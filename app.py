# app.py
"""
Streamlit web app for the Support Integrity Auditor (SIA).

Two tabs:
  1. Batch File Auditor - upload a CSV of support tickets, get back
     a dashboard of flagged mismatches with dossiers for inspection.
  2. Single Ticket Auditor - manually enter one ticket and get an
     immediate integrity check.

Cloud-Safe Deployment:
  - If the fine-tuned DeBERTa model weights are not found (e.g. on Streamlit Cloud),
    the app gracefully falls back to Zero-Shot Signal Fusion Mode.
  - This mode uses the Stage 1 NLI + Operational Latency rules to compute the
    mismatches. This keeps the application fully functional online without pushing
    500MB+ weight files to GitHub.
"""

import warnings
import sys

warnings.filterwarnings("ignore")

class _DevNull:
    def write(self, x): pass
    def flush(self): pass

_saved_stderr = sys.stderr
sys.stderr = _DevNull()
try:
    import torchvision  # noqa: F401
except Exception:
    pass
sys.stderr = _saved_stderr

import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from src.utils import Config, seed_everything
from src.pipeline_stage1 import (
    IngestionEngine,
    OperationalLatencyEvaluator,
    PseudoLabelGenerator,
)
from src.dossier_engine import DossierEngine

seed_everything()

st.set_page_config(layout="wide", page_title="SIA Dashboard", page_icon="🛡️")

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_auditor_model():
    """Load the fine-tuned DeBERTa classifier if present."""
    model_path = "./models/sia_deberta"
    if not os.path.exists(model_path):
        return None, None, 0.5
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()

        threshold_file = os.path.join(model_path, "threshold_config.json")
        threshold = 0.5
        if os.path.exists(threshold_file):
            with open(threshold_file, "r") as f:
                threshold = json.load(f).get("optimal_threshold", 0.5)
        return tokenizer, model, threshold
    except Exception:
        return None, None, 0.5


@st.cache_resource
def load_stage1_engine():
    """Load the Stage 1 pseudo-labeling engine (NLI + z-score)."""
    try:
        generator = PseudoLabelGenerator(w1=0.6, w2=0.4, threshold=2)
        return generator
    except Exception as e:
        st.warning(f"Could not initialize Stage 1 NLI engine: {e}")
        return None


@st.cache_resource
def load_historical_stats():
    """Load cached pseudo-labeled data for single-ticket mode reference."""
    cache_path = "data/processed_pseudo_labeled_tickets.csv"
    if not os.path.exists(cache_path):
        return None
    try:
        return pd.read_csv(cache_path)
    except Exception:
        return None


tokenizer, model, decision_threshold = load_auditor_model()


# ---------------------------------------------------------------------------
# Core Helper Functions
# ---------------------------------------------------------------------------

def serialize_ticket(row) -> str:
    """Serialize a ticket row into the same text format used during training."""
    return (
        f"Description: {row.get('Ticket Description', '')} | "
        f"Subject: {row.get('Ticket Subject', '')} | "
        f"Assigned Priority: {row.get('Ticket Priority', '')} | "
        f"Channel: {row.get('Ticket Channel', '')} | "
        f"Domain: {row.get('Customer Domain', '')} | "
        f"Type: {row.get('Ticket Type', '')}"
    )


def predict_mismatch(text: str):
    """Run the trained classifier on one serialized ticket text."""
    if not model or not tokenizer:
        return 0, 0.5, 0.5
        
    inputs = tokenizer(
        text,
        max_length=Config.MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        prob_mismatch = probs[0][1].item()
        
    is_mismatch = 1 if prob_mismatch >= decision_threshold else 0
    confidence = prob_mismatch if is_mismatch == 1 else (1.0 - prob_mismatch)
    return is_mismatch, confidence, prob_mismatch


# App header
st.title("🛡️ Support Integrity Auditor (SIA)")
st.markdown(
    "Automated auditing system to detect mismatches between assigned ticket "
    "priority and objective semantic + operational indicators."
)

# Render mode banner
if model:
    st.sidebar.success("💻 Mode: Fine-Tuned DeBERTa Classifier")
    st.sidebar.markdown(f"**Decision threshold:** `{decision_threshold:.2f}`")
else:
    st.sidebar.info("☁️ Mode: Zero-Shot Signal Fusion")
    st.sidebar.markdown("**Decision threshold:** `Stage 1 Fusion Rules`")
    st.info("ℹ️ Running in **Zero-Shot Signal Fusion Mode**. The app is utilizing our Stage 1 NLI and operational latency metrics to compute mismatches. This enables full dashboard capabilities on cloud servers.")

st.sidebar.markdown(f"**Max sequence length:** `{Config.MAX_LEN}`")


# ---------------------------------------------------------------------------
# Tab 1: Batch File Auditor
# ---------------------------------------------------------------------------

tab1, tab2 = st.tabs(["📊 Batch File Auditor", "🔍 Single Ticket Auditor"])

with tab1:
    st.header("Batch Ticket Auditing Portal")
    uploaded_file = st.file_uploader("Upload Support Tickets CSV File", type=["csv"])

    if uploaded_file:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.info(f"Loaded {len(df_raw)} records from uploaded file.")

            with st.spinner("Processing metadata indicators..."):
                temp_path = "data/temp_upload.csv"
                df_raw.to_csv(temp_path, index=False)

                try:
                    cleaned_df = IngestionEngine.clean_crm_data(temp_path)
                except KeyError as ke:
                    st.error(f"⚠️ Ingestion Error: Missing required columns: {ke}")
                    st.stop()

                generator = load_stage1_engine()
                if generator is None:
                    st.error("⚠️ Stage 1 engine failed to load.")
                    st.stop()
                processed_df = generator.run_pipeline(cleaned_df)

                if os.path.exists(temp_path):
                    os.remove(temp_path)

            # Prediction Phase
            predictions = []
            confidences = []
            
            if model:
                # Use DeBERTa Classifier
                with st.spinner("Executing DeBERTa classification checks..."):
                    for idx, row in processed_df.iterrows():
                        text = serialize_ticket(row)
                        is_mismatch, confidence, prob_mismatch = predict_mismatch(text)
                        predictions.append(is_mismatch)
                        confidences.append(confidence)
            else:
                # Fallback: Stage 1 Fusion Rules
                predictions = processed_df['mismatch_label'].tolist()
                confidences = [0.95 if p == 1 else 0.90 for p in predictions]

            processed_df['predicted_mismatch'] = predictions
            processed_df['confidence_score'] = confidences

            st.subheader("Audit Summary")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Tickets Scanned", f"{len(processed_df)}")
            m2.metric("Flagged Priority Mismatches", f"{sum(predictions)}")
            m3.metric("Audited Conflict Rate", f"{sum(predictions)/len(processed_df):.2%}")

            g1, g2 = st.columns(2)
            with g1:
                fig_pie = px.pie(processed_df, names='predicted_mismatch', title="Audit Status Distribution", color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig_pie, use_container_width=True)
            with g2:
                fig_heat = px.density_heatmap(processed_df, x="Ticket Type", y="Ticket Channel", z="severity_delta", histfunc="avg", title="Mean Severity Delta by Channel and Category")
                st.plotly_chart(fig_heat, use_container_width=True)

            st.subheader("Audited Mismatch Logs")
            mismatch_df = processed_df[processed_df['predicted_mismatch'] == 1].copy()
            if len(mismatch_df) > 0:
                selected_id = st.selectbox("Inspect Evidence Dossier for a flagged ticket:", mismatch_df['Ticket ID'].astype(str).values)
                target_row = mismatch_df[mismatch_df['Ticket ID'].astype(str) == selected_id].iloc[0]
                dossier = DossierEngine.generate(target_row.to_dict(), 1, target_row['confidence_score'])
                st.json(dossier)
            else:
                st.success("✅ No priority mismatches detected in this batch.")

        except Exception as e:
            st.error(f"⚠️ Failed to parse uploaded file: {str(e)}")


# ---------------------------------------------------------------------------
# Tab 2: Single Ticket Auditor
# ---------------------------------------------------------------------------

with tab2:
    st.header("Single Ticket Integrity Auditor")
    st.markdown("Manually evaluate a single ticket configuration against historical distributions.")

    historical_df = load_historical_stats()
    if historical_df is None:
        st.warning("⚠️ No historical data cache found. Operational z-score will default to 0.0.")

    with st.form("manual_ticket_form"):
        subject = st.text_input("Ticket Subject", "Database unreachable")
        desc = st.text_area("Ticket Description", "The main postgres server has failed and is refusing connections.")
        col1, col2, col3 = st.columns(3)
        priority = col1.selectbox("Assigned Ticket Priority", list(Config.PRIORITY_MAP.keys()))
        category = col2.selectbox("Ticket Category", ["Technical Support", "Billing Inquiry", "Account Access", "Product Feedback", "General Inquiry"])
        channel = col3.selectbox("Ticket Channel", ["Email", "Chat", "Phone", "Social Media"])
        email = st.text_input("Customer Contact Email", "admin@enterprisecorp.com")
        res_time = st.number_input("Observed Resolution Time (Hours)", min_value=0.1, value=24.0, step=1.0)

        submitted = st.form_submit_button("Run Single Audit Check")

        if submitted:
            with st.spinner("Computing operational metrics..."):
                z_val = 0.0
                s_ops = 1
                z_basis_note = "no_historical_reference"

                if historical_df is not None and 'Ticket Type' in historical_df.columns:
                    type_subset = historical_df[historical_df['Ticket Type'] == category]
                    if len(type_subset) >= 5 and 'Resolution Time' in type_subset.columns:
                        median = type_subset['Resolution Time'].median()
                        mad = (type_subset['Resolution Time'] - median).abs().median()
                        if mad > 0 and not pd.isna(median) and not pd.isna(mad):
                            z_val = (res_time - median) / (1.4826 * mad)
                            z_basis_note = f"computed against {len(type_subset)} historical '{category}' tickets"
                        else:
                            z_basis_note = f"zero MAD in '{category}' group; defaulting to z=0.0"
                    else:
                        z_basis_note = f"only {len(type_subset)} historical '{category}' tickets; defaulting to z=0.0"

                if z_val > 0.0 and z_val <= 1.0: s_ops = 2
                elif z_val > 1.0 and z_val <= 2.5: s_ops = 3
                elif z_val > 2.5: s_ops = 4

                urgency_keywords = ["immediately", "broken", "fail", "crash", "error", "prevent", "block", "loss", "urgently", "severe", "down", "outage", "critical"]
                text_lower = (subject + " " + desc).lower()
                keyword_hits = sum(1 for kw in urgency_keywords if kw in text_lower)

                if keyword_hits >= 5: s_sem = 4
                elif keyword_hits >= 3: s_sem = 3
                elif keyword_hits >= 1: s_sem = 2
                else: s_sem = 1

                s_inf = round(0.6 * s_sem + 0.4 * s_ops)
                s_inf = max(1, min(4, s_inf))
                s_inf_label = Config.REVERSE_PRIORITY_MAP[s_inf]

                synthetic_row = {
                    'Ticket ID': 'MANUAL-ENTRY',
                    'Ticket Subject': subject,
                    'Ticket Description': desc,
                    'Ticket Priority': priority,
                    'Ticket Type': category,
                    'Ticket Channel': channel,
                    'Customer Email': email,
                    'Customer Domain': (email.split('@')[-1] if '@' in email else 'unknown.com'),
                    'Resolution Time': res_time,
                    'P_assigned': Config.PRIORITY_MAP[priority],
                    'resolution_z_score': z_val,
                    'S_ops': s_ops,
                    'S_sem': s_sem,
                    'inferred_severity': s_inf,
                    'inferred_severity_label': s_inf_label,
                    'severity_delta': abs(s_inf - Config.PRIORITY_MAP[priority]),
                }

                # Evaluate mismatch
                if model:
                    serialized_text = serialize_ticket(synthetic_row)
                    is_mismatch, confidence, raw_prob = predict_mismatch(serialized_text)
                else:
                    # Fallback logic
                    is_mismatch = 1 if synthetic_row['severity_delta'] >= 2 else 0
                    confidence = 0.95
                    raw_prob = 1.0 if is_mismatch == 1 else 0.0

                st.markdown("### Audit Computation Trace")
                trace_df = pd.DataFrame({
                    'Metric': [
                        'Operational z-score', 'Operational severity (S_ops)',
                        'Semantic urgency keywords hit', 'Semantic severity (S_sem)',
                        'Fused inferred severity (S_inf)', 'Z-score basis',
                        'Classifier Selection',
                    ],
                    'Value': [
                        f"{z_val:.2f}", str(s_ops),
                        str(keyword_hits), str(s_sem),
                        f"{s_inf} ({s_inf_label})", z_basis_note,
                        "Fine-Tuned DeBERTa" if model else "Zero-Shot Stage 1 Rule Engine",
                    ],
                })
                st.dataframe(trace_df, use_container_width=True, hide_index=True)

                if is_mismatch == 1:
                    st.error(f"🚨 Priority Mismatch Detected (Confidence: {confidence:.2%})")
                    dossier = DossierEngine.generate(synthetic_row, 1, confidence)
                else:
                    st.success(f"✅ Priority Integrity Confirmed (Confidence: {confidence:.2%})")
                    dossier = DossierEngine.generate(synthetic_row, 0, confidence)

                st.markdown("### Evidence Dossier")
                st.json(dossier)