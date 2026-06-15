# src/pipeline_stage1.py
import torch  
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import pipeline
from sklearn.metrics import cohen_kappa_score
from src.utils import Config, seed_everything

seed_everything()

# src/pipeline_stage1.py
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import pipeline
from sklearn.metrics import cohen_kappa_score
from src.utils import Config, seed_everything

seed_everything()

# src/pipeline_stage1.py
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import pipeline
from sklearn.metrics import cohen_kappa_score
from src.utils import Config, seed_everything

seed_everything()

class IngestionEngine:
    @staticmethod
    def find_column(columns: list, target_keywords: list) -> str:
        """
        Helper function to scan column names using keywords (case-insensitive).
        """
        for col in columns:
            col_lower = str(col).lower().replace("_", " ").replace("-", " ").strip()
            if all(keyword in col_lower for keyword in target_keywords):
                return col
        return None

    @classmethod
    def clean_crm_data(cls, filepath: str) -> pd.DataFrame:
        """
        Loads and standardizes the Kaggle Customer Support Tickets dataset by auto-detecting columns.
        """
        df = pd.read_csv(filepath)
        cols = list(df.columns)
        
        # Exact mappings based on your CSV's printed columns
        mappings = {
            'Ticket Subject': ['subject'],
            'Ticket Description': ['description'],
            'Ticket Priority': ['priority'],
            'Ticket Type': ['category'],       # Maps 'Issue_Category' -> 'Ticket Type'
            'Ticket Channel': ['channel'],
            'Resolution Time': ['resolution'],  # Maps 'Resolution_Time_Hours' -> 'Resolution Time'
            'Customer Email': ['email']         # Maps 'Customer_Email' -> 'Customer Email'
        }
        
        mapped_columns = {}
        for target_key, keywords in mappings.items():
            found_col = cls.find_column(cols, keywords)
            if found_col:
                mapped_columns[found_col] = target_key
            else:
                raise KeyError(
                    f"Required feature matching keywords {keywords} (for '{target_key}') not found in CSV. "
                    f"Available columns are: {cols}"
                )
        
        # Rename identified columns to standardized keys
        df = df.rename(columns=mapped_columns)
        
        # Keep standardized columns for processing
        standard_cols = list(mappings.keys())
        id_col = cls.find_column(cols, ['id'])
        if id_col:
            df = df.rename(columns={id_col: 'Ticket ID'})
            standard_cols.append('Ticket ID')
        else:
            df['Ticket ID'] = df.index
            standard_cols.append('Ticket ID')
            
        df = df[standard_cols].copy()

        # Extract domain-tier feature from email: e.g. "user@corporation.com" -> "corporation.com"
        df['Customer Domain'] = df['Customer Email'].apply(
            lambda x: str(x).split('@')[-1] if '@' in str(x) else 'unknown.com'
        )

        # Strip string values and handle nulls
        string_cols = ['Ticket Subject', 'Ticket Description', 'Ticket Priority', 'Ticket Type', 'Ticket Channel', 'Customer Domain']
        for col in string_cols:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(["nan", ""], "Unknown")
        
        # Force Resolution Time to numeric
        df['Resolution Time'] = pd.to_numeric(df['Resolution Time'], errors='coerce')
        # Impute missing resolution times using group median
        df['Resolution Time'] = df.groupby('Ticket Type')['Resolution Time'].transform(
            lambda x: x.fillna(x.median() if not x.isna().all() else 24.0)
        )
            
        # Filter rows to valid priority values defined in Config
        df = df[df['Ticket Priority'].isin(Config.PRIORITY_MAP.keys())].reset_index(drop=True)
        return df

class SemanticUrgencyEvaluator:
    # src/pipeline_stage1.py 
    def __init__(self):
        # Determine best available backend (CUDA -> MPS -> CPU)
        if torch.cuda.is_available():
            self.device = 0
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = -1
            
        print(f"Initializing SemanticUrgencyEvaluator on device: {self.device}")
        
        self.classifier = pipeline(
            "zero-shot-classification",
            model=Config.NLI_MODEL_NAME,
            device=self.device
        )
        self.candidate_labels = ["Low urgency", "Medium urgency", "High urgency", "Critical urgency"]
        self.label_map = {
            "Low urgency": 1,
            "Medium urgency": 2,
            "High urgency": 3,
            "Critical urgency": 4
        }

    def evaluate_batch(self, descriptions: list, batch_size: int = 32) -> list:
        """
        Runs zero-shot inference over description strings in batches.
        """
        scores = []
        for i in tqdm(range(0, len(descriptions), batch_size), desc="Evaluating Semantic Urgency"):
            batch = descriptions[i : i + batch_size]
            results = self.classifier(
                batch,
                candidate_labels=self.candidate_labels,
                hypothesis_template="This customer support ticket indicates {}."
            )
            
            # Handle both single dictionary returns and lists from pipeline
            if not isinstance(results, list):
                results = [results]
                
            for res in results:
                # Get the label with the highest probability score
                top_label = res['labels'][0]
                scores.append(self.label_map[top_label])
                
        return scores
    
class OperationalLatencyEvaluator:
    @staticmethod
    def compute_z_scores(df: pd.DataFrame) -> pd.Series:
        """
        Calculates group-wise Median Absolute Deviation (MAD) robust Z-scores
        for ticket Resolution Times.
        """
        def get_group_z_scores(group):
            median = group.median()
            # Absolute deviations from median
            abs_dev = (group - median).abs()
            mad = abs_dev.median()
            
            # Prevent division by zero if MAD is 0
            if mad == 0:
                mad = 1e-5
                
            # Compute robust scale factor (1.4826 matches normal distribution scaling)
            z_score = (group - median) / (1.4826 * mad)
            return z_score
            
        return df.groupby('Ticket Type')['Resolution Time'].transform(get_group_z_scores)

    @classmethod
    def assign_operational_scores(cls, df: pd.DataFrame) -> pd.Series:
        """
        Bins Z-scores into normalized severity categories 1 to 4.
        """
        z_scores = cls.compute_z_scores(df)
        
        # Save Z-score back to dataframe for transparency/audit traces
        df['resolution_z_score'] = z_scores
        
        conditions = [
            (z_scores <= 0.0),
            (z_scores > 0.0) & (z_scores <= 1.0),
            (z_scores > 1.0) & (z_scores <= 2.5),
            (z_scores > 2.5)
        ]
        choices = [1, 2, 3, 4]
        return pd.Series(np.select(conditions, choices, default=2), index=df.index)
    
class PseudoLabelGenerator:
    def __init__(self, w1: float = 0.6, w2: float = 0.4, threshold: int = 2):
        self.w1 = w1
        self.w2 = w2
        self.threshold = threshold

    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        # Step A: Evaluate Semantic Urgency
        semantic_evaluator = SemanticUrgencyEvaluator()
        # Feed combined subject + description to give NLI context
        text_inputs = (df['Ticket Subject'] + " - " + df['Ticket Description']).tolist()
        df['S_sem'] = semantic_evaluator.evaluate_batch(text_inputs)
        
        # Step B: Evaluate Operational Latency
        df['S_ops'] = OperationalLatencyEvaluator.assign_operational_scores(df)
        
        # Step C: Weighted Fusion Calculation
        raw_inferred = (self.w1 * df['S_sem']) + (self.w2 * df['S_ops'])
        df['inferred_severity'] = np.round(raw_inferred).astype(int)
        
        # Map inferred value back to human-readable strings
        df['inferred_severity_label'] = df['inferred_severity'].map(Config.REVERSE_PRIORITY_MAP)
        
        # Step D: Map Human Priority to Ints
        df['P_assigned'] = df['Ticket Priority'].map(Config.PRIORITY_MAP)
        
        # Step E: Apply Mismatch Metric Rule
        # A mismatch exists if the absolute delta is greater than or equal to the threshold
        df['severity_delta'] = (df['inferred_severity'] - df['P_assigned']).abs()
        df['mismatch_label'] = (df['severity_delta'] >= self.threshold).astype(int)
        
        # Calculate consistency metrics
        kappa = cohen_kappa_score(df['S_sem'], df['S_ops'])
        print("\n=== STAGE 1 SIGNAL GENERATION REPORT ===")
        print(f"Agreement Metric (Cohen's Kappa) Between Semantic & Operational Signals: {kappa:.4f}")
        print(f"Total Consistent Instances (0): {len(df[df['mismatch_label'] == 0])}")
        print(f"Total Mismatched Instances (1): {len(df[df['mismatch_label'] == 1])}")
        print("=========================================\n")
        
        return df