# run_stage1.py
from src.pipeline_stage1 import IngestionEngine, PseudoLabelGenerator
import os

def main():
    csv_path = "data/customer_support_tickets.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find the dataset at '{csv_path}'. Please check Step 2.")
        return

    print("Step 1: Loading and cleaning dataset...")
    # Read the first 100 rows to quickly verify the pipeline without waiting long
    df = IngestionEngine.clean_crm_data(csv_path).head(100)
    print(f"Loaded {len(df)} rows successfully.")

    print("\nStep 2: Generating pseudo-labels (running semantic & operational metrics)...")
    generator = PseudoLabelGenerator(w1=0.6, w2=0.4, threshold=2)
    processed_df = generator.run_pipeline(df)

    # run_stage1.py (Modified Preview Block)
    print("\nStep 3: Sampling processed output:")
    preview_cols = ['Ticket Priority', 'Customer Domain', 'S_sem', 'S_ops', 'inferred_severity_label', 'mismatch_label']
    print(processed_df[preview_cols].head(10))

if __name__ == "__main__":
    main()