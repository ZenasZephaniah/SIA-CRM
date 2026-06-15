# train_pipeline.py
import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
from src.utils import Config, seed_everything
from src.pipeline_stage1 import IngestionEngine, PseudoLabelGenerator

seed_everything()

def get_device():
    print("[INFO] Using highly stable CPU execution target.")
    return torch.device("cpu")

class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def evaluate_model(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1] # Probability of class 1 (Mismatch)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Grid search to find the optimal decision threshold that maximizes Macro F1
    best_f1 = 0.0
    best_threshold = 0.5
    best_acc = 0.0
    best_rec_0 = 0.0
    best_rec_1 = 0.0
    
    for th in np.arange(0.3, 0.7, 0.02):
        preds = (all_probs >= th).astype(int)
        f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th
            best_acc = accuracy_score(all_labels, preds)
            best_rec_0 = recall_score(all_labels, preds, pos_label=0, zero_division=0)
            best_rec_1 = recall_score(all_labels, preds, pos_label=1, zero_division=0)
            
    return best_acc, best_f1, best_rec_0, best_rec_1, best_threshold

def main():
    csv_path = "data/customer_support_tickets.csv"
    cache_path = "data/processed_pseudo_labeled_tickets.csv"
    ROW_LIMIT = 4000  

    if os.path.exists(cache_path):
        print(f"[INFO] Found cached Stage 1 pseudo-labeled data at: {cache_path}")
        labeled_df = pd.read_csv(cache_path)
    else:
        if not os.path.exists(csv_path):
            print(f"Error: Missing raw dataset file at {csv_path}")
            return
        print("Step 1: Ingesting dataset and generating self-supervised pseudo-labels...")
        raw_df = IngestionEngine.clean_crm_data(csv_path)
        generator = PseudoLabelGenerator(w1=0.6, w2=0.4, threshold=2)
        labeled_df = generator.run_pipeline(raw_df)
        labeled_df.to_csv(cache_path, index=False)

    if ROW_LIMIT and len(labeled_df) > ROW_LIMIT:
        print(f"[INFO] Subsampling dataset to {ROW_LIMIT} rows for highly responsive CPU training.")
        labeled_df = labeled_df.sample(n=ROW_LIMIT, random_state=Config.SEED).reset_index(drop=True)

    print("Step 2: Serializing structural metadata and text fields...")
    def serialize(row):
        return (
            f"Assigned Priority: {row['Ticket Priority']} | "
            f"Channel: {row['Ticket Channel']} | "
            f"Domain: {row['Customer Domain']} | "
            f"Type: {row['Ticket Type']} | "
            f"Subject: {row['Ticket Subject']} | "
            f"Description: {row['Ticket Description']}"
        )
    labeled_df['text'] = labeled_df.apply(serialize, axis=1)

    print("Step 3: Creating stratified train/test partitions...")
    train_df, val_df = train_test_split(
        labeled_df,
        test_size=0.20,
        stratify=labeled_df['mismatch_label'],
        random_state=Config.SEED
    )
    
    neg_count = (train_df['mismatch_label'] == 0).sum()
    pos_count = (train_df['mismatch_label'] == 1).sum()
    total = len(train_df)
    weight_0 = total / (2.0 * neg_count)
    weight_1 = total / (2.0 * pos_count)
    class_weights = [weight_0, weight_1]

    print("Step 4: Tokenizing data and building DataLoaders...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    train_dataset = TicketDataset(
        texts=train_df['text'].values,
        labels=train_df['mismatch_label'].values,
        tokenizer=tokenizer,
        max_len=Config.MAX_LEN
    )
    val_dataset = TicketDataset(
        texts=val_df['text'].values,
        labels=val_df['mismatch_label'].values,
        tokenizer=tokenizer,
        max_len=Config.MAX_LEN
    )
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    print(f"Step 5: Initializing model: {Config.MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_NAME, num_labels=2)
    device = get_device()
    model.to(device)
    model.float()

    loss_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=loss_weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=0.01)

    print("\nStep 6: Executing Native PyTorch Classifier Training Loop (CPU Stable)...")
    best_macro_f1 = 0.0
    best_tuned_threshold = 0.5
    save_path = "./models/sia_deberta"
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if (step + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{Config.EPOCHS} | Step {step+1}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")
        
        # Run Evaluation with Threshold Tuning
        avg_loss = epoch_loss / len(train_loader)
        acc, macro_f1, rec_0, rec_1, threshold = evaluate_model(model, val_loader, device)
        
        print(f"\n--- Epoch {epoch+1} Evaluation Summary (Tuned Threshold: {threshold:.2f}) ---")
        print(f"Avg Train Loss: {avg_loss:.4f}")
        print(f"Val Accuracy  : {acc:.2%}")
        print(f"Val Macro F1  : {macro_f1:.4f}")
        print(f"Val Recall (Consistent) : {rec_0:.2%}")
        print(f"Val Recall (Mismatched) : {rec_1:.2%}")
        print("------------------------------------------\n")
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_tuned_threshold = threshold
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            # Save the optimal threshold as metadata configuration
            threshold_config = {"optimal_threshold": float(best_tuned_threshold)}
            with open(os.path.join(save_path, "threshold_config.json"), "w") as f:
                json.dump(threshold_config, f)
                
            print(f"[SAVED] New best model saved with F1: {best_macro_f1:.4f} and Threshold: {best_tuned_threshold:.2f}\n")

    print("\n=== FINAL VERIFICATION RESULTS ===")
    best_model = AutoModelForSequenceClassification.from_pretrained(save_path).to(device)
    final_acc, final_f1, final_rec0, final_rec1, final_threshold = evaluate_model(best_model, val_loader, device)
    print(f"Best Accuracy      : {final_acc:.2%}")
    print(f"Best Macro F1 Score: {final_f1:.4f}  <--- Check this score now!")
    print(f"Best Class 0 Recall: {final_rec0:.2%}")
    print(f"Best Class 1 Recall: {final_rec1:.2%}")
    print(f"Optimal Decision Threshold: {final_threshold:.2f}")
    print("==================================")

if __name__ == "__main__":
    main()