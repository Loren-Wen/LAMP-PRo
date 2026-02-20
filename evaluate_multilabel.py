import pandas as pd
import re
import numpy as np
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, auc, accuracy_score,
    matthews_corrcoef, hamming_loss, precision_score, recall_score, f1_score
)

def evaluate(path):
    df = pd.read_csv(path)
    print("metrics for", path)

    def clean_label_string(label_str):
        """Removes numpy-specific text and evaluates the string to a list."""
        cleaned_str = re.sub(r'np\.\w+\((.*?)\)', r'\1', str(label_str))
        try:
            return eval(cleaned_str)
        except:
            return [] 

    df['actual labels'] = df['actual labels'].apply(clean_label_string).apply(lambda x: [int(i) for i in x])
    df['predicted_label'] = df['predicted_label'].apply(clean_label_string).apply(lambda x: [int(i) for i in x])
    df['probability'] = df['probability'].apply(clean_label_string)

    # --- Prepare data arrays ---
    y_true = np.array(df['actual labels'].to_list())
    y_pred = np.array(df['predicted_label'].to_list())
    y_probs = np.array(df['probability'].to_list())

    # --- 1. Per-Class Confusion Matrix (One-vs-Rest) ---
    print("--- Per-Class Analysis (One-vs-Rest) ---")

    label_map = {
        (1,0,0): 'DBP', (0,1,0): 'RBP',
        (1,1,0): 'DRBP', (0,0,1): 'neither'
    }
    def list_to_class(row): return label_map.get(tuple(row), 'other')
    df['actual_class'] = df['actual labels'].apply(list_to_class)
    df['pred_class'] = df['predicted_label'].apply(list_to_class)

    classes = ['DBP', 'RBP', 'DRBP', 'neither']
    for c in classes:
        print(f"\n===== {c} vs Rest =====")
        y_true_binary = (df['actual_class'] == c)
        y_pred_binary = (df['pred_class'] == c)
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        print("Confusion Matrix:")
        print(cm)

        acc = accuracy_score(y_true_binary, y_pred_binary)
        mcc = matthews_corrcoef(y_true_binary, y_pred_binary)
        print(f"Accuracy: {acc:.4f}")
        print(f"MCC     : {mcc:.4f}")

    # --- 2. AUC & 1-AURC ---
    DBP_IDX, RBP_IDX = 0, 1

    def safe_roc_auc_score(y_true_col, y_prob_col, other_true_col=None):
        if other_true_col is not None:
            # Exclude DRBP samples (where both = 1)
            drbp_mask = (y_true_col == 1) & (other_true_col == 1)
            y_true_col = y_true_col[~drbp_mask]
            y_prob_col = y_prob_col[~drbp_mask]

        if len(np.unique(y_true_col)) < 2:
            return np.nan

        return roc_auc_score(y_true_col, y_prob_col)

    def calculate_aurc(y_true, y_probs, target_idx, cross_idx):
        tpr_points, cpr_points = [], []
        true_target_mask = (y_true[:, target_idx] == 1) & (y_true[:, cross_idx] == 0)
        true_cross_mask = (y_true[:, cross_idx] == 1) & (y_true[:, target_idx] == 0)

        if np.sum(true_cross_mask) == 0: 
            return np.nan

        target_probs = y_probs[:, target_idx]
        for thresh in np.linspace(0, 1, 101):
            preds = (target_probs >= thresh).astype(int)
            tpr = np.sum(preds[true_target_mask] == 1) / np.sum(true_target_mask) if np.sum(true_target_mask) > 0 else 0
            cpr = np.sum(preds[true_cross_mask] == 1) / np.sum(true_cross_mask)
            tpr_points.append(tpr)
            cpr_points.append(cpr)

        sorted_points = sorted(zip(tpr_points, cpr_points))
        return auc([p[0] for p in sorted_points], [p[1] for p in sorted_points])
    
    # --- 2b. DRBP AUC ---
    def calculate_drbp_auc(y_true, y_probs):
        # Positive = DRBP (both DBP=1 and RBP=1)
        true_drbp = ((y_true[:, DBP_IDX] == 1) & (y_true[:, RBP_IDX] == 1)).astype(int)
        prob_drbp = np.minimum(y_probs[:, DBP_IDX], y_probs[:, RBP_IDX])  
        # conservative: take min(DBP_prob, RBP_prob) as DRBP confidence

        if len(np.unique(true_drbp)) < 2:
            return np.nan

        return roc_auc_score(true_drbp, prob_drbp)

    print("\n\n--- Performance Summary (As per Research Paper) ---")
    auc_dbp = safe_roc_auc_score(y_true[:, DBP_IDX], y_probs[:, DBP_IDX], other_true_col=y_true[:, RBP_IDX])
    auc_rbp = safe_roc_auc_score(y_true[:, RBP_IDX], y_probs[:, RBP_IDX], other_true_col=y_true[:, DBP_IDX])

    aurc_dbp = calculate_aurc(y_true, y_probs, target_idx=DBP_IDX, cross_idx=RBP_IDX)
    aurc_rbp = calculate_aurc(y_true, y_probs, target_idx=RBP_IDX, cross_idx=DBP_IDX)
    
    auc_drbp = calculate_drbp_auc(y_true, y_probs)

    print(f"\n[DNA-binding] AUC: {auc_dbp:.4f} | 1-AURC: {1 - aurc_dbp:.4f}")
    print(f"[RNA-binding] AUC: {auc_rbp:.4f} | 1-AURC: {1 - aurc_rbp:.4f}")
    print(f"[DNA-RNA-binding] AUC: {auc_drbp:.4f}")

    # --- 3. Overall Multi-Label Performance Metrics ---
    print("\n\n--- Overall Multi-Label Performance Metrics ---")
    subset_acc = accuracy_score(y_true, y_pred)
    h_loss = hamming_loss(y_true, y_pred)
    print(f"Subset Accuracy      : {subset_acc:.4f} (Exact match ratio)")
    print(f"Hamming Score        : {1 - h_loss:.4f} (Label-based accuracy)")

    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)

    print(f"\nPrecision (Macro/Micro): {precision_macro:.4f} / {precision_micro:.4f}")
    print(f"Recall (Macro/Micro)   : {recall_macro:.4f} / {recall_micro:.4f}")
    print(f"F1-Score (Macro/Micro) : {f1_macro:.4f} / {f1_micro:.4f}")

    # Flattened MCC (not the true multi-class one, but across all labels)
    mcc_flat = matthews_corrcoef(y_true.ravel(), y_pred.ravel())
    print(f"\nMatthews Corr. Coef. (Flattened labels) : {mcc_flat:.4f}")
    print("")

if __name__ == "__main__":
	files=[r"/predictions_TEST474_FULL.csv",
	      r"/predictions_DRBP206_FULL.csv",
	      r"/predictions_PDB255_FULL.csv",
	       r"/predictions_EZL_FULL.csv"]
	for path in files:
	    evaluate(path)
