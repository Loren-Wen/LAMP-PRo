from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef, average_precision_score
import torch
def compute_metrics(y_true, y_logits):
    y_true = y_true.cpu().numpy()
    y_probs = torch.sigmoid(y_logits).cpu().numpy()
    y_pred = (torch.sigmoid(y_logits) > 0.5).int().cpu().numpy()

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'mcc': matthews_corrcoef(y_true.ravel(), y_pred.ravel()),
        'roc_auc': roc_auc_score(y_true, y_probs, average='macro'),
        'pr_auc': average_precision_score(y_true, y_probs, average='macro'),
    }