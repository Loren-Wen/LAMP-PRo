import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pickle
import os
import glob
import re
from transformers import AutoTokenizer, AutoModel
from script.dataloader import MyDataset
from script.Model import ModelClassifier
from script.FocalLoss import FocalLoss
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef, average_precision_score
from sklearn.metrics import multilabel_confusion_matrix
from ast import literal_eval
from sklearn.metrics import classification_report


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def check_label_embedding_alignment(emb_dir, label_vectors):
    files = [f for f in os.listdir(emb_dir) if f.startswith("seq_") and f.endswith(".pt")]
    files = sorted(files, key=natural_sort_key)

    if len(files) != len(label_vectors):
        raise ValueError(
            f"Number of embedding files ({len(files)}) does not match number of label vectors ({len(label_vectors)}).")

    for idx, file in enumerate(files):
        expected = f"seq_{idx}.pt"
        if file != expected:
            raise ValueError(f"Mismatch at index {idx}: expected '{expected}', found '{file}'")

    print(f"All embedding files in {emb_dir} correctly aligned with label vectors.")


def collate_fn(batch):
    embs, labels, filenames, input_ids = zip(*batch)
    padded_embs = pad_sequence(embs, batch_first=True)
    attn_masks = pad_sequence([torch.ones(e.shape[0]) for e in embs], batch_first=True)
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded_embs, attn_masks, labels, filenames, padded_input_ids


def compute_pos_weights(y_labels):
    if isinstance(y_labels, torch.Tensor):
        y_labels = y_labels.cpu().numpy()

    label_counts = np.sum(y_labels, axis=0)
    total = y_labels.shape[0]

    pos_weights_np = (total - label_counts) / (label_counts + 1e-6)
    pos_weights = torch.tensor(pos_weights_np, dtype=torch.float)
    pos_weights = torch.clamp(pos_weights, max=10)
    return pos_weights


def print_multilabel_confusion(y_true, y_pred, label_names=None):
    cm = multilabel_confusion_matrix(y_true, y_pred)
    if label_names is None:
        label_names = [f"Label {i}" for i in range(y_true.shape[1])]
    for i, matrix in enumerate(cm):
        print(f"\nConfusion matrix for {label_names[i]}:")
        print(f"TN: {matrix[0, 0]} | FP: {matrix[0, 1]}")
        print(f"FN: {matrix[1, 0]} | TP: {matrix[1, 1]}")


def compute_metrics(y_true, y_logits):
    y_true = y_true.cpu().numpy()
    y_probs = torch.sigmoid(y_logits).cpu().numpy()
    y_pred = (y_probs > 0.5).astype(int)
    print_multilabel_confusion(y_true, y_pred, label_names=["DBP", "RBP", "Neither"])
    print(classification_report(y_true, y_pred, target_names=["DBP", "RBP", "Neither"]))

    is_dbrp_true = np.logical_and(y_true[:, 0] == 1, y_true[:, 1] == 1).astype(int)
    is_neither_true = np.logical_and(y_true[:, 0] == 0, y_true[:, 1] == 0).astype(int)

    is_dbrp_pred = np.logical_and(y_pred[:, 0] == 1, y_pred[:, 1] == 1).astype(int)
    is_neither_pred = np.logical_and(y_pred[:, 0] == 0, y_pred[:, 1] == 0).astype(int)

    metrics = {
        'subset_accuracy': accuracy_score(y_true, y_pred),
        'hamming_accuracy': 1 - np.mean(np.abs(y_true - y_pred)),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'mcc': matthews_corrcoef(y_true.ravel(), y_pred.ravel()),
        'roc_auc': roc_auc_score(y_true, y_probs, average='macro'),
        'pr_auc': average_precision_score(y_true, y_probs, average='macro'),
        'f1_DBP': f1_score(y_true[:, 0], y_pred[:, 0], zero_division=0),
        'f1_RBP': f1_score(y_true[:, 1], y_pred[:, 1], zero_division=0),
        'f1_DRBP': f1_score(is_dbrp_true, is_dbrp_pred, zero_division=0),
        'f1_Neither': f1_score(is_neither_true, is_neither_pred, zero_division=0),
        'auc_DBP': roc_auc_score(y_true[:, 0], y_probs[:, 0]),
        'auc_RBP': roc_auc_score(y_true[:, 1], y_probs[:, 1]),
        'auc_DRBP': roc_auc_score(is_dbrp_true, is_dbrp_pred),
        'auc_Neither': roc_auc_score(is_neither_true, is_neither_pred),
    }
    return metrics


VALID_LABELS = torch.tensor([
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
], dtype=torch.float)


def invalid_label_penalty(y_pred_sigmoid, weight=0.1):
    y_pred_bin = (y_pred_sigmoid > 0.5).float()
    penalties = torch.stack([
        ~(torch.any(torch.all(y_pred_bin[i].unsqueeze(0) == VALID_LABELS.to(y_pred_bin.device), dim=1)))
        for i in range(y_pred_bin.size(0))
    ]).float().to(y_pred_bin.device)
    return weight * penalties.mean()


def train(model, train_loader, val_loader, loss_fn, epochs=15, lr=1e-4, patience=5):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=2, min_lr=1e-6)

    best_roc_auc = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_logits, train_labels = [], []

        for x, attention_mask, y, _, _ in train_loader:
            x, attention_mask, y = x.to(device), attention_mask.to(device), y.to(device)
            optim.zero_grad()
            pred, _, _ = model(x, attention_mask=attention_mask, labels=y)
            loss_main = loss_fn(pred, y)
            y_pred_sigmoid = torch.sigmoid(pred)
            loss_penalty = invalid_label_penalty(y_pred_sigmoid, weight=0.1)
            loss = loss_main + loss_penalty
            loss.backward()
            optim.step()
            total_loss += loss.item()
            train_logits.append(pred.detach())
            train_labels.append(y.detach())

        train_logits = torch.cat(train_logits)
        train_labels = torch.cat(train_labels)
        train_metrics = compute_metrics(train_labels, train_logits)

        model.eval()
        with torch.no_grad():
            all_logits, all_labels = [], []
            for x, attention_mask, y, _, _ in val_loader:
                x, attention_mask, y = x.to(device), attention_mask.to(device), y.to(device)
                logit, _, _ = model(x, attention_mask=attention_mask, labels=y)
                all_logits.append(logit.detach())
                all_labels.append(y.detach())
            logits = torch.cat(all_logits)
            labels = torch.cat(all_labels)
            val_metrics = compute_metrics(labels, logits)

        roc_auc = val_metrics["roc_auc"]
        scheduler.step(roc_auc)

        print(f"Epoch {epoch + 1} - Loss: {total_loss / len(train_loader):.4f} | "
              f"F1: {val_metrics['f1_macro']:.4f} | Acc: {val_metrics['subset_accuracy']:.4f} | MCC: {val_metrics['mcc']:.4f}")
        print(f"Train Loss   : {total_loss / len(train_loader):.4f}")
        print(f"Train Metrics: F1={train_metrics['f1_macro']:.4f}, "
              f"Subset Acc={train_metrics['subset_accuracy']:.4f}, "
              f"Hamming Acc={train_metrics['hamming_accuracy']:.4f}, "
              f"MCC={train_metrics['mcc']:.4f}, "
              f"AUC={train_metrics['roc_auc']:.4f}, PR_AUC={train_metrics['pr_auc']:.4f}")
        print(f"Val  Metrics: F1={val_metrics['f1_macro']:.4f}, "
              f"Subset Acc={val_metrics['subset_accuracy']:.4f}, "
              f"Hamming Acc={val_metrics['hamming_accuracy']:.4f}, "
              f"MCC={val_metrics['mcc']:.4f}, "
              f"AUC={val_metrics['roc_auc']:.4f}, PR_AUC={val_metrics['pr_auc']:.4f}")
        print(f"F1 Scores  → DBP: {val_metrics['f1_DBP']:.4f}, RBP: {val_metrics['f1_RBP']:.4f}, "
              f"DRBP: {val_metrics['f1_DRBP']:.4f}, Neither: {val_metrics['f1_Neither']:.4f}")
        print(f"AUC Scores → DBP: {val_metrics['auc_DBP']:.4f}, RBP: {val_metrics['auc_RBP']:.4f}, "
              f"DRBP: {val_metrics['auc_DRBP']:.4f}, Neither: {val_metrics['auc_Neither']:.4f}")

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join("Lamp/BestModel.pt")) 
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break


if __name__ == '__main__':
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "facebook/esm2_t30_150M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    esm = AutoModel.from_pretrained(model_name)

    train_data = pd.read_csv('Lamp/Training_with_Label_shuffled.csv', index_col=None)
    test_data = pd.read_csv('Lamp/Validation_with_Label_shuffled.csv', index_col=None)

    train_labels = train_data["label_vector"]
    test_labels = test_data["label_vector"]

   
    train_embedding_dir = os.path.expanduser('Lamp/embeddings/training/train')
    test_embedding_dir = os.path.expanduser('Lamp/embeddings/training/test')
    
    check_label_embedding_alignment(train_embedding_dir, train_labels.tolist())
    check_label_embedding_alignment(test_embedding_dir, test_labels.tolist())

    train_dataset = MyDataset(embedding_dir=train_embedding_dir, label_vectors=train_labels)
    test_dataset = MyDataset(embedding_dir=test_embedding_dir, label_vectors=test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    input_channel = 640
    NeuralNetwork = ModelClassifier(in_channel=640, num_labels=3, use_mhsa=True)
    NeuralNetwork.to(device)

    train_labels_tensor = torch.tensor([literal_eval(lbl) for lbl in train_labels.tolist()], dtype=torch.float)
    pos_weight = compute_pos_weights(train_labels_tensor).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train(NeuralNetwork, train_loader, test_loader, loss_fn=loss_fn)