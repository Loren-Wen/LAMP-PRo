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
from evaluate_multilabel import evaluate
from script.Model import ModelClassifier
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence
from scipy.ndimage import zoom
from sklearn.metrics import multilabel_confusion_matrix, classification_report


from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef, average_precision_score

if __name__ == '__main__':
    # Set Random Seed
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # GPU training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = "facebook/esm2_t30_150M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    esm = AutoModel.from_pretrained(model_name)


    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


    def check_label_embedding_alignment(emb_dir, label_vectors):
        files = [f for f in os.listdir(emb_dir) if f.startswith("seq_") and f.endswith(".pt")]
        files = sorted(files, key=natural_sort_key)

        if len(files) != len(label_vectors):
            raise ValueError(
                f"Number of embedding files ({len(files)}) does not match number of label vectors ({len(label_vectors)}).")

            # Check filename sequence
        for idx, file in enumerate(files):
            expected = f"seq_{idx}.pt"
            if file != expected:
                raise ValueError(f" Mismatch at index {idx}: expected '{expected}', found '{file}'")

        print(" All embedding files correctly aligned with label vectors.")


    def collate_fn(batch):
        embs, labels, filenames, input_ids = zip(*batch)
        padded_embs = pad_sequence(embs, batch_first=True)  # → [B, L_max, D]
        attn_masks = pad_sequence([torch.ones(e.shape[0]) for e in embs], batch_first=True)
        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = torch.stack(labels)
        return padded_embs, attn_masks, labels, filenames, padded_input_ids


    def rescale_attention_weights(label_weights, attention_mask):
        """
        Rescales attention weights to match original input lengths.

        Args:
            label_weights: [B, num_labels, N'] – attention weights
            attention_mask: [B, seq_len] – original mask for input

        Returns:
            attn_maps: list of [num_labels, original_len] per sample
        """
        attn_maps = []
        for i in range(label_weights.size(0)):
            original_len = int(attention_mask[i].sum().item())
            rescaled = []
            for l in range(label_weights.size(1)):  # Loop over labels
                weights = label_weights[i, l].detach().cpu().numpy()  # [N']
                rescaled_weights = zoom(weights, original_len / len(weights))  # [original_len]
                rescaled.append(rescaled_weights)
            attn_maps.append(np.stack(rescaled))  # [num_labels, original_len]
        return attn_maps


    def print_multilabel_confusion(y_true, y_pred, label_names=None):
        cm = multilabel_confusion_matrix(y_true, y_pred)

        if label_names is None:
            label_names = [f"Label {i}" for i in range(y_true.shape[1])]

        for i, matrix in enumerate(cm):
            print(f"\nConfusion matrix for {label_names[i]}:")
            print(f"TN: {matrix[0, 0]} | FP: {matrix[0, 1]}")
            print(f"FN: {matrix[1, 0]} | TP: {matrix[1, 1]}")

    VALID_LABELS = torch.tensor([
        [1, 0, 0],  # DBP
        [0, 1, 0],  # RBP
        [1, 1, 0],  # DBRP
        [0, 0, 1],  # Neither
    ], dtype=torch.float).to(device)


    def invalid_label_penalty(y_pred_sigmoid, weight=0.1):
        """
        y_pred_sigmoid: [B, 3] after sigmoid
        Returns: scalar penalty for invalid label predictions
        """
        y_pred_bin = (y_pred_sigmoid > 0.5).float()  # shape: [B, 3]

        # Check for invalid predictions by comparing with all valid labels
        penalties = torch.stack([
            ~(torch.any(torch.all(y_pred_bin[i].unsqueeze(0) == VALID_LABELS, dim=1)))
            for i in range(y_pred_bin.size(0))
        ]).float().to(y_pred_bin.device)  # [B]

        return weight * penalties.mean()

    def predict(model, dataloader, device, save_to_csv=None, embedding_paths=None):
        model.eval()
        predictions, probabilities, true_labels = [], [], []

        with torch.no_grad():
            for x, attention_mask, y, _, input_ids in dataloader:
                x = x.to(device)
                attention_mask = attention_mask.to(device)
                y = y.to(device)
                input_ids = input_ids.to(device)

                logits, attn_sa, label_weights = model(x, attention_mask=attention_mask, return_attn=True, labels=y)
                if label_weights is not None:
                    attn_maps = rescale_attention_weights(label_weights, attention_mask)
                    for i in range(x.size(0)):  # Loop over batch
                        attn_rescaled = attn_maps[i]  # [num_labels, original_len]
                        input_len = int(attention_mask[i].sum().item())
                        tokens = tokenizer.convert_ids_to_tokens(input_ids[i][:input_len].tolist())

                        for label_index in range(attn_rescaled.shape[0]):
                            clean_token_attention_pairs = [
                                (tok, attn_rescaled[label_index][j])
                                for j, tok in enumerate(tokens)
                                if tok not in tokenizer.all_special_tokens
                            ]

                            if not clean_token_attention_pairs:
                                continue  # Skip if no valid tokens

                            token_list, attention_list = zip(*clean_token_attention_pairs)
                            

                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)

                predictions.extend(preds)
                probabilities.extend(probs)
                true_labels.extend(y.cpu().numpy())

        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        true_labels = np.array(true_labels)


        if save_to_csv and embedding_paths:
            df = pd.DataFrame({
                "actual labels": [list(tl) for tl in true_labels],
                "predicted_label": [list(pred) for pred in predictions],
                "probability": [list(prob) for prob in probabilities]
            })
            df.to_csv(save_to_csv, index=False)
            print(f"[✅] Predictions saved to {save_to_csv}")

        return predictions, probabilities
    
    list_lables=[r"Lamp/test_dataset_TEST474_shuffled.csv",
                 r"Lamp/test_dataset_DRBP206_shuffled.csv",
                 r"Lamp/test_dataset_PDB255_shuffled.csv",
                 r"Lamp/test_dataset_EZL_shuffled.csv"]
    
    list_embeddings = [
        r"Lamp/embeddings/testing/TEST474",
        r"Lamp/embeddings/testing/DRBP206",
        r"Lamp/embeddings/testing/PDB255",
        r"Lamp/embeddings/testing/EZL"]

    
    list_save=[r"Lamp/predictions/predictions_TEST474_FULL.csv",
               r"Lamp/predictions/predictions_DRBP206_FULL.csv",
               r"Lamp/predictions/predictions_PDB255_FULL.csv",
               r"Lamp/predictions/predictions_EZL_FULL.csv"]
    for i in range(4):

        test_data = pd.read_csv(
            list_lables[i],
            index_col=None)



        test_sentences = test_data['sequence']
        test_labels = test_data["label_vector"]


        test_dataset = MyDataset(embedding_dir=os.path.expanduser(list_embeddings[i]),label_vectors=test_labels)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        input_channel = 640
        # Loading model
        NeuralNetwork = torch.load(os.path.expanduser('Lamp/BestModel.pt'),weights_only=False)

        NeuralNetwork.to(device)
        check_label_embedding_alignment(os.path.expanduser(list_embeddings[i]),test_labels.tolist())


        preds, probs = predict(NeuralNetwork, test_loader, device,save_to_csv = os.path.expanduser(list_save[i]),embedding_paths=test_dataset.embedding_paths)
        evaluate(list_save[i])