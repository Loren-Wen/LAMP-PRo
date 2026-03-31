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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from sklearn.metrics import multilabel_confusion_matrix, classification_report
import csv
from tqdm.auto import tqdm

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, f1_score, matthews_corrcoef, average_precision_score
)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def check_label_embedding_alignment(emb_dir, label_vectors):
    files = [f for f in os.listdir(emb_dir) if f.startswith("seq_") and f.endswith(".pt")]
    files = sorted(files, key=natural_sort_key)

    if len(files) != len(label_vectors):
        raise ValueError(
            f"Number of embedding files ({len(files)}) does not match number of label vectors ({len(label_vectors)})."
        )

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


def rescale_attention_weights(label_weights, attention_mask):
    attn_maps = []
    for i in range(label_weights.size(0)):
        original_len = int(attention_mask[i].sum().item())
        rescaled = []
        for l in range(label_weights.size(1)):
            weights = label_weights[i, l].detach().cpu().numpy()
            rescaled_weights = zoom(weights, original_len / len(weights))
            rescaled.append(rescaled_weights)
        attn_maps.append(np.stack(rescaled))
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
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
], dtype=torch.float)


def visualize_combined_attention(
    sequence, attention_weights, label_index, sample_index, dataset_name, k=30
):
    attention_weights = np.array(attention_weights)
    cmap = plt.colormaps.get_cmap('YlOrRd')
    label_map = {0: "DBP", 1: "RBP", 2: "Neither"}
    label_name = label_map.get(label_index, f"Unknown ({label_index})")
    norm = mcolors.Normalize(vmin=attention_weights.min(), vmax=attention_weights.max() + 1e-8)

    topk_idx = np.argsort(attention_weights)[-k:]
    topk_idx = np.sort(topk_idx)
    sliced_sequence = [sequence[i] for i in topk_idx]
    sliced_attention_weights = [attention_weights[i] for i in topk_idx]

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        figsize=(max(8, len(sliced_sequence) // 2), 5),
        gridspec_kw={'height_ratios': [1, 2]}
    )

    im = ax1.imshow(np.array(sliced_attention_weights)[np.newaxis, :], cmap=cmap, aspect='auto')
    total_length = len(sequence)
    ax1.set_title(f"Attention Heatmap - {label_name}", fontsize=18, fontweight='bold')
    ax1.set_yticks([])
    ax1.set_xticks(np.arange(len(sliced_sequence)))
    ax1.set_xticklabels(sliced_sequence, fontsize=12, fontweight='bold', rotation=75, ha='right')
    cbar = fig.colorbar(im, ax=ax1, orientation='vertical', pad=0.02)
    for label in cbar.ax.get_yticklabels():
        label.set_weight('bold')

    ax2.set_xlim(-0.5, len(sliced_sequence) - 0.5)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    plt.tight_layout(pad=3.0)

    image_dir = os.path.expanduser(f'/workspace/LAMP-PRo/Lamp/visualization/{dataset_name}/images')
    log_dir = os.path.expanduser(f'/workspace/LAMP-PRo/Lamp/visualization/{dataset_name}/log')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    image_filename = f"attn_label{label_index}_sample{sample_index}.pdf"
    save_path = os.path.join(image_dir, image_filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    log_csv_path = os.path.join(log_dir, f"attention_log_{dataset_name}.csv")
    file_exists = os.path.isfile(log_csv_path)
    with open(log_csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['image_filename', 'full_sequence', 'sequence_length'])
        writer.writerow([image_filename, "".join(sequence), total_length])


def invalid_label_penalty(y_pred_sigmoid, weight=0.1):
    y_pred_bin = (y_pred_sigmoid > 0.5).float()
    penalties = torch.stack([
        ~(torch.any(torch.all(y_pred_bin[i].unsqueeze(0) == VALID_LABELS, dim=1)))
        for i in range(y_pred_bin.size(0))
    ]).float().to(y_pred_bin.device)
    return weight * penalties.mean()


def predict(model, dataloader, device, save_to_csv=None, dataset_name=None, tokenizer=None):
    model.eval()
    predictions, probabilities, true_labels = [], [], []

    pbar = tqdm(dataloader, desc=f"Predicting [{dataset_name}]", total=len(dataloader), leave=False)
    with torch.no_grad():
        for i, (x, attention_mask, y, filenames, input_ids) in enumerate(pbar):
            x = x.to(device)
            attention_mask = attention_mask.to(device)
            y = y.to(device)
            input_ids = input_ids.to(device)

            logits, attn_sa, label_weights = model(x, attention_mask=attention_mask, return_attn=True, labels=y)

            if label_weights is not None and tokenizer is not None:
                attn_maps = rescale_attention_weights(label_weights, attention_mask)
                for j in range(x.size(0)):
                    attn_rescaled = attn_maps[j]
                    input_len = int(attention_mask[j].sum().item())
                    tokens = tokenizer.convert_ids_to_tokens(input_ids[j][:input_len].tolist())

                    for label_index in range(attn_rescaled.shape[0]):
                        clean_token_attention_pairs = [
                            (tok, attn_rescaled[label_index][k])
                            for k, tok in enumerate(tokens)
                            if tok not in tokenizer.all_special_tokens
                        ]

                        if not clean_token_attention_pairs:
                            continue

                        token_list, attention_list = zip(*clean_token_attention_pairs)
                        visualize_combined_attention(
                            token_list, attention_list, label_index,
                            sample_index=(i * dataloader.batch_size) + j,
                            dataset_name=dataset_name, k=30
                        )

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            predictions.extend(preds)
            probabilities.extend(probs)
            true_labels.extend(y.cpu().numpy())

            pbar.set_postfix({
                "batch": f"{i + 1}/{len(dataloader)}",
                "samples": len(predictions)
            })

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    true_labels = np.array(true_labels)

    if save_to_csv:
        df = pd.DataFrame({
            "actual labels": [list(tl) for tl in true_labels],
            "predicted_label": [list(pred) for pred in predictions],
            "probability": [list(prob) for prob in probabilities]
        })
        df.to_csv(save_to_csv, index=False)

    return predictions, probabilities


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

    list_labels = [
        "/workspace/LAMP-PRo/Lamp/test_dataset_TEST474_shuffled.csv",
        "/workspace/LAMP-PRo/Lamp/test_dataset_DRBP206_shuffled.csv",
        "/workspace/LAMP-PRo/Lamp/test_dataset_PDB255_shuffled.csv",
        "/workspace/LAMP-PRo/Lamp/test_dataset_EZL_shuffled.csv"
    ]

    list_embeddings = [
        "/workspace/LAMP-PRo/Lamp/embeddings/testing/TEST474",
        "/workspace/LAMP-PRo/Lamp/embeddings/testing/DRBP206",
        "/workspace/LAMP-PRo/Lamp/embeddings/testing/PDB255",
        "/workspace/LAMP-PRo/Lamp/embeddings/testing/EZL"
    ]

    list_save = [
        "/workspace/LAMP-PRo/Lamp/predictions/predictions_TEST474_FULL.csv",
        "/workspace/LAMP-PRo/Lamp/predictions/predictions_DRBP206_FULL.csv",
        "/workspace/LAMP-PRo/Lamp/predictions/predictions_PDB255_FULL.csv",
        "/workspace/LAMP-PRo/Lamp/predictions/predictions_EZL_FULL.csv"
    ]

    dataset_bar = tqdm(
        range(len(list_labels)),
        desc="Overall datasets",
        total=len(list_labels)
    )

    for i in dataset_bar:
        dataset_name = os.path.basename(list_labels[i]).replace("test_dataset_", "").replace("_shuffled.csv", "")
        dataset_bar.set_postfix({"dataset": dataset_name})

        test_data = pd.read_csv(list_labels[i], index_col=None)
        test_labels = test_data["label_vector"]

        test_embedding_dir = os.path.expanduser(list_embeddings[i])
        test_save_path = os.path.expanduser(list_save[i])

        test_dataset = MyDataset(embedding_dir=test_embedding_dir, label_vectors=test_labels)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        NeuralNetwork = ModelClassifier(in_channel=640, num_labels=3, use_mhsa=True)
        NeuralNetwork.load_state_dict(torch.load("/workspace/LAMP-PRo/Lamp/BestModel.pt", map_location=device))
        NeuralNetwork.to(device)

        check_label_embedding_alignment(test_embedding_dir, test_labels.tolist())

        preds, probs = predict(
            NeuralNetwork,
            test_loader,
            device,
            save_to_csv=test_save_path,
            dataset_name=dataset_name,
            tokenizer=tokenizer
        )

        evaluate(test_save_path)
