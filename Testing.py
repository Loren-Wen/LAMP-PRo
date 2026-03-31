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
from tqdm import tqdm


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

        for idx, file in enumerate(files):
            expected = f"seq_{idx}.pt"
            if file != expected:
                raise ValueError(f" Mismatch at index {idx}: expected '{expected}', found '{file}'")

        print(" All embedding files correctly aligned with label vectors.")


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
        [1, 0, 0],  # DBP
        [0, 1, 0],  # RBP
        [1, 1, 0],  # DBRP
        [0, 0, 1],  # Neither
    ], dtype=torch.float).to(device)


    def invalid_label_penalty(y_pred_sigmoid, weight=0.1):
        y_pred_bin = (y_pred_sigmoid > 0.5).float()
        penalties = torch.stack([
            ~(torch.any(torch.all(y_pred_bin[i].unsqueeze(0) == VALID_LABELS, dim=1)))
            for i in range(y_pred_bin.size(0))
        ]).float().to(y_pred_bin.device)
        return weight * penalties.mean()


    def predict(model, dataloader, device, dataset_name="", save_to_csv=None, embedding_paths=None):
        model.eval()
        predictions, probabilities, true_labels = [], [], []

        with torch.no_grad():
            batch_bar = tqdm(dataloader, desc=f"  Predicting [{dataset_name}]",
                             unit="batch", leave=False)

            for x, attention_mask, y, _, input_ids in batch_bar:
                x = x.to(device)
                attention_mask = attention_mask.to(device)
                y = y.to(device)
                input_ids = input_ids.to(device)

                logits, attn_sa, label_weights = model(x, attention_mask=attention_mask, return_attn=True, labels=y)
                if label_weights is not None:
                    attn_maps = rescale_attention_weights(label_weights, attention_mask)
                    for i in range(x.size(0)):
                        attn_rescaled = attn_maps[i]
                        input_len = int(attention_mask[i].sum().item())
                        tokens = tokenizer.convert_ids_to_tokens(input_ids[i][:input_len].tolist())

                        for label_index in range(attn_rescaled.shape[0]):
                            clean_token_attention_pairs = [
                                (tok, attn_rescaled[label_index][j])
                                for j, tok in enumerate(tokens)
                                if tok not in tokenizer.all_special_tokens
                            ]
                            if not clean_token_attention_pairs:
                                continue
                            token_list, attention_list = zip(*clean_token_attention_pairs)

                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)

                predictions.extend(preds)
                probabilities.extend(probs)
                true_labels.extend(y.cpu().numpy())

            batch_bar.close()

        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        true_labels = np.array(true_labels)

        if save_to_csv and embedding_paths:
            os.makedirs(os.path.dirname(save_to_csv), exist_ok=True)
            df = pd.DataFrame({
                "actual labels": [list(tl) for tl in true_labels],
                "predicted_label": [list(pred) for pred in predictions],
                "probability": [list(prob) for prob in probabilities]
            })
            df.to_csv(save_to_csv, index=False)

        return predictions, probabilities


    list_labels = [
        r"/workspace/LAMP-PRo/Lamp/test_dataset_TEST474_shuffled.csv",
        r"/workspace/LAMP-PRo/Lamp/test_dataset_DRBP206_shuffled.csv",
        r"/workspace/LAMP-PRo/Lamp/test_dataset_PDB255_shuffled.csv",
        r"/workspace/LAMP-PRo/Lamp/test_dataset_EZL_shuffled.csv",
    ]

    list_embeddings = [
        r"/workspace/LAMP-PRo/Lamp/embeddings/testing/TEST474",
        r"/workspace/LAMP-PRo/Lamp/embeddings/testing/DRBP206",
        r"/workspace/LAMP-PRo/Lamp/embeddings/testing/PDB255",
        r"/workspace/LAMP-PRo/Lamp/embeddings/testing/EZL",
    ]

    list_save = [
        r"/workspace/LAMP-PRo/Lamp/predictions/predictions_TEST474_FULL.csv",
        r"/workspace/LAMP-PRo/Lamp/predictions/predictions_DRBP206_FULL.csv",
        r"/workspace/LAMP-PRo/Lamp/predictions/predictions_PDB255_FULL.csv",
        r"/workspace/LAMP-PRo/Lamp/predictions/predictions_EZL_FULL.csv",
    ]

    dataset_names = ["TEST474", "DRBP206", "PDB255", "EZL"]

    # 外层进度条：遍历所有测试集
    dataset_bar = tqdm(range(4), desc="Evaluating datasets", unit="dataset")

    for i in dataset_bar:
        name = dataset_names[i]
        dataset_bar.set_postfix(current=name)
        tqdm.write(f"\n{'='*50}")
        tqdm.write(f"Processing dataset: {name}")
        tqdm.write(f"{'='*50}")

        test_data = pd.read_csv(list_labels[i], index_col=None)
        test_sentences = test_data['sequence']
        test_labels = test_data["label_vector"]

        test_dataset = MyDataset(
            embedding_dir=os.path.expanduser(list_embeddings[i]),
            label_vectors=test_labels
        )
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)


        NeuralNetwork = ModelClassifier(in_channel=640, num_labels=3, use_mhsa=True)
        state_dict = torch.load(
            os.path.expanduser('/workspace/LAMP-PRo/Lamp/BestModel.pt'),
            map_location=device,
            weights_only=True
        )
        NeuralNetwork.load_state_dict(state_dict)
        NeuralNetwork.to(device)

        check_label_embedding_alignment(
            os.path.expanduser(list_embeddings[i]),
            test_labels.tolist()
        )

        preds, probs = predict(
            NeuralNetwork, test_loader, device,
            dataset_name=name,
            save_to_csv=os.path.expanduser(list_save[i]),
            embedding_paths=test_dataset.embedding_paths
        )

        tqdm.write(f"  ✓ Predictions saved → {list_save[i]}")
        evaluate(list_save[i])

    dataset_bar.close()
    tqdm.write("\nAll datasets evaluated.")
