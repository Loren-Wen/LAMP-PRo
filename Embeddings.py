import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import pickle
from typing import List
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import os
import glob


def get_embeddings_with_overlap_batched(
    sequences: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    max_len: int = 1024,
    stride: int = 512,
) -> List[dict]:
    """
    Get ESM2 embeddings for a batch of sequences using overlapping chunks.
    Returns: list of dicts with 'embedding' and 'input_ids'.
    """
    all_outputs = []

    is_data_parallel = isinstance(model, torch.nn.DataParallel)
    hidden_size = model.module.config.hidden_size if is_data_parallel else model.config.hidden_size

    for seq in sequences:
        tokens = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        L = tokens["input_ids"].shape[1] - 2
        D = hidden_size

        full_embed = torch.zeros(L, D).to(device)
        full_input_ids = torch.full((L,), tokenizer.pad_token_id).to(device)
        counts = torch.zeros(L).to(device)

        for start in range(0, L, stride):
            end = min(start + max_len, L)
            chunk = seq[start:end]
            inputs = tokenizer(chunk, return_tensors="pt", add_special_tokens=True).to(device)

            with torch.no_grad():
                output = model(**inputs)
                rep = output.last_hidden_state.squeeze(0)[1:-1]
                ids = inputs["input_ids"].squeeze(0)[1:-1]

            full_embed[start:end] += rep
            full_input_ids[start:end] = ids
            counts[start:end] += 1

            if end == L:
                break

        full_embed = full_embed / counts.unsqueeze(1)

        all_outputs.append({
            'embedding': full_embed.cpu(),
            'input_ids': full_input_ids.cpu()
        })
    return all_outputs


class SequenceDataset(Dataset):
    def __init__(self, sequences: List[str]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def save_embeddings_to_disk(
    split_name: str,
    sequences: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_len: int,
    stride: int,
):
    dataset = SequenceDataset(sequences)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Determine the directory path based on the split name
    if split_name in ["train", "test"]:
        save_dir = os.path.expanduser(f"/workspace/LAMP-PRo/Lamp/embeddings/training/{split_name}")
    else:
        save_dir = os.path.expanduser(f"/workspace/LAMP-PRo/Lamp/embeddings/testing/{split_name}")
    
    os.makedirs(save_dir, exist_ok=True)

    all_lengths = []
    index = 0
    for batch in loader:
        outputs = get_embeddings_with_overlap_batched(
            batch, tokenizer, model, device, max_len=max_len, stride=stride
        )
        for item in outputs:
            torch.save(item, os.path.join(save_dir, f"seq_{index}.pt"))
            all_lengths.append(item["embedding"].shape[0])
            index += 1

    with open(os.path.join(save_dir, "lengths.pkl"), "wb") as f:
        pickle.dump(all_lengths, f)

    print(f"Saved {index} embeddings (and input_ids) for {split_name}.")


if __name__ == '__main__':
    # Set Random Seed
    seed_val = 41
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # GPU training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1000
    stride = 512
    max_len = 1024
    model_name = "facebook/esm2_t30_150M_UR50D"

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    model.to(device)
    model.eval()

    # Generate Embeddings for Train Datasets
    train_data = pd.read_csv('/workspace/LAMP-PRo/Lamp/Training_with_Label_shuffled.csv')
    test_data = pd.read_csv('/workspace/LAMP-PRo/Lamp/Validation_with_Label_shuffled.csv')

    save_embeddings_to_disk(
        "train",
        train_data['sequence'].tolist(),
        tokenizer, model, device, batch_size, max_len, stride
    )
    save_embeddings_to_disk(
        "test",
        test_data['sequence'].tolist(),
        tokenizer, model, device, batch_size, max_len, stride
    )

    # Generate Embeddings for Test Datasets
    test_files = [
        "/workspace/LAMP-PRo/Lamp/test_dataset_DRBP206_shuffled.csv",
        "/workspace/LAMP-PRo/Lamp/test_dataset_EZL_shuffled.csv",
        "/workspace/LAMP-PRo/Lamp/test_dataset_PDB255_shuffled.csv",
        "/workspace/LAMP-PRo/Lamp/test_dataset_TEST474_shuffled.csv"
    ]

    for file in test_files:
        df = pd.read_csv(file)
        dataset_name = os.path.basename(file).replace("test_dataset_", "").replace("_shuffled.csv", "")
        save_embeddings_to_disk(
            dataset_name,
            df['sequence'].tolist(),
            tokenizer, model, device, batch_size, max_len, stride
        )
