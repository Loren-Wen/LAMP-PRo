# import torch
# import torch.utils.data as Data
# import ast
#
#
# class MyDataset(Data.Dataset):
#     def __init__(self, embeddings, masks, labels):
#         self.embeddings = embeddings  # Tensor [N, L, D]
#         self.masks = masks            # Tensor [N, L]
#         self.labels = labels          # List[str] or List[List[int]]
#
#     def __len__(self):
#         return len(self.embeddings)
#
#     def __getitem__(self, index):
#         emb = self.embeddings[index]          # [L, D]
#         mask = self.masks[index]              # [L]
#         label_data = self.labels[index]
#
#         if isinstance(label_data, str):
#             label_list = ast.literal_eval(label_data)
#         else:
#             label_list = label_data
#
#         label = torch.tensor(label_list, dtype=torch.float)
#         return emb, mask, label

# import torch
# from torch.utils.data import Dataset
# import os
# import glob
# import re
# import ast
#
# def natural_sort_key(s):
#     return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
#
# class MyDataset(Dataset):
#     def __init__(self, embedding_dir, labels, prefix="seq"):
#         self.embedding_files = sorted(
#             glob.glob(os.path.join(embedding_dir, f"{prefix}_*_padded.pt")),
#             key=natural_sort_key
#         )
#         self.mask_files = sorted(
#             glob.glob(os.path.join(embedding_dir, f"{prefix}_*_mask.pt")),
#             key=natural_sort_key
#         )
#         self.label_files = labels
#
#         assert len(self.embedding_files) == len(self.mask_files) == len(self.label_files), \
#             "Mismatch in number of files!"
#
#     def __len__(self):
#         return len(self.embedding_files)
#
#     def __getitem__(self, index):
#         emb = torch.load(self.embedding_files[index], map_location='cpu')
#         mask = torch.load(self.mask_files[index], map_location='cpu')
#         label_data = self.label_files[index]
#
#         if isinstance(label_data, str):
#             label = torch.tensor(ast.literal_eval(label_data), dtype=torch.float)
#         else:
#             label = torch.tensor(label_data, dtype=torch.float)
#
#         return emb, mask, label

import os
import re
from torch.utils.data import Dataset
import torch
import ast

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

class MyDataset(Dataset):
    def __init__(self, embedding_dir, label_vectors):
        self.embedding_paths = sorted([
            os.path.join(embedding_dir, fname)
            for fname in os.listdir(embedding_dir)
            if fname.endswith('.pt')
        ], key=natural_sort_key)

        assert len(self.embedding_paths) == len(label_vectors)
        self.label_vectors = label_vectors

    def __len__(self):
        return len(self.embedding_paths)

    def __getitem__(self, idx):
        data = torch.load(self.embedding_paths[idx])  # [L, D]
        embedding = data["embedding"]
        input_ids = data["input_ids"]

        raw_label = self.label_vectors[idx]
        if isinstance(raw_label, str):
            label_list = ast.literal_eval(raw_label)
        else:
            label_list = raw_label

        label_vector = torch.tensor(label_list, dtype=torch.float)
        filename = os.path.basename(self.embedding_paths[idx])
        return embedding, label_vector, filename, input_ids


