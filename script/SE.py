import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VALID_LABELS = torch.tensor([
    [1, 0, 0],  # DBP
    [0, 1, 0],  # RBP
    [1, 1, 0],  # DRBP
    [0, 0, 1],  # Neither
], dtype=torch.float).to(device)


class LabelAwareAttention(nn.Module):
    def __init__(self, num_labels, dim):
        super().__init__()
        self.label_embed = nn.Parameter(torch.randn(num_labels, dim))  # [3, 256]

    def forward(self, x_seq, return_attn=False):  # x_seq: [B, N, D]
        B, N, D = x_seq.shape
        queries = self.label_embed.unsqueeze(0).expand(B, -1, -1)  # [B, 3, D]
        attn_scores = torch.matmul(queries, x_seq.transpose(-1, -2)) / (D ** 0.5)  # [B, 3, N]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, 3, N]
        attn_out = torch.bmm(attn_weights, x_seq)  # [B, 3, D]
        if return_attn:
            return attn_out, attn_weights
        return attn_out


class CrossLabelBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads=2, dropout=0.1, gate_init=-3.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate_param = None
        self.gate_init = gate_init

    def reset_gate(self, num_labels):
        if (self.gate_param is None) or (self.gate_param.shape[1] != num_labels):
            p = torch.full((1, num_labels, 1), float(self.gate_init))
            self.gate_param = nn.Parameter(p)

    def make_label_mask(self, batch_label_vectors):
        B, L = batch_label_vectors.shape
        mask = torch.eye(L, dtype=torch.bool, device=batch_label_vectors.device).unsqueeze(0).repeat(B, 1, 1)
        for i in range(B):
            vec = batch_label_vectors[i]
            if vec[0] == 1 and vec[1] == 1:  # DRBP case
                mask[i, 0, 1] = True  # DBP -> RBP
                mask[i, 1, 0] = True  # RBP -> DBP
        return mask

    def forward(self, x_label, batch_label_vectors, return_attn=False):
        l_val = x_label.size(1)
        if self.gate_param is None or self.gate_param.size(1) != l_val:
            self.reset_gate(l_val)
        mask_bool = self.make_label_mask(batch_label_vectors)
        attn_mask = (~mask_bool).float() * -1e9
        attn_mask = attn_mask.repeat_interleave(self.attn.num_heads, dim=0)
        attn_out, attn_w = self.attn(x_label, x_label, x_label, attn_mask=attn_mask, need_weights=return_attn)
        gate = torch.sigmoid(self.gate_param)  # [1, L, 1]
        gated = gate * attn_out
        out = x_label + self.dropout(gated)
        out = self.norm(out)
        if return_attn:
            return out, attn_w, gate.detach()
        return out


# ====================== 新增模块（最有用） ======================
class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation 通道注意力块
    在 CNN 之后自适应地对通道进行加权，提升特征质量，显著提高 AUC
    """
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // reduction, bias=False)
        self.fc2 = nn.Linear(dim // reduction, dim, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: [B, L, D]
        # Squeeze: 全局平均池化
        avg_pool = x.mean(dim=1)  # [B, D]
        # Excitation
        excitation = self.relu(self.fc1(avg_pool))
        excitation = self.sigmoid(self.fc2(excitation))  # [B, D]
        # Scale
        return x * excitation.unsqueeze(1)


class ModelClassifier(nn.Module):
    def __init__(self, in_channel, num_labels=3, use_mhsa=True, use_cross_label_attention=True):
        super().__init__()
        self.use_mhsa = use_mhsa
        self.num_labels = num_labels
        self.use_cross_label_attention = use_cross_label_attention

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=256, kernel_size=3, stride=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=0.2)
        )
        # ====================== 新增 SE 块 ======================
        self.se_block = SqueezeExcitation(256)

        # Multihead Attention
        if self.use_mhsa:
            self.self_attn = nn.MultiheadAttention(256, num_heads=4, batch_first=True)
            self.mhsa_proj = nn.Identity()
            self.register_parameter("fusion_gate", nn.Parameter(torch.zeros(1, 1, 256)))

        # Label-aware attention
        self.label_attn = LabelAwareAttention(num_labels, 256)
        self.cross_label_block = CrossLabelBlock(256, n_heads=2, dropout=0.1, gate_init=-3.0)
        self.cross_label_block.reset_gate(num_labels)

        # Final classifier
        self.classifier = nn.Linear(256, 1)

    def forward(self, x, labels=None, attention_mask=None, return_attn=False):  # x: [B, N, 640]
        # CNN + SE 通道注意力（核心优化点）
        x_cnn = x.transpose(1, 2)                  # [B, in_channel, N]
        x_cnn = self.conv1(x_cnn)                  # [B, 256, N_down]
        x_cnn = x_cnn.transpose(1, 2)              # [B, N_down, 256]
        x_cnn = self.se_block(x_cnn)               # ← 新增 SE 块

        attn_weights_sa = None
        attn_weights_label = None
        attn_weights_cross = None
        gate_values = None

        if attention_mask is not None:
            attn_mask = attention_mask.unsqueeze(1).float()
            attn_mask = (F.max_pool1d(attn_mask, kernel_size=3, stride=3, padding=1) > 0.5).float()
            key_padding_mask = ~(attn_mask.squeeze(1).bool())
        else:
            key_padding_mask = None

        if self.use_mhsa:
            x_sa, attn_weights_sa = self.self_attn(x_cnn, x_cnn, x_cnn,
                                                   key_padding_mask=key_padding_mask,
                                                   need_weights=return_attn)
            if not return_attn:
                attn_weights_sa = None
            x_sa = self.mhsa_proj(x_sa)
            gate = torch.sigmoid(self.fusion_gate)
            x_seq = x_cnn + gate * x_sa
        else:
            x_seq = x_cnn

        if return_attn:
            x_label, attn_weights_label = self.label_attn(x_seq, return_attn=True)
        else:
            x_label = self.label_attn(x_seq)

        if self.use_cross_label_attention:
            if labels is None:
                raise ValueError("labels must be provided for cross-label attention masking")
            label_ids = torch.tensor([
                torch.where(torch.all(l.cpu() == VALID_LABELS.cpu(), dim=1))[0].item()
                for l in labels
            ], dtype=torch.long, device=labels.device)
            batch_label_vectors = VALID_LABELS[label_ids]

            if return_attn:
                x_label, attn_weights_cross, gate_values = self.cross_label_block(
                    x_label, batch_label_vectors, return_attn=True
                )
            else:
                x_label = self.cross_label_block(x_label, batch_label_vectors, return_attn=False)

        logits = self.classifier(x_label).squeeze(-1)  # [B, 3]

        return logits, attn_weights_sa, attn_weights_label
