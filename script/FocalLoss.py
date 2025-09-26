import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # tensor of shape [C] or scalar
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_term = (1 - pt) ** self.gamma

        loss = focal_term * BCE_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):  # scalar alpha
                alpha_tensor = torch.full((logits.size(1),), self.alpha, dtype=torch.float32).to(logits.device)
            elif isinstance(self.alpha, (list, np.ndarray, torch.Tensor)):  # list or tensor of class-wise alpha
                alpha_tensor = torch.tensor(self.alpha, dtype=torch.float32).to(logits.device)
            else:
                raise ValueError("alpha must be float, int, list, np.ndarray, or torch.Tensor")
            loss *= alpha_tensor.unsqueeze(0)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
