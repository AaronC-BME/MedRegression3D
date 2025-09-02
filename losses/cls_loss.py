import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # softmax prob of the correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()

class TopKLoss(nn.Module):
    def __init__(self, k=10, reduction="mean"):
        super().__init__()
        self.k = k
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # shape: (B,)
        k = max(1, int(len(ce_loss) * self.k / 100))  # top k% elements
        topk_loss, _ = torch.topk(ce_loss, k)
        if self.reduction == "mean":
            return topk_loss.mean()
        else:
            return topk_loss.sum()