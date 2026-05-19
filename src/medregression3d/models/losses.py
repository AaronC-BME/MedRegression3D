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


def coral_loss(logits, levels, importance_weights=None):
    loss = F.binary_cross_entropy_with_logits(logits, levels, reduction='none')
    if importance_weights is not None:
        loss = loss * importance_weights.view(1,-1)
    return loss.mean()


def coral_focal_loss(logits, levels, alpha=0.25, gamma=2.0):
    prob = torch.sigmoid(logits)
    pt = torch.where(levels == 1, prob, 1 - prob)
    ce_loss = F.binary_cross_entropy_with_logits(logits, levels, reduction='none')
    focal_weight = (1 - pt) ** gamma
    if alpha is not None:
        alpha_factor = torch.where(levels == 1, alpha, 1 - alpha)
        ce_loss = ce_loss * alpha_factor
    return (focal_weight * ce_loss).mean()


def coral_topk_loss(logits, levels, k=20):
    bce = F.binary_cross_entropy_with_logits(logits, levels, reduction='none')
    topk_vals, _ = torch.topk(bce, k=min(k, bce.shape[1]), dim=1)
    return topk_vals.mean()


def combined_bce_focal_loss(logits, levels, alpha=0.25, gamma=2.0, focal_weight=0.5, importance_weights=None):
    bce_loss = F.binary_cross_entropy_with_logits(logits, levels, reduction='none')
    if importance_weights is not None:
        bce_loss = bce_loss * importance_weights.view(1, -1)
    prob = torch.sigmoid(logits)
    pt = torch.where(levels == 1, prob, 1 - prob)
    focal_term = (1 - pt) ** gamma
    if alpha is not None:
        alpha_factor = torch.where(levels == 1, alpha, 1 - alpha)
        focal_term = focal_term * alpha_factor
    focal_loss = focal_term * bce_loss
    total_loss = (1 - focal_weight) * bce_loss + focal_weight * focal_loss
    return total_loss.mean()


def combined_bce_topk_loss(logits, levels, topk=10, topk_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(logits, levels, reduction='none')  # shape [B, K-1]
    topk_vals, _ = torch.topk(bce, k=min(topk, bce.shape[1]), dim=1)  # shape [B, topk]
    print(f"Using the following k value: {min(topk, bce.shape[1])}")  # Debugging line
    topk_loss = topk_vals.mean()
    full_bce_loss = bce.mean()
    return (1 - topk_weight) * full_bce_loss + topk_weight * topk_loss


def label_to_levels(labels, num_classes):
    batch_size = labels.size(0)
    levels = torch.zeros(batch_size, num_classes - 1, device=labels.device)
    for i, label in enumerate(labels):
        levels[i, :int(label)] = 1
    return levels


def combined_coral_mae_loss(logits, levels, labels, mae_weight=0.2, importance_weights=None):
    coral = coral_loss(logits, levels, importance_weights)
    pred_soft_label = torch.sigmoid(logits).sum(dim=1)  # shape: [B]
    mae = F.l1_loss(pred_soft_label, labels.float())
    return coral + mae_weight * mae
