"""Step 9: Correction Technique - Class-Balanced Loss Functions.

Implements:
1. Weighted Cross-Entropy Loss using inverse class frequencies.
2. Focal Loss for mitigating extreme foreground class imbalance (e.g., D40 potholes vs. D00 cracks).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(class_counts: list[int] | dict[str, int], device: torch.device) -> torch.Tensor:
    if isinstance(class_counts, dict):
        counts = list(class_counts.values())
    else:
        counts = class_counts
    total = sum(counts)
    num_classes = len(counts)
    weights = [total / (num_classes * max(1, c)) for c in counts]
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    # Normalize weights so mean is 1.0
    return weights_tensor / weights_tensor.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
