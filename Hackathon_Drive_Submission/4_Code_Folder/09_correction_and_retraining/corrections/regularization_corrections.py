"""Step 9: Correction Technique - Regularization & Label Smoothing."""
from __future__ import annotations

import torch
import torch.nn as nn


def create_regularized_loss(
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.1,
) -> nn.Module:
    """Creates a regularized CrossEntropyLoss with label smoothing and optional class weighting."""
    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )
