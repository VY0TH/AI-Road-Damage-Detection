"""Step 7: Vision Transformer (ViT-B/16) Classifier Architecture."""
import torch
import torch.nn as nn
from torchvision import models


def create_vit(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
    model = models.vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model
