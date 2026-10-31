"""Vision Transformer (ViT-B/16) Model Definition and Factory."""
import torch.nn as nn
from torchvision import models


def get_vit_b16(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
    model = models.vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model
