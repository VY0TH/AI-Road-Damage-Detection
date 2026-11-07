"""Step 7: EfficientNet-B0 Classifier Architecture."""
import torch
import torch.nn as nn
from torchvision import models


def create_efficientnet(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model
