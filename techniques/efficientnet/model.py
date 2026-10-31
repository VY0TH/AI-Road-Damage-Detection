"""EfficientNet-B0 Model Definition and Factory."""
import torch.nn as nn
from torchvision import models


def get_efficientnet_b0(num_classes: int = 10, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model
