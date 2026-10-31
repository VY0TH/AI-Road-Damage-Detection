"""ResNet-50 Model Definition and Factory."""
import torch.nn as nn
from torchvision import models


def get_resnet50(num_classes: int = 10, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model
