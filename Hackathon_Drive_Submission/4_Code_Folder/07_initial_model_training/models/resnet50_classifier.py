"""Step 7: ResNet-50 Classifier Architecture."""
import torch
import torch.nn as nn
from torchvision import models


def create_resnet50(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model
