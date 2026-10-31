"""Conditional DCGAN Architecture for Damage Crop Generation."""
from __future__ import annotations

import torch
import torch.nn as nn

LATENT_DIM = 100
EMBED_DIM = 32
IMAGE_SIZE = 64


class ConditionalGenerator(nn.Module):
    def __init__(self, num_classes: int = 10, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, EMBED_DIM)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + EMBED_DIM, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.ReLU(True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        conditioned = torch.cat([noise, self.embedding(labels)], dim=1)
        return self.net(self.fc(conditioned).view(-1, 256, 4, 4))


class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, IMAGE_SIZE * IMAGE_SIZE)
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        b, _, h, w = images.shape
        conditioned = torch.cat([images, self.embedding(labels).view(b, 1, h, w)], dim=1)
        return self.net(conditioned)
