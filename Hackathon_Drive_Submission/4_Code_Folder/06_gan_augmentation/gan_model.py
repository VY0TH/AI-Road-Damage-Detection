"""Step 6: Conditional GAN Architecture for Image Augmentation.

Implements a class-conditional DCGAN with label embeddings:
- ConditionalGenerator: Maps (latent_dim + class_embed) -> 64x64 RGB synthetic crop
- ConditionalDiscriminator: Evaluates (RGB image + class_embed) -> real/fake validity
"""
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
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embed = self.embedding(labels)
        conditioned = torch.cat([noise, embed], dim=1)
        x = self.fc(conditioned).view(-1, 256, 4, 4)
        return self.net(x)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, IMAGE_SIZE * IMAGE_SIZE)
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=4, stride=2, padding=1),  # 3 image channels + 1 label channel
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        b, _, h, w = images.shape
        label_map = self.embedding(labels).view(b, 1, h, w)
        x = torch.cat([images, label_map], dim=1)
        return self.net(x)
