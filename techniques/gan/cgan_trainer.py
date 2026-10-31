"""Conditional GAN Trainer & Sample Generator."""
from __future__ import annotations

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "06_gan_augmentation"))

from gan_model import ConditionalGenerator, ConditionalDiscriminator, LATENT_DIM, IMAGE_SIZE
from train_gan import train_conditional_gan
from generate_gan_samples import generate_samples
