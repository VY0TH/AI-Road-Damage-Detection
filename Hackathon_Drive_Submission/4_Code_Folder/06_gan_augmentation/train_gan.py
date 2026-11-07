"""Step 6: Conditional GAN Training on TRAINING DATA ONLY.

Strictly trains the generator and discriminator using only real training images.
Validation and test data are never seen by the GAN.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

from gan_model import ConditionalDiscriminator, ConditionalGenerator, IMAGE_SIZE, LATENT_DIM

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class TrainingCropsDataset(Dataset):
    def __init__(self, train_dir: Path, class_names: list[str]):
        self.train_dir = Path(train_dir).resolve()
        # Validation guard: ensure directory is train only!
        if "val" in str(self.train_dir).lower() or "test" in str(self.train_dir).lower():
            raise ValueError(f"SECURITY VIOLATION: GAN must be trained on TRAINING data only! Rejected: {self.train_dir}")

        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.samples: list[tuple[Path, int]] = []

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        for c_name in self.class_names:
            c_dir = self.train_dir / c_name
            if c_dir.is_dir():
                for p in sorted(c_dir.iterdir()):
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                        self.samples.append((p, self.class_to_idx[c_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        p, label = self.samples[index]
        with Image.open(p) as img:
            t = self.transform(img.convert("RGB"))
        return t, label


def train_conditional_gan(
    train_dir: Path,
    output_checkpoint_dir: Path,
    class_names: list[str],
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.0002,
    device: str | None = None,
) -> Path:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_checkpoint_dir = Path(output_checkpoint_dir).resolve()
    output_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = TrainingCropsDataset(train_dir, class_names)
    if len(dataset) == 0:
        raise ValueError(f"No training samples found in {train_dir}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    net_g = ConditionalGenerator(num_classes=len(class_names), latent_dim=LATENT_DIM).to(dev)
    net_d = ConditionalDiscriminator(num_classes=len(class_names)).to(dev)

    criterion = nn.BCELoss()
    opt_g = torch.optim.Adam(net_g.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(net_d.parameters(), lr=lr, betas=(0.5, 0.999))

    print(f"Training Conditional GAN on {len(dataset)} real training samples for {epochs} epochs on {dev}...")

    for epoch in range(1, epochs + 1):
        total_d_loss, total_g_loss = 0.0, 0.0
        for real_imgs, labels in dataloader:
            b_size = real_imgs.size(0)
            real_imgs = real_imgs.to(dev)
            labels = labels.to(dev)

            real_target = torch.ones(b_size, 1, device=dev)
            fake_target = torch.zeros(b_size, 1, device=dev)

            # Train Discriminator
            opt_d.zero_grad()
            out_real = net_d(real_imgs, labels)
            d_loss_real = criterion(out_real, real_target)

            noise = torch.randn(b_size, LATENT_DIM, device=dev)
            fake_imgs = net_g(noise, labels)
            out_fake = net_d(fake_imgs.detach(), labels)
            d_loss_fake = criterion(out_fake, fake_target)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_d.step()
            total_d_loss += d_loss.item()

            # Train Generator
            opt_g.zero_grad()
            out_g = net_d(fake_imgs, labels)
            g_loss = criterion(out_g, real_target)
            g_loss.backward()
            opt_g.step()
            total_g_loss += g_loss.item()

        avg_d = total_d_loss / max(1, len(dataloader))
        avg_g = total_g_loss / max(1, len(dataloader))
        print(f"Epoch [{epoch:2d}/{epochs:2d}] - Loss D: {avg_d:.4f}, Loss G: {avg_g:.4f}")

    final_ckpt = output_checkpoint_dir / "cgan_generator_final.pt"
    torch.save(net_g.state_dict(), final_ckpt)
    print(f"Saved trained cGAN Generator checkpoint to: {final_ckpt}")
    return final_ckpt
