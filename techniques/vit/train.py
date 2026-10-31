"""Vision Transformer (ViT-B/16) Training Pipeline."""
from __future__ import annotations

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "01_load_image_dataset"))

from dataset_loader import ImageFolderDataset, create_dataloader
from model import get_vit_b16
import pipeline_config as cfg


def train_vit(
    train_dir: Path,
    val_dir: Path,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 5e-5,
) -> Path:
    dev = cfg.DEVICE
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((cfg.DEFAULT_IMAGE_SIZE, cfg.DEFAULT_IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((cfg.DEFAULT_IMAGE_SIZE, cfg.DEFAULT_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = ImageFolderDataset(train_dir, cfg.CLASS_NAMES, transform=train_transform)
    val_ds = ImageFolderDataset(val_dir, cfg.CLASS_NAMES, transform=val_transform)

    train_loader = create_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_ds, batch_size=batch_size, shuffle=False)

    model = get_vit_b16(num_classes=len(cfg.CLASS_NAMES), pretrained=True).to(dev)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = 0.0
    best_ckpt = output_dir / "vit_best.pt"

    print(f"Training ViT-B/16 on {len(train_ds)} samples for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(dev), labels.to(dev)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (preds.argmax(dim=1) == labels).sum().item()
            total += imgs.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(dev), labels.to(dev)
                preds = model(imgs)
                val_correct += (preds.argmax(dim=1) == labels).sum().item()
                val_total += imgs.size(0)

        val_acc = val_correct / max(1, val_total)
        print(f"Epoch [{epoch:2d}/{epochs:2d}] Loss: {train_loss/max(1,total):.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)

    print(f"ViT-B/16 checkpoint saved: {best_ckpt}")
    return best_ckpt


if __name__ == "__main__":
    train_vit(
        train_dir=cfg.DATASET_ROOT / "processed" / "efficientnet_crops" / "train",
        val_dir=cfg.DATASET_ROOT / "processed" / "efficientnet_crops" / "val",
        output_dir=cfg.RESULTS_DIR / "vit",
        epochs=1,
    )
