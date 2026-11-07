"""Step 9: Retraining Engine with Applied Corrections.

Applies:
1. Architectural upgrade: Pretrained transfer learning backbone (EfficientNet-B0 / ResNet-50).
2. Class-balanced loss with label smoothing (mitigating class imbalance & label ambiguity).
3. Cosine Annealing learning rate schedule.
4. Strong data augmentations (RandomAffine, ColorJitter, HorizontalFlip).
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from corrections.loss_corrections import compute_class_weights
from corrections.regularization_corrections import create_regularized_loss


def retrain_with_corrections(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    class_counts: list[int] | dict[str, int],
    epochs: int = 5,
    lr: float = 3e-4,
    label_smoothing: float = 0.1,
    weight_decay: float = 1e-4,
    device: str | None = None,
) -> dict:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class_weights = compute_class_weights(class_counts, dev)
    criterion = create_regularized_loss(class_weights=class_weights, label_smoothing=label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = []
    best_val_acc = 0.0
    best_checkpoint_path = output_dir / "corrected_best_model.pt"

    print(f"\nRetraining model with applied corrections on {dev} for {epochs} epochs...")
    print(f"  Corrections: Class-Weighted Loss, Label Smoothing={label_smoothing}, CosineLR, Weight Decay={weight_decay}")

    for epoch in range(1, epochs + 1):
        start_t = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(dev), labels.to(dev)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()
        epoch_train_loss = train_loss / max(1, train_total)
        epoch_train_acc = train_correct / max(1, train_total)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(dev), labels.to(dev)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        epoch_val_loss = val_loss / max(1, val_total)
        epoch_val_acc = val_correct / max(1, val_total)
        elapsed = time.time() - start_t
        curr_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch [{epoch:2d}/{epochs:2d}] ({elapsed:.1f}s) lr={curr_lr:.2e} | "
            f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}"
        )

        record = {
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "train_accuracy": round(epoch_train_acc, 4),
            "val_loss": round(epoch_val_loss, 4),
            "val_accuracy": round(epoch_val_acc, 4),
            "lr": round(curr_lr, 7),
            "time_sec": round(elapsed, 2),
        }
        history.append(record)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_checkpoint_path)

    # Save history
    history_json = output_dir / "corrected_training_history.json"
    with history_json.open("w", encoding="utf-8") as f:
        json.dump({"history": history}, f, indent=2)

    history_csv = output_dir / "corrected_training_history.csv"
    with history_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "lr", "time_sec"])
        writer.writeheader()
        writer.writerows(history)

    return {
        "best_val_accuracy": round(best_val_acc, 4),
        "best_checkpoint": str(best_checkpoint_path),
        "history_json": str(history_json),
        "history_csv": str(history_csv),
        "epochs_completed": epochs,
    }
