
from pathlib import Path
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
DATA_DIR = ROOT / "dataset" / "processed" / "cnn_lstm"
OUT_DIR = Path(os.environ.get("ROAD_DAMAGE_OUTPUT_DIR", str(ROOT / "results" / "cnn_lstm" / "segmentation")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 1
EPOCHS = int(os.environ.get("ROAD_DAMAGE_EPOCHS", "10"))
LEARNING_RATE = 0.0001
POSITIVE_WEIGHT = 1.0
PATIENCE = max(10, int(os.environ.get("ROAD_DAMAGE_EPOCHS", "10")))
SEED = 42
THRESHOLD = 0.5

DEVICE = torch.device("cpu")
torch.set_num_threads(4)

# Keep V2 safe
BEST_PATH = OUT_DIR / "best_cnn_lstm_spatial_v3.pt"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================================================
# DATASET
# =========================================================

class VideoDataset(Dataset):
    def __init__(self, split):
        self.files = sorted((DATA_DIR / split).glob("*.npz"))

        if not self.files:
            raise FileNotFoundError(
                f"No processed files found in: {DATA_DIR / split}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]

        with np.load(path) as data:
            frames = data["frames"].copy()
            masks = data["masks"].copy()

        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"{path.name}: expected frames (T,H,W,3), got {frames.shape}"
            )

        if masks.ndim != 3 or frames.shape[:3] != masks.shape:
            raise ValueError(
                f"{path.name}: frame/mask shape mismatch: "
                f"{frames.shape}, {masks.shape}"
            )

        # Frames: (T,H,W,C) -> (T,C,H,W), normalized to [0,1]
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        frames = frames / 255.0

        # Accept binary masks stored as 0/1 or 0/255.
        masks = torch.from_numpy(masks).float().unsqueeze(1)

        if masks.max().item() > 1.0:
            masks = masks / 255.0

        masks = (masks >= 0.5).float()

        return frames, masks


# =========================================================
# ALGORITHM 4: CNN + SPATIAL LSTM
# =========================================================

class CNNLSTMSegmenter(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        batch, steps, channels, height, width = x.shape

        features = self.cnn(
            x.reshape(batch * steps, channels, height, width)
        )

        _, feature_channels, fh, fw = features.shape

        features = features.reshape(
            batch, steps, feature_channels, fh, fw
        )

        # Run LSTM through time at each spatial location.
        sequences = features.permute(0, 3, 4, 1, 2)
        sequences = sequences.reshape(
            batch * fh * fw, steps, feature_channels
        )

        temporal, _ = self.lstm(sequences)

        temporal = temporal.reshape(
            batch, fh, fw, steps, feature_channels
        ).permute(0, 3, 4, 1, 2)

        spatial_flat = features.reshape(
            batch * steps, feature_channels, fh, fw
        )

        temporal_flat = temporal.reshape(
            batch * steps, feature_channels, fh, fw
        )

        combined = torch.cat(
            [spatial_flat, temporal_flat], dim=1
        )

        logits = self.decoder(combined)

        logits = F.interpolate(
            logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        return logits.reshape(
            batch, steps, 1, height, width
        )


# =========================================================
# COMBINED BCE + DICE LOSS
# =========================================================

def combined_loss(logits, targets):
    pos_weight = torch.tensor(
        POSITIVE_WEIGHT,
        dtype=logits.dtype,
        device=logits.device,
    )

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
    )

    probabilities = torch.sigmoid(logits)
    dims = (1, 2, 3, 4)

    intersection = (probabilities * targets).sum(dim=dims)

    denominator = (
        probabilities.sum(dim=dims)
        + targets.sum(dim=dims)
    )

    dice = (
        2.0 * intersection + 1.0
    ) / (
        denominator + 1.0
    )

    return bce + (1.0 - dice.mean())


# =========================================================
# TRAINING AND VALIDATION METRICS
# =========================================================

def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    tp = fp = fn = tn = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for frames, masks in loader:
            frames = frames.to(DEVICE)
            masks = masks.to(DEVICE)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(frames)
            loss = combined_loss(logits, masks)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered.")

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

                optimizer.step()

            predictions = torch.sigmoid(logits) >= THRESHOLD
            targets = masks >= 0.5

            tp += (predictions & targets).sum().item()
            fp += (predictions & ~targets).sum().item()
            fn += (~predictions & targets).sum().item()
            tn += (~predictions & ~targets).sum().item()

            total_loss += loss.item()

    dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
    iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)

    total_pixels = max(tp + fp + fn + tn, 1)

    return {
        "loss": total_loss / max(len(loader), 1),
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "pred_fg": (tp + fp) / total_pixels,
        "actual_fg": (tp + fn) / total_pixels,
    }


# =========================================================
# MAIN TRAINING LOOP
# =========================================================

def main():
    train_data = VideoDataset("train")
    val_data = VideoDataset("val")

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = CNNLSTMSegmenter().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_val_dice = -1.0
    best_epoch = 0
    patience_counter = 0

    print("=" * 60)
    print("ALGORITHM 4: CNN + SPATIAL LSTM — V3")
    print("=" * 60)
    print("Device:", DEVICE)
    print("Training videos:", len(train_data))
    print("Validation videos:", len(val_data))
    print("Epochs:", EPOCHS)
    print("Positive weight:", POSITIVE_WEIGHT)
    print("Checkpoint:", BEST_PATH)

    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        train_metrics = run_epoch(
            model, train_loader, optimizer
        )

        val_metrics = run_epoch(model, val_loader)

        scheduler.step(val_metrics["dice"])

        print(
            f"\nEpoch {epoch}/{EPOCHS} "
            f"({time.time() - start:.1f}s)"
        )

        for label, metrics in (
            ("Train", train_metrics),
            ("Val  ", val_metrics),
        ):
            print(
                f"{label} | Loss {metrics['loss']:.4f} "
                f"Dice {metrics['dice']:.4f} "
                f"IoU {metrics['iou']:.4f} "
                f"Precision {metrics['precision']:.4f} "
                f"Recall {metrics['recall']:.4f} "
                f"Pred FG {metrics['pred_fg'] * 100:.2f}% "
                f"GT FG {metrics['actual_fg'] * 100:.2f}%"
            )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_metrics["loss"],
                    "val_dice": val_metrics["dice"],
                    "val_iou": val_metrics["iou"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "positive_weight": POSITIVE_WEIGHT,
                    "mask_threshold": 128,
                    "inference_threshold": THRESHOLD,
                    "architecture": "CNNLSTMSegmenter",
                },
                BEST_PATH,
            )

            print("Saved best V3 checkpoint.")

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break

    print("\nTraining complete.")
    print("Best epoch:", best_epoch)
    print("Best validation Dice:", round(best_val_dice, 4))
    print("Checkpoint:", BEST_PATH)


if __name__ == "__main__":
    main()
