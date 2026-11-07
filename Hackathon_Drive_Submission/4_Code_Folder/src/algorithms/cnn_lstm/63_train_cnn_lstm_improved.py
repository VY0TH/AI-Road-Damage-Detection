
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==================================================
# CONFIGURATION
# ==================================================
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
DATA_DIR = ROOT / "dataset" / "processed" / "cnn_lstm"
OUT_DIR = ROOT / "results" / "cnn_lstm" / "improved"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 1
EPOCHS = 25
LEARNING_RATE = 0.0001
PATIENCE = 6
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

DEVICE = torch.device("cpu")


# ==================================================
# DATASET
# ==================================================
class VideoDataset(Dataset):
    def __init__(self, split):
        self.files = sorted((DATA_DIR / split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(
                f"No .npz files found in {DATA_DIR / split}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with np.load(self.files[index]) as data:
            frames = data["frames"].copy()
            masks = data["masks"].copy()

        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        frames /= 255.0

        masks = torch.from_numpy(masks).float().unsqueeze(1)
        masks = (masks > 0).float()

        return frames, masks


# ==================================================
# CNN + LSTM MODEL
# ==================================================
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

        x = x.reshape(batch * steps, channels, height, width)
        spatial = self.cnn(x)

        _, feature_channels, fh, fw = spatial.shape
        spatial = spatial.reshape(
            batch, steps, feature_channels, fh, fw
        )

        temporal_input = spatial.mean(dim=(-2, -1))
        temporal_output, _ = self.lstm(temporal_input)

        temporal_map = temporal_output.reshape(
            batch * steps, feature_channels, 1, 1
        ).expand(-1, -1, fh, fw)

        spatial_flat = spatial.reshape(
            batch * steps, feature_channels, fh, fw
        )

        combined = torch.cat([spatial_flat, temporal_map], dim=1)
        logits = self.decoder(combined)

        logits = F.interpolate(
            logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        return logits.reshape(batch, steps, 1, height, width)


# ==================================================
# LOSS
# ==================================================
def combined_loss(logits, targets):
    # Moderate foreground weighting avoids overwhelming
    # the loss when pothole pixels are rare.
    positive_weight = torch.tensor(
        3.0, dtype=logits.dtype, device=logits.device
    )

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=positive_weight,
    )

    probabilities = torch.sigmoid(logits)

    intersection = (probabilities * targets).sum(
        dim=(2, 3, 4)
    )
    denominator = probabilities.sum(
        dim=(2, 3, 4)
    ) + targets.sum(dim=(2, 3, 4))

    dice = (2 * intersection + 1.0) / (denominator + 1.0)
    dice_loss = 1.0 - dice.mean()

    return bce + dice_loss


# ==================================================
# EVALUATION
# ==================================================
def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_intersection = 0
    total_prediction = 0
    total_target = 0
    total_videos = 0

    with torch.set_grad_enabled(training):
        for frames, masks in loader:
            frames = frames.to(DEVICE)
            masks = masks.to(DEVICE)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(frames)
            loss = combined_loss(logits, masks)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()

            # Accumulate global pixel counts consistently.
            predictions = torch.sigmoid(logits) >= 0.5
            targets = masks >= 0.5

            total_intersection += (
                predictions & targets
            ).sum().item()
            total_prediction += predictions.sum().item()
            total_target += targets.sum().item()

            total_loss += loss.item()
            total_videos += frames.size(0)

    if total_videos == 0:
        raise RuntimeError("No videos were processed.")

    dice = (
        2 * total_intersection /
        (total_prediction + total_target)
        if total_prediction + total_target else 1.0
    )

    union = total_prediction + total_target - total_intersection
    iou = total_intersection / union if union else 1.0

    pred_fg = total_prediction / (
        len(loader.dataset) * 8 * 128 * 128
    )

    return total_loss / total_videos, dice, iou, pred_fg


# ==================================================
# TRAINING
# ==================================================
def main():
    train_data = VideoDataset("train")
    val_data = VideoDataset("val")

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=1,
        shuffle=False, num_workers=0
    )

    model = CNNLSTMSegmenter().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    best_dice = -1.0
    patience_count = 0
    best_path = OUT_DIR / "best_cnn_lstm_improved.pt"

    print("CNN + LSTM IMPROVED TRAINING")
    print("Device:", DEVICE)
    print("Training videos:", len(train_data))
    print("Validation videos:", len(val_data))
    print("Epochs:", EPOCHS)
    print("Checkpoint:", best_path)

    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        train_metrics = run_epoch(model, train_loader, optimizer)
        val_metrics = run_epoch(model, val_loader)

        train_loss, train_dice, train_iou, train_fg = train_metrics
        val_loss, val_dice, val_iou, val_fg = val_metrics

        print(f"\nEpoch {epoch}/{EPOCHS} ({time.time()-start:.1f}s)")
        print(
            f"Train | Loss {train_loss:.4f} "
            f"Dice {train_dice:.4f} IoU {train_iou:.4f} "
            f"Pred FG {train_fg*100:.2f}%"
        )
        print(
            f"Val   | Loss {val_loss:.4f} "
            f"Dice {val_dice:.4f} IoU {val_iou:.4f} "
            f"Pred FG {val_fg*100:.2f}%"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            patience_count = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "threshold": 0.5,
            }, best_path)

            print("Saved best Dice checkpoint.")
        else:
            patience_count += 1
            print(f"No Dice improvement: {patience_count}/{PATIENCE}")

        if patience_count >= PATIENCE:
            print("Early stopping.")
            break

    print("\nTraining complete.")
    print(f"Best global validation Dice: {best_dice:.4f}")
    print("Saved model:", best_path)


if __name__ == "__main__":
    main()
