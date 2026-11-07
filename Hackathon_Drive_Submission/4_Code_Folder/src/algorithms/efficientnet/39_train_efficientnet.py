
from pathlib import Path
import sys
import json
import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- CONFIGURATION ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
sys.path.insert(0, str(PROJECT / "algorithms"))
from image_dataset_loader import ImageFolderDataset
DATA = Path(os.environ.get("ROAD_DAMAGE_DATA_DIR", str(PROJECT / "dataset/processed/efficientnet_crops")))
GAN_AUGMENTATION_DIR = PROJECT / "results/gan_individual_samples"
USE_GAN_AUGMENTATION = os.environ.get(
    "ROAD_DAMAGE_USE_GAN_AUGMENT", "0"
).strip().lower() in {"1", "true", "yes", "on"}
RESUME_TRAINING = os.environ.get(
    "ROAD_DAMAGE_RESUME", "0"
).strip().lower() in {"1", "true", "yes", "on"}
OUT = Path(os.environ.get("ROAD_DAMAGE_OUTPUT_DIR", str(PROJECT / "results" / (
    "efficientnet_b0_gan_augmented"
    if USE_GAN_AUGMENTATION else "efficientnet_b0"
))))
EPOCHS = int(os.environ.get("ROAD_DAMAGE_EPOCHS", "10"))
OUT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
IMAGE_SIZE = 224
DEVICE = torch.device("cpu")

# Fixed class mapping for BOTH training and validation.
CLASS_NAMES = [
    "D00", "D01", "D0w0", "D10", "D11",
    "D20", "D40", "D43", "D44", "D50"
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# ---------------- TRANSFORMS ----------------
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

# ---------------- DATASET ----------------
if USE_GAN_AUGMENTATION and not GAN_AUGMENTATION_DIR.is_dir():
    raise FileNotFoundError(
        "GAN augmentation was enabled, but generated samples were "
        f"not found at {GAN_AUGMENTATION_DIR}. Generate samples "
        "with src/52_generate_gan_samples.py first."
    )

gan_roots = [GAN_AUGMENTATION_DIR] if USE_GAN_AUGMENTATION else []
train_data = ImageFolderDataset(
    DATA / "train", CLASS_NAMES, train_transform, extra_roots=gan_roots
)
train_data.gan_sample_count = len(train_data) - sum(
    1 for path, _ in train_data.samples if DATA / "train" in path.parents
)
val_data = ImageFolderDataset(DATA / "val", CLASS_NAMES, val_transform)

if not train_data.samples or not val_data.samples:
    raise ValueError("Training or validation dataset is empty.")

train_counts = np.bincount(
    [label for _, label in train_data.samples],
    minlength=len(CLASS_NAMES)
)
val_counts = np.bincount(
    [label for _, label in val_data.samples],
    minlength=len(CLASS_NAMES)
)

print("Device:", DEVICE)
print("Classes:", CLASS_NAMES)
print("Training crops:", len(train_data))
print("GAN training samples:", train_data.gan_sample_count)
print("Validation crops:", len(val_data))

print("\nClass distribution:")
for i, name in enumerate(CLASS_NAMES):
    print(
        f"{name}: train={train_counts[i]}, "
        f"validation={val_counts[i]}"
    )

# Weighted loss reduces the dominance of frequent classes.
# Rare classes still require more real examples for reliable learning.
class_weights = len(train_data) / (
    len(CLASS_NAMES) * np.maximum(train_counts, 1)
)
class_weights = torch.tensor(
    class_weights, dtype=torch.float32
)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)
val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ---------------- MODEL ----------------
try:
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )
    print("\nLoaded pretrained ImageNet weights.")
except Exception as exc:
    print("\nPretrained weights unavailable.")
    print("Reason:", exc)
    print("Using random initialization instead.")
    model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(CLASS_NAMES)
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(DEVICE)
)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-4
)

# ---------------- TRAINING ----------------
best_path = OUT / "best_efficientnet_b0.pt"
history_path = OUT / "training_history.json"
if best_path.exists() and not RESUME_TRAINING:
    raise FileExistsError(f"Refusing to overwrite existing classifier checkpoint: {best_path}")
history = []
start_epoch = 0
best_val_loss = float("inf")
if RESUME_TRAINING:
    if not best_path.exists() or not history_path.exists():
        raise FileNotFoundError("Resume requested, but checkpoint or training history is missing.")
    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    with history_path.open("r", encoding="utf-8") as file:
        history = json.load(file)
    start_epoch = max((int(row["epoch"]) for row in history), default=0)
    best_val_loss = min((float(row["val_loss"]) for row in history), default=float("inf"))
    if start_epoch >= EPOCHS:
        raise ValueError(f"Training history already has {start_epoch} epochs; requested total is {EPOCHS}.")
    print(f"Resuming from epoch {start_epoch}; best validation loss so far: {best_val_loss:.4f}")

for epoch in range(start_epoch, EPOCHS):
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * labels.size(0)
        train_correct += (
            outputs.argmax(1) == labels
        ).sum().item()
        train_total += labels.size(0)

    # Validation: no augmentation or gradient updates.
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * labels.size(0)
            val_correct += (
                outputs.argmax(1) == labels
            ).sum().item()
            val_total += labels.size(0)

    train_loss = train_loss_sum / max(train_total, 1)
    val_loss = val_loss_sum / max(val_total, 1)
    train_acc = train_correct / max(train_total, 1)
    val_acc = val_correct / max(val_total, 1)

    record = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "gan_augmentation_enabled": USE_GAN_AUGMENTATION,
        "gan_training_samples": train_data.gan_sample_count,
    }
    history.append(record)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: {train_acc:.4f} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val accuracy: {val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "class_to_idx": CLASS_TO_IDX,
            "image_size": IMAGE_SIZE,
            "epoch": epoch + 1,
            "gan_augmentation_enabled": USE_GAN_AUGMENTATION,
            "gan_training_samples": train_data.gan_sample_count,
        }, best_path)

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

# ---------------- BEST MODEL EVALUATION ----------------
checkpoint = torch.load(
    best_path, map_location=DEVICE, weights_only=False
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_true = []
all_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images.to(DEVICE))
        predictions = outputs.argmax(1).cpu().tolist()

        all_true.extend(labels.tolist())
        all_pred.extend(predictions)

print("\nValidation classification report:")
print(classification_report(
    all_true,
    all_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    zero_division=0
))

report = classification_report(
    all_true,
    all_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    zero_division=0,
    output_dict=True
)

with (OUT / "classification_report.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(report, file, indent=2)

matrix = confusion_matrix(
    all_true,
    all_pred,
    labels=list(range(len(CLASS_NAMES)))
)

np.savetxt(
    OUT / "confusion_matrix.csv",
    matrix,
    delimiter=",",
    fmt="%d"
)

print("\nTraining complete.")
print("Best checkpoint:", best_path)
print("History:", OUT / "training_history.json")
print("Report:", OUT / "classification_report.json")
print("Confusion matrix:", OUT / "confusion_matrix.csv")
