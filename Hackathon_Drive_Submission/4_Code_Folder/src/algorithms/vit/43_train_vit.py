
from pathlib import Path
import sys
import json
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from image_dataset_loader import ImageFolderDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
DATA = PROJECT / "dataset/processed/efficientnet_crops"
OUT = PROJECT / "results/vit"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
BATCH_SIZE = 2
EPOCHS = 5
IMAGE_SIZE = 224

CLASS_NAMES = [
    "D00", "D01", "D0w0", "D10", "D11",
    "D20", "D40", "D43", "D44", "D50"
]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

train_data = ImageFolderDataset(DATA / "train", CLASS_NAMES, train_transform)
val_data = ImageFolderDataset(DATA / "val", CLASS_NAMES, val_transform)

if not train_data.samples or not val_data.samples:
    raise ValueError("Training or validation crops are missing.")

counts = np.bincount(
    [label for _, label in train_data.samples],
    minlength=len(CLASS_NAMES)
)
weights = len(train_data) / (
    len(CLASS_NAMES) * np.maximum(counts, 1)
)
weights = torch.tensor(weights, dtype=torch.float32)

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_data, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)

print("Training crops:", len(train_data))
print("Validation crops:", len(val_data))
print("Device:", DEVICE)

# Use pretrained ImageNet weights if available.
try:
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    print("Loaded pretrained ImageNet weights.")
except Exception as exc:
    print("Pretrained weights unavailable:", exc)
    print("Using random initialization.")
    model = vit_b_16(weights=None)

model.heads.head = nn.Linear(
    model.heads.head.in_features, len(CLASS_NAMES)
)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

best_loss = float("inf")
best_path = OUT / "best_vit.pt"
history = []

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * labels.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(1)

            val_loss_sum += loss.item() * labels.size(0)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    train_loss = train_loss_sum / max(train_total, 1)
    val_loss = val_loss_sum / max(val_total, 1)
    train_acc = train_correct / max(train_total, 1)
    val_acc = val_correct / max(val_total, 1)

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc
    })

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
        f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
    )

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "image_size": IMAGE_SIZE,
            "epoch": epoch + 1
        }, best_path)

    with (OUT / "training_history.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(history, f, indent=2)

checkpoint = torch.load(
    best_path, map_location=DEVICE, weights_only=False
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images.to(DEVICE))
        y_true.extend(labels.tolist())
        y_pred.extend(outputs.argmax(1).cpu().tolist())

print("\nValidation report:")
print(classification_report(
    y_true, y_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    zero_division=0
))

report = classification_report(
    y_true, y_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    zero_division=0,
    output_dict=True
)

with (OUT / "classification_report.json").open(
    "w", encoding="utf-8"
) as f:
    json.dump(report, f, indent=2)

np.savetxt(
    OUT / "confusion_matrix.csv",
    confusion_matrix(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES)))
    ),
    delimiter=",",
    fmt="%d"
)

print("\nTraining complete.")
print("Best model:", best_path)
print("Results folder:", OUT)
