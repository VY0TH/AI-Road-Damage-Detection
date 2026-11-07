
import random
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report

# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())

DATA_ROOT = (
    PROJECT_ROOT / "dataset" / "subsets"
    / "multicountry" / "RDD2022"
)

TRAIN_IMAGES = DATA_ROOT / "split" / "train" / "images"
VAL_IMAGES = DATA_ROOT / "split" / "val" / "images"

# CORRECT LABEL PATHS
TRAIN_LABELS = DATA_ROOT / "yolo" / "labels" / "train"
VAL_LABELS = DATA_ROOT / "yolo" / "labels" / "val"

RESULT_DIR = PROJECT_ROOT / "results" / "cnn_lstm"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = RESULT_DIR / "best_cnn_lstm.pt"

IMAGE_SIZE = 128
SEQUENCE_LENGTH = 5
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.001
SEED = 42

DAMAGE_CLASSES = [
    "D00", "D01", "D10", "D11", "D20",
    "D40", "D43", "D44", "D50", "D0w0"
]
CLASS_NAMES = DAMAGE_CLASSES + ["No_Damage"]
NUM_CLASSES = len(CLASS_NAMES)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")

print("=" * 60)
print("CNN + LSTM ROAD DAMAGE CLASSIFICATION")
print("=" * 60)
print("Device:", DEVICE)
print("Project:", PROJECT_ROOT)
print("Train images:", TRAIN_IMAGES)
print("Train labels:", TRAIN_LABELS)
print("Validation images:", VAL_IMAGES)
print("Validation labels:", VAL_LABELS)


# ============================================================
# 2. IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 3. READ YOLO LABELS
# ============================================================

def get_image_label(image_path, label_dir):
    label_path = label_dir / f"{image_path.stem}.txt"

    if not label_path.is_file():
        raise FileNotFoundError(
            f"Missing label for image: {image_path.name}\n"
            f"Expected label: {label_path}"
        )

    class_ids = []

    with open(label_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()

            if not parts:
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                continue

            if 0 <= class_id < len(DAMAGE_CLASSES):
                class_ids.append(class_id)

    # Empty label means no mapped damage objects.
    if not class_ids:
        return len(DAMAGE_CLASSES)

    # One image-level label: most frequent damage class.
    return Counter(class_ids).most_common(1)[0][0]


# ============================================================
# 4. DATASET: GROUP IMAGES INTO FIXED-LENGTH SEQUENCES
# ============================================================

class RoadSequenceDataset(Dataset):

    def __init__(self, image_dir, label_dir, split_name):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        if not self.label_dir.is_dir():
            raise FileNotFoundError(
                f"Label directory not found: {self.label_dir}"
            )

        extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        images = sorted([
            p for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ])

        if not images:
            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

        image_labels = [
            get_image_label(image, self.label_dir)
            for image in images
        ]

        self.sequences = []

        for start in range(0, len(images), SEQUENCE_LENGTH):
            group_images = images[
                start:start + SEQUENCE_LENGTH
            ]
            group_labels = image_labels[
                start:start + SEQUENCE_LENGTH
            ]

            # Use complete sequences only.
            if len(group_images) != SEQUENCE_LENGTH:
                continue

            sequence_label = Counter(
                group_labels
            ).most_common(1)[0][0]

            self.sequences.append(
                (group_images, sequence_label)
            )

        if not self.sequences:
            raise RuntimeError(
                f"No complete sequences in {self.image_dir}"
            )

        print(
            f"{split_name}: {len(images)} images, "
            f"{len(self.sequences)} sequences"
        )

        print(
            f"{split_name} label distribution:",
            Counter(label for _, label in self.sequences)
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        image_paths, sequence_label = self.sequences[index]
        frames = []

        for image_path in image_paths:
            with Image.open(image_path) as image:
                frames.append(
                    transform(image.convert("RGB"))
                )

        sequence = torch.stack(frames)

        return sequence, torch.tensor(
            sequence_label, dtype=torch.long
        )


# ============================================================
# 5. CNN + LSTM MODEL
# ============================================================

class CNNLSTM(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape

        # Extract CNN features from each image.
        x = x.reshape(
            batch_size * seq_len,
            channels,
            height,
            width
        )

        features = self.cnn(x)

        # Restore batch and sequence dimensions.
        features = features.reshape(
            batch_size, seq_len, -1
        )

        _, (hidden, _) = self.lstm(features)

        return self.classifier(hidden[-1])


# ============================================================
# 6. VERIFY PATHS BEFORE TRAINING
# ============================================================

for path in [
    TRAIN_IMAGES, VAL_IMAGES, TRAIN_LABELS, VAL_LABELS
]:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Required directory does not exist: {path}"
        )

print("\nAll four dataset directories exist.")


# ============================================================
# 7. LOAD DATA
# ============================================================

train_dataset = RoadSequenceDataset(
    TRAIN_IMAGES, TRAIN_LABELS, "Training"
)

val_dataset = RoadSequenceDataset(
    VAL_IMAGES, VAL_LABELS, "Validation"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 8. INITIALIZE MODEL AND OPTIMIZER
# ============================================================

model = CNNLSTM(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

best_val_accuracy = -1.0
best_epoch = 0


# ============================================================
# 9. TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):
    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for sequences, labels in train_loader:
        sequences = sequences.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(sequences)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)
        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

    train_loss = train_loss_sum / max(train_total, 1)
    train_accuracy = train_correct / max(train_total, 1)

    # Validation
    model.eval()

    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().tolist()
            )
            all_labels.extend(
                labels.cpu().tolist()
            )

    val_loss = val_loss_sum / max(val_total, 1)
    val_accuracy = val_correct / max(val_total, 1)

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
        f"\nTrain Loss: {train_loss:.4f}"
        f" | Train Accuracy: {train_accuracy:.4f}"
        f"\nVal Loss: {val_loss:.4f}"
        f" | Val Accuracy: {val_accuracy:.4f}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "sequence_length": SEQUENCE_LENGTH,
            "image_size": IMAGE_SIZE,
            "best_val_accuracy": best_val_accuracy,
            "best_epoch": best_epoch
        }, MODEL_PATH)

        print(f"Best model saved: {MODEL_PATH}")


# ============================================================
# 10. FINAL VALIDATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("FINAL VALIDATION REPORT")
print("=" * 60)

print(classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    zero_division=0
))

print(f"Best epoch: {best_epoch}")
print(f"Best validation accuracy: {best_val_accuracy:.4f}")
print(f"Checkpoint: {MODEL_PATH}")

print(
    "\nNOTE: Images are grouped by sorted filename, not by "
    "verified video chronology. These results do not establish "
    "real temporal video-recognition performance."
)
