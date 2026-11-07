
from pathlib import Path
import os
from collections import Counter
import csv
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ---------------- CONFIGURATION ----------------

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
DATA_ROOT = ROOT / "dataset" / "subsets" / "multicountry" / "RDD2022" / "yolo_grouped"
OUTPUT_DIR = Path(os.environ.get("ROAD_DAMAGE_OUTPUT_DIR", str(ROOT / "results" / "resnet50_road_damage")))
CROP_DIR = OUTPUT_DIR / "crops"

CLASS_NAMES = [
    "D00", "D01", "D10", "D11", "D20",
    "D40", "D43", "D44", "D50", "D0w0"
]

NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = int(os.environ.get("ROAD_DAMAGE_EPOCHS", "10"))
LEARNING_RATE = 1e-4
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- DATA PREPARATION ----------------

def make_crops(split):
    image_dir = DATA_ROOT / "images" / split
    label_dir = DATA_ROOT / "labels" / split
    out_dir = CROP_DIR / split

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            f"Missing image or label directory for '{split}'.\n"
            f"Images: {image_dir}\nLabels: {label_dir}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    saved = 0

    image_files = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]

    for image_path in image_files:
        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        height, width = image.shape[:2]

        try:
            lines = label_path.read_text(
                encoding="utf-8"
            ).splitlines()
        except OSError:
            continue

        for line_index, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
            except ValueError:
                continue

            if not 0 <= class_id < NUM_CLASSES:
                continue

            # YOLO normalized center/width/height -> pixel coordinates.
            x1 = max(0, int((xc - bw / 2) * width))
            y1 = max(0, int((yc - bh / 2) * height))
            x2 = min(width, int((xc + bw / 2) * width))
            y2 = min(height, int((yc + bh / 2) * height))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            class_name = CLASS_NAMES[class_id]
            class_dir = out_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            crop_name = (
                f"{image_path.stem}_box{line_index}.jpg"
            )
            crop_path = class_dir / crop_name

            # Avoid repeatedly rewriting already generated crops.
            if not crop_path.exists():
                if not cv2.imwrite(str(crop_path), crop):
                    continue

            counts[class_name] += 1
            saved += 1

    print(f"\n{split.upper()}: generated/read {saved} crops")
    for name in CLASS_NAMES:
        print(f"  {name}: {counts[name]}")

    return counts


# ---------------- DATASET ----------------

class CropDataset(Dataset):
    def __init__(self, split, training=False):
        self.items = []
        self.training = training

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            *(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(8),
                    transforms.ColorJitter(
                        brightness=0.15, contrast=0.15
                    )
                ] if training else []
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        split_dir = CROP_DIR / split

        for class_id, class_name in enumerate(CLASS_NAMES):
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue

            for path in class_dir.glob("*.jpg"):
                self.items.append((path, class_id))

        if not self.items:
            raise RuntimeError(
                f"No training crops found in {split_dir}"
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, class_id = self.items[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, class_id


# ---------------- TRAINING ----------------

def main():
    print("Device:", DEVICE)
    print("Preparing crops from YOLO annotations...")

    make_crops("train")
    make_crops("val")

    train_ds = CropDataset("train", training=True)
    val_ds = CropDataset("val", training=False)

    workers = 0  # Reliable on Windows
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=workers
    )

    print(f"\nTraining crops: {len(train_ds)}")
    print(f"Validation crops: {len(val_ds)}")

    # Pretrained ResNet50 with a new 10-class output layer.
    # The first run requires cached weights or internet access.
    try:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            "ResNet50 pretrained weights could not be loaded. "
            "Check internet access or the Torch model cache."
        ) from exc

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    best_accuracy = -1.0
    history_path = OUTPUT_DIR / "training_history.csv"
    best_path = OUTPUT_DIR / "best_resnet50.pt"
    if best_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing classifier checkpoint: {best_path}")

    with history_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "val_accuracy"
        ])

        for epoch in range(EPOCHS):
            model.train()
            train_loss_sum = 0.0
            train_count = 0

            for images, labels in train_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * len(labels)
                train_count += len(labels)

            train_loss = train_loss_sum / max(train_count, 1)

            model.eval()
            val_loss_sum = 0.0
            correct = 0
            total = 0

            with torch.inference_mode():
                for images, labels in val_loader:
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)

                    logits = model(images)
                    loss = criterion(logits, labels)

                    val_loss_sum += loss.item() * len(labels)
                    predictions = logits.argmax(dim=1)
                    correct += (
                        predictions == labels
                    ).sum().item()
                    total += len(labels)

            val_loss = val_loss_sum / max(total, 1)
            accuracy = correct / max(total, 1)

            writer.writerow([
                epoch + 1, train_loss, val_loss, accuracy
            ])
            file.flush()

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"train loss={train_loss:.4f} | "
                f"val loss={val_loss:.4f} | "
                f"val accuracy={accuracy:.4f}"
            )

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": IMAGE_SIZE,
                    "epoch": epoch + 1,
                    "val_accuracy": accuracy
                }, best_path)

    print("\nTraining complete.")
    print("Best checkpoint:", best_path)
    print("Training history:", history_path)
    print(
        "Validation accuracy is an initial prototype metric. "
        "It is not directly comparable to YOLO mAP."
    )


if __name__ == "__main__":
    main()
