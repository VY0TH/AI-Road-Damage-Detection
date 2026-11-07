
from pathlib import Path
import os
import csv
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
CROP_ROOT = Path(os.environ.get("ROAD_DAMAGE_CROP_DIR", str(ROOT / "results" / "resnet50_road_damage" / "crops")))
OUTPUT_DIR = Path(os.environ.get("ROAD_DAMAGE_OUTPUT_DIR", str(ROOT / "results" / "vit_road_damage")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "D00", "D01", "D10", "D11", "D20",
    "D40", "D43", "D44", "D50", "D0w0"
]

IMAGE_SIZE = 224
BATCH_SIZE = 4
EPOCHS = int(os.environ.get("ROAD_DAMAGE_EPOCHS", "10"))
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class DamageCropDataset(Dataset):
    def __init__(self, split, training=False):
        self.samples = []

        for class_id, class_name in enumerate(CLASS_NAMES):
            folder = CROP_ROOT / split / class_name
            if folder.exists():
                for path in folder.glob("*.jpg"):
                    self.samples.append((path, class_id))

        if not self.samples:
            raise RuntimeError(
                f"No crops found for {split} in {CROP_ROOT}"
            )

        ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
        if training:
            ops += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(8)
            ]

        ops += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label


def main():
    print("Device:", DEVICE)
    print("Loading pretrained Vision Transformer...")

    train_ds = DamageCropDataset("train", training=True)
    val_ds = DamageCropDataset("val", training=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    print("Training crops:", len(train_ds))
    print("Validation crops:", len(val_ds))

    # First run downloads pretrained weights if not cached.
    try:
        weights = models.ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            "Could not load pretrained ViT weights. "
            "Check internet access or the Torch model cache."
        ) from exc

    # Freeze the pretrained backbone for a lightweight prototype run.
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.heads.head = nn.Linear(
        model.heads.head.in_features, len(CLASS_NAMES)
    )

   
    resume_path = os.environ.get("ROAD_DAMAGE_RESUME_CHECKPOINT")
    resume_epoch = 0
    best_accuracy = -1.0

    if resume_path:
        resume_checkpoint = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )

        old_classes = resume_checkpoint.get("class_names", [])
        if len(old_classes) != len(CLASS_NAMES) or set(old_classes) != set(CLASS_NAMES):
            raise ValueError(
                f"Checkpoint classes {old_classes} do not match {CLASS_NAMES}"
            )

        state = resume_checkpoint["model_state_dict"]

        # Remap the classifier head from the checkpoint's class order
        # into the current trainer's class order.
        old_head_weight = state["heads.head.weight"]
        old_head_bias = state["heads.head.bias"]
        new_weight = old_head_weight.clone()
        new_bias = old_head_bias.clone()

        for new_idx, class_name in enumerate(CLASS_NAMES):
            old_idx = old_classes.index(class_name)
            new_weight[new_idx] = old_head_weight[old_idx]
            new_bias[new_idx] = old_head_bias[old_idx]

        state["heads.head.weight"] = new_weight
        state["heads.head.bias"] = new_bias
        model.load_state_dict(state)

        resume_epoch = int(resume_checkpoint.get("epoch", 0))
        best_accuracy = float(resume_checkpoint.get("val_accuracy", -1.0))
        print(f"Resuming from epoch {resume_epoch}: {resume_path}")
        print("Classifier head remapped to the current class order.")

    # Train only the new classification head on the laptop.
    for parameter in model.heads.parameters():
        parameter.requires_grad = True

    model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = OUTPUT_DIR / "best_vit_road_damage.pt"
    history_path = OUTPUT_DIR / "training_history.csv"
    if checkpoint_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing classifier checkpoint: {checkpoint_path}")

    with history_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])

        for epoch in range(EPOCHS):
            reported_epoch = resume_epoch + epoch + 1
            model.train()
            total_loss = 0.0
            total_train = 0

            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(labels)
                total_train += len(labels)

            train_loss = total_loss / max(total_train, 1)

            model.eval()
            val_loss_sum = 0.0
            correct = 0
            total = 0

            with torch.inference_mode():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    logits = model(images)
                    loss = criterion(logits, labels)

                    val_loss_sum += loss.item() * len(labels)
                    correct += (logits.argmax(1) == labels).sum().item()
                    total += len(labels)

            val_loss = val_loss_sum / max(total, 1)
            accuracy = correct / max(total, 1)

            writer.writerow([
                reported_epoch, train_loss, val_loss, accuracy
            ])
            file.flush()

            print(
                f"Epoch {reported_epoch}/{resume_epoch + EPOCHS} | "
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
                    "epoch": reported_epoch,
                    "val_accuracy": accuracy
                }, checkpoint_path)

    print("\nViT prototype training complete.")
    print("Best checkpoint:", checkpoint_path)
    print("Training history:", history_path)
    print("The pretrained ViT backbone was frozen for this run.")


if __name__ == "__main__":
    main()
