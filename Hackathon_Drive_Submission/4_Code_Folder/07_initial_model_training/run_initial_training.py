"""Step 7 CLI Runner: Initial Model Training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "01_load_image_dataset"))

from dataset_loader import ImageFolderDataset, create_dataloader
from models.baseline_cnn import BaselineCNN
from models.efficientnet_classifier import create_efficientnet
from models.resnet50_classifier import create_resnet50
from models.vit_classifier import create_vit
from trainer import train_model
import pipeline_config as cfg


def run_step_7(
    model_name: str = "baseline_cnn",
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 1e-4,
) -> dict:
    train_dir = cfg.DATASET_ROOT / "processed" / "efficientnet_crops" / "train"
    val_dir = cfg.DATASET_ROOT / "processed" / "efficientnet_crops" / "val"
    out_dir = cfg.RESULTS_DIR / "initial_training" / model_name

    print(f"\n{'=' * 70}")
    print(f"[STEP 7] INITIAL MODEL TRAINING ({model_name.upper()})")
    print(f"  Train Dir: {train_dir}")
    print(f"  Val Dir  : {val_dir}")
    print(f"  Epochs   : {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"{'=' * 70}")

    # Standard Transforms
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

    num_classes = len(cfg.CLASS_NAMES)
    if model_name == "baseline_cnn":
        model = BaselineCNN(num_classes=num_classes)
    elif model_name == "efficientnet":
        model = create_efficientnet(num_classes=num_classes)
    elif model_name == "resnet50":
        model = create_resnet50(num_classes=num_classes)
    elif model_name == "vit":
        model = create_vit(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=out_dir,
        epochs=epochs,
        lr=lr,
    )

    print(f"\n[STEP 7 COMPLETED] Model training finished.")
    print(f"  Best Val Accuracy: {results['best_val_accuracy'] * 100:.2f}%")
    print(f"  Checkpoint: {results['best_checkpoint']}")
    print(f"  History   : {results['history_json']}")
    print(f"{'=' * 70}\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["baseline_cnn", "efficientnet", "resnet50", "vit"], default="baseline_cnn")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    run_step_7(args.model, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
