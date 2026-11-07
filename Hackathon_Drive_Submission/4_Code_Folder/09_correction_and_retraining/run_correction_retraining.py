"""Step 9 CLI Runner: Apply Corrections and Retrain."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "01_load_image_dataset"))
sys.path.insert(0, str(PROJECT_ROOT / "07_initial_model_training"))

from dataset_loader import ImageFolderDataset, create_dataloader
from models.efficientnet_classifier import create_efficientnet
from models.resnet50_classifier import create_resnet50
from retrainer import retrain_with_corrections
from compare_before_after import compare_models
import pipeline_config as cfg


def run_step_9(
    model_name: str = "efficientnet",
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 3e-4,
) -> dict:
    # Use augmented training data from Step 6 if present, else standard train crops
    train_dir = cfg.DATASET_ROOT / "processed" / "augmented_training_crops"
    if not train_dir.is_dir():
        train_dir = cfg.DATASET_ROOT / "processed" / "efficientnet_crops" / "train"

    val_dir = cfg.DATASET_ROOT / "processed" / "efficientnet_crops" / "val"
    out_dir = cfg.RESULTS_DIR / "corrected_retraining" / model_name

    print(f"\n{'=' * 70}")
    print(f"[STEP 9] APPLY CORRECTIONS & RETRAIN ({model_name.upper()})")
    print(f"  Training Data  : {train_dir}")
    print(f"  Validation Data: {val_dir} (100% Real Images Only)")
    print(f"  Corrections    : Architecture Upgrade, Class Weighting, Cosine LR, Label Smoothing")
    print(f"{'=' * 70}")

    # Enhanced Augmentation Transforms for Retraining
    train_transform = transforms.Compose([
        transforms.Resize((cfg.DEFAULT_IMAGE_SIZE, cfg.DEFAULT_IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
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
    if model_name == "efficientnet":
        model = create_efficientnet(num_classes=num_classes)
    elif model_name == "resnet50":
        model = create_resnet50(num_classes=num_classes)
    else:
        model = create_efficientnet(num_classes=num_classes)

    class_counts = train_ds.get_class_counts()

    retrain_res = retrain_with_corrections(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=out_dir,
        class_counts=class_counts,
        epochs=epochs,
        lr=lr,
        label_smoothing=0.1,
    )

    # Compare against initial baseline
    base_history = cfg.RESULTS_DIR / "initial_training" / "baseline_cnn" / "training_history.json"
    comparison = {}
    if base_history.is_file():
        comparison = compare_models(
            baseline_history_path=base_history,
            corrected_history_path=Path(retrain_res["history_json"]),
            output_dir=out_dir,
        )
        print(f"\nBefore vs. After Comparison:")
        print(f"  Baseline Accuracy : {comparison['baseline_best_val_acc']*100:.2f}%")
        print(f"  Corrected Accuracy: {comparison['corrected_best_val_acc']*100:.2f}%")
        print(f"  Accuracy Delta    : {comparison['accuracy_improvement']*100:+.2f}%")

    summary = {
        "model_name": model_name,
        "retrain_results": retrain_res,
        "comparison_with_baseline": comparison,
    }

    report_json = cfg.RESULTS_DIR / "step09_correction_retraining_summary.json"
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[STEP 9 COMPLETED] Retraining with corrections finished.")
    print(f"Report saved to: {report_json}")
    print(f"{'=' * 70}\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["efficientnet", "resnet50"], default="efficientnet")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    run_step_9(args.model, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
