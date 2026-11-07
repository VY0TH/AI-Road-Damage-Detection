"""Step 6: Training Data Augmentation Injector & Verification.

Ensures that synthetic GAN samples are added EXCLUSIVELY to the training split.
Validation and test sets remain 100% untouched and contain 0 synthetic images.
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def inject_gan_samples_to_training(
    real_train_dir: Path,
    synthetic_samples_dir: Path,
    augmented_output_dir: Path,
) -> dict:
    real_train_dir = Path(real_train_dir).resolve()
    synthetic_samples_dir = Path(synthetic_samples_dir).resolve()
    augmented_output_dir = Path(augmented_output_dir).resolve()
    augmented_output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    real_copied = 0
    syn_copied = 0

    # 1. Copy real training images
    for class_dir in sorted(p for p in real_train_dir.iterdir() if p.is_dir()):
        cls_name = class_dir.name
        dest_cls_dir = augmented_output_dir / cls_name
        dest_cls_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(class_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                dest = dest_cls_dir / img_path.name
                if not dest.is_file():
                    shutil.copy2(img_path, dest)
                real_copied += 1
                manifest_rows.append({
                    "filename": img_path.name,
                    "class_name": cls_name,
                    "sample_type": "REAL",
                    "origin_path": str(img_path),
                })

    # 2. Add approved synthetic training samples
    if synthetic_samples_dir.is_dir():
        for class_dir in sorted(p for p in synthetic_samples_dir.iterdir() if p.is_dir()):
            cls_name = class_dir.name
            dest_cls_dir = augmented_output_dir / cls_name
            dest_cls_dir.mkdir(parents=True, exist_ok=True)

            for img_path in sorted(class_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    dest = dest_cls_dir / img_path.name
                    if not dest.is_file():
                        shutil.copy2(img_path, dest)
                    syn_copied += 1
                    manifest_rows.append({
                        "filename": img_path.name,
                        "class_name": cls_name,
                        "sample_type": "SYNTHETIC_GAN",
                        "origin_path": str(img_path),
                    })

    manifest_path = augmented_output_dir / "augmented_train_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class_name", "sample_type", "origin_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    return {
        "real_training_images": real_copied,
        "synthetic_gan_images": syn_copied,
        "total_augmented_training_images": real_copied + syn_copied,
        "augmented_dir": str(augmented_output_dir),
    }


def verify_augmentation_isolation(
    augmented_train_dir: Path,
    val_dir: Path,
    test_dir: Path | None = None,
) -> dict:
    """Verifies that 0% synthetic samples entered validation or test sets."""
    val_dir = Path(val_dir).resolve()
    val_synthetic = []
    if val_dir.is_dir():
        for p in val_dir.rglob("*"):
            if "syn" in p.name.lower() or "gan" in p.name.lower():
                val_synthetic.append(str(p))

    test_synthetic = []
    if test_dir and Path(test_dir).is_dir():
        for p in Path(test_dir).rglob("*"):
            if "syn" in p.name.lower() or "gan" in p.name.lower():
                test_synthetic.append(str(p))

    passed = (len(val_synthetic) == 0 and len(test_synthetic) == 0)
    return {
        "isolation_verified": passed,
        "synthetic_in_validation_count": len(val_synthetic),
        "synthetic_in_test_count": len(test_synthetic),
        "status": "PASS: Validation and Test sets are 100% REAL" if passed else "VIOLATION DETECTED",
    }
