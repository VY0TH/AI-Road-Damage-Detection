"""Step 6 CLI Runner: Conditional GAN Augmentation on Training Data Only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_gan import train_conditional_gan
from generate_gan_samples import generate_samples
from inject_to_training import inject_gan_samples_to_training, verify_augmentation_isolation
import pipeline_config as cfg


def run_step_6(epochs: int = 1, generate_count: int = 10) -> dict:
    train_dir = cfg.DATASET_ROOT / "processed" / "efficientnet_crops" / "train"
    val_dir = cfg.DATASET_ROOT / "processed" / "efficientnet_crops" / "val"
    ckpt_dir = cfg.RESULTS_DIR / "gan_checkpoints"
    syn_samples_dir = cfg.RESULTS_DIR / "gan_augmented_training_samples"
    aug_train_dir = cfg.DATASET_ROOT / "processed" / "augmented_training_crops"

    print(f"\n{'=' * 70}")
    print(f"[STEP 6] CONDITIONAL GAN AUGMENTATION (TRAINING DATA ONLY)")
    print(f"{'=' * 70}")

    # 1. Train Conditional GAN strictly on training crops
    ckpt_path = train_conditional_gan(
        train_dir=train_dir,
        output_checkpoint_dir=ckpt_dir,
        class_names=cfg.CLASS_NAMES,
        epochs=epochs,
        batch_size=16,
    )

    # 2. Generate samples for minority classes (e.g. D40 pothole, D10, D20)
    samples_plan = {
        "D40": generate_count,
        "D10": generate_count // 2,
        "D20": generate_count // 2,
    }
    print(f"\nGenerating synthetic augmentation samples: {samples_plan}")
    gen_counts = generate_samples(
        generator_checkpoint=ckpt_path,
        output_dir=syn_samples_dir,
        class_names=cfg.CLASS_NAMES,
        samples_per_class=samples_plan,
    )
    print(f"Generated samples: {gen_counts}")

    # 3. Inject synthetic samples into training data ONLY
    injection_summary = inject_gan_samples_to_training(
        real_train_dir=train_dir,
        synthetic_samples_dir=syn_samples_dir,
        augmented_output_dir=aug_train_dir,
    )
    print(f"\nInjection Summary:")
    print(f"  Real Training Images     : {injection_summary['real_training_images']}")
    print(f"  Synthetic GAN Images     : {injection_summary['synthetic_gan_images']}")
    print(f"  Total Augmented Training : {injection_summary['total_augmented_training_images']}")

    # 4. Rigorous verification: Ensure validation and test sets remain 100% real
    isolation_audit = verify_augmentation_isolation(
        augmented_train_dir=aug_train_dir,
        val_dir=val_dir,
    )
    print(f"\nAugmentation Isolation Audit: {isolation_audit['status']}")

    summary = {
        "generator_checkpoint": str(ckpt_path),
        "generated_counts": gen_counts,
        "training_dataset": injection_summary,
        "isolation_audit": isolation_audit,
    }

    report_path = cfg.RESULTS_DIR / "step06_gan_augmentation_summary.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[STEP 6 COMPLETED] Report saved to: {report_path}")
    print(f"{'=' * 70}\n")
    return summary


if __name__ == "__main__":
    run_step_6()
