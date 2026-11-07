"""Master End-to-End Image Classification Pipeline Runner.

Orchestrates the complete 10-step sequence:
  Step 01: Load Image Dataset
  Step 02: Data Preprocessing
  Step 03: Data Leakage Checks (Target leakage • duplicate leakage • suspicious features)
  Step 04: CLAHE (when contrast enhancement is useful)
  Step 05: 1. Train / Validation / Test Split | 2. 10-Fold Cross-Validation
  Step 06: GAN Augmentation on TRAINING data only
  Step 07: Initial Model Training
  Step 08: Overfitting / Underfitting Detection & Rectification
  Step 09: Apply Correction Technique & Retrain
  Step 10: Final Test Evaluation

Usage:
  python run_full_pipeline.py --all
  python run_full_pipeline.py --step 1
  python run_full_pipeline.py --step 3
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

STEP_MODULES = {
    1: ("01_load_image_dataset.run_load_dataset", "run_step_1", "Load Image Dataset"),
    2: ("02_data_preprocessing.run_preprocessing", "run_step_2", "Data Preprocessing"),
    3: ("03_data_leakage_checks.run_leakage_checks", "run_step_3", "Data Leakage Checks"),
    4: ("04_clahe_enhancement.run_clahe", "run_step_4", "CLAHE Contrast Enhancement"),
    5: ("05_splitting_and_cross_validation.run_splitting_pipeline", "run_step_5", "Splitting & 10-Fold CV"),
    6: ("06_gan_augmentation.run_gan_pipeline", "run_step_6", "GAN Augmentation (Train Only)"),
    7: ("07_initial_model_training.run_initial_training", "run_step_7", "Initial Model Training"),
    8: ("08_overfitting_underfitting_detection.run_detection", "run_step_8", "Overfitting/Underfitting Detection"),
    9: ("09_correction_and_retraining.run_correction_retraining", "run_step_9", "Apply Corrections & Retrain"),
    10: ("10_final_test_evaluation.run_final_evaluation", "run_step_10", "Final Test Evaluation"),
}


def run_pipeline_step(step_num: int) -> bool:
    if step_num not in STEP_MODULES:
        print(f"Invalid step number: {step_num}")
        return False

    mod_name, func_name, title = STEP_MODULES[step_num]
    print(f"\n{'#' * 75}")
    print(f"  EXECUTING STEP {step_num:02d}: {title.upper()}")
    print(f"{'#' * 75}")

    start_time = time.time()
    try:
        module = importlib.import_module(mod_name)
        runner_fn = getattr(module, func_name)
        runner_fn()
        elapsed = time.time() - start_time
        print(f"\n[OK] Step {step_num:02d} ({title}) completed successfully in {elapsed:.2f}s.")
        return True
    except Exception as e:
        print(f"\n[ERROR] Step {step_num:02d} failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Run the complete 10-step sequence")
    parser.add_argument("--step", type=int, choices=range(1, 11), help="Run a specific step (1-10)")
    parser.add_argument("--from-step", type=int, choices=range(1, 11), default=1, help="Start from a specific step")
    args = parser.parse_args()

    if args.step:
        success = run_pipeline_step(args.step)
        sys.exit(0 if success else 1)

    if args.all:
        print("\n" + "=" * 75)
        print("  LAUNCHING FULL 10-STEP IMAGE CLASSIFICATION PIPELINE SEQUENCE")
        print("=" * 75)
        overall_start = time.time()

        for step_num in range(args.from_step, 11):
            ok = run_pipeline_step(step_num)
            if not ok:
                print(f"\nPipeline halted at Step {step_num}.")
                sys.exit(1)

        total_elapsed = time.time() - overall_start
        print("\n" + "=" * 75)
        print(f"  ALL 10 STEPS COMPLETED SUCCESSFULLY IN {total_elapsed:.2f}s!")
        print("=" * 75 + "\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
