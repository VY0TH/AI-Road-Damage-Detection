"""Step 5 Master Runner: Executes Splitting (Train/Val/Test) and 10-Fold CV Setup."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_val_test_split.run_split import run_train_val_test_split
from ten_fold_cross_validation.run_cv import run_10_fold_setup


def run_step_5() -> dict:
    print(f"\n{'=' * 70}")
    print(f"[STEP 5] DATASET SPLITTING & CROSS-VALIDATION")
    print(f"{'=' * 70}")

    split_res = run_train_val_test_split()
    cv_res = run_10_fold_setup()

    print(f"\n{'=' * 70}")
    print(f"[STEP 5 COMPLETED] Train/Val/Test Split and 10-Fold CV ready.")
    print(f"{'=' * 70}\n")
    return {"split": split_res, "cv": cv_res}


if __name__ == "__main__":
    run_step_5()
