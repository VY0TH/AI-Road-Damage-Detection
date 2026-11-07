"""Step 5.2 CLI Runner: Generate and Verify 10-Fold Cross-Validation Setup."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fold_generator import generate_10_folds
from cv_evaluator import CrossValidationAggregator
import pipeline_config as cfg


def run_10_fold_setup(source_dir: Path | None = None, output_dir: Path | None = None) -> dict:
    source = source_dir or (cfg.DATASET_ROOT / "processed" / "efficientnet_crops" / "train")
    if not source.is_dir():
        source = cfg.DATASET_ROOT / "processed" / "efficientnet_crops"

    target = output_dir or (cfg.RESULTS_DIR / "cross_validation_10fold")
    print(f"\n[STEP 5.2] Generating Stratified Group-Aware 10-Fold CV Configuration")
    print(f"  Source : {source}")
    print(f"  Target : {target}")

    cv_results = generate_10_folds(source, target, num_folds=10)

    print(f"\n10-Fold Partitioning Summary:")
    for fold_name, stats in cv_results["folds"].items():
        print(f"  {fold_name:8}: {stats['total_samples']} samples across {len(stats['class_counts'])} classes")

    report_path = target / "cv_setup_summary.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(cv_results, f, indent=2)

    print(f"\nManifest generated: {cv_results['manifest_path']}")
    print(f"Summary saved: {report_path}\n")
    return cv_results


if __name__ == "__main__":
    run_10_fold_setup()
