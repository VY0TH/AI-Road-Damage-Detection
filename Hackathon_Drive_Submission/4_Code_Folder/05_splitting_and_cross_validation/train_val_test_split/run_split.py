"""Step 5.1 CLI Runner: Execute Train / Validation / Test Split."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stratified_group_splitter import split_dataset
import pipeline_config as cfg


def run_train_val_test_split(
    source_dir: Path | None = None,
    output_dir: Path | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict:
    # Use preprocessed dataset or crops
    source = source_dir or (cfg.DATASET_ROOT / "processed" / "efficientnet_crops" / "train")
    if not source.is_dir():
        source = cfg.DATASET_ROOT / "processed" / "efficientnet_crops"

    target = output_dir or (cfg.DATASET_ROOT / "processed" / "train_val_test_70_15_15")
    print(f"\n[STEP 5.1] Performing Stratified Group-Aware Train/Val/Test Split")
    print(f"  Source : {source}")
    print(f"  Target : {target}")
    print(f"  Ratios : Train={train_ratio}, Val={val_ratio}, Test={test_ratio}")

    summary = split_dataset(
        source_dir=source,
        output_dir=target,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    print("\nSplit Summary:")
    for split_name, data in summary.items():
        print(f"  [{split_name.upper():5}]: {data['total']} total samples across {len(data['classes'])} classes")

    report_path = cfg.RESULTS_DIR / "step05_split_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Report saved to: {report_path}\n")
    return summary


if __name__ == "__main__":
    run_train_val_test_split()
