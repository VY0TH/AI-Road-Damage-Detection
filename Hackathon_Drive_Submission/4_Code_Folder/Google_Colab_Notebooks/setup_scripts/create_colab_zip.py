"""Helper Script: Create a lightweight clean zip archive of the project for Google Colab.

Packages:
- 10-step sequence folders (01 to 10)
- techniques folders
- Active classification dataset (dataset/processed/efficientnet_crops)
- Core scripts, config, requirements, and Colab notebook (~25 MB total)
"""
from __future__ import annotations

import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ZIP = PROJECT_ROOT.parent / "RoadDamageProject_Colab_Ready.zip"

INCLUDE_DIRS = [
    "models",
    "analysis",
    "evaluation",
    "techniques",
    "notebooks",
    "01_load_image_dataset",
    "02_data_preprocessing",
    "03_data_leakage_checks",
    "04_clahe_enhancement",
    "05_splitting_and_cross_validation",
    "06_gan_augmentation",
    "07_initial_model_training",
    "08_overfitting_underfitting_detection",
    "09_correction_and_retraining",
    "10_final_test_evaluation",
    "dataset/processed",
    "dataset/subsets/multicountry/RDD2022/split",
    "results/final_evaluation",
    "results/diagnostics",
    "results/cross_validation",
]

INCLUDE_FILES = [
    "run_algorithm.py",
    "pipeline_config.py",
    "run_full_pipeline.py",
    "requirements.txt",
    "README.md",
    "yolo12n.pt",
    "results/final_evaluation/Road_Damage_Comprehensive_Benchmark.xlsx",
]


def make_clean_zip() -> None:
    print(f"Creating Colab-ready project zip at: {OUTPUT_ZIP}")
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    included_count = 0
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add directories
        for rel_dir in INCLUDE_DIRS:
            target_dir = PROJECT_ROOT / rel_dir
            if not target_dir.is_dir():
                continue
            for f in target_dir.rglob("*"):
                if f.is_file() and not f.name.endswith((".pyc", ".ipynb_checkpoints")):
                    arcname = f"RoadDamageProject/{f.relative_to(PROJECT_ROOT)}"
                    zipf.write(f, arcname=arcname)
                    included_count += 1

        # Add single files
        for rel_file in INCLUDE_FILES:
            target_file = PROJECT_ROOT / rel_file
            if target_file.is_file():
                arcname = f"RoadDamageProject/{rel_file}"
                zipf.write(target_file, arcname=arcname)
                included_count += 1

    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"\n[SUCCESS] Project archived: {OUTPUT_ZIP}")
    print(f"  Files included: {included_count}")
    print(f"  Archive Size  : {size_mb:.2f} MB")
    print("  -> Ultra-lightweight and ready for Google Colab!")


if __name__ == "__main__":
    make_clean_zip()
