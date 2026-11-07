# Step 5: Dataset Splitting & 10-Fold Cross-Validation

## Overview
This step handles the two primary validation methodologies:

### 1. Train / Validation / Test Split (`train_val_test_split/`)
- Partitions the dataset into Train (70%), Validation (15%), and Test (15%) splits (or 80/20 train/test).
- **Stratified**: Retains balanced representation across all 10 road damage classes.
- **Group-Aware**: Guarantees that images with identical visual content or belonging to the same dashcam sequence/burst stay in the same split to prevent cross-split leakage.
- Generates `split_manifest.csv` and organized directory splits.

### 2. 10-Fold Cross-Validation (`ten_fold_cross_validation/`)
- Generates 10 stratified, group-aware cross-validation folds.
- Emits `10_fold_manifest.csv` and fold summaries.
- Provides CV evaluation aggregator tracking mean and std of Accuracy, Precision, Recall, and F1 across all folds.

## Usage
Run the full step:
```powershell
python 05_splitting_and_cross_validation/run_splitting_pipeline.py
```
Or run individually:
```powershell
python 05_splitting_and_cross_validation/train_val_test_split/run_split.py
python 05_splitting_and_cross_validation/ten_fold_cross_validation/run_cv.py
```
Outputs are written to `dataset/processed/train_val_test_70_15_15/` and `results/cross_validation_10fold/`.
