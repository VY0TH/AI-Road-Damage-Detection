# Step 3: Data Leakage Checks

## Overview
Rigorous validation to prevent overestimating model generalization ability:
1. **Target Leakage**:
   - Ensures no ground-truth annotations or proxy labels appear in image metadata, EXIF tags, or path variables.
   - Enforces the strict rule that preprocessing statistics (normalization mean/std) are computed exclusively on the training split.
2. **Duplicate Leakage**:
   - Computes SHA-256 digests to flag exact cross-split duplicate images.
   - Computes difference perceptual hashes (`dHash`) with Hamming distance metrics to detect continuous sequential dashcam frames spanning train and test partitions.
3. **Suspicious Features**:
   - Identifies spurious shortcuts, such as country/camera prefix bias where a particular class is sourced exclusively from one environment.
   - Checks for class-conditioned aspect ratio and dimension disparities.

## Files
- `check_target_leakage.py`: Metadata and featurization independence checks.
- `check_duplicate_leakage.py`: SHA-256 exact & dHash perceptual near-duplicate auditor.
- `check_suspicious_features.py`: Domain prefix correlation and shortcut detector.
- `run_leakage_checks.py`: Unified audit runner.

## Usage
```powershell
python 03_data_leakage_checks/run_leakage_checks.py
```
Reports are written to `results/step03_leakage_audit_report.json` and `.md`.
