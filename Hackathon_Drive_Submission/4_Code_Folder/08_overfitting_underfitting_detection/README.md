# Step 8: Overfitting / Underfitting Detection & Rectification

## Overview
Monitors learning trajectories and provides automated, quantitative fit diagnosis:
1. **Underfitting Detection**:
   - Model capacity is insufficient or learning is stunted.
   - Train accuracy < benchmark (< 65%), train loss and validation loss both remain high.
2. **Overfitting Detection**:
   - Generalization gap ($Acc_{train} - Acc_{val} \ge 10\%$).
   - Validation loss increases across recent epochs while training loss continues downward.
3. **Rectification Formulation**:
   - Computes targeted countermeasures:
     - For Overfitting: Higher Dropout, Weight Decay ($L_2$), Label Smoothing, advanced data augmentations.
     - For Underfitting: Higher capacity architectures (transfer learning via EfficientNet/ResNet50), Cosine Annealing learning rate schedule with warmup, class-weighted loss, longer training.

## Files
- `learning_curve_analyzer.py`: Visualizes loss and accuracy curves to publication-quality plots.
- `fit_detector.py`: Algorithmic rule-based detector.
- `rectification_advisor.py`: Generates the exact remediation strategy for Step 9.
- `run_detection.py`: CLI runner.

## Usage
```powershell
python 08_overfitting_underfitting_detection/run_detection.py --model-name baseline_cnn
```
Outputs are written to `results/learning_curves/` and `results/step08_fit_diagnosis_report.json` and `.md`.
