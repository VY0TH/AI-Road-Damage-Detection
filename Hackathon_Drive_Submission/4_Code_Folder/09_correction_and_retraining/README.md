# Step 9: Apply Correction Technique & Retrain

## Overview
Executes the engineering interventions formulated in Step 8 to overcome model limitations:
1. **Backbone Upgrade**: Employs transfer learning with pretrained representations (`EfficientNet-B0` / `ResNet-50`).
2. **Loss Correction**: Class-weighted cross-entropy loss computed from inverse class frequencies to elevate recall on rare damage types (e.g. D40 pothole).
3. **Regularization & Smoothing**: Label smoothing ($\alpha = 0.1$) to prevent overconfidence on noisy road pavement annotations.
4. **Learning Rate Scheduling**: Cosine Annealing with Warmup for smooth convergence.
5. **Data Augmentation**: Integrates GAN synthetic training samples (from Step 6) and spatial/color jitter.
6. **Comparative Verification**: Directly contrasts initial baseline vs. corrected retrained model.

## Files
- `corrections/loss_corrections.py`: Class-weighted loss and Focal Loss.
- `corrections/regularization_corrections.py`: Label smoothing loss and regularization.
- `retrainer.py`: PyTorch retraining loop with integrated corrections.
- `compare_before_after.py`: Generates comparative bar plots and delta calculations.
- `run_correction_retraining.py`: CLI runner.

## Usage
```powershell
python 09_correction_and_retraining/run_correction_retraining.py --model efficientnet --epochs 2
```
Outputs are written to `results/corrected_retraining/<model_name>/` and `results/step09_correction_retraining_summary.json`.
