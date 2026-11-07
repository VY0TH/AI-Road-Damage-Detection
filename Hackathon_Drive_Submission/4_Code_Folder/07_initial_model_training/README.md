# Step 7: Initial Model Training

## Overview
Establishes the baseline classification performance for road damage identification:
1. **Architectures**:
   - `BaselineCNN`: Custom 4-stage convolutional neural network.
   - `EfficientNet-B0`: Pretrained compound-scaled CNN.
   - `ResNet-50`: Deep residual network with bottleneck blocks.
   - `ViT-B/16`: Vision Transformer using 16x16 patch embeddings and self-attention.
2. **Metrics & Logging**:
   - Every epoch logs `train_loss`, `train_accuracy`, `val_loss`, and `val_accuracy` to both `training_history.json` and `training_history.csv`.
   - Generates the foundation logs required for Step 8 (Overfitting / Underfitting Detection).

## Files
- `models/`: Architectures for BaselineCNN, EfficientNet-B0, ResNet-50, and ViT-B/16.
- `trainer.py`: PyTorch training loop and metric logger.
- `run_initial_training.py`: CLI training launcher.

## Usage
```powershell
python 07_initial_model_training/run_initial_training.py --model baseline_cnn --epochs 2
```
Or train transfer learning models:
```powershell
python 07_initial_model_training/run_initial_training.py --model efficientnet --epochs 3
```
Checkpoints and history are saved to `results/initial_training/<model_name>/`.
