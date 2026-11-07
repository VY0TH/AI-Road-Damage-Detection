# Step 8: Overfitting / Underfitting Diagnosis & Rectification Report

**Model Analyzed**: `baseline_cnn`  
**Diagnosis Verdict**: `UNDERFITTING`  

## Quantitative Metrics
- Final Train Accuracy: `48.47%`
- Final Val Accuracy: `53.54%`
- Generalization Gap: `-5.07%`
- Train Loss: `1.4382`, Val Loss: `1.2598`

## Rectification Plan for Step 9
- **Upgrade Model Backbone (Transfer Learning via EfficientNet-B0 or ResNet-50)**
- **Learning Rate Warmup with Cosine Annealing**
- **Extended Training Epochs (Allow model more convergence time)**
- **Class-Balanced Loss (Weighted Cross-Entropy or Focal Loss for minority classes)**
- **Reduce Excessive Regularization**

### Hyperparameter Adjustments:
- `model_architecture`: Switch to pretrained EfficientNet-B0 or ResNet-50
- `learning_rate`: Use lr=5e-4 with CosineAnnealingLR scheduler
- `epochs`: Increase training duration to 5-10 epochs
- `loss_function`: Weighted Cross-Entropy with inverse class frequencies
