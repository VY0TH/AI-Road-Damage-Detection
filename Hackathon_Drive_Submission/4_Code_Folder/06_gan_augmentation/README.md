# Step 6: GAN Augmentation (TRAINING Data Only)

## Overview
Generative Adversarial Network (GAN) augmentation is utilized to address acute class imbalance (such as rare D40 potholes or specific crack classes) by synthesizing labeled damaged pavement crops.

### Non-Negotiable Core Rule: TRAINING Data Only
- **Training the GAN**: The generator and discriminator are trained **strictly on real training crops**.
- **Augmentation Injection**: Synthetic samples are added **exclusively to the training split**.
- **Evaluation Isolation**: Validation and final test partitions remain **100% real images**. Synthetic samples are NEVER allowed to enter validation or test splits under any circumstances.

## Files
- `gan_model.py`: Conditional Generator and Discriminator PyTorch neural networks.
- `train_gan.py`: GAN trainer with strict training-only input enforcement.
- `generate_gan_samples.py`: Class-conditioned sampler generating 224x224 RGB crops for minority classes.
- `inject_to_training.py`: Training-set injector and automated isolation auditor.
- `run_gan_pipeline.py`: End-to-end execution script.

## Usage
```powershell
python 06_gan_augmentation/run_gan_pipeline.py
```
Checkpoints are saved to `results/gan_checkpoints/`, samples to `results/gan_augmented_training_samples/`, and augmented training data to `dataset/processed/augmented_training_crops/`.
