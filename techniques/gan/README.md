# Technique: Conditional GAN Augmentation

## Overview
A class-conditional Generative Adversarial Network that synthesizes artificial training crops for underrepresented classes (such as D40 potholes or specific crack subcategories).

## Core Principles
1. **Training Data Exclusivity**: The GAN is strictly trained on real training images only.
2. **Evaluation Isolation**: Synthetic samples are injected into the training set only; validation and test splits remain 100% real.

## Components
- `cgan_architecture.py`: Generator and Discriminator PyTorch modules with label embedding conditioning.
- `cgan_trainer.py`: Consolidated training and sampling workflows.
