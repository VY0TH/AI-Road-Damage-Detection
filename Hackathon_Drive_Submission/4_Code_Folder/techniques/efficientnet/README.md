# Technique: EfficientNet-B0

## Overview
EfficientNet-B0 employs compound scaling (balancing depth, width, and resolution) to deliver high classification accuracy with low compute overhead.

## Components
- `model.py`: Model architecture factory with configurable dropout.
- `train.py`: Unified training script supporting both standard real data and GAN-augmented training.
- `inference.py`: Inference engine with integrated LAB-luminance CLAHE enhancement.
