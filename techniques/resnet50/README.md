# Technique: ResNet-50

## Overview
ResNet-50 uses 50 deep residual layers with bottleneck residual connections ($1\times1$, $3\times3$, $1\times1$ convolutions) to prevent vanishing gradients and capture multi-scale pavement cracking textures.

## Components
- `model.py`: ResNet-50 architecture with custom classification head.
- `train.py`: Unified trainer with class weighting and label smoothing.
- `inference.py`: Single/batch inference engine with optional CLAHE enhancement.
