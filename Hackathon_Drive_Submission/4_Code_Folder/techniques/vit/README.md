# Technique: Vision Transformer (ViT-B/16)

## Overview
ViT-B/16 treats image crops as sequences of $16\times16$ patches, using self-attention mechanisms to learn global context across the entire damage area.

## Components
- `model.py`: ViT-B/16 architecture definition.
- `train.py`: Transformer fine-tuning script.
- `inference.py`: Single/batch inference engine.
