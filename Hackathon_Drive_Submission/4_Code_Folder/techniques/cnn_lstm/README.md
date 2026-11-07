# Technique: CNN + Spatial LSTM Video Segmentation

## Overview
Processes sequential video frames from forward-facing vehicle cameras. It combines a 2D CNN encoder for frame-level feature maps with a ConvLSTM recurrent cell to model temporal pavement evolution, outputting dense binary damage segmentation masks.

## Components
- `model.py`: Spatial ConvLSTM PyTorch network.
- `video_processor.py`: Video sequence frame processor with binary mask overlays.
