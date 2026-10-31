# Technique: YOLO12n Object Detection & Localization

## Overview
YOLO12n performs anchor-free spatial bounding box localization on full-frame road imagery. It outputs bounding coordinates `[xmin, ymin, xmax, ymax]` and detector confidence scores for road damage regions.

## Components
- `dataset_converter.py`: Converts VOC XML annotations to normalized YOLO format and writes `data.yaml`.
- `train.py`: Training engine supporting baseline and high-recall fine-tuning.
- `detector.py`: Inference detector for images or video frames.
- `evaluate.py`: Standard COCO evaluation computing mAP@50 and mAP@50:95.
