"""YOLO Validation & mAP Evaluation Script."""
from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO


def evaluate_yolo(checkpoint_path: Path, data_yaml: Path, imgsz: int = 640) -> dict:
    model = YOLO(str(checkpoint_path))
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, split="val", plots=True)
    return {
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }
