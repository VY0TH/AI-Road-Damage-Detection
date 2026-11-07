"""YOLO12n Training & Fine-Tuning Pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def train_yolo(
    data_yaml: Path,
    output_dir: Path,
    epochs: int = 10,
    imgsz: int = 640,
    batch: int = 4,
    weights: str = "yolo12n.pt",
    device: str | None = None,
) -> Path:
    dev = device or ("0" if torch.cuda.is_available() else "cpu")
    data_yaml = Path(data_yaml).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    weights_path = Path(weights)
    if not weights_path.is_file():
        # Check root directory for weights
        cand = PROJECT_ROOT / weights
        if cand.is_file():
            weights_path = cand

    print(f"Training YOLO on {dev} for {epochs} epochs...")
    model = YOLO(str(weights_path))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=dev,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        save=True,
        plots=True,
    )

    best_pt = output_dir / "weights" / "best.pt"
    print(f"YOLO training completed. Best checkpoint: {best_pt}")
    return best_pt
