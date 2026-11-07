"""Train a separate YOLO checkpoint for road damage on the grouped split."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
DATA_YAML = ROOT / "dataset/subsets/multicountry/RDD2022/yolo_grouped/data.yaml"
BASE_WEIGHTS = ROOT / "yolo12n.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu", help="CUDA device, for example 0, or cpu")
    parser.add_argument("--name", default="yolo12n_grouped_10ep")
    args = parser.parse_args()

    if not DATA_YAML.is_file() or not BASE_WEIGHTS.is_file():
        raise FileNotFoundError("The checked-in dataset configuration or base YOLO12n weights are missing.")
    names = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))["names"]
    names_by_id = {int(index): str(name) for index, name in names.items()} if isinstance(names, dict) else dict(enumerate(map(str, names)))
    if names_by_id.get(5) != "D40":
        raise ValueError(f"Expected verified D40 pothole class at ID 5; found {names_by_id.get(5)!r}.")
    if not torch.cuda.is_available() and args.device != "cpu":
        raise RuntimeError(
            "CUDA is unavailable in this Python environment. Run this prepared training command "
            "on a CUDA machine; no CPU training was started."
        )
    output_project = ROOT / "results/training"
    output_path = output_project / args.name
    if output_path.exists():
        raise FileExistsError(f"Refusing to reuse or overwrite training output: {output_path}")

    model = YOLO(str(BASE_WEIGHTS))
    print(f"Starting new run: {output_path}")
    print(f"Device={args.device}, epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}")
    result = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0 if args.device == "cpu" else 4,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,
        patience=20,
        seed=42,
        deterministic=True,
        project=str(output_project),
        name=args.name,
        exist_ok=False,
        save=True,
        plots=True,
        val=True,
        amp=args.device != "cpu",
        close_mosaic=min(10, max(0, args.epochs - 1)),
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.0,
    )
    best = output_path / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError("Training returned without producing the expected best validation checkpoint.")
    print(f"Best validation checkpoint: {best}")
    print(f"Training results: {output_path}")
    print(f"Training return type: {type(result).__name__}")


if __name__ == "__main__":
    main()
