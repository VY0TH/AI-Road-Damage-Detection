"""Compare YOLO checkpoints on an image at several confidence thresholds.

From the repository root:
    python src/90_debug_pothole_detection.py dataset/subsets/multicountry/RDD2022/yolo/images/train/China_0059.jpg
    python src/90_debug_pothole_detection.py --find-pothole-image --split val
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
DATASET = ROOT / "dataset/subsets/multicountry/RDD2022/yolo"
DEFAULT_CHECKPOINT = ROOT / "results/training/yolo12n_30_epochs/weights/best.pt"
DEFAULT_THRESHOLDS = (0.01, 0.05, 0.10, 0.25, 0.40)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_pothole_image(split: str) -> Path:
    """Find an image paired with a valid D40 label, honoring any extension."""
    labels = DATASET / "labels" / split
    images = DATASET / "images" / split
    if not labels.is_dir() or not images.is_dir():
        raise FileNotFoundError(f"Missing {split} image or label directory under {DATASET}")
    class_id = pothole_class_id()
    for label_path in sorted(labels.glob("*.txt")):
        try:
            has_pothole = any(
                line.split() and line.split()[0] == str(class_id)
                for line in label_path.read_text(encoding="utf-8").splitlines()
            )
        except OSError:
            continue
        if not has_pothole:
            continue
        for image_path in sorted(images.glob(label_path.stem + ".*")):
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                return image_path
    raise FileNotFoundError(f"No image paired with a D40 (class {class_id}) annotation in {split} split")


def pothole_class_id() -> int:
    config_path = DATASET / "data.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset class map is missing: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    names = config.get("names", {})
    names = {int(key): value for key, value in names.items()} if isinstance(names, dict) else dict(enumerate(names))
    matches = [class_id for class_id, name in names.items() if str(name) == "D40"]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one D40 class in {config_path}, found: {matches}")
    return matches[0]


def dataset_class_names() -> dict[int, str]:
    config = yaml.safe_load((DATASET / "data.yaml").read_text(encoding="utf-8"))
    names = config.get("names", {})
    return ({int(key): str(value) for key, value in names.items()}
            if isinstance(names, dict)
            else {index: str(value) for index, value in enumerate(names)})


def draw_result(image, result, output_path: Path, ground_truth_path: Path | None = None):
    canvas = image.copy()
    names = result.names
    detections = []
    if ground_truth_path is not None and ground_truth_path.is_file():
        height, width = image.shape[:2]
        for line in ground_truth_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            class_id = int(fields[0])
            center_x, center_y, box_w, box_h = map(float, fields[1:5])
            x1 = int((center_x - box_w / 2) * width)
            y1 = int((center_y - box_h / 2) * height)
            x2 = int((center_x + box_w / 2) * width)
            y2 = int((center_y + box_h / 2) * height)
            class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 80, 20), 2)
            cv2.putText(canvas, f"GT {class_name}", (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 20), 2, cv2.LINE_AA)
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
            confidence = float(box.conf[0].item())
            coordinates = [int(v) for v in box.xyxy[0].detach().cpu().tolist()]
            detections.append((class_id, str(class_name), confidence, coordinates))
            x1, y1, x2, y2 = coordinates
            color = (0, 180, 0) if str(class_name) == "D40" else (0, 150, 240)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, f"{class_name} {confidence:.2f}", (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Could not write annotated image: {output_path}")
    return detections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path, help="Road image to inspect")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help=f"YOLO weights (default: {DEFAULT_CHECKPOINT.relative_to(ROOT)})")
    parser.add_argument("--find-pothole-image", action="store_true",
                        help="Find an image paired with a D40 annotation (ID read from data.yaml)")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/pothole_debug")
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--label-path", type=Path,
                        help="Optional YOLO label file to overlay in blue")
    args = parser.parse_args()

    image_path = find_pothole_image(args.split) if args.find_pothole_image else args.image
    if image_path is None:
        parser.error("provide an image path or use --find-pothole-image")
    image_path = image_path.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not image_path.is_file():
        parser.error(f"Image does not exist: {image_path}")
    if image_path.suffix.lower() not in IMAGE_SUFFIXES:
        parser.error(f"Unsupported image type: {image_path.suffix}")
    if not checkpoint.is_file():
        parser.error(f"YOLO checkpoint does not exist: {checkpoint}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"OpenCV could not decode image: {image_path}")
    if args.imgsz <= 0 or any(not 0 < threshold < 1 for threshold in args.thresholds):
        parser.error("imgsz must be positive and each confidence threshold must be between 0 and 1")
    label_path = args.label_path.expanduser().resolve() if args.label_path else None
    if label_path is None:
        try:
            relative_image = image_path.relative_to((DATASET / "images").resolve())
            label_path = (DATASET / "labels" / relative_image.parent / f"{image_path.stem}.txt").resolve()
        except ValueError:
            pass

    model = YOLO(str(checkpoint))
    checkpoint_names = (
        {int(key): str(value) for key, value in model.names.items()}
        if isinstance(model.names, dict)
        else {index: str(value) for index, value in enumerate(model.names)}
    )
    dataset_names = dataset_class_names()
    print(f"Image: {image_path} ({image.shape[1]}x{image.shape[0]})")
    print(f"Checkpoint: {checkpoint}")
    print(f"Checkpoint classes: {checkpoint_names}")
    print(f"Number of classes: {len(checkpoint_names)}")
    print(f"Dataset D40 pothole class ID: {pothole_class_id()}")
    print(f"Checkpoint class mapping compatible with dataset: {checkpoint_names == dataset_names}")
    if checkpoint_names != dataset_names:
        print(f"WARNING: dataset class names are {dataset_names}; ground-truth comparisons may be invalid.")
    print(f"Ground-truth label overlay: {label_path if label_path and label_path.is_file() else 'none'}")
    print(f"Inference device: {args.device}; image size: {args.imgsz}")
    print(f"Output directory: {args.output_dir.resolve()}")

    checkpoint_label = (
        "_".join(checkpoint.relative_to(ROOT).parts[:-1])
        if checkpoint.is_relative_to(ROOT) else checkpoint.parent.name
    )
    for threshold in args.thresholds:
        started = time.perf_counter()
        result = model.predict(source=image, conf=threshold, imgsz=args.imgsz,
                               device=args.device, verbose=False)[0]
        elapsed = time.perf_counter() - started
        output_path = args.output_dir / (
            f"{image_path.stem}_{checkpoint_label}"
            f"_s{args.imgsz}_conf_{threshold:.2f}.jpg"
        )
        detections = draw_result(image, result, output_path, label_path)
        print(f"\nconf={threshold:.2f}; detections={len(detections)}; inference={elapsed:.3f}s")
        for class_id, class_name, confidence, coordinates in detections:
            print(f"  id={class_id} name={class_name} confidence={confidence:.4f} xyxy={coordinates}")
        print(f"  annotated={output_path.resolve()} (exists={output_path.is_file()})")


if __name__ == "__main__":
    main()
