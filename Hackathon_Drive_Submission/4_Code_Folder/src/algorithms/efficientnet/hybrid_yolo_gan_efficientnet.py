"""Run YOLO detection followed by the GAN-augmented EfficientNetB0 classifier.

The conditional GAN is used to augment EfficientNet's training data. At inference
time, the saved GAN-augmented EfficientNet checkpoint classifies each YOLO crop;
the generator itself does not need to run.

Example (from the project root)::

    python src/algorithms/efficientnet/hybrid_yolo_gan_efficientnet.py path/to/road.jpg
    python src/algorithms/efficientnet/hybrid_yolo_gan_efficientnet.py path/to/images/
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
DEFAULT_YOLO_CHECKPOINT = ROOT / "results/training/yolo12n_30_epochs/weights/best.pt"
DEFAULT_CLASSIFIER_CHECKPOINT = ROOT / "results/efficientnet_b0_gan_augmented/best_efficientnet_b0.pt"
DEFAULT_OUTPUT_DIR = ROOT / "results/hybrid_yolo_gan_efficientnet"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _load_classifier(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"GAN-augmented EfficientNet checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not checkpoint.get("gan_augmentation_enabled") or int(checkpoint.get("gan_training_samples", 0)) < 1:
        raise ValueError(
            f"{checkpoint_path} is not marked as trained with GAN augmentation. "
            "Train the GAN-augmented classifier before using this hybrid."
        )

    class_names = checkpoint.get("class_names")
    if not isinstance(class_names, (list, tuple)) or not class_names:
        raise ValueError(f"Checkpoint has no valid class_names list: {checkpoint_path}")

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((int(checkpoint.get("image_size", 224)),) * 2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print(
        f"Loaded GAN-augmented EfficientNetB0: {checkpoint_path} "
        f"({checkpoint['gan_training_samples']} GAN training crops)"
    )
    return model, list(class_names), transform


def _classifier_prediction(model, class_names, transform, crop_bgr, device):
    # Match the CLAHE preprocessing used when the classifier crops were prepared.
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    luminance, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_bgr = cv2.cvtColor(
        cv2.merge((clahe.apply(luminance), a_channel, b_channel)),
        cv2.COLOR_LAB2BGR,
    )
    crop_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    class_id = int(probabilities.argmax().item())
    return class_names[class_id], float(probabilities[class_id].item())


def _class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def process_image(
    image_path: Path,
    detector,
    classifier,
    class_names,
    transform,
    device: torch.device,
    output_dir: Path,
    confidence: float,
    image_size: int,
):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    result = detector.predict(
        source=image,
        conf=confidence,
        iou=0.50,
        max_det=50,
        imgsz=image_size,
        device=str(device),
        verbose=False,
    )[0]

    annotated = image.copy()
    rows = []
    boxes = result.boxes
    if boxes is not None:
        height, width = image.shape[:2]
        for index, box in enumerate(boxes, start=1):
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
            x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width))
            y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue

            detector_id = int(box.cls[0].item())
            detector_name = _class_name(result.names, detector_id)
            detector_confidence = float(box.conf[0].item())
            crop_class, crop_confidence = _classifier_prediction(
                classifier, class_names, transform, image[y1:y2, x1:x2], device
            )
            agrees = detector_name == crop_class

            color = (40, 180, 80) if agrees else (0, 165, 255)
            label = (
                f"YOLO {detector_name} {detector_confidence:.2f} | "
                f"EffNet {crop_class} {crop_confidence:.2f}"
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )
            rows.append({
                "image": image_path.name,
                "detection_index": index,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "yolo_class": detector_name,
                "yolo_confidence": round(detector_confidence, 5),
                "efficientnet_class": crop_class,
                "efficientnet_confidence": round(crop_confidence, 5),
                "classes_agree": agrees,
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    output_image = output_dir / f"{image_path.stem}_yolo_gan_efficientnet.jpg"
    output_csv = output_dir / f"{image_path.stem}_yolo_gan_efficientnet.csv"
    if not cv2.imwrite(str(output_image), annotated):
        raise OSError(f"Could not save annotated image: {output_image}")

    columns = [
        "image", "detection_index", "x1", "y1", "x2", "y2",
        "yolo_class", "yolo_confidence", "efficientnet_class",
        "efficientnet_confidence", "classes_agree",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{image_path}: {len(rows)} detections")
    print(f"  Annotated image: {output_image}")
    print(f"  Results CSV:     {output_csv}")
    return len(rows)


def _image_paths(input_path: Path):
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image type: {input_path.suffix}")
        return [input_path]
    if input_path.is_dir():
        paths = sorted(
            path for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not paths:
            raise ValueError(f"No supported images found under: {input_path}")
        return paths
    raise FileNotFoundError(f"Input image or directory not found: {input_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect road damage with YOLO and classify each crop with GAN-augmented EfficientNetB0."
    )
    parser.add_argument("input", help="Image path or directory of images")
    parser.add_argument("--yolo-checkpoint", default=str(DEFAULT_YOLO_CHECKPOINT))
    parser.add_argument("--classifier-checkpoint", default=str(DEFAULT_CLASSIFIER_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--confidence", type=float, default=0.10, help="YOLO detection confidence cutoff")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--device", default="cpu", help="Inference device, for example cpu or cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = _resolve_path(args.input)
    yolo_checkpoint = _resolve_path(args.yolo_checkpoint)
    classifier_checkpoint = _resolve_path(args.classifier_checkpoint)
    output_dir = _resolve_path(args.output_dir)

    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be between 0 and 1")
    if args.imgsz < 32:
        raise ValueError("--imgsz must be at least 32")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available. Use --device cpu.")
    device = torch.device(args.device)
    if not yolo_checkpoint.is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {yolo_checkpoint}")

    paths = _image_paths(input_path)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    detector = YOLO(str(yolo_checkpoint))
    classifier, class_names, transform = _load_classifier(classifier_checkpoint, device)
    print(f"Loaded YOLO detector: {yolo_checkpoint}")
    print(f"Device: {device}; images: {len(paths)}")

    total_detections = 0
    for image_path in paths:
        total_detections += process_image(
            image_path, detector, classifier, class_names, transform,
            device, output_dir, args.confidence, args.imgsz,
        )
    print(f"\nComplete: {len(paths)} image(s), {total_detections} detection(s).")


if __name__ == "__main__":
    main()
