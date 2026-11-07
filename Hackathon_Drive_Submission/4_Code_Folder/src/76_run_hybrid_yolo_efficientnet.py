"""Compatibility launcher; implementation is grouped by algorithm."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "algorithms" / "efficientnet" / "76_run_hybrid_yolo_efficientnet.py"), run_name="__main__")

from pathlib import Path
import argparse
import csv
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT = Path(__file__).resolve().parents[3]

YOLO_WEIGHTS = (
    PROJECT / "results/training/yolo12n_improved_30_epochs/weights/best.pt"
)
EFFICIENTNET_WEIGHTS = (
    PROJECT / "results/efficientnet_b0/best_efficientnet_b0.pt"
)

OUTPUT_DIR = PROJECT / "results/hybrid_yolo_efficientnet"

# IMPORTANT: These must match the class order used when training
# the EfficientNet classifier.
EFFICIENNET_CLASSES = [
    "D00", "D01", "D0w0", "D10", "D11",
    "D20", "D40", "D43", "D44", "D50"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
DETECTION_CONFIDENCE = 0.25


# --------------------------------------------------
# LOAD EFFICIENTNET CHECKPOINT
# --------------------------------------------------

def load_efficientnet(checkpoint_path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"EfficientNet checkpoint not found: {checkpoint_path}"
        )

    model = models.efficientnet_b0(weights=None)
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        input_features, len(EFFICIENNET_CLASSES)
    )

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    # Support common checkpoint formats.
    if isinstance(checkpoint, dict):
        state_dict = None
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break

        if state_dict is None:
            # A checkpoint may itself be the state dictionary.
            if checkpoint and all(
                isinstance(k, str) for k in checkpoint.keys()
            ):
                state_dict = checkpoint

        if state_dict is None:
            raise ValueError(
                "Cannot find model weights in EfficientNet checkpoint. "
                "Inspect its keys and adapt the loader."
            )
    else:
        raise ValueError("Unexpected EfficientNet checkpoint format.")

    # Handle checkpoints saved from torch.nn.DataParallel.
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    print(f"Loaded EfficientNetB0: {checkpoint_path}")
    return model


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def apply_clahe(bgr_crop):
    """Apply CLAHE to LAB luminance while preserving color."""
    lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB)
    luminance, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(luminance)

    lab_enhanced = cv2.merge(
        (enhanced, a_channel, b_channel)
    )
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def classify_crop(model, crop_bgr):
    enhanced_crop = apply_clahe(crop_bgr)
    rgb = cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_index = probabilities.max(dim=1)

    index = int(predicted_index.item())
    return (
        EFFICIENNET_CLASSES[index],
        float(confidence.item()),
        enhanced_crop
    )


# --------------------------------------------------
# YOLO12 + EFFICIENTNET HYBRID INFERENCE
# --------------------------------------------------

def run_pipeline(image_path, yolo_model, efficientnet):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    original = cv2.imread(str(image_path))
    if original is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = original.shape[:2]
    annotated = original.copy()
    rows = []

    prediction = yolo_model.predict(
        source=original,
        conf=DETECTION_CONFIDENCE,
        verbose=False,
        device=0 if DEVICE.type == "cuda" else "cpu"
    )[0]

    boxes = prediction.boxes

    if boxes is None or len(boxes) == 0:
        print("No detections passed the confidence threshold.")

    else:
        for detection_index, box in enumerate(boxes):
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].detach().cpu().tolist()
            )
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = original[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            yolo_index = int(box.cls[0].item())
            yolo_name = yolo_model.names[yolo_index]
            yolo_confidence = float(box.conf[0].item())

            effnet_name, effnet_confidence, enhanced_crop = (
                classify_crop(efficientnet, crop)
            )

            agreement = (
                str(yolo_name) == effnet_name
            )

            color = (0, 200, 0) if agreement else (0, 165, 255)
            label = (
                f"YOLO:{yolo_name} {yolo_confidence:.2f} | "
                f"EffNet:{effnet_name} {effnet_confidence:.2f}"
            )

            cv2.rectangle(
                annotated, (x1, y1), (x2, y2), color, 2
            )
            cv2.putText(
                annotated,
                label[:110],
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA
            )

            rows.append({
                "image": image_path.name,
                "detection_index": detection_index,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "yolo_class": str(yolo_name),
                "yolo_confidence": yolo_confidence,
                "efficientnet_class": effnet_name,
                "efficientnet_confidence": effnet_confidence,
                "class_agreement": agreement
            })

            print(
                f"Detection {detection_index}: "
                f"YOLO={yolo_name} ({yolo_confidence:.3f}), "
                f"EfficientNet={effnet_name} "
                f"({effnet_confidence:.3f}), "
                f"agreement={agreement}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_image = OUTPUT_DIR / f"{image_path.stem}_hybrid.jpg"
    output_csv = OUTPUT_DIR / f"{image_path.stem}_hybrid.csv"

    if not cv2.imwrite(str(output_image), annotated):
        raise IOError(f"Failed to save annotated image: {output_image}")

    fieldnames = [
        "image", "detection_index",
        "x1", "y1", "x2", "y2",
        "yolo_class", "yolo_confidence",
        "efficientnet_class", "efficientnet_confidence",
        "class_agreement"
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDetections processed: {len(rows)}")
    print(f"Annotated image: {output_image}")
    print(f"CSV results:      {output_csv}")

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="YOLO12 + EfficientNetB0 road damage inference"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to an input road image"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DETECTION_CONFIDENCE,
        help="YOLO detection confidence threshold"
    )
    args = parser.parse_args()

    global DETECTION_CONFIDENCE
    DETECTION_CONFIDENCE = args.conf

    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(
            f"YOLO checkpoint not found: {YOLO_WEIGHTS}"
        )

    print("Device:", DEVICE)
    yolo_model = YOLO(str(YOLO_WEIGHTS))
    efficientnet = load_efficientnet(EFFICIENTNET_WEIGHTS)

    run_pipeline(args.image, yolo_model, efficientnet)


if __name__ == "__main__":
    main()
