
from pathlib import Path
import csv

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())

# Use the newer YOLO checkpoint
YOLO_CHECKPOINT = (
    ROOT / "results" / "training" / "yolo12n_improved_30_epochs"
    / "weights" / "best.pt"
)

CLASSIFIER_CHECKPOINT = (
    ROOT / "results" / "efficientnet_b0" / "best_efficientnet_b0.pt"
)

OUTPUT_DIR = ROOT / "results" / "hybrid_yolo_efficientnet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224

# Low threshold for debugging; may produce false positives
DETECTION_CONFIDENCE = 0.001
DEVICE = "cpu"

CLASS_NAMES = [
    "D00", "D01", "D0w0", "D10", "D11",
    "D20", "D40", "D43", "D44", "D50"
]

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


# =========================================================
# LOAD CLASSIFIER
# =========================================================

def load_classifier():
    if not CLASSIFIER_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Classifier checkpoint not found:\n{CLASSIFIER_CHECKPOINT}"
        )

    checkpoint = torch.load(
        CLASSIFIER_CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    class_names = checkpoint.get("class_names", CLASS_NAMES)

    if list(class_names) != CLASS_NAMES:
        raise ValueError(
            f"Class order mismatch.\n"
            f"Checkpoint: {class_names}\n"
            f"Script: {CLASS_NAMES}"
        )

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(class_names),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("EfficientNetB0 loaded.")
    print("Classifier classes:", class_names)

    return model, list(class_names)


# =========================================================
# LOAD DETECTOR
# =========================================================

def load_detector():
    if not YOLO_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"YOLO checkpoint not found:\n{YOLO_CHECKPOINT}"
        )

    detector = YOLO(str(YOLO_CHECKPOINT))
    print("YOLO checkpoint loaded:", YOLO_CHECKPOINT)
    print("Detector classes:", detector.names)

    return detector


# =========================================================
# SELECT IMAGE
# =========================================================

def choose_image():
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        selected = filedialog.askopenfilename(
            title="Select a road image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )
    finally:
        root.destroy()

    return Path(selected) if selected else None


# =========================================================
# RUN HYBRID INFERENCE
# =========================================================

def process_image(detector, classifier, class_names, image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Cannot read image:\n{image_path}")

    result = detector.predict(
        source=image,
        conf=DETECTION_CONFIDENCE,
        iou=0.45,
        max_det=50,
        imgsz=416,
        device=DEVICE,
        verbose=False,
    )[0]

    annotated = image.copy()
    rows = []

    if result.boxes is None or len(result.boxes) == 0:
        print("No detections found by YOLO.")
    else:
        for index, box in enumerate(result.boxes):
            x1, y1, x2, y2 = (
                box.xyxy[0].cpu().numpy().astype(int)
            )

            x1 = max(0, min(x1, image.shape[1] - 1))
            x2 = max(0, min(x2, image.shape[1]))
            y1 = max(0, min(y1, image.shape[0] - 1))
            y2 = max(0, min(y2, image.shape[0]))

            if x2 <= x1 or y2 <= y1:
                continue

            crop_bgr = image[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

            # Apply CLAHE to the crop's luminance channel
            lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8),
            )
            enhanced_l = clahe.apply(l_channel)

            enhanced_lab = cv2.merge(
                (enhanced_l, a_channel, b_channel)
            )
            enhanced_rgb = cv2.cvtColor(
                enhanced_lab, cv2.COLOR_LAB2RGB
            )

            crop_pil = Image.fromarray(enhanced_rgb)
            tensor = TRANSFORM(crop_pil).unsqueeze(0).to(DEVICE)

            with torch.inference_mode():
                logits = classifier(tensor)
                probabilities = torch.softmax(logits, dim=1)[0]
                classifier_id = int(probabilities.argmax().item())
                classifier_conf = float(
                    probabilities[classifier_id].item()
                )

            classifier_name = class_names[classifier_id]

            detector_id = int(box.cls[0].item())
            detector_conf = float(box.conf[0].item())
            detector_name = str(result.names[detector_id])

            label = (
                f"{detector_name} {detector_conf:.2f} | "
                f"{classifier_name} {classifier_conf:.2f}"
            )

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
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
                "yolo_confidence": round(detector_conf, 5),
                "efficientnet_class": classifier_name,
                "efficientnet_confidence": round(classifier_conf, 5),
                "classes_agree": detector_name == classifier_name,
            })

            print(
                f"Detection {index + 1}: "
                f"YOLO={detector_name} ({detector_conf:.3f}) | "
                f"EfficientNet={classifier_name} "
                f"({classifier_conf:.3f})"
            )

    output_image = OUTPUT_DIR / f"{image_path.stem}_hybrid.jpg"
    output_csv = OUTPUT_DIR / f"{image_path.stem}_hybrid.csv"

    if not cv2.imwrite(str(output_image), annotated):
        raise RuntimeError(f"Could not save image: {output_image}")

    columns = [
        "image", "detection_index",
        "x1", "y1", "x2", "y2",
        "yolo_class", "yolo_confidence",
        "efficientnet_class", "efficientnet_confidence",
        "classes_agree",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("\nHybrid inference complete.")
    print("Detections:", len(rows))
    print("Annotated image:", output_image)
    print("Results CSV:", output_csv)

    if rows:
        agreement = np.mean(
            [row["classes_agree"] for row in rows]
        ) * 100
        print(
            f"YOLO / EfficientNet class agreement: "
            f"{agreement:.2f}%"
        )
        print("Note: Agreement is not classification accuracy.")


def main():
    print("=" * 60)
    print("ROAD DAMAGE — YOLO12N + EFFICIENTNETB0")
    print("=" * 60)

    detector = load_detector()
    classifier, class_names = load_classifier()
    image_path = choose_image()

    if image_path is None:
        print("No image selected.")
        return

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    process_image(detector, classifier, class_names, image_path)


if __name__ == "__main__":
    main()
