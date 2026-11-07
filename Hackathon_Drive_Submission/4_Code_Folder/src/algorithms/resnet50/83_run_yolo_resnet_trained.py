
from pathlib import Path
import csv

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())

YOLO_PATH = ROOT / "results" / "training" / \
    "yolo12n_improved_30_epochs" / "weights" / "best.pt"

RESNET_PATH = ROOT / "results" / "resnet50_road_damage" / \
    "best_resnet50.pt"

OUTPUT_DIR = ROOT / "results" / "yolo_resnet_trained"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 416
CONFIDENCE = 0.10

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_resnet():
    checkpoint = torch.load(
        RESNET_PATH, map_location=DEVICE, weights_only=False
    )

    class_names = checkpoint["class_names"]

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names


def classify_crop(model, crop, class_names):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)
        confidence, index = probabilities.max(dim=1)

    return (
        class_names[index.item()],
        float(confidence.item())
    )


def main():
    if not YOLO_PATH.exists():
        raise FileNotFoundError(f"YOLO checkpoint missing: {YOLO_PATH}")
    if not RESNET_PATH.exists():
        raise FileNotFoundError(f"ResNet checkpoint missing: {RESNET_PATH}")

    print("Device:", DEVICE)
    detector = YOLO(str(YOLO_PATH))
    classifier, class_names = load_resnet()

    image_path = Path(
        input("Enter full path to a road image: ").strip().strip('"')
    )
    if not image_path.is_absolute():
        image_path = ROOT / image_path

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    result = detector.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        verbose=False
    )[0]

    annotated = image.copy()
    rows = []

    if result.boxes is not None:
        for index, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].cpu().tolist()
            )

            height, width = image.shape[:2]
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            yolo_id = int(box.cls[0].item())
            yolo_class = str(result.names[yolo_id])
            yolo_conf = float(box.conf[0].item())
            resnet_class, resnet_conf = classify_crop(
                classifier, crop, class_names
            )

            label = (
                f"YOLO:{yolo_class} {yolo_conf:.2f} | "
                f"ResNet:{resnet_class} {resnet_conf:.2f}"
            )

            cv2.rectangle(
                annotated, (x1, y1), (x2, y2), (0, 255, 0), 2
            )
            cv2.putText(
                annotated, label[:90],
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1, cv2.LINE_AA
            )

            rows.append({
                "image": image_path.name,
                "detection_index": index + 1,
                "yolo_class_id": yolo_id,
                "yolo_class": yolo_class,
                "yolo_confidence": yolo_conf,
                "resnet_class": resnet_class,
                "resnet_confidence": resnet_conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })

    out_image = OUTPUT_DIR / f"{image_path.stem}_hybrid.jpg"
    out_csv = OUTPUT_DIR / f"{image_path.stem}_hybrid.csv"

    cv2.imwrite(str(out_image), annotated)

    fields = [
        "image", "detection_index", "yolo_class_id", "yolo_class",
        "yolo_confidence", "resnet_class", "resnet_confidence",
        "x1", "y1", "x2", "y2"
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("Detections:", len(rows))
    print("Annotated image:", out_image)
    print("CSV results:", out_csv)


if __name__ == "__main__":
    main()
