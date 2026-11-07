
from pathlib import Path
import csv
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())

YOLO_CHECKPOINT = (
    ROOT / "results" / "training" /
    "yolo12n_improved_30_epochs" / "weights" / "best.pt"
)

# ResNet50 ImageNet weights for the initial pipeline integration test.
# This is NOT yet a road-damage-trained classifier.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE = 0.10
IMAGE_SIZE = 416

OUTPUT_DIR = ROOT / "results" / "yolo_resnet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Matches the 10 class names in the YOLO dataset configuration.
CLASS_NAMES = [
    "D00", "D01", "D10", "D11", "D20",
    "D40", "D43", "D44", "D50", "D0w0"
]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_resnet():
    # Downloads pretrained weights if they are not already cached.
    try:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            "Could not load pretrained ResNet50 weights. "
            "Check your internet connection or model cache."
        ) from exc

    model.eval()
    model.to(DEVICE)
    return model


def classify_crop(model, crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = probabilities.max(dim=1)

    # ImageNet class prediction is used only to test pipeline execution.
    label = models.ResNet50_Weights.DEFAULT.meta["categories"][
        predicted.item()
    ]
    return label, float(confidence.item())


def main():
    print("Device:", DEVICE)
    print("Loading YOLO detector...")

    if not YOLO_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"YOLO checkpoint not found: {YOLO_CHECKPOINT}"
        )

    detector = YOLO(str(YOLO_CHECKPOINT))
    classifier = load_resnet()

    image_path = input(
        "Enter the full path to a road image: "
    ).strip().strip('"')

    image_path = Path(image_path)
    if not image_path.is_absolute():
        image_path = ROOT / image_path

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Could not read the supplied image.")

    result = detector.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        verbose=False
    )[0]

    annotated = image.copy()
    rows = []

    if result.boxes is None or len(result.boxes) == 0:
        print("No road-damage detections at this confidence threshold.")

    else:
        for index, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].cpu().tolist()
            )
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            yolo_id = int(box.cls[0].item())
            yolo_conf = float(box.conf[0].item())

            resnet_label, resnet_conf = classify_crop(
                classifier, crop
            )

            yolo_label = (
                CLASS_NAMES[yolo_id]
                if 0 <= yolo_id < len(CLASS_NAMES)
                else str(yolo_id)
            )

            label = (
                f"YOLO:{yolo_label} "
                f"{yolo_conf:.2f} | "
                f"ResNet(ImageNet):{resnet_label} "
                f"{resnet_conf:.2f}"
            )

            cv2.rectangle(
                annotated, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )
            cv2.putText(
                annotated, label[:100],
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 255, 0), 1,
                cv2.LINE_AA
            )

            rows.append({
                "image": image_path.name,
                "detection_index": index + 1,
                "yolo_class": yolo_label,
                "yolo_confidence": yolo_conf,
                "resnet_imagenet_label": resnet_label,
                "resnet_confidence": resnet_conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })

    output_image = OUTPUT_DIR / f"{image_path.stem}_resnet.jpg"
    output_csv = OUTPUT_DIR / f"{image_path.stem}_resnet.csv"

    cv2.imwrite(str(output_image), annotated)

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image", "detection_index", "yolo_class",
                "yolo_confidence", "resnet_imagenet_label",
                "resnet_confidence", "x1", "y1", "x2", "y2"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Detections: {len(rows)}")
    print(f"Annotated image: {output_image}")
    print(f"CSV results: {output_csv}")
    print(
        "\nIMPORTANT: ResNet is currently ImageNet-pretrained, "
        "not trained to classify your road-damage classes."
    )


if __name__ == "__main__":
    main()
