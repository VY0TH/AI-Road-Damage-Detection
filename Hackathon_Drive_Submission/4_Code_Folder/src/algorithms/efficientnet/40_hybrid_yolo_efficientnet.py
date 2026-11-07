
from pathlib import Path
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from torchvision import models, transforms
from torch import nn

# ---------------- PATHS ----------------
PROJECT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())

YOLO_WEIGHTS = (
    PROJECT
    / "results/training/model1_yolo12_efficientnet/"
      "baseline_15_epochs/weights/best.pt"
)

EFFICIENTNET_WEIGHTS = (
    PROJECT / "results/efficientnet_b0/best_efficientnet_b0.pt"
)

OUTPUT_DIR = PROJECT / "results/hybrid_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use your existing validation images by default.
IMAGE_DIR = (
    PROJECT / "dataset/subsets/multicountry/RDD2022/"
    "split/val/images"
)

DEVICE = torch.device("cpu")
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.25

# Must match the class order used to train EfficientNetB0.
CLASS_NAMES = [
    "D00", "D01", "D0w0", "D10", "D11",
    "D20", "D40", "D43", "D44", "D50"
]

# ---------------- LOAD YOLO ----------------
if not YOLO_WEIGHTS.exists():
    raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}")

if not EFFICIENTNET_WEIGHTS.exists():
    raise FileNotFoundError(
        f"EfficientNet weights not found: {EFFICIENTNET_WEIGHTS}"
    )

detector = YOLO(str(YOLO_WEIGHTS))
print("YOLO detector loaded.")

# ---------------- LOAD EFFICIENTNET ----------------
checkpoint = torch.load(
    EFFICIENTNET_WEIGHTS,
    map_location=DEVICE,
    weights_only=False
)

saved_classes = checkpoint.get("class_names")
if saved_classes != CLASS_NAMES:
    raise ValueError(
        f"Class mapping mismatch. Checkpoint: {saved_classes}"
    )

classifier = models.efficientnet_b0(weights=None)
classifier.classifier[1] = nn.Linear(
    classifier.classifier[1].in_features,
    len(CLASS_NAMES)
)
classifier.load_state_dict(checkpoint["model_state_dict"])
classifier.to(DEVICE)
classifier.eval()

print("EfficientNetB0 classifier loaded.")

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ---------------- PREDICTION ----------------
def classify_crop(crop_bgr):
    """Return EfficientNet's predicted class and confidence."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(crop_rgb)
    tensor = preprocess(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = classifier(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        class_id = int(probabilities.argmax().item())
        confidence = float(probabilities[class_id].item())

    return CLASS_NAMES[class_id], confidence


def process_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Skipping unreadable image: {image_path}")
        return

    height, width = image.shape[:2]
    annotated = image.copy()

    results = detector.predict(
        source=image,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
        device="cpu"
    )

    detection_count = 0

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            coords = box.xyxy[0].cpu().tolist()
            x1, y1, x2, y2 = [int(v) for v in coords]

            # Clip box coordinates to image dimensions.
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # YOLO detector prediction.
            yolo_id = int(box.cls[0].item())
            yolo_conf = float(box.conf[0].item())
            yolo_name = detector.names[yolo_id]

            # EfficientNet classification of the detected region.
            eff_name, eff_conf = classify_crop(crop)

            # Draw bounding box and both model predictions.
            color = (0, 200, 0)
            cv2.rectangle(
                annotated, (x1, y1), (x2, y2), color, 2
            )

            label = (
                f"YOLO:{yolo_name} {yolo_conf:.2f} | "
                f"EffNet:{eff_name} {eff_conf:.2f}"
            )

            text_y = max(20, y1 - 8)
            cv2.putText(
                annotated,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA
            )

            detection_count += 1
            print(
                f"{image_path.name} | "
                f"YOLO={yolo_name} ({yolo_conf:.3f}) | "
                f"EfficientNet={eff_name} ({eff_conf:.3f})"
            )

    output_path = OUTPUT_DIR / f"{image_path.stem}_hybrid.jpg"
    cv2.imwrite(str(output_path), annotated)

    print(
        f"Processed {image_path.name}: "
        f"{detection_count} detections -> {output_path}"
    )


def main():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMAGE_DIR}")

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

    # First test a small number of images on CPU.
    test_images = image_paths[:10]
    print(f"\nTesting hybrid model on {len(test_images)} images.")

    for image_path in test_images:
        process_image(image_path)

    print("\nHybrid inference completed.")
    print("Annotated results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
