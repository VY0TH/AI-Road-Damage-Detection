
from pathlib import Path
import cv2
import torch
from torch import nn
from PIL import Image
from ultralytics import YOLO
from torchvision.models import vit_b_16
from torchvision import transforms

PROJECT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())

YOLO_WEIGHTS = (
    PROJECT / "results/training/model1_yolo12_efficientnet/"
    "baseline_15_epochs/weights/best.pt"
)
VIT_WEIGHTS = PROJECT / "results/vit/best_vit.pt"

IMAGE_DIR = (
    PROJECT / "dataset/subsets/multicountry/RDD2022/"
    "split/val/images"
)
OUTPUT_DIR = PROJECT / "results/hybrid_vit_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.25

CLASS_NAMES = [
    "D00", "D01", "D0w0", "D10", "D11",
    "D20", "D40", "D43", "D44", "D50"
]

for path in (YOLO_WEIGHTS, VIT_WEIGHTS):
    if not path.exists():
        raise FileNotFoundError(f"Model weights not found: {path}")

# Load YOLO detector
detector = YOLO(str(YOLO_WEIGHTS))
print("YOLO12n loaded.")

# Load trained ViT classifier
checkpoint = torch.load(
    VIT_WEIGHTS,
    map_location=DEVICE,
    weights_only=False
)

if checkpoint.get("class_names") != CLASS_NAMES:
    raise ValueError(
        f"Class mapping mismatch: {checkpoint.get('class_names')}"
    )

model = vit_b_16(weights=None)
model.heads.head = nn.Linear(
    model.heads.head.in_features,
    len(CLASS_NAMES)
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()
print("Vision Transformer loaded.")

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def classify_crop(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        class_id = int(probabilities.argmax().item())
        confidence = float(probabilities[class_id].item())

    return CLASS_NAMES[class_id], confidence


def process_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        print("Could not read:", image_path)
        return

    height, width = image.shape[:2]
    annotated = image.copy()
    count = 0

    results = detector.predict(
        source=image,
        conf=CONFIDENCE_THRESHOLD,
        device="cpu",
        verbose=False
    )

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].cpu().tolist()
            )

            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            yolo_id = int(box.cls[0].item())
            yolo_name = detector.names[yolo_id]
            yolo_conf = float(box.conf[0].item())

            vit_name, vit_conf = classify_crop(crop)

            label = (
                f"YOLO:{yolo_name} {yolo_conf:.2f} | "
                f"ViT:{vit_name} {vit_conf:.2f}"
            )

            cv2.rectangle(
                annotated,
                (x1, y1), (x2, y2),
                (0, 200, 0), 2
            )
            cv2.putText(
                annotated,
                label,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 200, 0),
                1,
                cv2.LINE_AA
            )

            print(
                f"{image_path.name}: YOLO={yolo_name} "
                f"({yolo_conf:.3f}), ViT={vit_name} "
                f"({vit_conf:.3f})"
            )
            count += 1

    output_path = OUTPUT_DIR / f"{image_path.stem}_vit.jpg"
    if not cv2.imwrite(str(output_path), annotated):
        print("Failed to save:", output_path)
        return

    print(f"{image_path.name}: {count} detections -> {output_path}")


def main():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMAGE_DIR}")

    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    if not images:
        raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

    test_images = images[:10]
    print(f"Testing hybrid YOLO12n + ViT on {len(test_images)} images.")

    for image_path in test_images:
        process_image(image_path)

    print("Finished. Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
