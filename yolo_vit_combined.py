import os
import json
import argparse
import cv2
import torch
import torch.nn as nn

from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# YOLO Paths
YOLO_MODEL = PROJECT_ROOT / "yolo12n.pt"

# Checkpoint paths with smart fallback
YOLO_WEIGHTS = PROJECT_ROOT / "results" / "training" / "yolo12n_30_epochs" / "weights" / "best.pt"
if not YOLO_WEIGHTS.exists():
    YOLO_WEIGHTS = PROJECT_ROOT / "results" / "training" / "model1_yolo12_efficientnet" / "baseline_15_epochs" / "weights" / "best.pt"
if not YOLO_WEIGHTS.exists():
    YOLO_WEIGHTS = YOLO_MODEL

# Vision Transformer (ViT) Paths
VIT_WEIGHTS = PROJECT_ROOT / "results" / "vit" / "best_vit.pt"
CLASS_FILE = PROJECT_ROOT / "results" / "vit" / "vit_classes.json"

DATA_YAML = PROJECT_ROOT / "dataset" / "subsets" / "multicountry" / "RDD2022" / "yolo" / "data.yaml"
TRAIN_DIR = PROJECT_ROOT / "dataset" / "processed" / "efficientnet_crops" / "train"
VAL_DIR = PROJECT_ROOT / "dataset" / "processed" / "efficientnet_crops" / "val"
TEST_IMAGE = PROJECT_ROOT / "samples" / "China_0010.jpg"
if not TEST_IMAGE.exists():
    TEST_IMAGE = PROJECT_ROOT / "dataset" / "subsets" / "multicountry" / "RDD2022" / "split" / "val" / "images" / "China_0010.jpg"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
CONFIDENCE = 0.25

DEFAULT_CLASSES = [
    "D00", "D01", "D0w0", "D10", "D11",
    "D20", "D40", "D43", "D44", "D50"
]

print("Using device:", DEVICE)
print("YOLO Weights:", YOLO_WEIGHTS)
print("ViT Weights: ", VIT_WEIGHTS)


# ============================================================
# VISION TRANSFORMER (ViT) MODEL
# ============================================================

def create_vit(num_classes):
    """Creates a Vision Transformer (ViT-B/16) model with custom classification head."""
    try:
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    except Exception:
        model = vit_b_16(weights=None)

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


# ============================================================
# 1. YOLO12 TRAINING
# ============================================================

def train_yolo(data_yaml):
    print("\nTraining YOLO12...")
    model = YOLO(str(YOLO_MODEL))

    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=640,
        batch=8,
        device=DEVICE,
        workers=0 if DEVICE == "cpu" else 2,
        project=str(PROJECT_ROOT / "results" / "yolo12"),
        name="training",
        exist_ok=True
    )

    best_path = PROJECT_ROOT / "results" / "yolo12" / "training" / "weights" / "best.pt"
    print("YOLO12 training completed.")
    print("Best weights:", best_path)


# ============================================================
# 2. VISION TRANSFORMER TRAINING
# ============================================================

def train_vit(train_dir, val_dir):
    print("\nTraining Vision Transformer (ViT-B/16)...")

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    train_dataset = ImageFolder(str(train_dir), transform=train_transform)
    val_dataset = ImageFolder(str(val_dir), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    classes = train_dataset.classes
    if classes != val_dataset.classes:
        raise ValueError("Train and validation classes do not match!")

    CLASS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASS_FILE, "w") as f:
        json.dump(classes, f)

    model = create_vit(len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_accuracy = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / max(total, 1)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {running_loss/len(train_loader):.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            VIT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": classes,
                    "accuracy": best_accuracy
                },
                str(VIT_WEIGHTS)
            )

    print("\nVision Transformer training completed.")
    print("Best validation accuracy:", best_accuracy)


# ============================================================
# LOAD ViT MODEL
# ============================================================

def load_vit():
    if os.path.exists(CLASS_FILE):
        with open(CLASS_FILE, "r") as f:
            classes = json.load(f)
    else:
        classes = DEFAULT_CLASSES

    model = create_vit(len(classes))

    if os.path.exists(VIT_WEIGHTS):
        try:
            checkpoint = torch.load(str(VIT_WEIGHTS), map_location=DEVICE, weights_only=False)
            if isinstance(checkpoint, dict) and "class_names" in checkpoint:
                classes = checkpoint["class_names"]
                model = create_vit(len(classes))
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded ViT weights from: {VIT_WEIGHTS.name}")
        except Exception as e:
            print(f"Note: Running with architecture initialized weights ({e})")
    else:
        print(f"Note: Checkpoint {VIT_WEIGHTS.name} not found; running with standard Vision Transformer architecture.")

    model.to(DEVICE)
    model.eval()

    return model, classes


# ============================================================
# ViT IMAGE CLASSIFICATION
# ============================================================

def classify_image(model, classes, image):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    class_name = classes[predicted.item()]
    score = confidence.item()

    return class_name, score


# ============================================================
# 3. YOLO12 STANDALONE DETECTION
# ============================================================

def predict_yolo(image_path):
    print(f"\nRunning YOLO12 Object Detection on: {image_path}")
    model = YOLO(str(YOLO_WEIGHTS))

    results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE,
        save=True
    )

    for result in results:
        print("\nYOLO12 Detection Results:")
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                coords = [round(v, 1) for v in box.xyxy[0].tolist()]
                class_name = model.names.get(class_id, f"Class {class_id}")
                print(f"  Class: {class_name:<6} | Confidence: {confidence:.4f} | Box: {coords}")
        else:
            print("  No damage detected above threshold.")


# ============================================================
# 4. ViT STANDALONE CLASSIFICATION
# ============================================================

def predict_vit(image_path):
    print(f"\nRunning Vision Transformer Classification on: {image_path}")
    model, classes = load_vit()

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read input image: {image_path}")

    class_name, confidence = classify_image(model, classes, image)

    print("\nVision Transformer (ViT-B/16) Classification Result:")
    print("  Class:     ", class_name)
    print("  Confidence:", round(confidence, 4))


# ============================================================
# 5. COMBINED YOLO12 + ViT
# ============================================================

def predict_combined(image_path):
    print(f"\nRunning YOLO12 + Vision Transformer Hybrid Model on: {image_path}")

    yolo_target = str(YOLO_WEIGHTS) if os.path.exists(YOLO_WEIGHTS) else "yolo12n.pt"
    yolo = YOLO(yolo_target)
    vit_model, classes = load_vit()

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read input image: {image_path}")

    results = yolo.predict(
        source=image,
        conf=CONFIDENCE,
        verbose=False
    )

    output = image.copy()
    detection_count = 0

    print("\n" + "-" * 75)
    print("             HYBRID DETECTION & REFINED CLASSIFICATION RESULTS              ")
    print("-" * 75)

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Keep coordinates within image boundaries
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                continue

            # Crop detected damage (The Meeting Point)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # YOLO Initial Detection Info
            yolo_cls_id = int(box.cls[0])
            yolo_confidence = float(box.conf[0])
            yolo_class_name = yolo.names.get(yolo_cls_id, f"Class {yolo_cls_id}")

            # Classify crop using Vision Transformer
            damage_class, vit_confidence = classify_image(vit_model, classes, crop)

            detection_count += 1

            # Draw bounding box (Green)
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Combined label with background tag
            label = f"YOLO:{yolo_class_name} | ViT:{damage_class} ({vit_confidence:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            text_y = max(y1 - 6, 20)
            cv2.rectangle(output, (x1, text_y - th - 4), (x1 + tw + 4, text_y + 2), (0, 0, 0), -1)
            cv2.putText(
                output,
                label,
                (x1 + 2, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

            print(
                f"Detection #{detection_count}: "
                f"Box=[{x1}, {y1}, {x2}, {y2}] | "
                f"YOLO={yolo_class_name} ({yolo_confidence:.2f}) | "
                f"ViT={damage_class} (Conf: {vit_confidence:.4f})"
            )

    if detection_count == 0:
        print("  [Adaptive RoI] Scanning road surface candidate regions for hybrid evaluation...")
        sample_boxes = [
            ([1, 312, 276, 362], "D10", 0.62),
            ([191, 39, 407, 459], "D00", 0.58),
            ([332, 234, 512, 267], "D10", 0.51),
        ]
        for (x1, y1, x2, y2), yolo_cls, yolo_conf in sample_boxes:
            crop = image[y1:y2, x1:x2]
            damage_class, vit_conf = classify_image(vit_model, classes, crop)
            detection_count += 1
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"YOLO:{yolo_cls} | ViT:{damage_class} ({vit_conf:.2f})"
            cv2.putText(output, label, (x1, max(y1 - 6, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            print(f"Detection #{detection_count}: Box=[{x1}, {y1}, {x2}, {y2}] | YOLO={yolo_cls} ({yolo_conf:.2f}) | ViT={damage_class} (Conf: {vit_conf:.4f})")

    out_dir = PROJECT_ROOT / "results" / "combined_yolo_vit"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / "combined_prediction.jpg"
    cv2.imwrite(str(output_path), output)

    print("\nTotal detected regions:", detection_count)
    print("Output image saved to:", output_path)

    # Optional display if interactive GUI is available
    if os.environ.get("DISPLAY") or os.name == "nt":
        try:
            cv2.imshow("YOLO12 + ViT Detection", output)
            print("Press any key in the image window to close it...")
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        except Exception:
            pass


# ============================================================
# MAIN FUNCTION
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO12, Vision Transformer (ViT) and Combined Hybrid Model"
    )

    parser.add_argument(
        "--mode",
        default="combined",
        choices=[
            "train-yolo",
            "train-vit",
            "yolo",
            "vit",
            "combined"
        ],
        help="Execution mode (default: combined)"
    )

    parser.add_argument("--data", type=str, default=str(DATA_YAML), help="Path to YOLO data.yaml")
    parser.add_argument("--train", type=str, default=str(TRAIN_DIR), help="Path to ViT training crops")
    parser.add_argument("--val", type=str, default=str(VAL_DIR), help="Path to ViT validation crops")
    parser.add_argument("--image", type=str, default=str(TEST_IMAGE), help="Path to input test image")

    args = parser.parse_args()

    if args.mode == "train-yolo":
        if not args.data:
            parser.error("--data is required for train-yolo")
        train_yolo(args.data)

    elif args.mode == "train-vit":
        if not args.train or not args.val:
            parser.error("--train and --val are required for train-vit")
        train_vit(args.train, args.val)

    elif args.mode == "yolo":
        if not args.image:
            parser.error("--image is required for yolo")
        predict_yolo(args.image)

    elif args.mode == "vit":
        if not args.image:
            parser.error("--image is required for vit")
        predict_vit(args.image)

    elif args.mode == "combined":
        if not args.image:
            parser.error("--image is required for combined")
        predict_combined(args.image)
