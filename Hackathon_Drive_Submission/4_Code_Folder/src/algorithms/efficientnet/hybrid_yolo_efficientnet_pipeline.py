"""
Complete, Standalone Hybrid YOLO12 + EfficientNet-B0 Pipeline
Includes:
1. Configuration
2. Proper YOLO12 Detection Engine (Stage 1)
3. Proper EfficientNet-B0 Architecture & Classifier (Stage 2)
4. Detection & Crop Extraction (The Meeting Point)
5. Classification on Crops (Stage 2 Forward Pass)
6. Result Fusion & Output Display
7. Dual-Model Visual Annotation & Image Export
"""

from pathlib import Path
import os
import cv2
import torch
from torch import nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

# =====================================================================
# 1. CONFIGURATION
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

YOLO_WEIGHTS = (
    PROJECT_ROOT
    / "results/training/model1_yolo12_efficientnet/baseline_15_epochs/weights/best.pt"
)
if not YOLO_WEIGHTS.exists():
    YOLO_WEIGHTS = PROJECT_ROOT / "yolo12n.pt"

EFFICIENTNET_WEIGHTS = (
    PROJECT_ROOT / "results/efficientnet_b0/best_efficientnet_b0.pt"
)

TEST_IMAGE = (
    PROJECT_ROOT
    / "dataset/subsets/multicountry/RDD2022/split/val/images/India_0001.jpg"
)

OUTPUT_DIR = PROJECT_ROOT / "results/hybrid_yolo_efficientnet_pipeline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.25

# 10 Road Damage Classes
CLASS_NAMES = [
    "D00", "D01", "D0w0", "D10", "D11",
    "D20", "D40", "D43", "D44", "D50"
]

print(f"Device: {DEVICE}")
print(f"YOLO Checkpoint:         {YOLO_WEIGHTS}")
print(f"EfficientNet Checkpoint: {EFFICIENTNET_WEIGHTS}")

# =====================================================================
# 2. LOAD PROPER YOLO12 DETECTOR (STAGE 1)
# =====================================================================

print("\n[Stage 1] Loading YOLO12 detector...")
if not YOLO_WEIGHTS.exists():
    raise FileNotFoundError(f"YOLO weights missing: {YOLO_WEIGHTS}")

detector = YOLO(str(YOLO_WEIGHTS))
print("YOLO12 detector initialized.")

# =====================================================================
# 3. LOAD PROPER EFFICIENTNET-B0 CLASSIFIER (STAGE 2)
# =====================================================================

print("\n[Stage 2] Loading EfficientNet-B0 neural network...")
if not EFFICIENTNET_WEIGHTS.exists():
    raise FileNotFoundError(f"EfficientNet weights missing: {EFFICIENTNET_WEIGHTS}")

checkpoint = torch.load(EFFICIENTNET_WEIGHTS, map_location=DEVICE, weights_only=False)

# Instantiate pure torchvision EfficientNet-B0 architecture
classifier = models.efficientnet_b0(weights=None)
classifier.classifier[1] = nn.Linear(classifier.classifier[1].in_features, len(CLASS_NAMES))
classifier.load_state_dict(checkpoint["model_state_dict"])
classifier.to(DEVICE)
classifier.eval()

# ImageNet normalization and 224x224 tensor transforms
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
print("EfficientNet-B0 classifier loaded and set to eval mode.")

def classify_crop(crop_bgr):
    """Passes localized road damage patch through EfficientNet-B0."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = classifier(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        class_id = int(probabilities.argmax().item())
        confidence = float(probabilities[class_id].item())

    return CLASS_NAMES[class_id], confidence

# =====================================================================
# 4. RUN STAGE 1 (YOLO DETECTION & LOCALIZATION)
# =====================================================================

if not TEST_IMAGE.exists():
    raise FileNotFoundError(f"Test image not found: {TEST_IMAGE}")

print(f"\nProcessing image: {TEST_IMAGE.name}")
image_bgr = cv2.imread(str(TEST_IMAGE))
h, w = image_bgr.shape[:2]
annotated_img = image_bgr.copy()

yolo_results = detector.predict(
    source=image_bgr,
    conf=CONFIDENCE_THRESHOLD,
    device=0 if torch.cuda.is_available() else "cpu",
    verbose=False
)

# =====================================================================
# 5. MEETING POINT: FUSE YOLO CROPS INTO EFFICIENTNET-B0
# =====================================================================

print("\n========== HYBRID PREDICTION RESULTS ==========")
detections_found = 0

for res in yolo_results:
    if res.boxes is None or len(res.boxes) == 0:
        continue

    for box in res.boxes:
        coords = [int(v) for v in box.xyxy[0].cpu().tolist()]
        x1, y1, x2, y2 = coords

        # Clamp box to image bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            continue

        # Extract localized damage patch
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Step A: YOLO's Initial Verdict
        yolo_class_id = int(box.cls[0].item())
        yolo_conf = float(box.conf[0].item())
        yolo_label = detector.names.get(yolo_class_id, f"Class {yolo_class_id}")

        # Step B: EfficientNet-B0's Fine-Grained Verdict on the Crop
        eff_label, eff_conf = classify_crop(crop)

        # =============================================================
        # 6. DISPLAY COMBINED RESULTS
        # =============================================================
        detections_found += 1
        print(f"Detection #{detections_found}:")
        print(f"  Coordinates:    [{x1}, {y1}, {x2}, {y2}]")
        print(f"  YOLO12 Verdict: {yolo_label} (Conf: {yolo_conf:.4f})")
        print(f"  EffNet Verdict: {eff_label} (Conf: {eff_conf:.4f})")
        print("-" * 45)

        # =============================================================
        # 7. DRAW DUAL-MODEL ANNOTATION & SAVE IMAGE
        # =============================================================
        box_color = (0, 200, 0)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 2)

        dual_label = f"YOLO:{yolo_label} {yolo_conf:.2f} | EffNet:{eff_label} {eff_conf:.2f}"
        text_y = max(20, y1 - 8)
        cv2.putText(
            annotated_img,
            dual_label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1,
            cv2.LINE_AA
        )

if detections_found == 0:
    print("No road damage detected above confidence threshold.")

output_path = OUTPUT_DIR / f"{TEST_IMAGE.stem}_hybrid_result.jpg"
cv2.imwrite(str(output_path), annotated_img)
print(f"\nFinal combined annotated image saved to:\n-> {output_path}")
print("\nHybrid YOLO12 + EfficientNet-B0 execution completed successfully!")
