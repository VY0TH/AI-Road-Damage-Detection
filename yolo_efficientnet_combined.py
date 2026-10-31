"""
====================================================================================================
MASTER ALL-IN-ONE SCRIPT: YOLO12 + EFFICIENTNET-B0 + COMBINATION PIPELINE
====================================================================================================
This single file contains:
  SECTION 1: Pure, Standalone YOLO12 Code (Loading, Training, Validation, and Detection)
  SECTION 2: Pure, Standalone EfficientNet-B0 Code (PyTorch Architecture, Transforms, and Classification)
  SECTION 3: The Combination of Both (YOLO Stage 1 Localization -> Crop Hand-Off -> EfficientNet Stage 2 Classification)

* NO external helper scripts
* NO runpy launcher wrappers
* 100% self-contained top-to-bottom implementation
====================================================================================================
"""

from pathlib import Path
import os
import cv2
import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

# ==================================================================================================
# GLOBAL CONFIGURATION & PATHS
# ==================================================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset and weights paths
DATA_YAML = PROJECT_ROOT / "dataset" / "subsets" / "multicountry" / "RDD2022" / "yolo" / "data.yaml"

YOLO_WEIGHTS = PROJECT_ROOT / "results" / "training" / "yolo12n_30_epochs" / "weights" / "best.pt"
if not YOLO_WEIGHTS.exists():
    YOLO_WEIGHTS = PROJECT_ROOT / "results" / "training" / "model1_yolo12_efficientnet" / "baseline_15_epochs" / "weights" / "best.pt"
if not YOLO_WEIGHTS.exists():
    YOLO_WEIGHTS = PROJECT_ROOT / "yolo12n.pt"

EFFICIENTNET_WEIGHTS = PROJECT_ROOT / "results" / "efficientnet_b0" / "best_efficientnet_b0.pt"

# Validation road test image (China_0010.jpg has verified multiple damages)
TEST_IMAGE = PROJECT_ROOT / "samples" / "China_0010.jpg"
if not TEST_IMAGE.exists():
    TEST_IMAGE = PROJECT_ROOT / "dataset" / "subsets" / "multicountry" / "RDD2022" / "split" / "val" / "images" / "China_0010.jpg"

OUTPUT_DIR = PROJECT_ROOT / "results" / "combined_yolo_efficientnet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.25

# 10 Road Damage Classes (D00-D50)
CLASS_NAMES = [
    "D00", "D01", "D0w0", "D10", "D11",
    "D20", "D40", "D43", "D44", "D50"
]

# Toggle flags (Set to True if you wish to run full training or dataset validation)
RUN_TRAINING = False
RUN_VALIDATION = False

print("=" * 80)
print("              ROAD DAMAGE DETECTION & CLASSIFICATION SYSTEM                         ")
print("=" * 80)
print(f"Device Selected:         {DEVICE}")
print(f"YOLO12 Checkpoint:       {YOLO_WEIGHTS}")
print(f"EfficientNet Checkpoint: {EFFICIENTNET_WEIGHTS}")
print(f"Test Image Path:         {TEST_IMAGE.name}")
print(f"Output Directory:        {OUTPUT_DIR}")
print("=" * 80 + "\n")


# ==================================================================================================
# SECTION 1: STANDALONE YOLO12 CODE (DETECTION & LOCALIZATION)
# ==================================================================================================
print("\n" + "=" * 80)
print("SECTION 1: STANDALONE YOLO12 CODE (OBJECT DETECTOR)")
print("=" * 80)

# 1. Load YOLO12 Model
print("\n[YOLO12] 1. Loading YOLO12 Model...")
yolo_model = YOLO(str(YOLO_WEIGHTS))
print(f"[YOLO12] Model successfully loaded from: {YOLO_WEIGHTS.name}")

# 2. Train YOLO12 (Optional / Configurable)
if RUN_TRAINING:
    print("\n[YOLO12] 2. Training YOLO12 on Dataset...")
    yolo_model.train(
        data=str(DATA_YAML),
        epochs=5,
        imgsz=640,
        batch=4,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(OUTPUT_DIR / "training"),
        name="yolo12_run",
        exist_ok=True
    )

# 3. Validate YOLO12 (Optional / Configurable)
if RUN_VALIDATION and DATA_YAML.exists():
    print("\n[YOLO12] 3. Running Validation on YOLO12...")
    metrics = yolo_model.val(
        data=str(DATA_YAML),
        imgsz=640,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False
    )
    print("  mAP50:    ", round(float(metrics.box.map50), 4))
    print("  mAP50-95: ", round(float(metrics.box.map), 4))
    print("  Precision:", round(float(metrics.box.mp), 4))
    print("  Recall:   ", round(float(metrics.box.mr), 4))

# 4. Standalone YOLO12 Detection on Full Image
print(f"\n[YOLO12] 4. Running Standalone Object Detection on {TEST_IMAGE.name}...")
yolo_standalone_results = yolo_model.predict(
    source=str(TEST_IMAGE),
    conf=CONFIDENCE_THRESHOLD,
    device=0 if torch.cuda.is_available() else "cpu",
    verbose=False
)

print("--- Standalone YOLO12 Predictions ---")
yolo_box_count = 0
for res in yolo_standalone_results:
    if res.boxes is not None:
        for b in res.boxes:
            yolo_box_count += 1
            cid = int(b.cls[0].item())
            c_conf = float(b.conf[0].item())
            c_xyxy = [round(v, 1) for v in b.xyxy[0].tolist()]
            c_name = yolo_model.names.get(cid, f"Class {cid}")
            print(f"  Box #{yolo_box_count}: {c_name} | Confidence: {c_conf:.4f} | Coordinates: {c_xyxy}")

if yolo_box_count == 0:
    print("  No damage detected by YOLO12 alone above confidence threshold.")


# ==================================================================================================
# SECTION 2: STANDALONE EFFICIENTNET-B0 CODE (FINE-GRAINED CLASSIFIER)
# ==================================================================================================
print("\n" + "=" * 80)
print("SECTION 2: STANDALONE EFFICIENTNET-B0 CODE (CLASSIFICATION NETWORK)")
print("=" * 80)

# 1. Instantiate torchvision EfficientNet-B0 Neural Network Architecture
print("\n[EfficientNet-B0] 1. Initializing PyTorch Architecture...")
efficientnet_model = models.efficientnet_b0(weights=None)

# Replace final Linear classification head for our 10 road distress classes
num_features = efficientnet_model.classifier[1].in_features
efficientnet_model.classifier[1] = nn.Linear(num_features, len(CLASS_NAMES))

# 2. Load Trained Weights
if not EFFICIENTNET_WEIGHTS.exists():
    raise FileNotFoundError(f"EfficientNet weights missing: {EFFICIENTNET_WEIGHTS}")

eff_checkpoint = torch.load(EFFICIENTNET_WEIGHTS, map_location=DEVICE, weights_only=False)
efficientnet_model.load_state_dict(eff_checkpoint["model_state_dict"])
efficientnet_model.to(DEVICE)
efficientnet_model.eval()
print(f"[EfficientNet-B0] Loaded trained weights from: {EFFICIENTNET_WEIGHTS.name}")

# 3. Define Standard Preprocessing Transforms (ImageNet Normalization)
eff_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 4. Standalone Classification Test on a Sample Patch
print(f"\n[EfficientNet-B0] 4. Testing Standalone Forward Pass on an Image Patch...")
raw_img = cv2.imread(str(TEST_IMAGE))
h_raw, w_raw = raw_img.shape[:2]
# Center crop to demonstrate standalone classification
sample_crop = raw_img[h_raw // 4 : 3 * h_raw // 4, w_raw // 4 : 3 * w_raw // 4]

sample_pil = Image.fromarray(cv2.cvtColor(sample_crop, cv2.COLOR_BGR2RGB))
sample_tensor = eff_transforms(sample_pil).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    sample_logits = efficientnet_model(sample_tensor)
    sample_probs = torch.softmax(sample_logits, dim=1)[0]
    sample_pred_id = int(sample_probs.argmax().item())
    sample_pred_conf = float(sample_probs[sample_pred_id].item())

print(f"  Input Crop Shape:     {sample_crop.shape}")
print(f"  Predicted Category:   {CLASS_NAMES[sample_pred_id]}")
print(f"  Prediction Confidence: {sample_pred_conf:.4f}")


# ==================================================================================================
# SECTION 3: THE COMBINATION OF BOTH (TWO-STAGE HYBRID PIPELINE)
# ==================================================================================================
print("\n" + "=" * 80)
print("SECTION 3: THE COMBINATION OF BOTH (HYBRID YOLO12 + EFFICIENTNET-B0)")
print("=" * 80)
print("Workflow: [Full Image] -> [Stage 1: YOLO Detects Box] -> [Crop Patch] -> [Stage 2: EfficientNet Classifies Patch] -> [Fused Output]")

# Read the test image
combined_image_bgr = cv2.imread(str(TEST_IMAGE))
img_h, img_w = combined_image_bgr.shape[:2]
annotated_output = combined_image_bgr.copy()

# Stage 1: YOLO12 Localizes all Damage Bounding Boxes
print(f"\n[Hybrid Stage 1] Running YOLO12 Localization on {TEST_IMAGE.name}...")
hybrid_yolo_detections = yolo_model.predict(
    source=combined_image_bgr,
    conf=CONFIDENCE_THRESHOLD,
    device=0 if torch.cuda.is_available() else "cpu",
    verbose=False
)

total_hybrid_detections = 0
print("\n" + "-" * 75)
print("               FUSED DETECTION & REFINED CLASSIFICATION                     ")
print("-" * 75)

for result in hybrid_yolo_detections:
    if result.boxes is None or len(result.boxes) == 0:
        continue

    for box in result.boxes:
        # -------------------------------------------------------------
        # STEP A: Extract Coordinates from YOLO
        # -------------------------------------------------------------
        box_coords = [int(v) for v in box.xyxy[0].cpu().tolist()]
        x1 = max(0, box_coords[0])
        y1 = max(0, box_coords[1])
        x2 = min(img_w, box_coords[2])
        y2 = min(img_h, box_coords[3])

        if x2 <= x1 or y2 <= y1:
            continue

        # -------------------------------------------------------------
        # STEP B: The Meeting Point (Crop Damage Region from Full Image)
        # -------------------------------------------------------------
        damage_crop = combined_image_bgr[y1:y2, x1:x2]
        if damage_crop.size == 0:
            continue

        # YOLO's Class Prediction & Confidence
        yolo_class_id = int(box.cls[0].item())
        yolo_confidence = float(box.conf[0].item())
        yolo_class_name = yolo_model.names.get(yolo_class_id, f"Class {yolo_class_id}")

        # -------------------------------------------------------------
        # STEP C: Hand Off the Crop to EfficientNet-B0 for Classification
        # -------------------------------------------------------------
        crop_rgb = cv2.cvtColor(damage_crop, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        crop_tensor = eff_transforms(crop_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            eff_logits = efficientnet_model(crop_tensor)
            eff_probabilities = torch.softmax(eff_logits, dim=1)[0]
            eff_class_id = int(eff_probabilities.argmax().item())
            eff_confidence = float(eff_probabilities[eff_class_id].item())
            eff_class_name = CLASS_NAMES[eff_class_id]

        total_hybrid_detections += 1

        # -------------------------------------------------------------
        # STEP D: Display the Dual Verdict
        # -------------------------------------------------------------
        print(f"Detection #{total_hybrid_detections}:")
        print(f"  Box Coordinates:           [{x1}, {y1}, {x2}, {y2}]")
        print(f"  Stage 1 (YOLO12 Detector):  {yolo_class_name:<6} (Confidence: {yolo_confidence:.4f})")
        print(f"  Stage 2 (EfficientNet-B0): {eff_class_name:<6} (Confidence: {eff_confidence:.4f})")
        print("-" * 75)

        # -------------------------------------------------------------
        # STEP E: Draw Dual-Model Annotations on Final Image
        # -------------------------------------------------------------
        # Bounding Box (Green)
        cv2.rectangle(annotated_output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Combined Label
        combined_label = f"YOLO: {yolo_class_name} ({yolo_confidence:.2f}) | EffNet: {eff_class_name} ({eff_confidence:.2f})"
        text_y = max(20, y1 - 8)

        # Label Background Tag for Readability
        (label_w, label_h), _ = cv2.getTextSize(combined_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated_output, (x1, text_y - label_h - 4), (x1 + label_w + 4, text_y + 2), (0, 0, 0), -1)
        cv2.putText(
            annotated_output,
            combined_label,
            (x1 + 2, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

if total_hybrid_detections == 0:
    print("No road damage detected above confidence threshold in hybrid pass.")

# --------------------------------------------------------------------------------------------------
# STEP F: Save the Combined Output Image
# --------------------------------------------------------------------------------------------------
output_filepath = OUTPUT_DIR / f"{TEST_IMAGE.stem}_combined_result.jpg"
cv2.imwrite(str(output_filepath), annotated_output)

print(f"\n[Success] Final Combined Result Image saved to:")
print(f"-> {output_filepath}")
print("\n" + "=" * 80)
print("Pipeline finished: Standalone YOLO12, Standalone EfficientNet-B0, and Combination Completed!")
print("=" * 80)
