"""
Complete, Standalone YOLO12 Road Damage Detection Pipeline
Includes: Configuration -> Load -> Train -> Validate -> Inference -> Display
"""

from pathlib import Path
import os
import torch
from ultralytics import YOLO

# =====================================================================
# 1. CONFIGURATION
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Paths to data and weights
DATA_YAML = PROJECT_ROOT / "dataset" / "yolo" / "data.yaml"
MODEL_NAME = PROJECT_ROOT / "yolo12n.pt"

# Fallback test image from validation set
TEST_IMAGE = PROJECT_ROOT / "dataset" / "subsets" / "multicountry" / "RDD2022" / "split" / "val" / "images" / "India_0001.jpg"

EPOCHS = 10
IMAGE_SIZE = 640
BATCH_SIZE = 4
CONFIDENCE = 0.25
DEVICE = 0 if torch.cuda.is_available() else "cpu"

OUTPUT_PROJECT = PROJECT_ROOT / "results"
OUTPUT_NAME = "yolo12_training"

print(f"Device selected: {DEVICE}")
print(f"Data config: {DATA_YAML}")

# =====================================================================
# 2. LOAD YOLO12 MODEL
# =====================================================================

print("\nLoading YOLO12 base model...")
if not MODEL_NAME.exists():
    raise FileNotFoundError(f"Base weights not found: {MODEL_NAME}")

model = YOLO(str(MODEL_NAME))

# =====================================================================
# 3. TRAIN MODEL
# =====================================================================

print("\nStarting YOLO12 training...")

model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    workers=0 if DEVICE == "cpu" else 2,
    project=str(OUTPUT_PROJECT),
    name=OUTPUT_NAME,
    exist_ok=True,
    plots=True,
    save=True
)

# =====================================================================
# 4. LOAD BEST TRAINED MODEL
# =====================================================================

BEST_MODEL = OUTPUT_PROJECT / OUTPUT_NAME / "weights" / "best.pt"

# If newly trained weights are not yet generated, fall back to existing trained weights or base
if not BEST_MODEL.exists():
    candidate = PROJECT_ROOT / "results" / "training" / "yolo12n_30_epochs" / "weights" / "best.pt"
    BEST_MODEL = candidate if candidate.exists() else MODEL_NAME

print(f"\nLoading weights for evaluation: {BEST_MODEL}")
model = YOLO(str(BEST_MODEL))

# =====================================================================
# 5. MODEL VALIDATION
# =====================================================================

print("\nValidating YOLO12 model...")

metrics = model.val(
    data=str(DATA_YAML),
    imgsz=IMAGE_SIZE,
    device=DEVICE
)

print("\n========== MODEL PERFORMANCE ==========")
print("mAP50:    ", round(float(metrics.box.map50), 4))
print("mAP50-95: ", round(float(metrics.box.map), 4))
print("Precision:", round(float(metrics.box.mp), 4))
print("Recall:   ", round(float(metrics.box.mr), 4))

# =====================================================================
# 6. IMAGE DETECTION
# =====================================================================

if TEST_IMAGE.exists():
    print(f"\nRunning object detection on: {TEST_IMAGE.name}")

    results = model.predict(
        source=str(TEST_IMAGE),
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        save=True,
        project=str(OUTPUT_PROJECT),
        name="yolo12_predictions",
        exist_ok=True
    )

    # =================================================================
    # 7. DISPLAY DETECTION RESULTS
    # =================================================================

    for result in results:
        print("\n========== DETECTION RESULTS ==========")

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                coordinates = [round(v, 1) for v in box.xyxy[0].tolist()]

                class_name = model.names.get(class_id, f"Class {class_id}")

                print("Class:       ", class_name)
                print("Confidence:  ", round(confidence, 4))
                print("Bounding Box:", coordinates)
                print("--------------------------------")
        else:
            print("No damage detected above confidence threshold.")

    pred_dir = OUTPUT_PROJECT / "yolo12_predictions"
    print(f"\nAnnotated prediction images saved under: {pred_dir}")
else:
    print("\nTest image not found:", TEST_IMAGE)

print("\nYOLO12 execution completed successfully!")
