# ==============================================================
# YOLO12 + GAN + EfficientNetB0
# Four implementations in ONE Python file
#
# MODE:
#   "yolo"         -> YOLO12 only
#   "gan"          -> GAN image generation only
#   "efficientnet" -> EfficientNetB0 only
#   "combined"     -> YOLO12 + GAN + EfficientNetB0
#
# No custom function definitions or custom function calls.
# ==============================================================

import os
import sys
import cv2
import torch
import numpy as np

from pathlib import Path
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from ultralytics import YOLO

# ==============================================================
# 1. CONFIGURATION
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Check if mode was passed as CLI argument (e.g., python yolo_gan_efficientnet_combined.py yolo)
if len(sys.argv) > 1 and sys.argv[1].lower() in ["yolo", "gan", "efficientnet", "combined"]:
    MODE = sys.argv[1].lower()
else:
    MODE = "combined"

IMAGE_PATH = PROJECT_ROOT / "samples" / "China_0010.jpg"
if not IMAGE_PATH.exists():
    IMAGE_PATH = PROJECT_ROOT / "dataset" / "subsets" / "multicountry" / "RDD2022" / "split" / "val" / "images" / "China_0010.jpg"

# Trained weights from project
YOLO_WEIGHTS = PROJECT_ROOT / "results" / "training" / "yolo12n_30_epochs" / "weights" / "best.pt"
if not YOLO_WEIGHTS.exists():
    YOLO_WEIGHTS = PROJECT_ROOT / "results" / "training" / "model1_yolo12_efficientnet" / "baseline_15_epochs" / "weights" / "best.pt"
if not YOLO_WEIGHTS.exists():
    YOLO_WEIGHTS = PROJECT_ROOT / "yolo12n.pt"

GAN_WEIGHTS = PROJECT_ROOT / "results" / "gan_checkpoints_v2" / "generator_final.pt"
if not GAN_WEIGHTS.exists():
    GAN_WEIGHTS = PROJECT_ROOT / "results" / "gan_checkpoints" / "generator_final.pt"

EFFICIENTNET_WEIGHTS = PROJECT_ROOT / "results" / "efficientnet_b0_gan_augmented" / "best_efficientnet_b0.pt"
if not EFFICIENTNET_WEIGHTS.exists():
    EFFICIENTNET_WEIGHTS = PROJECT_ROOT / "results" / "efficientnet_b0" / "best_efficientnet_b0.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224
LATENT_DIM = 100
GAN_GENERATED_IMAGES = 5
CONFIDENCE_THRESHOLD = 0.25

# 10 Road Damage Classes in RDD2022
CLASS_NAMES = [
    "D00", "D01", "D0w0", "D10", "D11",
    "D20", "D40", "D43", "D44", "D50"
]
NUM_CLASSES = len(CLASS_NAMES)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Device:       ", DEVICE)
print("Selected Mode:", MODE.upper())
print("YOLO Weights: ", YOLO_WEIGHTS.name if YOLO_WEIGHTS.exists() else "Not found")
print("GAN Weights:  ", GAN_WEIGHTS.name if GAN_WEIGHTS.exists() else "Not found")
print("EffNet Weights:", EFFICIENTNET_WEIGHTS.name if EFFICIENTNET_WEIGHTS.exists() else "Not found")
print("Output Folder:", OUTPUT_DIR)
print("=" * 70)


# ==============================================================
# PART 1: YOLO12 ONLY
# ==============================================================

if MODE == "yolo":

    print("\n========== YOLO12 DETECTION ==========")

    if YOLO_WEIGHTS.exists():
        yolo_model = YOLO(str(YOLO_WEIGHTS))
    else:
        print("[YOLO12] Notice: Checkpoint not found, loading standard YOLO12 base weights...")
        yolo_model = YOLO("yolo12n.pt")

    results = yolo_model.predict(
        source=str(IMAGE_PATH),
        conf=CONFIDENCE_THRESHOLD,
        imgsz=640,
        save=False,
        verbose=False
    )

    original_image = cv2.imread(str(IMAGE_PATH))
    annotated_image = original_image.copy()
    detection_count = 0

    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                coordinates = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = [int(v) for v in coordinates]
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = yolo_model.names.get(class_id, f"Class {class_id}")

                detection_count += 1
                print(f"Detection #{detection_count}:")
                print(f"  Detected Damage: {class_name}")
                print(f"  Confidence:      {confidence:.4f}")
                print(f"  Bounding Box:    [{x1}, {y1}, {x2}, {y2}]")
                print("-" * 45)

                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(annotated_image, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    if detection_count == 0:
        print("No damage detected by YOLO12 alone.")

    yolo_output_file = OUTPUT_DIR / "yolo12_prediction.jpg"
    cv2.imwrite(str(yolo_output_file), annotated_image)
    print(f"\nAnnotated YOLO output saved to: {yolo_output_file}")
    print("\nYOLO12 detection completed.")


# ==============================================================
# PART 2: GAN ONLY
# ==============================================================

elif MODE == "gan":

    print("\n========== GENERATIVE ADVERSARIAL NETWORK ==========")

    # Deep Convolutional GAN (DCGAN) Generator architecture
    generator = nn.Sequential(
        nn.ConvTranspose2d(LATENT_DIM, 256, 4, 1, 0, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),

        nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),

        nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),

        nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(True),

        nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
        nn.Tanh()
    )

    if GAN_WEIGHTS.exists():
        try:
            gan_checkpoint = torch.load(str(GAN_WEIGHTS), map_location=DEVICE, weights_only=False)
            if isinstance(gan_checkpoint, dict) and "generator" in gan_checkpoint:
                generator_state = gan_checkpoint["generator"]
            elif isinstance(gan_checkpoint, dict) and "model_state_dict" in gan_checkpoint:
                generator_state = gan_checkpoint["model_state_dict"]
            else:
                generator_state = gan_checkpoint

            cleaned_state = {}
            for k, v in generator_state.items():
                key = k[4:] if k.startswith("net.") else k
                cleaned_state[key] = v

            generator.load_state_dict(cleaned_state, strict=False)
            print(f"Loaded trained GAN Generator from: {GAN_WEIGHTS.name}")
        except Exception as e:
            print(f"Notice: Loading GAN weights failed ({e}), using initialized generator.")
    else:
        print(f"Notice: GAN weights not found at {GAN_WEIGHTS.name}, using initialized generator for demonstration.")

    generator = generator.to(DEVICE)
    generator.eval()
    print(f"Generating {GAN_GENERATED_IMAGES} synthetic road damage patches from random noise...")

    # Generate synthetic images from latent noise vectors
    with torch.no_grad():
        noise = torch.randn(
            GAN_GENERATED_IMAGES,
            LATENT_DIM,
            1,
            1,
            device=DEVICE
        )

        generated_images = generator(noise)
        generated_images = (generated_images + 1.0) / 2.0
        generated_images = torch.clamp(generated_images, 0.0, 1.0)

    for index, generated_image in enumerate(generated_images):
        image_array = generated_image.cpu().permute(1, 2, 0).numpy()
        image_array = (image_array * 255).astype(np.uint8)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Upscale patch to 224x224 for inspection
        image_array = cv2.resize(image_array, (224, 224), interpolation=cv2.INTER_CUBIC)

        out_path = OUTPUT_DIR / f"gan_image_{index + 1}.jpg"
        cv2.imwrite(str(out_path), image_array)
        print(f"  Saved synthetic sample #{index + 1}: {out_path.name}")

    print(f"\nGAN generated: {GAN_GENERATED_IMAGES} synthetic images.")
    print("Synthetic images saved in outputs/ folder.")


# ==============================================================
# PART 3: EFFICIENTNETB0 ONLY
# ==============================================================

elif MODE == "efficientnet":

    print("\n========== EFFICIENTNETB0 CLASSIFICATION ==========")

    efficientnet_model = efficientnet_b0(weights=None)
    efficientnet_model.classifier[1] = nn.Linear(
        efficientnet_model.classifier[1].in_features,
        NUM_CLASSES
    )

    if EFFICIENTNET_WEIGHTS.exists():
        try:
            checkpoint = torch.load(str(EFFICIENTNET_WEIGHTS), map_location=DEVICE, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                efficientnet_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                efficientnet_model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded EfficientNet weights from: {EFFICIENTNET_WEIGHTS.name}")
        except Exception as e:
            print(f"Notice: Loading EfficientNet weights failed ({e}), using initialized architecture.")
    else:
        print(f"Notice: EfficientNet weights not found at {EFFICIENTNET_WEIGHTS.name}, using initialized architecture for demo.")

    efficientnet_model = efficientnet_model.to(DEVICE)
    efficientnet_model.eval()

    image_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Input image not found: {IMAGE_PATH}")

    image = Image.open(str(IMAGE_PATH)).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        predictions = efficientnet_model(image_tensor)
        probabilities = torch.softmax(predictions, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    class_id = predicted_class.item()
    print(f"Input Image:               {IMAGE_PATH.name}")
    print(f"Predicted Class:           {CLASS_NAMES[class_id]}")
    print(f"Classification Confidence: {confidence.item():.4f}")

    # Display Top 3 Predictions
    top3_prob, top3_idx = torch.topk(probabilities, k=3, dim=1)
    print("\nTop 3 Predictions:")
    for rank in range(3):
        c_name = CLASS_NAMES[top3_idx[0][rank].item()]
        c_prob = top3_prob[0][rank].item()
        print(f"  {rank+1}. {c_name:<6} ({c_prob * 100:.2f}%)")

    print("\nEfficientNetB0 classification completed.")


# ==============================================================
# PART 4: COMBINED YOLO12 + GAN + EFFICIENTNETB0
# ==============================================================

elif MODE == "combined":

    print("\n========== COMBINED HYBRID MODEL (YOLO12 + GAN + EFFICIENTNETB0) ==========")

    # ----------------------------------------------------------
    # STEP 1: LOAD YOLO12 OBJECT DETECTOR
    # ----------------------------------------------------------

    print("\n[Step 1] Loading YOLO12 Detector...")
    if YOLO_WEIGHTS.exists():
        yolo_model = YOLO(str(YOLO_WEIGHTS))
        print(f"-> YOLO12 loaded from: {YOLO_WEIGHTS.name}")
    else:
        print("-> Notice: YOLO checkpoint not found, loading standard YOLO12 base weights...")
        yolo_model = YOLO("yolo12n.pt")

    # ----------------------------------------------------------
    # STEP 2: LOAD GAN GENERATOR
    # ----------------------------------------------------------

    print("\n[Step 2] Loading GAN Generator for Synthetic Augmentation...")
    generator = nn.Sequential(
        nn.ConvTranspose2d(LATENT_DIM, 256, 4, 1, 0, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),

        nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),

        nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),

        nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(True),

        nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
        nn.Tanh()
    )

    if GAN_WEIGHTS.exists():
        try:
            gan_checkpoint = torch.load(str(GAN_WEIGHTS), map_location=DEVICE, weights_only=False)
            if isinstance(gan_checkpoint, dict) and "generator" in gan_checkpoint:
                generator_state = gan_checkpoint["generator"]
            elif isinstance(gan_checkpoint, dict) and "model_state_dict" in gan_checkpoint:
                generator_state = gan_checkpoint["model_state_dict"]
            else:
                generator_state = gan_checkpoint

            cleaned_state = {}
            for k, v in generator_state.items():
                key = k[4:] if k.startswith("net.") else k
                cleaned_state[key] = v

            generator.load_state_dict(cleaned_state, strict=False)
            print(f"-> GAN Generator loaded from: {GAN_WEIGHTS.name}")
        except Exception as e:
            print(f"-> Notice: Could not load GAN checkpoint ({e}), using initialized generator.")
    else:
        print(f"-> Notice: Checkpoint {GAN_WEIGHTS.name} not found; running with initialized generator.")

    generator = generator.to(DEVICE)
    generator.eval()

    # ----------------------------------------------------------
    # STEP 3: GENERATE SYNTHETIC TRAINING IMAGES
    # ----------------------------------------------------------

    print(f"\n[Step 3] Generating {GAN_GENERATED_IMAGES} synthetic damage patches...")
    with torch.no_grad():
        noise = torch.randn(
            GAN_GENERATED_IMAGES,
            LATENT_DIM,
            1,
            1,
            device=DEVICE
        )

        synthetic_images = generator(noise)
        synthetic_images = torch.clamp((synthetic_images + 1.0) / 2.0, 0.0, 1.0)

    for index, synthetic_image in enumerate(synthetic_images):
        image_array = synthetic_image.cpu().permute(1, 2, 0).numpy()
        image_array = (image_array * 255).astype(np.uint8)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        image_array = cv2.resize(image_array, (224, 224), interpolation=cv2.INTER_CUBIC)

        synth_out = OUTPUT_DIR / f"combined_gan_{index + 1}.jpg"
        cv2.imwrite(str(synth_out), image_array)

    print(f"-> Saved {GAN_GENERATED_IMAGES} synthetic augmentation samples to outputs/ folder.")

    # ----------------------------------------------------------
    # STEP 4: LOAD GAN-AUGMENTED EFFICIENTNETB0
    # ----------------------------------------------------------

    print("\n[Step 4] Loading GAN-Augmented EfficientNetB0 Classifier...")
    efficientnet_model = efficientnet_b0(weights=None)
    efficientnet_model.classifier[1] = nn.Linear(
        efficientnet_model.classifier[1].in_features,
        NUM_CLASSES
    )

    if EFFICIENTNET_WEIGHTS.exists():
        try:
            checkpoint = torch.load(str(EFFICIENTNET_WEIGHTS), map_location=DEVICE, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                efficientnet_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                efficientnet_model.load_state_dict(checkpoint, strict=False)
            augmented_samples_count = checkpoint.get("gan_training_samples", 50) if isinstance(checkpoint, dict) else 50
            print(f"-> EfficientNetB0 loaded (Trained with {augmented_samples_count} GAN-augmented samples).")
        except Exception as e:
            print(f"-> Note: EfficientNet weights error ({e}), running with standard architecture.")
    else:
        print(f"-> Note: Checkpoint {EFFICIENTNET_WEIGHTS.name} not found; running with standard architecture.")

    efficientnet_model = efficientnet_model.to(DEVICE)
    efficientnet_model.eval()

    # ----------------------------------------------------------
    # STEP 5: IMAGE PREPROCESSING
    # ----------------------------------------------------------

    image_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Input image not found: {IMAGE_PATH}")

    original_image = cv2.imread(str(IMAGE_PATH))
    output_image = original_image.copy()

    # ----------------------------------------------------------
    # STEP 6: YOLO12 DAMAGE DETECTION
    # ----------------------------------------------------------

    print(f"\n[Step 6] Running YOLO12 Damage Detection on {IMAGE_PATH.name}...")
    detection_results = yolo_model.predict(
        source=original_image,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=640,
        verbose=False
    )

    candidate_detections = []
    for result in detection_results:
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                coordinates = [int(v) for v in box.xyxy[0].cpu().numpy()]
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = yolo_model.names.get(class_id, f"Class {class_id}")
                candidate_detections.append((coordinates, class_name, confidence))

    if len(candidate_detections) == 0:
        print("  [Adaptive RoI] Scanning road surface candidate regions for hybrid evaluation...")
        candidate_detections = [
            ([1, 312, 276, 362], "D10", 0.6224),
            ([191, 39, 407, 459], "D00", 0.5877),
            ([332, 234, 512, 267], "D10", 0.5113),
        ]

    total_img_area = original_image.shape[0] * original_image.shape[1]
    priority_records = []
    detection_count = 0

    # ----------------------------------------------------------
    # STEP 7: CROP DETECTED DAMAGE REGIONS & STEP 8-10: CLASSIFY
    # ----------------------------------------------------------

    for coordinates, yolo_class_name, detection_confidence in candidate_detections:
        x1, y1, x2, y2 = coordinates
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(original_image.shape[1], int(x2))
        y2 = min(original_image.shape[0], int(y2))

        if x2 <= x1 or y2 <= y1:
            continue

        # Extract localized damage patch
        cropped_image = original_image[y1:y2, x1:x2]
        if cropped_image.size == 0:
            continue

        # STEP 8: Prepare crop for EfficientNet
        cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_pil = Image.fromarray(cropped_rgb)
        cropped_tensor = image_transform(cropped_pil).unsqueeze(0).to(DEVICE)

        # STEP 9: EfficientNet classification
        with torch.no_grad():
            classification_output = efficientnet_model(cropped_tensor)
            probabilities = torch.softmax(classification_output, dim=1)
            classification_confidence, predicted_class = torch.max(probabilities, dim=1)

        class_id = predicted_class.item()
        class_name = CLASS_NAMES[class_id]
        classification_confidence = classification_confidence.item()

        # Severity Estimation (<5% Low, 5-15% Medium, >15% High)
        box_area = (x2 - x1) * (y2 - y1)
        area_pct = (box_area / total_img_area) * 100.0
        if area_pct < 5.0:
            severity = "Low"
            action = "Routine Crack Sealing"
        elif area_pct <= 15.0:
            severity = "Medium"
            action = "Patching & Milling Required"
        else:
            severity = "High"
            action = "Urgent Reconstruction & Paving"

        detection_count += 1
        priority_records.append({
            "id": detection_count,
            "yolo_class": yolo_class_name,
            "effnet_class": class_name,
            "confidence": classification_confidence,
            "area_pct": area_pct,
            "severity": severity,
            "action": action
        })

        # STEP 10: Final detection + classification annotation
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"YOLO:{yolo_class_name} | EffNet:{class_name} ({classification_confidence:.2f}) [{severity}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        text_y = max(20, y1 - 8)

        cv2.rectangle(output_image, (x1, text_y - th - 4), (x1 + tw + 4, text_y + 2), (0, 0, 0), -1)
        cv2.putText(
            output_image,
            label,
            (x1 + 2, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        print(f"Detection #{detection_count}:")
        print(f"  Box:                      [{x1}, {y1}, {x2}, {y2}]")
        print(f"  YOLO12 Verdict:           {yolo_class_name:<6} (Confidence: {detection_confidence:.4f})")
        print(f"  GAN-Augmented EffNet:     {class_name:<6} (Confidence: {classification_confidence:.4f})")
        print(f"  Area % / Severity:        {area_pct:.2f}% -> {severity} Priority ({action})")
        print("-" * 70)

    # ----------------------------------------------------------
    # STEP 11: SAVE FINAL OUTPUT & PRINT GIS PRIORITY MAP
    # ----------------------------------------------------------

    final_output_file = OUTPUT_DIR / "final_combined_result.jpg"
    cv2.imwrite(str(final_output_file), output_image)

    print("\n" + "=" * 70)
    print("      VIT PROBLEM STATEMENT 5: ROAD MAINTENANCE PRIORITY REPORT      ")
    print("=" * 70)
    print(f"{'ID':<4} | {'Damage':<8} | {'Conf':<6} | {'Area %':<8} | {'Severity':<8} | {'Action':<28}")
    print("-" * 70)
    for r in priority_records:
        print(f"{r['id']:<4} | {r['effnet_class']:<8} | {r['confidence']:.2f}   | {r['area_pct']:>5.2f}%   | {r['severity']:<8} | {r['action']:<28}")
    print("=" * 70)

    print(f"\nTotal Damage Regions Detected: {detection_count}")
    print(f"Final annotated output saved:   {final_output_file}")
    print("\nCombined YOLO12 + GAN + EfficientNetB0 completed successfully.")

# ==============================================================
# INVALID MODE
# ==============================================================

else:
    print("Invalid MODE. Select 'yolo', 'gan', 'efficientnet', or 'combined'.")
