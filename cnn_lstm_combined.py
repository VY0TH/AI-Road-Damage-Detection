# ============================================================
# CNN + LSTM
# Three Independent Implementations in One Python File
# No Custom Function Definitions or Calls
# ============================================================

import os
import sys
import cv2
import torch
import numpy as np

from pathlib import Path
from PIL import Image
from torch import nn
from torch.optim import Adam
from torchvision import transforms, models

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Check if mode was passed as CLI argument (e.g., python cnn_lstm_combined.py cnn)
if len(sys.argv) > 1 and sys.argv[1].lower() in ["cnn", "lstm", "combined"]:
    MODE = sys.argv[1].lower()
else:
    MODE = "combined"

# Options:
# "cnn"      -> CNN only
# "lstm"     -> LSTM only
# "combined" -> CNN + LSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_PATH = PROJECT_ROOT / "samples" / "China_0010.jpg"
if not IMAGE_PATH.exists():
    IMAGE_PATH = PROJECT_ROOT / "dataset" / "subsets" / "multicountry" / "RDD2022" / "split" / "val" / "images" / "China_0010.jpg"
VIDEO_PATH = PROJECT_ROOT / "results" / "pothole_debug" / "dashboard_smoke_20261014_172243" / "0081_segmentation_smoke.mp4"

# Trained weights from project
COMBINED_WEIGHTS = PROJECT_ROOT / "results" / "cnn_lstm" / "best_cnn_lstm.pt"

# Road damage classes from project
CLASS_NAMES = [
    "D00", "D01", "D10", "D11", "D20",
    "D40", "D43", "D44", "D50", "D0w0",
    "No_Damage"
]
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 128
SEQUENCE_LENGTH = 5

print("=" * 60)
print(f"Executing Mode: {MODE.upper()}")
print("Device:        ", DEVICE)
print("Weights File:  ", COMBINED_WEIGHTS)
print("=" * 60)


# ============================================================
# PART 1: CNN MODEL (SEPARATE)
# ============================================================

if MODE == "cnn":

    print("\n========== CNN CLASSIFICATION (SEPARATE) ==========")

    # Define the 4-stage convolutional spatial feature extractor
    cnn_model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )

    cnn_classifier = nn.Linear(128, NUM_CLASSES)

    # Load trained weights
    if not COMBINED_WEIGHTS.exists():
        raise FileNotFoundError(f"Model weights not found: {COMBINED_WEIGHTS}")

    checkpoint = torch.load(str(COMBINED_WEIGHTS), map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    # Extract CNN layer weights
    cnn_state = {}
    for k, v in state_dict.items():
        if k.startswith("cnn."):
            cnn_state[k[4:]] = v
    cnn_model.load_state_dict(cnn_state)

    cnn_model = cnn_model.to(DEVICE)
    cnn_classifier = cnn_classifier.to(DEVICE)
    cnn_model.eval()
    cnn_classifier.eval()

    image_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    image = Image.open(str(IMAGE_PATH)).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = cnn_model(image_tensor)
        output = cnn_classifier(features)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    predicted_class = CLASS_NAMES[predicted.item()]
    print(f"Input Image:         {IMAGE_PATH.name}")
    print(f"CNN Feature Vector:  {list(features.shape)}")
    print(f"CNN Predicted Class: {predicted_class}")
    print(f"Confidence:          {confidence.item():.4f}")


# ============================================================
# PART 2: LSTM MODEL (SEPARATE)
# ============================================================

elif MODE == "lstm":

    print("\n========== LSTM CLASSIFICATION (SEPARATE) ==========")

    INPUT_SIZE = 128
    HIDDEN_SIZE = 64
    NUM_LAYERS = 1

    lstm_model = nn.LSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        batch_first=True
    )

    lstm_classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
    )

    # Load trained weights
    if not COMBINED_WEIGHTS.exists():
        raise FileNotFoundError(f"Model weights not found: {COMBINED_WEIGHTS}")

    checkpoint = torch.load(str(COMBINED_WEIGHTS), map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    lstm_state = {}
    classifier_state = {}
    for k, v in state_dict.items():
        if k.startswith("lstm."):
            lstm_state[k[5:]] = v
        elif k.startswith("classifier."):
            classifier_state[k[11:]] = v

    lstm_model.load_state_dict(lstm_state)
    lstm_classifier.load_state_dict(classifier_state)

    lstm_model = lstm_model.to(DEVICE)
    lstm_classifier = lstm_classifier.to(DEVICE)
    lstm_model.eval()
    lstm_classifier.eval()

    # Sequential input representing consecutive road observations
    sequence = torch.randn(
        1,
        SEQUENCE_LENGTH,
        INPUT_SIZE
    ).to(DEVICE)

    with torch.no_grad():
        lstm_output, (hidden, cell) = lstm_model(sequence)
        final_output = lstm_classifier(lstm_output[:, -1, :])
        probabilities = torch.softmax(final_output, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    predicted_class = CLASS_NAMES[predicted.item()]
    print(f"Input Sequence Shape: {list(sequence.shape)} (batch, seq_len, features)")
    print(f"LSTM Hidden State:    {list(hidden.shape)}")
    print(f"LSTM Predicted Class: {predicted_class}")
    print(f"Confidence:           {confidence.item():.4f}")


# ============================================================
# PART 3: CNN + LSTM COMBINED
# ============================================================

elif MODE == "combined":

    print("\n========== CNN + LSTM COMBINED MODEL ==========")

    # --------------------------------------------------------
    # STEP 1: CNN FEATURE EXTRACTOR
    # --------------------------------------------------------

    cnn_model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )

    # --------------------------------------------------------
    # STEP 2: LSTM NETWORK
    # --------------------------------------------------------

    cnn_feature_size = 128
    hidden_size = 64
    num_layers = 1

    lstm_model = nn.LSTM(
        input_size=cnn_feature_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True
    )

    # --------------------------------------------------------
    # STEP 3: FINAL CLASSIFIER
    # --------------------------------------------------------

    classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(hidden_size, NUM_CLASSES)
    )

    # --------------------------------------------------------
    # STEP 4: LOAD TRAINED WEIGHTS
    # --------------------------------------------------------

    if COMBINED_WEIGHTS.exists():
        try:
            checkpoint = torch.load(str(COMBINED_WEIGHTS), map_location=DEVICE, weights_only=False)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

            cnn_state = {}
            lstm_state = {}
            classifier_state = {}

            for k, v in state_dict.items():
                if k.startswith("cnn."):
                    cnn_state[k[4:]] = v
                elif k.startswith("lstm."):
                    lstm_state[k[5:]] = v
                elif k.startswith("classifier."):
                    classifier_state[k[11:]] = v

            cnn_model.load_state_dict(cnn_state, strict=False)
            lstm_model.load_state_dict(lstm_state, strict=False)
            classifier.load_state_dict(classifier_state, strict=False)
            print(f"Loaded trained weights from: {COMBINED_WEIGHTS.name}")
        except Exception as e:
            print(f"Note: Running with architecture initialized weights ({e})")
    else:
        print(f"Note: Checkpoint {COMBINED_WEIGHTS.name} not found; running with model architecture initialized weights.")

    cnn_model = cnn_model.to(DEVICE)
    lstm_model = lstm_model.to(DEVICE)
    classifier = classifier.to(DEVICE)

    cnn_model.eval()
    lstm_model.eval()
    classifier.eval()

    # --------------------------------------------------------
    # STEP 5: IMAGE PREPROCESSING
    # --------------------------------------------------------

    image_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # --------------------------------------------------------
    # STEP 6: READ VIDEO OR CONSECUTIVE ROAD FRAMES
    # --------------------------------------------------------

    feature_sequence = []

    # If video file exists, extract frames from video
    if VIDEO_PATH.exists():
        print(f"Reading sequence from video: {VIDEO_PATH.name}")
        video = cv2.VideoCapture(str(VIDEO_PATH))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames > 0:
            frame_indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
            for frame_index in frame_indices:
                video.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                success, frame = video.read()
                if not success:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                image_tensor = image_transform(pil_image).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    features = cnn_model(image_tensor)

                feature_sequence.append(features.squeeze(0))
        video.release()

    # Fallback to consecutive road validation images if needed
    if len(feature_sequence) != SEQUENCE_LENGTH:
        val_img_dir = PROJECT_ROOT / "dataset" / "subsets" / "multicountry" / "RDD2022" / "split" / "val" / "images"
        sorted_images = sorted(list(val_img_dir.glob("*.jpg")))[:SEQUENCE_LENGTH] if val_img_dir.exists() else []
        if not sorted_images and IMAGE_PATH.is_file():
            sorted_images = [IMAGE_PATH] * SEQUENCE_LENGTH
        print(f"Using {len(sorted_images)} road images as temporal sequence:")
        feature_sequence = []
        for img_p in sorted_images:
            print(f"  Frame: {img_p.name}")
            pil_img = Image.open(str(img_p)).convert("RGB")
            img_tensor = image_transform(pil_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                features = cnn_model(img_tensor)
            feature_sequence.append(features.squeeze(0))

    # --------------------------------------------------------
    # STEP 7: CNN FEATURES TO LSTM
    # --------------------------------------------------------

    if len(feature_sequence) != SEQUENCE_LENGTH:
        raise RuntimeError("Could not extract the required number of frames.")

    sequence_tensor = torch.stack(feature_sequence).unsqueeze(0).to(DEVICE)
    print(f"\nStacked Temporal Sequence Tensor: {list(sequence_tensor.shape)} (batch, seq_len, 128)")

    # --------------------------------------------------------
    # STEP 8: LSTM TEMPORAL LEARNING
    # --------------------------------------------------------

    with torch.no_grad():
        lstm_output, (hidden, cell) = lstm_model(sequence_tensor)
        final_features = lstm_output[:, -1, :]

        # ----------------------------------------------------
        # STEP 9: FINAL CLASSIFICATION
        # ----------------------------------------------------

        output = classifier(final_features)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    predicted_class = CLASS_NAMES[predicted.item()]

    print("\n" + "=" * 50)
    print("           CNN + LSTM COMBINED PREDICTION           ")
    print("=" * 50)
    print(f"Predicted Damage Class: {predicted_class}")
    print(f"Prediction Confidence:  {confidence.item():.4f}")
    print("=" * 50)

    # Print distribution across top 3 categories
    top3_prob, top3_idx = torch.topk(probabilities, k=3, dim=1)
    print("\nTop 3 Predictions:")
    for rank in range(3):
        cls_name = CLASS_NAMES[top3_idx[0][rank].item()]
        prob_val = top3_prob[0][rank].item()
        print(f"  {rank+1}. {cls_name:<10} ({prob_val * 100:.2f}%)")

    print("\nCombined CNN + LSTM processing completed successfully.")

else:
    print("Invalid MODE. Choose 'cnn', 'lstm', or 'combined'.")
