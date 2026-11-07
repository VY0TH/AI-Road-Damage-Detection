
from pathlib import Path
import importlib.util

import cv2
import numpy as np
import torch

# =========================================================
# PATHS
# =========================================================
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())

CHECKPOINT = (
    ROOT / "results" / "cnn_lstm" / "segmentation"
    / "best_cnn_lstm_spatial.pt"
)

DATA_FILE = (
    ROOT / "dataset" / "processed" / "cnn_lstm"
    / "val" / "0061.npz"
)

OUTPUT = (
    ROOT / "results" / "cnn_lstm" / "segmentation"
    / "prediction_preview.png"
)

THRESHOLD = 0.49
DEVICE = torch.device("cpu")


# =========================================================
# VERIFY FILES
# =========================================================
if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT}"
    )

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Validation file not found: {DATA_FILE}"
    )


# =========================================================
# LOAD THE CORRECT MODEL ARCHITECTURE
# =========================================================
training_script = (
    ROOT / "src" / "algorithms" / "cnn_lstm" / "58_train_cnn_lstm_segmentation.py"
)

spec = importlib.util.spec_from_file_location(
    "cnn_lstm_training", training_script
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

model = module.CNNLSTMSegmenter().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# =========================================================
# LOAD VALIDATION VIDEO
# =========================================================
with np.load(DATA_FILE) as data:
    frames = data["frames"].copy()
    masks = data["masks"].copy()

x = (
    torch.from_numpy(frames)
    .permute(0, 3, 1, 2)
    .float()
    / 255.0
)
x = x.unsqueeze(0).to(DEVICE)


# =========================================================
# PREDICT ALL FRAMES
# =========================================================
with torch.no_grad():
    logits = model(x)
    probabilities = torch.sigmoid(logits)[0, :, 0].cpu().numpy()

# Report foreground percentages for the whole sequence.
ground_truth_ratio = (masks > 0).mean() * 100
predicted_ratio = (probabilities >= THRESHOLD).mean() * 100

print("Checkpoint:", CHECKPOINT.name)
print("Checkpoint epoch:", checkpoint["epoch"])
print("Recorded validation Dice:", checkpoint["val_dice"])
print("Threshold:", THRESHOLD)
print(f"Ground-truth foreground: {ground_truth_ratio:.2f}%")
print(f"Predicted foreground: {predicted_ratio:.2f}%")


# =========================================================
# CREATE PREVIEW
# First frame | Ground truth | Prediction
# =========================================================
frame_index = 4

rgb = frames[frame_index].astype(np.uint8)

ground_truth = (
    (masks[frame_index] > 0).astype(np.uint8) * 255
)

prediction = (
    (probabilities[frame_index] >= THRESHOLD)
    .astype(np.uint8) * 255
)

ground_truth_rgb = cv2.cvtColor(
    ground_truth, cv2.COLOR_GRAY2RGB
)

prediction_rgb = cv2.cvtColor(
    prediction, cv2.COLOR_GRAY2RGB
)

# Add labels to make the output easy to interpret.
def add_label(image, label):
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(
        image, label, (5, 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (255, 255, 255), 1, cv2.LINE_AA
    )
    return image

rgb = add_label(rgb, "ROAD FRAME")
ground_truth_rgb = add_label(
    ground_truth_rgb, "GROUND TRUTH"
)
prediction_rgb = add_label(
    prediction_rgb, "PREDICTION"
)

preview = np.concatenate(
    [rgb, ground_truth_rgb, prediction_rgb],
    axis=1,
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

success = cv2.imwrite(
    str(OUTPUT),
    cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
)

if not success:
    raise RuntimeError(f"Could not save preview: {OUTPUT}")

print("Preview saved:", OUTPUT)
