
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

import cv2
import numpy as np
import torch

# =========================================================
# CONFIGURATION — FINAL TEST EVALUATION
# =========================================================

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())

DATA_DIR = ROOT / "dataset" / "processed" / "cnn_lstm" / "test"

CHECKPOINT = (
    ROOT / "results" / "cnn_lstm" / "segmentation"
    / "best_cnn_lstm_spatial_v2.pt"
)

OUT_DIR = (
    ROOT / "results" / "cnn_lstm" / "segmentation"
    / "test_evaluation"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
THRESHOLD = 0.5

torch.set_num_threads(4)


# =========================================================
# LOAD MODEL
# =========================================================

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

script = ROOT / "src" / "algorithms" / "cnn_lstm" / "58_train_cnn_lstm_segmentation.py"

spec = spec_from_file_location("training_module", script)
module = module_from_spec(spec)
spec.loader.exec_module(module)

model = module.CNNLSTMSegmenter().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("=" * 55)
print("FINAL TEST SET EVALUATION — SPATIAL LSTM V2")
print("=" * 55)
print("Checkpoint:", CHECKPOINT.name)
print("Threshold:", THRESHOLD)
print("Device:", DEVICE)


# =========================================================
# LOAD TEST FILES
# =========================================================

files = sorted(DATA_DIR.glob("*.npz"))

if not files:
    raise FileNotFoundError(f"No test files found in {DATA_DIR}")

print("Test videos:", len(files))

total_intersection = 0
total_prediction = 0
total_target = 0
total_correct = 0
total_pixels = 0

rows = []


# =========================================================
# EVALUATE
# =========================================================

with torch.no_grad():
    for file in files:
        with np.load(file) as data:
            frames = data["frames"].copy()
            masks = data["masks"].copy()

        x = (
            torch.from_numpy(frames)
            .permute(0, 3, 1, 2)
            .float() / 255.0
        ).unsqueeze(0).to(DEVICE)

        probabilities = torch.sigmoid(model(x))[0, :, 0].cpu().numpy()

        predictions = probabilities >= THRESHOLD
        targets = masks > 0

        intersection = int(
            np.logical_and(predictions, targets).sum()
        )
        pred_count = int(predictions.sum())
        target_count = int(targets.sum())

        union = pred_count + target_count - intersection

        dice = (
            2 * intersection / (pred_count + target_count)
            if pred_count + target_count else 1.0
        )

        iou = intersection / union if union else 1.0

        accuracy = float(np.mean(predictions == targets))

        total_intersection += intersection
        total_prediction += pred_count
        total_target += target_count
        total_correct += int((predictions == targets).sum())
        total_pixels += targets.size

        gt_fg = 100 * targets.mean()
        pred_fg = 100 * predictions.mean()

        rows.append({
            "name": file.name,
            "dice": dice,
            "iou": iou,
            "accuracy": accuracy,
            "gt_fg": gt_fg,
            "pred_fg": pred_fg,
        })

        # Save first-frame preview:
        # RGB | Ground truth | Prediction | Error overlay
        rgb = frames[0].copy()

        gt = targets[0].astype(np.uint8) * 255
        pred = predictions[0].astype(np.uint8) * 255

        overlay = rgb.copy()

        tp = predictions[0] & targets[0]
        fp = predictions[0] & ~targets[0]
        fn = ~predictions[0] & targets[0]

        overlay[tp] = [0, 255, 0]    # Correct damage
        overlay[fp] = [255, 0, 0]    # False positive
        overlay[fn] = [0, 0, 255]    # Missed damage

        panels = [
            rgb,
            cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB),
            overlay,
        ]

        preview = np.concatenate(panels, axis=1)

        output_file = OUT_DIR / f"{file.stem}_test_preview.png"

        success = cv2.imwrite(
            str(output_file),
            cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
        )

        if not success:
            print("WARNING: Could not save", output_file)

        print(
            f"{file.name}: Dice={dice:.4f}, "
            f"IoU={iou:.4f}, "
            f"GT FG={gt_fg:.2f}%, "
            f"Pred FG={pred_fg:.2f}%"
        )


# =========================================================
# OVERALL METRICS
# =========================================================

micro_dice = (
    2 * total_intersection / (total_prediction + total_target)
    if total_prediction + total_target else 1.0
)

global_union = total_prediction + total_target - total_intersection

micro_iou = (
    total_intersection / global_union
    if global_union else 1.0
)

print("\n" + "=" * 55)
print("FINAL TEST RESULTS")
print("=" * 55)

print("Test videos:", len(files))
print(f"Mean per-video Dice: {np.mean([r['dice'] for r in rows]):.4f}")
print(f"Mean per-video IoU:  {np.mean([r['iou'] for r in rows]):.4f}")
print(f"Global pixel Dice:   {micro_dice:.4f}")
print(f"Global pixel IoU:    {micro_iou:.4f}")
print(f"Pixel accuracy:      {total_correct / total_pixels:.4f}")
print(f"Ground-truth FG:     {100 * total_target / total_pixels:.2f}%")
print(f"Predicted FG:        {100 * total_prediction / total_pixels:.2f}%")
print("Previews saved to:", OUT_DIR)
