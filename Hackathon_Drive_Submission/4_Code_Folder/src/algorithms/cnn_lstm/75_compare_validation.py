
from pathlib import Path
import importlib.util

import numpy as np
import cv2
import torch

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())

MODEL_SCRIPT = ROOT / "src" / "algorithms" / "cnn_lstm" / "58_train_cnn_lstm_segmentation.py"
DATA_DIR = ROOT / "dataset" / "processed" / "cnn_lstm" / "val"
CHECKPOINT = (
    ROOT / "results" / "cnn_lstm" / "segmentation"
    / "best_cnn_lstm_spatial_v3.pt"
)
OUTPUT_DIR = ROOT / "results" / "cnn_lstm" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
THRESHOLD = 0.70


def main():
    spec = importlib.util.spec_from_file_location(
        "training_module", MODEL_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.CNNLSTMSegmenter().to(DEVICE)
    checkpoint = torch.load(
        CHECKPOINT, map_location=DEVICE, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    files = sorted(DATA_DIR.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No validation samples in {DATA_DIR}")

    print("Validation comparison")
    print("-" * 65)

    results = []

    for path in files:
        with np.load(path) as data:
            frames = data["frames"].astype(np.float32) / 255.0
            gt = data["masks"]

        x = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0)

        with torch.inference_mode():
            logits = model(x.to(DEVICE))
            probabilities = torch.sigmoid(logits)[0, :, 0].cpu().numpy()

        pred = probabilities >= THRESHOLD
        target = gt > 0
        print("Ground-truth mask values:", np.unique(gt))

        intersection = np.logical_and(pred, target).sum()
        union = np.logical_or(pred, target).sum()
        dice_denominator = pred.sum() + target.sum()

        dice = (
            2 * intersection / dice_denominator
            if dice_denominator else 1.0
        )
        iou = intersection / union if union else 1.0

        pred_pct = pred.mean() * 100
        gt_pct = target.mean() * 100

        results.append((path.name, dice, iou, pred_pct, gt_pct))

        print(
            f"{path.name}: Dice={dice:.4f}, IoU={iou:.4f}, "
            f"Predicted={pred_pct:.2f}%, Ground truth={gt_pct:.2f}%"
        )

        # Save a visual comparison for the first validation sample.
        if path == files[0]:
            frame = cv2.cvtColor(frames[4], cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (384, 384))

            prediction = cv2.resize(
                pred[4].astype(np.uint8) * 255,
                (384, 384),
                interpolation=cv2.INTER_NEAREST,
            )
            ground_truth = cv2.resize(
                target[4].astype(np.uint8) * 255,
                (384, 384),
                interpolation=cv2.INTER_NEAREST,
            )

            cv2.imwrite(str(OUTPUT_DIR / "validation_original.jpg"), frame)
            cv2.imwrite(str(OUTPUT_DIR / "validation_prediction.jpg"), prediction)
            cv2.imwrite(str(OUTPUT_DIR / "validation_ground_truth.jpg"), ground_truth)

    print("-" * 65)
    print(f"Mean Dice: {np.mean([r[1] for r in results]):.4f}")
    print(f"Mean IoU:  {np.mean([r[2] for r in results]):.4f}")
    print(f"Mean predicted coverage: {np.mean([r[3] for r in results]):.2f}%")
    print(f"Mean ground-truth coverage: {np.mean([r[4] for r in results]):.2f}%")
    print(f"\nImages saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
