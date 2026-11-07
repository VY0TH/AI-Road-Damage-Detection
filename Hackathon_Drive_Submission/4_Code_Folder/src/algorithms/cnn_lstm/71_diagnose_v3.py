
from pathlib import Path
import numpy as np
import torch

from importlib.util import spec_from_file_location, module_from_spec

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
DATA_DIR = ROOT / "dataset" / "processed" / "cnn_lstm" / "val"
CHECKPOINT = (
    ROOT / "results" / "cnn_lstm" / "segmentation"
    / "best_cnn_lstm_spatial_v3.pt"
)
MODEL_SCRIPT = ROOT / "src" / "algorithms" / "cnn_lstm" / "58_train_cnn_lstm_segmentation.py"

DEVICE = torch.device("cpu")
torch.set_num_threads(4)


def load_model():
    spec = spec_from_file_location("training_module", MODEL_SCRIPT)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.CNNLSTMSegmenter().to(DEVICE)
    checkpoint = torch.load(
        CHECKPOINT, map_location=DEVICE, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT}")

    files = sorted(DATA_DIR.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No validation files in {DATA_DIR}")

    model = load_model()
    thresholds = [0.3, 0.5, 0.7, 0.9]

    print("\nV3 VALIDATION PREDICTION DIAGNOSTIC")
    print("-" * 78)
    print(
        f"{'Video':<18} {'GT damage %':>12} "
        + " ".join(f"{t:>10.2f}" for t in thresholds)
    )
    print("-" * 78)

    all_predictions = {t: [] for t in thresholds}
    all_targets = []

    for path in files:
        with np.load(path) as data:
            frames = data["frames"].copy()
            masks = data["masks"].copy()

        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        x = x.unsqueeze(0).to(DEVICE)

        y = masks.astype(np.float32)
        if y.max() > 1:
            y = y / 255.0
        y = y >= 0.5

        with torch.inference_mode():
            logits = model(x)
            probabilities = torch.sigmoid(logits)[0, :, 0].cpu().numpy()

        gt_pct = 100.0 * y.mean()
        coverages = []

        for threshold in thresholds:
            pred = probabilities >= threshold
            coverage = 100.0 * pred.mean()
            coverages.append(coverage)
            all_predictions[threshold].append(pred.reshape(-1))
        all_targets.append(y.reshape(-1))

        print(
            f"{path.name:<18} {gt_pct:>11.2f}% "
            + " ".join(f"{c:>9.2f}%" for c in coverages)
        )

    targets = np.concatenate(all_targets).astype(bool)

    print("-" * 78)
    print("OVERALL VALIDATION METRICS")
    print(f"Ground-truth damage: {100 * targets.mean():.2f}%")

    for threshold in thresholds:
        pred = np.concatenate(all_predictions[threshold]).astype(bool)
        tp = np.logical_and(pred, targets).sum()
        fp = np.logical_and(pred, ~targets).sum()
        fn = np.logical_and(~pred, targets).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        dice = 2 * tp / max(2 * tp + fp + fn, 1)

        print(
            f"Threshold {threshold:.2f} | "
            f"Predicted coverage {100 * pred.mean():.2f}% | "
            f"Precision {precision:.3f} | "
            f"Recall {recall:.3f} | Dice {dice:.3f}"
        )


if __name__ == "__main__":
    main()
