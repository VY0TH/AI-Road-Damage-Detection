
from pathlib import Path
import importlib.util
import numpy as np
import torch

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
DATA_DIR = ROOT / "dataset" / "processed" / "cnn_lstm" / "val"

MODELS = [
    {
        "name": "Spatial LSTM",
        "module": "58_train_cnn_lstm_segmentation.py",
        "checkpoint": ROOT / "results/cnn_lstm/segmentation/best_cnn_lstm_spatial.pt",
    },
    {
        "name": "Improved LSTM",
        "module": "63_train_cnn_lstm_improved.py",
        "checkpoint": ROOT / "results/cnn_lstm/improved/best_cnn_lstm_improved.pt",
    },
]

THRESHOLD = 0.5
DEVICE = torch.device("cpu")


def load_module(filename):
    path = ROOT / "src" / "algorithms" / "cnn_lstm" / filename
    spec = importlib.util.spec_from_file_location(
        path.stem, path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def evaluate(model, files):
    intersection = 0
    pred_total = 0
    gt_total = 0
    per_video_dice = []
    per_video_iou = []
    gt_pixels = 0
    pred_pixels = 0
    total_pixels = 0

    model.eval()

    with torch.no_grad():
        for file in files:
            data = np.load(file)

            frames = data["frames"].astype(np.float32)
            masks = data["masks"].astype(np.float32)

            # Frames: T,H,W,C -> 1,T,C,H,W
            x = torch.from_numpy(frames).permute(
                0, 3, 1, 2
            ).unsqueeze(0).to(DEVICE)

            # Masks: T,H,W -> 1,T,1,H,W
            y = torch.from_numpy(masks).unsqueeze(0).unsqueeze(2)
            y = (y > 0.5).float().to(DEVICE)

            # Match the usual image normalization
            if x.max() > 1:
                x = x / 255.0

            logits = model(x)

            # Support models returning B,T,1,H,W or B,T,H,W
            if logits.ndim == 4:
                logits = logits.unsqueeze(2)

            probs = torch.sigmoid(logits)
            pred = (probs >= THRESHOLD).float()

            inter = (pred * y).sum().item()
            p_sum = pred.sum().item()
            g_sum = y.sum().item()

            dice = (2 * inter + 1e-7) / (p_sum + g_sum + 1e-7)
            iou = (inter + 1e-7) / (p_sum + g_sum - inter + 1e-7)

            per_video_dice.append(dice)
            per_video_iou.append(iou)

            intersection += inter
            pred_total += p_sum
            gt_total += g_sum
            gt_pixels += g_sum
            pred_pixels += p_sum
            total_pixels += y.numel()

    global_dice = (
        2 * intersection + 1e-7
    ) / (pred_total + gt_total + 1e-7)

    global_iou = (
        intersection + 1e-7
    ) / (pred_total + gt_total - intersection + 1e-7)

    return {
        "videos": len(files),
        "mean_video_dice": np.mean(per_video_dice),
        "mean_video_iou": np.mean(per_video_iou),
        "global_dice": global_dice,
        "global_iou": global_iou,
        "gt_foreground_pct": 100 * gt_pixels / total_pixels,
        "pred_foreground_pct": 100 * pred_pixels / total_pixels,
    }


def main():
    files = sorted(DATA_DIR.glob("*.npz"))

    if not files:
        raise FileNotFoundError(f"No validation NPZ files found: {DATA_DIR}")

    print(f"Validation videos: {len(files)}")
    print(f"Threshold: {THRESHOLD}")

    for item in MODELS:
        print(f"\n--- {item['name']} ---")

        if not item["checkpoint"].exists():
            print(f"Checkpoint missing: {item['checkpoint']}")
            continue

        module = load_module(item["module"])
        model = module.CNNLSTMSegmenter().to(DEVICE)

        checkpoint = torch.load(
            item["checkpoint"],
            map_location=DEVICE,
            weights_only=False,
        )
        model.load_state_dict(get_state_dict(checkpoint))

        metrics = evaluate(model, files)

        for key, value in metrics.items():
            if isinstance(value, (float, np.floating)):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()