
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from importlib.util import spec_from_file_location, module_from_spec

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
CHECKPOINT = ROOT / "results/cnn_lstm/segmentation/best_cnn_lstm_spatial.pt"
DATA_DIR = ROOT / "dataset/processed/cnn_lstm/val"

torch.set_num_threads(4)
device = torch.device("cpu")

# Load the same model architecture used for training.
script = ROOT / "src/algorithms/cnn_lstm/58_train_cnn_lstm_segmentation.py"
spec = spec_from_file_location("training_module", script)
module = module_from_spec(spec)
spec.loader.exec_module(module)

model = module.CNNLSTMSegmenter().to(device)
checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

files = sorted(DATA_DIR.glob("*.npz"))
if not files:
    raise FileNotFoundError(f"No validation files found in {DATA_DIR}")

thresholds = np.arange(0.30, 0.501, 0.01)
results = {round(float(t), 2): [] for t in thresholds}

with torch.no_grad():
    for file in files:
        with np.load(file) as data:
            frames = data["frames"].copy()
            masks = data["masks"].copy()

        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        x = x.unsqueeze(0).to(device)

        probabilities = torch.sigmoid(model(x))[0, :, 0].numpy()
        targets = masks > 0

        for threshold in results:
            predictions = probabilities >= threshold
            intersection = np.logical_and(predictions, targets).sum()
            denominator = predictions.sum() + targets.sum()

            dice = (
                2.0 * intersection / denominator
                if denominator > 0 else 1.0
            )
            results[threshold].append(float(dice))

print(f"Evaluated {len(files)} validation videos")
print("\nThreshold | Mean Dice")
print("---------------------")

for threshold, scores in results.items():
    print(f"   {threshold:.2f}    | {np.mean(scores):.4f}")

best = max(results, key=lambda t: np.mean(results[t]))
print(f"\nBest validation threshold: {best:.2f}")
print(f"Mean validation Dice: {np.mean(results[best]):.4f}")
print("Use this threshold for validation visualization only.")
