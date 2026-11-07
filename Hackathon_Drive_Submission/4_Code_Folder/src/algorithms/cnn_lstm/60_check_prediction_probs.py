
from pathlib import Path
import importlib.util
import numpy as np
import torch

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())

DATA_FILE = ROOT / "dataset/processed/cnn_lstm/val/0061.npz"
CHECKPOINT = ROOT / "results/cnn_lstm/segmentation/best_cnn_lstm_segmentation.pt"
TRAIN_SCRIPT = ROOT / "src/algorithms/cnn_lstm/58_train_cnn_lstm_segmentation.py"

spec = importlib.util.spec_from_file_location("training", TRAIN_SCRIPT)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)

model = training.CNNLSTMSegmenter()
checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with np.load(DATA_FILE) as data:
    frames = data["frames"].copy()
    masks = data["masks"].copy()

x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
x = x.unsqueeze(0)

with torch.no_grad():
    probabilities = torch.sigmoid(model(x))[0, :, 0].numpy()

print("Checkpoint epoch:", checkpoint["epoch"])
print("Probability percentiles:")

for percentile in [0, 10, 25, 50, 75, 90, 95, 99, 100]:
    value = np.percentile(probabilities, percentile)
    print(f"  P{percentile}: {value:.6f}")

print(f"\nGround-truth foreground: {(masks > 0).mean() * 100:.2f}%")
print("\nForeground by threshold:")

for threshold in [0.40, 0.42, 0.44, 0.46, 0.48, 0.50]:
    ratio = (probabilities >= threshold).mean() * 100
    print(f"  {threshold:.2f}: {ratio:.2f}%")
