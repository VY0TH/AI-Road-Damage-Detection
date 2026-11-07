
from pathlib import Path
import numpy as np

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
DATA = ROOT / "dataset" / "processed" / "cnn_lstm"

for split in ("train", "val", "test"):
    files = sorted((DATA / split).glob("*.npz"))
    ratios = []

    for file in files:
        with np.load(file) as data:
            masks = data["masks"]
            ratios.append(float((masks > 0).mean()))

    if ratios:
        print(f"\n{split.upper()} — {len(ratios)} videos")
        print(f"Empty masks: {sum(r == 0 for r in ratios)}")
        print(f"Mean foreground: {np.mean(ratios) * 100:.3f}%")
        print(f"Minimum foreground: {np.min(ratios) * 100:.3f}%")
        print(f"Maximum foreground: {np.max(ratios) * 100:.3f}%")
