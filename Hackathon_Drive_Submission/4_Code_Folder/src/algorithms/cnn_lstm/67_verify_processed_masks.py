
from pathlib import Path
import numpy as np

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
DATA_DIR = ROOT / "dataset" / "processed" / "cnn_lstm"

for split in ("train", "val", "test"):
    folder = DATA_DIR / split
    files = sorted(folder.glob("*.npz"))

    print(f"\n--- {split.upper()} ---")
    print("Videos:", len(files))

    if not files:
        print("No processed files found.")
        continue

    percentages = []

    for path in files:
        with np.load(path) as data:
            frames = data["frames"]
            masks = data["masks"]

            assert frames.shape[0] == masks.shape[0], (
                f"Sequence mismatch: {path.name}"
            )
            assert frames.shape[1:3] == masks.shape[1:3], (
                f"Spatial mismatch: {path.name}"
            )

            unique_values = np.unique(masks)
            foreground = 100 * np.mean(masks > 0)
            percentages.append(foreground)

            print(
                f"{path.name}: "
                f"frames={frames.shape[0]}, "
                f"size={frames.shape[1:3]}, "
                f"mask_values={unique_values.tolist()}, "
                f"foreground={foreground:.2f}%"
            )

    print(f"Mean foreground: {np.mean(percentages):.2f}%")
    print(f"Min foreground: {np.min(percentages):.2f}%")
    print(f"Max foreground: {np.max(percentages):.2f}%")
