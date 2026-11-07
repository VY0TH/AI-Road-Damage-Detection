
from pathlib import Path
import cv2
import numpy as np

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())

MASK = (
    ROOT / "dataset" / "videos" / "Pothole Videos"
    / "pothole_video" / "val" / "mask" / "0061.mp4"
)

cap = cv2.VideoCapture(str(MASK))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open mask: {MASK}")

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
indices = np.linspace(0, total - 1, 4, dtype=int)

for index in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()

    if not ok:
        print("Could not read frame", index)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    values, counts = np.unique(gray, return_counts=True)

    print(f"\nFrame {index}")
    print("Most common pixel values:",
          sorted(zip(counts.tolist(), values.tolist()), reverse=True)[:10])
    print("Minimum:", gray.min(), "Maximum:", gray.max())
    print("Nonzero percentage:", 100 * np.mean(gray > 0))

cap.release()
