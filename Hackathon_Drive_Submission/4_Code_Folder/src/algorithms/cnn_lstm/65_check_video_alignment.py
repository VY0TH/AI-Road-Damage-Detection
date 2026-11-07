
from pathlib import Path
import cv2
import numpy as np

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())

DATASET = (
    ROOT / "dataset" / "videos" / "Pothole Videos"
    / "pothole_video"
)

SPLIT = "val"
VIDEO_NAME = "0061.mp4"  # Change if this file doesn't exist


def inspect(path):
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        print("Could not open:", path)
        return None

    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }

    cap.release()
    return info


rgb_path = DATASET / SPLIT / "rgb" / VIDEO_NAME
mask_path = DATASET / SPLIT / "mask" / VIDEO_NAME

print("RGB exists:", rgb_path.exists())
print("Mask exists:", mask_path.exists())

if rgb_path.exists() and mask_path.exists():
    rgb_info = inspect(rgb_path)
    mask_info = inspect(mask_path)

    print("\nRGB video:", rgb_info)
    print("Mask video:", mask_info)

    if rgb_info and mask_info:
        print("\nFrame counts match:",
              rgb_info["frames"] == mask_info["frames"])
        print("FPS matches:",
              abs(rgb_info["fps"] - mask_info["fps"]) < 0.01)
else:
    print("Change VIDEO_NAME to an existing matching RGB/mask filename.")
