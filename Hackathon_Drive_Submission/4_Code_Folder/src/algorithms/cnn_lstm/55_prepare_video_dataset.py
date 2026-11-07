
from pathlib import Path
import cv2
import csv
import numpy as np

# Project paths
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())

DATASET = (
    ROOT / "dataset" / "videos" / "Pothole Videos"
    / "pothole_video"
)

OUTPUT = ROOT / "dataset" / "processed" / "cnn_lstm"

# CPU-friendly settings
IMAGE_SIZE = 128
SEQUENCE_LENGTH = 8

# Start small; preserve official train/val/test splits.
MAX_VIDEOS = {
    "train": 40,
    "val": 10,
    "test": 10,
}


def read_sampled_frames(video_path, count, grayscale=False):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return None

    indices = np.linspace(
        0, total - 1, count, dtype=int
    )

    frames = []

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()

        if not ok:
            cap.release()
            return None

        if grayscale:
            frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2GRAY
            )
            frame = cv2.resize(
                frame, (IMAGE_SIZE, IMAGE_SIZE)
            )
            frame = (frame > 0).astype(np.uint8)
        else:
            frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )
            frame = cv2.resize(
                frame, (IMAGE_SIZE, IMAGE_SIZE)
            )

        frames.append(frame)

    cap.release()
    return np.stack(frames)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT / "manifest.csv"

    rows = []

    for split, limit in MAX_VIDEOS.items():
        rgb_dir = DATASET / split / "rgb"
        mask_dir = DATASET / split / "mask"

        out_dir = OUTPUT / split
        out_dir.mkdir(parents=True, exist_ok=True)

        video_paths = sorted(rgb_dir.glob("*.mp4"))[:limit]

        print(f"\n{split}: preparing {len(video_paths)} videos")

        for video_path in video_paths:
            mask_path = mask_dir / video_path.name

            if not mask_path.exists():
                print(f"Missing mask: {video_path.name}")
                continue

            frames = read_sampled_frames(
                video_path, SEQUENCE_LENGTH
            )

            masks = read_sampled_frames(
                mask_path, SEQUENCE_LENGTH, grayscale=True
            )

            if frames is None or masks is None:
                print(f"Skipping unreadable pair: {video_path.name}")
                continue

            # Binary sequence label:
            # 1 = at least one sampled mask frame contains
            # a nonzero pixel; 0 = no nonzero mask pixels.
            # Verify mask encoding before relying on this label.
            label = int(np.any(masks > 0))

            output_file = out_dir / (
                video_path.stem + ".npz"
            )

            np.savez_compressed(
                output_file,
                frames=frames.astype(np.uint8),
                masks=masks.astype(np.uint8),
                label=np.int64(label),
            )

            rows.append({
                "split": split,
                "video": video_path.name,
                "processed_file": str(output_file.relative_to(ROOT)),
                "label": label,
            })

            print(
                f"  {video_path.name}: saved, "
                f"label={label}"
            )

    with open(
        manifest_path, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split", "video", "processed_file", "label"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nPreprocessing complete.")
    print(f"Successfully processed: {len(rows)} videos")
    print(f"Output folder: {OUTPUT}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
