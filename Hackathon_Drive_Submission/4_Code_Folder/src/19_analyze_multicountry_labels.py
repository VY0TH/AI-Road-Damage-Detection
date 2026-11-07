
from pathlib import Path
from collections import Counter
import yaml

# Project root
ROOT = Path(__file__).resolve().parent.parent

# Your existing multi-country YOLO dataset
DATASET = (
    ROOT / "dataset" / "subsets" / "multicountry"
    / "RDD2022" / "yolo"
)

SPLITS = ["train", "val"]
CLASS_NAMES = [
    "D00", "D01", "D10", "D11", "D20",
    "D40", "D43", "D44", "D50", "D0w0"
]

print("=" * 65)
print("MULTI-COUNTRY YOLO LABEL ANALYSIS")
print("=" * 65)
print("Dataset:", DATASET)

if not DATASET.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET}"
    )

for split in SPLITS:
    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    if not image_dir.exists() or not label_dir.exists():
        print(f"\n{split.upper()}: folders missing")
        continue

    images = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]

    class_counts = Counter()
    images_with_damage = 0
    images_without_damage = 0
    invalid_lines = 0
    missing_labels = 0
    total_boxes = 0

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            missing_labels += 1
            continue

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        valid_boxes_in_image = 0

        for line in lines:
            parts = line.split()

            if len(parts) != 5:
                invalid_lines += 1
                continue

            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                invalid_lines += 1
                continue

            # Check valid class and normalized box coordinates
            if not (0 <= class_id < len(CLASS_NAMES)):
                invalid_lines += 1
                continue

            if not all(0 <= v <= 1 for v in (x, y, w, h)):
                invalid_lines += 1
                continue

            if w <= 0 or h <= 0:
                invalid_lines += 1
                continue

            class_counts[class_id] += 1
            total_boxes += 1
            valid_boxes_in_image += 1

        if valid_boxes_in_image:
            images_with_damage += 1
        else:
            images_without_damage += 1

    print(f"\n{'-' * 50}")
    print(f"SPLIT: {split.upper()}")
    print(f"{'-' * 50}")
    print("Images:", len(images))
    print("Images with damage boxes:", images_with_damage)
    print("Images without damage boxes:", images_without_damage)
    print("Missing label files:", missing_labels)
    print("Invalid label lines:", invalid_lines)
    print("Total valid boxes:", total_boxes)
    print("\nClass distribution (bounding boxes):")

    for class_id, name in enumerate(CLASS_NAMES):
        print(f"{name:5s}: {class_counts[class_id]}")

print("\nAnalysis complete. No files were modified.")
