
from pathlib import Path
from collections import Counter

ROOT = Path("dataset/subsets/multicountry/RDD2022/yolo")
SPLITS = ["train", "val"]

CLASS_NAMES = [
    "D00", "D01", "D10", "D11", "D20",
    "D40", "D43", "D44", "D50", "D0w0"
]

for split in SPLITS:
    labels_dir = ROOT / "labels" / split

    if not labels_dir.exists():
        print(f"Missing label directory: {labels_dir.resolve()}")
        continue

    counts = Counter()
    images_with_objects = 0
    empty_labels = 0

    for label_file in labels_dir.glob("*.txt"):
        lines = [
            line.strip()
            for line in label_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if not lines:
            empty_labels += 1
            continue

        images_with_objects += 1

        for line in lines:
            parts = line.split()
            if len(parts) == 5:
                try:
                    class_id = int(parts[0])
                    if 0 <= class_id < len(CLASS_NAMES):
                        counts[CLASS_NAMES[class_id]] += 1
                except ValueError:
                    pass

    print(f"\n{'=' * 35}")
    print(f"SPLIT: {split.upper()}")
    print(f"Images with objects: {images_with_objects}")
    print(f"Empty label files: {empty_labels}")
    print("Bounding-box counts:")

    for name in CLASS_NAMES:
        print(f"{name:5s}: {counts[name]}")

print("\nAnalysis complete. No files were modified.")
