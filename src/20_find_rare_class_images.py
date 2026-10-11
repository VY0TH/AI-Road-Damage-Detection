
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
DATASET = (
    ROOT / "dataset" / "subsets" / "multicountry"
    / "RDD2022" / "yolo"
)

CLASS_NAMES = [
    "D00", "D01", "D10", "D11", "D20",
    "D40", "D43", "D44", "D50", "D0w0"
]

rare_classes = {1, 3, 6, 9}  # D01, D11, D43, D0w0

print("=" * 60)
print("RARE-CLASS IMAGE REPORT")
print("=" * 60)

for split in ("train", "val"):
    label_dir = DATASET / "labels" / split
    images_by_class = defaultdict(list)

    if not label_dir.exists():
        print(f"Missing folder: {label_dir}")
        continue

    for label_file in sorted(label_dir.glob("*.txt")):
        found = set()

        for line in label_file.read_text(
            encoding="utf-8"
        ).splitlines():
            parts = line.split()

            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                continue

            if 0 <= class_id < len(CLASS_NAMES):
                found.add(class_id)

        for class_id in found:
            if class_id in rare_classes:
                images_by_class[class_id].append(label_file.stem)

    print(f"\n{split.upper()}")

    for class_id in sorted(rare_classes):
        files = images_by_class[class_id]
        print(f"\n{CLASS_NAMES[class_id]}: {len(files)} images")

        for filename in files:
            print("  ", filename)

print("\nReport complete. No files were modified.")
