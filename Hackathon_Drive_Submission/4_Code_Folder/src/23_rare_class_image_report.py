
from pathlib import Path
from collections import defaultdict

ROOT = Path("dataset/subsets/multicountry/RDD2022/yolo")
RARE_CLASSES = {
    1: "D01",
    3: "D11",
    6: "D43",
    9: "D0w0",
}

for split in ("train", "val"):
    labels_dir = ROOT / "labels" / split
    print(f"\n{'=' * 45}")
    print(f"{split.upper()} RARE-CLASS IMAGE REPORT")
    print("=" * 45)

    if not labels_dir.exists():
        print("Missing folder:", labels_dir.resolve())
        continue

    images = defaultdict(list)

    for label_file in labels_dir.glob("*.txt"):
        try:
            lines = label_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                continue

            if class_id in RARE_CLASSES:
                images[class_id].append(label_file.stem)
    
    for class_id, name in RARE_CLASSES.items():
        unique_images = sorted(set(images[class_id]))
        print(f"\n{name}: {len(unique_images)} images")
        for stem in unique_images:
            print(" ", stem)

print("\nReport complete. No files modified.")
