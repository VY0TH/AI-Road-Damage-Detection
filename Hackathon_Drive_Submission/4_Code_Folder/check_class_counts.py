
from collections import Counter
from pathlib import Path
import yaml

yaml_path = Path("dataset/subsets/multicountry/RDD2022/yolo_grouped/data.yaml")
config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
root = yaml_path.parent
names = config["names"]

print("Classes:", names)

for split in ("train", "val"):
    counts = Counter()
    image_count = 0
    empty_labels = 0
    missing_labels = 0

    source = config[split]
    entries = source if isinstance(source, list) else [source]

    for entry in entries:
        path = Path(entry)
        if not path.is_absolute():
            path = root / path

        if path.is_file():
            images = [path]
        elif path.is_dir():
            images = [
                p for p in path.rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ]
        else:
            print(f"WARNING: path not found: {path}")
            continue

        for image_path in images:
            image_count += 1

            # Standard YOLO layout: images/... maps to labels/...
            label_path = Path(str(image_path).replace("images", "labels"))
            label_path = label_path.with_suffix(".txt")

            if not label_path.exists():
                missing_labels += 1
                continue

            lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            if not lines:
                empty_labels += 1
                continue

            for line in lines:
                parts = line.split()
                try:
                    class_id = int(parts[0])
                except (ValueError, IndexError):
                    continue

                if 0 <= class_id < len(names):
                    counts[class_id] += 1

    print(f"\n{split.upper()}: images={image_count}, "
          f"empty_labels={empty_labels}, missing_labels={missing_labels}")

    for class_id, name in enumerate(names):
        print(f"  {class_id}: {name}: {counts[class_id]} objects")
