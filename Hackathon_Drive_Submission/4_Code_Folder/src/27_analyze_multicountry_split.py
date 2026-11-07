
from pathlib import Path
from collections import Counter
import yaml

CONFIG = Path("dataset/subsets/multicountry/RDD2022/yolo/data.yaml")

def main():
    if not CONFIG.exists():
        print(f"ERROR: Configuration not found: {CONFIG}")
        return

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    root = (CONFIG.parent / config.get("path", ".")).resolve()

    print("=" * 65)
    print("MULTICOUNTRY DATASET AUDIT")
    print("=" * 65)
    print("Dataset root:", root)
    print("Classes:", config["names"])

    for split in ("train", "val"):
        image_dir = (root / config[split]).resolve()
        label_dir = root / "labels" / split

        if not image_dir.exists() or not label_dir.exists():
            print(f"\nERROR: Missing {split} images or labels directory.")
            print("Images:", image_dir)
            print("Labels:", label_dir)
            continue

        images = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )

        class_counts = Counter()
        country_counts = Counter()
        missing = []
        empty = 0
        malformed = []
        object_total = 0

        for image_path in images:
            country_counts[image_path.name.split("_")[0]] += 1
            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                missing.append(image_path.name)
                continue

            lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            if not lines:
                empty += 1
                continue

            for line in lines:
                parts = line.split()
                try:
                    if len(parts) != 5:
                        raise ValueError("Expected 5 fields")

                    class_id = int(parts[0])
                    box = [float(v) for v in parts[1:]]

                    if not 0 <= class_id < len(config["names"]):
                        raise ValueError("Invalid class ID")

                    x, y, w, h = box
                    if not (
                        0 <= x <= 1 and 0 <= y <= 1
                        and 0 < w <= 1 and 0 < h <= 1
                    ):
                        raise ValueError("Invalid normalized box")

                    class_counts[class_id] += 1
                    object_total += 1

                except (ValueError, IndexError):
                    malformed.append((image_path.name, line))

        print(f"\n--- {split.upper()} ---")
        print("Images:", len(images))
        print("Label files:", len(list(label_dir.glob("*.txt"))))
        print("Missing label files:", len(missing))
        print("Empty label files:", empty)
        print("Malformed/invalid annotations:", len(malformed))
        print("Total annotated objects:", object_total)

        print("\nImages by country:")
        for country, count in sorted(country_counts.items()):
            print(f"  {country}: {count}")

        print("\nObjects by class:")
        for class_id, name in enumerate(config["names"]):
            print(f"  {class_id} ({name}): {class_counts[class_id]}")

        if missing:
            print("Missing examples:", missing[:10])
        if malformed:
            print("Invalid examples:", malformed[:5])

    print("\nAudit finished. No files were modified.")

if __name__ == "__main__":
    main()
