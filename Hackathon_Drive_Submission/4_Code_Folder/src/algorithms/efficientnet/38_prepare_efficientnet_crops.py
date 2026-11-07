
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter
import csv
import cv2

PROJECT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
ROOT = PROJECT / "dataset/subsets/multicountry/RDD2022"
OUTPUT = PROJECT / "dataset/processed/efficientnet_crops"

SPLITS = {
    "train": ROOT / "split/train/images",
    "val": ROOT / "split/val/images",
}

XML_DIR = ROOT / "annotations/xmls"

CLASSES = [
    "D00", "D01", "D10", "D11", "D20",
    "D40", "D43", "D44", "D50", "D0w0"
]

CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def main():
    counts = {split: Counter() for split in SPLITS}
    manifest_path = OUTPUT / "manifest.csv"
    OUTPUT.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = Counter()

    for split, image_dir in SPLITS.items():
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")

        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            # Source images use their original filename stem for XML lookup.
            xml_path = XML_DIR / f"{image_path.stem}.xml"
            if not xml_path.exists():
                skipped["missing_xml"] += 1
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                skipped["unreadable_image"] += 1
                continue

            try:
                xml_root = ET.parse(xml_path).getroot()
            except ET.ParseError:
                skipped["invalid_xml"] += 1
                continue

            height, width = image.shape[:2]

            for obj_index, obj in enumerate(xml_root.findall("object")):
                name = obj.findtext("name", default="").strip()

                # Repair is deliberately excluded from the 10 detector classes.
                if name not in CLASS_TO_ID:
                    skipped[f"excluded_{name or 'unknown'}"] += 1
                    continue

                box = obj.find("bndbox")
                if box is None:
                    skipped["missing_box"] += 1
                    continue

                try:
                    xmin = max(0, int(float(box.findtext("xmin"))))
                    ymin = max(0, int(float(box.findtext("ymin"))))
                    xmax = min(width, int(float(box.findtext("xmax"))))
                    ymax = min(height, int(float(box.findtext("ymax"))))
                except (TypeError, ValueError):
                    skipped["invalid_coordinates"] += 1
                    continue

                if xmax <= xmin or ymax <= ymin:
                    skipped["invalid_box"] += 1
                    continue

                crop = image[ymin:ymax, xmin:xmax]
                if crop.size == 0:
                    skipped["empty_crop"] += 1
                    continue

                # Use class folders expected by torchvision ImageFolder.
                crop_dir = OUTPUT / split / name
                crop_dir.mkdir(parents=True, exist_ok=True)

                crop_name = f"{image_path.stem}_obj{obj_index}.jpg"
                crop_path = crop_dir / crop_name

                if not cv2.imwrite(str(crop_path), crop):
                    skipped["write_failed"] += 1
                    continue

                counts[split][name] += 1
                rows.append({
                    "split": split,
                    "crop_path": str(crop_path.relative_to(PROJECT)),
                    "class_name": name,
                    "class_id": CLASS_TO_ID[name],
                    "source_image": image_path.name,
                    "source_xml": xml_path.name,
                })

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "split", "crop_path", "class_name", "class_id",
            "source_image", "source_xml"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print("\nCrop generation complete.")
    print("Output:", OUTPUT)
    print("Manifest:", manifest_path)
    print("Total crops:", len(rows))

    for split in SPLITS:
        print(f"\n{split.upper()} crops:")
        for name in CLASSES:
            print(f"  {name}: {counts[split][name]}")

    print("\nSkipped items:")
    for reason, count in skipped.items():
        print(f"  {reason}: {count}")

if __name__ == "__main__":
    main()
