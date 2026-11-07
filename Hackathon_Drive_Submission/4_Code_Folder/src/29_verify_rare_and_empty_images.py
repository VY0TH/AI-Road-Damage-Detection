
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT / "dataset/subsets/multicountry/RDD2022"
YOLO_ROOT = DATASET_ROOT / "yolo"
RAW_ROOT = PROJECT / "dataset/raw"

OUTPUT = PROJECT / "results/rare_class_review"
OUTPUT.mkdir(parents=True, exist_ok=True)

RARE = {
    "India_0002",
    "India_0000",
    "India_0039",
    "Japan_0000",
    "Japan_0008",
    "China_0100",
    "China_0101",
    "China_0102",
}

IMAGE_DIRS = [
    DATASET_ROOT / "images",
    DATASET_ROOT / "split/train/images",
    DATASET_ROOT / "split/val/images",
]

XML_DIRS = [
    DATASET_ROOT / "annotations/xmls",
    DATASET_ROOT / "split/train/annotations/xmls",
    DATASET_ROOT / "split/val/annotations/xmls",
]


def find_file(stem, folders, extensions):
    for folder in folders:
        if not folder.exists():
            continue

        for ext in extensions:
            candidate = folder / f"{stem}{ext}"
            if candidate.is_file():
                return candidate

    # Fallback: search raw data if the file is not in the main dataset.
    for folder in [RAW_ROOT]:
        if folder.exists():
            for ext in extensions:
                matches = folder.rglob(f"{stem}{ext}")
                for match in matches:
                    if match.is_file():
                        return match

    return None


def find_image(stem):
    return find_file(
        stem,
        IMAGE_DIRS,
        [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]
    )


def find_xml(stem):
    return find_file(stem, XML_DIRS, [".xml", ".XML"])


def main():
    print("Reviewing selected images...")

    for stem in sorted(RARE):
        image_path = find_image(stem)

        if image_path is None:
            print(f"Image not found: {stem}")
            continue

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        draw = ImageDraw.Draw(image)

        label_path = None
        for split in ("train", "val"):
            candidate = YOLO_ROOT / "labels" / split / f"{stem}.txt"
            if candidate.is_file():
                label_path = candidate
                break

        label_lines = []
        if label_path:
            label_lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        width, height = image.size

        for line in label_lines:
            parts = line.split()
            if len(parts) != 5:
                print(f"Skipping malformed label: {stem}: {line}")
                continue

            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])

            x1 = max(0, int((x - w / 2) * width))
            y1 = max(0, int((y - h / 2) * height))
            x2 = min(width - 1, int((x + w / 2) * width))
            y2 = min(height - 1, int((y + h / 2) * height))

            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            draw.text((x1, max(0, y1 - 15)), f"class {cls}", fill="red")

        destination = OUTPUT / f"{stem}_review.jpg"
        image.save(destination)

        xml_path = find_xml(stem)

        print(f"\n{stem}")
        print("  Image:", image_path)
        print("  YOLO label:", label_path or "(not found)")
        print("  YOLO objects:", len(label_lines))
        print("  Original XML:", xml_path or "(not found)")
        print("  Review image:", destination)

        if xml_path:
            try:
                root = ET.parse(xml_path).getroot()
                names = [
                    obj.findtext("name", default="unknown")
                    for obj in root.findall("object")
                ]
                print("  Original XML classes:", names or "(no objects)")
            except ET.ParseError:
                print("  Original XML could not be parsed")

    print("\nFinished. Original dataset files were not modified.")
    print("Review folder:", OUTPUT)


if __name__ == "__main__":
    main()
