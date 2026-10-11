
from pathlib import Path
from collections import defaultdict
import hashlib
import random
import shutil
import xml.etree.ElementTree as ET

from PIL import Image


# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parent.parent

SOURCE_IMAGES = ROOT / "dataset/raw/India/India/train/images"
SOURCE_XML = ROOT / "dataset/raw/India/India/train/annotations/xmls"

YOLO_ROOT = ROOT / "dataset/yolo"

TRAIN_IMAGES = YOLO_ROOT / "images/train"
VAL_IMAGES = YOLO_ROOT / "images/val"
TRAIN_LABELS = YOLO_ROOT / "labels/train"
VAL_LABELS = YOLO_ROOT / "labels/val"

TRAIN_RATIO = 0.80
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Preserve the existing 10-class mapping.
CLASSES = [
    "D00", "D01", "D10", "D11", "D20",
    "D40", "D43", "D44", "D50", "D0w0"
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def sha256_file(path):
    """Hash file contents to identify exact duplicate images."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def read_annotation(xml_path, image_width, image_height):
    """Convert Pascal VOC boxes into normalized YOLO labels."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = []

    for obj in root.findall("object"):
        class_name = obj.findtext("name", "").strip()

        if class_name not in CLASS_TO_ID:
            raise ValueError(
                f"Unknown class '{class_name}' in {xml_path}"
            )

        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bounding box in {xml_path}")

        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))

        # Clip boxes to image boundaries.
        xmin = max(0.0, min(xmin, image_width))
        xmax = max(0.0, min(xmax, image_width))
        ymin = max(0.0, min(ymin, image_height))
        ymax = max(0.0, min(ymax, image_height))

        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"Invalid bounding box in {xml_path}")

        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        labels.append(
            f"{CLASS_TO_ID[class_name]} "
            f"{x_center:.6f} {y_center:.6f} "
            f"{width:.6f} {height:.6f}"
        )

    return labels


def main():
    print("=" * 65)
    print("REBUILDING CLEAN YOLO TRAIN / VALIDATION SPLIT")
    print("=" * 65)

    # Validate source paths before touching generated folders.
    if not SOURCE_IMAGES.is_dir() or not SOURCE_XML.is_dir():
        print("ERROR: Original image or XML folder was not found.")
        print("Images:", SOURCE_IMAGES)
        print("XMLs:  ", SOURCE_XML)
        return

    images = sorted(
        p for p in SOURCE_IMAGES.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print("ERROR: No source images found.")
        return

    print(f"Original images found: {len(images)}")

    # Prepare and validate all annotations BEFORE clearing output.
    samples = []
    print("Validating source images and XML annotations...")

    try:
        for index, image_path in enumerate(images, start=1):
            xml_path = SOURCE_XML / f"{image_path.stem}.xml"

            if not xml_path.is_file():
                raise FileNotFoundError(
                    f"Missing annotation for {image_path.name}: {xml_path}"
                )

            with Image.open(image_path) as image:
                width, height = image.size

            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions: {image_path}")

            yolo_labels = read_annotation(xml_path, width, height)
            image_hash = sha256_file(image_path)

            samples.append({
                "image": image_path,
                "labels": yolo_labels,
                "hash": image_hash,
            })

            if index % 500 == 0 or index == len(images):
                print(f"Validated {index}/{len(images)}")

    except Exception as error:
        print("\nERROR: Validation failed.")
        print(error)
        print("Existing generated folders were not cleared.")
        return

    # Group identical images so no exact duplicate crosses splits.
    hash_groups = defaultdict(list)

    for sample in samples:
        hash_groups[sample["hash"]].append(sample)

    groups = list(hash_groups.values())
    random.Random(SEED).shuffle(groups)

    # Target approximately 20% of all source images for validation.
    target_val_count = round(len(samples) * (1 - TRAIN_RATIO))

    val_samples = []
    train_samples = []

    for group in groups:
        if len(val_samples) < target_val_count:
            val_samples.extend(group)
        else:
            train_samples.extend(group)

    if not train_samples or not val_samples:
        print("ERROR: The split would produce an empty partition.")
        return

    train_hashes = {s["hash"] for s in train_samples}
    val_hashes = {s["hash"] for s in val_samples}

    if train_hashes & val_hashes:
        print("ERROR: Duplicate leakage detected before writing.")
        return

    print("\nPlanned split:")
    print(f"TRAIN: {len(train_samples)} images")
    print(f"VAL:   {len(val_samples)} images")
    print(f"Exact duplicate groups in source: {len(groups)}")

    # Only generated TRAIN and VAL directories will be replaced.
    output_dirs = [
        TRAIN_IMAGES, VAL_IMAGES, TRAIN_LABELS, VAL_LABELS
    ]

    print("\nRebuilding generated TRAIN and VAL folders...")

    for folder in output_dirs:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

    def write_samples(items, image_dir, label_dir):
        for sample in items:
            image_path = sample["image"]
            destination_image = image_dir / image_path.name
            destination_label = label_dir / f"{image_path.stem}.txt"

            shutil.copy2(image_path, destination_image)

            # Empty file is correct for images without damage boxes.
            destination_label.write_text(
                "\n".join(sample["labels"]),
                encoding="utf-8"
            )

    write_samples(train_samples, TRAIN_IMAGES, TRAIN_LABELS)
    write_samples(val_samples, VAL_IMAGES, VAL_LABELS)

    # Write dataset configuration.
    yaml_lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(CLASSES)}",
        "names:",
    ]

    for index, name in enumerate(CLASSES):
        yaml_lines.append(f"  {index}: {name}")

    (YOLO_ROOT / "data.yaml").write_text(
        "\n".join(yaml_lines) + "\n",
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("SPLIT REBUILD COMPLETE")
    print("=" * 65)
    print(f"TRAIN images: {len(list(TRAIN_IMAGES.iterdir()))}")
    print(f"VAL images:   {len(list(VAL_IMAGES.iterdir()))}")
    print(f"TRAIN labels: {len(list(TRAIN_LABELS.glob('*.txt')))}")
    print(f"VAL labels:   {len(list(VAL_LABELS.glob('*.txt')))}")
    print(f"Configuration: {YOLO_ROOT / 'data.yaml'}")
    print("Exact hash overlap: 0")
    print("Original raw dataset: untouched")
    print("Generated test folders: untouched")


if __name__ == "__main__":
    main()
