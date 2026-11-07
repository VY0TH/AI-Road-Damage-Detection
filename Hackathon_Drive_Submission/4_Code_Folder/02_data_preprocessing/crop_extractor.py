"""Step 2: Crop Extractor from Object Annotations.

Extracts labeled bounding boxes from raw road damage images and Pascal VOC XML
or YOLO annotations into dedicated class-named classification crop folders.
"""
from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_crops_voc(
    images_dir: Path,
    xmls_dir: Path,
    output_dir: Path,
    target_size: int = 224,
    margin_ratio: float = 0.05,
) -> tuple[int, dict[str, int]]:
    images_dir = Path(images_dir).resolve()
    xmls_dir = Path(xmls_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    class_counts: dict[str, int] = {}
    total_crops = 0

    xml_files = sorted(xmls_dir.glob("*.xml"))
    for xml_path in xml_files:
        stem = xml_path.stem
        # Locate corresponding image
        img_path = None
        for ext in IMAGE_EXTENSIONS:
            cand = images_dir / f"{stem}{ext}"
            if cand.is_file():
                img_path = cand
                break
        if img_path is None:
            continue

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            with Image.open(img_path) as img:
                img_w, img_h = img.size
                rgb_img = img.convert("RGB")

                for obj_idx, obj in enumerate(root.findall("object")):
                    cls_name = (obj.findtext("name") or "").strip()
                    if not cls_name:
                        continue

                    bndbox = obj.find("bndbox")
                    if bndbox is None:
                        continue

                    try:
                        xmin = float(bndbox.findtext("xmin") or 0)
                        ymin = float(bndbox.findtext("ymin") or 0)
                        xmax = float(bndbox.findtext("xmax") or 0)
                        ymax = float(bndbox.findtext("ymax") or 0)
                    except ValueError:
                        continue

                    # Margin expansion
                    box_w = xmax - xmin
                    box_h = ymax - ymin
                    pad_w = box_w * margin_ratio
                    pad_h = box_h * margin_ratio

                    x1 = max(0, int(xmin - pad_w))
                    y1 = max(0, int(ymin - pad_h))
                    x2 = min(img_w, int(xmax + pad_w))
                    y2 = min(img_h, int(ymax + pad_h))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = rgb_img.crop((x1, y1, x2, y2))
                    crop_resized = crop.resize((target_size, target_size), Image.Resampling.LANCZOS)

                    dest_class_dir = output_dir / cls_name
                    dest_class_dir.mkdir(parents=True, exist_ok=True)
                    crop_filename = f"{stem}_obj{obj_idx}.jpg"
                    crop_dest = dest_class_dir / crop_filename

                    crop_resized.save(crop_dest, format="JPEG", quality=95)

                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    total_crops += 1

                    manifest_rows.append({
                        "crop_filename": crop_filename,
                        "class_name": cls_name,
                        "source_image": img_path.name,
                        "source_xml": xml_path.name,
                        "crop_box": f"{x1},{y1},{x2},{y2}",
                        "crop_path": str(crop_dest.relative_to(output_dir)),
                    })
        except Exception as e:
            print(f"Error extracting crops from {xml_path}: {e}")

    manifest_path = output_dir / "crop_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["crop_filename", "class_name", "source_image", "source_xml", "crop_box", "crop_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return total_crops, class_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Path to images directory")
    parser.add_argument("--annotations", type=Path, required=True, help="Path to XML annotations")
    parser.add_argument("--output", type=Path, required=True, help="Output crops directory")
    parser.add_argument("--size", type=int, default=224, help="Target crop size")
    args = parser.parse_args()

    total, counts = extract_crops_voc(args.images, args.annotations, args.output, args.size)
    print(f"Extracted {total} crops into {args.output}")
    for k, v in sorted(counts.items()):
        print(f"  {k:10}: {v} crops")


if __name__ == "__main__":
    main()
