
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "dataset/subsets/multicountry/RDD2022"
XML_DIR = ROOT / "annotations/xmls"
IMAGE_DIR = ROOT / "images"
OUTPUT = PROJECT / "results/repair_review"
OUTPUT.mkdir(parents=True, exist_ok=True)

count = 0

for xml_path in sorted(XML_DIR.glob("*.xml")):
    root = ET.parse(xml_path).getroot()
    repairs = [
        obj for obj in root.findall("object")
        if obj.findtext("name", default="").strip() == "Repair"
    ]

    if not repairs:
        continue

    image_path = IMAGE_DIR / f"{xml_path.stem}.jpg"
    if not image_path.exists():
        print("Image missing:", image_path.name)
        continue

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    draw = ImageDraw.Draw(image)

    for obj in repairs:
        box = obj.find("bndbox")
        x1 = int(float(box.findtext("xmin")))
        y1 = int(float(box.findtext("ymin")))
        x2 = int(float(box.findtext("xmax")))
        y2 = int(float(box.findtext("ymax")))

        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1, max(0, y1 - 15)), "Repair", fill="yellow")

    destination = OUTPUT / f"{xml_path.stem}_repair.jpg"
    image.save(destination)
    print("Saved:", destination.name)

    count += 1
    if count == 5:
        break

print(f"\nFinished: {count} images saved to {OUTPUT}")
