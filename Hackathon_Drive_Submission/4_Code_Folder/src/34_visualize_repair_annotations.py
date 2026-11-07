
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "dataset/subsets/multicountry/RDD2022"

stem = "China_0195"
image_path = ROOT / "images" / f"{stem}.jpg"
xml_path = ROOT / "annotations" / "xmls" / f"{stem}.xml"
output_dir = PROJECT / "results/repair_review"
output_dir.mkdir(parents=True, exist_ok=True)

image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(image)
root = ET.parse(xml_path).getroot()

for obj in root.findall("object"):
    name = obj.findtext("name", default="unknown")
    box = obj.find("bndbox")

    x1 = int(float(box.findtext("xmin")))
    y1 = int(float(box.findtext("ymin")))
    x2 = int(float(box.findtext("xmax")))
    y2 = int(float(box.findtext("ymax")))

    draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
    draw.text((x1, max(0, y1 - 15)), name, fill="yellow")

output_path = output_dir / f"{stem}_xml_review.jpg"
image.save(output_path)
print("Saved:", output_path)
