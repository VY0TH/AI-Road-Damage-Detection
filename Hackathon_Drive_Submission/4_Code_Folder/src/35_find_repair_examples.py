
from pathlib import Path
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
XML_DIR = (
    PROJECT
    / "dataset/subsets/multicountry/RDD2022/annotations/xmls"
)

found = 0

for xml_path in sorted(XML_DIR.glob("*.xml")):
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        continue

    repair_objects = [
        obj for obj in root.findall("object")
        if obj.findtext("name", default="").strip() == "Repair"
    ]

    if repair_objects:
        print(f"\nImage: {xml_path.stem}.jpg")
        print(f"Repair objects: {len(repair_objects)}")

        for obj in repair_objects:
            box = obj.find("bndbox")
            if box is not None:
                coords = [
                    box.findtext(key)
                    for key in ("xmin", "ymin", "xmax", "ymax")
                ]
                print("  Box:", coords)

        found += 1
        if found >= 5:
            break

print(f"\nFinished. Found {found} example images.")
