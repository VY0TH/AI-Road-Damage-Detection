
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
XML_DIR = (
    PROJECT
    / "dataset/subsets/multicountry/RDD2022/annotations/xmls"
)

class_counts = Counter()
image_counts = Counter()
xml_count = 0
errors = []

for xml_path in XML_DIR.glob("*.xml"):
    try:
        root = ET.parse(xml_path).getroot()
        xml_count += 1

        names = [
            obj.findtext("name", default="unknown").strip()
            for obj in root.findall("object")
        ]

        for name in names:
            class_counts[name] += 1
            image_counts[name] += 1

    except ET.ParseError:
        errors.append(xml_path.name)

print("XML files processed:", xml_count)
print("\nObjects by annotation class:")

for name, count in class_counts.most_common():
    print(f"{name}: {count}")

print("\nXML parsing errors:", len(errors))
if errors:
    print(errors[:20])
