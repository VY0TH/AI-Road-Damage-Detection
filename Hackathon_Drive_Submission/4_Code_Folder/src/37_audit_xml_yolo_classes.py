
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "dataset/subsets/multicountry/RDD2022"

EXPECTED = {
    "D00", "D01", "D10", "D11", "D20",
    "D40", "D43", "D44", "D50", "D0w0"
}

xml_counts = Counter()
unknown_classes = Counter()
repair_only = 0
mixed_repair = 0
errors = []

for xml_path in sorted((ROOT / "annotations/xmls").glob("*.xml")):
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        errors.append(xml_path.name)
        continue

    names = [
        obj.findtext("name", default="unknown").strip()
        for obj in root.findall("object")
    ]

    xml_counts.update(names)
    unknown_classes.update(name for name in names
                           if name not in EXPECTED and name != "Repair")

    if "Repair" in names:
        if set(names) == {"Repair"}:
            repair_only += 1
        else:
            mixed_repair += 1

print("XML class counts:")
for name, count in xml_counts.most_common():
    print(f"  {name}: {count}")

print("\nRepair-only images:", repair_only)
print("Images with Repair and other classes:", mixed_repair)
print("Unexpected non-Repair classes:", dict(unknown_classes))
print("XML parsing errors:", len(errors))

if unknown_classes or errors:
    print("AUDIT NEEDS INVESTIGATION")
else:
    print("CLASS AUDIT PASSED: no unexpected non-Repair classes.")
