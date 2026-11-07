from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

base = Path("dataset/subsets/multicountry/RDD2022")
xml_dir = base / "split" / "train" / "annotations" / "xmls"
label_dir = base / "yolo_grouped" / "labels" / "train"

expected = {
    "D00", "D01", "D10", "D11", "D20",
    "D40", "D43", "D44", "D50", "D0w0"
}

xml_counts = Counter()
xml_images = Counter()
for path in xml_dir.glob("*.xml"):
    root = ET.parse(path).getroot()
    seen = set()
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name:
            xml_counts[name] += 1
            seen.add(name)
    xml_images.update(seen)

print("=== XML OBJECT COUNTS ===")
for name, count in sorted(xml_counts.items()):
    print(f"{name}: {count}")

print("\n=== XML CLASS NAMES NOT IN YOLO YAML ===")
print(sorted(set(xml_counts) - expected))

label_counts = Counter()
bad_rows = []
label_files = list(label_dir.glob("*.txt"))

for path in label_files:
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            bad_rows.append((path.name, line_no, line, "non-integer class ID"))
            continue

        if not 0 <= class_id < 10:
            bad_rows.append((path.name, line_no, line, "class ID outside 0-9"))
        else:
            label_counts[class_id] += 1

names = ["D00", "D01", "D10", "D11", "D20",
         "D40", "D43", "D44", "D50", "D0w0"]

print("\n=== YOLO TRAINING LABEL OBJECT COUNTS ===")
for class_id, name in enumerate(names):
    print(f"{class_id} ({name}): {label_counts[class_id]}")

print(f"\nYOLO label files: {len(label_files)}")
print(f"Invalid label rows: {len(bad_rows)}")
for row in bad_rows[:10]:
    print(row)

print("\n=== XML CLASSES WITH ZERO YOLO OBJECTS ===")
print(sorted(name for name in xml_counts if name in expected
             and label_counts[names.index(name)] == 0))
