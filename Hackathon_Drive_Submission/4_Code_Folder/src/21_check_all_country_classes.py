
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter

ROOT = Path(
    "dataset/raw/RoadDamageProjectdatasetrawRDD2022_FULL/RDD2022"
)

TARGET_CLASSES = {"D01", "D11", "D43", "D0w0"}
counts = Counter()
xml_count = 0
errors = 0

if not ROOT.exists():
    print("Folder not found:", ROOT.resolve())
    raise SystemExit(1)

# Scan only XML annotation folders, not every file in the dataset.
xml_files = list(ROOT.glob("**/annotations/**/*.xml"))

print("XML files found:", len(xml_files), flush=True)

for i, xml_path in enumerate(xml_files, start=1):
    try:
        for obj in ET.parse(xml_path).findall(".//object"):
            name = (obj.findtext("name") or "").strip()
            if name in TARGET_CLASSES:
                counts[name] += 1
        xml_count += 1
    except (ET.ParseError, OSError):
        errors += 1

    if i % 500 == 0:
        print(f"Scanned {i}/{len(xml_files)} XML files", flush=True)

print("\nRARE-CLASS SUMMARY")
print("=" * 35)

for cls in sorted(TARGET_CLASSES):
    print(f"{cls}: {counts[cls]} annotated objects")

print("XML files processed:", xml_count)
print("Unreadable/invalid XML files:", errors)
print("Finished.")
