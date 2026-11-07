
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

ROOT = Path(
    "dataset/raw/RoadDamageProjectdatasetrawRDD2022_FULL/RDD2022"
)

TARGETS = {"D01", "D11", "D43", "D0w0"}
MAX_EXAMPLES = 20

if not ROOT.exists():
    print("Dataset folder not found:", ROOT.resolve())
    raise SystemExit(1)

counts = Counter()
examples = defaultdict(list)
xml_files = list(ROOT.glob("**/annotations/**/*.xml"))

print("XML files to inspect:", len(xml_files), flush=True)

for i, xml_path in enumerate(xml_files, 1):
    try:
        root = ET.parse(xml_path).getroot()
        classes_in_image = {
            (obj.findtext("name") or "").strip()
            for obj in root.findall(".//object")
        }

        for cls in classes_in_image & TARGETS:
            counts[cls] += 1
            if len(examples[cls]) < MAX_EXAMPLES:
                examples[cls].append(xml_path)

    except (ET.ParseError, OSError):
        continue

    if i % 5000 == 0:
        print(f"Scanned {i}/{len(xml_files)}", flush=True)

print("\nRARE-CLASS CANDIDATES")
print("=" * 45)

for cls in sorted(TARGETS):
    print(f"\n{cls}: found in {counts[cls]} images")
    for path in examples[cls]:
        print(" ", path)

print("\nComplete. No files were modified.")
