from pathlib import Path
import xml.etree.ElementTree as ET

base = Path("dataset/subsets/multicountry/RDD2022")
xml_dir = base / "split/train/annotations/xmls"
label_dir = base / "yolo_grouped/labels/train"

samples = [
    "China_0005",
    "China_0026",
    "China_0031",
    "China_0047",
    "China_0059",
]

names = ["D00", "D01", "D10", "D11", "D20",
         "D40", "D43", "D44", "D50", "D0w0"]

for stem in samples:
    xml_path = xml_dir / f"{stem}.xml"
    label_path = label_dir / f"{stem}.txt"

    print(f"\n=== {stem} ===")

    if xml_path.exists():
        root = ET.parse(xml_path).getroot()
        print("XML objects:")
        for obj in root.findall("object"):
            name = obj.findtext("name", default="?")
            box = obj.find("bndbox")
            coords = (
                [box.findtext(k) for k in ("xmin", "ymin", "xmax", "ymax")]
                if box is not None else []
            )
            print(f"  {name}: {coords}")
    else:
        print("XML file missing")

    print("YOLO labels:")
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                try:
                    cls = int(parts[0])
                    name = names[cls] if 0 <= cls < len(names) else "INVALID"
                    print(f"  {name}: {' '.join(parts[1:5])}")
                except ValueError:
                    print(f"  Invalid row: {line}")
    else:
        print("Label file missing")
