
from pathlib import Path
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "dataset/subsets/multicountry/RDD2022"

def main():
    for split in ("train", "val"):
        labels_dir = ROOT / "yolo" / "labels" / split
        images_dir = ROOT / "yolo" / "images" / split

        empty = [
            p for p in labels_dir.glob("*.txt")
            if not p.read_text(encoding="utf-8").strip()
        ]

        print(f"\n{split.upper()}: {len(empty)} empty labels")
        checked = 0
        xml_objects_found = 0
        missing_xml = 0

        for label in empty:
            xml_path = ROOT / "annotations" / "xmls" / f"{label.stem}.xml"

            if not xml_path.exists():
                missing_xml += 1
                continue

            try:
                root = ET.parse(xml_path).getroot()
                objects = root.findall("object")
            except ET.ParseError:
                print("INVALID XML:", xml_path.name)
                continue

            checked += 1

            if objects:
                xml_objects_found += 1
                names = [o.findtext("name", default="?") for o in objects]
                print(f"MISMATCH: {label.stem} has empty YOLO label "
                      f"but XML contains {names}")

        print("Empty labels checked against XML:", checked)
        print("XML files with objects despite empty labels:",
              xml_objects_found)
        print("Missing XML files:", missing_xml)

if __name__ == "__main__":
    main()
