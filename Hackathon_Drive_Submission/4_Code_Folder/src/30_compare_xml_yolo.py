
from pathlib import Path
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "dataset/subsets/multicountry/RDD2022"
YOLO_ROOT = ROOT / "yolo"

CLASS_NAMES = {
    0: "D00",
    1: "D01",
    2: "D10",
    3: "D11",
    4: "D20",
    5: "D40",
    6: "D43",
    7: "D44",
    8: "D50",
    9: "D0w0",
}

IMAGES = ["India_0039", "Japan_0000", "Japan_0008"]

def read_xml(stem):
    path = ROOT / "annotations" / "xmls" / f"{stem}.xml"
    root = ET.parse(path).getroot()
    result = []

    for obj in root.findall("object"):
        name = obj.findtext("name", default="unknown")
        box = obj.find("bndbox")
        if box is None:
            continue

        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))
        result.append((name, xmin, ymin, xmax, ymax))

    size = root.find("size")
    width = float(size.findtext("width"))
    height = float(size.findtext("height"))
    return width, height, result

def main():
    for stem in IMAGES:
        print(f"\n{'=' * 50}\n{stem}")

        width, height, xml_objects = read_xml(stem)
        print("\nOriginal XML objects:")

        for name, x1, y1, x2, y2 in xml_objects:
            print(f"  {name}: ({x1:.1f}, {y1:.1f}, "
                  f"{x2:.1f}, {y2:.1f})")

        label_path = None
        for split in ("train", "val"):
            candidate = YOLO_ROOT / "labels" / split / f"{stem}.txt"
            if candidate.exists():
                label_path = candidate
                break

        if label_path is None:
            print("YOLO label file NOT FOUND")
            continue

        print("\nYOLO objects converted back to pixel coordinates:")

        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                print("  INVALID LINE:", line)
                continue

            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            x1 = (x - w / 2) * width
            y1 = (y - h / 2) * height
            x2 = (x + w / 2) * width
            y2 = (y + h / 2) * height

            name = CLASS_NAMES.get(cls, f"UNKNOWN_CLASS_{cls}")
            print(f"  {name}: ({x1:.1f}, {y1:.1f}, "
                  f"{x2:.1f}, {y2:.1f})")

        print("\nCompare each YOLO class and coordinate "
              "with the XML objects above.")

if __name__ == "__main__":
    main()
