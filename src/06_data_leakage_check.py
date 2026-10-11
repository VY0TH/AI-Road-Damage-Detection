import os
import hashlib
import xml.etree.ElementTree as ET
from collections import defaultdict
from PIL import Image

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\Vyom Thakur\Desktop\AI hackathon\RoadDamageProject"

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "subsets",
    "small",
    "India"
)

IMAGE_DIR = os.path.join(
    DATASET_ROOT,
    "images"
)

XML_DIR = os.path.join(
    DATASET_ROOT,
    "annotations",
    "xmls"
)

CLASSES = {
    "D00", "D01", "D10", "D11", "D20",
    "D40", "D43", "D44", "D50", "D0w0"
}


# ============================================================
# HASH FUNCTION
# ============================================================

def get_sha256(path):

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)

    return sha256.hexdigest()


# ============================================================
# IMAGE FILES
# ============================================================

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

xml_files = [
    f for f in os.listdir(XML_DIR)
    if f.lower().endswith(".xml")
]

print("=" * 60)
print("DATA LEAKAGE CHECK")
print("=" * 60)

print(f"\nImages found : {len(image_files)}")
print(f"XML files    : {len(xml_files)}")


# ============================================================
# CHECK 1 — IMAGE/XML MATCHING
# ============================================================

print("\n" + "=" * 60)
print("1. IMAGE ↔ XML CONSISTENCY")
print("=" * 60)

missing_xml = []
extra_xml = []

image_names = {
    os.path.splitext(f)[0]
    for f in image_files
}

xml_names = {
    os.path.splitext(f)[0]
    for f in xml_files
}

for name in image_names - xml_names:
    missing_xml.append(name)

for name in xml_names - image_names:
    extra_xml.append(name)

print(f"Images without XML : {len(missing_xml)}")
print(f"XML without image  : {len(extra_xml)}")

if missing_xml:
    print("\nMissing XML examples:")
    for x in missing_xml[:10]:
        print(" ", x)

if extra_xml:
    print("\nExtra XML examples:")
    for x in extra_xml[:10]:
        print(" ", x)


# ============================================================
# CHECK 2 — EXACT DUPLICATE IMAGES
# ============================================================

print("\n" + "=" * 60)
print("2. EXACT DUPLICATE IMAGE CHECK")
print("=" * 60)

hash_to_files = defaultdict(list)

for image_file in image_files:

    path = os.path.join(IMAGE_DIR, image_file)

    try:
        file_hash = get_sha256(path)
        hash_to_files[file_hash].append(image_file)

    except Exception as e:
        print(f"Error hashing {image_file}: {e}")


duplicate_groups = [
    files
    for files in hash_to_files.values()
    if len(files) > 1
]

duplicate_count = sum(
    len(group) - 1
    for group in duplicate_groups
)

print(f"Duplicate groups : {len(duplicate_groups)}")
print(f"Duplicate files  : {duplicate_count}")

if duplicate_groups:

    print("\nDuplicate examples:")

    for group in duplicate_groups[:10]:
        print(" ", group)

else:
    print("No exact duplicate images found.")


# ============================================================
# CHECK 3 — NEAR DUPLICATES
# ============================================================

print("\n" + "=" * 60)
print("3. NEAR-DUPLICATE IMAGE CHECK")
print("=" * 60)

print("Checking perceptual similarity...")

# Simple perceptual hash
def average_hash(path, size=16):

    image = Image.open(path).convert("L")
    image = image.resize((size, size))

    pixels = list(image.getdata())

    average = sum(pixels) / len(pixels)

    bits = [
        1 if pixel >= average else 0
        for pixel in pixels
    ]

    return bits


def hamming_distance(hash1, hash2):

    return sum(
        a != b
        for a, b in zip(hash1, hash2)
    )


image_hashes = {}

for image_file in image_files:

    path = os.path.join(IMAGE_DIR, image_file)

    try:
        image_hashes[image_file] = average_hash(path)

    except Exception as e:
        print(f"Error processing {image_file}: {e}")


near_duplicates = []

files = list(image_hashes.keys())

# Threshold is intentionally conservative.
# These are candidates, NOT automatically deleted.
THRESHOLD = 5

for i in range(len(files)):

    for j in range(i + 1, len(files)):

        file1 = files[i]
        file2 = files[j]

        distance = hamming_distance(
            image_hashes[file1],
            image_hashes[file2]
        )

        if distance <= THRESHOLD:

            near_duplicates.append(
                (file1, file2, distance)
            )


print(
    f"Potential near-duplicate pairs : "
    f"{len(near_duplicates)}"
)

if near_duplicates:

    print("\nPotential near-duplicates:")

    for file1, file2, distance in near_duplicates[:20]:

        print(
            f"  {file1} <-> {file2} "
            f"(distance={distance})"
        )

    print(
        "\nIMPORTANT: These are only candidates. "
        "They are NOT automatically removed."
    )

else:

    print("No obvious near-duplicate pairs found.")


# ============================================================
# CHECK 4 — XML ANNOTATIONS
# ============================================================

print("\n" + "=" * 60)
print("4. ANNOTATION VALIDATION")
print("=" * 60)

invalid_xml = []
unknown_classes = []
invalid_boxes = []
object_count = 0

for xml_file in xml_files:

    xml_path = os.path.join(XML_DIR, xml_file)

    try:

        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")

        if size is None:
            invalid_xml.append(
                (xml_file, "Missing image size")
            )
            continue

        width_element = size.find("width")
        height_element = size.find("height")

        if width_element is None or height_element is None:
            invalid_xml.append(
                (xml_file, "Missing width/height")
            )
            continue

        width = float(width_element.text)
        height = float(height_element.text)

        objects = root.findall("object")

        for obj in objects:

            object_count += 1

            name = obj.find("name")

            if name is None:
                invalid_xml.append(
                    (xml_file, "Object without class")
                )
                continue

            class_name = name.text.strip()

            if class_name not in CLASSES:

                unknown_classes.append(
                    (xml_file, class_name)
                )

            bbox = obj.find("bndbox")

            if bbox is None:

                invalid_boxes.append(
                    (xml_file, "Missing bounding box")
                )
                continue

            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            # Check coordinates
            if (
                xmin < 0
                or ymin < 0
                or xmax > width
                or ymax > height
                or xmin >= xmax
                or ymin >= ymax
            ):

                invalid_boxes.append(
                    (
                        xml_file,
                        f"Invalid box "
                        f"({xmin},{ymin},{xmax},{ymax})"
                    )
                )

    except Exception as e:

        invalid_xml.append(
            (xml_file, str(e))
        )


print(f"Total annotation objects : {object_count}")
print(f"Invalid XML              : {len(invalid_xml)}")
print(f"Unknown classes          : {len(unknown_classes)}")
print(f"Invalid bounding boxes   : {len(invalid_boxes)}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL LEAKAGE CHECK SUMMARY")
print("=" * 60)

checks = {
    "Image/XML mismatch": len(missing_xml) + len(extra_xml),
    "Exact duplicate groups": len(duplicate_groups),
    "Near duplicate pairs": len(near_duplicates),
    "Invalid XML": len(invalid_xml),
    "Unknown classes": len(unknown_classes),
    "Invalid bounding boxes": len(invalid_boxes)
}

for name, value in checks.items():

    status = "PASS" if value == 0 else "REVIEW"

    print(
        f"{name:30} : "
        f"{value:5}  [{status}]"
    )


print("\nFinished.")
print("=" * 60)