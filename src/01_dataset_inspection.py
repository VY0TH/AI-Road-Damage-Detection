import os
import xml.etree.ElementTree as ET
from collections import Counter

# ============================================================
# RDD2022 INDIA - DATASET INSPECTION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "raw",
    "India",
    "India"
)

TRAIN_PATH = os.path.join(DATASET_PATH, "train")

IMAGE_PATH = os.path.join(TRAIN_PATH, "images")

ANNOTATION_PATH = os.path.join(
    TRAIN_PATH,
    "annotations",
    "xmls"
)

print("=" * 70)
print("        AI ROAD DAMAGE DETECTION")
print("        RDD2022 INDIA DATASET INSPECTION")
print("=" * 70)

# Check dataset
print("\n[1] CHECKING DATASET")

print("Dataset path:")
print(DATASET_PATH)

if not os.path.exists(DATASET_PATH):
    print("\nERROR: Dataset path not found!")
    exit()

print("Dataset path: OK")

if not os.path.exists(IMAGE_PATH):
    print("ERROR: Images folder not found!")
    exit()

if not os.path.exists(ANNOTATION_PATH):
    print("ERROR: XML annotations folder not found!")
    exit()

print("Images folder: OK")
print("Annotations folder: OK")

# Find images
print("\n[2] IMAGE COUNT")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

images = []

for file in os.listdir(IMAGE_PATH):
    extension = os.path.splitext(file)[1].lower()

    if extension in IMAGE_EXTENSIONS:
        images.append(file)

print("Total training images:", len(images))

# Find XML files
print("\n[3] ANNOTATION COUNT")

xml_files = [
    file
    for file in os.listdir(ANNOTATION_PATH)
    if file.lower().endswith(".xml")
]

print("Total XML files:", len(xml_files))

# Match images and XML
print("\n[4] IMAGE / XML MATCHING")

image_names = {
    os.path.splitext(file)[0]
    for file in images
}

xml_names = {
    os.path.splitext(file)[0]
    for file in xml_files
}

missing_xml = image_names - xml_names
missing_images = xml_names - image_names

print("Images without XML :", len(missing_xml))
print("XML without image  :", len(missing_images))

# Read XML
print("\n[5] READING XML ANNOTATIONS")

class_counter = Counter()

images_with_damage = 0
images_without_damage = 0
total_objects = 0
invalid_xml = 0

for xml_file in xml_files:

    xml_path = os.path.join(
        ANNOTATION_PATH,
        xml_file
    )

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        objects = root.findall("object")

        if len(objects) == 0:

            images_without_damage += 1

        else:

            images_with_damage += 1

            for obj in objects:

                name = obj.find("name")

                if name is not None:

                    class_name = name.text.strip()

                    class_counter[class_name] += 1
                    total_objects += 1

    except Exception as error:

        invalid_xml += 1

        print("Invalid XML:", xml_file)
        print(error)

# Class distribution
print("\n[6] DAMAGE CLASS DISTRIBUTION")

for class_name, count in class_counter.most_common():

    print(f"{class_name:10} : {count}")

# Summary
print("\n" + "=" * 70)
print("                    DATASET SUMMARY")
print("=" * 70)

print("Training images        :", len(images))
print("XML annotations        :", len(xml_files))
print("Images with damage     :", images_with_damage)
print("Images without damage  :", images_without_damage)
print("Total damage objects   :", total_objects)
print("Invalid XML files      :", invalid_xml)
print("Images missing XML     :", len(missing_xml))
print("XML missing image      :", len(missing_images))

print("\nClasses:")

for class_name, count in class_counter.most_common():
    print(f"  {class_name} : {count}")

print("\n" + "=" * 70)
print("              INSPECTION COMPLETE")
print("=" * 70)