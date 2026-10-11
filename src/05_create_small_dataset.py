import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict

# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = r"C:\Users\Vyom Thakur\Desktop\AI hackathon\RoadDamageProject"

SOURCE_IMAGES = os.path.join(
    PROJECT_ROOT,
    "dataset", "raw", "India", "India", "train", "images"
)

SOURCE_XML = os.path.join(
    PROJECT_ROOT,
    "dataset", "raw", "India", "India", "train", "annotations", "xmls"
)

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset", "subsets", "small", "India"
)

OUTPUT_IMAGES = os.path.join(OUTPUT_ROOT, "images")
OUTPUT_XML = os.path.join(OUTPUT_ROOT, "annotations", "xmls")

# Number of images
DAMAGED_COUNT = 100
NON_DAMAGED_COUNT = 100

# Reproducible random selection
random.seed(42)

# RDD2022 classes
CLASSES = [
    "D00",
    "D01",
    "D10",
    "D11",
    "D20",
    "D40",
    "D43",
    "D44",
    "D50",
    "D0w0"
]


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_XML, exist_ok=True)


# ============================================================
# READ XML ANNOTATIONS
# ============================================================

damaged_images = []
non_damaged_images = []

class_to_images = defaultdict(set)

image_files = [
    f for f in os.listdir(SOURCE_IMAGES)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Total source images found: {len(image_files)}")

for image_file in image_files:

    image_name = os.path.splitext(image_file)[0]
    xml_file = image_name + ".xml"
    xml_path = os.path.join(SOURCE_XML, xml_file)

    if not os.path.exists(xml_path):
        continue

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        objects = root.findall("object")

        if len(objects) == 0:
            # No damage
            non_damaged_images.append(image_file)

        else:
            # Damage exists
            damaged_images.append(image_file)

            for obj in objects:
                name = obj.find("name")

                if name is not None:
                    class_name = name.text.strip()

                    if class_name in CLASSES:
                        class_to_images[class_name].add(image_file)

    except Exception as e:
        print(f"Error reading {xml_file}: {e}")


# ============================================================
# PRINT DATASET INFORMATION
# ============================================================

print("\nDataset information")
print("-------------------")
print(f"Damaged images     : {len(damaged_images)}")
print(f"Non-damaged images : {len(non_damaged_images)}")

print("\nClass availability:")
for class_name in CLASSES:
    print(
        f"{class_name:5} : "
        f"{len(class_to_images[class_name])} images"
    )


# ============================================================
# SELECT DAMAGED IMAGES
# ============================================================

selected_damaged = set()

# First: try to include every class
print("\nSelecting representative images...")

for class_name in CLASSES:

    candidates = list(
        class_to_images[class_name] - selected_damaged
    )

    if candidates:
        selected = random.choice(candidates)
        selected_damaged.add(selected)

        print(
            f"Included {class_name}: {selected}"
        )

    else:
        print(
            f"WARNING: No unused image found for {class_name}"
        )


# Fill remaining damaged images randomly
remaining_damaged = list(
    set(damaged_images) - selected_damaged
)

random.shuffle(remaining_damaged)

needed = DAMAGED_COUNT - len(selected_damaged)

if needed > 0:
    selected_damaged.update(
        remaining_damaged[:needed]
    )

selected_damaged = list(selected_damaged)


# ============================================================
# SELECT NON-DAMAGED IMAGES
# ============================================================

random.shuffle(non_damaged_images)

selected_non_damaged = non_damaged_images[
    :NON_DAMAGED_COUNT
]


# ============================================================
# FINAL CHECK
# ============================================================

print("\nFinal selection")
print("----------------")
print(f"Damaged     : {len(selected_damaged)}")
print(f"Non-damaged : {len(selected_non_damaged)}")
print(f"Total       : {len(selected_damaged) + len(selected_non_damaged)}")


# ============================================================
# COPY FILES
# ============================================================

print("\nCopying files...")

selected_images = selected_damaged + selected_non_damaged

copied = 0
missing_xml = 0

for image_file in selected_images:

    source_image = os.path.join(
        SOURCE_IMAGES,
        image_file
    )

    destination_image = os.path.join(
        OUTPUT_IMAGES,
        image_file
    )

    shutil.copy2(
        source_image,
        destination_image
    )

    # Copy corresponding XML
    image_name = os.path.splitext(image_file)[0]
    xml_file = image_name + ".xml"

    source_xml = os.path.join(
        SOURCE_XML,
        xml_file
    )

    destination_xml = os.path.join(
        OUTPUT_XML,
        xml_file
    )

    if os.path.exists(source_xml):

        shutil.copy2(
            source_xml,
            destination_xml
        )

    else:
        missing_xml += 1

    copied += 1


# ============================================================
# SAVE A SIMPLE MANIFEST
# ============================================================

manifest_path = os.path.join(
    OUTPUT_ROOT,
    "manifest.txt"
)

with open(manifest_path, "w", encoding="utf-8") as f:

    f.write("RDD2022 India Small Dataset\n")
    f.write("===========================\n\n")

    f.write(f"Random seed: 42\n")
    f.write(f"Damaged images: {len(selected_damaged)}\n")
    f.write(f"Non-damaged images: {len(selected_non_damaged)}\n")
    f.write(f"Total images: {len(selected_images)}\n\n")

    f.write("Images:\n")

    for image in sorted(selected_images):
        f.write(image + "\n")


# ============================================================
# DONE
# ============================================================

print("\n========================================")
print("SMALL DATASET CREATED SUCCESSFULLY")
print("========================================")

print(f"\nImages copied : {copied}")
print(f"Missing XML   : {missing_xml}")

print("\nOutput:")
print(OUTPUT_ROOT)

print("\nStructure:")
print("small/")
print("└── India/")
print("    ├── images/")
print("    ├── annotations/")
print("    │   └── xmls/")
print("    └── manifest.txt")