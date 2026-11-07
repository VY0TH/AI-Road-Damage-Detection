import os
import random
import shutil
import xml.etree.ElementTree as ET

PROJECT_ROOT = r"C:\Users\Vyom Thakur\Desktop\AI hackathon\RoadDamageProject"

SOURCE_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset", "subsets", "small", "India"
)

SOURCE_IMAGES = os.path.join(
    SOURCE_ROOT, "images"
)

SOURCE_XML = os.path.join(
    SOURCE_ROOT, "annotations", "xmls"
)

OUTPUT_ROOT = SOURCE_ROOT

TRAIN_IMAGES = os.path.join(
    OUTPUT_ROOT, "train", "images"
)

TRAIN_XML = os.path.join(
    OUTPUT_ROOT, "train", "annotations", "xmls"
)

VAL_IMAGES = os.path.join(
    OUTPUT_ROOT, "val", "images"
)

VAL_XML = os.path.join(
    OUTPUT_ROOT, "val", "annotations", "xmls"
)

random.seed(42)

os.makedirs(TRAIN_IMAGES, exist_ok=True)
os.makedirs(TRAIN_XML, exist_ok=True)
os.makedirs(VAL_IMAGES, exist_ok=True)
os.makedirs(VAL_XML, exist_ok=True)


# ------------------------------------------------------------
# Read all images
# ------------------------------------------------------------

images = [
    f for f in os.listdir(SOURCE_IMAGES)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Total images: {len(images)}")


# ------------------------------------------------------------
# Separate damaged / non-damaged
# ------------------------------------------------------------

damaged = []
non_damaged = []

for image in images:

    name = os.path.splitext(image)[0]
    xml_path = os.path.join(
        SOURCE_XML,
        name + ".xml"
    )

    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = root.findall("object")

    if len(objects) > 0:
        damaged.append(image)
    else:
        non_damaged.append(image)


print(f"Damaged images     : {len(damaged)}")
print(f"Non-damaged images : {len(non_damaged)}")


# ------------------------------------------------------------
# Shuffle
# ------------------------------------------------------------

random.shuffle(damaged)
random.shuffle(non_damaged)


# ------------------------------------------------------------
# 80/20 split
# ------------------------------------------------------------

def split_list(items):

    train_count = int(len(items) * 0.8)

    return (
        items[:train_count],
        items[train_count:]
    )


damaged_train, damaged_val = split_list(damaged)
normal_train, normal_val = split_list(non_damaged)

train_images = damaged_train + normal_train
val_images = damaged_val + normal_val

random.shuffle(train_images)
random.shuffle(val_images)


# ------------------------------------------------------------
# Copy files
# ------------------------------------------------------------

def copy_dataset(images, image_dest, xml_dest):

    for image in images:

        source_image = os.path.join(
            SOURCE_IMAGES,
            image
        )

        destination_image = os.path.join(
            image_dest,
            image
        )

        shutil.copy2(
            source_image,
            destination_image
        )

        xml_name = os.path.splitext(image)[0] + ".xml"

        source_xml = os.path.join(
            SOURCE_XML,
            xml_name
        )

        destination_xml = os.path.join(
            xml_dest,
            xml_name
        )

        shutil.copy2(
            source_xml,
            destination_xml
        )


copy_dataset(
    train_images,
    TRAIN_IMAGES,
    TRAIN_XML
)

copy_dataset(
    val_images,
    VAL_IMAGES,
    VAL_XML
)


# ------------------------------------------------------------
# Save split manifest
# ------------------------------------------------------------

manifest = os.path.join(
    OUTPUT_ROOT,
    "split_manifest.txt"
)

with open(manifest, "w", encoding="utf-8") as f:

    f.write("RDD2022 India Small Dataset\n")
    f.write("===========================\n\n")

    f.write("Random seed: 42\n")
    f.write("Split: 80/20\n\n")

    f.write(
        f"Training images: {len(train_images)}\n"
    )

    f.write(
        f"Validation images: {len(val_images)}\n\n"
    )

    f.write("TRAIN:\n")

    for image in sorted(train_images):
        f.write(image + "\n")

    f.write("\nVALIDATION:\n")

    for image in sorted(val_images):
        f.write(image + "\n")


# ------------------------------------------------------------
# Final output
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("80/20 SPLIT COMPLETE")
print("=" * 50)

print(f"Training images   : {len(train_images)}")
print(f"Validation images : {len(val_images)}")

print("\nTraining:")
print(TRAIN_IMAGES)

print("\nValidation:")
print(VAL_IMAGES)

print("\nDone.")