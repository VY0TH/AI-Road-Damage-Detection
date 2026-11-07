import os
import shutil
import random
import xml.etree.ElementTree as ET
from PIL import Image


# ============================================================
# RDD2022 XML -> YOLO FORMAT CONVERTER
# 80/20 TRAIN-VALIDATION SPLIT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# ============================================================
# RAW DATASET
# ============================================================

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "raw",
    "India",
    "India"
)

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "yolo"
)


# ============================================================
# RDD2022 CLASSES
# ============================================================

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

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASSES)
}


# ============================================================
# RAW DATA PATHS
# ============================================================

TRAIN_IMAGE_SRC = os.path.join(
    DATASET_ROOT,
    "train",
    "images"
)

TRAIN_XML_SRC = os.path.join(
    DATASET_ROOT,
    "train",
    "annotations",
    "xmls"
)

TEST_IMAGE_SRC = os.path.join(
    DATASET_ROOT,
    "test",
    "images"
)

TEST_XML_SRC = os.path.join(
    DATASET_ROOT,
    "test",
    "annotations",
    "xmls"
)


# ============================================================
# YOLO OUTPUT PATHS
# ============================================================

TRAIN_IMAGE_OUT = os.path.join(
    OUTPUT_ROOT,
    "images",
    "train"
)

VAL_IMAGE_OUT = os.path.join(
    OUTPUT_ROOT,
    "images",
    "val"
)

TEST_IMAGE_OUT = os.path.join(
    OUTPUT_ROOT,
    "images",
    "test"
)

TRAIN_LABEL_OUT = os.path.join(
    OUTPUT_ROOT,
    "labels",
    "train"
)

VAL_LABEL_OUT = os.path.join(
    OUTPUT_ROOT,
    "labels",
    "val"
)

TEST_LABEL_OUT = os.path.join(
    OUTPUT_ROOT,
    "labels",
    "test"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for folder in [
    TRAIN_IMAGE_OUT,
    VAL_IMAGE_OUT,
    TEST_IMAGE_OUT,
    TRAIN_LABEL_OUT,
    VAL_LABEL_OUT,
    TEST_LABEL_OUT
]:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# XML -> YOLO CONVERSION
# ============================================================

def convert_xml_to_yolo(
    xml_path,
    image_width,
    image_height
):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    yolo_lines = []

    for obj in root.findall("object"):

        name_element = obj.find("name")

        if name_element is None:
            continue

        class_name = name_element.text.strip()

        if class_name not in CLASS_TO_ID:

            print(
                f"WARNING: Unknown class "
                f"{class_name} in {xml_path}"
            )

            continue

        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        xmin = float(
            bbox.find("xmin").text
        )

        ymin = float(
            bbox.find("ymin").text
        )

        xmax = float(
            bbox.find("xmax").text
        )

        ymax = float(
            bbox.find("ymax").text
        )

        # ----------------------------------------------------
        # CLAMP BOUNDING BOX
        # ----------------------------------------------------

        xmin = max(
            0,
            min(xmin, image_width)
        )

        xmax = max(
            0,
            min(xmax, image_width)
        )

        ymin = max(
            0,
            min(ymin, image_height)
        )

        ymax = max(
            0,
            min(ymax, image_height)
        )

        # ----------------------------------------------------
        # IGNORE INVALID BOXES
        # ----------------------------------------------------

        if xmax <= xmin or ymax <= ymin:
            continue

        # ----------------------------------------------------
        # PASCAL VOC -> YOLO
        # ----------------------------------------------------

        x_center = (
            (xmin + xmax) / 2
        ) / image_width

        y_center = (
            (ymin + ymax) / 2
        ) / image_height

        width = (
            xmax - xmin
        ) / image_width

        height = (
            ymax - ymin
        ) / image_height

        class_id = CLASS_TO_ID[class_name]

        yolo_lines.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    return yolo_lines


# ============================================================
# PROCESS TRAIN / VALIDATION IMAGE
# ============================================================

def process_annotated_image(
    image_filename,
    image_source,
    xml_source,
    image_destination,
    label_destination
):

    image_path = os.path.join(
        image_source,
        image_filename
    )

    xml_filename = (
        os.path.splitext(image_filename)[0]
        + ".xml"
    )

    xml_path = os.path.join(
        xml_source,
        xml_filename
    )

    # --------------------------------------------------------
    # XML MUST EXIST FOR TRAIN/VAL
    # --------------------------------------------------------

    if not os.path.exists(xml_path):

        print(
            f"WARNING: XML missing for "
            f"{image_filename}"
        )

        return False

    # --------------------------------------------------------
    # GET IMAGE SIZE
    # --------------------------------------------------------

    try:

        with Image.open(image_path) as img:

            image_width, image_height = img.size

    except Exception as e:

        print(
            f"ERROR reading {image_filename}: {e}"
        )

        return False

    # --------------------------------------------------------
    # CONVERT XML
    # --------------------------------------------------------

    yolo_lines = convert_xml_to_yolo(
        xml_path,
        image_width,
        image_height
    )

    # --------------------------------------------------------
    # COPY IMAGE
    # --------------------------------------------------------

    destination_image = os.path.join(
        image_destination,
        image_filename
    )

    shutil.copy2(
        image_path,
        destination_image
    )

    # --------------------------------------------------------
    # CREATE YOLO LABEL
    # --------------------------------------------------------

    label_filename = (
        os.path.splitext(image_filename)[0]
        + ".txt"
    )

    destination_label = os.path.join(
        label_destination,
        label_filename
    )

    with open(
        destination_label,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(yolo_lines)
        )

    return True


# ============================================================
# START
# ============================================================

print("=" * 70)
print("RDD2022 XML -> YOLO DATASET CONVERSION")
print("=" * 70)

print("\nDataset:")
print(DATASET_ROOT)


# ============================================================
# SHOW CLASSES
# ============================================================

print("\nClasses:")

for class_name, class_id in CLASS_TO_ID.items():

    print(
        f"{class_id}: {class_name}"
    )


# ============================================================
# GET TRAINING IMAGES
# ============================================================

train_images = [
    f
    for f in os.listdir(TRAIN_IMAGE_SRC)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]

print(
    f"\nTraining images found: "
    f"{len(train_images)}"
)


# ============================================================
# 80 / 20 SPLIT
# ============================================================

random.seed(42)

random.shuffle(train_images)

split_index = int(
    len(train_images) * 0.80
)

train_files = train_images[
    :split_index
]

val_files = train_images[
    split_index:
]

print(
    f"Training split:   "
    f"{len(train_files)}"
)

print(
    f"Validation split: "
    f"{len(val_files)}"
)


# ============================================================
# TRAIN
# ============================================================

print("\n[1] Converting TRAIN data...")

train_success = 0

for filename in train_files:

    success = process_annotated_image(
        filename,
        TRAIN_IMAGE_SRC,
        TRAIN_XML_SRC,
        TRAIN_IMAGE_OUT,
        TRAIN_LABEL_OUT
    )

    if success:
        train_success += 1


print(
    f"TRAIN converted: "
    f"{train_success}/{len(train_files)}"
)


# ============================================================
# VALIDATION
# ============================================================

print(
    "\n[2] Converting VALIDATION data..."
)

val_success = 0

for filename in val_files:

    success = process_annotated_image(
        filename,
        TRAIN_IMAGE_SRC,
        TRAIN_XML_SRC,
        VAL_IMAGE_OUT,
        VAL_LABEL_OUT
    )

    if success:
        val_success += 1


print(
    f"VAL converted: "
    f"{val_success}/{len(val_files)}"
)


# ============================================================
# TEST DATA
#
# Test images may NOT have XML annotations.
# Therefore we simply copy the images.
# ============================================================

test_images = []

if os.path.exists(TEST_IMAGE_SRC):

    test_images = [
        f
        for f in os.listdir(TEST_IMAGE_SRC)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

print(
    f"\nTest images found: "
    f"{len(test_images)}"
)


print("\n[3] Copying TEST images...")

test_success = 0

for filename in test_images:

    source_image = os.path.join(
        TEST_IMAGE_SRC,
        filename
    )

    destination_image = os.path.join(
        TEST_IMAGE_OUT,
        filename
    )

    shutil.copy2(
        source_image,
        destination_image
    )

    test_success += 1


print(
    f"TEST images copied: "
    f"{test_success}/{len(test_images)}"
)

print(
    "\nNOTE: Test XML annotations are "
    "not required for this conversion."
)


# ============================================================
# CREATE DATA.YAML
# ============================================================

yaml_path = os.path.join(
    OUTPUT_ROOT,
    "data.yaml"
)

# Use forward slashes for YAML
yaml_root = OUTPUT_ROOT.replace(
    "\\",
    "/"
)

with open(
    yaml_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"path: {yaml_root}\n"
    )

    f.write(
        "train: images/train\n"
    )

    f.write(
        "val: images/val\n"
    )

    f.write(
        "test: images/test\n\n"
    )

    f.write(
        f"nc: {len(CLASSES)}\n\n"
    )

    f.write("names:\n")

    for class_id, class_name in enumerate(CLASSES):

        f.write(
            f"  {class_id}: {class_name}\n"
        )


# ============================================================
# CREATE CLASSES.TXT
# ============================================================

classes_path = os.path.join(
    OUTPUT_ROOT,
    "classes.txt"
)

with open(
    classes_path,
    "w",
    encoding="utf-8"
) as f:

    for class_name in CLASSES:

        f.write(
            class_name + "\n"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CONVERSION COMPLETE")
print("=" * 70)

print(
    "\nYOLO dataset created at:"
)

print(
    OUTPUT_ROOT
)

print("\nDataset structure:")

print(
    "  dataset/yolo/images/train"
)

print(
    "  dataset/yolo/images/val"
)

print(
    "  dataset/yolo/images/test"
)

print(
    "  dataset/yolo/labels/train"
)

print(
    "  dataset/yolo/labels/val"
)

print(
    "  dataset/yolo/labels/test"
)

print(
    "\n  dataset/yolo/data.yaml"
)

print(
    "  dataset/yolo/classes.txt"
)

print("\n" + "=" * 70)