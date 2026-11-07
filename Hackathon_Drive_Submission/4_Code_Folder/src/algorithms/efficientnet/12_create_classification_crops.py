import os
import cv2
import xml.etree.ElementTree as ET
from collections import Counter

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

OUTPUT_ROOT = os.path.join(
    DATASET_ROOT,
    "classification"
)

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
# CLAHE
# ============================================================

def apply_clahe(image):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    enhanced = cv2.merge(
        (l, a, b)
    )

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR
    )

    return enhanced


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split):

    image_dir = os.path.join(
        DATASET_ROOT,
        split,
        "images"
    )

    xml_dir = os.path.join(
        DATASET_ROOT,
        split,
        "annotations",
        "xmls"
    )

    print("\n" + "=" * 60)
    print(f"PROCESSING {split.upper()}")
    print("=" * 60)

    counters = Counter()

    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    for image_file in image_files:

        image_path = os.path.join(
            image_dir,
            image_file
        )

        xml_name = (
            os.path.splitext(image_file)[0]
            + ".xml"
        )

        xml_path = os.path.join(
            xml_dir,
            xml_name
        )

        if not os.path.exists(xml_path):
            print(
                f"WARNING: XML missing: {image_file}"
            )
            continue

        image = cv2.imread(image_path)

        if image is None:
            print(
                f"WARNING: Could not read: {image_file}"
            )
            continue

        height, width = image.shape[:2]

        tree = ET.parse(xml_path)
        root = tree.getroot()

        objects = root.findall("object")

        object_index = 0

        for obj in objects:

            class_element = obj.find("name")

            if class_element is None:
                continue

            class_name = class_element.text.strip()

            if class_name not in CLASSES:
                print(
                    f"WARNING: Unknown class "
                    f"{class_name}"
                )
                continue

            bbox = obj.find("bndbox")

            if bbox is None:
                continue

            xmin = int(
                float(
                    bbox.find("xmin").text
                )
            )

            ymin = int(
                float(
                    bbox.find("ymin").text
                )
            )

            xmax = int(
                float(
                    bbox.find("xmax").text
                )
            )

            ymax = int(
                float(
                    bbox.find("ymax").text
                )
            )

            # Clamp coordinates
            xmin = max(0, min(xmin, width - 1))
            ymin = max(0, min(ymin, height - 1))
            xmax = max(0, min(xmax, width))
            ymax = max(0, min(ymax, height))

            # Ignore invalid boxes
            if xmin >= xmax or ymin >= ymax:
                continue

            # Crop damage
            crop = image[
                ymin:ymax,
                xmin:xmax
            ]

            if crop.size == 0:
                continue

            # Apply CLAHE
            crop = apply_clahe(crop)

            # Create class folder
            class_dir = os.path.join(
                OUTPUT_ROOT,
                split,
                class_name
            )

            os.makedirs(
                class_dir,
                exist_ok=True
            )

            # Unique filename
            base_name = os.path.splitext(
                image_file
            )[0]

            crop_name = (
                f"{base_name}"
                f"_object_{object_index}.jpg"
            )

            output_path = os.path.join(
                class_dir,
                crop_name
            )

            cv2.imwrite(
                output_path,
                crop
            )

            counters[class_name] += 1

            object_index += 1

    # --------------------------------------------------------
    # Print class distribution
    # --------------------------------------------------------

    print("\nClass distribution:")

    total = 0

    for class_name in CLASSES:

        count = counters[class_name]

        print(
            f"{class_name:5} : {count}"
        )

        total += count

    print(
        f"\nTotal classification crops: {total}"
    )


# ============================================================
# TRAIN + VALIDATION
# ============================================================

process_split("train")
process_split("val")


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION DATASET CREATED")
print("=" * 60)

print("\nOutput:")
print(OUTPUT_ROOT)