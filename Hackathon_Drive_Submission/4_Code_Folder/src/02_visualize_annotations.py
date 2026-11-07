import os
import random
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# ============================================================
# RDD2022 INDIA - ANNOTATION VISUALIZATION
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

IMAGE_PATH = os.path.join(
    DATASET_PATH,
    "train",
    "images"
)

ANNOTATION_PATH = os.path.join(
    DATASET_PATH,
    "train",
    "annotations",
    "xmls"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "annotation_visualization"
)

os.makedirs(OUTPUT_PATH, exist_ok=True)

NUMBER_OF_IMAGES = 6


# ============================================================
# FIND XML FILES WITH DAMAGE
# ============================================================

xml_files = [
    file
    for file in os.listdir(ANNOTATION_PATH)
    if file.lower().endswith(".xml")
]

damage_xmls = []

for xml_file in xml_files:

    xml_path = os.path.join(
        ANNOTATION_PATH,
        xml_file
    )

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        if len(root.findall("object")) > 0:
            damage_xmls.append(xml_file)

    except Exception:
        pass


# ============================================================
# SELECT RANDOM IMAGES
# ============================================================

sample_size = min(
    NUMBER_OF_IMAGES,
    len(damage_xmls)
)

selected_xmls = random.sample(
    damage_xmls,
    sample_size
)


print("=" * 60)
print("RDD2022 ANNOTATION VISUALIZATION")
print("=" * 60)

print("Selected images:", sample_size)
print("Output folder:")
print(OUTPUT_PATH)


# ============================================================
# CREATE VISUALIZATIONS
# ============================================================

for xml_file in selected_xmls:

    xml_path = os.path.join(
        ANNOTATION_PATH,
        xml_file
    )

    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename_element = root.find("filename")

    if filename_element is None:
        continue

    filename = filename_element.text

    image_file = os.path.join(
        IMAGE_PATH,
        filename
    )

    if not os.path.exists(image_file):

        print("Image not found:", filename)
        continue

    image = Image.open(image_file)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.imshow(image)

    objects = root.findall("object")

    for obj in objects:

        name_element = obj.find("name")

        if name_element is None:
            continue

        class_name = name_element.text.strip()

        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        width = xmax - xmin
        height = ymax - ymin

        rectangle = patches.Rectangle(
            (xmin, ymin),
            width,
            height,
            fill=False,
            linewidth=2
        )

        ax.add_patch(rectangle)

        ax.text(
            xmin,
            max(ymin - 5, 0),
            class_name,
            fontsize=10,
            bbox=dict(
                alpha=0.7
            )
        )

    ax.set_title(
        f"RDD2022 - {filename}"
    )

    ax.axis("off")

    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_PATH,
        f"annotated_{filename.replace('.jpg', '.png')}"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output_file)


print("\nVisualization complete.")