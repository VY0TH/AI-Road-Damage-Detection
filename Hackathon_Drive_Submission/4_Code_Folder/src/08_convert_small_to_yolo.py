import os
import shutil
import xml.etree.ElementTree as ET

PROJECT_ROOT = r"C:\Users\Vyom Thakur\Desktop\AI hackathon\RoadDamageProject"

SOURCE_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "subsets",
    "small",
    "India"
)

OUTPUT_ROOT = os.path.join(
    SOURCE_ROOT,
    "yolo"
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

CLASS_TO_ID = {
    name: i
    for i, name in enumerate(CLASSES)
}


def convert_split(split):

    source_images = os.path.join(
        SOURCE_ROOT,
        split,
        "images"
    )

    source_xml = os.path.join(
        SOURCE_ROOT,
        split,
        "annotations",
        "xmls"
    )

    output_images = os.path.join(
        OUTPUT_ROOT,
        "images",
        split
    )

    output_labels = os.path.join(
        OUTPUT_ROOT,
        "labels",
        split
    )

    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)

    image_files = [
        f for f in os.listdir(source_images)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    converted = 0
    total_objects = 0

    for image_file in image_files:

        image_path = os.path.join(
            source_images,
            image_file
        )

        image_name = os.path.splitext(
            image_file
        )[0]

        xml_path = os.path.join(
            source_xml,
            image_name + ".xml"
        )

        if not os.path.exists(xml_path):
            print(
                f"WARNING: XML missing for {image_file}"
            )
            continue

        # Copy image
        shutil.copy2(
            image_path,
            os.path.join(
                output_images,
                image_file
            )
        )

        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")

        width = float(
            size.find("width").text
        )

        height = float(
            size.find("height").text
        )

        yolo_lines = []

        for obj in root.findall("object"):

            class_name = obj.find(
                "name"
            ).text.strip()

            if class_name not in CLASS_TO_ID:
                print(
                    f"WARNING: Unknown class "
                    f"{class_name} in {xml_path}"
                )
                continue

            class_id = CLASS_TO_ID[class_name]

            bbox = obj.find("bndbox")

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

            # Convert VOC → YOLO
            x_center = (
                (xmin + xmax) / 2
            ) / width

            y_center = (
                (ymin + ymax) / 2
            ) / height

            box_width = (
                xmax - xmin
            ) / width

            box_height = (
                ymax - ymin
            ) / height

            yolo_lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}"
            )

            total_objects += 1

        label_path = os.path.join(
            output_labels,
            image_name + ".txt"
        )

        with open(
            label_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "\n".join(yolo_lines)
            )

        converted += 1

    print(
        f"{split.upper():5} converted: "
        f"{converted}/{len(image_files)}"
    )

    print(
        f"{split.upper():5} objects: "
        f"{total_objects}"
    )


# ============================================================
# CONVERT TRAIN + VAL
# ============================================================

print("=" * 60)
print("RDD2022 SMALL DATASET → YOLO")
print("=" * 60)

convert_split("train")
convert_split("val")


# ============================================================
# CLASSES.TXT
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
        f.write(class_name + "\n")


# ============================================================
# DATA.YAML
# ============================================================

yaml_path = os.path.join(
    OUTPUT_ROOT,
    "data.yaml"
)

with open(
    yaml_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"path: {OUTPUT_ROOT.replace(os.sep, '/')}\n"
    )

    f.write(
        "train: images/train\n"
    )

    f.write(
        "val: images/val\n\n"
    )

    f.write(
        "nc: 10\n"
    )

    f.write("names:\n")

    for i, class_name in enumerate(CLASSES):
        f.write(
            f"  {i}: {class_name}\n"
        )


print("\n" + "=" * 60)
print("YOLO CONVERSION COMPLETE")
print("=" * 60)

print("\nOutput:")
print(OUTPUT_ROOT)

print("\nClasses:")
for i, class_name in enumerate(CLASSES):
    print(f"{i}: {class_name}")