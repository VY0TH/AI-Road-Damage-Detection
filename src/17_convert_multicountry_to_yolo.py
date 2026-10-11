import os
import shutil
import random
import xml.etree.ElementTree as ET

# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = r"C:\Users\Vyom Thakur\Desktop\AI hackathon\RoadDamageProject"

SOURCE_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "subsets",
    "multicountry",
    "RDD2022",
    "split"
)

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "subsets",
    "multicountry",
    "RDD2022",
    "yolo"
)

# ============================================================
# CLASS MAPPING
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
# CREATE FOLDERS
# ============================================================

for split in ["train", "val"]:

    os.makedirs(
        os.path.join(
            OUTPUT_ROOT,
            "images",
            split
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            OUTPUT_ROOT,
            "labels",
            split
        ),
        exist_ok=True
    )


# ============================================================
# CONVERT XML → YOLO
# ============================================================

def convert_split(split):

    image_dir = os.path.join(
        SOURCE_ROOT,
        split,
        "images"
    )

    xml_dir = os.path.join(
        SOURCE_ROOT,
        split,
        "annotations",
        "xmls"
    )

    output_image_dir = os.path.join(
        OUTPUT_ROOT,
        "images",
        split
    )

    output_label_dir = os.path.join(
        OUTPUT_ROOT,
        "labels",
        split
    )

    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        )
    ]

    converted = 0
    total_objects = 0
    class_counts = {
        class_name: 0
        for class_name in CLASSES
    }

    errors = 0

    print("\n" + "=" * 60)
    print(f"PROCESSING {split.upper()}")
    print("=" * 60)

    for image_file in image_files:

        image_path = os.path.join(
            image_dir,
            image_file
        )

        xml_file = (
            os.path.splitext(
                image_file
            )[0]
            + ".xml"
        )

        xml_path = os.path.join(
            xml_dir,
            xml_file
        )

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        shutil.copy2(
            image_path,
            os.path.join(
                output_image_dir,
                image_file
            )
        )

        # ----------------------------------------------------
        # If XML doesn't exist
        # ----------------------------------------------------

        if not os.path.exists(xml_path):

            print(
                f"WARNING: XML missing: "
                f"{xml_file}"
            )

            # Empty YOLO label
            label_path = os.path.join(
                output_label_dir,
                os.path.splitext(
                    image_file
                )[0] + ".txt"
            )

            open(
                label_path,
                "w"
            ).close()

            errors += 1
            continue

        try:

            tree = ET.parse(
                xml_path
            )

            root = tree.getroot()

            # ------------------------------------------------
            # Image dimensions
            # ------------------------------------------------

            size = root.find("size")

            if size is not None:

                width = int(
                    size.find(
                        "width"
                    ).text
                )

                height = int(
                    size.find(
                        "height"
                    ).text
                )

            else:

                from PIL import Image

                image = Image.open(
                    image_path
                )

                width, height = image.size

            # ------------------------------------------------
            # YOLO labels
            # ------------------------------------------------

            yolo_lines = []

            for obj in root.findall(
                "object"
            ):

                name = obj.find(
                    "name"
                )

                if name is None:
                    continue

                class_name = (
                    name.text.strip()
                )

                if class_name not in CLASS_TO_ID:

                    print(
                        f"WARNING: Unknown "
                        f"class {class_name}"
                    )

                    continue

                bbox = obj.find(
                    "bndbox"
                )

                if bbox is None:
                    continue

                xmin = float(
                    bbox.find(
                        "xmin"
                    ).text
                )

                ymin = float(
                    bbox.find(
                        "ymin"
                    ).text
                )

                xmax = float(
                    bbox.find(
                        "xmax"
                    ).text
                )

                ymax = float(
                    bbox.find(
                        "ymax"
                    ).text
                )

                # --------------------------------------------
                # Clamp bounding box
                # --------------------------------------------

                xmin = max(
                    0,
                    min(xmin, width)
                )

                ymin = max(
                    0,
                    min(ymin, height)
                )

                xmax = max(
                    0,
                    min(xmax, width)
                )

                ymax = max(
                    0,
                    min(ymax, height)
                )

                # --------------------------------------------
                # Reject invalid boxes
                # --------------------------------------------

                if (
                    xmax <= xmin
                    or
                    ymax <= ymin
                ):

                    continue

                # --------------------------------------------
                # Convert to YOLO format
                # --------------------------------------------

                x_center = (
                    (xmin + xmax)
                    / 2
                ) / width

                y_center = (
                    (ymin + ymax)
                    / 2
                ) / height

                box_width = (
                    xmax - xmin
                ) / width

                box_height = (
                    ymax - ymin
                ) / height

                class_id = CLASS_TO_ID[
                    class_name
                ]

                yolo_lines.append(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{box_width:.6f} "
                    f"{box_height:.6f}"
                )

                total_objects += 1

                class_counts[
                    class_name
                ] += 1

            # ------------------------------------------------
            # Save label
            # ------------------------------------------------

            label_path = os.path.join(
                output_label_dir,
                os.path.splitext(
                    image_file
                )[0] + ".txt"
            )

            with open(
                label_path,
                "w",
                encoding="utf-8"
            ) as f:

                if yolo_lines:

                    f.write(
                        "\n".join(
                            yolo_lines
                        )
                    )

            converted += 1

        except Exception as error:

            print(
                f"ERROR: {image_file}"
            )

            print(error)

            errors += 1

    # ========================================================
    # SPLIT SUMMARY
    # ========================================================

    print(
        f"\n{split.upper()} converted: "
        f"{converted}/{len(image_files)}"
    )

    print(
        f"{split.upper()} objects: "
        f"{total_objects}"
    )

    print(
        f"{split.upper()} errors: "
        f"{errors}"
    )

    print("\nClass distribution:")

    for class_name in CLASSES:

        print(
            f"{class_name:5} : "
            f"{class_counts[class_name]}"
        )

    return converted, total_objects


# ============================================================
# CONVERT TRAIN + VAL
# ============================================================

train_converted, train_objects = (
    convert_split("train")
)

val_converted, val_objects = (
    convert_split("val")
)


# ============================================================
# CREATE data.yaml
# ============================================================

data_yaml_path = os.path.join(
    OUTPUT_ROOT,
    "data.yaml"
)

with open(
    data_yaml_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "path: "
        + OUTPUT_ROOT.replace(
            "\\",
            "/"
        )
        + "\n"
    )

    f.write(
        "train: images/train\n"
    )

    f.write(
        "val: images/val\n"
    )

    f.write(
        f"nc: {len(CLASSES)}\n"
    )

    f.write(
        "names:\n"
    )

    for index, class_name in enumerate(
        CLASSES
    ):

        f.write(
            f"  {index}: "
            f"{class_name}\n"
        )


# ============================================================
# CREATE classes.txt
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
print("YOLO CONVERSION COMPLETE")
print("=" * 70)

print(
    f"\nTRAIN images: "
    f"{train_converted}"
)

print(
    f"VAL images  : "
    f"{val_converted}"
)

print(
    f"\nTRAIN objects: "
    f"{train_objects}"
)

print(
    f"VAL objects  : "
    f"{val_objects}"
)

print("\nClasses:")

for index, class_name in enumerate(
    CLASSES
):

    print(
        f"{index}: {class_name}"
    )

print("\ndata.yaml:")
print(data_yaml_path)

print("\nYOLO dataset:")
print(OUTPUT_ROOT)

print("\nDONE")