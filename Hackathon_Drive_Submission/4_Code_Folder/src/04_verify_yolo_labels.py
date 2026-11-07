import os
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# ============================================================
# YOLO LABEL VERIFICATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

YOLO_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "yolo"
)

IMAGE_DIR = os.path.join(
    YOLO_ROOT,
    "images",
    "train"
)

LABEL_DIR = os.path.join(
    YOLO_ROOT,
    "labels",
    "train"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "yolo_verification"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CLASSES
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


# ============================================================
# GET IMAGES
# ============================================================

images = [
    f
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]


random.seed(42)

selected_images = random.sample(
    images,
    min(6, len(images))
)


print("=" * 60)
print("YOLO LABEL VERIFICATION")
print("=" * 60)

print(
    f"Training images available: {len(images)}"
)

print(
    f"Checking: {len(selected_images)} images"
)


# ============================================================
# PROCESS IMAGES
# ============================================================

for image_name in selected_images:

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    label_name = (
        os.path.splitext(image_name)[0]
        + ".txt"
    )

    label_path = os.path.join(
        LABEL_DIR,
        label_name
    )

    if not os.path.exists(label_path):

        print(
            f"WARNING: Label missing for "
            f"{image_name}"
        )

        continue


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = Image.open(image_path)

    width, height = image.size


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    ax.imshow(image)


    # --------------------------------------------------------
    # READ YOLO LABEL
    # --------------------------------------------------------

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        lines = f.readlines()


    object_count = 0


    for line in lines:

        line = line.strip()

        if not line:
            continue


        parts = line.split()

        if len(parts) != 5:

            print(
                f"Invalid label in "
                f"{label_name}: {line}"
            )

            continue


        class_id = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])


        # ----------------------------------------------------
        # YOLO NORMALIZED -> PIXELS
        # ----------------------------------------------------

        x_center *= width
        y_center *= height

        box_width *= width
        box_height *= height


        xmin = (
            x_center -
            box_width / 2
        )

        ymin = (
            y_center -
            box_height / 2
        )


        # ----------------------------------------------------
        # DRAW BOX
        # ----------------------------------------------------

        rectangle = patches.Rectangle(
            (xmin, ymin),
            box_width,
            box_height,
            fill=False,
            linewidth=2
        )

        ax.add_patch(rectangle)


        # ----------------------------------------------------
        # CLASS LABEL
        # ----------------------------------------------------

        if 0 <= class_id < len(CLASSES):

            class_name = CLASSES[class_id]

        else:

            class_name = f"UNKNOWN-{class_id}"


        ax.text(
            xmin,
            max(0, ymin - 5),
            class_name,
            fontsize=10,
            bbox=dict(
                alpha=0.7
            )
        )


        object_count += 1


    ax.set_title(
        f"{image_name} | Objects: {object_count}"
    )

    ax.axis("off")

    plt.tight_layout()


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_name = (
        "yolo_" +
        os.path.splitext(image_name)[0] +
        ".png"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


    print(
        f"Saved: {output_name} "
        f"({object_count} objects)"
    )


print("\n" + "=" * 60)
print("YOLO VERIFICATION COMPLETE")
print("=" * 60)

print(
    "\nOpen:"
)

print(
    OUTPUT_DIR
)