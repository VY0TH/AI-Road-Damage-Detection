import os
import random
import cv2

PROJECT_ROOT = r"C:\Users\Vyom Thakur\Desktop\AI hackathon\RoadDamageProject"

YOLO_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "subsets",
    "multicountry",
    "RDD2022",
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
    "multicountry_yolo_verification"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
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

# ------------------------------------------------------------
# SELECT RANDOM IMAGES
# ------------------------------------------------------------

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]

random.seed(42)

sample_size = min(
    12,
    len(image_files)
)

samples = random.sample(
    image_files,
    sample_size
)

print("=" * 70)
print("MULTI-COUNTRY YOLO VERIFICATION")
print("=" * 70)

print(
    f"\nChecking {sample_size} random images..."
)


# ------------------------------------------------------------
# PROCESS
# ------------------------------------------------------------

for index, image_file in enumerate(
    samples,
    start=1
):

    image_path = os.path.join(
        IMAGE_DIR,
        image_file
    )

    label_file = (
        os.path.splitext(
            image_file
        )[0]
        + ".txt"
    )

    label_path = os.path.join(
        LABEL_DIR,
        label_file
    )

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            f"ERROR: Could not read "
            f"{image_file}"
        )

        continue

    height, width = image.shape[:2]

    object_count = 0

    # --------------------------------------------------------
    # Read YOLO labels
    # --------------------------------------------------------

    if os.path.exists(label_path):

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:

            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        for line in lines:

            parts = line.split()

            if len(parts) != 5:

                print(
                    f"WARNING: Bad label "
                    f"in {label_file}"
                )

                continue

            class_id = int(
                parts[0]
            )

            x_center = float(
                parts[1]
            )

            y_center = float(
                parts[2]
            )

            box_width = float(
                parts[3]
            )

            box_height = float(
                parts[4]
            )

            # ------------------------------------------------
            # Convert normalized YOLO coordinates to pixels
            # ------------------------------------------------

            x_center *= width
            y_center *= height

            box_width *= width
            box_height *= height

            xmin = int(
                x_center - box_width / 2
            )

            ymin = int(
                y_center - box_height / 2
            )

            xmax = int(
                x_center + box_width / 2
            )

            ymax = int(
                y_center + box_height / 2
            )

            # Clamp
            xmin = max(
                0,
                min(xmin, width - 1)
            )

            ymin = max(
                0,
                min(ymin, height - 1)
            )

            xmax = max(
                0,
                min(xmax, width - 1)
            )

            ymax = max(
                0,
                min(ymax, height - 1)
            )

            # ------------------------------------------------
            # Class name
            # ------------------------------------------------

            if (
                0 <= class_id
                < len(CLASSES)
            ):

                class_name = CLASSES[
                    class_id
                ]

            else:

                class_name = (
                    f"UNKNOWN_{class_id}"
                )

            # ------------------------------------------------
            # Draw box
            # ------------------------------------------------

            cv2.rectangle(
                image,
                (xmin, ymin),
                (xmax, ymax),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                class_name,
                (
                    xmin,
                    max(
                        20,
                        ymin - 5
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            object_count += 1

    # --------------------------------------------------------
    # Add image information
    # --------------------------------------------------------

    cv2.putText(
        image,
        image_file,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        image,
        f"Objects: {object_count}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"sample_{index:02d}_{image_file}"
    )

    cv2.imwrite(
        output_path,
        image
    )

    print(
        f"{index:02d}. "
        f"{image_file} "
        f"-> {object_count} objects"
    )


print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)

print("\nSaved images:")
print(OUTPUT_DIR)

print("\nOpen that folder and check the boxes visually.")