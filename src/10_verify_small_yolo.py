import os
import random
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = r"C:\Users\Vyom Thakur\Desktop\AI hackathon\RoadDamageProject"

YOLO_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset", "subsets", "small", "India", "yolo"
)

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "results", "small_yolo_verification"
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

random.seed(42)

os.makedirs(OUTPUT_ROOT, exist_ok=True)


def verify_split(split, number_of_images=10):

    image_dir = os.path.join(
        YOLO_ROOT,
        "images",
        split
    )

    label_dir = os.path.join(
        YOLO_ROOT,
        "labels",
        split
    )

    output_dir = os.path.join(
        OUTPUT_ROOT,
        split
    )

    os.makedirs(output_dir, exist_ok=True)

    images = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    selected = random.sample(
        images,
        min(number_of_images, len(images))
    )

    print(f"\n{split.upper()}")
    print("-" * 40)

    for image_file in selected:

        image_path = os.path.join(
            image_dir,
            image_file
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        draw = ImageDraw.Draw(image)

        width, height = image.size

        label_file = (
            os.path.splitext(image_file)[0]
            + ".txt"
        )

        label_path = os.path.join(
            label_dir,
            label_file
        )

        object_count = 0

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
                    continue

                class_id = int(parts[0])

                x_center = float(parts[1]) * width
                y_center = float(parts[2]) * height
                box_width = float(parts[3]) * width
                box_height = float(parts[4]) * height

                xmin = x_center - box_width / 2
                ymin = y_center - box_height / 2
                xmax = x_center + box_width / 2
                ymax = y_center + box_height / 2

                draw.rectangle(
                    [xmin, ymin, xmax, ymax],
                    outline="red",
                    width=3
                )

                label = (
                    CLASSES[class_id]
                    if 0 <= class_id < len(CLASSES)
                    else str(class_id)
                )

                draw.text(
                    (xmin, max(0, ymin - 20)),
                    label,
                    fill="red"
                )

                object_count += 1

        output_path = os.path.join(
            output_dir,
            "verified_" + image_file
        )

        image.save(output_path)

        print(
            f"{image_file} -> "
            f"{object_count} objects"
        )


# Verify both sets
verify_split("train", 10)
verify_split("val", 10)

print("\n" + "=" * 60)
print("YOLO VISUAL VERIFICATION COMPLETE")
print("=" * 60)

print("\nResults:")
print(OUTPUT_ROOT)