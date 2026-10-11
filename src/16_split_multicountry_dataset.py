import os
import random
import shutil

# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = r"C:\Users\Vyom Thakur\Desktop\AI hackathon\RoadDamageProject"

SOURCE_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "subsets",
    "multicountry",
    "RDD2022"
)

SOURCE_IMAGES = os.path.join(
    SOURCE_ROOT,
    "images"
)

SOURCE_XML = os.path.join(
    SOURCE_ROOT,
    "annotations",
    "xmls"
)

OUTPUT_ROOT = os.path.join(
    SOURCE_ROOT,
    "split"
)

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

TRAIN_RATIO = 0.80

COUNTRIES = [
    "India",
    "Japan",
    "China",
    "Czech",
    "Norway",
    "United_States"
]


# ============================================================
# CLEAN PREVIOUS SPLIT
# ============================================================

if os.path.exists(OUTPUT_ROOT):

    print("Removing previous split...")

    shutil.rmtree(
        OUTPUT_ROOT
    )


# ============================================================
# CREATE FOLDERS
# ============================================================

for split in ["train", "val"]:

    os.makedirs(
        os.path.join(
            OUTPUT_ROOT,
            split,
            "images"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            OUTPUT_ROOT,
            split,
            "annotations",
            "xmls"
        ),
        exist_ok=True
    )


# ============================================================
# GET ALL IMAGES
# ============================================================

all_images = [
    f
    for f in os.listdir(SOURCE_IMAGES)
    if f.lower().endswith(
        (
            ".jpg",
            ".jpeg",
            ".png"
        )
    )
]


print("=" * 70)
print("RDD2022 MULTI-COUNTRY 80/20 SPLIT")
print("=" * 70)

print(
    f"\nTotal images found: "
    f"{len(all_images)}"
)


# ============================================================
# SPLIT COUNTRY BY COUNTRY
# ============================================================

total_train = 0
total_val = 0


for country in COUNTRIES:

    print("\n" + "-" * 60)
    print(f"Processing: {country}")
    print("-" * 60)

    # --------------------------------------------------------
    # Find images belonging to country
    # --------------------------------------------------------

    country_images = [
        f
        for f in all_images
        if f.startswith(
            country + "_"
        )
    ]

    print(
        f"Total: {len(country_images)}"
    )

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    random.shuffle(
        country_images
    )

    # --------------------------------------------------------
    # 80/20
    # --------------------------------------------------------

    train_count = int(
        len(country_images)
        * TRAIN_RATIO
    )

    train_images = (
        country_images[:train_count]
    )

    val_images = (
        country_images[train_count:]
    )

    print(
        f"Train: {len(train_images)}"
    )

    print(
        f"Val  : {len(val_images)}"
    )

    # --------------------------------------------------------
    # Copy TRAIN
    # --------------------------------------------------------

    for image_file in train_images:

        source_image = os.path.join(
            SOURCE_IMAGES,
            image_file
        )

        destination_image = os.path.join(
            OUTPUT_ROOT,
            "train",
            "images",
            image_file
        )

        shutil.copy2(
            source_image,
            destination_image
        )

        # XML
        xml_file = (
            os.path.splitext(
                image_file
            )[0]
            + ".xml"
        )

        source_xml = os.path.join(
            SOURCE_XML,
            xml_file
        )

        destination_xml = os.path.join(
            OUTPUT_ROOT,
            "train",
            "annotations",
            "xmls",
            xml_file
        )

        if os.path.exists(
            source_xml
        ):

            shutil.copy2(
                source_xml,
                destination_xml
            )

    # --------------------------------------------------------
    # Copy VALIDATION
    # --------------------------------------------------------

    for image_file in val_images:

        source_image = os.path.join(
            SOURCE_IMAGES,
            image_file
        )

        destination_image = os.path.join(
            OUTPUT_ROOT,
            "val",
            "images",
            image_file
        )

        shutil.copy2(
            source_image,
            destination_image
        )

        # XML
        xml_file = (
            os.path.splitext(
                image_file
            )[0]
            + ".xml"
        )

        source_xml = os.path.join(
            SOURCE_XML,
            xml_file
        )

        destination_xml = os.path.join(
            OUTPUT_ROOT,
            "val",
            "annotations",
            "xmls",
            xml_file
        )

        if os.path.exists(
            source_xml
        ):

            shutil.copy2(
                source_xml,
                destination_xml
            )

    total_train += len(
        train_images
    )

    total_val += len(
        val_images
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("80/20 SPLIT COMPLETE")
print("=" * 70)

print(
    f"\nTotal images : "
    f"{len(all_images)}"
)

print(
    f"Train        : "
    f"{total_train}"
)

print(
    f"Validation   : "
    f"{total_val}"
)

print(
    f"\nTrain ratio  : "
    f"{total_train / len(all_images):.2%}"
)

print(
    f"Val ratio    : "
    f"{total_val / len(all_images):.2%}"
)

print("\nOutput:")

print(
    os.path.join(
        OUTPUT_ROOT,
        "train"
    )
)

print(
    os.path.join(
        OUTPUT_ROOT,
        "val"
    )
)

print("\nDone.")