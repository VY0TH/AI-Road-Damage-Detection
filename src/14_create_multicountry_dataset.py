import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict

# ============================================================
# PROJECT SETTINGS
# ============================================================

PROJECT_ROOT = r"C:\Users\Vyom Thakur\Desktop\AI hackathon\RoadDamageProject"

SOURCE_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "raw",
    "RoadDamageProjectdatasetrawRDD2022_FULL",
    "RDD2022"
)

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "subsets",
    "multicountry",
    "RDD2022"
)

OUTPUT_IMAGES = os.path.join(
    OUTPUT_ROOT,
    "images"
)

OUTPUT_XML = os.path.join(
    OUTPUT_ROOT,
    "annotations",
    "xmls"
)

# ============================================================
# DATASET SIZE
# ============================================================

IMAGES_PER_COUNTRY = 200

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ============================================================
# COUNTRY GROUPS
# ============================================================

COUNTRIES = {
    "India": [
        "India"
    ],

    "Japan": [
        "Japan"
    ],

    "Czech": [
        "Czech"
    ],

    "Norway": [
        "Norway"
    ],

    "United_States": [
        "United_States"
    ],

    # China combines both RDD2022 China subsets
    "China": [
        "China_Drone",
        "China_MotorBike"
    ]
}

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


# ============================================================
# CLEAN PREVIOUS OUTPUT
# ============================================================

if os.path.exists(OUTPUT_ROOT):

    print("=" * 70)
    print("REMOVING PREVIOUS INCOMPLETE DATASET")
    print("=" * 70)

    print(
        f"\nRemoving:\n{OUTPUT_ROOT}"
    )

    shutil.rmtree(
        OUTPUT_ROOT
    )

# Create fresh folders
os.makedirs(
    OUTPUT_IMAGES,
    exist_ok=True
)

os.makedirs(
    OUTPUT_XML,
    exist_ok=True
)


# ============================================================
# FIND TRAIN IMAGE + XML PAIRS
# ============================================================

def find_pairs(country_folders):

    pairs = []

    for folder_name in country_folders:

        country_root = os.path.join(
            SOURCE_ROOT,
            folder_name
        )

        print(
            f"\nSearching: {folder_name}"
        )

        if not os.path.exists(country_root):

            print(
                f"WARNING: Folder not found:"
            )

            print(country_root)

            continue

        image_dirs = []
        xml_dirs = []

        # ----------------------------------------------------
        # Find TRAIN/images only
        # ----------------------------------------------------

        for root, dirs, files in os.walk(
            country_root
        ):

            folder_name_lower = (
                os.path.basename(root).lower()
            )

            parent_name_lower = (
                os.path.basename(
                    os.path.dirname(root)
                ).lower()
            )

            if (
                folder_name_lower == "images"
                and
                parent_name_lower == "train"
            ):

                image_dirs.append(root)

        # ----------------------------------------------------
        # Find XML annotation folders
        # ----------------------------------------------------

        for root, dirs, files in os.walk(
            country_root
        ):

            if (
                os.path.basename(root).lower()
                == "xmls"
            ):

                xml_dirs.append(root)

        # ----------------------------------------------------
        # Build XML lookup
        # ----------------------------------------------------

        xml_lookup = {}

        for xml_dir in xml_dirs:

            for filename in os.listdir(
                xml_dir
            ):

                if filename.lower().endswith(
                    ".xml"
                ):

                    base_name = os.path.splitext(
                        filename
                    )[0]

                    xml_lookup[base_name] = os.path.join(
                        xml_dir,
                        filename
                    )

        # ----------------------------------------------------
        # Match TRAIN images with XML
        # ----------------------------------------------------

        for image_dir in image_dirs:

            for filename in os.listdir(
                image_dir
            ):

                if not filename.lower().endswith(
                    (
                        ".jpg",
                        ".jpeg",
                        ".png"
                    )
                ):

                    continue

                base_name = os.path.splitext(
                    filename
                )[0]

                if base_name in xml_lookup:

                    image_path = os.path.join(
                        image_dir,
                        filename
                    )

                    xml_path = xml_lookup[
                        base_name
                    ]

                    pairs.append(
                        (
                            image_path,
                            xml_path
                        )
                    )

    return pairs


# ============================================================
# READ DAMAGE CLASSES FROM XML
# ============================================================

def get_classes(xml_path):

    classes = set()

    try:

        tree = ET.parse(
            xml_path
        )

        root = tree.getroot()

        objects = root.findall(
            "object"
        )

        for obj in objects:

            name_element = obj.find(
                "name"
            )

            if name_element is None:
                continue

            class_name = (
                name_element.text.strip()
            )

            if class_name in CLASSES:

                classes.add(
                    class_name
                )

    except Exception as error:

        print(
            f"WARNING: XML error:"
        )

        print(xml_path)

        print(error)

    return classes


# ============================================================
# MAIN DATASET CREATION
# ============================================================

print("\n" + "=" * 70)
print("RDD2022 MULTI-COUNTRY DATASET")
print("=" * 70)

print("\nSource:")
print(SOURCE_ROOT)

print("\nDataset configuration:")
print("Countries/groups : 6")
print("Images per group : 200")
print("Total images     : 1200")

print("\nChina = China_Drone + China_MotorBike")


# Store everything selected
all_selected = []

# Store statistics
country_statistics = {}


# ============================================================
# PROCESS EACH COUNTRY
# ============================================================

for country, folders in COUNTRIES.items():

    print("\n" + "=" * 70)
    print(f"PROCESSING: {country}")
    print("=" * 70)

    # --------------------------------------------------------
    # Find pairs
    # --------------------------------------------------------

    pairs = find_pairs(
        folders
    )

    print(
        f"\nMatched image/XML pairs: "
        f"{len(pairs)}"
    )

    if len(pairs) < IMAGES_PER_COUNTRY:

        print(
            f"WARNING: Only "
            f"{len(pairs)} matched images "
            f"available for {country}."
        )

    if len(pairs) == 0:

        print(
            f"ERROR: No data found for "
            f"{country}."
        )

        continue

    # --------------------------------------------------------
    # Separate damaged and non-damaged
    # --------------------------------------------------------

    damaged = []
    non_damaged = []

    for image_path, xml_path in pairs:

        classes = get_classes(
            xml_path
        )

        item = (
            image_path,
            xml_path,
            classes
        )

        if len(classes) > 0:

            damaged.append(
                item
            )

        else:

            non_damaged.append(
                item
            )

    print(
        f"Damaged images     : "
        f"{len(damaged)}"
    )

    print(
        f"Non-damaged images : "
        f"{len(non_damaged)}"
    )

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    random.shuffle(
        damaged
    )

    random.shuffle(
        non_damaged
    )

    selected = []

    selected_paths = set()

    # ========================================================
    # STEP 1 — RARE CLASS COVERAGE
    # ========================================================

    print("\nEnsuring class coverage...")

    # Calculate how many images contain each class
    class_availability = {}

    for class_name in CLASSES:

        count = sum(
            1
            for item in damaged
            if class_name in item[2]
        )

        class_availability[
            class_name
        ] = count

    # Rare classes first
    classes_by_rarity = sorted(
        CLASSES,
        key=lambda x: class_availability[x]
    )

    for class_name in classes_by_rarity:

        candidates = [
            item
            for item in damaged
            if (
                class_name in item[2]
                and
                item[0] not in selected_paths
            )
        ]

        if len(candidates) > 0:

            chosen = random.choice(
                candidates
            )

            selected.append(
                chosen
            )

            selected_paths.add(
                chosen[0]
            )

            print(
                f"Added {class_name}: "
                f"{os.path.basename(chosen[0])}"
            )


    # ========================================================
    # STEP 2 — ADD MORE DAMAGED IMAGES
    # ========================================================

    # Target roughly 50% damaged images
    target_damaged = min(
        IMAGES_PER_COUNTRY // 2,
        len(damaged)
    )

    # Current number of damaged images
    current_damaged = len(
        selected
    )

    # Number still required
    needed_damaged = (
        target_damaged
        - current_damaged
    )

    if needed_damaged > 0:

        remaining_damaged = [
            item
            for item in damaged
            if item[0]
            not in selected_paths
        ]

        random.shuffle(
            remaining_damaged
        )

        extra_damaged = (
            remaining_damaged[
                :needed_damaged
            ]
        )

        for item in extra_damaged:

            selected.append(
                item
            )

            selected_paths.add(
                item[0]
            )


    # ========================================================
    # STEP 3 — ADD NON-DAMAGED IMAGES
    # ========================================================

    target_non_damaged = (
        IMAGES_PER_COUNTRY
        - target_damaged
    )

    available_non_damaged = (
        len(non_damaged)
    )

    actual_non_damaged = min(
        target_non_damaged,
        available_non_damaged
    )

    for item in non_damaged[
        :actual_non_damaged
    ]:

        selected.append(
            item
        )

        selected_paths.add(
            item[0]
        )


    # ========================================================
    # STEP 4 — FILL REMAINING SLOTS
    # ========================================================

    if len(selected) < IMAGES_PER_COUNTRY:

        print(
            "\nNot enough balanced "
            "images."
        )

        print(
            "Filling remaining slots "
            "from available images..."
        )

        remaining_items = [
            item
            for item in pairs
            if item[0]
            not in selected_paths
        ]

        random.shuffle(
            remaining_items
        )

        needed = (
            IMAGES_PER_COUNTRY
            - len(selected)
        )

        extra_items = (
            remaining_items[
                :needed
            ]
        )

        for image_path, xml_path in extra_items:

            classes = get_classes(
                xml_path
            )

            item = (
                image_path,
                xml_path,
                classes
            )

            selected.append(
                item
            )

            selected_paths.add(
                image_path
            )


    # ========================================================
    # LIMIT TO EXACTLY 200
    # ========================================================

    selected = selected[
        :IMAGES_PER_COUNTRY
    ]

    print(
        f"\nSelected for {country}: "
        f"{len(selected)}"
    )


    # ========================================================
    # CLASS STATISTICS
    # ========================================================

    class_counts = defaultdict(
        int
    )

    for (
        image_path,
        xml_path,
        classes
    ) in selected:

        for class_name in classes:

            class_counts[
                class_name
            ] += 1


    country_statistics[
        country
    ] = {
        "total": len(selected),
        "classes": dict(
            class_counts
        )
    }


    # ========================================================
    # COPY SELECTED FILES
    # ========================================================

    print(
        f"\nCopying {len(selected)} "
        f"images..."
    )

    for index, (
        image_path,
        xml_path,
        classes
    ) in enumerate(selected):

        original_filename = os.path.basename(
            image_path
        )

        extension = os.path.splitext(
            original_filename
        )[1]

        # ----------------------------------------------------
        # New unique filename
        # ----------------------------------------------------

        new_base_name = (
            f"{country}_{index:04d}"
        )

        new_image_name = (
            new_base_name
            + extension
        )

        new_xml_name = (
            new_base_name
            + ".xml"
        )

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        destination_image = os.path.join(
            OUTPUT_IMAGES,
            new_image_name
        )

        shutil.copy2(
            image_path,
            destination_image
        )

        # ----------------------------------------------------
        # Read XML
        # ----------------------------------------------------

        tree = ET.parse(
            xml_path
        )

        root = tree.getroot()

        # ----------------------------------------------------
        # Update filename
        # ----------------------------------------------------

        filename_element = (
            root.find("filename")
        )

        if filename_element is not None:

            filename_element.text = (
                new_image_name
            )

        # ----------------------------------------------------
        # Update path
        # ----------------------------------------------------

        path_element = (
            root.find("path")
        )

        if path_element is not None:

            path_element.text = (
                new_image_name
            )

        # ----------------------------------------------------
        # Save XML
        # ----------------------------------------------------

        destination_xml = os.path.join(
            OUTPUT_XML,
            new_xml_name
        )

        tree.write(
            destination_xml,
            encoding="utf-8",
            xml_declaration=True
        )

        # ----------------------------------------------------
        # Add to global manifest
        # ----------------------------------------------------

        all_selected.append(
            {
                "country": country,
                "image": new_image_name,
                "xml": new_xml_name
            }
        )


# ============================================================
# CREATE MANIFEST
# ============================================================

manifest_path = os.path.join(
    OUTPUT_ROOT,
    "manifest.txt"
)

with open(
    manifest_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "RDD2022 Multi-Country "
        "Development Dataset\n"
    )

    file.write(
        "========================================\n\n"
    )

    file.write(
        "Random seed: 42\n"
    )

    file.write(
        "Images per country/group: 200\n"
    )

    file.write(
        "Total target: 1200\n\n"
    )

    # --------------------------------------------------------
    # Country statistics
    # --------------------------------------------------------

    for country, stats in (
        country_statistics.items()
    ):

        file.write(
            f"{country}: "
            f"{stats['total']} images\n"
        )

        file.write(
            "Class distribution:\n"
        )

        for class_name in CLASSES:

            count = stats[
                "classes"
            ].get(
                class_name,
                0
            )

            file.write(
                f"  {class_name}: "
                f"{count}\n"
            )

        file.write("\n")

    # --------------------------------------------------------
    # Selected files
    # --------------------------------------------------------

    file.write(
        "\nSelected files:\n"
    )

    for item in all_selected:

        file.write(
            f"{item['country']} | "
            f"{item['image']} | "
            f"{item['xml']}\n"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MULTI-COUNTRY DATASET COMPLETE")
print("=" * 70)

print("\nCountry summary:")

total = 0

for country, stats in (
    country_statistics.items()
):

    print(
        f"{country:15} : "
        f"{stats['total']} images"
    )

    total += stats["total"]

print("-" * 45)

print(
    f"{'TOTAL':15} : "
    f"{total} images"
)

print("\nOutput folder:")
print(OUTPUT_ROOT)

print("\nImages:")
print(OUTPUT_IMAGES)

print("\nAnnotations:")
print(OUTPUT_XML)

print("\nManifest:")
print(manifest_path)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)