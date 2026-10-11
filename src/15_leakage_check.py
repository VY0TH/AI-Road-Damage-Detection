import os
import hashlib
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

PROJECT_ROOT = r"C:\Users\Vyom Thakur\Desktop\AI hackathon\RoadDamageProject"

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "subsets",
    "multicountry",
    "RDD2022"
)

IMAGE_DIR = os.path.join(
    DATASET_ROOT,
    "images"
)

XML_DIR = os.path.join(
    DATASET_ROOT,
    "annotations",
    "xmls"
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
# HASH FUNCTION
# ============================================================

def get_hash(file_path):

    md5 = hashlib.md5()

    with open(
        file_path,
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            md5.update(chunk)

    return md5.hexdigest()


# ============================================================
# START
# ============================================================

print("=" * 70)
print("RDD2022 MULTI-COUNTRY LEAKAGE CHECK")
print("=" * 70)


# ============================================================
# GET FILES
# ============================================================

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(
        (
            ".jpg",
            ".jpeg",
            ".png"
        )
    )
]

xml_files = [
    f for f in os.listdir(XML_DIR)
    if f.lower().endswith(".xml")
]


print("\nBasic dataset counts")
print("-" * 50)

print(
    f"Images : {len(image_files)}"
)

print(
    f"XMLs   : {len(xml_files)}"
)


# ============================================================
# IMAGE ↔ XML MATCHING
# ============================================================

image_bases = {
    os.path.splitext(f)[0]
    for f in image_files
}

xml_bases = {
    os.path.splitext(f)[0]
    for f in xml_files
}


missing_xml = (
    image_bases - xml_bases
)

missing_images = (
    xml_bases - image_bases
)


print("\nImage/XML integrity")
print("-" * 50)

print(
    f"Images without XML : "
    f"{len(missing_xml)}"
)

print(
    f"XML without image  : "
    f"{len(missing_images)}"
)


if missing_xml:

    print("\nMissing XML examples:")

    for item in list(
        missing_xml
    )[:10]:

        print(
            " ",
            item
        )


if missing_images:

    print("\nMissing image examples:")

    for item in list(
        missing_images
    )[:10]:

        print(
            " ",
            item
        )


# ============================================================
# XML VALIDATION + CLASS DISTRIBUTION
# ============================================================

invalid_xml = []

class_counts = Counter()

image_damage_status = {}

country_counts = Counter()


print("\nChecking XML files...")


for xml_file in xml_files:

    xml_path = os.path.join(
        XML_DIR,
        xml_file
    )

    try:

        tree = ET.parse(
            xml_path
        )

        root = tree.getroot()

        objects = root.findall(
            "object"
        )

        if len(objects) == 0:

            image_damage_status[
                xml_file
            ] = "non-damaged"

        else:

            image_damage_status[
                xml_file
            ] = "damaged"

        for obj in objects:

            name = obj.find(
                "name"
            )

            if name is None:
                continue

            class_name = (
                name.text.strip()
            )

            class_counts[
                class_name
            ] += 1

    except Exception as error:

        invalid_xml.append(
            (
                xml_file,
                str(error)
            )
        )


print("\nXML validation")
print("-" * 50)

print(
    f"Invalid XML files : "
    f"{len(invalid_xml)}"
)


if invalid_xml:

    print("\nInvalid XML examples:")

    for filename, error in invalid_xml[:10]:

        print(
            filename,
            "->",
            error
        )


# ============================================================
# COUNTRY DISTRIBUTION
# ============================================================

for image_file in image_files:

    if "_" in image_file:

        country = image_file.split(
            "_"
        )[0]

        # United_States needs special handling
        if image_file.startswith(
            "United_States_"
        ):

            country = "United_States"

        country_counts[
            country
        ] += 1


print("\nCountry distribution")
print("-" * 50)

for country, count in sorted(
    country_counts.items()
):

    print(
        f"{country:20} : {count}"
    )


# ============================================================
# DUPLICATE CONTENT CHECK
# ============================================================

print("\nCalculating image hashes...")
print("This may take a little time.")


hash_to_files = defaultdict(
    list
)

for index, image_file in enumerate(
    image_files,
    start=1
):

    image_path = os.path.join(
        IMAGE_DIR,
        image_file
    )

    file_hash = get_hash(
        image_path
    )

    hash_to_files[
        file_hash
    ].append(
        image_file
    )

    if index % 200 == 0:

        print(
            f"Checked {index}/"
            f"{len(image_files)}"
        )


duplicate_groups = [
    files
    for files in hash_to_files.values()
    if len(files) > 1
]


duplicate_count = sum(
    len(files)
    for files in duplicate_groups
)


print("\nDuplicate image check")
print("-" * 50)

print(
    f"Duplicate groups : "
    f"{len(duplicate_groups)}"
)

print(
    f"Images involved  : "
    f"{duplicate_count}"
)


if duplicate_groups:

    print("\nDuplicate examples:")

    for group in duplicate_groups[:10]:

        print(
            " ",
            group
        )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nDamage class distribution")
print("-" * 50)

for class_name in CLASSES:

    print(
        f"{class_name:5} : "
        f"{class_counts.get(class_name, 0)}"
    )


# ============================================================
# DAMAGED / NON-DAMAGED
# ============================================================

damaged_count = sum(
    1
    for status in image_damage_status.values()
    if status == "damaged"
)

non_damaged_count = sum(
    1
    for status in image_damage_status.values()
    if status == "non-damaged"
)


print("\nDamage status")
print("-" * 50)

print(
    f"Damaged images     : "
    f"{damaged_count}"
)

print(
    f"Non-damaged images : "
    f"{non_damaged_count}"
)


# ============================================================
# FINAL VERDICT
# ============================================================

print("\n" + "=" * 70)
print("LEAKAGE CHECK SUMMARY")
print("=" * 70)

problems = 0

if missing_xml:
    print("❌ Missing XML files")
    problems += 1
else:
    print("✅ Image/XML matching")

if missing_images:
    print("❌ XML files without images")
    problems += 1
else:
    print("✅ No orphan XML files")

if invalid_xml:
    print("❌ Invalid XML files")
    problems += 1
else:
    print("✅ XML validity")

if duplicate_groups:
    print("⚠️ Duplicate images detected")
    problems += 1
else:
    print("✅ No exact duplicate images")

if len(image_files) == 1200:
    print("✅ Exactly 1200 images")
else:
    print(
        f"⚠️ Expected 1200 images, "
        f"found {len(image_files)}"
    )
    problems += 1


print("\n")


if problems == 0:

    print(
        "RESULT: DATASET PASSED "
        "BASIC LEAKAGE CHECK"
    )

else:

    print(
        f"RESULT: {problems} "
        f"ISSUE(S) NEED ATTENTION"
    )


print("\nDone.")