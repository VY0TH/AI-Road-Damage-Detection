import os
import re

PROJECT_ROOT = r"C:\Users\Vyom Thakur\Desktop\AI hackathon\RoadDamageProject"

LABEL_MAP = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "raw",
    "label_map.pbtxt"
)

print("=" * 60)
print("RDD2022 LABEL MAP")
print("=" * 60)

if not os.path.exists(LABEL_MAP):
    print("\nERROR: label_map.pbtxt not found.")
    print("Expected:")
    print(LABEL_MAP)
    exit()

with open(LABEL_MAP, "r", encoding="utf-8") as f:
    text = f.read()

# Find each item { ... }
items = re.findall(
    r'item\s*\{(.*?)\}',
    text,
    re.DOTALL
)

print(f"\nClasses found: {len(items)}\n")

for item in items:

    class_id = re.search(
        r'id:\s*(\d+)',
        item
    )

    class_name = re.search(
        r'name:\s*[\'"]([^\'"]+)[\'"]',
        item
    )

    display_name = re.search(
        r'display_name:\s*[\'"]([^\'"]+)[\'"]',
        item
    )

    print("-" * 40)

    if class_id:
        print("ID          :", class_id.group(1))

    if class_name:
        print("Name        :", class_name.group(1))

    if display_name:
        print("Display name:", display_name.group(1))


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)