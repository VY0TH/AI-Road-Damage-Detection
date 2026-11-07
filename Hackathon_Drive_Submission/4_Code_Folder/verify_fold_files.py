from pathlib import Path
import csv

base = Path("dataset/subsets/multicountry/RDD2022/yolo_grouped")
manifest = Path("results/cross_validation/fold_manifest.csv")

with manifest.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

missing_images = []
missing_labels = []

for row in rows:
    name = Path(row["image"]).name
    country = name.split("_")[0]

    # Locate each image in either grouped split.
    found_image = any(
        (base / "images" / split / name).exists()
        for split in ("train", "val")
    )
    found_label = any(
        (base / "labels" / split / f"{Path(name).stem}.txt").exists()
        for split in ("train", "val")
    )

    if not found_image:
        missing_images.append(name)
    if not found_label:
        missing_labels.append(name)

print("Manifest images:", len(rows))
print("Missing grouped images:", len(missing_images))
print("Missing grouped labels:", len(missing_labels))
print("Missing image examples:", missing_images[:10])
print("Missing label examples:", missing_labels[:10])
