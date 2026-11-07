from pathlib import Path
from collections import Counter
import csv
import random
import xml.etree.ElementTree as ET

base = Path("dataset/subsets/multicountry/RDD2022")
image_dir = base / "split/train/images"
xml_dir = base / "split/train/annotations/xmls"
out = Path("results/cross_validation/fold_manifest.csv")

SEED = 42
N_FOLDS = 10
YOLO_CLASSES = {
    "D00", "D01", "D10", "D11", "D20",
    "D40", "D43", "D44", "D50", "D0w0"
}

images = sorted(
    p for p in image_dir.iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if len(images) != 960:
    raise SystemExit(f"Expected 960 training images; found {len(images)}")

missing_xml = [p.name for p in images if not (xml_dir / f"{p.stem}.xml").exists()]
if missing_xml:
    raise SystemExit(f"Missing XML annotations: {missing_xml[:10]}")

rng = random.Random(SEED)
rng.shuffle(images)

fold_sizes = [len(images) // N_FOLDS] * N_FOLDS
for i in range(len(images) % N_FOLDS):
    fold_sizes[i] += 1

rows = []
start = 0

for fold, size in enumerate(fold_sizes, start=1):
    validation_images = images[start:start + size]
    start += size

    for img in validation_images:
        xp = xml_dir / f"{img.stem}.xml"
        root = ET.parse(xp).getroot()
        classes = sorted({
            (obj.findtext("name") or "").strip()
            for obj in root.findall("object")
            if (obj.findtext("name") or "").strip() in YOLO_CLASSES
        })

        rows.append({
            "image": str(img).replace("\\", "/"),
            "xml": str(xp).replace("\\", "/"),
            "country": img.stem.split("_")[0],
            "classes": "|".join(classes),
            "fold": fold
        })

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["image", "xml", "country", "classes", "fold"]
    )
    writer.writeheader()
    writer.writerows(rows)

# Validate fold sizes and unique image assignment
fold_counts = Counter(row["fold"] for row in rows)
image_counts = Counter(row["image"] for row in rows)

assert len(rows) == 960
assert len(image_counts) == 960
assert all(count == 1 for count in image_counts.values())
assert all(fold_counts[i] == 96 for i in range(1, 11))
assert all(
    "Repair" not in row["classes"].split("|")
    for row in rows
)

print("Manifest regenerated successfully.")
print(f"Total images: {len(rows)}")
print(f"Unique images: {len(image_counts)}")
print(f"Folds: {N_FOLDS}")
print(f"Images per validation fold: {dict(sorted(fold_counts.items()))}")
print(f"Repair included as a YOLO class: No")
print(f"Manifest: {out}")
print("\nValidation image counts by country and fold:")
for fold in range(1, N_FOLDS + 1):
    counts = Counter(r["country"] for r in rows if r["fold"] == fold)
    print(f"Fold {fold}: {dict(sorted(counts.items()))}")
