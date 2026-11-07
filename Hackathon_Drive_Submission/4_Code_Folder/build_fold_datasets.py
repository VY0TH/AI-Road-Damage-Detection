from pathlib import Path
import csv
import shutil
import yaml

project = Path.cwd()
dataset = project / "dataset/subsets/multicountry/RDD2022/yolo_grouped"
manifest = project / "results/cross_validation/fold_manifest.csv"
output_root = project / "results/cross_validation"

classes = [
    "D00", "D01", "D10", "D11", "D20",
    "D40", "D43", "D44", "D50", "D0w0"
]

with manifest.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if len(rows) != 960:
    raise SystemExit(f"Expected 960 rows, found {len(rows)}")

# Find each source image and label in the existing grouped dataset.
sources = {}
for row in rows:
    name = Path(row["image"]).name
    stem = Path(name).stem

    image_path = next(
        (dataset / "images" / split / name
         for split in ("train", "val")
         if (dataset / "images" / split / name).is_file()),
        None
    )
    label_path = next(
        (dataset / "labels" / split / f"{stem}.txt"
         for split in ("train", "val")
         if (dataset / "labels" / split / f"{stem}.txt").is_file()),
        None
    )

    if image_path is None or label_path is None:
        raise SystemExit(f"Missing image or label for {name}")

    sources[name] = (image_path, label_path)

for fold in range(1, 11):
    fold_root = output_root / f"fold_{fold}"

    # Refuse to overwrite an existing fold dataset.
    if (fold_root / "images").exists() or (fold_root / "labels").exists():
        raise SystemExit(
            f"{fold_root} already contains images/labels. "
            "Stop to avoid overwriting existing data."
        )

    train_rows = [r for r in rows if int(r["fold"]) != fold]
    val_rows = [r for r in rows if int(r["fold"]) == fold]

    for split, split_rows in (("train", train_rows), ("val", val_rows)):
        image_dest = fold_root / "images" / split
        label_dest = fold_root / "labels" / split
        image_dest.mkdir(parents=True, exist_ok=True)
        label_dest.mkdir(parents=True, exist_ok=True)

        for row in split_rows:
            name = Path(row["image"]).name
            src_image, src_label = sources[name]
            shutil.copy2(src_image, image_dest / name)
            shutil.copy2(src_label, label_dest / f"{Path(name).stem}.txt")

    config = {
        "path": str(fold_root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes
    }
    with (fold_root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(
        f"Fold {fold}: copied {len(train_rows)} train and "
        f"{len(val_rows)} validation images"
    )

print("\nAll fold datasets created.")
print("Original dataset and trained weights were not modified.")
