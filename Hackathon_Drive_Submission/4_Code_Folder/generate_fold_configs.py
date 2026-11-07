from pathlib import Path
import csv
import yaml

base = Path("dataset/subsets/multicountry/RDD2022")
manifest = Path("results/cross_validation/fold_manifest.csv")
output_root = Path("results/cross_validation")

classes = [
    "D00", "D01", "D10", "D11", "D20",
    "D40", "D43", "D44", "D50", "D0w0"
]

with manifest.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if len(rows) != 960:
    raise SystemExit(f"Expected 960 manifest rows, found {len(rows)}")

for fold in range(1, 11):
    fold_root = output_root / f"fold_{fold}"
    train_list = fold_root / "train.txt"
    val_list = fold_root / "val.txt"
    fold_root.mkdir(parents=True, exist_ok=True)

    train_images = [
        str((Path(r["image"]).resolve())).replace("\\", "/")
        for r in rows if int(r["fold"]) != fold
    ]
    val_images = [
        str((Path(r["image"]).resolve())).replace("\\", "/")
        for r in rows if int(r["fold"]) == fold
    ]

    if len(train_images) != 864 or len(val_images) != 96:
        raise SystemExit(f"Fold {fold}: unexpected train/validation sizes")

    train_list.write_text("\n".join(train_images) + "\n", encoding="utf-8")
    val_list.write_text("\n".join(val_images) + "\n", encoding="utf-8")

    config = {
        "path": str(base.resolve()).replace("\\", "/"),
        "train": str(train_list.resolve()).replace("\\", "/"),
        "val": str(val_list.resolve()).replace("\\", "/"),
        "nc": len(classes),
        "names": classes
    }

    with (fold_root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"Fold {fold}: train={len(train_images)}, val={len(val_images)}")

print(f"\nCreated 10 fold configurations under: {output_root.resolve()}")
