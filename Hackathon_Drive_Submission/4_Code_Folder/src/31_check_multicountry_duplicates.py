
from pathlib import Path
from collections import defaultdict
import hashlib

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "dataset/subsets/multicountry/RDD2022/yolo"
TRAIN = ROOT / "images/train"
VAL = ROOT / "images/val"

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    if not TRAIN.exists() or not VAL.exists():
        print("ERROR: Train or validation image folder not found.")
        return

    train_hashes = defaultdict(list)
    val_hashes = defaultdict(list)

    for folder, output in [(TRAIN, train_hashes), (VAL, val_hashes)]:
        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }:
                output[sha256(path)].append(path.name)

    overlap = set(train_hashes) & set(val_hashes)

    print("Train images:", sum(map(len, train_hashes.values())))
    print("Validation images:", sum(map(len, val_hashes.values())))
    print("Exact duplicate groups across splits:", len(overlap))

    for digest in sorted(overlap):
        print("TRAIN:", train_hashes[digest])
        print("VAL:  ", val_hashes[digest])

    if overlap:
        print("\nWARNING: Exact duplicate images occur in both splits.")
    else:
        print("\nPASS: No exact image duplicates across train and validation.")

if __name__ == "__main__":
    main()
