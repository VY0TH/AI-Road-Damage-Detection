
from pathlib import Path
from collections import defaultdict
import hashlib

# Project root
ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = ROOT / "dataset" / "yolo" / "images" / "train"
VAL_DIR = ROOT / "dataset" / "yolo" / "images" / "val"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path):
    """Calculate the exact SHA-256 hash of an image."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_hash_map(folder):
    """Map each image hash to its file paths."""
    result = defaultdict(list)

    if not folder.exists():
        print(f"ERROR: Folder not found: {folder}")
        return result

    images = [
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"\nScanning {folder.name}: {len(images)} images")

    for index, image_path in enumerate(images, start=1):
        try:
            result[sha256_file(image_path)].append(image_path)
        except OSError as error:
            print(f"Could not read {image_path}: {error}")

        if index % 500 == 0 or index == len(images):
            print(f"Processed {index}/{len(images)}")

    return result


def main():
    print("=" * 60)
    print("TRAIN / VALIDATION DUPLICATE CHECK")
    print("=" * 60)

    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        print("ERROR: Train or validation folder does not exist.")
        print("Check the dataset/yolo/images directory.")
        return

    train_hashes = build_hash_map(TRAIN_DIR)
    val_hashes = build_hash_map(VAL_DIR)

    shared_hashes = set(train_hashes) & set(val_hashes)

    affected_train = sum(
        len(train_hashes[h]) for h in shared_hashes
    )
    affected_val = sum(
        len(val_hashes[h]) for h in shared_hashes
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Unique image hashes in TRAIN: {len(train_hashes)}")
    print(f"Unique image hashes in VAL:   {len(val_hashes)}")
    print(f"Shared exact-duplicate groups: {len(shared_hashes)}")
    print(f"Affected TRAIN files: {affected_train}")
    print(f"Affected VAL files:   {affected_val}")

    if shared_hashes:
        print("\nWARNING: Identical images exist in both splits.")
        print("First 20 duplicate groups:")

        for number, image_hash in enumerate(
            sorted(shared_hashes), start=1
        ):
            if number > 20:
                break

            print(f"\nDuplicate group {number}")
            print(f"SHA-256: {image_hash}")

            for path in train_hashes[image_hash]:
                print(f"  TRAIN: {path.name}")

            for path in val_hashes[image_hash]:
                print(f"  VAL:   {path.name}")

        print("\nNo files were modified.")
        print("Correct the split-generation script before training.")
    else:
        print("\nPASS: No exact duplicates shared between TRAIN and VAL.")
        print("Note: This check does not detect visually similar images.")

    print("\nCheck complete.")


if __name__ == "__main__":
    main()
