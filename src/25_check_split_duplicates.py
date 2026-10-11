
from pathlib import Path
from collections import defaultdict
import hashlib

# Project paths
ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "dataset" / "yolo" / "images"

TRAIN_DIR = IMAGE_ROOT / "train"
VAL_DIR = IMAGE_ROOT / "val"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_image_hash(image_path):
    """Calculate SHA-256 hash of an image file."""
    sha256 = hashlib.sha256()

    with image_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def scan_images(folder):
    """Return image hashes and their file paths."""
    hashes = defaultdict(list)

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder}"
        )

    image_paths = sorted(
        path for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"Scanning {folder.name}: {len(image_paths)} images")

    for index, image_path in enumerate(image_paths, start=1):
        try:
            digest = get_image_hash(image_path)
            hashes[digest].append(image_path)
        except OSError as error:
            print(f"Could not read {image_path.name}: {error}")

        if index % 500 == 0 or index == len(image_paths):
            print(f"  Processed {index}/{len(image_paths)}")

    return hashes


def main():
    print("=" * 60)
    print("TRAIN / VALIDATION DUPLICATE CHECK")
    print("=" * 60)

    try:
        train_hashes = scan_images(TRAIN_DIR)
        val_hashes = scan_images(VAL_DIR)
    except FileNotFoundError as error:
        print(f"\nERROR: {error}")
        print("Check that dataset/yolo/images/train and val exist.")
        return

    train_count = sum(len(paths) for paths in train_hashes.values())
    val_count = sum(len(paths) for paths in val_hashes.values())

    # Hashes present in both splits
    shared_hashes = set(train_hashes) & set(val_hashes)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Train images: {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Exact hashes shared between train and validation: "
          f"{len(shared_hashes)}")

    if shared_hashes:
        print("\nFirst 10 duplicate groups:")

        for digest in sorted(shared_hashes)[:10]:
            print(f"\nSHA-256: {digest[:16]}...")
            for path in train_hashes[digest]:
                print(f"  TRAIN: {path.name}")
            for path in val_hashes[digest]:
                print(f"  VAL:   {path.name}")

        print("\nWARNING: Investigate the shared images before training.")
    else:
        print("\nNo exact cross-split duplicates found.")

    print("\nCheck complete. No files were modified.")


if __name__ == "__main__":
    main()
