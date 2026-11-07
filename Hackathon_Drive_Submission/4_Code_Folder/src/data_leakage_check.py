"""Run a combined image split and annotation leakage audit.

Checks train/validation filename overlap, exact duplicate image content across
splits, image/XML matching, malformed XML, and annotation class counts. This is
a read-only audit; it never moves or deletes dataset files.

Default dataset: ``dataset/subsets/multicountry/RDD2022``

Run from the project ``src`` directory::

    python data_leakage_check.py
    python data_leakage_check.py --dataset-root path/to/RDD2022
"""

import argparse
import hashlib
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / ".git").is_dir()
)
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "dataset" / "subsets" / "multicountry" / "RDD2022"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_split_dir(dataset_root: Path, split: str) -> Path | None:
    """Find common image layouts, preferring the YOLO image split if present."""
    candidates = (
        dataset_root / "yolo" / "images" / split,
        dataset_root / "images" / split,
        dataset_root / "split" / split / "images",
        dataset_root / split / "images",
    )
    return next((path for path in candidates if path.is_dir()), None)


def image_files(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_images(paths: list[Path], split: str) -> dict[str, list[Path]]:
    hashes: dict[str, list[Path]] = defaultdict(list)
    failures = []
    for index, path in enumerate(paths, start=1):
        try:
            hashes[sha256_file(path)].append(path)
        except OSError as exc:
            failures.append((path, str(exc)))
        if index % 500 == 0 or index == len(paths):
            print(f"  Hashed {index}/{len(paths)} {split} images")
    for path, error in failures:
        print(f"  Could not hash {path}: {error}")
    return hashes


def print_examples(title: str, values: list[str], limit: int = 10) -> None:
    print(f"{title}: {len(values)}")
    for value in sorted(values)[:limit]:
        print(f"  {value}")
    if len(values) > limit:
        print(f"  ... and {len(values) - limit} more")


def check_annotations(dataset_root: Path, train_images: list[Path], val_images: list[Path]) -> None:
    xml_dir = dataset_root / "annotations" / "xmls"
    if not xml_dir.is_dir():
        print("\nIMAGE/XML CHECK: skipped (annotations/xmls directory not found)")
        return

    xml_paths = sorted(xml_dir.glob("*.xml"))
    image_stems = {path.stem for path in train_images + val_images}
    xml_stems = {path.stem for path in xml_paths}
    print("\nIMAGE/XML CHECK")
    print(f"  XML files: {len(xml_paths)}")
    print_examples("  Images without XML", list(image_stems - xml_stems))
    print_examples("  XML without image", list(xml_stems - image_stems))

    invalid_xml = []
    classes = Counter()
    for path in xml_paths:
        try:
            root = ET.parse(path).getroot()
            for obj in root.findall("object"):
                name = (obj.findtext("name") or "").strip()
                if name:
                    classes[name] += 1
        except (ET.ParseError, OSError) as exc:
            invalid_xml.append(f"{path.name}: {exc}")

    print_examples("  Invalid XML", invalid_xml)
    print("  Annotated object counts:")
    if classes:
        for name, count in sorted(classes.items()):
            print(f"    {name}: {count}")
    else:
        print("    (no named objects found)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
        help="Dataset root containing train/val images and optionally annotations/xmls",
    )
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()

    split_dirs = {split: find_split_dir(dataset_root, split) for split in ("train", "val")}
    missing = [split for split, folder in split_dirs.items() if folder is None]
    if missing:
        print(f"ERROR: Could not locate {', '.join(missing)} image split(s) under {dataset_root}")
        print("Expected layouts such as yolo/images/train and yolo/images/val.")
        return 2

    train_paths = image_files(split_dirs["train"])
    val_paths = image_files(split_dirs["val"])
    print("=" * 68)
    print("COMBINED DATA LEAKAGE CHECK")
    print("=" * 68)
    print(f"Dataset root: {dataset_root}")
    print(f"Train images: {len(train_paths)} ({split_dirs['train']})")
    print(f"Validation images: {len(val_paths)} ({split_dirs['val']})")

    train_names = Counter(path.name.casefold() for path in train_paths)
    val_names = Counter(path.name.casefold() for path in val_paths)
    shared_names = sorted(set(train_names) & set(val_names))
    print("\nFILENAME OVERLAP")
    print(f"  Shared image filenames: {len(shared_names)}")
    for name in shared_names[:10]:
        print(f"  {name} (train={train_names[name]}, val={val_names[name]})")
    if len(shared_names) > 10:
        print(f"  ... and {len(shared_names) - 10} more")

    print("\nEXACT CONTENT DUPLICATES")
    train_hashes = hash_images(train_paths, "train")
    val_hashes = hash_images(val_paths, "validation")
    shared_hashes = sorted(set(train_hashes) & set(val_hashes))
    print(f"  Duplicate hash groups across splits: {len(shared_hashes)}")
    print(f"  Affected train files: {sum(len(train_hashes[key]) for key in shared_hashes)}")
    print(f"  Affected validation files: {sum(len(val_hashes[key]) for key in shared_hashes)}")
    for digest in shared_hashes[:10]:
        print(f"  SHA-256 {digest}")
        for path in train_hashes[digest]:
            print(f"    TRAIN: {path}")
        for path in val_hashes[digest]:
            print(f"    VAL:   {path}")
    if len(shared_hashes) > 10:
        print(f"  ... and {len(shared_hashes) - 10} more groups")

    check_annotations(dataset_root, train_paths, val_paths)

    print("\nSUMMARY")
    if shared_names or shared_hashes:
        print("  WARNING: possible train/validation leakage was found; inspect the overlaps above.")
    else:
        print("  PASS: no shared filenames or exact duplicate image content across splits.")
    print("  Near-duplicate visual content is not checked by this script.")
    print("  No dataset files were modified.")
    return 1 if shared_names or shared_hashes else 0


if __name__ == "__main__":
    raise SystemExit(main())
