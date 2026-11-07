"""Prepare a class-folder image dataset for classifier training.

Reads ``<source>/<split>/<class>/<image>`` and writes resized RGB images to
the same layout beneath a separate output directory. The input dataset is
never modified. Pixel normalization and random augmentation should remain in
the model's training transforms rather than being baked into saved files.

Example:
    python algorithms/image_dataset_preprocessor.py
    python algorithms/image_dataset_preprocessor.py --size 256
    python algorithms/image_dataset_preprocessor.py --source path/to/crops \
        --output path/to/preprocessed
"""

import argparse
import csv
from pathlib import Path

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / ".git").is_dir()
)
DEFAULT_SOURCE = PROJECT_ROOT / "dataset" / "processed" / "efficientnet_crops"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "processed" / "image_dataset_preprocessed"
SPLITS = ("train", "val")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def preprocess_dataset(source: Path, output: Path, size: int) -> tuple[int, int]:
    """Resize readable images, returning (processed, skipped) counts."""
    source = source.resolve()
    output = output.resolve()

    if size < 1:
        raise ValueError("Image size must be a positive integer.")
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset directory not found: {source}")
    if source == output or source in output.parents:
        raise ValueError("Output must not be the source directory or inside it.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}. Choose a new output path."
        )

    rows = []
    processed = 0
    skipped = 0
    output.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        split_dir = source / split
        if not split_dir.is_dir():
            print(f"Skipping missing split: {split_dir}")
            continue

        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            destination_class = output / split / class_dir.name
            for image_path in sorted(class_dir.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                relative_path = image_path.relative_to(source)
                destination = output / relative_path.with_suffix(".jpg")
                try:
                    with Image.open(image_path) as image:
                        prepared = image.convert("RGB").resize(
                            (size, size), Image.Resampling.LANCZOS
                        )
                        destination_class.mkdir(parents=True, exist_ok=True)
                        prepared.save(destination, format="JPEG", quality=95)
                except (OSError, UnidentifiedImageError) as exc:
                    skipped += 1
                    print(f"Skipping unreadable image {image_path}: {exc}")
                    continue

                processed += 1
                rows.append({
                    "split": split,
                    "class_name": class_dir.name,
                    "source_path": str(relative_path),
                    "output_path": str(destination.relative_to(output)),
                    "width": size,
                    "height": size,
                })

    manifest = output / "preprocessing_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "class_name", "source_path", "output_path", "width", "height"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return processed, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=224, help="Output width and height")
    args = parser.parse_args()

    processed, skipped = preprocess_dataset(args.source, args.output, args.size)
    print(f"Preprocessed images: {processed}")
    print(f"Skipped unreadable images: {skipped}")
    print(f"Output dataset: {args.output.resolve()}")
    print(f"Manifest: {(args.output.resolve() / 'preprocessing_manifest.csv')}")


if __name__ == "__main__":
    main()
