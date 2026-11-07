"""Step 2: Image Preprocessor.

Performs:
- RGB channel standardization
- Lanczos high-quality resizing with optional aspect ratio letterboxing
- Corrupt image filtering and validation
- Metadata manifest generation
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def resize_image(
    image: Image.Image,
    target_size: tuple[int, int] = (224, 224),
    keep_aspect_ratio: bool = True,
    pad_color: tuple[int, int, int] = (128, 128, 128),
) -> Image.Image:
    """Resize PIL Image to target_size. If keep_aspect_ratio is True, adds letterbox padding."""
    if not keep_aspect_ratio:
        return image.resize(target_size, Image.Resampling.LANCZOS)

    target_w, target_h = target_size
    img_w, img_h = image.size
    ratio = min(target_w / img_w, target_h / img_h)
    new_w, new_h = max(1, int(img_w * ratio)), max(1, int(img_h * ratio))

    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", target_size, pad_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def preprocess_folder(
    source_dir: Path,
    output_dir: Path,
    target_size: int = 224,
    keep_aspect_ratio: bool = True,
) -> tuple[int, int]:
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")
    if source_dir == output_dir or source_dir in output_dir.parents:
        raise ValueError("Output must not be the source directory or inside it.")

    output_dir.mkdir(parents=True, exist_ok=True)
    processed, skipped = 0, 0
    records = []

    # Find all subdirectories (splits or classes)
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        rel_path = path.relative_to(source_dir)
        dest_path = output_dir / rel_path.with_suffix(".jpg")
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(path) as img:
                rgb_img = img.convert("RGB")
                final_img = resize_image(
                    rgb_img,
                    target_size=(target_size, target_size),
                    keep_aspect_ratio=keep_aspect_ratio,
                )
                final_img.save(dest_path, format="JPEG", quality=95)
            processed += 1
            records.append({
                "source": str(rel_path),
                "destination": str(dest_path.relative_to(output_dir)),
                "width": target_size,
                "height": target_size,
            })
        except (OSError, UnidentifiedImageError) as e:
            skipped += 1
            print(f"Skipping corrupt image {path}: {e}")

    manifest_path = output_dir / "preprocessing_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "destination", "width", "height"])
        writer.writeheader()
        writer.writerows(records)

    return processed, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Input directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--size", type=int, default=224, help="Target image size")
    parser.add_argument("--stretch", action="store_true", help="Stretch instead of aspect-ratio letterboxing")
    args = parser.parse_args()

    proc, skip = preprocess_folder(args.source, args.output, args.size, not args.stretch)
    print(f"Preprocessing completed: {proc} processed, {skip} skipped.")


if __name__ == "__main__":
    main()
