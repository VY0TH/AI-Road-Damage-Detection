"""Step 1: Dataset Inspection and Validation Tool.

Scans an image dataset folder, validates file headers, computes dimension
statistics, checks class balance, and verifies file integrity.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def inspect_directory(
    dataset_dir: Path,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict:
    dataset_dir = Path(dataset_dir).resolve()
    report = {
        "dataset_path": str(dataset_dir),
        "splits_found": [],
        "total_images": 0,
        "class_distribution": {},
        "dimension_stats": {},
        "format_distribution": {},
        "corrupt_files": [],
    }

    widths, heights = [], []
    formats = Counter()
    class_dist = {}

    target_splits = [s for s in splits if (dataset_dir / s).is_dir()]
    if not target_splits:
        target_splits = ["."]  # Check root directly if no sub-splits exist

    for split in target_splits:
        split_dir = dataset_dir / split if split != "." else dataset_dir
        report["splits_found"].append(split)
        class_dist[split] = Counter()

        # Class subdirectories
        subdirs = [p for p in split_dir.iterdir() if p.is_dir()]
        if subdirs:
            for c_dir in sorted(subdirs):
                c_name = c_dir.name
                for img_path in sorted(c_dir.iterdir()):
                    if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                        report["total_images"] += 1
                        class_dist[split][c_name] += 1
                        formats[img_path.suffix.lower()] += 1
                        try:
                            with Image.open(img_path) as img:
                                w, h = img.size
                                widths.append(w)
                                heights.append(h)
                        except (UnidentifiedImageError, OSError) as e:
                            report["corrupt_files"].append({"file": str(img_path), "error": str(e)})
        else:
            for img_path in sorted(split_dir.iterdir()):
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    report["total_images"] += 1
                    formats[img_path.suffix.lower()] += 1
                    try:
                        with Image.open(img_path) as img:
                            w, h = img.size
                            widths.append(w)
                            heights.append(h)
                    except (UnidentifiedImageError, OSError) as e:
                        report["corrupt_files"].append({"file": str(img_path), "error": str(e)})

    report["class_distribution"] = {s: dict(c) for s, c in class_dist.items()}
    report["format_distribution"] = dict(formats)

    if widths and heights:
        report["dimension_stats"] = {
            "min_width": min(widths),
            "max_width": max(widths),
            "mean_width": round(sum(widths) / len(widths), 1),
            "min_height": min(heights),
            "max_height": max(heights),
            "mean_height": round(sum(heights) / len(heights), 1),
        }

    return report


def print_report(report: dict) -> None:
    print("=" * 65)
    print("STEP 1: IMAGE DATASET INSPECTION REPORT")
    print("=" * 65)
    print(f"Dataset Path : {report['dataset_path']}")
    print(f"Splits Found : {', '.join(report['splits_found'])}")
    print(f"Total Images : {report['total_images']}")
    print(f"Image Formats: {report['format_distribution']}")

    if report.get("dimension_stats"):
        ds = report["dimension_stats"]
        print(f"\nDimensions Summary:")
        print(f"  Width : min={ds['min_width']}, max={ds['max_width']}, mean={ds['mean_width']}px")
        print(f"  Height: min={ds['min_height']}, max={ds['max_height']}, mean={ds['mean_height']}px")

    print("\nClass Distribution:")
    for split, classes in report.get("class_distribution", {}).items():
        print(f"  [{split.upper()}]: {sum(classes.values())} total")
        for cls_name, count in sorted(classes.items()):
            print(f"    - {cls_name:10}: {count:5} samples")

    if report["corrupt_files"]:
        print(f"\nWARNING: Found {len(report['corrupt_files'])} corrupt/unreadable files:")
        for cf in report["corrupt_files"][:5]:
            print(f"  {cf['file']}: {cf['error']}")
    else:
        print("\nIntegrity Check: PASSED (0 corrupt images detected)")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset" / "processed" / "efficientnet_crops",
        help="Path to the dataset directory to inspect",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Save report to JSON file")
    args = parser.parse_args()

    report = inspect_directory(args.dataset_dir)
    print_report(report)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {args.output_json}")


if __name__ == "__main__":
    main()
