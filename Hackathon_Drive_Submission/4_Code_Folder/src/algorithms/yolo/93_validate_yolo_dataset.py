"""Audit YOLO image/label pairing, boxes, corruption, counts, and split leakage."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
DATA_YAML = ROOT / "dataset/subsets/multicountry/RDD2022/yolo/data.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dhash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((9, 8)).tobytes())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return bits


def _image_map(folder: Path):
    return {
        path.stem: path for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def _audit_split(split: str, names: dict[int, str], dataset_root: Path):
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Missing YOLO image or label directory for split {split}.")
    images = _image_map(image_dir)
    labels = {path.stem: path for path in label_dir.glob("*.txt")}
    report = {
        "images": len(images), "labels": len(labels),
        "missing_labels": [], "orphan_labels": [], "empty_labels": 0,
        "objects_by_class": Counter(), "invalid_lines": [], "corrupt_images": [],
        "image_hashes": {}, "dhashes": {},
    }
    for stem, image_path in images.items():
        label_path = labels.get(stem)
        if label_path is None:
            report["missing_labels"].append(image_path.name)
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            report["corrupt_images"].append({"image": image_path.name, "error": str(exc)})
            continue
        try:
            report["image_hashes"][stem] = _hash_file(image_path)
            report["dhashes"][stem] = _dhash(image_path)
        except OSError as exc:
            report["corrupt_images"].append({"image": image_path.name, "error": str(exc)})
        if label_path is None:
            continue
        rows = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            report["empty_labels"] += 1
            continue
        for line_number, row in enumerate(rows, start=1):
            parts = row.split()
            try:
                if len(parts) != 5:
                    raise ValueError("expected class_id x_center y_center width height")
                class_id = int(parts[0])
                x, y, width, height = map(float, parts[1:])
                if class_id not in names:
                    raise ValueError(f"class ID {class_id} is absent from data.yaml")
                if not all(0.0 <= value <= 1.0 for value in (x, y, width, height)):
                    raise ValueError("normalized coordinates and size must be in [0,1]")
                # Labels are rounded to six decimals; permit the resulting
                # sub-pixel boundary error while still rejecting real overflow.
                tolerance = 1e-5
                if width == 0 or height == 0 or x - width / 2 < -tolerance or y - height / 2 < -tolerance or x + width / 2 > 1 + tolerance or y + height / 2 > 1 + tolerance:
                    raise ValueError("box is zero-area or extends outside the normalized image")
                report["objects_by_class"][names[class_id]] += 1
            except (ValueError, OverflowError) as exc:
                report["invalid_lines"].append({"label": label_path.name, "line": line_number, "error": str(exc)})
    report["orphan_labels"] = sorted(set(labels) - set(images))
    report["objects_by_class"] = dict(report["objects_by_class"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, default=DATA_YAML)
    parser.add_argument("--output", type=Path, default=ROOT / "results/dataset_audit/yolo_integrity_report.json")
    parser.add_argument("--near-duplicate-distance", type=int, default=4)
    args = parser.parse_args()
    data_yaml = args.data_yaml if args.data_yaml.is_absolute() else ROOT / args.data_yaml
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {data_yaml}")
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    raw_names = config["names"]
    names = {int(index): str(name) for index, name in raw_names.items()} if isinstance(raw_names, dict) else dict(enumerate(map(str, raw_names)))
    if int(config.get("nc", len(names))) != len(names):
        raise ValueError("data.yaml nc does not match its names list")
    raw_root = Path(config.get("path", data_yaml.parent))
    dataset_root = raw_root if raw_root.is_absolute() else ROOT / raw_root
    train = _audit_split("train", names, dataset_root)
    val = _audit_split("val", names, dataset_root)

    train_hashes, val_hashes = train.pop("image_hashes"), val.pop("image_hashes")
    train_dhashes, val_dhashes = train.pop("dhashes"), val.pop("dhashes")
    exact = []
    train_by_hash = {}
    for stem, digest in train_hashes.items():
        train_by_hash.setdefault(digest, []).append(stem)
    for stem, digest in val_hashes.items():
        for train_stem in train_by_hash.get(digest, []):
            exact.append({"train": train_stem, "val": stem})
    near = []
    for train_stem, train_hash in train_dhashes.items():
        for val_stem, val_hash in val_dhashes.items():
            distance = (train_hash ^ val_hash).bit_count()
            if distance <= args.near_duplicate_distance:
                near.append({"train": train_stem, "val": val_stem, "dhash_distance": distance})
    report = {
        "dataset_yaml": str(data_yaml.relative_to(ROOT)) if data_yaml.is_relative_to(ROOT) else str(data_yaml),
        "class_mapping": names,
        "split_reports": {"train": train, "val": val},
        "exact_cross_split_duplicates": exact,
        "near_cross_split_candidates": near,
        "near_duplicate_method": f"64-bit dHash, Hamming distance <= {args.near_duplicate_distance}",
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Train: {train['images']} images, {train['labels']} labels, {sum(train['objects_by_class'].values())} objects")
    print(f"Val: {val['images']} images, {val['labels']} labels, {sum(val['objects_by_class'].values())} objects")
    print(f"Missing labels: train={len(train['missing_labels'])}, val={len(val['missing_labels'])}")
    print(f"Invalid boxes/IDs: train={len(train['invalid_lines'])}, val={len(val['invalid_lines'])}")
    print(f"Corrupt images: train={len(train['corrupt_images'])}, val={len(val['corrupt_images'])}")
    print(f"Exact cross-split duplicates: {len(exact)}; near-duplicate candidates: {len(near)}")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
