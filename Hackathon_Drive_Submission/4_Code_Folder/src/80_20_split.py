"""Unified 80/20 train/validation splitter for Road Damage Project datasets.

Supports Pascal VOC image/XML datasets and existing YOLO train/val datasets.
Outputs are written to a new directory; a non-empty output is never replaced.

Examples (run from ``src``)::

    # Multi-country images + VOC XML -> VOC split
    python 80_20_split.py --mode voc

    # Small India images + XML -> VOC split
    python 80_20_split.py --mode voc \
        --source dataset/subsets/small/India \
        --output dataset/subsets/small/India/split_80_20

    # Raw India images + XML -> YOLO train/val labels
    python 80_20_split.py --mode voc --format yolo \
        --images dataset/raw/India/India/train/images \
        --annotations dataset/raw/India/India/train/annotations/xmls \
        --output dataset/processed/india_yolo_80_20

    # Re-split an existing YOLO dataset, grouping cross-split near duplicates
    python 80_20_split.py --mode yolo \
        --source dataset/subsets/multicountry/RDD2022/yolo \
        --output dataset/subsets/multicountry/RDD2022/yolo_grouped_80_20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / ".git").is_dir()
)
DEFAULT_SOURCE = ROOT / "dataset" / "subsets" / "multicountry" / "RDD2022"
CLASSES = ["D00", "D01", "D10", "D11", "D20", "D40", "D43", "D44", "D50", "D0w0"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _ensure_new_output(output: Path, input_paths: list[Path]) -> None:
    output = output.resolve()
    for source in input_paths:
        source = source.resolve()
        if source == output or output in source.parents:
            raise ValueError(f"Output must be separate from input paths: {output}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dhash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((9, 8)).tobytes())
    value = 0
    for row in range(8):
        for col in range(8):
            value = (value << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return value


def _stratified_split(items: list[dict], ratio: float, seed: int, stratify_country: bool):
    """Stratify while keeping byte-identical images in one partition."""
    exact_groups = defaultdict(list)
    for item in items:
        exact_groups[item["sha"]].append(item)
    strata = defaultdict(list)
    for members in exact_groups.values():
        item = members[0]
        group = item["status"]
        if stratify_country:
            group = (item["country"], group)
        strata[group].append(members)

    rng = random.Random(seed)
    train, val = [], []
    for group in sorted(strata, key=str):
        members = strata[group]
        rng.shuffle(members)
        train_count = int(sum(len(member_group) for member_group in members) * ratio)
        if len(members) > 1:
            train_count = min(max(train_count, 1), sum(len(member_group) for member_group in members) - 1)
        current_count = 0
        for member_group in members:
            if current_count < train_count:
                train.extend(member_group)
                current_count += len(member_group)
            else:
                val.extend(member_group)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _voc_to_yolo(xml_path: Path, width: int, height: int) -> tuple[list[str], set[str]]:
    root = ET.parse(xml_path).getroot()
    labels, present_classes = [], set()
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in CLASS_TO_ID:
            raise ValueError(f"Unknown class {name!r} in {xml_path}")
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bndbox in {xml_path}")
        x1 = float(box.findtext("xmin"))
        y1 = float(box.findtext("ymin"))
        x2 = float(box.findtext("xmax"))
        y2 = float(box.findtext("ymax"))
        x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
        y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bounding box in {xml_path}")
        xc, yc = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
        bw, bh = (x2 - x1) / width, (y2 - y1) / height
        labels.append(f"{CLASS_TO_ID[name]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        present_classes.add(name)
    return labels, present_classes


def run_voc(args) -> dict:
    source = _resolve(args.source)
    images_dir = _resolve(args.images) if args.images else source / "images"
    xml_dir = _resolve(args.annotations) if args.annotations else source / "annotations" / "xmls"
    output = _resolve(args.output) if args.output else source / "split_80_20"
    _ensure_new_output(output, [images_dir, xml_dir])
    if not images_dir.is_dir() or not xml_dir.is_dir():
        raise FileNotFoundError(f"Expected image directory {images_dir} and XML directory {xml_dir}")

    records = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        xml_path = xml_dir / f"{image_path.stem}.xml"
        if not xml_path.is_file():
            if args.allow_missing_xml:
                labels, names, status = [], set(), "normal"
            else:
                raise FileNotFoundError(f"Missing XML for {image_path.name}: {xml_path}")
        else:
            # Parsing up front means malformed annotations fail before writing output.
            with Image.open(image_path) as image:
                labels, names = _voc_to_yolo(xml_path, *image.size)
            status = "damaged" if names else "normal"
        records.append({
            "image": image_path,
            "xml": xml_path if xml_path.is_file() else None,
            "labels": labels,
            "classes": names,
            "sha": _sha256(image_path),
            "status": status,
            "country": image_path.stem.split("_", 1)[0],
        })

    if len(records) < 2:
        raise ValueError(f"Need at least two images to split; found {len(records)}")
    train, val = _stratified_split(records, args.train_ratio, args.seed, args.stratify_country)
    if not train or not val:
        raise ValueError("Split produced an empty train or validation set")

    for split, subset in (("train", train), ("val", val)):
        for record in subset:
            image_path = record["image"]
            if args.format == "voc":
                image_out = output / split / "images" / image_path.name
                image_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, image_out)
                if record["xml"]:
                    xml_out = output / split / "annotations" / "xmls" / record["xml"].name
                    xml_out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(record["xml"], xml_out)
            else:
                image_out = output / "images" / split / image_path.name
                label_out = output / "labels" / split / f"{image_path.stem}.txt"
                image_out.parent.mkdir(parents=True, exist_ok=True)
                label_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, image_out)
                label_out.write_text("\n".join(record["labels"]), encoding="utf-8")

    report = {
        "mode": "voc",
        "format": args.format,
        "source_images": str(images_dir),
        "source_annotations": str(xml_dir),
        "output": str(output),
        "train_ratio_requested": args.train_ratio,
        "seed": args.seed,
        "stratify_country": args.stratify_country,
        "image_counts": {"train": len(train), "val": len(val)},
        "damage_status_counts": {
            split: dict(Counter(record["status"] for record in subset))
            for split, subset in (("train", train), ("val", val))
        },
    }
    if args.format == "yolo":
        yaml_lines = [f"path: {output.as_posix()}", "train: images/train", "val: images/val", "", f"nc: {len(CLASSES)}", "names:"]
        yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASSES))
        (output / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    report_path = output / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


class _DisjointSet:
    def __init__(self, keys):
        self.parent = {key: key for key in keys}

    def find(self, key):
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def run_yolo(args) -> dict:
    source = _resolve(args.source)
    if not (source / "images" / "train").is_dir() and (source / "yolo" / "images" / "train").is_dir():
        source = source / "yolo"
    output = _resolve(args.output) if args.output else source.with_name(source.name + "_grouped_80_20")
    _ensure_new_output(output, [source])
    records = {}
    for split in ("train", "val"):
        image_dir, label_dir = source / "images" / split, source / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing YOLO images/labels for split {split} under {source}")
        for image_path in sorted(image_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label file for {image_path}")
            key = f"{split}/{image_path.relative_to(image_dir).as_posix()}"
            label_lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            records[key] = {
                "split": split, "image": image_path, "label": label_path,
                "sha": _sha256(image_path), "dhash": _dhash(image_path),
                "classes": [int(line.split()[0]) for line in label_lines],
            }
    if len(records) < 2:
        raise ValueError("Need at least two images across YOLO train/val")

    keys = sorted(records)
    groups = _DisjointSet(keys)
    duplicate_edges = []
    # Compare pairs from opposite current splits to keep this audit manageable.
    train_keys = [key for key in keys if records[key]["split"] == "train"]
    val_keys = [key for key in keys if records[key]["split"] == "val"]
    for train_key in train_keys:
        left = records[train_key]
        for val_key in val_keys:
            right = records[val_key]
            distance = (left["dhash"] ^ right["dhash"]).bit_count()
            exact = left["sha"] == right["sha"]
            if exact or distance <= args.near_duplicate_distance:
                groups.union(train_key, val_key)
                duplicate_edges.append({"train": train_key, "val": val_key, "dhash_distance": distance, "exact": exact})

    components = defaultdict(list)
    for key in keys:
        components[groups.find(key)].append(key)
    target_split = {}
    moved_from_val = []
    rng = random.Random(args.seed)
    components_list = list(components.values())
    rng.shuffle(components_list)
    target_val = round(len(records) * (1.0 - args.train_ratio))
    val_count = 0
    for members in components_list:
        existing_splits = {records[key]["split"] for key in members}
        if existing_splits == {"train", "val"}:
            destination = "train"  # preserve the established grouped-split policy
        elif existing_splits == {"val"} and val_count < target_val:
            destination = "val"
            val_count += len(members)
        else:
            destination = "train"
        for key in members:
            target_split[key] = destination
            if records[key]["split"] == "val" and destination == "train":
                moved_from_val.append(key)

    class_counts = {"train": Counter(), "val": Counter()}
    for key, destination in target_split.items():
        record = records[key]
        image_out = output / "images" / destination / record["image"].name
        label_out = output / "labels" / destination / record["label"].name
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["image"], image_out)
        shutil.copy2(record["label"], label_out)
        class_counts[destination].update(record["classes"])

    source_yaml = source / "data.yaml"
    if source_yaml.is_file():
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to copy the source YOLO data.yaml") from exc
        config = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
        config.update({"path": output.as_posix(), "train": "images/train", "val": "images/val"})
        (output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    report = {
        "mode": "yolo",
        "source": str(source),
        "output": str(output),
        "train_ratio_requested": args.train_ratio,
        "seed": args.seed,
        "near_duplicate_distance": args.near_duplicate_distance,
        "near_duplicate_method": "64-bit dHash plus exact SHA-256 across original train/val pairs",
        "near_duplicate_edges": duplicate_edges,
        "moved_validation_images_to_train": moved_from_val,
        "image_counts": {split: sum(value == split for value in target_split.values()) for split in ("train", "val")},
        "object_counts_by_class_id": {split: dict(counter) for split, counter in class_counts.items()},
        "group_count": len(components),
    }
    (output / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("voc", "yolo"), required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--images", type=Path, help="VOC source image folder override")
    parser.add_argument("--annotations", type=Path, help="VOC XML folder override")
    parser.add_argument("--format", choices=("voc", "yolo"), default="voc", help="Output format for --mode voc")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-country-stratify", action="store_false", dest="stratify_country", help="Do not split country-prefixed image groups independently")
    parser.set_defaults(stratify_country=True)
    parser.add_argument("--allow-missing-xml", action="store_true", help="Treat missing VOC annotations as undamaged images")
    parser.add_argument("--near-duplicate-distance", type=int, default=4, help="YOLO dHash Hamming threshold")
    args = parser.parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        parser.error("--train-ratio must be between 0 and 1")
    if args.near_duplicate_distance < 0:
        parser.error("--near-duplicate-distance must be non-negative")

    report = run_voc(args) if args.mode == "voc" else run_yolo(args)
    print(f"Created {args.mode} 80/20 split at: {report['output']}")
    print(f"Train: {report['image_counts']['train']}; validation: {report['image_counts']['val']}")
    print(f"Report: {Path(report['output']) / 'split_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
