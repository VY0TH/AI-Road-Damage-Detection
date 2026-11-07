"""Create a separate train/validation split that keeps near-duplicate images together."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
SOURCE = ROOT / "dataset/subsets/multicountry/RDD2022/yolo"
SOURCE_YAML = SOURCE / "data.yaml"
DEFAULT_OUTPUT = ROOT / "dataset/subsets/multicountry/RDD2022/yolo_grouped"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _dhash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((9, 8)).tobytes())
    result = 0
    for row in range(8):
        for col in range(8):
            result = (result << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return result


class DisjointSet:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--near-duplicate-distance", type=int, default=4)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty grouped dataset: {output}")
    if not SOURCE_YAML.is_file():
        raise FileNotFoundError(f"Source data.yaml not found: {SOURCE_YAML}")

    files = {}
    hashes = {}
    exact_hashes = {}
    for split in ("train", "val"):
        image_dir = SOURCE / "images" / split
        label_dir = SOURCE / "labels" / split
        for image_path in sorted(image_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing source label for {image_path}")
            key = f"{split}/{image_path.name}"
            files[key] = (split, image_path, label_path)
            hashes[key] = _dhash(image_path)
            exact_hashes[key] = _sha256(image_path)

    groups = DisjointSet(files)
    keys = sorted(files)
    duplicate_edges = []
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            if files[left][0] == files[right][0]:
                continue
            distance = (hashes[left] ^ hashes[right]).bit_count()
            exact_duplicate = exact_hashes[left] == exact_hashes[right]
            if exact_duplicate or distance <= args.near_duplicate_distance:
                groups.union(left, right)
                duplicate_edges.append({"left": left, "right": right, "dhash_distance": distance, "exact": exact_duplicate})

    components = defaultdict(list)
    for key in keys:
        components[groups.find(key)].append(key)
    destination_split = {}
    moved_from_val = []
    for members in components.values():
        split_set = {files[key][0] for key in members}
        target = "train" if split_set == {"train", "val"} else next(iter(split_set))
        for key in members:
            original_split = files[key][0]
            destination_split[key] = target
            if original_split == "val" and target == "train":
                moved_from_val.append(key)

    class_counts = {"train": Counter(), "val": Counter()}
    for key, target in destination_split.items():
        original_split, image_path, label_path = files[key]
        image_target = output / "images" / target / image_path.name
        label_target = output / "labels" / target / label_path.name
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, image_target)
        shutil.copy2(label_path, label_target)
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                class_counts[target][int(parts[0])] += 1

    config = yaml.safe_load(SOURCE_YAML.read_text(encoding="utf-8"))
    config["path"] = str(output.relative_to(ROOT)).replace("\\", "/")
    config["train"] = "images/train"
    config["val"] = "images/val"
    (output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    report = {
        "source_yaml": str(SOURCE_YAML.relative_to(ROOT)),
        "output_yaml": str((output / "data.yaml").relative_to(ROOT)),
        "near_duplicate_method": f"64-bit dHash plus exact SHA-256; group if Hamming distance <= {args.near_duplicate_distance}",
        "near_duplicate_edges": duplicate_edges,
        "moved_validation_images_to_train": moved_from_val,
        "image_counts": {split: sum(1 for value in destination_split.values() if value == split) for split in ("train", "val")},
        "object_counts_by_class_id": {split: dict(counter) for split, counter in class_counts.items()},
        "group_count": len(components),
    }
    report_path = output / "grouped_split_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Created grouped dataset at: {output}")
    print(f"Train images: {report['image_counts']['train']}; validation images: {report['image_counts']['val']}")
    print(f"Near-duplicate edges grouped: {len(duplicate_edges)}; moved validation images to train: {len(moved_from_val)}")
    print(f"Data config: {output / 'data.yaml'}")
    print(f"Report: {report_path}")


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
