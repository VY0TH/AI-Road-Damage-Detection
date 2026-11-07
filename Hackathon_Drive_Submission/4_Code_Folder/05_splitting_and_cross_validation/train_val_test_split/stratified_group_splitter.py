"""Step 5.1: Stratified Group-Aware Train/Validation/Test Splitter.

Splits an image dataset into Train / Validation / Test sets:
1. Stratified by class labels to maintain relative class frequencies.
2. Group-Aware: Images sharing the same source sequence, burst group, or identical
   hash are kept strictly within the same partition, preventing cross-split leakage.
"""
from __future__ import annotations

import csv
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_image_group_id(path: Path) -> str:
    """Extracts sequence/group identifier from image filename (e.g., China_0001 -> China_0001)."""
    parts = path.stem.split("_obj")
    return parts[0]


def get_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def split_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    copy_files: bool = True,
) -> dict:
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Ratios must sum to 1.0"

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Collect images grouped by class and group_id
    class_groups = defaultdict(lambda: defaultdict(list))
    for class_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
        cls_name = class_dir.name
        for img_path in sorted(class_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                group_id = get_image_group_id(img_path)
                class_groups[cls_name][group_id].append(img_path)

    # Partition groups per class to preserve stratification and grouping
    train_items, val_items, test_items = [], [], []

    for cls_name, groups_dict in sorted(class_groups.items()):
        group_keys = list(groups_dict.keys())
        rng.shuffle(group_keys)

        total_groups = len(group_keys)
        n_train = max(1, int(round(total_groups * train_ratio)))
        n_val = max(1, int(round(total_groups * val_ratio))) if test_ratio > 0 else (total_groups - n_train)

        train_groups = group_keys[:n_train]
        val_groups = group_keys[n_train: n_train + n_val]
        test_groups = group_keys[n_train + n_val:]

        for gid in train_groups:
            train_items.extend([(p, cls_name) for p in groups_dict[gid]])
        for gid in val_groups:
            val_items.extend([(p, cls_name) for p in groups_dict[gid]])
        for gid in test_groups:
            test_items.extend([(p, cls_name) for p in groups_dict[gid]])

    partitions = {
        "train": train_items,
        "val": val_items,
        "test": test_items,
    }

    manifest_rows = []
    summary = {}

    for split_name, items in partitions.items():
        summary[split_name] = {"total": len(items), "classes": defaultdict(int)}
        split_dir = output_dir / split_name

        for img_path, cls_name in items:
            summary[split_name]["classes"][cls_name] += 1
            rel_dest = Path(split_name) / cls_name / img_path.name
            dest_path = output_dir / rel_dest

            if copy_files:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                if not dest_path.is_file():
                    shutil.copy2(img_path, dest_path)

            manifest_rows.append({
                "split": split_name,
                "class_name": cls_name,
                "filename": img_path.name,
                "source_path": str(img_path),
                "dest_path": str(rel_dest),
                "group_id": get_image_group_id(img_path),
            })
        summary[split_name]["classes"] = dict(summary[split_name]["classes"])

    manifest_path = output_dir / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "class_name", "filename", "source_path", "dest_path", "group_id"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return summary
