"""Step 5.2: 10-Fold Cross-Validation Fold Generator.

Creates 10 stratified, group-aware cross-validation folds.
Ensures that images from the same camera run, sequence, or near-duplicate group
remain in the exact same fold, eliminating data leakage across CV iterations.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_group_id(path: Path) -> str:
    return path.stem.split("_obj")[0]


def generate_10_folds(
    source_dir: Path,
    output_dir: Path,
    num_folds: int = 10,
    seed: int = 42,
) -> dict:
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Collect images grouped by class and group_id
    class_groups = defaultdict(lambda: defaultdict(list))
    for class_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
        cls_name = class_dir.name
        for img_path in sorted(class_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                gid = get_group_id(img_path)
                class_groups[cls_name][gid].append(img_path)

    # Initialize folds
    folds: list[list[tuple[Path, str]]] = [[] for _ in range(num_folds)]
    fold_class_counts: list[dict[str, int]] = [defaultdict(int) for _ in range(num_folds)]

    # Distribute groups into folds using round-robin on shuffled groups per class
    for cls_name, groups_dict in sorted(class_groups.items()):
        group_keys = list(groups_dict.keys())
        rng.shuffle(group_keys)

        for idx, gid in enumerate(group_keys):
            assigned_fold = idx % num_folds
            for img_path in groups_dict[gid]:
                folds[assigned_fold].append((img_path, cls_name))
                fold_class_counts[assigned_fold][cls_name] += 1

    # Save manifest for each fold
    manifest_rows = []
    fold_stats = {}
    for fold_idx in range(num_folds):
        fold_items = folds[fold_idx]
        fold_stats[f"fold_{fold_idx}"] = {
            "total_samples": len(fold_items),
            "class_counts": dict(fold_class_counts[fold_idx]),
        }
        for img_path, cls_name in fold_items:
            manifest_rows.append({
                "fold": fold_idx,
                "class_name": cls_name,
                "filename": img_path.name,
                "path": str(img_path),
                "group_id": get_group_id(img_path),
            })

    cv_manifest_path = output_dir / "10_fold_manifest.csv"
    with cv_manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["fold", "class_name", "filename", "path", "group_id"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return {
        "num_folds": num_folds,
        "manifest_path": str(cv_manifest_path),
        "folds": fold_stats,
    }
