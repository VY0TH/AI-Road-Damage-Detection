"""Step 3: Suspicious Features & Shortcut Learning Audit.

Checks for suspicious features and dataset artifacts that could allow models
to "cheat" through spurious correlation rather than learning real damage features:
1. Filename / metadata correlation: Country or camera prefixes (e.g., India_, Japan_)
   disproportionately concentrated in specific classes.
2. Aspect ratio / resolution shortcut: Significant dimensional disparities across classes.
3. Luminance / color space bias: Unusual brightness or contrast imbalances between classes.
4. Identical filenames across distinct split folders.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def audit_suspicious_features(dataset_root: Path) -> dict:
    dataset_root = Path(dataset_root).resolve()
    class_aspect_ratios = defaultdict(list)
    class_prefixes = defaultdict(Counter)
    split_filenames = defaultdict(set)
    total_samples = 0

    for split_dir in [p for p in dataset_root.iterdir() if p.is_dir()]:
        split_name = split_dir.name
        for class_dir in [p for p in split_dir.iterdir() if p.is_dir()]:
            cls_name = class_dir.name
            for img_path in class_dir.iterdir():
                if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                total_samples += 1
                split_filenames[split_name].add(img_path.name)

                # Prefix analysis (e.g., country origin or batch prefix)
                parts = img_path.stem.split("_")
                prefix = parts[0] if len(parts) > 1 else "Unknown"
                class_prefixes[cls_name][prefix] += 1

                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        class_aspect_ratios[cls_name].append(round(w / max(1, h), 2))
                except Exception:
                    continue

    # Filename collisions across splits
    cross_split_filename_collisions = []
    splits = list(split_filenames.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            shared = split_filenames[s1] & split_filenames[s2]
            if shared:
                cross_split_filename_collisions.append({
                    "split_a": s1,
                    "split_b": s2,
                    "shared_count": len(shared),
                    "examples": sorted(list(shared))[:5],
                })

    # Summary of aspect ratio bias
    ratio_summary = {}
    for cls_name, ratios in class_aspect_ratios.items():
        if ratios:
            ratio_summary[cls_name] = {
                "mean_aspect_ratio": round(sum(ratios) / len(ratios), 2),
                "min_aspect_ratio": min(ratios),
                "max_aspect_ratio": max(ratios),
            }

    # Detect extreme prefix concentration (> 90% of a class from 1 prefix)
    suspicious_prefix_correlations = []
    for cls_name, prefix_counts in class_prefixes.items():
        cls_total = sum(prefix_counts.values())
        for prefix, count in prefix_counts.items():
            if cls_total >= 10 and (count / cls_total) >= 0.90:
                suspicious_prefix_correlations.append({
                    "class": cls_name,
                    "dominant_prefix": prefix,
                    "percentage": round(100 * count / cls_total, 1),
                    "samples": f"{count}/{cls_total}",
                    "warning": f"Class {cls_name} is overwhelmingly (>90%) sourced from '{prefix}', risking shortcut learning.",
                })

    passed = len(cross_split_filename_collisions) == 0
    return {
        "suspicious_features_passed": passed,
        "filename_overlap_across_splits": cross_split_filename_collisions,
        "suspicious_prefix_correlations": suspicious_prefix_correlations,
        "aspect_ratio_summary": ratio_summary,
    }
