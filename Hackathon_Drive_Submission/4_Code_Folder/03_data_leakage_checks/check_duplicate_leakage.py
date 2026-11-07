"""Step 3: Duplicate Leakage Detection.

Checks for two types of duplicate leakage across train/validation/test splits:
1. Exact Content Duplicates (Byte-for-byte SHA-256 hash collisions).
2. Perceptual Near-Duplicates (Difference hash / dHash with Hamming distance threshold),
   crucial for continuous dashcam or drone video frames where sequential frames
   are nearly identical.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(path: Path, hash_size: int = 8) -> int:
    """Compute difference hash (dHash) for perceptual visual similarity."""
    with Image.open(path) as img:
        # Resize to (hash_size + 1, hash_size) grayscale
        resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = list(resized.getdata())
    diff = []
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            diff.append(left > right)
    decimal_val = 0
    for idx, bit in enumerate(diff):
        if bit:
            decimal_val |= 1 << idx
    return decimal_val


def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def audit_duplicate_leakage(
    dataset_root: Path,
    splits: tuple[str, ...] = ("train", "val", "test"),
    near_dup_threshold: int = 4,  # Hamming distance <= 4 considered near-duplicate
    max_near_dup_checks: int = 1000,
) -> dict:
    dataset_root = Path(dataset_root).resolve()
    split_files: dict[str, list[Path]] = {}
    split_hashes: dict[str, dict[str, list[Path]]] = {}
    split_dhashes: dict[str, list[tuple[Path, int]]] = {}

    for s in splits:
        s_dir = dataset_root / s
        if not s_dir.is_dir():
            continue
        files = sorted(
            p for p in s_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        split_files[s] = files
        split_hashes[s] = defaultdict(list)
        for f in files:
            split_hashes[s][compute_sha256(f)].append(f)

    # 1. Exact Duplicate Cross-Split Leakage
    active_splits = list(split_files.keys())
    exact_leakage_groups = []
    for i in range(len(active_splits)):
        for j in range(i + 1, len(active_splits)):
            s1, s2 = active_splits[i], active_splits[j]
            shared_hashes = set(split_hashes[s1].keys()) & set(split_hashes[s2].keys())
            for h in shared_hashes:
                exact_leakage_groups.append({
                    "split_a": s1,
                    "split_b": s2,
                    "sha256": h,
                    "files_a": [str(p.name) for p in split_hashes[s1][h]],
                    "files_b": [str(p.name) for p in split_hashes[s2][h]],
                })

    # 2. Perceptual Near-Duplicates between Train and Val/Test
    near_duplicates = []
    if "train" in split_files:
        train_dhashes = [(f, compute_dhash(f)) for f in split_files["train"][:max_near_dup_checks]]
        for other_split in [s for s in active_splits if s != "train"]:
            other_dhashes = [(f, compute_dhash(f)) for f in split_files[other_split][:max_near_dup_checks]]
            for f_train, h_train in train_dhashes:
                for f_other, h_other in other_dhashes:
                    dist = hamming_distance(h_train, h_other)
                    if dist <= near_dup_threshold:
                        near_duplicates.append({
                            "train_file": f_train.name,
                            "other_split": other_split,
                            "other_file": f_other.name,
                            "hamming_distance": dist,
                        })
                        if len(near_duplicates) >= 20:
                            break
                if len(near_duplicates) >= 20:
                    break

    passed = len(exact_leakage_groups) == 0 and len(near_duplicates) == 0
    return {
        "duplicate_leakage_passed": passed,
        "exact_cross_split_duplicates_count": len(exact_leakage_groups),
        "exact_duplicates": exact_leakage_groups[:10],
        "near_duplicates_sample_count": len(near_duplicates),
        "near_duplicates_sample": near_duplicates[:10],
    }
