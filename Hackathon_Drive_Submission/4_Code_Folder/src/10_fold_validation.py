"""Prepare, preflight, or train leakage-controlled 10-fold YOLO validation.

The workflow requires a manually reviewed ``image,group_id`` CSV so visually
similar images stay in the same fold. Training requires generic pretrained
weights located outside this project's ``results`` directory.

Run from the project root or ``src`` directory::

    python src/10_fold_validation.py prepare
    python src/10_fold_validation.py preflight
    python src/10_fold_validation.py train --weights path/to/yolo12n.pt

Use ``--help`` after a command for dataset, fold, and training options.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import yaml


ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / ".git").is_dir()
)
CLASSES = ["D00", "D01", "D10", "D11", "D20", "D40", "D43", "D44", "D50", "D0w0"]
DEFAULT_IMAGES = ROOT / "dataset/subsets/multicountry/RDD2022/yolo/images/train"
DEFAULT_LABELS = ROOT / "dataset/subsets/multicountry/RDD2022/yolo/labels/train"
DEFAULT_GROUPS = ROOT / "results/cross_validation_clean/reviewed_groups.csv"
DEFAULT_OUTPUT = ROOT / "results/cross_validation_10fold_merged"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_reviewed_groups(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Reviewed image groups CSV not found: {path}. It must have image and group_id columns."
        )
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"image", "group_id"}.issubset(rows[0]):
        raise ValueError("Reviewed groups CSV requires image and group_id columns.")
    groups = {}
    for row in rows:
        stem = Path(row["image"].strip()).stem
        group = row["group_id"].strip()
        if not stem or not group:
            raise ValueError(f"Blank image or group_id in reviewed groups CSV: {row}")
        if stem in groups and groups[stem] != group:
            raise ValueError(f"Conflicting group IDs for image stem {stem}")
        groups[stem] = group
    return groups


def read_yolo_classes(label_path: Path) -> set[str]:
    present = set()
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected 5 YOLO fields at {label_path}:{line_number}: {line}")
        try:
            class_id = int(fields[0])
        except ValueError as exc:
            raise ValueError(f"Invalid class ID at {label_path}:{line_number}") from exc
        if class_id < 0 or class_id >= len(CLASSES):
            raise ValueError(f"Class ID out of range at {label_path}:{line_number}: {class_id}")
        for value in fields[1:]:
            number = float(value)
            if not 0.0 <= number <= 1.0:
                raise ValueError(f"YOLO coordinate outside [0,1] at {label_path}:{line_number}")
        present.add(CLASSES[class_id])
    return present


def collect_samples(images_dir: Path, labels_dir: Path, groups: dict[str, str]) -> list[dict]:
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing images or labels directory: {images_dir}, {labels_dir}")
    images = sorted(
        path.resolve() for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No supported images found in {images_dir}")
    samples = []
    seen_stems = set()
    for image in images:
        if image.stem in seen_stems:
            raise ValueError(f"Duplicate image stem in input directory: {image.stem}")
        seen_stems.add(image.stem)
        label = (labels_dir / f"{image.stem}.txt").resolve()
        if not label.is_file():
            raise FileNotFoundError(f"Missing YOLO label for {image}: {label}")
        if image.stem not in groups:
            raise ValueError(f"Image missing from reviewed groups CSV: {image.name}")
        samples.append({
            "image": image,
            "label": label,
            "group": groups[image.stem],
            "classes": read_yolo_classes(label),
            "country": image.stem.split("_", 1)[0],
        })
    extra = set(groups) - seen_stems
    if extra:
        raise ValueError(f"Reviewed groups CSV has stems absent from active image set: {sorted(extra)[:10]}")
    return samples


def assign_folds(samples: list[dict], folds: int, seed: int):
    if folds < 2:
        raise ValueError("At least two folds are required.")
    grouped = defaultdict(list)
    for index, row in enumerate(samples):
        grouped[row["group"]].append(index)
    if len(grouped) < folds:
        raise ValueError(f"Need at least {folds} distinct reviewed groups; found {len(grouped)}")

    totals = Counter(name for row in samples for name in row["classes"])
    target_size = len(samples) / folds
    assigned = [[] for _ in range(folds)]
    sizes = [0] * folds
    class_counts = [Counter() for _ in range(folds)]
    bundles = list(grouped.values())
    random.Random(seed).shuffle(bundles)
    bundles.sort(key=lambda group: (
        -sum(1 / max(1, totals[name]) for index in group for name in samples[index]["classes"]),
        -len(group),
    ))

    for group in bundles:
        group_classes = Counter(
            name for index in group for name in samples[index]["classes"]
        )

        def score(fold_index: int) -> float:
            size_cost = sum(
                ((sizes[k] + (len(group) if k == fold_index else 0) - target_size) / target_size) ** 2
                for k in range(folds)
            )
            class_cost = 0.0
            for k in range(folds):
                for name in CLASSES:
                    target = max(totals[name] / folds, 0.25)
                    value = class_counts[k][name] + (group_classes[name] if k == fold_index else 0)
                    class_cost += ((value - target) / target) ** 2
            return size_cost + 0.35 * class_cost

        target_fold = min(range(folds), key=score)
        assigned[target_fold].extend(group)
        sizes[target_fold] += len(group)
        class_counts[target_fold].update(group_classes)

    minimum_size = 0.85 * target_size
    missing_d40 = [index + 1 for index, counts in enumerate(class_counts) if counts["D40"] == 0]
    if min(sizes) < minimum_size or missing_d40:
        d40_by_fold = [counts["D40"] for counts in class_counts]
        raise ValueError(
            f"Unsafe fold balance: sizes={sizes}, D40 counts={d40_by_fold}. "
            "Review group assignments before creating folds."
        )
    return assigned, sizes, class_counts


def prepare(args) -> Path:
    groups_path = resolve(args.reviewed_groups)
    images_dir, labels_dir = resolve(args.images), resolve(args.labels)
    output = resolve(args.output)
    groups = load_reviewed_groups(groups_path)
    samples = collect_samples(images_dir, labels_dir, groups)
    assigned, sizes, class_counts = assign_folds(samples, args.folds, args.seed)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty fold output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = output / "grouped_fold_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["image", "label", "country", "classes", "group_id", "fold"]
        )
        writer.writeheader()
        for fold_number, indices in enumerate(assigned, 1):
            for index in indices:
                row = samples[index]
                writer.writerow({
                    "image": row["image"].as_posix(),
                    "label": row["label"].as_posix(),
                    "country": row["country"],
                    "classes": "|".join(sorted(row["classes"])),
                    "group_id": row["group"],
                    "fold": fold_number,
                })

    for fold_number, valid_indices in enumerate(assigned, 1):
        fold_dir = output / f"fold_{fold_number}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        valid_set = set(valid_indices)
        train_indices = [index for index in range(len(samples)) if index not in valid_set]
        train_list = fold_dir / "train.txt"
        val_list = fold_dir / "val.txt"
        train_list.write_text("\n".join(samples[i]["image"].as_posix() for i in train_indices) + "\n", encoding="utf-8")
        val_list.write_text("\n".join(samples[i]["image"].as_posix() for i in valid_indices) + "\n", encoding="utf-8")
        config = {
            "path": images_dir.parent.parent.as_posix(),
            "train": train_list.resolve().as_posix(),
            "val": val_list.resolve().as_posix(),
            "nc": len(CLASSES),
            "names": CLASSES,
        }
        (fold_dir / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(
            f"Fold {fold_number}: train={len(train_indices)}, validation={len(valid_indices)}, "
            f"images_with_D40={class_counts[fold_number - 1]['D40']}"
        )

    report = {
        "folds": args.folds,
        "seed": args.seed,
        "image_count": len(samples),
        "group_count": len({row["group"] for row in samples}),
        "fold_sizes": sizes,
        "d40_images_by_fold": [counts["D40"] for counts in class_counts],
        "reviewed_groups_csv": str(groups_path),
    }
    (output / "fold_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Prepared folds under: {output}")
    print(f"Assignment manifest: {manifest_path}")
    return output


def _load_fold_paths(config_path: Path) -> tuple[list[Path], list[Path], dict]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    names = config.get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names)]
    if config.get("nc") != len(CLASSES) or names != CLASSES:
        raise ValueError(f"Class mapping mismatch in {config_path}")

    def read_list(key: str) -> list[Path]:
        list_path = Path(config[key])
        if not list_path.is_absolute():
            list_path = config_path.parent / list_path
        if list_path.suffix.lower() != ".txt":
            raise ValueError(f"{config_path}: {key} must point to an image-list .txt file")
        base = Path(config.get("path", config_path.parent))
        if not base.is_absolute():
            base = config_path.parent / base
        return [
            item if item.is_absolute() else base / item
            for item in (Path(line.strip()) for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip())
        ]

    return read_list("train"), read_list("val"), config


def preflight(args) -> list[dict]:
    output = resolve(args.output)
    groups = load_reviewed_groups(resolve(args.reviewed_groups))
    configs, validation_sets = [], []
    for fold_number in range(1, args.folds + 1):
        config_path = output / f"fold_{fold_number}" / "data.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Fold config missing: {config_path}. Run the prepare command first.")
        train, val, _ = _load_fold_paths(config_path)
        train_set = {str(path.resolve()) for path in train}
        val_set = {str(path.resolve()) for path in val}
        if not train_set or not val_set or train_set & val_set:
            raise ValueError(f"Fold {fold_number} has empty or overlapping train/validation lists")
        for image_path in train + val:
            if not image_path.is_file():
                raise FileNotFoundError(f"Fold {fold_number} image missing: {image_path}")
            if image_path.stem not in groups:
                raise ValueError(f"Image not found in reviewed groups CSV: {image_path}")
            parts = image_path.parts
            if "images" not in parts:
                raise ValueError(f"Cannot derive YOLO label path from {image_path}")
            image_pos = parts.index("images")
            label_path = Path(*parts[:image_pos], "labels", *parts[image_pos + 1:-1], image_path.stem + ".txt")
            if not label_path.is_file():
                raise FileNotFoundError(f"Fold {fold_number} label missing: {label_path}")
        train_groups = {groups[Path(path).stem] for path in train_set}
        val_groups = {groups[Path(path).stem] for path in val_set}
        if train_groups & val_groups:
            raise ValueError(f"Fold {fold_number} splits a manually reviewed duplicate group")
        if not any(
            "5" == line.split()[0]
            for image_path in val
            for line in Path(*image_path.parts[:image_path.parts.index("images")], "labels", *image_path.parts[image_path.parts.index("images") + 1:-1], image_path.stem + ".txt").read_text(encoding="utf-8").splitlines()
            if line.split()
        ):
            raise ValueError(f"Fold {fold_number} validation has no D40 (class ID 5) objects")
        validation_sets.append((train_set, val_set))
        configs.append({"fold": fold_number, "config": config_path, "train_images": len(train), "validation_images": len(val)})

    validation_images = [image for _train, val in validation_sets for image in val]
    universe = set(validation_images)
    if len(validation_images) != len(universe):
        raise ValueError("Validation folds repeat images")
    for fold_number, (train_set, val_set) in enumerate(validation_sets, 1):
        if train_set != universe - val_set:
            raise ValueError(f"Fold {fold_number} train set is not the complement of its validation fold")
    print(json.dumps({"preflight": "passed", "folds": configs}, indent=2, default=str))
    return configs


def train(args, configs: list[dict]) -> None:
    weights = resolve(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Generic pretrained weights not found: {weights}")
    try:
        weights.resolve().relative_to((ROOT / "results").resolve())
    except ValueError:
        pass
    else:
        raise ValueError("Use generic pretrained weights outside results/; project-trained weights contaminate CV.")

    from ultralytics import YOLO, settings

    output = ROOT / "results" / "cross_validation_10fold_training" / datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite training output: {output}")
    output.mkdir(parents=True)
    settings.update({"runs_dir": str((ROOT / "results").resolve())})
    state = {"state": "running", "weights": str(weights.resolve()), "epochs": args.epochs, "completed_folds": []}
    status_path = output / "status.json"
    status_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    fold_results = []
    try:
        for entry in configs:
            fold_number = entry["fold"]
            state["current_fold"] = fold_number
            status_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            model = YOLO(str(weights.resolve()))
            result = model.train(
                data=str(entry["config"].resolve()), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=0,
                project=str(output.resolve()), name=f"fold_{fold_number}", exist_ok=False,
                pretrained=True, plots=False, save=True, val=True, amp=False, cache=False,
                seed=args.seed + fold_number, deterministic=True, patience=args.epochs,
            )
            metrics = {
                key: float(value) for key, value in getattr(result, "results_dict", {}).items()
                if isinstance(value, (int, float))
            }
            record = {
                "fold": fold_number,
                "metrics": metrics,
                "best_checkpoint": str(output / f"fold_{fold_number}" / "weights" / "best.pt"),
            }
            fold_results.append(record)
            (output / f"fold_{fold_number}" / "fold_metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            state["completed_folds"].append(fold_number)
            status_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except KeyboardInterrupt:
        state["state"] = "interrupted"
        status_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        raise
    except Exception as exc:
        state["state"] = "failed"
        state["error"] = str(exc)
        status_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        raise

    metric_keys = set.intersection(*(set(item["metrics"]) for item in fold_results)) if fold_results else set()
    summary = {"weights": str(weights.resolve()), "epochs": args.epochs, "folds": fold_results, "means": {}, "stddevs": {}}
    for key in metric_keys:
        values = [item["metrics"][key] for item in fold_results]
        summary["means"][key] = statistics.mean(values)
        summary["stddevs"][key] = statistics.stdev(values) if len(values) > 1 else 0.0
    (output / "cross_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    state["state"] = "completed"
    state.pop("current_fold", None)
    status_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"Completed 10-fold training outputs: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "train"))
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--reviewed-groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", type=Path, help="Generic pretrained YOLO checkpoint, outside results/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.folds < 2 or args.epochs < 1 or args.imgsz < 1 or args.batch < 1:
        parser.error("folds, epochs, imgsz, and batch must be positive (folds >= 2)")

    if args.command == "prepare":
        prepare(args)
        return 0
    configs = preflight(args)
    if args.command == "train":
        if args.weights is None:
            parser.error("train requires --weights pointing to generic pretrained weights")
        train(args, configs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
