"""Step 1: Dataset Loader for Class-Folder and Manifest-Based Image Datasets.

Provides:
- ImageFolderDataset: PyTorch Dataset supporting `<split>/<class>/<image>` directory layouts
- ManifestDataset: PyTorch Dataset loading from CSV manifests
- DataLoader factory functions with configurable batch size, shuffling, transforms, and workers.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class ImageFolderDataset(Dataset):
    """Loads labeled RGB images from class folders, optionally including extra sample roots."""

    def __init__(
        self,
        root: str | Path,
        class_names: Sequence[str],
        transform: Callable | None = None,
        extra_roots: Sequence[str | Path] = (),
        extensions: set[str] | None = None,
    ):
        self.root = Path(root)
        self.class_names = list(class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.transform = transform
        self.extensions = {ext.lower() for ext in (extensions or IMAGE_EXTENSIONS)}
        self.samples: list[tuple[Path, int]] = []

        all_roots = [self.root] + [Path(p) for p in extra_roots]
        for sample_root in all_roots:
            if not sample_root.is_dir():
                continue
            for class_name, label in self.class_to_idx.items():
                class_dir = sample_root / class_name
                if class_dir.is_dir():
                    for path in sorted(class_dir.iterdir()):
                        if path.is_file() and path.suffix.lower() in self.extensions:
                            self.samples.append((path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor | Image.Image, int]:
        path, label = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def get_class_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.class_names}
        for _, label in self.samples:
            counts[self.class_names[label]] += 1
        return counts


class ManifestDataset(Dataset):
    """Loads images based on a CSV manifest containing columns: crop_path / image_path, class_name."""

    def __init__(
        self,
        manifest_path: str | Path,
        class_names: Sequence[str],
        base_dir: str | Path | None = None,
        split_filter: str | None = None,
        transform: Callable | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.class_names = list(class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.base_dir = Path(base_dir) if base_dir else self.manifest_path.parent
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with self.manifest_path.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split_filter and row.get("split") and row["split"] != split_filter:
                    continue
                path_str = row.get("crop_path") or row.get("image_path") or row.get("source_path")
                if not path_str:
                    continue
                path = Path(path_str)
                if not path.is_absolute():
                    path = self.base_dir / path
                class_name = row.get("class_name") or row.get("label")
                if class_name in self.class_to_idx and path.is_file():
                    self.samples.append((path, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor | Image.Image, int]:
        path, label = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def get_class_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.class_names}
        for _, label in self.samples:
            counts[self.class_names[label]] += 1
        return counts


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """Convenience wrapper for PyTorch DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )
