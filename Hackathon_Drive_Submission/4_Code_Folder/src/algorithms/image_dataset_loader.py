"""Shared PyTorch image dataset loader for class-named folders.

Expected layout::

    root/<class name>/<image file>

Transforms are supplied by each model so the loader does not impose a model-
specific image size, normalization, or augmentation policy.
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ImageFolderDataset(Dataset):
    """Load labeled RGB images from class folders, optionally adding samples.

    ``extra_roots`` are additional class-folder roots (for example generated
    training images). They use the same class names and label indices as root.
    """

    def __init__(self, root, class_names, transform=None, extra_roots=(), extensions=None):
        self.root = Path(root)
        self.class_names = list(class_names)
        self.class_to_idx = {name: index for index, name in enumerate(self.class_names)}
        self.transform = transform
        self.extensions = {ext.lower() for ext in (extensions or IMAGE_EXTENSIONS)}
        self.samples = []

        for sample_root in (self.root, *(Path(path) for path in extra_roots)):
            for class_name, label in self.class_to_idx.items():
                class_dir = sample_root / class_name
                if class_dir.is_dir():
                    self.samples.extend(
                        (path, label)
                        for path in sorted(class_dir.iterdir())
                        if path.is_file() and path.suffix.lower() in self.extensions
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
