"""Train a class-conditional GAN and generate labeled augmentation samples.

This merges the selected-class conditional GAN trainer with its sample
generator. Training reads only ``split=train`` rows from the crop manifest;
generation writes images to class-named folders for classifier augmentation.

Examples from the project root::

    python src/algorithms/gan/gan_augmentation.py train
    python src/algorithms/gan/gan_augmentation.py generate

Generated images are experimental and should be reviewed before use in model
training. The class folder records the requested conditioning label, not a
guarantee that the generated image visibly belongs to that class.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image


ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / ".git").is_dir()
)
DEFAULT_MANIFEST = ROOT / "dataset/processed/efficientnet_crops/manifest.csv"
DEFAULT_CHECKPOINT_DIR = ROOT / "results/conditional_gan_selected_checkpoints"
DEFAULT_PREVIEW_DIR = ROOT / "results/conditional_gan_selected"
DEFAULT_SAMPLE_DIR = ROOT / "results/gan_individual_samples"
DEFAULT_CLASSES = ["D00", "D10", "D20", "D40", "D44"]
IMAGE_SIZE = 64
LATENT_DIM = 100
EMBED_DIM = 32


class CropDataset(Dataset):
    def __init__(self, rows: list[dict], class_to_id: dict[str, int], root: Path):
        self.rows = rows
        self.class_to_id = class_to_id
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = Path(row["crop_path"])
        if not image_path.is_absolute():
            image_path = self.root / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Training crop not found: {image_path}")
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.class_to_id[row["class_name"]]


class ConditionalGenerator(nn.Module):
    def __init__(self, num_classes: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, EMBED_DIM)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + EMBED_DIM, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.ReLU(True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        conditioned = torch.cat([noise, self.embedding(labels)], dim=1)
        return self.net(self.fc(conditioned).view(-1, 256, 4, 4))


class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, IMAGE_SIZE * IMAGE_SIZE)
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0),
        )

    def forward(self, images, labels):
        label_map = self.embedding(labels).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        return self.net(torch.cat([images, label_map], dim=1)).view(-1)


def read_training_rows(manifest: Path, root: Path, classes: list[str]) -> list[dict]:
    if not manifest.is_file():
        raise FileNotFoundError(f"Crop manifest not found: {manifest}")
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"split", "crop_path", "class_name"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Manifest must contain columns: {sorted(required)}")
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
            if (row.get("split") or "").strip().lower() == "train"
        ]
    rows = [row for row in rows if row["crop_path"] and row["class_name"] in classes]
    if not rows:
        raise ValueError("No training crops found for the selected classes.")
    for row in rows:
        path = Path(row["crop_path"])
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            raise FileNotFoundError(f"Manifest crop is missing: {path}")
    return rows


def train(args) -> None:
    root = ROOT
    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else ROOT / args.checkpoint_dir
    preview_dir = args.preview_dir if args.preview_dir.is_absolute() else ROOT / args.preview_dir
    classes = [name.strip() for name in args.classes.split(",") if name.strip()]
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("--classes must be a non-empty comma-separated list without duplicates")
    if args.batch_size < 2:
        raise ValueError("Batch size must be at least 2 for the generator's batch normalization")

    rows = read_training_rows(manifest, root, classes)
    counts = {name: sum(row["class_name"] == name for row in rows) for name in classes}
    print("Selected classes:", classes)
    print("Training crop counts:", counts)
    print("Device:", args.device)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    class_to_id = {name: index for index, name in enumerate(classes)}
    dataset = CropDataset(rows, class_to_id, root)
    if len(dataset) < args.batch_size:
        raise ValueError(f"Only {len(dataset)} selected training crops; batch size is {args.batch_size}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    generator = ConditionalGenerator(len(classes)).to(device)
    discriminator = ConditionalDiscriminator(len(classes)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    preview_labels = torch.arange(len(classes), device=device)
    preview_noise = torch.randn(len(classes), LATENT_DIM, device=device)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        d_total = g_total = 0.0
        batches = 0
        for real_images, labels in loader:
            real_images, labels = real_images.to(device), labels.to(device)
            batch_size = labels.size(0)
            real_targets = torch.ones(batch_size, device=device)
            fake_targets = torch.zeros(batch_size, device=device)

            optimizer_d.zero_grad(set_to_none=True)
            real_logits = discriminator(real_images, labels)
            loss_real = criterion(real_logits, real_targets)
            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_images = generator(noise, labels)
            loss_fake = criterion(discriminator(fake_images.detach(), labels), fake_targets)
            d_loss = (loss_real + loss_fake) / 2
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            generated = generator(noise, labels)
            g_loss = criterion(discriminator(generated, labels), real_targets)
            g_loss.backward()
            optimizer_g.step()

            d_total += d_loss.item()
            g_total += g_loss.item()
            batches += 1

        if batches == 0:
            raise RuntimeError("No GAN batches were produced; increase dataset size or reduce batch size")
        print(f"Epoch {epoch + 1}/{args.epochs} | D loss {d_total / batches:.4f} | G loss {g_total / batches:.4f}")

        generator.eval()
        with torch.no_grad():
            preview = generator(preview_noise, preview_labels)
            save_image(preview, preview_dir / f"preview_epoch_{epoch + 1:03d}.png", normalize=True, value_range=(-1, 1), nrow=min(5, len(classes)))
        checkpoint = {
            "model_state_dict": generator.state_dict(),
            "classes": classes,
            "class_to_id": class_to_id,
            "latent_dim": LATENT_DIM,
            "image_size": IMAGE_SIZE,
            "epoch": epoch + 1,
        }
        torch.save(checkpoint, checkpoint_dir / "conditional_generator_latest.pt")
        torch.save(discriminator.state_dict(), checkpoint_dir / "discriminator_latest.pt")

    torch.save(checkpoint, checkpoint_dir / "conditional_generator_final.pt")
    print("Final generator:", checkpoint_dir / "conditional_generator_final.pt")
    print("Preview images:", preview_dir)
    print("Review generated images before adding them to classifier training.")


def generate(args) -> None:
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Generator checkpoint not found: {checkpoint_path}")
    if args.samples_per_class < 1:
        raise ValueError("--samples-per-class must be positive")
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    class_to_id = checkpoint["class_to_id"]
    latent_dim = int(checkpoint["latent_dim"])
    generator = ConditionalGenerator(len(classes), latent_dim).to(device)
    generator.load_state_dict(checkpoint["model_state_dict"])
    generator.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    with torch.no_grad():
        for class_name in classes:
            class_id = class_to_id[class_name]
            noise = torch.randn(args.samples_per_class, latent_dim, device=device)
            labels = torch.full((args.samples_per_class,), class_id, dtype=torch.long, device=device)
            generated = generator(noise, labels)
            class_dir = output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for index, image_tensor in enumerate(generated, 1):
                save_image(image_tensor, class_dir / f"sample_{index:03d}.png", normalize=True, value_range=(-1, 1))
            print(f"{class_name}: generated {args.samples_per_class} images in {class_dir}")
    print("Generated samples are experimental; visually inspect them before training with them.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="Train the conditional GAN from training crops")
    train_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    train_parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    train_parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    train_parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=0.0002)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default="cpu")
    train_parser.set_defaults(func=train)

    generate_parser = subparsers.add_parser("generate", help="Generate class-labeled sample images from a checkpoint")
    generate_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_DIR / "conditional_generator_final.pt")
    generate_parser.add_argument("--output", type=Path, default=DEFAULT_SAMPLE_DIR)
    generate_parser.add_argument("--samples-per-class", type=int, default=10)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--device", default="cpu")
    generate_parser.set_defaults(func=generate)

    args = parser.parse_args()
    if getattr(args, "epochs", 1) < 1 or getattr(args, "batch_size", 2) < 2:
        parser.error("epochs must be positive and batch-size must be at least 2")
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
