
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
MANIFEST = ROOT / "dataset" / "processed" / "efficientnet_crops" / "manifest.csv"
OUTPUT = ROOT / "results" / "conditional_gan_selected"
CHECKPOINTS = ROOT / "results" / "conditional_gan_selected_checkpoints"

OUTPUT.mkdir(parents=True, exist_ok=True)
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 64
LATENT_DIM = 100
EMBED_DIM = 32
BATCH_SIZE = 8
EPOCHS = 20
LR = 0.0002
SEED = 42
DEVICE = torch.device("cpu")

torch.manual_seed(SEED)
random.seed(SEED)

if not MANIFEST.exists():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

df = pd.read_csv(MANIFEST)
required = {"split", "crop_path", "class_name"}

if not required.issubset(df.columns):
    raise ValueError(f"Manifest must contain columns: {required}")

# Use training rows only; never train on validation crops.
df = df[df["split"].astype(str).str.lower() == "train"].copy()
df = df.dropna(subset=["crop_path", "class_name"])

# Train the GAN only on classes with enough training examples.
SELECTED_CLASSES = ["D00", "D10", "D20", "D40", "D44"]

df["class_name"] = df["class_name"].astype(str)
df = df[df["class_name"].isin(SELECTED_CLASSES)].copy()

classes = SELECTED_CLASSES
class_to_id = {name: idx for idx, name in enumerate(classes)}
df["label"] = df["class_name"].map(class_to_id)

print("Selected GAN classes:", class_to_id)
print("Training crops used:", len(df))
print(df["class_name"].value_counts().to_string())
class_to_id = {name: idx for idx, name in enumerate(classes)}
df["label"] = df["class_name"].astype(str).map(class_to_id)

print("Classes:", class_to_id)
print("Training crops in manifest:", len(df))

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class CropDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = ROOT / str(row["crop_path"])

        if not path.exists():
            raise FileNotFoundError(f"Crop missing: {path}")

        with Image.open(path) as image:
            image = transform(image.convert("RGB"))

        return image, int(row["label"])


dataset = CropDataset(df)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=False
)


# Conditional generator: noise + class embedding.
class Generator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, EMBED_DIM)

        self.fc = nn.Sequential(
            nn.Linear(LATENT_DIM + EMBED_DIM, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.ReLU(True)
        )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        embedded = self.embedding(labels)
        x = torch.cat([noise, embedded], dim=1)
        x = self.fc(x).view(-1, 256, 4, 4)
        return self.net(x)


# Conditional discriminator receives image + class label.
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, IMAGE_SIZE * IMAGE_SIZE)

        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 4, 1, 0)
        )

    def forward(self, images, labels):
        label_map = self.embedding(labels).view(
            -1, 1, IMAGE_SIZE, IMAGE_SIZE
        )
        x = torch.cat([images, label_map], dim=1)
        return self.net(x).view(-1)


num_classes = len(classes)
generator = Generator(num_classes).to(DEVICE)
discriminator = Discriminator(num_classes).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
opt_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

# Fixed class previews, one sample per class.
preview_labels = torch.arange(num_classes, device=DEVICE)
preview_noise = torch.randn(num_classes, LATENT_DIM, device=DEVICE)

for epoch in range(EPOCHS):
    generator.train()
    discriminator.train()
    d_total = 0.0
    g_total = 0.0
    batches = 0

    for real_images, labels in loader:
        real_images = real_images.to(DEVICE)
        labels = labels.to(DEVICE)
        batch = len(labels)

        real_targets = torch.ones(batch, device=DEVICE)
        fake_targets = torch.zeros(batch, device=DEVICE)

        # Train discriminator.
        opt_d.zero_grad()
        real_logits = discriminator(real_images, labels)
        loss_real = criterion(real_logits, real_targets)

        noise = torch.randn(batch, LATENT_DIM, device=DEVICE)
        fake_images = generator(noise, labels)
        fake_logits = discriminator(fake_images.detach(), labels)
        loss_fake = criterion(fake_logits, fake_targets)

        d_loss = (loss_real + loss_fake) / 2
        d_loss.backward()
        opt_d.step()

        # Train generator.
        opt_g.zero_grad()
        noise = torch.randn(batch, LATENT_DIM, device=DEVICE)
        fake_images = generator(noise, labels)
        fake_logits = discriminator(fake_images, labels)

        g_loss = criterion(fake_logits, real_targets)
        g_loss.backward()
        opt_g.step()

        d_total += d_loss.item()
        g_total += g_loss.item()
        batches += 1

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"D Loss: {d_total / batches:.4f} | "
        f"G Loss: {g_total / batches:.4f}"
    )

    generator.eval()
    with torch.no_grad():
        previews = generator(preview_noise, preview_labels)
        save_image(
            previews,
            OUTPUT / f"preview_epoch_{epoch + 1:02d}.png",
            normalize=True,
            nrow=5
        )

    torch.save(
        generator.state_dict(),
        CHECKPOINTS / "generator_latest.pt"
    )
    torch.save(
        discriminator.state_dict(),
        CHECKPOINTS / "discriminator_latest.pt"
    )

# Save final generator and class mapping.
torch.save(
    {
        "model_state_dict": generator.state_dict(),
        "classes": classes,
        "class_to_id": class_to_id,
        "latent_dim": LATENT_DIM
    },
    CHECKPOINTS / "conditional_generator_final.pt"
)

print("\nConditional GAN training complete.")
print("Preview folder:", OUTPUT)
print("Checkpoint:", CHECKPOINTS / "conditional_generator_final.pt")
print("WARNING: Inspect previews; class conditioning does not guarantee correct images.")
