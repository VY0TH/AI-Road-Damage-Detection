
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# =========================================================
# CONFIGURATION
# =========================================================

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())

TRAIN_DIR = ROOT / "dataset" / "processed" / "efficientnet_crops" / "train"
OUTPUT_DIR = ROOT / "results" / "gan_generated_v2"
CHECKPOINT_DIR = ROOT / "results" / "gan_checkpoints_v2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 64
LATENT_DIM = 100
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.0002
BETA1 = 0.5
NUM_SYNTHETIC_IMAGES = 20
SEED = 42

DEVICE = torch.device("cpu")

torch.manual_seed(SEED)
random.seed(SEED)

print("Device:", DEVICE)
print("Training folder:", TRAIN_DIR)
print("Output folder:", OUTPUT_DIR)


# =========================================================
# DATASET
# =========================================================

class RoadCropDataset(Dataset):
    def __init__(self, directory):
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        self.files = sorted(
            p for p in Path(directory).rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        )

        if not self.files:
            raise FileNotFoundError(
                f"No training images found in {directory}"
            )

        print("Real training images:", len(self.files))

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            return self.transform(image.convert("RGB"))


dataset = RoadCropDataset(TRAIN_DIR)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True
)


# =========================================================
# GENERATOR
# =========================================================

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

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

    def forward(self, noise):
        return self.net(noise)


# =========================================================
# DISCRIMINATOR
# =========================================================

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 4, 1, 0, bias=False)
        )

    def forward(self, images):
        return self.net(images).reshape(-1)


# =========================================================
# INITIALIZATION
# =========================================================

def initialize_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)

    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

generator.apply(initialize_weights)
discriminator.apply(initialize_weights)

criterion = nn.BCEWithLogitsLoss()

optimizer_g = torch.optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE,
    betas=(BETA1, 0.999)
)

optimizer_d = torch.optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE,
    betas=(BETA1, 0.999)
)

fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, device=DEVICE)


# =========================================================
# TRAINING
# =========================================================

for epoch in range(EPOCHS):
    generator.train()
    discriminator.train()

    g_total = 0.0
    d_total = 0.0
    batches = 0

    for real_images in loader:
        real_images = real_images.to(DEVICE)
        batch_size = real_images.size(0)

        real_targets = torch.ones(batch_size, device=DEVICE)
        fake_targets = torch.zeros(batch_size, device=DEVICE)

        # Train discriminator
        optimizer_d.zero_grad()

        real_logits = discriminator(real_images)
        loss_real = criterion(real_logits, real_targets)

        noise = torch.randn(
            batch_size, LATENT_DIM, 1, 1, device=DEVICE
        )
        fake_images = generator(noise)

        fake_logits = discriminator(fake_images.detach())
        loss_fake = criterion(fake_logits, fake_targets)

        d_loss = (loss_real + loss_fake) / 2
        d_loss.backward()
        optimizer_d.step()

        # Train generator
        optimizer_g.zero_grad()

        noise = torch.randn(
            batch_size, LATENT_DIM, 1, 1, device=DEVICE
        )
        generated = generator(noise)
        generated_logits = discriminator(generated)

        g_loss = criterion(generated_logits, real_targets)
        g_loss.backward()
        optimizer_g.step()

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
        preview = generator(fixed_noise)

        save_image(
            preview,
            OUTPUT_DIR / f"preview_epoch_{epoch + 1:02d}.png",
            normalize=True,
            nrow=4
        )

    torch.save(
        generator.state_dict(),
        CHECKPOINT_DIR / "generator_latest.pt"
    )

    torch.save(
        discriminator.state_dict(),
        CHECKPOINT_DIR / "discriminator_latest.pt"
    )


# =========================================================
# GENERATE SYNTHETIC SAMPLES
# =========================================================

generator.eval()

with torch.no_grad():
    for start in range(0, NUM_SYNTHETIC_IMAGES, BATCH_SIZE):
        count = min(BATCH_SIZE, NUM_SYNTHETIC_IMAGES - start)

        noise = torch.randn(
            count, LATENT_DIM, 1, 1, device=DEVICE
        )
        generated = generator(noise)

        for offset, image_tensor in enumerate(generated):
            number = start + offset

            save_image(
                image_tensor,
                OUTPUT_DIR / f"synthetic_{number:04d}.png",
                normalize=True
            )

torch.save(
    generator.state_dict(),
    CHECKPOINT_DIR / "generator_final.pt"
)

print("\nGAN training completed.")
print("Synthetic images:", NUM_SYNTHETIC_IMAGES)
print("Preview:", OUTPUT_DIR / f"preview_epoch_{EPOCHS:02d}.png")
print("Checkpoint:", CHECKPOINT_DIR / "generator_final.pt")
print(
    "WARNING: Generated samples have no verified class labels. "
    "Do not add them to labelled classifier folders automatically."
)
