
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import save_image

# ---------------- PATHS ----------------
PROJECT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())

CHECKPOINT = (
    PROJECT
    / "results"
    / "conditional_gan_selected_checkpoints"
    / "conditional_generator_final.pt"
)

OUTPUT_DIR = PROJECT / "results" / "gan_individual_samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
IMAGE_SIZE = 64
EMBED_DIM = 32
SAMPLES_PER_CLASS = 10

# ---------------- GENERATOR ----------------
class Generator(nn.Module):
    def __init__(self, num_classes, latent_dim):
        super().__init__()

        self.embedding = nn.Embedding(num_classes, EMBED_DIM)

        self.fc = nn.Sequential(
            nn.Linear(latent_dim + EMBED_DIM, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.ReLU(True),
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
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        embedded = self.embedding(labels)
        x = torch.cat([noise, embedded], dim=1)
        x = self.fc(x).view(-1, 256, 4, 4)
        return self.net(x)


# ---------------- LOAD CHECKPOINT ----------------
if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

classes = checkpoint["classes"]
class_to_id = checkpoint["class_to_id"]
latent_dim = checkpoint["latent_dim"]

generator = Generator(len(classes), latent_dim).to(DEVICE)
generator.load_state_dict(checkpoint["model_state_dict"])
generator.eval()

print("Loaded classes:", classes)

# ---------------- GENERATE SAMPLES ----------------
with torch.no_grad():
    for class_name in classes:
        class_id = class_to_id[class_name]

        noise = torch.randn(
            SAMPLES_PER_CLASS, latent_dim, device=DEVICE
        )
        labels = torch.full(
            (SAMPLES_PER_CLASS,),
            class_id,
            dtype=torch.long,
            device=DEVICE,
        )

        generated = generator(noise, labels)

        class_dir = OUTPUT_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i, image_tensor in enumerate(generated):
            output_path = class_dir / f"sample_{i + 1:02d}.png"

            # Convert GAN output from [-1, 1] to [0, 1].
            save_image(
                image_tensor,
                output_path,
                normalize=True,
                value_range=(-1, 1),
            )

        print(
            f"{class_name}: saved {SAMPLES_PER_CLASS} samples "
            f"to {class_dir}"
        )

print("\nGeneration complete.")
print("Output folder:", OUTPUT_DIR)
print("WARNING: Generated images are experimental, not verified labels.")

