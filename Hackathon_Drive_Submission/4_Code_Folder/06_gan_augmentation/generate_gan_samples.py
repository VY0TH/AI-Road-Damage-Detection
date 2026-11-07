"""Step 6: Synthetic Augmentation Sample Generator.

Uses the trained generator to create synthetic training samples for under-represented
minority classes (such as D40 Pothole, D10 Transverse Crack, D20 Alligator Crack).
"""
from __future__ import annotations

from pathlib import Path
import torch
from torchvision.utils import save_image
from PIL import Image

from gan_model import ConditionalGenerator, LATENT_DIM


def generate_samples(
    generator_checkpoint: Path,
    output_dir: Path,
    class_names: list[str],
    samples_per_class: dict[str, int],
    target_size: int = 224,
    device: str | None = None,
) -> dict[str, int]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    net_g = ConditionalGenerator(num_classes=len(class_names), latent_dim=LATENT_DIM).to(dev)
    if generator_checkpoint.is_file():
        state_dict = torch.load(generator_checkpoint, map_location=dev)
        net_g.load_state_dict(state_dict)
    net_g.eval()

    generated_counts = {}
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    with torch.no_grad():
        for cls_name, count in samples_per_class.items():
            if cls_name not in class_to_idx or count <= 0:
                continue

            cls_dir = output_dir / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            label_idx = class_to_idx[cls_name]

            labels = torch.full((count,), label_idx, dtype=torch.long, device=dev)
            noise = torch.randn(count, LATENT_DIM, device=dev)
            fakes = net_g(noise, labels)  # shape: (count, 3, 64, 64) in [-1, 1]

            # Denormalize to [0, 1]
            fakes = (fakes * 0.5) + 0.5
            fakes = torch.clamp(fakes, 0.0, 1.0)

            for i in range(count):
                sample_path = cls_dir / f"gan_syn_{cls_name}_{i:04d}.jpg"
                save_image(fakes[i], sample_path)
                # Resize to standard size (224x224)
                if target_size != 64:
                    with Image.open(sample_path) as im:
                        im.resize((target_size, target_size), Image.Resampling.LANCZOS).save(sample_path)

            generated_counts[cls_name] = count

    return generated_counts
