"""ResNet-50 Inference Engine."""
from __future__ import annotations

import sys
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "04_clahe_enhancement"))

from clahe_processor import apply_clahe_to_image
from model import get_resnet50
import pipeline_config as cfg


class ResNet50Classifier:
    def __init__(self, checkpoint_path: Path | None = None, apply_clahe: bool = True):
        self.dev = cfg.DEVICE
        self.apply_clahe = apply_clahe
        self.class_names = cfg.CLASS_NAMES
        self.model = get_resnet50(num_classes=len(self.class_names)).to(self.dev)

        if checkpoint_path and Path(checkpoint_path).is_file():
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.dev))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((cfg.DEFAULT_IMAGE_SIZE, cfg.DEFAULT_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, image: Image.Image) -> tuple[str, float, dict[str, float]]:
        if self.apply_clahe:
            image = apply_clahe_to_image(image)

        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.dev)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(probs.argmax())
        pred_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])
        all_probs = {name: float(probs[i]) for i, name in enumerate(self.class_names)}
        return pred_class, confidence, all_probs
