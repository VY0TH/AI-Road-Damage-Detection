
from collections import Counter

import torch.nn as nn
from torchvision import models
from ultralytics import YOLO

from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())

def report(name, model):
    counts = Counter(
        type(module).__name__
        for module in model.modules()
        if isinstance(module, (
            nn.ReLU, nn.SiLU, nn.GELU,
            nn.Sigmoid, nn.Tanh, nn.LeakyReLU
        ))
    )

    print(f"\n{'=' * 48}")
    print(name)
    print("=" * 48)

    if not counts:
        print("No matching activation modules found.")
    else:
        for activation, count in sorted(counts.items()):
            print(f"{activation}: {count}")

    print("\nExample module locations:")
    shown = 0
    for module_name, module in model.named_modules():
        if isinstance(module, (
            nn.ReLU, nn.SiLU, nn.GELU,
            nn.Sigmoid, nn.Tanh, nn.LeakyReLU
        )):
            print(f"{module_name}: {type(module).__name__}")
            shown += 1
            if shown == 8:
                break


report("ResNet50", models.resnet50(weights=None))
report("EfficientNetB0", models.efficientnet_b0(weights=None))
report("Vision Transformer B/16", models.vit_b_16(weights=None))

yolo_path = (
    ROOT / "results" / "training" /
    "yolo12n_improved_30_epochs" / "weights" / "best.pt"
)

if yolo_path.exists():
    yolo = YOLO(str(yolo_path))
    report("YOLO12n checkpoint", yolo.model)
else:
    print("\nYOLO checkpoint not found:", yolo_path)
