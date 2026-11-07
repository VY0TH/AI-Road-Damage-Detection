from pathlib import Path
import torch
import ultralytics
import yaml

print("Python packages")
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Ultralytics:", ultralytics.__version__)

weights = Path("results/training/yolo12n_30_epochs/weights/best.pt")
config = Path("results/cross_validation/fold_1/data.yaml")

print("\nModel and configuration")
print("YOLO weights exist:", weights.exists(), weights)
print("Fold 1 config exists:", config.exists(), config)

if config.exists():
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    print("Classes:", data["nc"])
    print("Train list exists:", Path(data["train"]).exists())
    print("Validation list exists:", Path(data["val"]).exists())

if not weights.exists():
    print("\nSTOP: YOLO weights were not found at the expected path.")
