
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
DATA_YAML = ROOT / "dataset" / "yolo" / "data.yaml"

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset configuration missing: {DATA_YAML}")

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device: CPU")

# YOLO12 nano baseline
model = YOLO("yolo12n.pt")

results = model.train(
    data=str(DATA_YAML),
    epochs=3,             # Smoke test only
    imgsz=416,
    batch=2,
    device="cpu",
    workers=0,
    patience=3,
    project=str(ROOT / "results" / "training"),
    name="yolo12n_smoke_test",
    exist_ok=True,
    plots=True,
    save=True,
    verbose=True
)

print("\nSmoke test finished.")
print("Review the training results under results/training/yolo12n_smoke_test")
