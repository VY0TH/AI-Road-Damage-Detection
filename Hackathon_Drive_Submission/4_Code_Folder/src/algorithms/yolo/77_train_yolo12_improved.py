
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())

DATA_YAML = (
    ROOT / "dataset" / "subsets" / "multicountry"
    / "RDD2022" / "yolo" / "data.yaml"
)
BASE_WEIGHTS = ROOT / "yolo12n.pt"

OUTPUT_DIR = ROOT / "results" / "training"
RUN_NAME = "yolo12n_improved_30_epochs"

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

if not BASE_WEIGHTS.exists():
    raise FileNotFoundError(f"Base weights not found: {BASE_WEIGHTS}")

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Training device: CPU")
print("Dataset:", DATA_YAML)
print("Output:", OUTPUT_DIR / RUN_NAME)

model = YOLO(str(BASE_WEIGHTS))

model.train(
    data=str(DATA_YAML),
    epochs=30,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
    patience=10,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    cos_lr=True,
    weight_decay=0.0005,
    seed=42,
    deterministic=True,
    project=str(OUTPUT_DIR),
    name=RUN_NAME,
    exist_ok=False,
    plots=True,
    save=True,
    verbose=True,
)

print("\nTraining finished.")
print(
    "Best checkpoint:",
    OUTPUT_DIR / RUN_NAME / "weights" / "best.pt",
)
print(
    "Review results:",
    OUTPUT_DIR / RUN_NAME,
)
