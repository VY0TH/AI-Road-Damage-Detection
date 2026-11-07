
from pathlib import Path
from ultralytics import YOLO

PROJECT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())

DATA_YAML = (
    PROJECT
    / "dataset/subsets/multicountry/RDD2022/yolo/data.yaml"
)

BASE_WEIGHTS = PROJECT / "yolo12n.pt"

OUTPUT_DIR = PROJECT / "results/training/yolo12n_30_epochs"

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

if not BASE_WEIGHTS.exists():
    raise FileNotFoundError(
        f"Base weights not found: {BASE_WEIGHTS}\n"
        "Locate your YOLO12n pretrained weights before running."
    )

model = YOLO(str(BASE_WEIGHTS))

model.train(
    data=str(DATA_YAML),
    epochs=30,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
    patience=10,
    project=str(OUTPUT_DIR.parent),
    name=OUTPUT_DIR.name,
    exist_ok=True,
    plots=True,
    seed=42
)

print("\nTraining completed.")
print("Best checkpoint:", OUTPUT_DIR / "weights" / "best.pt")
