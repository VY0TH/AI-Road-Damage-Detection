
from pathlib import Path
from ultralytics import YOLO
import csv

PROJECT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())

WEIGHTS = (
    PROJECT
    / "results/training/model1_yolo12_efficientnet/"
    / "baseline_15_epochs/weights/best.pt"
)

DATA_YAML = (
    PROJECT
    / "dataset/subsets/multicountry/RDD2022/yolo/data.yaml"
)

OUTPUT_DIR = PROJECT / "results/yolo_full_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not WEIGHTS.exists():
    raise FileNotFoundError(f"YOLO weights not found: {WEIGHTS}")

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"Dataset YAML not found: {DATA_YAML}"
    )

model = YOLO(str(WEIGHTS))

print("Evaluating YOLO on the validation dataset...")
print("This may take some time on CPU.")

metrics = model.val(
    data=str(DATA_YAML),
    split="val",
    device="cpu",
    batch=4,
    workers=0,
    plots=True,
    project=str(OUTPUT_DIR),
    name="evaluation",
    exist_ok=True
)

box = metrics.box

results = {
    "precision": float(box.mp),
    "recall": float(box.mr),
    "mAP50": float(box.map50),
    "mAP50_95": float(box.map),
}

report_path = OUTPUT_DIR / "metrics.csv"

with report_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Metric", "Value"])

    for name, value in results.items():
        writer.writerow([name, f"{value:.6f}"])

print("\nFULL VALIDATION RESULTS")
for name, value in results.items():
    print(f"{name}: {value:.4f}")

print("\nMetrics saved to:", report_path)
print("Plots saved under:", OUTPUT_DIR)
