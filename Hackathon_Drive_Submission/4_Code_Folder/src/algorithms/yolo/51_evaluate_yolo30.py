
from pathlib import Path
import csv
from ultralytics import YOLO

# ---------------- PATHS ----------------
PROJECT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())

YOLO_WEIGHTS = (
    PROJECT
    / "results"
    / "training"
    / "yolo12n_30_epochs"
    / "weights"
    / "best.pt"
)

DATA_YAML = (
    PROJECT
    / "dataset"
    / "subsets"
    / "multicountry"
    / "RDD2022"
    / "yolo"
    / "data.yaml"
)

OUTPUT_DIR = PROJECT / "results" / "yolo30_full_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- CHECK FILES ----------------
if not YOLO_WEIGHTS.exists():
    raise FileNotFoundError(
        f"YOLO checkpoint not found: {YOLO_WEIGHTS}"
    )

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"Dataset YAML not found: {DATA_YAML}"
    )

# ---------------- LOAD MODEL ----------------
print("Loading YOLO12n 30-epoch model...")
model = YOLO(str(YOLO_WEIGHTS))

# ---------------- VALIDATE ----------------
print("Evaluating on the validation dataset...")
print("This may take some time on CPU.")

metrics = model.val(
    data=str(DATA_YAML),
    split="val",
    imgsz=640,

    batch=4,
    device="cpu",
    workers=0,
    plots=True,
    project=str(OUTPUT_DIR),
    name="evaluation",
    exist_ok=True
)

# ---------------- EXTRACT METRICS ----------------
precision = float(metrics.box.mp)
recall = float(metrics.box.mr)
map50 = float(metrics.box.map50)
map5095 = float(metrics.box.map)

print("\n========== YOLO12n 30-EPOCH RESULTS ==========")
print(f"Precision:  {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall:     {recall:.4f} ({recall * 100:.2f}%)")
print(f"mAP@50:     {map50:.4f} ({map50 * 100:.2f}%)")
print(f"mAP@50-95:  {map5095:.4f} ({map5095 * 100:.2f}%)")

# ---------------- SAVE CSV ----------------
csv_path = OUTPUT_DIR / "metrics.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Metric", "Value", "Percentage"])
    writer.writerow(["Precision", precision, precision * 100])
    writer.writerow(["Recall", recall, recall * 100])
    writer.writerow(["mAP50", map50, map50 * 100])
    writer.writerow(["mAP50-95", map5095, map5095 * 100])

print("\nMetrics saved to:", csv_path)
print("Evaluation plots saved under:", OUTPUT_DIR)
