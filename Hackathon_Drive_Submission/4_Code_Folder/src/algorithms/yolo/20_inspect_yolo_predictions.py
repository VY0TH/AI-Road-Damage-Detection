import os
from ultralytics import YOLO

PROJECT_ROOT = (
    r"C:\Users\Vyom Thakur\Desktop\AI hackathon\RoadDamageProject"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "training",
    "model1_yolo12_efficientnet",
    "smoke_test",
    "weights",
    "best.pt"
)

SOURCE_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "subsets",
    "multicountry",
    "RDD2022",
    "yolo",
    "images",
    "val"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "prediction_inspection"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

results = model.predict(
    source=SOURCE_DIR,
    imgsz=512,
    device="cpu",
    conf=0.10,
    save=True,
    project=OUTPUT_DIR,
    name="yolo12_smoke_test",
    exist_ok=True
)

print("Prediction inspection complete.")
print("Open:", os.path.join(OUTPUT_DIR, "yolo12_smoke_test"))