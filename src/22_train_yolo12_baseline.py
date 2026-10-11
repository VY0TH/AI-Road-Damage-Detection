import os
from ultralytics import YOLO

ROOT = r"C:\Users\Vyom Thakur\Desktop\AI hackathon\RoadDamageProject"

DATA = os.path.join(
    ROOT,
    "dataset", "subsets", "multicountry",
    "RDD2022", "yolo", "data.yaml"
)

OUTPUT = os.path.join(
    ROOT, "results", "training", "model1_yolo12_efficientnet"
)

model = YOLO("yolo12n.pt")

model.train(
    data=DATA,
    epochs=15,
    imgsz=512,
    batch=4,
    device="cpu",
    workers=0,
    patience=15,
    seed=42,
    project=OUTPUT,
    name="baseline_15_epochs",
    exist_ok=True,
    pretrained=True
)

print("\nTraining complete.")
print("Results:", os.path.join(OUTPUT, "baseline_15_epochs"))