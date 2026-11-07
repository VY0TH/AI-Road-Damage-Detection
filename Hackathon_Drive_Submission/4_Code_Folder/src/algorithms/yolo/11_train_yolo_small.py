from ultralytics import YOLO
import os

PROJECT_ROOT = r"C:\Users\Vyom Thakur\Desktop\AI hackathon\RoadDamageProject"

DATA_YAML = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "subsets",
    "small",
    "India",
    "yolo",
    "data.yaml"
)

PROJECT_NAME = "model1_yolo12_efficientnet"

model = YOLO("yolo12n.pt")

results = model.train(
    data=DATA_YAML,

    # Small CPU test
    epochs=3,
    imgsz=320,
    batch=4,

    # Your laptop has no NVIDIA CUDA
    device="cpu",

    # Don't waste time waiting for tiny improvements
    patience=2,

    workers=0,

    project=os.path.join(
        PROJECT_ROOT,
        "results",
        "training"
    ),

    name=PROJECT_NAME,

    exist_ok=True,

    verbose=True
)

print("\n" + "=" * 60)
print("YOLO12n SMOKE TEST COMPLETE")
print("=" * 60)

print("\nResults saved in:")
print(
    os.path.join(
        PROJECT_ROOT,
        "results",
        "training",
        PROJECT_NAME
    )
)