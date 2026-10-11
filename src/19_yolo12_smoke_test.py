from ultralytics import YOLO
import os

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\Vyom Thakur\Desktop\AI hackathon\RoadDamageProject"

DATA_YAML = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "subsets",
    "multicountry",
    "RDD2022",
    "yolo",
    "data.yaml"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "training",
    "model1_yolo12_efficientnet"
)

# ============================================================
# SETTINGS
# ============================================================

EPOCHS = 3
IMAGE_SIZE = 512
BATCH_SIZE = 4

# ============================================================
# START
# ============================================================

print("=" * 70)
print("YOLO12n MULTI-COUNTRY SMOKE TEST")
print("=" * 70)

print("\nDataset:")
print(DATA_YAML)

print("\nSettings:")
print("Epochs :", EPOCHS)
print("Image  :", IMAGE_SIZE)
print("Batch  :", BATCH_SIZE)
print("Device : CPU")

# ============================================================
# LOAD YOLO12n
# ============================================================

model = YOLO(
    "yolo12n.pt"
)

# ============================================================
# TRAIN
# ============================================================

results = model.train(
    data=DATA_YAML,

    epochs=EPOCHS,

    imgsz=IMAGE_SIZE,

    batch=BATCH_SIZE,

    device="cpu",

    workers=0,

    patience=2,

    project=RESULTS_DIR,

    name="smoke_test",

    exist_ok=True,

    pretrained=True,

    verbose=True
)

# ============================================================
# VALIDATE
# ============================================================

print("\n" + "=" * 70)
print("RUNNING VALIDATION")
print("=" * 70)

metrics = model.val(
    data=DATA_YAML,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device="cpu"
)

print("\n" + "=" * 70)
print("YOLO12n SMOKE TEST COMPLETE")
print("=" * 70)

print("\nResults saved in:")

print(
    os.path.join(
        RESULTS_DIR,
        "smoke_test"
    )
)