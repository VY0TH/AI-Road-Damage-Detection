
from pathlib import Path
import csv

import matplotlib.pyplot as plt

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
OUTPUT_DIR = ROOT / "results" / "learning_curves"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    if not path.exists():
        print(f"File not found: {path}")
        return []

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def plot_resnet():
    path = ROOT / "results" / "resnet50_road_damage" / "training_history.csv"
    rows = read_csv(path)

    if not rows:
        print("No ResNet50 history available.")
        return

    epochs, train_loss, val_loss, val_acc = [], [], [], []

    for row in rows:
        try:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_accuracy"]))
        except (KeyError, ValueError):
            continue

    if not epochs:
        print("No valid ResNet50 rows found.")
        return

    plt.figure()
    plt.plot(epochs, train_loss, marker="o", label="Training loss")
    plt.plot(epochs, val_loss, marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("ResNet50: Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output = OUTPUT_DIR / "resnet50_loss.png"
    plt.savefig(output, dpi=200)
    plt.close()
    print("Saved:", output)

    plt.figure()
    plt.plot(epochs, val_acc, marker="o", label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ResNet50: Validation Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output = OUTPUT_DIR / "resnet50_validation_accuracy.png"
    plt.savefig(output, dpi=200)
    plt.close()
    print("Saved:", output)


def plot_yolo():
    path = (
        ROOT / "results" / "training" /
        "yolo12n_improved_30_epochs" / "results.csv"
    )

    rows = read_csv(path)

    if not rows:
        print("YOLO results.csv not found at expected path.")
        return

    # Ultralytics may include spaces in CSV column names.
    rows = [
        {key.strip(): value for key, value in row.items()}
        for row in rows
    ]

    def find_column(candidates):
        for key in rows[0]:
            if key in candidates:
                return key
        return None

    epoch_col = find_column(["epoch"])
    train_col = find_column(["train/box_loss"])
    val_col = find_column(["val/box_loss"])
    map_col = find_column(["metrics/mAP50(B)", "metrics/mAP50"])

    if not epoch_col:
        print("YOLO epoch column not found.")
        return

    epochs, train_loss, val_loss, map50 = [], [], [], []

    for row in rows:
        try:
            epochs.append(int(float(row[epoch_col]) + 1))
            if train_col and val_col:
                train_loss.append(float(row[train_col]))
                val_loss.append(float(row[val_col]))
            if map_col:
                map50.append(float(row[map_col]))
        except (ValueError, TypeError):
            continue

    if train_loss and val_loss:
        plt.figure()
        plt.plot(epochs[:len(train_loss)], train_loss,
                 marker="o", label="Training box loss")
        plt.plot(epochs[:len(val_loss)], val_loss,
                 marker="o", label="Validation box loss")
        plt.xlabel("Epoch")
        plt.ylabel("Box loss")
        plt.title("YOLO12n: Training vs Validation Box Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        output = OUTPUT_DIR / "yolo_training_validation_loss.png"
        plt.savefig(output, dpi=200)
        plt.close()
        print("Saved:", output)

    if map50:
        plt.figure()
        plt.plot(epochs[:len(map50)], map50,
                 marker="o", label="Validation mAP50")
        plt.xlabel("Epoch")
        plt.ylabel("mAP50")
        plt.title("YOLO12n: Validation Detection Performance")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        output = OUTPUT_DIR / "yolo_validation_map50.png"
        plt.savefig(output, dpi=200)
        plt.close()
        print("Saved:", output)


if __name__ == "__main__":
    plot_resnet()
    plot_yolo()
    print("\nFinished generating available learning curves.")
