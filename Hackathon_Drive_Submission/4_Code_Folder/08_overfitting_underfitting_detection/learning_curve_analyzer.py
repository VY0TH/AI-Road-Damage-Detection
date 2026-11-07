"""Step 8: Learning Curve Analyzer & Visualizer.

Plots training and validation loss and accuracy curves from epoch history logs.
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_history(history_path: Path) -> pd.DataFrame:
    history_path = Path(history_path)
    if not history_path.is_file():
        raise FileNotFoundError(f"History file not found: {history_path}")

    if history_path.suffix.lower() == ".json":
        data = json.loads(history_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("history") or data.get("epochs") or []
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(history_path)

    if "epoch" not in df.columns:
        df["epoch"] = range(1, len(df) + 1)
    return df


def plot_learning_curves(df: pd.DataFrame, output_dir: Path, model_name: str = "Model") -> dict[str, str]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = {}

    epochs = df["epoch"]

    # 1. Loss Plot
    if "train_loss" in df.columns and "val_loss" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, df["train_loss"], marker="o", label="Training Loss", color="#1f77b4", linewidth=2)
        plt.plot(epochs, df["val_loss"], marker="s", label="Validation Loss", color="#d62728", linewidth=2)
        plt.title(f"{model_name} - Training vs. Validation Loss", fontsize=13, fontweight="bold")
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Loss", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=10)
        plt.tight_layout()
        loss_plot_path = output_dir / f"{model_name.lower()}_loss_curve.png"
        plt.savefig(loss_plot_path, dpi=150)
        plt.close()
        generated_plots["loss_plot"] = str(loss_plot_path)

    # 2. Accuracy Plot
    if "train_accuracy" in df.columns and "val_accuracy" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, df["train_accuracy"] * 100, marker="o", label="Training Accuracy", color="#2ca02c", linewidth=2)
        plt.plot(epochs, df["val_accuracy"] * 100, marker="s", label="Validation Accuracy", color="#ff7f0e", linewidth=2)
        plt.title(f"{model_name} - Training vs. Validation Accuracy", fontsize=13, fontweight="bold")
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Accuracy (%)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=10)
        plt.tight_layout()
        acc_plot_path = output_dir / f"{model_name.lower()}_accuracy_curve.png"
        plt.savefig(acc_plot_path, dpi=150)
        plt.close()
        generated_plots["accuracy_plot"] = str(acc_plot_path)

    return generated_plots
