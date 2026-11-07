"""Step 10: Confusion Matrix Visualizer."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
    normalize: bool = True,
    title: str = "Final Test Evaluation - Confusion Matrix",
) -> str:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1e-6)
        display_data = cm_norm
        fmt = ".2f"
    else:
        display_data = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(display_data, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label (Ground Truth)",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = display_data.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = f"{display_data[i, j]:{fmt}}" if normalize else str(display_data[i, j])
            ax.text(
                j, i, val,
                ha="center", va="center",
                color="white" if display_data[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)
