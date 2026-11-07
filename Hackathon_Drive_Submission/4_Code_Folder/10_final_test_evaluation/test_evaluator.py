"""Step 10: Final Test Set Evaluation Engine.

Executes rigorous evaluation on the untouched test partition (100% real images):
- Overall Accuracy
- Macro-F1 and Weighted-F1
- Per-class Precision, Recall, F1, and Support
- Confusion Matrix computation
- Inference latency benchmark
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader


def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: list[str],
    device: str | None = None,
) -> dict:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    model.eval()

    all_preds = []
    all_targets = []
    total_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(dev)
            start_t = time.perf_counter()
            outputs = model(images)
            total_time += time.perf_counter() - start_t

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            total_samples += images.size(0)

    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)

    # Confusion Matrix
    cm = confusion_matrix(all_targets_arr, all_preds_arr, labels=range(len(class_names)))

    # Classification Report
    report_dict = classification_report(
        all_targets_arr,
        all_preds_arr,
        target_names=class_names,
        labels=range(len(class_names)),
        output_dict=True,
        zero_division=0,
    )

    avg_latency_ms = (total_time / max(1, total_samples)) * 1000.0

    return {
        "accuracy": round(float(report_dict["accuracy"]), 4),
        "macro_f1": round(float(report_dict["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report_dict["weighted avg"]["f1-score"]), 4),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "samples_evaluated": total_samples,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_raw": cm,
    }
