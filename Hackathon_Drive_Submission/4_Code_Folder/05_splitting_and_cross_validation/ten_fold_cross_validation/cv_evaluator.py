"""Step 5.2: 10-Fold Cross-Validation Evaluator & Metrics Aggregator."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np


class CrossValidationAggregator:
    """Aggregates training and validation results across the 10 cross-validation folds."""

    def __init__(self, num_folds: int = 10):
        self.num_folds = num_folds
        self.fold_metrics: list[dict[str, float]] = []

    def record_fold(self, fold_id: int, accuracy: float, precision: float, recall: float, f1: float):
        self.fold_metrics.append({
            "fold": fold_id,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

    def compute_summary(self) -> dict:
        if not self.fold_metrics:
            return {"status": "No folds recorded"}

        accuracies = [m["accuracy"] for m in self.fold_metrics]
        precisions = [m["precision"] for m in self.fold_metrics]
        recalls = [m["recall"] for m in self.fold_metrics]
        f1s = [m["f1"] for m in self.fold_metrics]

        return {
            "num_folds_evaluated": len(self.fold_metrics),
            "accuracy_mean": round(float(np.mean(accuracies)), 4),
            "accuracy_std": round(float(np.std(accuracies)), 4),
            "precision_mean": round(float(np.mean(precisions)), 4),
            "precision_std": round(float(np.std(precisions)), 4),
            "recall_mean": round(float(np.mean(recalls)), 4),
            "recall_std": round(float(np.std(recalls)), 4),
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_std": round(float(np.std(f1s)), 4),
            "per_fold": self.fold_metrics,
        }
