"""Plot real training histories and summarize overfitting evidence cautiously."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
HISTORIES = {
    "YOLO12n 30 epochs": "results/training/yolo12n_30_epochs/results.csv",
    "YOLO12n improved run": "results/training/yolo12n_improved_30_epochs/results.csv",
    "YOLO12n road damage run": "results/yolo12n_road_damage/results.csv",
    "EfficientNetB0": "results/efficientnet_b0/training_history.json",
    "ResNet50 road damage": "results/resnet50_road_damage/training_history.csv",
    "ResNet50 earlier run": "results/resnet50/training_history.json",
    "ViT-B/16": "results/vit/training_history.json",
}


def _read_history(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("history", value.get("epochs", []))
        return pd.DataFrame(value)
    return pd.read_csv(path)


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _plot_history(name: str, frame: pd.DataFrame, output_dir: Path) -> list[str]:
    outputs = []
    x = "epoch" if "epoch" in frame.columns else None
    if x is None:
        frame = frame.copy()
        frame["epoch"] = range(1, len(frame) + 1)
        x = "epoch"

    train_losses = [column for column in frame.columns if column.startswith("train/") and "loss" in column]
    val_losses = [column for column in frame.columns if column.startswith("val/") and "loss" in column]
    if "train_loss" in frame:
        train_losses = ["train_loss"]
    if "val_loss" in frame:
        val_losses = ["val_loss"]
    loss_columns = train_losses + val_losses
    if loss_columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        for column in loss_columns:
            ax.plot(frame[x], frame[column], marker="o", label=column)
        ax.set(title=f"{name}: training and validation loss", xlabel="Epoch", ylabel="Loss")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        filename = f"{_safe_name(name)}_loss.png"
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)
        outputs.append(filename)

    metric_columns = [
        column for column in frame.columns
        if column in {"train_accuracy", "val_accuracy", "val_dice", "val_iou"}
        or column.startswith("metrics/") and ("mAP" in column or "precision" in column or "recall" in column)
    ]
    if metric_columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        for column in metric_columns:
            ax.plot(frame[x], frame[column], marker="o", label=column)
        ax.set(title=f"{name}: validation metrics", xlabel="Epoch", ylabel="Metric")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        filename = f"{_safe_name(name)}_metrics.png"
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)
        outputs.append(filename)
    return outputs


def _evidence(name: str, frame: pd.DataFrame) -> str:
    if len(frame) < 5:
        return f"Insufficient evidence: only {len(frame)} recorded epoch(s)."
    if {"train_loss", "val_loss", "train_accuracy", "val_accuracy"}.issubset(frame.columns):
        tail = frame.tail(min(3, len(frame)))
        accuracy_gap = float(tail["train_accuracy"].iloc[-1] - tail["val_accuracy"].iloc[-1])
        best_val_loss = float(frame["val_loss"].min())
        final_val_loss = float(frame["val_loss"].iloc[-1])
        val_accuracy_peak = float(frame["val_accuracy"].max())
        final_val_accuracy = float(frame["val_accuracy"].iloc[-1])
        signs = accuracy_gap >= 0.10 or (
            final_val_loss > best_val_loss * 1.10
            and val_accuracy_peak - final_val_accuracy >= 0.03
        )
        if signs:
            return (
                f"Overfitting signs near the end: final train/validation accuracy gap "
                f"{accuracy_gap:.3f}; validation loss best/final "
                f"{best_val_loss:.3f}/{final_val_loss:.3f}; validation accuracy peak/final "
                f"{val_accuracy_peak:.3f}/{final_val_accuracy:.3f}. This is a small logged run, "
                "not a controlled generalization study."
            )
        return (
            f"No strong overfitting signal in the recorded curve: final train/validation accuracy "
            f"gap {accuracy_gap:.3f}; validation loss best/final "
            f"{best_val_loss:.3f}/{final_val_loss:.3f}."
        )
    map50 = "metrics/mAP50(B)"
    if map50 in frame and len(frame) >= 5:
        values = pd.to_numeric(frame[map50], errors="coerce").dropna()
        if values.empty:
            return "Insufficient evidence: validation mAP values are absent."
        peak_epoch = int(frame.loc[values.idxmax(), "epoch"])
        best = float(values.max())
        final = float(values.iloc[-1])
        if peak_epoch < int(frame["epoch"].iloc[-1]) and best > final * 1.05:
            return (
                f"Validation mAP50 peaked at epoch {peak_epoch} ({best:.4f}) and ended at "
                f"{final:.4f}; this is a late validation decline, but the available curve does "
                "not by itself establish overfitting."
            )
        return (
            f"Poor validation mAP50 (best {best:.4f}); low mAP alone does not establish "
            "underfitting or overfitting."
        )
    return "Insufficient comparable training and validation metrics for a fit diagnosis."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/overfitting_analysis")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    findings = []
    for name, relative in HISTORIES.items():
        path = ROOT / relative
        if not path.is_file():
            findings.append({"name": name, "source": relative, "status": "history missing"})
            continue
        try:
            frame = _read_history(path)
        except Exception as exc:
            findings.append({
                "name": name, "source": relative, "status": "history unreadable",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            continue
        if frame.empty:
            findings.append({"name": name, "source": relative, "status": "no epoch rows"})
            continue
        plots = _plot_history(name, frame, output_dir)
        findings.append({
            "name": name,
            "source": relative,
            "status": "analyzed",
            "epochs": len(frame),
            "plots": plots,
            "finding": _evidence(name, frame),
        })

    segmentation_path = ROOT / "results/cnn_lstm/segmentation/final_test_report.txt"
    segmentation_text = segmentation_path.read_text(encoding="utf-8") if segmentation_path.is_file() else ""
    report_lines = [
        "# Overfitting and underfitting analysis",
        "",
        "This report describes recorded training histories only. A low score by itself is not proof of underfitting; crop-classification curves do not measure end-to-end detection.",
        "",
        "## Model findings",
        "",
    ]
    for finding in findings:
        report_lines += [f"### {finding['name']}", "", f"- History: `{finding['source']}`"]
        if finding["status"] != "analyzed":
            report_lines += [f"- Status: {finding['status']}"]
        else:
            report_lines += [f"- Epochs recorded: {finding['epochs']}", f"- Finding: {finding['finding']}"]
            report_lines += [f"- Plot: `{plot}`" for plot in finding["plots"]]
        report_lines.append("")

    report_lines += [
        "### CNN + Spatial LSTM",
        "",
        "- No per-epoch segmentation training history was found, so a learning curve or V3 overfit diagnosis cannot be produced.",
    ]
    if segmentation_text:
        report_lines.append("- Existing test report is for the V2 checkpoint on a selected 10-video subset, not the dashboard's V3 checkpoint.")
        for line in segmentation_text.splitlines():
            if "Mean per-video" in line or "Global pixel" in line or "Predicted foreground" in line:
                report_lines.append(f"- {line.strip()}")
    report_lines += [
        "",
        "## Interpretation",
        "",
        "- ViT-B/16 shows end-of-run overfitting signs in its five-epoch crop-classification history: training accuracy continues upward while validation accuracy declines from its peak and validation loss rises in the final epoch.",
        "- EfficientNetB0's validation loss oscillates late, but its final validation accuracy is close to training accuracy; the logged curve is not strong evidence of severe overfitting.",
        "- ResNet histories are short (three and ten epochs); the three-epoch road-damage run is insufficient to diagnose fit. The earlier ten-epoch history shows no large final accuracy gap.",
        "- The selected YOLO run reaches its best recorded mAP50 at epoch 29 and declines at epoch 30. This one-epoch decline is a warning to retain the best checkpoint, not proof of overfitting. Its full-validation mAP remains poor.",
        "- The improved YOLO run has only eight of thirty planned epochs logged. Its weak metrics do not by themselves distinguish underfitting from label or training problems.",
        "- GAN classifier and dashboard V3 segmentation epoch histories are absent; no fit diagnosis is possible for those models.",
        "",
        "No training was started and no checkpoints were changed by this analysis.",
    ]
    report_path = output_dir / "overfitting_underfitting_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    machine_report = output_dir / "learning_curve_inventory.json"
    machine_report.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")
    print(f"History inventory: {machine_report}")
    for finding in findings:
        print(f"{finding['name']}: {finding['status']}")


if __name__ == "__main__":
    main()
