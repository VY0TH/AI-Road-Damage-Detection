"""Step 10 CLI Runner: Final Test Evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "01_load_image_dataset"))
sys.path.insert(0, str(PROJECT_ROOT / "07_initial_model_training"))

from dataset_loader import ImageFolderDataset, create_dataloader
from models.baseline_cnn import BaselineCNN
from models.efficientnet_classifier import create_efficientnet
from models.resnet50_classifier import create_resnet50
from test_evaluator import evaluate_test_set
from confusion_matrix_plotter import plot_confusion_matrix
from generate_final_report import generate_markdown_report
import pipeline_config as cfg


def run_step_10(
    checkpoint_path: Path | None = None,
    test_dir: Path | None = None,
    model_name: str = "baseline_cnn",
) -> dict:
    # 1. Resolve test dataset (held-out test set from Step 5, or val set if test not yet created)
    if test_dir is None:
        cand_test = cfg.DATASET_ROOT / "processed" / "train_val_test_70_15_15" / "test"
        if cand_test.is_dir():
            test_dir = cand_test
        else:
            test_dir = cfg.DATASET_ROOT / "processed" / "efficientnet_crops" / "val"

    # 2. Resolve model checkpoint
    if checkpoint_path is None:
        cand_ckpt = cfg.RESULTS_DIR / "initial_training" / model_name / "best_model.pt"
        if cand_ckpt.is_file():
            checkpoint_path = cand_ckpt
        else:
            # Check corrected model checkpoint
            cand_corr = cfg.RESULTS_DIR / "corrected_retraining" / "efficientnet" / "corrected_best_model.pt"
            if cand_corr.is_file():
                checkpoint_path = cand_corr
                model_name = "efficientnet"
            else:
                checkpoint_path = None

    print(f"\n{'=' * 70}")
    print(f"[STEP 10] FINAL TEST EVALUATION")
    print(f"  Test Data  : {test_dir} (100% Real Images Only)")
    print(f"  Model      : {model_name}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"{'=' * 70}")

    val_transform = transforms.Compose([
        transforms.Resize((cfg.DEFAULT_IMAGE_SIZE, cfg.DEFAULT_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_ds = ImageFolderDataset(test_dir, cfg.CLASS_NAMES, transform=val_transform)
    test_loader = create_dataloader(test_ds, batch_size=16, shuffle=False)

    num_classes = len(cfg.CLASS_NAMES)
    if model_name == "baseline_cnn":
        model = BaselineCNN(num_classes=num_classes)
    elif model_name == "efficientnet":
        model = create_efficientnet(num_classes=num_classes)
    elif model_name == "resnet50":
        model = create_resnet50(num_classes=num_classes)
    else:
        model = BaselineCNN(num_classes=num_classes)

    if checkpoint_path and Path(checkpoint_path).is_file():
        import torch
        model.load_state_dict(torch.load(checkpoint_path, map_location=cfg.DEVICE))
        print(f"Loaded trained checkpoint: {checkpoint_path}")

    # Run Evaluation
    eval_results = evaluate_test_set(
        model=model,
        test_loader=test_loader,
        class_names=cfg.CLASS_NAMES,
    )

    # Plot Confusion Matrix
    out_dir = cfg.RESULTS_DIR / "final_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    cm_plot = out_dir / "test_confusion_matrix.png"
    plot_confusion_matrix(
        cm=eval_results["confusion_matrix_raw"],
        class_names=cfg.CLASS_NAMES,
        output_path=cm_plot,
        title=f"Final Test Confusion Matrix ({model_name.upper()})",
    )

    # Generate Reports
    md_report = generate_markdown_report(
        eval_results=eval_results,
        cm_plot_path=str(cm_plot),
        output_path=out_dir / "final_evaluation_report.md",
        model_name=model_name,
    )

    report_json = out_dir / "final_evaluation_report.json"
    with report_json.open("w", encoding="utf-8") as f:
        # Exclude raw ndarray before json dump
        save_dict = {k: v for k, v in eval_results.items() if k != "confusion_matrix_raw"}
        save_dict["confusion_matrix_plot"] = str(cm_plot)
        json.dump(save_dict, f, indent=2)

    print(f"\nFinal Test Results:")
    print(f"  Accuracy       : {eval_results['accuracy'] * 100:.2f}%")
    print(f"  Macro-F1       : {eval_results['macro_f1']:.4f}")
    print(f"  Weighted-F1    : {eval_results['weighted_f1']:.4f}")
    print(f"  Avg Latency    : {eval_results['avg_latency_ms']:.2f} ms")
    print(f"  Confusion Plot : {cm_plot}")
    print(f"  Report (MD)    : {md_report}")
    print(f"  Report (JSON)  : {report_json}")
    print(f"{'=' * 70}\n")

    return eval_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default="baseline_cnn")
    args = parser.parse_args()

    run_step_10(args.checkpoint, args.test_dir, args.model_name)


if __name__ == "__main__":
    main()
