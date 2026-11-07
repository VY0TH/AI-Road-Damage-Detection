"""CPU smoke tests for the unified dashboard's image pipelines.

Run from the project root:
    python src/91_smoke_dashboard.py
    python src/91_smoke_dashboard.py --video dataset/videos/.../0081.mp4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = Path(__file__).with_name("89_road_damage_app.py")


def load_app_module():
    spec = importlib.util.spec_from_file_location("road_damage_dashboard", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import dashboard module: {APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_ground_truth(label_path: Path, width: int, height: int, class_id: int):
    boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Malformed label at {label_path}:{line_number}")
        found_id = int(fields[0])
        if found_id != class_id:
            continue
        cx, cy, box_w, box_h = map(float, fields[1:])
        boxes.append((
            round((cx - box_w / 2) * width), round((cy - box_h / 2) * height),
            round((cx + box_w / 2) * width), round((cy + box_h / 2) * height),
        ))
    return boxes


def box_iou(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_b = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return intersection / max(1, area_a + area_b - intersection)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default="dataset/subsets/multicountry/RDD2022/yolo/images/val/China_0000.jpg",
        help="Validation image with a paired YOLO label containing D40.",
    )
    parser.add_argument(
        "--data-yaml",
        default="dataset/subsets/multicountry/RDD2022/yolo/data.yaml",
        help="YOLO dataset configuration containing the image's paired labels.",
    )
    parser.add_argument("--video", help="Optional held-out RGB video for CNN + Spatial LSTM smoke test.")
    args = parser.parse_args()

    app = load_app_module()
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = ROOT / image_path
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Input image is unreadable: {image_path}")

    data_yaml = Path(args.data_yaml)
    if not data_yaml.is_absolute():
        data_yaml = ROOT / data_yaml
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data["names"]
    class_names = list(names.values()) if isinstance(names, dict) else list(names)
    pothole_id = class_names.index(app.POTHOLE_CLASS_NAME)
    if app.POTHOLE_CLASS_NAME != "D40":
        raise AssertionError("Dashboard pothole class name differs from dataset configuration.")
    if image_path.parent.name not in {"train", "val", "test"}:
        raise ValueError("Image must be inside a split folder so its paired label can be resolved.")
    label_path = data_yaml.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
    if not label_path.is_file():
        raise FileNotFoundError(f"Paired YOLO label not found: {label_path}")
    gt_boxes = read_ground_truth(label_path, image.shape[1], image.shape[0], pothole_id)
    if not gt_boxes:
        raise ValueError(f"Ground-truth label has no {app.POTHOLE_CLASS_NAME} objects: {label_path}")

    run_dir = ROOT / "results/pothole_debug" / datetime.now().strftime("dashboard_smoke_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    original_copy = run_dir / f"{image_path.stem}_original.jpg"
    if not cv2.imwrite(str(original_copy), image):
        raise OSError(f"Could not write original image copy: {original_copy}")
    report = {
        "image": str(image_path), "label": str(label_path),
        "dataset_class_id": pothole_id, "dataset_class_name": app.POTHOLE_CLASS_NAME,
        "ground_truth_potholes": gt_boxes, "original_image_copy": str(original_copy), "pipelines": {},
    }

    for model_name in (
        "YOLO12n + EfficientNetB0", "YOLO12n + ResNet50", "YOLO12n + ViT-B/16",
    ):
        bundle = app.load_models(model_name)
        inference_started = time.perf_counter()
        confidence = 0.075
        annotated, rows = app._infer_frame(image.copy(), bundle, confidence, model_name, image_path.name)
        inference_seconds = round(time.perf_counter() - inference_started, 5)
        potholes = [row for row in rows if row["detector_class"] == app.POTHOLE_CLASS_NAME]
        matched_gt = set()
        detections = []
        for row in potholes:
            box = (row["x1"], row["y1"], row["x2"], row["y2"])
            overlaps = [(box_iou(box, gt), index) for index, gt in enumerate(gt_boxes) if index not in matched_gt]
            best_iou, best_index = max(overlaps, default=(0.0, -1))
            if best_iou >= 0.5:
                matched_gt.add(best_index)
            detections.append({
                "detector_class": row["detector_class"],
                "detector_confidence": row["detector_confidence"],
                "classifier_class": row["classifier_class"],
                "classifier_confidence": row["class_confidence"],
                "xyxy": list(box), "best_unmatched_gt_iou": round(best_iou, 5),
            })
        output = run_dir / f"{model_name.replace('/', '-').replace('+', 'and').replace(' ', '_')}.jpg"
        if not cv2.imwrite(str(output), annotated):
            raise OSError(f"Could not write annotated image: {output}")
        report["pipelines"][model_name] = {
            "status": "matched ground truth" if matched_gt else "failed: no D40 box matched ground truth at IoU >= 0.5",
            "detector_checkpoint": app.MODELS[model_name].get("detector"),
            "classifier_checkpoint": app.MODELS[model_name].get("classifier"),
            "confidence_threshold": confidence,
            "detections": detections, "distinct_ground_truths_matched": len(matched_gt),
            "inference_seconds": inference_seconds, "annotated_output": str(output),
        }

    blank = np.full((512, 512, 3), 255, dtype=np.uint8)
    bundle = app.load_models("YOLO12n + ViT-B/16")
    annotated, rows = app._infer_frame(blank, bundle, 0.90, "YOLO12n + ViT-B/16", "zero_detection_blank.png")
    zero_output = run_dir / "zero_detection_blank.png"
    if not cv2.imwrite(str(zero_output), annotated):
        raise OSError(f"Could not write zero-detection image: {zero_output}")
    report["zero_detection_test"] = {
        "status": "passed" if not rows else "failed: detections on blank image",
        "confidence": 0.90, "detections": len(rows), "output": str(zero_output),
    }

    if args.video:
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = ROOT / video_path
        if not video_path.is_file():
            raise FileNotFoundError(f"Input video not found: {video_path}")

        class Upload:
            name = video_path.name

            def getvalue(self):
                return video_path.read_bytes()

        video_started = time.perf_counter()
        output_bytes, rows, sampled_frames, duration = app._run_video(
            Upload(), "CNN + Spatial LSTM segmentation", 0.10, 0.25,
        )
        processing_seconds = round(time.perf_counter() - video_started, 5)
        video_output = run_dir / f"{video_path.stem}_segmentation_smoke.mp4"
        video_output.write_bytes(output_bytes)
        report["segmentation_video_smoke"] = {
            "input": str(video_path), "checkpoint": app.MODELS["CNN + Spatial LSTM segmentation"]["classifier"],
            "status": "inference smoke passed; segmentation metrics not evaluated for configured V3 checkpoint",
            "input_duration_seconds": duration, "output_frames": sampled_frames,
            "processing_seconds": processing_seconds,
            "processing_throughput_frames_per_second": round(sampled_frames / processing_seconds, 3) if processing_seconds else None,
            "output": str(video_output), "frame_rows": len(rows),
        }

    report["overall_status"] = (
        "passed" if all(
            value["status"] == "matched ground truth"
            for value in report["pipelines"].values()
        ) and report["zero_detection_test"]["status"] == "passed"
        else "failed: at least one check did not pass"
    )
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
