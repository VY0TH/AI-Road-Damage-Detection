r"""Unified Streamlit frontend for the five road damage model pipelines.

Run from the project root with:
    python -m streamlit run src/89_road_damage_app.py
"""
from __future__ import annotations

import io
import importlib.util
import json
import sqlite3
import tempfile
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
TORCH_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
YOLO_DEVICE = "0" if torch.cuda.is_available() else "cpu"
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "webp"]
VIDEO_EXTENSIONS = ["mp4", "mov", "avi", "mkv"]
CLASS_NAMES = ["D00", "D01", "D0w0", "D10", "D11", "D20", "D40", "D43", "D44", "D50"]
# Verified in the active YOLO data.yaml and all detector checkpoints by name.
POTHOLE_CLASS_NAME = "D40"
MIN_POTHOLE_RESCUE_DETECTOR_CONFIDENCE = 0.03
POTHOLE_WEAK_SCORE_CEILING = 0.05
MIN_POTHOLE_RESCUE_CLASSIFIER_CONFIDENCE = 0.90
MIN_POTHOLE_RESCUE_SIDE_PIXELS = 18
MIN_CLASSIFIER_CONFIDENCE = 0.60
MAX_UNVERIFIED_BOX_AREA_RATIO = 0.12
# Validation-only ablation on 237 grouped real images: area > .12 plus this
# confirmation threshold removed five unmatched broad proposals and no matched
# non-D40 boxes. Keep broad D40 proposals untouched because that class is sparse.
MIN_BROAD_BOX_CLASSIFIER_CONFIDENCE = 0.97
# Existing experimental coverage thresholds from src/72_estimate_severity.py.
SEVERITY_LOW_MAX = 5.0
SEVERITY_MEDIUM_MAX = 15.0

YOLO_CHECKPOINT = "results/training/yolo12n_30_epochs/weights/best.pt"
D40_BALANCED_CHECKPOINT = "results/training/yolo12n_d40_balanced_10ep_20261015/weights/best.pt"
MODELS = {
    "YOLO12n + EfficientNetB0": {
        "description": "D40-balanced YOLO12n localizes damage; EfficientNetB0 classifies each detected crop.",
        "kind": "efficientnet", "detector": D40_BALANCED_CHECKPOINT,
        "classifier": "results/efficientnet_b0/best_efficientnet_b0.pt",
        "validation_evidence": "Grouped real validation: D40 P .231, R .103, F1 .143 at confidence .075 (IoU .50).",
    },
    "YOLO12n + ResNet50": {
        "description": "D40-balanced YOLO12n localizes damage; ResNet50 classifies each detected crop.",
        "kind": "resnet", "detector": D40_BALANCED_CHECKPOINT,
        "classifier": "results/resnet50_road_damage/best_resnet50.pt",
        "validation_evidence": "Grouped real validation: D40 P .222, R .069, F1 .105 at confidence .075 (IoU .50).",
    },
    "YOLO12n + ViT-B/16": {
        "description": "D40-balanced YOLO12n localizes damage; ViT-B/16 classifies each detected crop.",
        "kind": "vit", "detector": D40_BALANCED_CHECKPOINT,
        "classifier": "results/vit/best_vit.pt",
        "validation_evidence": "Grouped real validation: D40 P .176, R .103, F1 .130 at confidence .075 (IoU .50).",
    },
    "CNN + Spatial LSTM segmentation": {
        "description": "Experimental temporal model for binary damaged-area masks. Video only; it does not name damage types.",
        "kind": "segmentation",
        "classifier": "results/cnn_lstm/segmentation/best_cnn_lstm_spatial_v3.pt",
        "validation_evidence": "Independent 10-video test: Dice .784, IoU .645, pixel accuracy .916 at threshold .50.",
    },
    "YOLO12n + GAN + EfficientNetB0": {
        "description": "Experimental synthetic images are used only as training augmentation; validation stays real-only.",
        "kind": "gan_hybrid", "detector": YOLO_CHECKPOINT,
        "classifier": "results/efficientnet_b0_gan_augmented/best_efficientnet_b0.pt",
        "augmentation_checkpoint": "results/conditional_gan_selected_checkpoints/conditional_generator_final.pt",
    },
}

DATABASE_PATH = ROOT / "results" / "road_damage.db"


def _checkpoint_path(relative: str) -> Path:
    return ROOT / relative


def _required_checkpoints(config):
    paths = []
    for key in ("detector", "classifier", "augmentation_checkpoint"):
        relative = config.get(key)
        if relative:
            paths.append((key, _checkpoint_path(relative)))
    return paths


def _checkpoint_status(config):
    statuses = []
    for key, path in _required_checkpoints(config):
        exists = path.is_file()
        if key == "classifier" and config.get("kind") == "gan_hybrid":
            exists = _gan_classifier_is_ready(path)
        statuses.append((key, path, exists))
    return statuses


def _parse_user_location(latitude_text, longitude_text):
    latitude_text = latitude_text.strip()
    longitude_text = longitude_text.strip()
    if not latitude_text and not longitude_text:
        return None
    if not latitude_text or not longitude_text:
        raise ValueError("Enter both latitude and longitude, or leave both blank.")
    try:
        latitude, longitude = float(latitude_text), float(longitude_text)
    except ValueError as exc:
        raise ValueError("Latitude and longitude must be numeric.") from exc
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Coordinates are outside valid latitude/longitude ranges.")
    return {"latitude": latitude, "longitude": longitude, "source": "user supplied"}


def _model_status(config):
    """Return a user-facing availability and evaluation status."""
    statuses = _checkpoint_status(config)
    if config.get("kind") == "gan_hybrid" and not _gan_classifier_is_ready(
        _checkpoint_path(config["classifier"])
    ):
        return "Unavailable", "Generated samples are noise-like and fail the visual quality check; the augmented classifier has not been trained."
    if config.get("kind") == "segmentation":
        if any(not exists for _key, _path, exists in statuses):
            return "Unavailable", "Required segmentation checkpoint is missing."
        if config.get("validation_evidence"):
            return "Available", config["validation_evidence"]
        return "Not evaluated", "The dashboard supports video mask inference, but this checkpoint has no recorded independent test evaluation."
    missing = [key for key, _path, exists in statuses if not exists]
    if missing:
        return "Unavailable", "Required model checkpoint is missing or not verified."
    if config.get("validation_evidence"):
        return "Available", config["validation_evidence"] + " Not an independent test score."
    return "Available", "Image smoke-tested; end-to-end validation status: Not evaluated."


def _render_model_status():
    st.markdown("#### Model status")
    status_models = [
        "YOLO12n + EfficientNetB0", "YOLO12n + ResNet50", "YOLO12n + ViT-B/16",
        "CNN + Spatial LSTM segmentation", "YOLO12n + GAN + EfficientNetB0",
    ]
    for start in range(0, len(status_models), 2):
        status_cols = st.columns(2)
        for col, status_name in zip(status_cols, status_models[start:start + 2]):
            status, reason = _model_status(MODELS[status_name])
            with col:
                st.markdown(f"**{status_name}**")
                st.caption(f"{status} · {reason}")
    with st.expander("GAN training"):
        gan_classifier = _checkpoint_path(MODELS["YOLO12n + GAN + EfficientNetB0"]["classifier"])
        if _gan_classifier_is_ready(gan_classifier):
            st.caption("Status: Trained. The conditional GAN generated 50 crops across five classes; EfficientNetB0 trained with these added to the training split. Validation used real crops only.")
            st.caption(f"Classifier checkpoint: {gan_classifier.relative_to(ROOT)}")
        else:
            st.caption("Status: Unavailable. The GAN generator exists, but the augmented EfficientNet classifier has not been trained.")
            st.caption("Generate samples and train the classifier from the project root. Synthetic samples are added to training only; validation uses real crops.")
            st.code("python src/algorithms/gan/52_generate_gan_samples.py\n$env:ROAD_DAMAGE_USE_GAN_AUGMENT = '1'\npython src/algorithms/efficientnet/39_train_efficientnet.py", language="powershell")


def _gan_classifier_is_ready(path):
    if not path.is_file():
        return False
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return bool(checkpoint.get("gan_augmentation_enabled")) and int(
            checkpoint.get("gan_training_samples", 0)
        ) > 0
    except Exception:
        return False


@st.cache_resource(show_spinner="Loading detection and classification models…")
def load_models(model_name: str):
    config = MODELS[model_name]
    kind = config["kind"]
    if kind == "gan_hybrid" and not _gan_classifier_is_ready(
        _checkpoint_path(config["classifier"])
    ):
        raise RuntimeError(
            "The GAN-augmented EfficientNet checkpoint has not been trained. "
            "Set ROAD_DAMAGE_USE_GAN_AUGMENT=1 and run src/39_train_efficientnet.py; "
            "that run saves to results/efficientnet_b0_gan_augmented/."
        )
    detector_path = _checkpoint_path(config["detector"]) if config.get("detector") else None
    if kind == "yolo":
        if detector_path is None or not detector_path.is_file():
            raise FileNotFoundError(f"YOLO checkpoint not found: {detector_path}")
        return YOLO(str(detector_path)), None, [], None

    classifier_path = _checkpoint_path(config["classifier"])
    paths = [classifier_path]
    if detector_path is not None:
        paths.append(detector_path)
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required checkpoint(s) not found: " + ", ".join(missing))

    checkpoint = torch.load(classifier_path, map_location=TORCH_DEVICE, weights_only=False)
    if kind == "segmentation":
        model_script = ROOT / "src" / "58_train_cnn_lstm_segmentation.py"
        spec = importlib.util.spec_from_file_location("road_damage_segmenter", model_script)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load the segmenter architecture from {model_script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        segmenter = module.CNNLSTMSegmenter().to(TORCH_DEVICE)
        state = checkpoint.get("model_state_dict", checkpoint)
        segmenter.load_state_dict(state)
        segmenter.eval()
        return None, segmenter, [], None

    detector = YOLO(str(detector_path))
    detector_names = detector.names
    detector_name_values = (
        list(detector_names.values()) if isinstance(detector_names, dict)
        else list(detector_names)
    )
    if POTHOLE_CLASS_NAME not in detector_name_values:
        raise RuntimeError(
            f"Detector {detector_path.relative_to(ROOT)} does not define the verified "
            f"pothole class {POTHOLE_CLASS_NAME!r}; refusing to apply pothole-specific filtering."
        )

    names = list(checkpoint.get("class_names", CLASS_NAMES))
    if len(names) != len(set(names)) or set(names) != set(detector_name_values):
        raise RuntimeError(
            f"Detector and classifier class-name sets differ: detector={detector_name_values}, "
            f"classifier={names}. Predictions cannot be safely aligned by name."
        )
    if POTHOLE_CLASS_NAME not in names:
        raise RuntimeError(
            f"Classifier {classifier_path.relative_to(ROOT)} does not define "
            f"{POTHOLE_CLASS_NAME!r}; it cannot verify low-confidence pothole candidates."
        )
    if kind in {"efficientnet", "gan_hybrid"}:
        classifier = models.efficientnet_b0(weights=None)
        classifier.classifier[1] = nn.Linear(classifier.classifier[1].in_features, len(names))
    elif kind == "resnet":
        classifier = models.resnet50(weights=None)
        classifier.fc = nn.Linear(classifier.fc.in_features, len(names))
    else:
        classifier = models.vit_b_16(weights=None)
        classifier.heads.head = nn.Linear(classifier.heads.head.in_features, len(names))
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.to(TORCH_DEVICE).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return detector, classifier, names, transform


def _classify(classifier, class_names, transform, crop_bgr):
    # The training crops are generated with LAB-luminance CLAHE by
    # src/12_create_classification_crops.py; apply that same preprocessing here.
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    luminance, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(
        cv2.merge((clahe.apply(luminance), a_channel, b_channel)),
        cv2.COLOR_LAB2BGR,
    )
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(TORCH_DEVICE)
    with torch.inference_mode():
        probs = torch.softmax(classifier(tensor), dim=1)[0]
    index = int(probs.argmax().item())
    return class_names[index], float(probs[index].item())


def _tile_starts(length, tile_size, overlap):
    if length <= tile_size:
        return [0]
    stride = max(1, int(tile_size * (1.0 - overlap)))
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _image_tiles(frame, tile_size=640, overlap=0.25):
    """Return overlapping tiles with their top-left offsets in the original image."""
    height, width = frame.shape[:2]
    xs = _tile_starts(width, tile_size, overlap)
    ys = _tile_starts(height, tile_size, overlap)
    for y in ys:
        for x in xs:
            yield frame[y:min(y + tile_size, height), x:min(x + tile_size, width)], x, y


def _suppress_duplicate_boxes(candidates, iou_threshold=0.50, containment_threshold=0.90, max_boxes=30):
    """Merge repeated tile proposals while preferring a tight box over a broad one."""
    ranked = sorted(
        candidates,
        key=lambda item: (item["rank_score"], -item["area"]),
        reverse=True,
    )
    kept = []
    for candidate in ranked:
        x1, y1, x2, y2 = candidate["box"]
        area = candidate["area"]
        duplicate_index = None
        for accepted in kept:
            ax1, ay1, ax2, ay2 = accepted["box"]
            intersection_w = max(0, min(x2, ax2) - max(x1, ax1))
            intersection_h = max(0, min(y2, ay2) - max(y1, ay1))
            intersection = intersection_w * intersection_h
            if not intersection:
                continue
            accepted_area = accepted["area"]
            union = area + accepted_area - intersection
            iou = intersection / union
            containment = intersection / min(area, accepted_area)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
            center_distance = ((cx - acx) ** 2 + (cy - acy) ** 2) ** 0.5
            candidate_diagonal = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            accepted_diagonal = ((ax2 - ax1) ** 2 + (ay2 - ay1) ** 2) ** 0.5
            near_duplicate = intersection > 0 and (
                center_distance <= 0.65 * min(candidate_diagonal, accepted_diagonal)
            )
            if iou >= iou_threshold or near_duplicate:
                duplicate_index = kept.index(accepted)
                break
            if containment >= containment_threshold:
                # A tight pothole box beats a broad box that merely encloses it.
                if area < accepted_area * 0.85:
                    duplicate_index = kept.index(accepted)
                else:
                    duplicate_index = -1
                break
        if duplicate_index is None:
            kept.append(candidate)
        elif duplicate_index >= 0 and candidate["area"] < kept[duplicate_index]["area"]:
            kept[duplicate_index] = candidate
        if len(kept) >= max_boxes:
            break

    # A broad box can overlap several real potholes without meeting pairwise
    # containment rules. Remove it when it encloses multiple tighter detections.
    filtered = []
    for candidate in kept:
        x1, y1, x2, y2 = candidate["box"]
        enclosed = 0
        for other in kept:
            if other is candidate or other["area"] >= candidate["area"] * 0.50:
                continue
            ox1, oy1, ox2, oy2 = other["box"]
            intersection = max(0, min(x2, ox2) - max(x1, ox1)) * max(0, min(y2, oy2) - max(y1, oy1))
            if intersection / max(1, other["area"]) >= containment_threshold:
                enclosed += 1
        if enclosed < 2:
            filtered.append(candidate)
    return filtered


def _infer_frame(frame, models_bundle, confidence, model_name, source_name, frame_number=None, timestamp=None):
    inference_started = time.perf_counter()
    detector, classifier, class_names, transform = models_bundle
    config = MODELS[model_name]
    annotated = frame.copy()
    candidates = []
    height, width = frame.shape[:2]
    for tile, offset_x, offset_y in _image_tiles(frame):
        prediction = detector.predict(
            source=tile, conf=0.001, iou=0.50, max_det=30,
            imgsz=640, device=YOLO_DEVICE, verbose=False,
        )[0]
        boxes = prediction.boxes
        if boxes is None:
            continue
        tile_height, tile_width = tile.shape[:2]
        for box in boxes:
            tx1, ty1, tx2, ty2 = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
            tx1, tx2 = max(0, tx1), min(tile_width, tx2)
            ty1, ty2 = max(0, ty1), min(tile_height, ty2)
            if tx2 <= tx1 or ty2 <= ty1:
                continue
            cls_id = int(box.cls[0].item())
            detector_names = prediction.names
            if isinstance(detector_names, dict):
                detector_name = str(detector_names.get(cls_id, cls_id))
            elif cls_id < len(detector_names):
                detector_name = str(detector_names[cls_id])
            else:
                detector_name = str(cls_id)
            detector_conf = float(box.conf[0].item())
            is_weak_pothole_candidate = (
                detector_name == POTHOLE_CLASS_NAME
                and detector_conf < max(confidence, POTHOLE_WEAK_SCORE_CEILING)
            )
            if detector_conf < confidence and detector_name != POTHOLE_CLASS_NAME:
                continue

            if config["kind"] == "yolo":
                label, class_conf = detector_name, detector_conf
            else:
                crop = tile[ty1:ty2, tx1:tx2]
                label, class_conf = _classify(classifier, class_names, transform, crop)
                if is_weak_pothole_candidate:
                    if (
                        detector_conf < MIN_POTHOLE_RESCUE_DETECTOR_CONFIDENCE
                        or label != POTHOLE_CLASS_NAME
                        or class_conf < MIN_POTHOLE_RESCUE_CLASSIFIER_CONFIDENCE
                        or (tx2 - tx1) < MIN_POTHOLE_RESCUE_SIDE_PIXELS
                        or (ty2 - ty1) < MIN_POTHOLE_RESCUE_SIDE_PIXELS
                    ):
                        continue
                elif class_conf < MIN_CLASSIFIER_CONFIDENCE:
                    continue
                elif detector_name != POTHOLE_CLASS_NAME and label != detector_name:
                    # Disagreement on non-pothole classes is a common source of
                    # broad false boxes; keep D40 localization visible because
                    # crop classifiers currently confuse some true potholes.
                    continue

            candidate_area_ratio = (
                max(1, (tx2 - tx1) * (ty2 - ty1))
                / max(1, tile_width * tile_height)
            )
            if (
                config["kind"] != "yolo"
                and candidate_area_ratio > MAX_UNVERIFIED_BOX_AREA_RATIO
                and (label != detector_name or class_conf < MIN_BROAD_BOX_CLASSIFIER_CONFIDENCE)
            ):
                # The active detector produces broad, false D20 boxes on a
                # clean road; only retain similarly broad proposals when the
                # crop classifier strongly confirms the detector's class.
                continue

            x1, y1 = tx1 + offset_x, ty1 + offset_y
            x2, y2 = tx2 + offset_x, ty2 + offset_y
            box_area = max(1, (x2 - x1) * (y2 - y1))
            tile_area = max(1, tile_width * tile_height)
            compactness = 1.0 - min(1.0, box_area / tile_area)
            rank_score = class_conf if is_weak_pothole_candidate else detector_conf
            candidates.append({
                "box": (x1, y1, x2, y2), "label": label,
                "class_confidence": class_conf, "detector_class": detector_name,
                "detector_confidence": detector_conf, "area": box_area,
                "rank_score": rank_score + 0.05 * compactness,
            })

    selected = _suppress_duplicate_boxes(candidates, max_boxes=30)
    rows = []
    for index, candidate in enumerate(selected, start=1):
        x1, y1, x2, y2 = candidate["box"]
        label = candidate["label"]
        class_conf = candidate["class_confidence"]
        detector_name = candidate["detector_class"]
        detector_conf = candidate["detector_confidence"]
        color = (46, 125, 83)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        if config["kind"] == "yolo":
            text = f"{detector_name} {detector_conf:.2f}"
        elif label == detector_name:
            text = f"{detector_name} det {detector_conf:.2f} cls {class_conf:.2f}"
        else:
            text = f"det {detector_name} {detector_conf:.2f} | cls {label} {class_conf:.2f}"
        cv2.putText(annotated, text, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        rows.append({
            "source": source_name, "frame": frame_number, "timestamp_seconds": timestamp,
            "model": model_name, "detection": index, "damage_class": detector_name,
            "classifier_class": label,
            "class_confidence": round(class_conf, 5), "detector_class": detector_name,
            "detector_confidence": round(detector_conf, 5),
            "inference_seconds_per_frame": None,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    elapsed = round(time.perf_counter() - inference_started, 5)
    for row in rows:
        row["inference_seconds_per_frame"] = elapsed
        row["detector_checkpoint"] = MODELS[model_name].get("detector")
        row["classifier_checkpoint"] = MODELS[model_name].get("classifier")
    return annotated, rows


def _run_images(files, model_name, confidence):
    bundle = load_models(model_name)
    results, rows = [], []
    progress = st.progress(0, text="Preparing images…")
    for index, uploaded in enumerate(files):
        data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            st.warning(f"Could not read {uploaded.name}; skipped.")
            continue
        inference_started = time.perf_counter()
        annotated, frame_rows = _infer_frame(image, bundle, confidence, model_name, uploaded.name)
        inference_seconds = round(time.perf_counter() - inference_started, 5)
        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok:
            results.append({
                "name": uploaded.name, "original": uploaded.getvalue(),
                "image": encoded.tobytes(), "detections": len(frame_rows),
                "inference_seconds": inference_seconds,
            })
        rows.extend(frame_rows)
        progress.progress((index + 1) / len(files), text=f"Processed {uploaded.name}")
    progress.empty()
    return results, rows


def _run_video(uploaded, model_name, confidence, sample_seconds):
    bundle = load_models(model_name)
    kind = MODELS[model_name]["kind"]
    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / ("input" + suffix)
        output_path = Path(temp_dir) / "annotated.mp4"
        input_path.write_bytes(uploaded.getvalue())
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError("The uploaded video could not be opened.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        step = max(1, round(fps * sample_seconds))
        sequence_length = 8
        segmentation_window_size = max(sequence_length, round(fps * 2.0))
        output_fps = (
            sequence_length * fps / segmentation_window_size
            if kind == "segmentation" else fps / step
        )
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not create the annotated video output.")
        rows, processed, frame_no = [], 0, 0
        progress = st.progress(0, text="Processing video…")
        try:
            if kind == "segmentation":
                window_size = segmentation_window_size
                sample_positions = np.linspace(0, window_size - 1, sequence_length).round().astype(int)
                window_start = 0
                while True:
                    original_frames = []
                    for _ in range(window_size):
                        ok, frame = cap.read()
                        if not ok:
                            break
                        original_frames.append(frame)
                        frame_no += 1
                    if not original_frames:
                        break
                    sampled = [original_frames[min(int(pos), len(original_frames) - 1)] for pos in sample_positions]
                    rgb = [cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (128, 128), interpolation=cv2.INTER_AREA) for frame in sampled]
                    tensor = torch.from_numpy(np.stack(rgb).astype(np.float32) / 255.0).permute(0, 3, 1, 2).unsqueeze(0).to(TORCH_DEVICE)
                    with torch.inference_mode():
                        probabilities = torch.sigmoid(bundle[1](tensor))[0, :, 0].cpu().numpy()
                    for index, (frame, probability) in enumerate(zip(sampled, probabilities)):
                        if sample_positions[index] >= len(original_frames):
                            continue
                        mask = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR) >= 0.5
                        overlay = frame.copy()
                        tint = np.zeros_like(overlay)
                        tint[:] = (55, 95, 220)
                        blended = cv2.addWeighted(overlay, 0.55, tint, 0.45, 0)
                        overlay[mask] = blended[mask]
                        coverage = float(mask.mean() * 100)
                        severity = (
                            "Low" if coverage < SEVERITY_LOW_MAX else
                            "Medium" if coverage < SEVERITY_MEDIUM_MAX else
                            "High"
                        )
                        cv2.putText(overlay, f"Experimental {severity}  |  mask {coverage:.1f}% area", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                        writer.write(overlay)
                        timestamp = window_start / fps + int(sample_positions[index]) / fps
                        rows.append({
                            "source": uploaded.name, "frame": window_start + int(sample_positions[index]),
                            "timestamp_seconds": round(timestamp, 3), "model": model_name,
                            "detection": index + 1, "damage_class": "Binary damage mask",
                            "class_confidence": None, "detector_class": None,
                            "detector_confidence": None, "damage_coverage_percent": round(coverage, 2),
                            "estimated_severity": severity,
                        })
                        processed += 1
                    window_start += window_size
                    progress.progress(min(1.0, frame_no / max(1, total)), text=f"Analyzed video through {window_start / fps:.1f}s")
                    if len(original_frames) < window_size:
                        break
            else:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_no % step == 0:
                        timestamp = frame_no / fps
                        annotated, frame_rows = _infer_frame(frame, bundle, confidence, model_name, uploaded.name, frame_no, round(timestamp, 3))
                        rows.extend(frame_rows)
                        writer.write(annotated)
                        processed += 1
                        progress.progress(min(1.0, frame_no / max(1, total)), text=f"Analyzed {timestamp:.1f}s")
                    frame_no += 1
        finally:
            cap.release()
            writer.release()
        progress.empty()
        video_bytes = output_path.read_bytes()
    return video_bytes, rows, processed, round(frame_no / fps, 2)


def _save_run(model_name, mode, rows, source_names=None, outputs=None, location=None):
    """Keep session summaries and append durable app records without changing legacy rows."""
    history = st.session_state.get("runs", [])
    run_id = uuid.uuid4().hex
    output_dir = ROOT / "results" / "dashboard_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_outputs = []
    for output_index, output in enumerate(outputs or [], start=1):
        suffix = ".mp4" if mode == "Video" else ".jpg"
        safe_stem = Path(output["name"]).stem[:80] or "road_damage"
        output_path = output_dir / f"{run_id}_{output_index}_{safe_stem}{suffix}"
        output_path.write_bytes(output["image"] if mode == "Image" else output["video"])
        saved_outputs.append(str(output_path.relative_to(ROOT)))

    timestamp = pd.Timestamp.now(tz="UTC").isoformat()
    config = MODELS[model_name]
    checkpoint_paths = {
        key: value for key, value in config.items()
        if key.endswith("checkpoint") or key in {"detector", "classifier"}
    }
    output_metadata = [
        {
            "source": output["name"],
            "inference_seconds": output.get("inference_seconds"),
            "processing_seconds": output.get("processing_seconds"),
            "processed_frames": output.get("processed_frames"),
            "input_duration_seconds": output.get("input_duration_seconds"),
        }
        for output in outputs or []
    ]
    connection = sqlite3.connect(DATABASE_PATH)
    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS dashboard_runs (
                run_id TEXT PRIMARY KEY,
                timestamp_utc TEXT NOT NULL,
                model TEXT NOT NULL,
                input_type TEXT NOT NULL,
                source_names_json TEXT NOT NULL,
                output_paths_json TEXT NOT NULL,
                detection_count INTEGER NOT NULL,
                run_payload_json TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS dashboard_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES dashboard_runs(run_id),
                source TEXT,
                frame INTEGER,
                timestamp_seconds REAL,
                detection INTEGER,
                damage_class TEXT,
                classifier_class TEXT,
                class_confidence REAL,
                detector_class TEXT,
                detector_confidence REAL,
                x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
                inference_seconds_per_frame REAL,
                detector_checkpoint TEXT,
                classifier_checkpoint TEXT,
                row_payload_json TEXT NOT NULL
            )
        """)
        connection.execute(
            "INSERT INTO dashboard_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, timestamp, model_name, mode,
             json.dumps(source_names or [], ensure_ascii=False),
             json.dumps(saved_outputs), len(rows),
             json.dumps({
                 "output_paths": saved_outputs,
                 "checkpoints": checkpoint_paths,
                 "outputs": output_metadata,
                 "location": location,
             }, ensure_ascii=False)),
        )
        detection_columns = (
            "source", "frame", "timestamp_seconds", "detection", "damage_class",
            "classifier_class", "class_confidence", "detector_class",
            "detector_confidence", "x1", "y1", "x2", "y2",
            "inference_seconds_per_frame", "detector_checkpoint", "classifier_checkpoint",
        )
        for row in rows:
            values = [row.get(column) for column in detection_columns]
            connection.execute(
                "INSERT INTO dashboard_detections "
                "(run_id, " + ", ".join(detection_columns) + ", row_payload_json) "
                "VALUES (" + ", ".join("?" for _ in range(len(detection_columns) + 2)) + ")",
                [run_id, *values, json.dumps(row, ensure_ascii=False)],
            )
        connection.commit()
    finally:
        connection.close()

    history.append({"run_id": run_id, "model": model_name, "input_type": mode, "rows": rows})
    st.session_state.runs = history


def _dashboard():
    all_rows = [row for run in st.session_state.get("runs", []) for row in run["rows"]]
    if not all_rows:
        st.info("Run an image or video analysis to populate this dashboard.")
        return
    df = pd.DataFrame(all_rows)
    left, center, right = st.columns(3)
    left.metric("Road damage detections", len(df))
    center.metric("Files analyzed", df["source"].nunique())
    scores = df["class_confidence"].dropna() if "class_confidence" in df else pd.Series(dtype=float)
    right.metric("Mean model score", f"{scores.mean():.1%}" if not scores.empty else "N/A")
    a, b = st.columns(2)
    with a:
        st.markdown("#### Damage classes")
        st.bar_chart(df["damage_class"].value_counts().sort_values(ascending=False), color="#3f8066")
    with b:
        st.markdown("#### Results by model")
        st.bar_chart(df.groupby("model").size().sort_values(ascending=False), color="#8caf9e")
    st.markdown("#### Detection records")
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("Download analysis CSV", df.to_csv(index=False).encode("utf-8"), "road_damage_analysis.csv", "text/csv")


def _model_comparison():
    rows = []

    def add_classifier(name, checkpoint_rel, history_rel, report_rel=None):
        checkpoint = _checkpoint_path(checkpoint_rel)
        history_path = _checkpoint_path(history_rel)
        values = {}
        if history_path.is_file():
            if history_path.suffix.lower() == ".json":
                history = json.loads(history_path.read_text(encoding="utf-8"))
                values = history[-1] if history else {}
            else:
                history = pd.read_csv(history_path)
                values = history.iloc[-1].to_dict() if not history.empty else {}
        macro_f1 = None
        if report_rel and _checkpoint_path(report_rel).is_file():
            report = json.loads(_checkpoint_path(report_rel).read_text(encoding="utf-8"))
            macro_f1 = report.get("macro avg", {}).get("f1-score")
        rows.append({
            "Model": name,
            "Inference checkpoint": "Available" if checkpoint.is_file() else "Missing",
            "Logged epoch": values.get("epoch"),
            "Validation accuracy": values.get("val_accuracy"),
            "Validation loss": values.get("val_loss"),
            "Validation macro F1": macro_f1,
            "Metric source": str(history_path.relative_to(ROOT)) if history_path.is_file() else "No training log",
        })

    add_classifier(
        "EfficientNetB0", "results/efficientnet_b0/best_efficientnet_b0.pt",
        "results/efficientnet_b0/training_history.json",
        "results/efficientnet_b0/classification_report.json",
    )
    add_classifier(
        "ResNet50", "results/resnet50_road_damage/best_resnet50.pt",
        "results/resnet50_road_damage/training_history.csv",
    )
    add_classifier(
        "ViT-B/16", "results/vit/best_vit.pt",
        "results/vit/training_history.json", "results/vit/classification_report.json",
    )

    detector_log = _checkpoint_path("results/training/yolo12n_30_epochs/results.csv")
    detector_eval = _checkpoint_path("results/yolo30_full_validation/metrics.csv")
    detector_report_paths = sorted(
        ROOT.glob("results/pothole_debug/full_validation_*/yolo12n_30_epochs_report/validation_metrics.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    detector_checkpoint = _checkpoint_path(YOLO_CHECKPOINT)
    detector_values = {}
    if detector_log.is_file():
        log = pd.read_csv(detector_log)
        detector_values = log.iloc[-1].to_dict() if not log.empty else {}
    detector_eval_values = {}
    detector_report_path = detector_report_paths[0] if detector_report_paths else None
    if detector_report_path is not None:
        try:
            detector_eval_values = json.loads(detector_report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            detector_eval_values = {}
    elif detector_eval.is_file():
        eval_frame = pd.read_csv(detector_eval)
        if {"Metric", "Value"}.issubset(eval_frame.columns):
            detector_eval_values = dict(zip(eval_frame["Metric"], eval_frame["Value"]))
    pothole_metrics = detector_eval_values.get("per_class", {}).get(POTHOLE_CLASS_NAME, {})
    rows.append({
        "Model": "YOLO12n detector",
        "Inference checkpoint": "Available" if detector_checkpoint.is_file() else "Missing",
        "Logged epoch": detector_values.get("epoch"),
        "Validation accuracy": None,
        "Validation loss": detector_values.get("val/box_loss"),
        "Validation macro F1": None,
        "Detector precision": detector_eval_values.get("precision", detector_eval_values.get("Precision", detector_values.get("metrics/precision(B)"))),
        "Detector recall": detector_eval_values.get("recall", detector_eval_values.get("Recall", detector_values.get("metrics/recall(B)"))),
        "Detector F1": detector_eval_values.get("f1"),
        "Detector mAP50": detector_eval_values.get("map50", detector_eval_values.get("mAP50", detector_values.get("metrics/mAP50(B)"))),
        "Detector mAP50-95": detector_eval_values.get("map50_95", detector_eval_values.get("mAP50-95")),
        "D40 precision": pothole_metrics.get("precision"),
        "D40 recall": pothole_metrics.get("recall"),
        "D40 mAP50": pothole_metrics.get("map50"),
        "Metric source": str(detector_report_path.relative_to(ROOT)) if detector_report_path else str(detector_eval.relative_to(ROOT)) if detector_eval_values else str(detector_log.relative_to(ROOT)) if detector_log.is_file() else "No detector log",
    })

    segmentation_path = _checkpoint_path(MODELS["CNN + Spatial LSTM segmentation"]["classifier"])
    if segmentation_path.is_file():
        checkpoint = torch.load(segmentation_path, map_location="cpu", weights_only=True)
        rows.append({
            "Model": "CNN + Spatial LSTM (experimental)",
            "Inference checkpoint": "Available",
            "Logged epoch": checkpoint.get("epoch"),
            "Validation loss": checkpoint.get("val_loss"),
            "Validation Dice": checkpoint.get("val_dice"),
            "Validation IoU": checkpoint.get("val_iou"),
            "Metric source": str(segmentation_path.relative_to(ROOT)) + " metadata",
        })
    else:
        rows.append({"Model": "CNN + Spatial LSTM (experimental)", "Inference checkpoint": "Missing"})

    gan_path = _checkpoint_path(MODELS["YOLO12n + GAN + EfficientNetB0"]["classifier"])
    gan_history_path = _checkpoint_path("results/efficientnet_b0_gan_augmented/training_history.json")
    gan_values = {}
    if gan_history_path.is_file():
        gan_history = json.loads(gan_history_path.read_text(encoding="utf-8"))
        gan_values = gan_history[-1] if gan_history else {}
    gan_report_path = _checkpoint_path("results/efficientnet_b0_gan_augmented/classification_report.json")
    gan_report = json.loads(gan_report_path.read_text(encoding="utf-8")) if gan_report_path.is_file() else {}
    rows.append({
        "Model": "YOLO12n + GAN + EfficientNetB0",
        "Inference checkpoint": "Available" if _gan_classifier_is_ready(gan_path) else "Unavailable",
        "Logged epoch": gan_values.get("epoch"),
        "Validation accuracy": gan_values.get("val_accuracy"),
        "Validation loss": gan_values.get("val_loss"),
        "Validation macro F1": gan_report.get("macro avg", {}).get("f1-score"),
        "Metric source": str(gan_history_path.relative_to(ROOT)) if gan_history_path.is_file() else "No GAN-augmented classifier training log",
    })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    detector_comparisons = sorted(
        ROOT.glob("results/detector_candidate_comparison_*/comparison.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    st.markdown("### Detector checkpoint comparison")
    if detector_comparisons:
        comparison_path = detector_comparisons[0]
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        detector_rows = []
        for name, values in comparison.get("models", {}).items():
            if values.get("status") != "evaluated":
                continue
            pothole = values.get("per_class", {}).get(POTHOLE_CLASS_NAME, {})
            detector_rows.append({
                "Checkpoint": name,
                "Precision": values.get("precision"),
                "Recall": values.get("recall"),
                "F1": values.get("f1"),
                "mAP50": values.get("map50"),
                "mAP50-95": values.get("map50_95"),
                "D40 recall": pothole.get("recall"),
                "D40 mAP50": pothole.get("map50"),
            })
        st.caption(
            "All listed detector checkpoints use the same real validation split and settings. "
            f"Report: {comparison_path.relative_to(ROOT)}"
        )
        if detector_rows:
            st.dataframe(pd.DataFrame(detector_rows), width="stretch", hide_index=True)
    else:
        st.info("Not evaluated: no same-split detector checkpoint comparison is available.")

    st.markdown("### Training curves")
    histories = {
        "EfficientNetB0": "results/efficientnet_b0/training_history.json",
        "ResNet50": "results/resnet50_road_damage/training_history.csv",
        "ViT-B/16": "results/vit/training_history.json",
        "GAN-augmented EfficientNet": "results/efficientnet_b0_gan_augmented/training_history.json",
        "YOLO12n": "results/training/yolo12n_30_epochs/results.csv",
    }
    available = [name for name, rel in histories.items() if _checkpoint_path(rel).is_file()]
    if available:
        selected = st.selectbox("Recorded training log", available)
        path = _checkpoint_path(histories[selected])
        history = pd.read_json(path) if path.suffix.lower() == ".json" else pd.read_csv(path)
        if "epoch" in history.columns:
            if selected == "YOLO12n":
                curve_columns = [column for column in ("metrics/mAP50(B)", "metrics/precision(B)", "metrics/recall(B)") if column in history]
            else:
                curve_columns = [column for column in ("train_loss", "val_loss", "train_accuracy", "val_accuracy") if column in history]
            if curve_columns:
                st.line_chart(history.set_index("epoch")[curve_columns])

    st.markdown("### Available confusion matrices")
    detector_matrices = [
        _checkpoint_path("results/training/yolo12n_30_epochs/confusion_matrix.png"),
        _checkpoint_path("results/training/yolo12n_30_epochs/confusion_matrix_normalized.png"),
    ]
    available_detector_matrices = [path for path in detector_matrices if path.is_file()]
    if available_detector_matrices:
        columns = st.columns(len(available_detector_matrices))
        for column, path in zip(columns, available_detector_matrices):
            column.image(str(path), caption=path.name, width="stretch")
    matrices = {
        "EfficientNetB0": ("results/efficientnet_b0/confusion_matrix.csv", CLASS_NAMES),
        "ViT-B/16": ("results/vit/confusion_matrix.csv", CLASS_NAMES),
        "GAN-augmented EfficientNet": ("results/efficientnet_b0_gan_augmented/confusion_matrix.csv", CLASS_NAMES),
    }
    for name, (relative, labels) in matrices.items():
        path = _checkpoint_path(relative)
        if path.is_file():
            matrix = pd.read_csv(path, header=None)
            if matrix.shape == (len(labels), len(labels)):
                matrix.index = labels
                matrix.columns = labels
                with st.expander(f"{name} confusion matrix · {relative}"):
                    st.dataframe(matrix, width="stretch")


def _database_history():
    st.markdown("## Stored detection history")
    if not DATABASE_PATH.is_file():
        st.info(f"No SQLite database found at {DATABASE_PATH.relative_to(ROOT)}.")
        return
    try:
        with sqlite3.connect(f"file:{DATABASE_PATH.as_posix()}?mode=ro", uri=True) as connection:
            table_names = set(pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'", connection,
            )["name"].tolist())
            legacy = (
                pd.read_sql_query("SELECT * FROM detections ORDER BY id DESC", connection)
                if "detections" in table_names else pd.DataFrame()
            )
            app_runs = (
                pd.read_sql_query("SELECT * FROM dashboard_runs ORDER BY timestamp_utc DESC", connection)
                if "dashboard_runs" in table_names else pd.DataFrame()
            )
            app_detections = (
                pd.read_sql_query(
                    "SELECT d.*, r.timestamp_utc, r.model, r.input_type "
                    "FROM dashboard_detections d JOIN dashboard_runs r USING(run_id) "
                    "ORDER BY d.id DESC", connection,
                ) if "dashboard_detections" in table_names else pd.DataFrame()
            )
    except sqlite3.Error as exc:
        st.error(f"Could not read the SQLite history: {exc}")
        return
    st.markdown("### Existing detection records")
    if legacy.empty:
        st.info("No legacy detections are stored.")
    else:
        st.caption("The pre-existing detections table is read without changing its schema.")
        frame = legacy
        if "efficientnet_class" in frame:
            classes = sorted(frame["efficientnet_class"].dropna().astype(str).unique())
            chosen = st.multiselect("EfficientNetB0 class", classes, default=classes)
            frame = frame[frame["efficientnet_class"].astype(str).isin(chosen)].copy()
        c1, c2, c3 = st.columns(3)
        c1.metric("Stored detections", len(frame))
        c2.metric("Images represented", frame["image"].nunique() if "image" in frame else "N/A")
        if "classes_agree" in frame:
            agreement = pd.to_numeric(frame["classes_agree"], errors="coerce").mean()
            c3.metric("Class agreement", f"{agreement:.1%}" if pd.notna(agreement) else "N/A")
        st.dataframe(frame, width="stretch", hide_index=True)
        st.download_button(
            "Download stored detections CSV", frame.to_csv(index=False).encode("utf-8"),
            "stored_road_detections.csv", "text/csv",
        )
        if {"latitude", "longitude"}.issubset(frame.columns):
            located = frame.dropna(subset=["latitude", "longitude"])
            located = located[
                located.latitude.between(-90, 90) & located.longitude.between(-180, 180)
            ]
            if not located.empty:
                st.markdown("#### Recorded locations")
                st.map(located.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])
            else:
                st.info("No stored detections have valid GPS coordinates.")
        else:
            st.info("The detections table has no GPS columns.")

    st.markdown("### Dashboard analyses")
    if app_runs.empty:
        st.info("No dashboard analyses have been saved yet.")
    else:
        st.dataframe(
            app_runs[["timestamp_utc", "model", "input_type", "source_names_json", "detection_count", "output_paths_json"]],
            width="stretch", hide_index=True,
        )
        locations = []
        for record in app_runs["run_payload_json"]:
            try:
                location = json.loads(record).get("location")
            except (TypeError, json.JSONDecodeError):
                location = None
            if not location or location.get("source") != "user supplied":
                continue
            latitude = location.get("latitude")
            longitude = location.get("longitude")
            if (
                isinstance(latitude, (int, float)) and isinstance(longitude, (int, float))
                and -90 <= latitude <= 90 and -180 <= longitude <= 180
            ):
                locations.append({"lat": latitude, "lon": longitude})
        if locations:
            st.markdown("#### User supplied locations")
            st.map(pd.DataFrame(locations))
        else:
            st.info("No dashboard runs have user supplied GPS coordinates.")
        st.download_button(
            "Download dashboard detections CSV",
            app_detections.to_csv(index=False).encode("utf-8"),
            "dashboard_detections.csv", "text/csv",
        )
        if not app_detections.empty:
            st.dataframe(app_detections.drop(columns="row_payload_json"), width="stretch", hide_index=True)


def main():
    st.set_page_config(page_title="RoadSight | Damage Analysis", page_icon="🛣️", layout="wide")
    st.markdown("""
    <style>
    .stApp { background: #f6f8f7; color: #23332d; }
    [data-testid="stHeader"] { background: rgba(246,248,247,.92); }
    div[data-testid="stAlert"] { color: #222b25; background: #fff4d6; border-color: #d7a72f; }
    .block-container { max-width: 1320px; padding-top: 2rem; }
    [data-testid="stMetric"] { background: white; border: 1px solid #e5ebe7; border-radius: 14px; padding: 16px 18px; }
    [data-testid="stFileUploader"] { background: white; border-radius: 14px; }
    .hero { background: linear-gradient(120deg,#e7f1ec,#f2f6f3); border: 1px solid #dfebe4; padding: 25px 30px; border-radius: 18px; margin: 4px 0 20px; }
    .hero p { color: #607168; margin: 0; }
    .status-warning { color: #222b25; background: #fff4d6; border: 1px solid #d7a72f; border-left: 5px solid #9a6500; border-radius: 8px; padding: 12px 16px; margin: 10px 0 16px; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1 style="margin:0 0 6px">Road damage analysis</h1><p>Choose a trained model, analyze your own road images or videos, and review every result in one dashboard.</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### RoadSight")
        page = st.radio(
            "Workspace",
            ["Image analysis", "Video analysis", "Analysis dashboard", "Model comparison", "Database history"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(f"Inference device · {TORCH_DEVICE.upper()}")

    if page == "Analysis dashboard":
        st.markdown("## Analysis dashboard")
        st.caption("A summary of results from this app session.")
        _render_model_status()
        _dashboard()
        return
    if page == "Model comparison":
        _model_comparison()
        return
    if page == "Database history":
        _database_history()
        return

    model_options = [name for name in MODELS if MODELS[name]["kind"] != "gan_hybrid"]
    if page == "Image analysis":
        model_options = [name for name in model_options if MODELS[name]["kind"] != "segmentation"]
    default_model = "YOLO12n + ViT-B/16"
    model_name = st.selectbox(
        "Algorithm", model_options,
        index=model_options.index(default_model),
        key="algorithm_selector_v2",
    )
    selected_config = MODELS[model_name]
    st.caption(selected_config["description"])
    _render_model_status()
    with st.expander("Optional GPS location (enter only if known)"):
        location_col1, location_col2 = st.columns(2)
        latitude_text = location_col1.text_input("Latitude", key="analysis_latitude", placeholder="e.g. 37.7749")
        longitude_text = location_col2.text_input("Longitude", key="analysis_longitude", placeholder="e.g. -122.4194")
        st.caption("These coordinates are saved as user supplied. Leave both blank if the location is unknown.")
    with st.expander("Model checkpoint details"):
        for key, path, exists in _checkpoint_status(selected_config):
            label = {"detector": "YOLO detector", "classifier": "Classifier/segmenter", "augmentation_checkpoint": "GAN generator"}.get(key, key)
            st.caption(f"{label}: {'Available' if exists else 'Unavailable'} · {path.relative_to(ROOT)}")
    if selected_config["kind"] == "segmentation":
        st.warning("Experimental binary segmentation: masks can estimate area coverage, but they do not classify damage types.")
    if selected_config["kind"] == "gan_hybrid" and not _gan_classifier_is_ready(
        _checkpoint_path(selected_config["classifier"])
    ):
        st.markdown('<div class="status-warning"><b>GAN pipeline unavailable.</b> The augmented EfficientNet checkpoint has not been trained and verified, so this pipeline cannot run inference.</div>', unsafe_allow_html=True)
    confidence = 0.10
    if selected_config["kind"] != "segmentation":
        default_confidence = 0.075 if selected_config.get("validation_evidence") else 0.10
        confidence = st.slider("Detection confidence", 0.01, 0.90, default_confidence, 0.005, key=f"confidence_{model_name}", help="Higher values reduce weak boxes. Very weak D40 candidates also need a strong D40 crop score and a box at least 18 pixels wide and high.")

    if page == "Image analysis":
        st.markdown("## Analyze images")
        st.write("Select one or more image files from your device. Annotated images and a detection table appear after analysis.")
        uploads = st.file_uploader("Choose road images", type=IMAGE_EXTENSIONS, accept_multiple_files=True, key="image_upload")
        if st.button("Analyze images", type="primary", disabled=not uploads, width="content"):
            st.session_state.pop("last_images", None)
            st.session_state.pop("last_rows", None)
            try:
                location = _parse_user_location(latitude_text, longitude_text)
                with st.spinner("Running selected model…"):
                    images, rows = _run_images(uploads, model_name, confidence)
                st.session_state.last_images = images
                st.session_state.last_rows = rows
                st.session_state.last_model = model_name
                _save_run(
                    model_name, "Image", rows, [item.name for item in uploads], images,
                    location=location,
                )
            except Exception as exc:
                st.error(f"Image analysis failed: {type(exc).__name__}: {exc}")
                st.exception(exc)
        if st.session_state.get("last_images"):
            st.markdown("### Image results")
            for result in st.session_state.last_images:
                st.markdown(f"#### {result['name']}")
                left, right = st.columns(2)
                with left:
                    st.image(result["original"], caption="Original", width="stretch")
                with right:
                    st.image(result["image"], caption=f"Annotated · {result['detections']} detections", width="stretch")
                result_config = MODELS[st.session_state.last_model]
                checkpoint_names = [result_config[key] for key in ("detector", "classifier") if result_config.get(key)]
                st.caption(f"Algorithm: {st.session_state.last_model} · Checkpoint(s): {'; '.join(checkpoint_names)} · Inference: {result['inference_seconds']:.3f} s")
                st.download_button(f"Download {Path(result['name']).stem} result", result["image"], f"{Path(result['name']).stem}_annotated.jpg", "image/jpeg", key=f"download_{result['name']}")
            df = pd.DataFrame(st.session_state.get("last_rows", []))
            st.markdown("### Detection details")
            st.dataframe(df, width="stretch", hide_index=True)
            st.download_button("Download detections CSV", df.to_csv(index=False).encode("utf-8"), "image_detections.csv", "text/csv", key="image_csv")
            st.info("No road damage detected. The annotated image is unchanged; try a lower confidence threshold or another model." if df.empty else f"{len(df)} road damage detection(s) across {len(st.session_state.last_images)} image(s).")
    else:
        st.markdown("## Analyze a video")
        st.write("Upload a road video. The app analyzes frames at your chosen interval and returns a video with detections drawn on sampled frames.")
        video = st.file_uploader("Choose a road video", type=VIDEO_EXTENSIONS, accept_multiple_files=False, key="video_upload")
        if selected_config["kind"] == "segmentation":
            sample_seconds = 0.25
            st.caption("The CNN + Spatial LSTM uses eight frames from each two-second window, matching the project video demo preprocessing.")
        else:
            sample_seconds = st.select_slider("Analyze one frame every", options=[0.5, 1, 2, 3, 5], value=1, format_func=lambda x: f"{x:g} second{'s' if x != 1 else ''}")
        if video is not None:
            st.video(video.getvalue())
        if st.button("Analyze video", type="primary", disabled=video is None):
            st.session_state.pop("last_video", None)
            st.session_state.pop("last_video_rows", None)
            try:
                location = _parse_user_location(latitude_text, longitude_text)
                processing_started = time.perf_counter()
                with st.spinner("Processing video frames…"):
                    video_bytes, rows, frame_count, duration = _run_video(video, model_name, confidence, sample_seconds)
                processing_seconds = round(time.perf_counter() - processing_started, 3)
                st.session_state.last_video = video_bytes
                st.session_state.last_video_name = video.name
                st.session_state.last_video_rows = rows
                _save_run(
                    model_name, "Video", rows, [video.name],
                    [{
                        "name": video.name, "video": video_bytes,
                        "processing_seconds": processing_seconds,
                        "processed_frames": frame_count,
                        "input_duration_seconds": duration,
                    }],
                    location=location,
                )
                fps = frame_count / processing_seconds if processing_seconds else 0
                st.success(f"Analyzed {frame_count} frames across {duration:g} seconds in {processing_seconds:.2f} seconds ({fps:.2f} processed frames/second).")
            except Exception as exc:
                st.error(f"Video analysis failed: {type(exc).__name__}: {exc}")
                st.exception(exc)
        if st.session_state.get("last_video"):
            st.markdown("### Annotated video")
            st.video(st.session_state.last_video)
            st.download_button("Download annotated video", st.session_state.last_video, f"{Path(st.session_state.last_video_name).stem}_annotated.mp4", "video/mp4")
            df = pd.DataFrame(st.session_state.get("last_video_rows", []))
            st.markdown("### Detection details")
            st.dataframe(df, width="stretch", hide_index=True)
            st.download_button("Download detections CSV", df.to_csv(index=False).encode("utf-8"), "video_detections.csv", "text/csv", key="video_csv")
            st.info("No road damage detected in sampled frames." if df.empty else f"{len(df)} road damage detection(s) across sampled video frames.")


if __name__ == "__main__":
    main()
