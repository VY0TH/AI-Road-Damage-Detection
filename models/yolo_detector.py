r"""
====================================================================================================
ALGORITHM 5: YOLO12 OBJECT DETECTOR (STANDALONE LOCALIZER & STAGE-1 PROPOSAL GENERATOR)
====================================================================================================
Module: models/yolo_detector.py
Class: YOLO12DetectorModel

Architecture Description:
  - Real-time single-stage object detector using YOLO12n (nano) architecture.
  - Features:
      * CSPDarknet-style backbone with multi-scale feature pyramids (P3, P4, P5).
      * Path Aggregation Network (PANet) fusing low-level edge details with high-level semantic context.
      * Decoupled detection head separating bounding-box regression from multi-class categorization.
  - Predicts bounding box tuples: $(\hat{x}_1, \hat{y}_1, \hat{x}_2, \hat{y}_2, \hat{s}_{conf}, \hat{c}_{class})$.

Hybrid Role:
  - Serves as **Stage 1** in `HybridYOLOClassifierModel`.
  - Scans full-resolution road frames (e.g., 1920x1080 or 640x640), generating candidate distress regions.
  - Provides bounding box coordinates $[x_1, y_1, x_2, y_2]$ used to crop damage patches from the raw image.
  - These crops are then enhanced with LAB CLAHE and passed to Stage 2 (EfficientNet / ResNet / ViT).
====================================================================================================
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.base_model import BaseRoadDamageModel


class YOLO12DetectorModel(BaseRoadDamageModel):
    """
    Standardized Standalone & Hybrid-Compatible YOLO12 Detector for Road Damage Localization.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(
            model_name="YOLO12n",
            num_classes=10,
            device=device,
        )
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Determine checkpoint path with fallback hierarchy
        default_weights = PROJECT_ROOT / "results" / "training" / "yolo12n_30_epochs" / "weights" / "best.pt"
        if not default_weights.is_file():
            default_weights = PROJECT_ROOT / "results" / "training" / "model1_yolo12_efficientnet" / "baseline_15_epochs" / "weights" / "best.pt"
        if not default_weights.is_file():
            default_weights = PROJECT_ROOT / "yolo12n.pt"

        self.weights_path = Path(weights_path) if weights_path else default_weights

        # ------------------------------------------------------------------------------------------
        # PART A: LOAD ULTRALYTICS YOLO MODEL
        # Variable: self.yolo
        # ------------------------------------------------------------------------------------------
        print(f"[{self.model_name}] Loading detector weights from: {self.weights_path}")
        self.yolo = YOLO(str(self.weights_path))

        # Synchronize class names from YOLO checkpoint metadata
        yolo_names = self.yolo.names
        if isinstance(yolo_names, dict):
            self.class_names = [yolo_names[i] for i in sorted(yolo_names.keys())]
        elif isinstance(yolo_names, list):
            self.class_names = yolo_names
        self.num_classes = len(self.class_names)

    def forward(self, x: Any) -> Any:
        """Forward pass delegated to YOLO inference engine."""
        return self.yolo(x, verbose=False)

    def predict(
        self,
        input_data: Union[str, Path, np.ndarray, Image.Image],
        conf_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes bounding box detection on an input image.
        
        Returns:
            Dict containing:
                - 'boxes': List of [x1, y1, x2, y2] pixel coordinates
                - 'confidences': List of float confidence scores
                - 'class_ids': List of integer category IDs
                - 'class_names': List of predicted class strings (e.g. 'D40', 'D00')
                - 'detections_count': Total number of detected damage regions
        """
        conf = conf_threshold if conf_threshold is not None else self.confidence_threshold
        yolo_device = 0 if self.device.type == "cuda" else "cpu"

        # If input is a path or numpy array, pass directly to YOLO
        source = str(input_data) if isinstance(input_data, (str, Path)) else np.array(input_data)
        results = self.yolo.predict(
            source=source,
            conf=conf,
            iou=self.iou_threshold,
            device=yolo_device,
            verbose=False,
        )

        boxes_list: List[List[float]] = []
        confidences_list: List[float] = []
        class_ids_list: List[int] = []
        class_names_list: List[str] = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for b in boxes:
                xyxy = [float(v) for v in b.xyxy[0].cpu().tolist()]
                score = float(b.conf[0].item())
                cid = int(b.cls[0].item())
                cname = self.class_names[cid] if cid < len(self.class_names) else f"Class_{cid}"

                boxes_list.append(xyxy)
                confidences_list.append(score)
                class_ids_list.append(cid)
                class_names_list.append(cname)

        primary_class = class_names_list[0] if class_names_list else "No_Damage"
        max_conf = max(confidences_list) if confidences_list else 0.0

        return {
            "model_name": self.model_name,
            "predicted_class": primary_class,
            "confidence": max_conf,
            "boxes": boxes_list,
            "confidences": confidences_list,
            "class_ids": class_ids_list,
            "class_names": class_names_list,
            "detections_count": len(boxes_list),
            "metadata": {
                "weights": str(self.weights_path),
                "device": str(self.device),
                "conf_threshold": conf,
            }
        }

    def plot_output(
        self,
        input_data: Union[str, Path, np.ndarray, Image.Image],
        output_save_path: Optional[Union[str, Path]] = None,
        show: bool = False
    ) -> np.ndarray:
        """
        Plots the image with overlaid bounding boxes, class labels, and confidences.
        """
        # Load image as BGR for OpenCV annotation
        if isinstance(input_data, (str, Path)):
            img = cv2.imread(str(input_data))
        elif isinstance(input_data, Image.Image):
            img = cv2.cvtColor(np.array(input_data.convert("RGB")), cv2.COLOR_RGB2BGR)
        elif isinstance(input_data, np.ndarray):
            img = input_data.copy()
        else:
            raise TypeError("Unsupported image format for plotting")

        preds = self.predict(img)
        annotated = img.copy()

        # Class color mapping (Vibrant palette for distress distinction)
        COLOR_MAP = {
            "D00": (0, 255, 0),     # Bright Green (Longitudinal Crack)
            "D01": (50, 205, 50),
            "D10": (255, 165, 0),   # Orange (Transverse Crack)
            "D11": (255, 140, 0),
            "D20": (0, 140, 255),   # Deep Orange / Red (Alligator Cracking)
            "D40": (0, 0, 255),     # Crimson Red (Pothole - High Hazard)
            "D43": (255, 0, 255),   # Magenta
            "D44": (255, 20, 147),
            "D50": (255, 255, 0),   # Cyan / Yellow
            "D0w0": (128, 0, 128),
        }

        for box, conf, cname in zip(preds["boxes"], preds["confidences"], preds["class_names"]):
            x1, y1, x2, y2 = [int(v) for v in box]
            color = COLOR_MAP.get(cname, (0, 255, 255))

            # Draw bounding box rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label banner with solid background
            label_text = f"{cname} | {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - text_h - 6)), (x1 + text_w + 4, y1), color, -1)
            cv2.putText(
                annotated, label_text, (x1 + 2, max(text_h, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )

        # Matplotlib visualization
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(annotated_rgb)
        ax.set_title(
            f"Algorithm 5: YOLO12n Object Detection\n"
            f"Total Detections: {preds['detections_count']} | Threshold: {self.confidence_threshold}",
            fontsize=12, fontweight="bold", pad=12
        )
        ax.axis("off")
        plt.tight_layout()

        if output_save_path:
            save_p = Path(output_save_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_p), dpi=150, bbox_inches="tight")
            print(f"[{self.model_name}] Detection plot saved to: {save_p}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return annotated_rgb
