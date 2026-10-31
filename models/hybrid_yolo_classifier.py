r"""
====================================================================================================
ALGORITHM 9: HYBRID YOLO + DEEP CLASSIFIER (TWO-STAGE LOCALIZATION & VERIFICATION)
====================================================================================================
Module: models/hybrid_yolo_classifier.py
Class: HybridYOLOClassifierModel

Architecture & Hybrid Integration Description:
  This class joins two independent algorithm classes into a cohesive two-stage computer vision system:
  
  STAGE 1: Object Localization (from YOLO12DetectorModel)
    - Component Used: `self.detector = YOLO12DetectorModel()`
    - Output Extracted: Bounding box coordinates $[x_1, y_1, x_2, y_2]$, detector confidence $s_{det}$,
      and preliminary class label $c_{det}$.
    - Role: Scans the raw high-resolution road frame and proposes candidate distress regions.

  INTER-STAGE HANDOFF: RoI Extraction & Photometric LAB CLAHE Contrast Enhancement
    - The raw image patch $I[y_1:y_2, x_1:x_2]$ is cropped.
    - Converted to LAB color space; CLAHE (clip_limit=2.0, tile_grid=8x8) is applied exclusively
      to the Luminance (L) channel to enhance asphalt crack edges and pothole depressions without color distortion.

  STAGE 2: Fine-Grained Distress Classification (from ResNet-50 / EfficientNet-B0 / ViT-B/16)
    - Component Used: `self.classifier = BaseRoadDamageModel()` (EfficientNet, ResNet, or ViT)
    - Input Ingested: The preprocessed, CLAHE-enhanced RoI crop resized to (3, 224, 224).
    - Output Calculated: Exact posterior class probabilities $P(c | crop)$ and verified class $c_{cls}$.

  FUSION & VERIFICATION DECISION LOGIC:
    1. Agreement Check: Verifies whether $c_{det} == c_{cls}$.
    2. Area Ratio Guard: Calculates box area ratio $A_{ratio} = \frac{(x_2 - x_1)(y_2 - y_1)}{W \times H}$.
       If $A_{ratio} > 0.12$ (broad candidate) and classifier disagrees or confidence $< 0.75$, the false
       broad proposal is suppressed.
    3. Fused Confidence Calculation:
           $S_{fused} = \alpha \cdot s_{det} + (1 - \alpha) \cdot s_{cls} \quad (\text{default } \alpha = 0.4)$
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.base_model import BaseRoadDamageModel
from models.efficientnet_classifier import EfficientNetB0ClassifierModel
from models.resnet50_classifier import ResNet50ClassifierModel
from models.vit_classifier import VisionTransformerClassifierModel
from models.yolo_detector import YOLO12DetectorModel


class HybridYOLOClassifierModel(BaseRoadDamageModel):
    """
    Two-Stage Hybrid Architecture combining YOLO12 Detection with Deep Feature Classification.
    """

    def __init__(
        self,
        detector: Optional[YOLO12DetectorModel] = None,
        classifier: Optional[BaseRoadDamageModel] = None,
        classifier_type: str = "vit",  # 'vit', 'efficientnet', or 'resnet'
        alpha_fusion: float = 0.4,
        max_unverified_area_ratio: float = 0.12,
        min_classifier_confidence: float = 0.60,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(
            model_name=f"HybridYOLO_{classifier_type.upper()}",
            num_classes=10,
            device=device,
        )
        self.classifier_type = classifier_type.lower()
        self.alpha_fusion = alpha_fusion
        self.max_unverified_area_ratio = max_unverified_area_ratio
        self.min_classifier_confidence = min_classifier_confidence

        # ------------------------------------------------------------------------------------------
        # PART 1: STAGE-1 YOLO DETECTOR COMPONENT
        # Variable: self.detector
        # Class Used: YOLO12DetectorModel
        # Extracted variables: xyxy bounding box coordinates and detector confidence scores
        # ------------------------------------------------------------------------------------------
        if detector is not None:
            self.detector = detector
        else:
            self.detector = YOLO12DetectorModel(device=self.device)

        # ------------------------------------------------------------------------------------------
        # PART 2: STAGE-2 DEEP CROP CLASSIFIER COMPONENT
        # Variable: self.classifier
        # Class Used: VisionTransformerClassifierModel / EfficientNetB0ClassifierModel / ResNet50ClassifierModel
        # Ingests: RoI crop extracted from YOLO bounding box coordinates
        # ------------------------------------------------------------------------------------------
        if classifier is not None:
            self.classifier = classifier
        else:
            if self.classifier_type == "vit":
                self.classifier = VisionTransformerClassifierModel(device=self.device)
            elif self.classifier_type == "resnet":
                self.classifier = ResNet50ClassifierModel(device=self.device)
            else:
                self.classifier = EfficientNetB0ClassifierModel(device=self.device)

        # Synchronize target class names
        self.class_names = self.classifier.class_names
        self.num_classes = len(self.class_names)

        # Setup LAB CLAHE processor for intermediate RoI hand-off
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _apply_clahe_to_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Intermediate processing step: Enhances luminance contrast in asphalt distress patches.
        Calculation:
          1. BGR -> LAB color space
          2. Split L (Luminance), A, B channels
          3. Apply CLAHE to L channel: L_enhanced = CLAHE(L)
          4. Merge (L_enhanced, A, B) -> Convert back to BGR
        """
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        enhanced_l = self.clahe.apply(l_chan)
        merged = cv2.merge((enhanced_l, a_chan, b_chan))
        enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return enhanced_bgr

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass delegates to Stage-2 classifier directly."""
        return self.classifier(x)

    def predict(
        self,
        input_data: Union[str, Path, np.ndarray, Image.Image],
        detection_threshold: float = 0.10,
    ) -> Dict[str, Any]:
        """
        Executes complete Two-Stage Hybrid Inference:
          1. YOLO12 detects all candidate bounding boxes.
          2. Each box is cropped and enhanced via LAB CLAHE.
          3. Classifier verifies class label and computes fine-grained confidence.
          4. Fusion rule filters out false-positive proposals.
        """
        # Load raw image as BGR numpy array
        if isinstance(input_data, (str, Path)):
            img_bgr = cv2.imread(str(input_data))
        elif isinstance(input_data, Image.Image):
            img_bgr = cv2.cvtColor(np.array(input_data.convert("RGB")), cv2.COLOR_BGR2BGR)
        elif isinstance(input_data, np.ndarray):
            img_bgr = input_data.copy()
        else:
            raise TypeError("Unsupported image format for hybrid prediction.")

        img_h, img_w = img_bgr.shape[:2]
        total_img_area = float(img_h * img_w)

        # ------------------------------------------------------------------------------------------
        # STEP 1: EXECUTE STAGE-1 YOLO LOCALIZATION
        # ------------------------------------------------------------------------------------------
        raw_detections = self.detector.predict(img_bgr, conf_threshold=detection_threshold)

        verified_boxes: List[List[float]] = []
        verified_confidences: List[float] = []
        verified_classes: List[str] = []
        detector_classes: List[str] = []
        classifier_classes: List[str] = []
        agreement_flags: List[bool] = []

        # ------------------------------------------------------------------------------------------
        # STEP 2: LOOP OVER LOCALIZED PROPOSALS & PERFORM VERIFICATION
        # ------------------------------------------------------------------------------------------
        for box, s_det, c_det in zip(
            raw_detections["boxes"],
            raw_detections["confidences"],
            raw_detections["class_names"]
        ):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            box_w, box_h = x2 - x1, y2 - y1
            if box_w < 8 or box_h < 8:
                continue

            # Calculate box area ratio relative to full image
            box_area = float(box_w * box_h)
            area_ratio = box_area / total_img_area

            # Extract RoI crop and apply LAB CLAHE
            roi_crop = img_bgr[y1:y2, x1:x2]
            enhanced_roi = self._apply_clahe_to_crop(roi_crop)

            # Stage-2 Classification on enhanced patch
            cls_result = self.classifier.predict(enhanced_roi)
            c_cls = cls_result["predicted_class"]
            s_cls = cls_result["confidence"]

            classes_agree = (c_det == c_cls)

            # Verification rule for broad boxes: suppress false broad proposals
            if area_ratio > self.max_unverified_area_ratio and not classes_agree and s_cls < 0.75:
                continue

            # Calculate fused confidence score: S_fused = alpha * s_det + (1 - alpha) * s_cls
            s_fused = self.alpha_fusion * s_det + (1.0 - self.alpha_fusion) * s_cls

            # Final decision: Trust classifier class if confident, otherwise retain detector class
            final_class = c_cls if s_cls >= self.min_classifier_confidence else c_det

            verified_boxes.append([x1, y1, x2, y2])
            verified_confidences.append(float(s_fused))
            verified_classes.append(final_class)
            detector_classes.append(c_det)
            classifier_classes.append(c_cls)
            agreement_flags.append(classes_agree)

        primary_class = verified_classes[0] if verified_classes else "No_Damage"
        max_conf = max(verified_confidences) if verified_confidences else 0.0

        return {
            "model_name": self.model_name,
            "predicted_class": primary_class,
            "confidence": max_conf,
            "boxes": verified_boxes,
            "confidences": verified_confidences,
            "class_names": verified_classes,
            "detector_classes": detector_classes,
            "classifier_classes": classifier_classes,
            "agreements": agreement_flags,
            "detections_count": len(verified_boxes),
            "metadata": {
                "detector_model": self.detector.model_name,
                "classifier_model": self.classifier.model_name,
                "alpha_fusion": self.alpha_fusion,
            }
        }

    def plot_output(
        self,
        input_data: Union[str, Path, np.ndarray, Image.Image],
        output_save_path: Optional[Union[str, Path]] = None,
        show: bool = False
    ) -> np.ndarray:
        """
        Direct plotting method: Renders hybrid detections showing both Stage-1 and Stage-2 labels.
        """
        if isinstance(input_data, (str, Path)):
            img = cv2.imread(str(input_data))
        elif isinstance(input_data, Image.Image):
            img = cv2.cvtColor(np.array(input_data.convert("RGB")), cv2.COLOR_BGR2RGB)
        elif isinstance(input_data, np.ndarray):
            img = input_data.copy()
        else:
            raise TypeError("Unsupported format for plotting")

        preds = self.predict(img)
        annotated = img.copy()

        for box, conf, final_c, det_c, cls_c, agree in zip(
            preds["boxes"],
            preds["confidences"],
            preds["class_names"],
            preds["detector_classes"],
            preds["classifier_classes"],
            preds["agreements"]
        ):
            x1, y1, x2, y2 = [int(v) for v in box]
            box_color = (0, 255, 0) if agree else (0, 165, 255)  # Green if both agree, Orange if refined

            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            label = f"{final_c} ({conf:.2f}) [YOLO:{det_c}|{self.classifier_type.upper()}:{cls_c}]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), box_color, -1)
            cv2.putText(
                annotated, label, (x1 + 2, max(th, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA
            )

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.imshow(annotated_rgb)
        ax.set_title(
            f"Algorithm 9: Two-Stage Hybrid (YOLO12n + {self.classifier.model_name})\n"
            f"Total Verified Detections: {preds['detections_count']} | Agreement: {sum(preds['agreements'])}/{len(preds['agreements'])}",
            fontsize=12, fontweight="bold", pad=12
        )
        ax.axis("off")
        plt.tight_layout()

        if output_save_path:
            save_p = Path(output_save_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_p), dpi=150, bbox_inches="tight")
            print(f"[{self.model_name}] Hybrid detection plot saved to: {save_p}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return annotated_rgb


# ==================================================================================================
# STANDALONE EXECUTION & DIRECT PLOTTING COMMAND
# ==================================================================================================
if __name__ == "__main__":
    print(f"\n{'=' * 80}")
    print("ALGORITHM 9: HYBRID YOLO + CLASSIFIER - STANDALONE EXECUTION & TEST")
    print(f"{'=' * 80}")

    hybrid_model = HybridYOLOClassifierModel(classifier_type="vit")
    print(f"Instantiated: {hybrid_model.model_name} on {hybrid_model.device}")
    print(f"Detector Component:   {hybrid_model.detector.model_name}")
    print(f"Classifier Component: {hybrid_model.classifier.model_name}")

    sample_img = PROJECT_ROOT / "dataset" / "subsets" / "multicountry" / "RDD2022" / "split" / "val" / "images" / "China_0010.jpg"
    if sample_img.is_file():
        out_plot = PROJECT_ROOT / "results" / "diagnostics" / "algo_09_hybrid_yolo_vit_prediction.png"
        hybrid_model.plot_output(sample_img, output_save_path=out_plot)
        print(f"Plot saved successfully to: {out_plot}")
    print(f"{'=' * 80}\n")
