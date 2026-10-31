r"""
====================================================================================================
ALGORITHM 3: EFFICIENTNET-B0 COMPOUND-SCALED CLASSIFIER (STANDALONE & HYBRID STAGE-2)
====================================================================================================
Module: models/efficientnet_classifier.py
Class: EfficientNetB0ClassifierModel

Architecture Description:
  - EfficientNet-B0 relies on compound scaling across depth, width, and resolution:
        $\text{depth: } d = \alpha^\phi, \quad \text{width: } w = \beta^\phi, \quad \text{resolution: } r = \gamma^\phi$
    constrained by $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ such that total FLOPS increase by $2^\phi$.
  - Core block: Mobile Inverted Bottleneck Convolution (MBConv) with depthwise separable convolutions
    and Squeeze-and-Excitation (SE) channel attention:
        $\mathbf{s} = \sigma(W_2 \delta(W_1 \mathbf{z}))$
    where channel weights $\mathbf{s}$ recalibrate feature maps to highlight crack boundaries over uniform asphalt.
  - Latent Representation: 1,280-dimensional feature embedding from `AdaptiveAvgPool2d(1, 1)`.
  - Head: Dropout(0.2) -> Linear(1280 -> 10).

Hybrid Role:
  - Primary Stage-2 Crop Classifier in `HybridYOLOClassifierModel` and `yolo_gan_efficientnet_combined.py`.
  - EfficientNet receives the bounding-box crops localized by YOLO, which are enhanced via LAB CLAHE,
    and calculates fine-grained damage probabilities, preventing false positives from shadow edges.
====================================================================================================
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.base_model import BaseRoadDamageModel


class EfficientNetB0ClassifierModel(BaseRoadDamageModel):
    """
    Standardized Standalone & Hybrid-Compatible EfficientNet-B0 Classifier for Road Damage Crops.
    """

    def __init__(
        self,
        num_classes: int = 10,
        class_names: Optional[List[str]] = None,
        dropout_rate: float = 0.2,
        pretrained: bool = False,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(
            model_name="EfficientNetB0",
            num_classes=num_classes,
            class_names=class_names,
            device=device,
        )
        self.dropout_rate = dropout_rate

        # ------------------------------------------------------------------------------------------
        # PART A: EFFICIENTNET-B0 BACKBONE (MBConv BLOCKS)
        # Variable: self.backbone
        # ------------------------------------------------------------------------------------------
        try:
            if pretrained:
                weights = models.EfficientNet_B0_Weights.DEFAULT
                self.backbone = models.efficientnet_b0(weights=weights)
            else:
                self.backbone = models.efficientnet_b0(weights=None)
        except Exception:
            self.backbone = models.efficientnet_b0(weights=None)

        # ------------------------------------------------------------------------------------------
        # PART B: ADAPTING THE CLASSIFIER HEAD
        # Variable: in_features = 1280
        # Replaces final linear classification head with road distress output dimension.
        # ------------------------------------------------------------------------------------------
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.feature_dim = in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate, inplace=True),
            nn.Linear(in_features, self.num_classes)
        )

        # Preprocessing transforms (ImageNet mean & std normalization)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Load weights if present
        default_weights = PROJECT_ROOT / "results" / "efficientnet_b0" / "best_efficientnet_b0.pt"
        target_weights = Path(weights_path) if weights_path else default_weights
        if target_weights.is_file():
            self.load_checkpoint(target_weights, strict=False)

        self.to(self.device)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the 1,280-dimensional latent feature vector prior to the final classifier.
        Used for SMOTE latent interpolation and hybrid multimodal representations.
        
        Calculation:
          1. MBConv stages 0-8: (B, 3, 224, 224) -> (B, 1280, 7, 7)
          2. AdaptiveAvgPool2d: (B, 1280, 7, 7) -> (B, 1280, 1, 1)
          3. Flatten: (B, 1280, 1, 1) -> (B, 1280)
        """
        features_map = self.backbone.features(x)
        pooled = self.backbone.avgpool(features_map)
        feature_vector = torch.flatten(pooled, 1)  # Shape: (B, 1280)
        return feature_vector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation through EfficientNet-B0.
        Calculation: logits = self.backbone(x)
        """
        return self.backbone(x)

    def predict(self, input_data: Union[str, Path, np.ndarray, Image.Image, torch.Tensor]) -> Dict[str, Any]:
        """
        Executes fine-grained road distress prediction on an image or localized RoI patch.
        """
        self.eval()

        if isinstance(input_data, (str, Path)):
            pil_img = Image.open(str(input_data)).convert("RGB")
            tensor_input = self.transform(pil_img).unsqueeze(0).to(self.device)
        elif isinstance(input_data, np.ndarray):
            if input_data.ndim == 3 and input_data.shape[2] == 3:
                rgb = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
            else:
                pil_img = Image.fromarray(input_data)
            tensor_input = self.transform(pil_img).unsqueeze(0).to(self.device)
        elif isinstance(input_data, Image.Image):
            tensor_input = self.transform(input_data.convert("RGB")).unsqueeze(0).to(self.device)
        elif isinstance(input_data, torch.Tensor):
            tensor_input = input_data.unsqueeze(0).to(self.device) if input_data.ndim == 3 else input_data.to(self.device)
        else:
            raise TypeError(f"Unsupported input type for prediction: {type(input_data)}")

        with torch.no_grad():
            logits = self.forward(tensor_input)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probabilities))
        pred_class = self.class_names[pred_idx]
        confidence = float(probabilities[pred_idx])

        return {
            "model_name": self.model_name,
            "predicted_class": pred_class,
            "predicted_index": pred_idx,
            "confidence": confidence,
            "raw_probabilities": probabilities,
            "metadata": {
                "device": str(self.device),
                "latent_features_dim": self.feature_dim,
            }
        }


# ==================================================================================================
# STANDALONE EXECUTION & DIRECT PLOTTING COMMAND
# ==================================================================================================
if __name__ == "__main__":
    print(f"\n{'=' * 80}")
    print("ALGORITHM 3: EFFICIENTNET-B0 - STANDALONE EXECUTION & TEST")
    print(f"{'=' * 80}")

    model = EfficientNetB0ClassifierModel()
    print(f"Instantiated: {model.model_name} on {model.device}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

    dummy_input = torch.randn(2, 3, 224, 224).to(model.device)
    dummy_output = model(dummy_input)
    print(f"Forward pass output shape: {dummy_output.shape} (Expected: [2, 10])")

    features = model.extract_features(dummy_input)
    print(f"Extracted feature embedding shape: {features.shape} (Expected: [2, 1280])")

    sample_img = PROJECT_ROOT / "dataset" / "subsets" / "multicountry" / "RDD2022" / "split" / "val" / "images" / "China_0010.jpg"
    if sample_img.is_file():
        out_plot = PROJECT_ROOT / "results" / "diagnostics" / "algo_03_efficientnet_prediction.png"
        model.plot_output(sample_img, output_save_path=out_plot)
        print(f"Plot saved successfully to: {out_plot}")
    print(f"{'=' * 80}\n")
