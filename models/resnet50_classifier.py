r"""
====================================================================================================
ALGORITHM 2: RESNET-50 DEEP RESIDUAL CLASSIFIER (STANDALONE & STAGE-2 HYBRID VERIFIER)
====================================================================================================
Module: models/resnet50_classifier.py
Class: ResNet50ClassifierModel

Architecture Description:
  - 50-Layer Deep Residual Network with Bottleneck Residual Blocks.
  - Mathematical formulation of Residual Learning:
        $\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$
    where $\mathbf{x}$ and $\mathbf{y}$ are the input and output vectors of the layers considered,
    and $\mathcal{F}(\mathbf{x}, \{W_i\})$ represents the residual mapping to be learned.
  - Skip/shortcut connections bypass intermediate convolutional layers, mitigating the vanishing
    gradient problem and enabling deeper representations of crack topology and pothole depth.
  - Backbone: 7x7 Conv + MaxPool -> Layer 1 (3 blocks) -> Layer 2 (4 blocks) -> Layer 3 (6 blocks) -> Layer 4 (3 blocks)
  - Latent Representation: 2,048-dimensional global average pooled feature vector.
  - Classification Head: Linear(2048 -> num_classes)

Hybrid Role:
  - In `HybridYOLOClassifierModel`: Serves as the Stage-2 Crop Classifier. Once YOLO localizes
    the candidate bounding box, the RoI patch is extracted, enhanced via CLAHE, and passed into
    `ResNet50ClassifierModel` to verify or reject low-confidence distress proposals.
  - In `SMOTEAugmenter`: The 2,048-dimensional latent vector before `self.fc` is extracted and
    interpolated in feature space to synthesize representations of minority classes (D01, D11).
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


class ResNet50ClassifierModel(BaseRoadDamageModel):
    """
    Standardized Standalone & Hybrid-Compatible ResNet-50 Classifier for Road Distress.
    """

    def __init__(
        self,
        num_classes: int = 10,
        class_names: Optional[List[str]] = None,
        pretrained: bool = False,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(
            model_name="ResNet50",
            num_classes=num_classes,
            class_names=class_names,
            device=device,
        )

        # ------------------------------------------------------------------------------------------
        # PART A: BASE RESNET-50 BACKBONE INITIALIZATION
        # Variable: self.backbone
        # ------------------------------------------------------------------------------------------
        try:
            if pretrained:
                weights = models.ResNet50_Weights.DEFAULT
                self.backbone = models.resnet50(weights=weights)
            else:
                self.backbone = models.resnet50(weights=None)
        except Exception:
            self.backbone = models.resnet50(weights=None)

        # ------------------------------------------------------------------------------------------
        # PART B: REPLACING THE FINAL FULLY CONNECTED LAYER (fc)
        # Variable: in_features = 2048
        # Replaces ImageNet's 1,000 classes with road damage distress categories (D00-D50).
        # ------------------------------------------------------------------------------------------
        in_features = self.backbone.fc.in_features  # 2048
        self.feature_dim = in_features
        self.backbone.fc = nn.Linear(in_features, self.num_classes)

        # ------------------------------------------------------------------------------------------
        # PART C: PREPROCESSING & NORMALIZATION
        # Variable: self.transform
        # Resizes RoI crop to 224x224 and normalizes with ImageNet statistics (RGB channels).
        # ------------------------------------------------------------------------------------------
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Load weights if provided, or search default project checkpoint
        default_weights = PROJECT_ROOT / "results" / "resnet50_road_damage" / "best_resnet50.pt"
        target_weights = Path(weights_path) if weights_path else default_weights
        if target_weights.is_file():
            self.load_checkpoint(target_weights, strict=False)

        self.to(self.device)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the 2,048-dimensional latent feature vector prior to the final linear layer.
        Used by SMOTE and hybrid latent feature fusion.
        
        Calculation:
          1. Conv1 + BN + ReLU + MaxPool: (B, 3, 224, 224) -> (B, 64, 56, 56)
          2. Layer 1 (Residual Bottlenecks): -> (B, 256, 56, 56)
          3. Layer 2 (Residual Bottlenecks): -> (B, 512, 28, 28)
          4. Layer 3 (Residual Bottlenecks): -> (B, 1024, 14, 14)
          5. Layer 4 (Residual Bottlenecks): -> (B, 2048, 7, 7)
          6. AvgPool: -> (B, 2048, 1, 1) -> Flatten -> (B, 2048)
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        feature_vector = torch.flatten(x, 1)  # Shape: (B, 2048)
        return feature_vector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates forward logits through ResNet-50.
        Calculation: logits = self.backbone(x)
        """
        return self.backbone(x)

    def predict(self, input_data: Union[str, Path, np.ndarray, Image.Image, torch.Tensor]) -> Dict[str, Any]:
        """
        Executes prediction with softmax probabilities for an input road distress image/crop.
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
    print("ALGORITHM 2: RESNET-50 - STANDALONE EXECUTION & TEST")
    print(f"{'=' * 80}")

    model = ResNet50ClassifierModel()
    print(f"Instantiated: {model.model_name} on {model.device}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

    dummy_input = torch.randn(2, 3, 224, 224).to(model.device)
    dummy_output = model(dummy_input)
    print(f"Forward pass output shape: {dummy_output.shape} (Expected: [2, 10])")

    features = model.extract_features(dummy_input)
    print(f"Extracted feature embedding shape: {features.shape} (Expected: [2, 2048])")

    sample_img = PROJECT_ROOT / "dataset" / "subsets" / "multicountry" / "RDD2022" / "split" / "val" / "images" / "China_0010.jpg"
    if sample_img.is_file():
        out_plot = PROJECT_ROOT / "results" / "diagnostics" / "algo_02_resnet50_prediction.png"
        model.plot_output(sample_img, output_save_path=out_plot)
        print(f"Plot saved successfully to: {out_plot}")
    print(f"{'=' * 80}\n")
