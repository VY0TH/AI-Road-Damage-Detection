"""
====================================================================================================
ALGORITHM 1: BASELINE CNN (STANDALONE & HYBRID FEATURE EXTRACTOR)
====================================================================================================
Module: models/baseline_cnn.py
Class: BaselineCNNModel

Architecture Description:
  - 4-Stage Hierarchical Convolutional Neural Network
  - Stage 1: Conv2d(3 -> 32, k=3, p=1) + BatchNorm2d + ReLU + MaxPool2d(2, 2)  -> Spatial: 112x112
  - Stage 2: Conv2d(32 -> 64, k=3, p=1) + BatchNorm2d + ReLU + MaxPool2d(2, 2) -> Spatial: 56x56
  - Stage 3: Conv2d(64 -> 128, k=3, p=1) + BatchNorm2d + ReLU + MaxPool2d(2, 2)-> Spatial: 28x28
  - Stage 4: Conv2d(128 -> 256, k=3, p=1) + BatchNorm2d + ReLU + AdaptiveAvgPool2d(4, 4)
  - Classifier Head: Flatten -> Dropout(0.3) -> Linear(4096 -> 256) -> ReLU -> Linear(256 -> 10)

Hybrid Role:
  - The `self.features` convolutional backbone provides localized edge, texture, and distress
    representations that can be transferred directly into the ConvLSTM or Autoencoder encoders.
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
from torchvision import transforms

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.base_model import BaseRoadDamageModel


class BaselineCNNModel(BaseRoadDamageModel):
    """
    Standardized Standalone Baseline Convolutional Neural Network for Road Damage Classification.
    """

    def __init__(
        self,
        num_classes: int = 10,
        class_names: Optional[List[str]] = None,
        dropout_rate: float = 0.3,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(
            model_name="BaselineCNN",
            num_classes=num_classes,
            class_names=class_names,
            device=device,
        )
        self.dropout_rate = dropout_rate

        # ------------------------------------------------------------------------------------------
        # PART A: CONVOLUTIONAL FEATURE EXTRACTOR
        # Variable: self.features
        # Used in: Standalone forward pass AND as the spatial front-end for hybrid algorithms
        # Output shape for 224x224 input: (Batch, 256, 4, 4)
        # ------------------------------------------------------------------------------------------
        self.features = nn.Sequential(
            # Stage 1: Captures fine texture, asphalt grain, and micro-cracks
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Shape: (B, 32, 112, 112)

            # Stage 2: Combines edges into distress orientations (longitudinal/transverse)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Shape: (B, 64, 56, 56)

            # Stage 3: Mid-level distress topology (alligator cracking networks, pothole rims)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Shape: (B, 128, 28, 28)

            # Stage 4: High-level semantic damage representation with adaptive spatial pooling
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),           # Shape: (B, 256, 4, 4)
        )

        # ------------------------------------------------------------------------------------------
        # PART B: DENSE CLASSIFICATION HEAD
        # Variable: self.classifier
        # Calculates: Unnormalized class logits across the 10 target categories
        # Flattened dimension: 256 channels * 4 height * 4 width = 4,096 features
        # ------------------------------------------------------------------------------------------
        self.feature_dim = 256 * 4 * 4  # 4096
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_classes),
        )

        # Image preprocessing pipeline (Standard ImageNet normalization for road crops)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.to(self.device)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts spatial feature embeddings prior to classification.
        Used when combining this CNN with SMOTE or Autoencoder latent analysis.
        
        Args:
            x (torch.Tensor): Input batch of shape (B, 3, 224, 224)
            
        Returns:
            torch.Tensor: Flattened feature embeddings of shape (B, 4096)
        """
        spatial_maps = self.features(x)  # Shape: (B, 256, 4, 4)
        flattened = torch.flatten(spatial_maps, 1)  # Shape: (B, 4096)
        return flattened

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates raw class logits from image tensor.
        Calculation: logits = classifier(features(x))
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits

    def predict(self, input_data: Union[str, Path, np.ndarray, Image.Image, torch.Tensor]) -> Dict[str, Any]:
        """
        Executes prediction with softmax probabilities for an input image.
        """
        self.eval()

        # Handle diverse input formats
        if isinstance(input_data, (str, Path)):
            pil_img = Image.open(str(input_data)).convert("RGB")
            tensor_input = self.transform(pil_img).unsqueeze(0).to(self.device)
        elif isinstance(input_data, np.ndarray):
            # Convert BGR (OpenCV) to RGB
            if input_data.ndim == 3 and input_data.shape[2] == 3:
                rgb = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
            else:
                pil_img = Image.fromarray(input_data)
            tensor_input = self.transform(pil_img).unsqueeze(0).to(self.device)
        elif isinstance(input_data, Image.Image):
            tensor_input = self.transform(input_data.convert("RGB")).unsqueeze(0).to(self.device)
        elif isinstance(input_data, torch.Tensor):
            if input_data.ndim == 3:
                tensor_input = input_data.unsqueeze(0).to(self.device)
            else:
                tensor_input = input_data.to(self.device)
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
                "feature_dimension": self.feature_dim,
            }
        }


# ==================================================================================================
# STANDALONE EXECUTION & DIRECT PLOTTING COMMAND
# ==================================================================================================
if __name__ == "__main__":
    print(f"\n{'=' * 80}")
    print("ALGORITHM 1: BASELINE CNN - STANDALONE EXECUTION & TEST")
    print(f"{'=' * 80}")

    model = BaselineCNNModel()
    print(f"Instantiated: {model.model_name} on {model.device}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass on a dummy tensor
    dummy_input = torch.randn(2, 3, 224, 224).to(model.device)
    dummy_output = model(dummy_input)
    print(f"Forward pass output shape: {dummy_output.shape} (Expected: [2, 10])")

    # Test direct plotting if sample image exists
    sample_img = PROJECT_ROOT / "dataset" / "subsets" / "multicountry" / "RDD2022" / "split" / "val" / "images" / "China_0010.jpg"
    if sample_img.is_file():
        out_plot = PROJECT_ROOT / "results" / "diagnostics" / "algo_01_baseline_cnn_prediction.png"
        model.plot_output(sample_img, output_save_path=out_plot)
        print(f"Plot saved successfully to: {out_plot}")
    print(f"{'=' * 80}\n")
