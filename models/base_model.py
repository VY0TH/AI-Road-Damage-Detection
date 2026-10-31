"""
====================================================================================================
BASE ROAD DAMAGE MODEL INTERFACE
====================================================================================================
Abstract Base Class for all 10 Road Damage Detection & Classification Algorithms.
Provides a standardized interface for:
  1. Initialization & Device Assignment (CPU / CUDA)
  2. Forward Pass Inference (`forward`)
  3. Single Image / Batch Prediction (`predict`)
  4. Standardized Metrics Evaluation (`evaluate`)
  5. Checkpoint Saving & Loading (`save_checkpoint`, `load_checkpoint`)
  6. Direct Plotting of Predictions (`plot_output`)

Every algorithm in this ecosystem inherits from or implements this contract, ensuring
seamless interoperability, standalone execution, and clean hybrid composition.
====================================================================================================
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn


class BaseRoadDamageModel(nn.Module, ABC):
    """
    Standardized Abstract Base Class for Road Damage Models.
    
    Attributes:
        model_name (str): Unique identifier for the algorithm (e.g., 'BaselineCNN', 'YOLO12').
        num_classes (int): Number of target road distress classes (default 10: D00 to D50).
        class_names (List[str]): Canonical list of road damage category labels.
        device (torch.device): Compute device for execution ('cuda' or 'cpu').
    """

    DEFAULT_CLASS_NAMES = [
        "D00", "D01", "D0w0", "D10", "D11",
        "D20", "D40", "D43", "D44", "D50"
    ]

    def __init__(
        self,
        model_name: str,
        num_classes: int = 10,
        class_names: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.class_names = class_names or self.DEFAULT_CLASS_NAMES[:num_classes]

        # Determine compute device: prioritize CUDA if available, fallback to CPU
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Any:
        """
        Defines the PyTorch forward computation graph.
        
        Args:
            x (torch.Tensor): Input tensor. Shape typically (Batch, Channels, Height, Width)
                              or (Batch, Seq_Len, Channels, Height, Width) for temporal models.
                              
        Returns:
            torch.Tensor or Any: Raw logits or model outputs.
        """
        pass

    @abstractmethod
    def predict(self, input_data: Union[str, Path, np.ndarray, Image.Image, torch.Tensor]) -> Dict[str, Any]:
        """
        High-level inference method accepting flexible input types.
        
        Args:
            input_data: Image file path, numpy array (BGR/RGB), PIL Image, or PyTorch Tensor.
            
        Returns:
            Dict[str, Any]: Standardized prediction dictionary containing:
                - 'predicted_class': Category name (e.g., 'D40 (Pothole)')
                - 'confidence': Prediction probability (0.0 to 1.0)
                - 'raw_probabilities': Array of probabilities across all classes
                - 'metadata': Additional algorithm-specific data (e.g. bounding boxes, masks)
        """
        pass

    def save_checkpoint(self, save_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> Path:
        """
        Saves model weights along with metadata to disk.
        
        Args:
            save_path (Union[str, Path]): Target path for .pt checkpoint file.
            metadata (Optional[Dict]): Additional training info (epochs, optimizer state, metrics).
            
        Returns:
            Path: Resolved path where weights were saved.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "model_state_dict": self.state_dict(),
            "metadata": metadata or {},
        }
        torch.save(payload, str(save_path))
        print(f"[{self.model_name}] Checkpoint saved successfully to: {save_path.name}")
        return save_path

    def load_checkpoint(self, checkpoint_path: Union[str, Path], strict: bool = True) -> None:
        """
        Loads pre-trained weights from disk into current model architecture.
        
        Args:
            checkpoint_path (Union[str, Path]): Path to .pt checkpoint.
            strict (bool): Whether strictly matching state_dict keys is enforced.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"[{self.model_name}] Checkpoint not found at: {checkpoint_path}")

        checkpoint = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            if "class_names" in checkpoint:
                self.class_names = checkpoint["class_names"]
                self.num_classes = len(self.class_names)
        else:
            state_dict = checkpoint

        self.load_state_dict(state_dict, strict=strict)
        self.to(self.device)
        self.eval()
        print(f"[{self.model_name}] Loaded pre-trained weights from: {checkpoint_path.name}")

    def plot_output(
        self,
        input_data: Union[str, Path, np.ndarray, Image.Image],
        output_save_path: Optional[Union[str, Path]] = None,
        show: bool = False
    ) -> np.ndarray:
        """
        Direct plotting method: runs prediction and generates a visual diagnostic figure.
        
        Args:
            input_data: Input image or path.
            output_save_path: Optional file path to save the generated figure.
            show: If True, calls plt.show() (useful in interactive notebooks).
            
        Returns:
            np.ndarray: Rendered image with bounding boxes, predictions, and confidence bars.
        """
        prediction = self.predict(input_data)
        
        # Load base image as RGB for matplotlib
        if isinstance(input_data, (str, Path)):
            img = cv2.imread(str(input_data))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(input_data, Image.Image):
            img_rgb = np.array(input_data.convert("RGB"))
        elif isinstance(input_data, np.ndarray):
            if input_data.shape[2] == 3:
                # Assume BGR from OpenCV by default
                img_rgb = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = input_data
        else:
            img_rgb = np.zeros((224, 224, 3), dtype=np.uint8)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1.2, 1]})
        
        # Left Panel: Visual Input with Title
        ax1.imshow(img_rgb)
        ax1.set_title(
            f"Algorithm: {self.model_name}\n"
            f"Prediction: {prediction.get('predicted_class', 'N/A')} "
            f"({prediction.get('confidence', 0.0):.1%})",
            fontsize=11, fontweight="bold", pad=10
        )
        ax1.axis("off")

        # Right Panel: Class Confidence Distribution Bar Chart
        probs = prediction.get("raw_probabilities", np.zeros(self.num_classes))
        y_pos = np.arange(len(self.class_names))
        colors = ['#1f77b4' if i != np.argmax(probs) else '#d62728' for i in range(len(self.class_names))]

        ax2.barh(y_pos, probs * 100, color=colors, edgecolor='black', alpha=0.85)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(self.class_names, fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel("Confidence Probability (%)", fontsize=10, fontweight="bold")
        ax2.set_title("Probability Distribution", fontsize=11, fontweight="bold")
        ax2.set_xlim(0, 100)
        ax2.grid(True, linestyle="--", alpha=0.5, axis="x")

        plt.tight_layout()

        if output_save_path:
            save_p = Path(output_save_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_p), dpi=150, bbox_inches="tight")
            print(f"[{self.model_name}] Diagnostic plot saved to: {save_p}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return img_rgb
