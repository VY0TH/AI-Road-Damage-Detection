r"""
====================================================================================================
ALGORITHM 7: CONVOLUTIONAL AUTOENCODER (STANDALONE & SPATIAL LATENT ENCODER)
====================================================================================================
Module: models/autoencoder.py
Class: RoadDamageAutoencoderModel

Architecture Description:
  - Deep Convolutional Autoencoder for Unsupervised Feature Compression & Distress Reconstruction:
    1. Encoder $\mathcal{E}_\theta$: Compresses spatial image $\mathbf{x} \in \mathbb{R}^{3 \times 112 \times 112}$
       into low-dimensional spatial bottleneck tensor $\mathbf{z} \in \mathbb{R}^{128 \times 14 \times 14}$
       or 256-dimensional global latent vector $\mathbf{z}_{flat}$.
    2. Bottleneck: Captures intrinsic asphalt surface manifold, suppressing stochastic grain noise.
    3. Decoder $\mathcal{D}_\phi$: Reconstructs the original road patch $\hat{\mathbf{x}} = \mathcal{D}_\phi(\mathbf{z})$
       via transposed convolutions: $128 \rightarrow 64 \rightarrow 32 \rightarrow 3$.
  - Reconstruction Loss:
        $\mathcal{L}_{rec}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{HW} \sum_{i=1}^H \sum_{j=1}^W \|\mathbf{x}_{i,j} - \hat{\mathbf{x}}_{i,j}\|_2^2$
  - Anomaly Scoring: Road distresses (potholes, cracks) produce sharp reconstruction residuals:
        $\mathcal{A}(i, j) = |\mathbf{x}_{i,j} - \hat{\mathbf{x}}_{i,j}| > \tau_{anomaly}$.

Hybrid Role:
  - **Shared directly inside `HybridLSTMAutoencoderModel` (Algorithm 10)**:
    * `self.encoder` is isolated and used as the spatial compressor for each frame $\mathbf{x}_t$.
    * `self.decoder` is isolated and attached to the temporal LSTM output to decode predicted states $\mathcal{H}_t$.
  - Latent vector $\mathbf{z}_{flat}$ is also suitable for feature-space SMOTE minority oversampling.
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
import torch.nn as nn
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.base_model import BaseRoadDamageModel


class RoadDamageAutoencoderModel(BaseRoadDamageModel):
    """
    Standardized Standalone & Hybrid-Compatible Convolutional Autoencoder.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(
            model_name="ConvolutionalAutoencoder",
            num_classes=latent_dim,
            device=device,
        )
        self.latent_dim = latent_dim

        # ------------------------------------------------------------------------------------------
        # PART A: CONVOLUTIONAL ENCODER
        # Variable: self.encoder
        # Used in: Standalone forward pass AND isolated inside HybridLSTMAutoencoderModel
        # Input shape: (B, 3, 112, 112) -> Output shape: (B, 128, 14, 14)
        # ------------------------------------------------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # (B, 32, 56, 56)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # (B, 64, 28, 28)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),          # (B, 128, 14, 14)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Bottleneck projections for vector representations
        self.flatten_dim = 128 * 14 * 14  # 25,088
        self.fc_encode = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        # ------------------------------------------------------------------------------------------
        # PART B: CONVOLUTIONAL DECODER
        # Variable: self.decoder
        # Used in: Standalone forward pass AND isolated inside HybridLSTMAutoencoderModel
        # Input shape: (B, 128, 14, 14) -> Output shape: (B, 3, 112, 112)
        # ------------------------------------------------------------------------------------------
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 64, 28, 28)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # (B, 32, 56, 56)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # (B, 3, 112, 112)
            nn.Sigmoid(),  # Bound output pixels to [0.0, 1.0]
        )

        self.to(self.device)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates spatial feature map and 1D latent bottleneck vector.
        
        Args:
            x (torch.Tensor): Input image batch of shape (B, 3, 112, 112)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - spatial_latent: (B, 128, 14, 14)
                - vector_latent: (B, latent_dim)
        """
        spatial_latent = self.encoder(x)
        flat = torch.flatten(spatial_latent, 1)
        vector_latent = self.fc_encode(flat)
        return spatial_latent, vector_latent

    def decode(self, spatial_or_vector: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the original road image from latent features.
        
        Args:
            spatial_or_vector: Either (B, 128, 14, 14) spatial map or (B, latent_dim) vector.
            
        Returns:
            torch.Tensor: Reconstructed RGB image tensor of shape (B, 3, 112, 112)
        """
        if spatial_or_vector.ndim == 2:
            flat = self.fc_decode(spatial_or_vector)
            spatial_map = flat.view(-1, 128, 14, 14)
        else:
            spatial_map = spatial_or_vector
        return self.decoder(spatial_map)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full Autoencoder pass: Input -> Encode -> Decode -> Reconstructed Output.
        Calculation: x_hat = decoder(encoder(x))
        """
        spatial_latent = self.encoder(x)
        x_reconstructed = self.decoder(spatial_latent)
        return x_reconstructed

    def compute_anomaly_map(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Calculates pixel-wise reconstruction residual and overall anomaly score.
        Calculation: residual = mean(|x - x_reconstructed|, channel_dim)
        """
        x_recon = self.forward(x)
        residual = torch.mean(torch.abs(x - x_recon), dim=1, keepdim=True)
        anomaly_score = float(torch.mean(residual).item())
        return residual, anomaly_score

    def predict(self, input_data: Union[str, Path, np.ndarray, Image.Image, torch.Tensor]) -> Dict[str, Any]:
        """
        Runs reconstruction inference and detects road distress anomalies.
        """
        self.eval()

        if isinstance(input_data, (str, Path)):
            raw = cv2.imread(str(input_data))
            rgb = cv2.cvtColor(cv2.resize(raw, (112, 112)), cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensor_input = tensor.unsqueeze(0).to(self.device)
        elif isinstance(input_data, np.ndarray):
            rgb = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB) if input_data.shape[2] == 3 else input_data
            resized = cv2.resize(rgb, (112, 112))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            tensor_input = tensor.unsqueeze(0).to(self.device)
        elif isinstance(input_data, Image.Image):
            resized = input_data.convert("RGB").resize((112, 112))
            tensor = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float() / 255.0
            tensor_input = tensor.unsqueeze(0).to(self.device)
        elif isinstance(input_data, torch.Tensor):
            tensor_input = input_data.unsqueeze(0).to(self.device) if input_data.ndim == 3 else input_data.to(self.device)
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

        with torch.no_grad():
            x_recon = self.forward(tensor_input)
            residual, score = self.compute_anomaly_map(tensor_input)

        recon_np = x_recon.squeeze().permute(1, 2, 0).cpu().numpy()
        residual_np = residual.squeeze().cpu().numpy()

        is_distress = score > 0.08  # Empirically calibrated threshold for asphalt distress

        return {
            "model_name": self.model_name,
            "predicted_class": "Anomaly Distress Detected" if is_distress else "Normal Road Surface",
            "confidence": min(1.0, score * 10.0),
            "reconstruction_error": score,
            "reconstructed_image": recon_np,
            "residual_anomaly_map": residual_np,
            "metadata": {
                "device": str(self.device),
                "latent_dim": self.latent_dim,
            }
        }

    def plot_output(
        self,
        input_data: Union[str, Path, np.ndarray, Image.Image],
        output_save_path: Optional[Union[str, Path]] = None,
        show: bool = False
    ) -> np.ndarray:
        """
        Direct plotting method: Original Road Image | Reconstructed Output | Anomaly Residual Map.
        """
        preds = self.predict(input_data)

        if isinstance(input_data, (str, Path)):
            raw = cv2.imread(str(input_data))
            orig_rgb = cv2.cvtColor(cv2.resize(raw, (112, 112)), cv2.COLOR_BGR2RGB)
        else:
            orig_rgb = np.zeros((112, 112, 3), dtype=np.uint8)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))
        ax1.imshow(orig_rgb)
        ax1.set_title("Input Road Patch", fontsize=11, fontweight="bold")
        ax1.axis("off")

        recon_clipped = np.clip(preds["reconstructed_image"], 0.0, 1.0)
        ax2.imshow(recon_clipped)
        ax2.set_title("Autoencoder Reconstruction", fontsize=11, fontweight="bold")
        ax2.axis("off")

        im3 = ax3.imshow(preds["residual_anomaly_map"], cmap="inferno")
        ax3.set_title(
            f"Residual Anomaly Map\nLoss: {preds['reconstruction_error']:.4f} | {preds['predicted_class']}",
            fontsize=11, fontweight="bold"
        )
        ax3.axis("off")
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

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

        return orig_rgb
