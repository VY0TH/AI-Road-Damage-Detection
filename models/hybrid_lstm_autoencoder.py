r"""
====================================================================================================
ALGORITHM 10: HYBRID LSTM + AUTOENCODER (SPATIOTEMPORAL ANOMALY & RECONSTRUCTION)
====================================================================================================
Module: models/hybrid_lstm_autoencoder.py
Class: HybridLSTMAutoencoderModel

Architecture & Hybrid Integration Description:
  This hybrid architecture integrates components from two foundational classes:
  
  PART 1: SPATIAL ENCODER (Inherited from RoadDamageAutoencoderModel)
    - Component Used: `self.spatial_encoder = autoencoder.encoder`
    - Ingests: Video frame sequence $\mathbf{X} \in \mathbb{R}^{B \times T \times 3 \times 112 \times 112}$
    - Output Calculated: Spatial feature map of shape $(B \times T, 128, 14, 14)$, projected to 1D
      latent vector $\mathbf{z}_t \in \mathbb{R}^{256}$ at each time-step $t$.
    - Role: Performs spatial compression, learning invariant representations of normal asphalt surface.

  PART 2: TEMPORAL LSTM SEQUENCE RECURRENT ENGINE
    - Component Used: `self.temporal_lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True)`
    - Ingests: Temporal sequence of latent vectors $(\mathbf{z}_1, \mathbf{z}_2, \dots, \mathbf{z}_T)$.
    - Equations of Transition:
          $\mathbf{i}_t = \sigma(\mathbf{W}_{ii} \mathbf{z}_t + \mathbf{b}_{ii} + \mathbf{W}_{hi} \mathbf{h}_{t-1} + \mathbf{b}_{hi})$
          $\mathbf{f}_t = \sigma(\mathbf{W}_{if} \mathbf{z}_t + \mathbf{b}_{if} + \mathbf{W}_{hf} \mathbf{h}_{t-1} + \mathbf{b}_{hf})$
          $\mathbf{g}_t = \tanh(\mathbf{W}_{ig} \mathbf{z}_t + \mathbf{b}_{ig} + \mathbf{W}_{hg} \mathbf{h}_{t-1} + \mathbf{b}_{hg})$
          $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \mathbf{g}_t$
          $\mathbf{o}_t = \sigma(\mathbf{W}_{io} \mathbf{z}_t + \mathbf{b}_{io} + \mathbf{W}_{ho} \mathbf{h}_{t-1} + \mathbf{b}_{ho})$
          $\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$
    - Role: Models temporal transition between consecutive road survey frames (tracking continuous road pavement).

  PART 3: SPATIAL DECODER (Inherited from RoadDamageAutoencoderModel)
    - Component Used: `self.spatial_decoder = autoencoder.decoder`
    - Ingests: The temporal hidden state $\mathbf{h}_t$ reshaped back into spatial tensor $(128, 14, 14)$.
    - Output Calculated: Reconstructed road frame $\hat{\mathbf{x}}_t \in \mathbb{R}^{3 \times 112 \times 112}$.

  PART 4: SPATIOTEMPORAL ANOMALY SCORING
    - Normal road surfaces exhibit smooth temporal transitions and low reconstruction loss.
    - Potholes, sudden longitudinal cracks, and severe distress trigger high temporal prediction error:
          $\mathcal{E}_t = \frac{1}{HW} \sum_{i,j} \|\mathbf{x}_{t,i,j} - \hat{\mathbf{x}}_{t,i,j}\|_2^2$
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

from models.autoencoder import RoadDamageAutoencoderModel
from models.base_model import BaseRoadDamageModel


class HybridLSTMAutoencoderModel(BaseRoadDamageModel):
    """
    Hybrid Spatiotemporal Architecture combining Autoencoder Spatial Compression with LSTM Temporal Modeling.
    """

    def __init__(
        self,
        autoencoder: Optional[RoadDamageAutoencoderModel] = None,
        latent_dim: int = 256,
        lstm_hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(
            model_name="HybridLSTM_Autoencoder",
            num_classes=latent_dim,
            device=device,
        )
        self.latent_dim = latent_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        # ------------------------------------------------------------------------------------------
        # PART 1: BASE AUTOENCODER MODEL
        # Variable: self.base_autoencoder
        # Source Class: RoadDamageAutoencoderModel
        # ------------------------------------------------------------------------------------------
        if autoencoder is not None:
            self.base_autoencoder = autoencoder
        else:
            self.base_autoencoder = RoadDamageAutoencoderModel(latent_dim=latent_dim, device=self.device)

        # ------------------------------------------------------------------------------------------
        # PART 2: EXTRACT ENCODER & DECODER COMPONENTS FROM AUTOENCODER
        # Variables: self.spatial_encoder, self.spatial_decoder
        # ------------------------------------------------------------------------------------------
        self.spatial_encoder = self.base_autoencoder.encoder
        self.fc_encode = self.base_autoencoder.fc_encode
        self.fc_decode = self.base_autoencoder.fc_decode
        self.spatial_decoder = self.base_autoencoder.decoder

        # ------------------------------------------------------------------------------------------
        # PART 3: TEMPORAL LSTM SEQUENCE RECURRENT CELL
        # Variable: self.temporal_lstm
        # Takes sequence of latent representations (B, T, latent_dim) -> Outputs (B, T, lstm_hidden_dim)
        # ------------------------------------------------------------------------------------------
        self.temporal_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.2 if num_lstm_layers > 1 else 0.0,
        )

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward computation across temporal sequence:
          1. Spatial encode each frame: x_t -> z_t
          2. LSTM recurrent sequence update: (z_1 ... z_T) -> (h_1 ... h_T)
          3. Spatial decode: h_t -> x_hat_t
          
        Args:
            x (torch.Tensor): Video sequence of shape (B, T, 3, 112, 112)
                              or single frame of shape (B, 3, 112, 112).
                              
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - x_reconstructed: Reconstructed video sequence (B, T, 3, 112, 112)
                - temporal_embeddings: Recurrent hidden vectors (B, T, lstm_hidden_dim)
        """
        if x.ndim == 4:
            x = x.unsqueeze(1)  # Expand to sequence length 1: (B, 1, 3, 112, 112)

        batch_size, seq_len, channels, height, width = x.shape

        # Step 1: Reshape to pass all frames through the spatial encoder concurrently
        x_flattened = x.view(batch_size * seq_len, channels, height, width)
        spatial_features = self.spatial_encoder(x_flattened)  # Shape: (B*T, 128, 14, 14)
        flat_spatial = torch.flatten(spatial_features, 1)
        latent_vectors = self.fc_encode(flat_spatial)         # Shape: (B*T, latent_dim)

        # Reshape into temporal sequence for the LSTM: (B, T, latent_dim)
        latent_seq = latent_vectors.view(batch_size, seq_len, self.latent_dim)

        # Step 2: Temporal LSTM processing
        lstm_out, _ = self.temporal_lstm(latent_seq)          # Shape: (B, T, lstm_hidden_dim)

        # Step 3: Decode LSTM hidden states back to spatial frames
        lstm_out_flat = lstm_out.contiguous().view(batch_size * seq_len, self.lstm_hidden_dim)
        decoded_flat = self.fc_decode(lstm_out_flat)
        decoded_spatial = decoded_flat.view(batch_size * seq_len, 128, 14, 14)
        x_recon_flat = self.spatial_decoder(decoded_spatial)  # Shape: (B*T, 3, 112, 112)

        # Reshape back to video sequence: (B, T, 3, 112, 112)
        x_reconstructed = x_recon_flat.view(batch_size, seq_len, channels, height, width)

        return x_reconstructed, lstm_out

    def predict(self, input_data: Union[str, Path, np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        Executes hybrid spatiotemporal anomaly detection.
        """
        self.eval()

        if isinstance(input_data, (str, Path)):
            raw = cv2.imread(str(input_data))
            rgb = cv2.cvtColor(cv2.resize(raw, (112, 112)), cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensor_input = tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        elif isinstance(input_data, np.ndarray):
            if input_data.ndim == 3:
                rgb = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB) if input_data.shape[2] == 3 else input_data
                res = cv2.resize(rgb, (112, 112))
                tensor = torch.from_numpy(res).permute(2, 0, 1).float() / 255.0
                tensor_input = tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            elif input_data.ndim == 4:
                # Video sequence: (T, H, W, C)
                tensors = []
                for f in input_data:
                    res = cv2.resize(f, (112, 112))
                    tensors.append(torch.from_numpy(res).permute(2, 0, 1).float() / 255.0)
                tensor_input = torch.stack(tensors, dim=0).unsqueeze(0).to(self.device)
            else:
                raise ValueError("Numpy input must be 3D image or 4D video sequence.")
        elif isinstance(input_data, torch.Tensor):
            tensor_input = input_data.to(self.device)
            if tensor_input.ndim == 4:
                tensor_input = tensor_input.unsqueeze(1)
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

        with torch.no_grad():
            x_recon, latent_lstm = self.forward(tensor_input)

            # Calculate pixel reconstruction residual on the latest frame
            residual = torch.mean(torch.abs(tensor_input[:, -1] - x_recon[:, -1]), dim=1, keepdim=True)
            reconstruction_error = float(torch.mean(residual).item())

        is_anomaly = reconstruction_error > 0.08
        recon_np = x_recon[:, -1].squeeze().permute(1, 2, 0).cpu().numpy()
        residual_np = residual.squeeze().cpu().numpy()

        return {
            "model_name": self.model_name,
            "predicted_class": "Spatiotemporal Distress Detected" if is_anomaly else "Normal Consistent Pavement",
            "confidence": min(1.0, reconstruction_error * 10.0),
            "reconstruction_error": reconstruction_error,
            "reconstructed_frame": recon_np,
            "residual_map": residual_np,
            "metadata": {
                "device": str(self.device),
                "latent_dim": self.latent_dim,
                "lstm_hidden_dim": self.lstm_hidden_dim,
            }
        }

    def plot_output(
        self,
        input_data: Union[str, Path, np.ndarray],
        output_save_path: Optional[Union[str, Path]] = None,
        show: bool = False
    ) -> np.ndarray:
        """
        Direct plotting method: Original Frame | LSTM-Autoencoder Reconstruction | Spatiotemporal Error.
        """
        preds = self.predict(input_data)

        if isinstance(input_data, (str, Path)):
            raw = cv2.imread(str(input_data))
            orig_rgb = cv2.cvtColor(cv2.resize(raw, (112, 112)), cv2.COLOR_BGR2RGB)
        else:
            orig_rgb = np.zeros((112, 112, 3), dtype=np.uint8)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
        ax1.imshow(orig_rgb)
        ax1.set_title("Input Frame (Sequence)", fontsize=11, fontweight="bold")
        ax1.axis("off")

        recon_clipped = np.clip(preds["reconstructed_frame"], 0.0, 1.0)
        ax2.imshow(recon_clipped)
        ax2.set_title("LSTM-Autoencoder Reconstruction", fontsize=11, fontweight="bold")
        ax2.axis("off")

        im3 = ax3.imshow(preds["residual_map"], cmap="magma")
        ax3.set_title(
            f"Spatiotemporal Error\nLoss: {preds['reconstruction_error']:.4f} | {preds['predicted_class']}",
            fontsize=11, fontweight="bold"
        )
        ax3.axis("off")
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        plt.tight_layout()

        if output_save_path:
            save_p = Path(output_save_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_p), dpi=150, bbox_inches="tight")
            print(f"[{self.model_name}] Hybrid diagnostic plot saved to: {save_p}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return orig_rgb


# ==================================================================================================
# STANDALONE EXECUTION & DIRECT PLOTTING COMMAND
# ==================================================================================================
if __name__ == "__main__":
    print(f"\n{'=' * 80}")
    print("ALGORITHM 10: HYBRID LSTM + AUTOENCODER - STANDALONE EXECUTION & TEST")
    print(f"{'=' * 80}")

    model = HybridLSTMAutoencoderModel()
    print(f"Instantiated: {model.model_name} on {model.device}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass test on video sequence of shape (Batch=2, SeqLen=4, Channels=3, H=112, W=112)
    dummy_seq = torch.randn(2, 4, 3, 112, 112).to(model.device)
    recon_seq, lstm_h = model(dummy_seq)
    print(f"Reconstruction sequence shape: {recon_seq.shape} (Expected: [2, 4, 3, 112, 112])")
    print(f"LSTM hidden sequence shape:     {lstm_h.shape}    (Expected: [2, 4, 256])")

    sample_img = PROJECT_ROOT / "dataset" / "subsets" / "multicountry" / "RDD2022" / "split" / "val" / "images" / "China_0010.jpg"
    if sample_img.is_file():
        out_plot = PROJECT_ROOT / "results" / "diagnostics" / "algo_10_hybrid_lstm_autoencoder_prediction.png"
        model.plot_output(sample_img, output_save_path=out_plot)
        print(f"Plot saved successfully to: {out_plot}")
    print(f"{'=' * 80}\n")
