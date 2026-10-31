r"""
====================================================================================================
ALGORITHM 6: SPATIAL CONVLSTM (STANDALONE SPATIOTEMPORAL SEQUENCE MODEL)
====================================================================================================
Module: models/conv_lstm.py
Class: SpatialConvLSTMModel

Architecture Description:
  - Convolutional Long Short-Term Memory (ConvLSTM) replaces conventional fully-connected matrix
    multiplications in standard LSTMs with 2D spatial convolution operators ($*$).
  - Mathematical equations for ConvLSTM cell transitions at time-step $t$:
        $\mathbf{i}_t = \sigma(\mathbf{W}_{xi} * \mathcal{X}_t + \mathbf{W}_{hi} * \mathcal{H}_{t-1} + \mathbf{b}_i)$
        $\mathbf{f}_t = \sigma(\mathbf{W}_{xf} * \mathcal{X}_t + \mathbf{W}_{hf} * \mathcal{H}_{t-1} + \mathbf{b}_f)$
        $\widetilde{\mathcal{C}}_t = \tanh(\mathbf{W}_{xc} * \mathcal{X}_t + \mathbf{W}_{hc} * \mathcal{H}_{t-1} + \mathbf{b}_c)$
        $\mathcal{C}_t = \mathbf{f}_t \odot \mathcal{C}_{t-1} + \mathbf{i}_t \odot \widetilde{\mathcal{C}}_t$
        $\mathbf{o}_t = \sigma(\mathbf{W}_{xo} * \mathcal{X}_t + \mathbf{W}_{ho} * \mathcal{H}_{t-1} + \mathbf{b}_o)$
        $\mathcal{H}_t = \mathbf{o}_t \odot \tanh(\mathcal{C}_t)$
  - Preserves 2D spatial topology $(H, W)$ while propagating recurrent temporal memory across video frames.
  - Generates binary pixel-level segmentation masks for damaged road surface tracking.

Hybrid Role:
  - In `HybridLSTMAutoencoderModel`, ConvLSTM acts as the temporal latent recurrent processor:
    it ingests compressed spatial latent representations $\mathcal{Z}_t$ from the Autoencoder Encoder
    and outputs predictive state $\mathcal{H}_t$ to the Autoencoder Decoder.
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


class ConvLSTMCell(nn.Module):
    """
    Single-layer 2D Convolutional LSTM Cell.
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.padding = kernel_size // 2

        # Fused convolution for all 4 gates (input, forget, cell candidate, output)
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, _, height, width = x.shape
        device = x.device

        if state is None:
            h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
            c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        else:
            h, c = state

        # Concatenate current input and previous hidden state along channel dimension
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        # Split into four gate components
        i_gate, f_gate, c_candidate, o_gate = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        c_cand = torch.tanh(c_candidate)
        o = torch.sigmoid(o_gate)

        # Update cell state and hidden state
        c_next = f * c + i * c_cand
        h_next = o * torch.tanh(c_next)

        return h_next, (h_next, c_next)


class SpatialConvLSTMModel(BaseRoadDamageModel):
    """
    End-to-End Spatiotemporal Convolutional LSTM Model for Video Road Damage Segmentation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(
            model_name="SpatialConvLSTM",
            num_classes=1,  # Binary road distress mask
            device=device,
        )

        # ------------------------------------------------------------------------------------------
        # PART A: SPATIAL CNN ENCODER
        # Variable: self.cnn
        # Downsamples raw frame (B*T, 3, H, W) to compact spatial feature representation (B*T, 64, H/4, W/4)
        # ------------------------------------------------------------------------------------------
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------------------------------
        # PART B: SPATIAL LSTM TEMPORAL ENGINE
        # Variable: self.lstm
        # Runs LSTM recurrence across frame sequence at each spatial coordinate
        # ------------------------------------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        # ------------------------------------------------------------------------------------------
        # PART C: SPATIAL DECODER / SEGMENTATION HEAD
        # Variable: self.decoder
        # Combines spatial features + temporal features (128 channels) and maps to binary mask logits
        # ------------------------------------------------------------------------------------------
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, 1),
        )

        # Default weights checkpoint
        default_weights = PROJECT_ROOT / "results" / "cnn_lstm" / "segmentation" / "best_cnn_lstm_spatial_v3.pt"
        if not default_weights.is_file():
            default_weights = PROJECT_ROOT / "results" / "cnn_lstm" / "best_cnn_lstm.pt"
        target_weights = Path(weights_path) if weights_path else default_weights
        if target_weights.is_file():
            try:
                self.load_checkpoint(target_weights, strict=True)
            except Exception as e:
                print(f"[{self.model_name}] Checkpoint note: {e}")

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward spatiotemporal computation.
        
        Args:
            x (torch.Tensor): Video sequence tensor of shape (Batch, Seq_Len, Channels, Height, Width)
                              or (Batch, Channels, Height, Width) for single frame.
                              
        Returns:
            torch.Tensor: Predicted distress mask logits of shape (Batch, Seq_Len, 1, Height, Width).
        """
        if x.ndim == 4:
            x = x.unsqueeze(1)
        batch, steps, channels, height, width = x.shape

        features = self.cnn(
            x.reshape(batch * steps, channels, height, width)
        )

        _, feature_channels, fh, fw = features.shape

        features = features.reshape(
            batch, steps, feature_channels, fh, fw
        )

        # Run LSTM through time at each spatial location.
        sequences = features.permute(0, 3, 4, 1, 2)
        sequences = sequences.reshape(
            batch * fh * fw, steps, feature_channels
        )

        temporal, _ = self.lstm(sequences)

        temporal = temporal.reshape(
            batch, fh, fw, steps, feature_channels
        ).permute(0, 3, 4, 1, 2)

        spatial_flat = features.reshape(
            batch * steps, feature_channels, fh, fw
        )

        temporal_flat = temporal.reshape(
            batch * steps, feature_channels, fh, fw
        )

        combined = torch.cat(
            [spatial_flat, temporal_flat], dim=1
        )

        logits = self.decoder(combined)

        logits = torch.nn.functional.interpolate(
            logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        return logits.reshape(
            batch, steps, 1, height, width
        )

    def predict(self, input_data: Union[str, Path, np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        Runs video sequence segmentation inference.
        """
        self.eval()

        if isinstance(input_data, (str, Path)):
            # Load single image or first frame
            frame = cv2.imread(str(input_data))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (128, 128))
            tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
            tensor_input = tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        elif isinstance(input_data, np.ndarray):
            if input_data.ndim == 3:
                rgb = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB) if input_data.shape[2] == 3 else input_data
                resized = cv2.resize(rgb, (128, 128))
                tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
                tensor_input = tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            elif input_data.ndim == 4:
                # Video sequence: (T, H, W, C)
                seq_tensors = []
                for f in input_data:
                    res = cv2.resize(f, (128, 128))
                    t = torch.from_numpy(res).permute(2, 0, 1).float() / 255.0
                    seq_tensors.append(t.unsqueeze(0))
                tensor_input = torch.cat(seq_tensors, dim=0).unsqueeze(0).to(self.device)
            else:
                raise ValueError("Numpy input must be 3D image or 4D video sequence.")
        elif isinstance(input_data, torch.Tensor):
            tensor_input = input_data.to(self.device)
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

        with torch.no_grad():
            mask_logits = self.forward(tensor_input)
            probs = torch.sigmoid(mask_logits).squeeze().cpu().numpy()

        binary_mask = (probs > 0.5).astype(np.uint8)
        damage_ratio = float(np.mean(binary_mask))

        return {
            "model_name": self.model_name,
            "predicted_class": "Road Damage Region" if damage_ratio > 0.01 else "Clean Road",
            "confidence": float(np.max(probs)) if probs.size > 0 else 0.0,
            "damage_pixel_ratio": damage_ratio,
            "probability_map": probs,
            "binary_mask": binary_mask,
            "metadata": {
                "device": str(self.device),
                "resolution": "(128, 128)",
            }
        }

    def plot_output(
        self,
        input_data: Union[str, Path, np.ndarray],
        output_save_path: Optional[Union[str, Path]] = None,
        show: bool = False
    ) -> np.ndarray:
        """
        Direct plotting method: displays original frame, predicted damage heatmap, and binary overlay.
        """
        preds = self.predict(input_data)
        prob_map = preds["probability_map"]
        if prob_map.ndim == 3:
            prob_map = prob_map[-1]  # Show latest frame in sequence

        if isinstance(input_data, (str, Path)):
            orig = cv2.imread(str(input_data))
            orig_rgb = cv2.cvtColor(cv2.resize(orig, (128, 128)), cv2.COLOR_BGR2RGB)
        else:
            orig_rgb = np.zeros((128, 128, 3), dtype=np.uint8)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
        ax1.imshow(orig_rgb)
        ax1.set_title("Input Road Frame", fontsize=11, fontweight="bold")
        ax1.axis("off")

        im2 = ax2.imshow(prob_map, cmap="jet", vmin=0.0, vmax=1.0)
        ax2.set_title(f"Damage Heatmap (Max: {preds['confidence']:.2f})", fontsize=11, fontweight="bold")
        ax2.axis("off")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        overlay = orig_rgb.copy()
        mask_bool = preds["binary_mask"] > 0
        if mask_bool.ndim == 3:
            mask_bool = mask_bool[-1]
        overlay[mask_bool] = [255, 0, 0]  # Highlight red
        ax3.imshow(overlay)
        ax3.set_title(f"Segmented Distress (Coverage: {preds['damage_pixel_ratio']:.1%})", fontsize=11, fontweight="bold")
        ax3.axis("off")

        plt.tight_layout()

        if output_save_path:
            save_p = Path(output_save_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_p), dpi=150, bbox_inches="tight")
            print(f"[{self.model_name}] Segmentation plot saved to: {save_p}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return orig_rgb
