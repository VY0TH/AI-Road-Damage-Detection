"""Road Video Damage Segmentation Processor."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import torch

from model import CNNSpatialLSTM


class RoadVideoSegmenter:
    def __init__(self, checkpoint_path: Path | None = None, threshold: float = 0.5):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.model = CNNSpatialLSTM().to(self.dev)

        if checkpoint_path and Path(checkpoint_path).is_file():
            state = torch.load(checkpoint_path, map_location=self.dev)
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            self.model.load_state_dict(state)
        self.model.eval()

    def process_frames(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        # frames: list of BGR images
        t = len(frames)
        if t == 0:
            return []
        h, w = frames[0].shape[:2]

        resized_frames = [cv2.resize(f, (256, 256)) for f in frames]
        rgb_tensor = torch.tensor(
            np.array(resized_frames, dtype=np.float32) / 255.0
        ).permute(0, 3, 1, 2).unsqueeze(0).to(self.dev)  # (1, T, 3, 256, 256)

        with torch.no_grad():
            logits = self.model(rgb_tensor)  # (1, T, 1, 256, 256)
            probs = torch.sigmoid(logits)[0, :, 0].cpu().numpy()

        masks = []
        for i in range(t):
            mask_bin = (probs[i] >= self.threshold).astype(np.uint8) * 255
            mask_orig = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
            masks.append(mask_orig)
        return masks
