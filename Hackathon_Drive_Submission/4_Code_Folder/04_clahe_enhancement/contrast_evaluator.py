"""Step 4: Contrast Evaluator (determines when contrast enhancement is useful).

Computes:
1. Root-Mean-Square (RMS) Contrast of the luminance channel.
2. Michelson Contrast.
3. Shannon Entropy of pixel intensities.

Decision Rule:
Contrast enhancement via CLAHE is beneficial when:
- RMS contrast is low (< threshold, typically 0.18-0.22 on normalized [0,1] scale)
- Image is underexposed/overcast or in shadow where damage features lack dynamic range.
- High-contrast images are kept unchanged to prevent boosting background noise or speckles.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def compute_contrast_metrics(image: Image.Image | np.ndarray) -> dict[str, float]:
    if isinstance(image, Image.Image):
        gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # 1. RMS Contrast: standard deviation of luminance
    rms_contrast = float(np.std(gray))

    # 2. Michelson Contrast: (max - min) / (max + min + eps)
    p_min, p_max = float(np.min(gray)), float(np.max(gray))
    michelson = (p_max - p_min) / (p_max + p_min + 1e-6)

    # 3. Shannon Entropy
    hist, _ = np.histogram(gray, bins=256, range=(0, 1), density=True)
    hist = hist[hist > 0]
    entropy = -float(np.sum(hist * np.log2(hist)))

    return {
        "rms_contrast": round(rms_contrast, 4),
        "michelson_contrast": round(michelson, 4),
        "entropy": round(entropy, 4),
        "mean_luminance": round(float(np.mean(gray)), 4),
    }


def is_contrast_enhancement_useful(
    image: Image.Image | np.ndarray,
    rms_threshold: float = 0.20,
    entropy_threshold: float = 7.0,
) -> tuple[bool, str]:
    metrics = compute_contrast_metrics(image)
    reasons = []

    if metrics["rms_contrast"] < rms_threshold:
        reasons.append(f"Low RMS contrast ({metrics['rms_contrast']:.3f} < {rms_threshold})")
    if metrics["mean_luminance"] < 0.35:
        reasons.append(f"Under-exposed / shadowed road surface ({metrics['mean_luminance']:.2f} < 0.35)")
    if metrics["entropy"] < entropy_threshold and metrics["rms_contrast"] < 0.25:
        reasons.append(f"Low visual dynamic range (entropy {metrics['entropy']:.2f})")

    if reasons:
        return True, "; ".join(reasons)
    return False, "Sufficient natural contrast present; CLAHE would risk over-amplifying background noise."
