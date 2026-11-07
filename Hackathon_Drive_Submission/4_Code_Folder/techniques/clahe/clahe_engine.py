"""Technique: CLAHE (Contrast-Limited Adaptive Histogram Equalization)."""
import cv2
import numpy as np
from PIL import Image


def apply_clahe(image: Image.Image | np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> Image.Image | np.ndarray:
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        bgr = image

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced_l = clahe.apply(l)

    merged = cv2.merge((enhanced_l, a, b))
    enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    if is_pil:
        return Image.fromarray(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB))
    return enhanced_bgr
