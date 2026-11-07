"""Step 4: CLAHE (Contrast Limited Adaptive Histogram Equalization) Processor.

Applies CLAHE exclusively to the Luminance (L) channel of the LAB color space,
preserving chromaticity (A and B channels) to avoid color distortion while
enhancing subtle surface texture on asphalt and dark potholes.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def apply_clahe_to_image(
    image: Image.Image | np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> Image.Image:
    """Applies CLAHE to the L channel of an RGB image."""
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        bgr = image

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(enhanced_rgb) if is_pil else enhanced_bgr
