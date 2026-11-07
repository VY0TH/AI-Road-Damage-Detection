def apply_clahe(crop_bgr):
    """Enhance local contrast without changing image dimensions."""
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )

    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)