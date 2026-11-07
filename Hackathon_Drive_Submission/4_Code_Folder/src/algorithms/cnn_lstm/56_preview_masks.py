
from pathlib import Path
import cv2
import numpy as np

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").is_dir())
DATA = ROOT / "dataset" / "processed" / "cnn_lstm" / "train"
OUTPUT = ROOT / "results" / "cnn_lstm" / "previews"

OUTPUT.mkdir(parents=True, exist_ok=True)

files = sorted(DATA.glob("*.npz"))[:5]

if not files:
    raise FileNotFoundError(f"No processed videos found in {DATA}")

for file in files:
    with np.load(file) as data:
        rgb = data["frames"][0]
        mask = data["masks"][0]

    # Convert binary mask to a visible black-and-white image.
    mask_image = (mask > 0).astype(np.uint8) * 255
    mask_rgb = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2RGB)

    # Place road frame on the left and mask on the right.
    preview = np.concatenate([rgb, mask_rgb], axis=1)

    output_file = OUTPUT / f"{file.stem}_preview.png"
    cv2.imwrite(
        str(output_file),
        cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
    )

    print(f"Saved: {output_file}")

print(f"\nCreated {len(files)} previews in: {OUTPUT}")
