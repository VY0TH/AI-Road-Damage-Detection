"""Step 2 CLI Runner: Execute Data Preprocessing Pipeline."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_preprocessor import preprocess_folder
import pipeline_config as cfg


def run_step_2(source_dir: Path | None = None, output_dir: Path | None = None) -> dict:
    source = source_dir or cfg.PROCESSED_DATA_DIR
    target = output_dir or (cfg.DATASET_ROOT / "processed" / "preprocessed_standardized")
    print(f"\n[STEP 2] Running Image Preprocessing:")
    print(f"  Source : {source}")
    print(f"  Target : {target}")
    print(f"  Size   : {cfg.DEFAULT_IMAGE_SIZE}x{cfg.DEFAULT_IMAGE_SIZE}")

    proc, skip = preprocess_folder(
        source_dir=source,
        output_dir=target,
        target_size=cfg.DEFAULT_IMAGE_SIZE,
        keep_aspect_ratio=True,
    )

    print(f"[STEP 2 COMPLETED] Processed: {proc} images, Skipped/Corrupt: {skip}")
    print(f"  Preprocessed dataset ready at: {target}\n")
    return {"processed": proc, "skipped": skip, "output_dir": str(target)}


if __name__ == "__main__":
    run_step_2()
