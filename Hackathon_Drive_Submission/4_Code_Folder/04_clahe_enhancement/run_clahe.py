"""Step 4 CLI Runner: Adaptive CLAHE Enhancement Pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from clahe_processor import apply_clahe_to_image
from contrast_evaluator import compute_contrast_metrics, is_contrast_enhancement_useful
import pipeline_config as cfg

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def run_step_4(
    source_dir: Path | None = None,
    output_dir: Path | None = None,
    max_demo_images: int = 10,
    adaptive_mode: bool = True,
) -> dict:
    source = source_dir or cfg.PROCESSED_DATA_DIR
    target = output_dir or (cfg.DATASET_ROOT / "processed" / "crops_clahe_enhanced")
    demo_dir = cfg.RESULTS_DIR / "clahe_comparisons"
    demo_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"[STEP 4] CLAHE CONTRAST ENHANCEMENT (ADAPTIVE)")
    print(f"Source: {source}")
    print(f"Mode  : {'Adaptive (enhances only when useful)' if adaptive_mode else 'Global (all images)'}")
    print(f"{'=' * 70}")

    image_paths = sorted([
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_paths:
        print("No images found to process.")
        return {}

    analyzed = 0
    enhanced = 0
    skipped_high_contrast = 0
    demo_saved = 0
    records = []

    for path in image_paths:
        analyzed += 1
        with Image.open(path) as img:
            rgb_img = img.convert("RGB")
            pre_metrics = compute_contrast_metrics(rgb_img)
            useful, reason = is_contrast_enhancement_useful(rgb_img)

            if useful or not adaptive_mode:
                enhanced_img = apply_clahe_to_image(rgb_img)
                post_metrics = compute_contrast_metrics(enhanced_img)
                enhanced += 1

                # Save a side-by-side comparison for demonstration
                if demo_saved < max_demo_images:
                    w, h = rgb_img.size
                    comparison = Image.new("RGB", (w * 2, h))
                    comparison.paste(rgb_img, (0, 0))
                    comparison.paste(enhanced_img, (w, 0))
                    demo_path = demo_dir / f"compare_{path.stem}.jpg"
                    comparison.save(demo_path, format="JPEG", quality=90)
                    demo_saved += 1
                    records.append({
                        "file": path.name,
                        "class": path.parent.name,
                        "decision": "ENHANCED",
                        "reason": reason,
                        "pre_rms": pre_metrics["rms_contrast"],
                        "post_rms": post_metrics["rms_contrast"],
                        "contrast_gain": round(post_metrics["rms_contrast"] - pre_metrics["rms_contrast"], 4),
                        "comparison_path": str(demo_path),
                    })
            else:
                skipped_high_contrast += 1

    summary = {
        "total_analyzed": analyzed,
        "enhanced_count": enhanced,
        "skipped_high_contrast": skipped_high_contrast,
        "enhancement_rate": round(100 * enhanced / max(1, analyzed), 1),
        "demo_comparisons_saved": demo_saved,
        "demo_dir": str(demo_dir),
        "sample_records": records,
    }

    output_json = cfg.RESULTS_DIR / "step04_clahe_contrast_report.json"
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAnalysis Summary:")
    print(f"  Total Images Inspected : {analyzed}")
    print(f"  Contrast Enhancement Useful: {enhanced} ({summary['enhancement_rate']}%)")
    print(f"  Natural Contrast Preserved : {skipped_high_contrast}")
    print(f"  Comparison Samples Saved   : {demo_saved} into {demo_dir}")
    print(f"Report saved to: {output_json}")
    print(f"{'=' * 70}\n")
    return summary


if __name__ == "__main__":
    run_step_4()
