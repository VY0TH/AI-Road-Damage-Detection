"""Compatibility launcher; implementation is grouped by algorithm."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "algorithms" / "efficientnet" / "53_hybrid_yolo_clahe_efficientnet.py"), run_name="__main__")
