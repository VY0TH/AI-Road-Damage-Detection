"""Compatibility launcher; implementation is grouped by algorithm."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "algorithms" / "vit" / "44_hybrid_yolo_vit.py"), run_name="__main__")
