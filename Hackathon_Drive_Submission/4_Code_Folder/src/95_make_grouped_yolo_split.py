"""Compatibility launcher; implementation is grouped by algorithm."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "algorithms" / "yolo" / "95_make_grouped_yolo_split.py"), run_name="__main__")
