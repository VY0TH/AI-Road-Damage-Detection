"""Compatibility launcher; implementation is grouped by algorithm."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "algorithms" / "resnet50" / "83_run_yolo_resnet_trained.py"), run_name="__main__")
