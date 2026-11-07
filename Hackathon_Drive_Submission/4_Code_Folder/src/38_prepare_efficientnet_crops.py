"""Compatibility launcher; implementation is grouped by algorithm."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "algorithms" / "efficientnet" / "38_prepare_efficientnet_crops.py"), run_name="__main__")
