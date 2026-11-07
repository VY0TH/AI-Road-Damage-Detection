"""Compatibility launcher; implementation is grouped by algorithm."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "algorithms" / "cnn_lstm" / "70_standalone_video_demo.py"), run_name="__main__")
