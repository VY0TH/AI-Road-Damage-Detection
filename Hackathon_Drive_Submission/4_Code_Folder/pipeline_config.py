"""Central pipeline configuration for the 10-step Image Classification sequence."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
PROCESSED_DATA_DIR = DATASET_ROOT / "processed" / "efficientnet_crops"
RAW_DATA_DIR = DATASET_ROOT / "raw"
SUBSETS_DIR = DATASET_ROOT / "subsets" / "multicountry" / "RDD2022"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Standard 10 road damage classes in CRDDC / RDD2022
CLASS_NAMES = [
    "D00",   # Longitudinal Crack
    "D01",   # Longitudinal Construction Joint Crack
    "D0w0",  # Longitudinal Crack with Water
    "D10",   # Transverse Crack
    "D11",   # Transverse Construction Joint Crack
    "D20",   # Alligator Crack
    "D40",   # Pothole
    "D43",   # Crosswalk Blur
    "D44",   # White Line Blur
    "D50",   # Rutting / Bump / Pothole group
]

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

# Default image configurations
DEFAULT_IMAGE_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Device configuration
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
