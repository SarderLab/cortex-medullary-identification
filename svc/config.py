"""Defaults for the standalone cortex/medulla SVC workflow."""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_XLSX = Path(os.environ.get(
    "SVC_TRAIN_XLSX",
    "/blue/pinaki.sarder/j.fermin/DigitAb_Master/Segmentation/csv_files/case_split_70_15_151.xlsx",
))
DEFAULT_FEATURE_ROOT = Path(os.environ.get(
    "SVC_TRAIN_FEATURE_ROOT",
    "/orange/pinaki.sarder/j.fermin/cort_med_wsi/UNI_V1_features",
))
DEFAULT_MODEL_DIR = Path(os.environ.get("SVC_MODEL_DIR", str(PROJECT_ROOT / "models")))

BEST_N_COMPONENTS = 1024
BEST_C = 2.0
DEFAULT_THRESHOLD = 0.0
DEFAULT_CACHE_SIZE_MB = 4096.0
DEFAULT_INFERENCE_BATCH_SIZE = 32768

LABEL_NAMES = {0: "cortical", 1: "medullary"}
RAW_LABEL_MAPPING = {
    -1: "excluded",
    0: "background (excluded)",
    1: "cortical",
    2: "medullary",
}
