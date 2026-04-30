import os
import logging

# Project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

TRUFOR_WEIGHTS = os.path.join(WEIGHTS_DIR, "trufor.pth")
CATNET_WEIGHTS = os.path.join(WEIGHTS_DIR, "catnet.pth")

TESSERACT_CMD = "tesseract"

# Preprocessing config
PDF_DPI = 200
IMAGE_MAX_DIM = 4096

# Detection thresholds
THRESHOLD = {
    "C1": 0.55,
    "C2": 0.45,
    "C3": 0.50,
    "C4": 0.50,
    "C5": 0.55,
    "C6": 0.60,
    "C7": None,
    "C8": None,
    "C9": 0.45,
}

# C7 parameters
C7_IQR_MULTIPLIER = 2.0
C7_MIN_STRETCH_FACTOR = 1.10

# C8 parameters
C8_SIGMA_LAP_THRESHOLD = 3.5
C8_ENTROPY_BG_THRESHOLD = 1.2

# C9 parameters
C9_FIELD_NOISE_RATIO = 0.20

C9_FIELD_LABELS = [
    "name", "date", "dob", "amount", "total", "bill no", "mrd",
    "uhid", "patient", "doctor", "diagnosis", "age", "sex",
    "admission", "discharge", "pan", "gstin"
]

# Bounding box parameters
BBOX_MORPH_CLOSE_KERNEL = 15
BBOX_MORPH_OPEN_KERNEL = 5
BBOX_MIN_AREA = 100

# Fusion weights
FUSION_WEIGHTS = {
    "C1": [0.45, 0.45, 0.10],
    "C2": [0.30, 0.10, 0.60],
    "C3": [0.50, 0.40, 0.10],
    "C4": [0.55, 0.35, 0.10],
    "C5": [0.70, 0.20, 0.10],
    "C6": [0.80, 0.10, 0.10],
    "C7": [0.00, 0.00, 1.00],
    "C9": [0.50, 0.10, 0.40],
}

# pHash configuration
PHASH_PATCH_SIZE = 64
PHASH_STRIDE = 32
PHASH_DISTANCE_THRESHOLD = 10

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s"
)

# Logger helper

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
