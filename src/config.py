import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUTS_DIR = DATA_DIR / "inputs"
OUTPUTS_DIR = DATA_DIR / "outputs"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"

# Ensure directories exist (wrapped for Modal compatibility)
try:
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass  # Running in read-only environment (e.g., Modal)

# Processing defaults
DEFAULT_UPSCALE_FACTOR = 4
DEFAULT_TRIM_PADDING = 24
DEFAULT_MAX_COLORS = 12
DEFAULT_VTRACER_FILTER_SPECKLE = 4
DEFAULT_VTRACER_COLOR_PRECISION = 6

# Background removal defaults
DEFAULT_RMBG_THRESHOLD = 0.5
DEFAULT_RMBG_ERODE_PIXELS = 1

# API settings
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
EMAIL_FROM = os.getenv("EMAIL_FROM")