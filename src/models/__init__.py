from .base import BackgroundRemover
from .rembg_model import RembgRemover
from .birefnet_model import BiRefNetRemover
from .birefnet_hr_model import BiRefNetHRRemover
from .ben2_model import BEN2Remover
from .rmbg_model import RMBGRemover

# SAM2 requires optional dependency - import lazily
try:
    from .sam2_model import SAM2Remover
except ImportError:
    SAM2Remover = None

__all__ = [
    "BackgroundRemover",
    "RembgRemover",
    "BiRefNetRemover",
    "BiRefNetHRRemover",
    "BEN2Remover",
    "RMBGRemover",
    "SAM2Remover",
]
