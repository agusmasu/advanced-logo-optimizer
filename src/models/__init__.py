from .base import BackgroundRemover
from .birefnet_hr_model import BiRefNetHRRemover

# Lazy imports for optional models (avoids requiring all dependencies)
def __getattr__(name):
    if name == "RembgRemover":
        from .rembg_model import RembgRemover
        return RembgRemover
    elif name == "BiRefNetRemover":
        from .birefnet_model import BiRefNetRemover
        return BiRefNetRemover
    elif name == "BEN2Remover":
        from .ben2_model import BEN2Remover
        return BEN2Remover
    elif name == "RMBGRemover":
        from .rmbg_model import RMBGRemover
        return RMBGRemover
    elif name == "SAM2Remover":
        try:
            from .sam2_model import SAM2Remover
            return SAM2Remover
        except ImportError:
            return None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BackgroundRemover",
    "RembgRemover",
    "BiRefNetRemover",
    "BiRefNetHRRemover",
    "BEN2Remover",
    "RMBGRemover",
    "SAM2Remover",
]
