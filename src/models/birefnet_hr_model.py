import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from .base import BackgroundRemover

# Pre-loaded model references (set by Modal at startup)
_preloaded_model = None
_preloaded_device = None


class BiRefNetHRRemover(BackgroundRemover):
    """Background remover using BiRefNet-HR (high resolution, best for fine details)."""

    def __init__(self):
        self._model = None
        self._device = None

    def _load_model(self):
        global _preloaded_model, _preloaded_device

        # Use pre-loaded model if available (set by Modal)
        if _preloaded_model is not None:
            self._model = _preloaded_model
            self._device = _preloaded_device
            return

        if self._model is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet_HR",
                trust_remote_code=True,
            )
            self._model.to(self._device)
            self._model.eval()

    def remove(self, input_path: str, output_path: str) -> Image.Image:
        self._load_model()

        image = Image.open(input_path).convert("RGB")
        original_size = image.size

        # BiRefNet-HR can handle higher resolution - use 2048x2048
        transform = transforms.Compose([
            transforms.Resize((2048, 2048)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        input_tensor = transform(image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            preds = self._model(input_tensor)[-1].sigmoid()

        mask = preds[0].squeeze().cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask).resize(original_size, Image.LANCZOS)

        rgba_image = image.convert("RGBA")
        rgba_image.putalpha(mask_image)
        rgba_image.save(output_path, "PNG")

        return rgba_image


def remove_background(input_path: str, output_path: str) -> Image.Image:
    """Convenience function for backward compatibility."""
    return BiRefNetHRRemover().remove(input_path, output_path)
