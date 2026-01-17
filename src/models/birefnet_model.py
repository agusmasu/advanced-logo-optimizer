import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from .base import BackgroundRemover


class BiRefNetRemover(BackgroundRemover):
    """Background remover using BiRefNet (state-of-the-art quality)."""

    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        self.model_name = model_name
        self._model = None
        self._device = None

    def _load_model(self):
        if self._model is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = AutoModelForImageSegmentation.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self._model.to(self._device)
            self._model.eval()

    def remove(self, input_path: str, output_path: str) -> Image.Image:
        self._load_model()

        image = Image.open(input_path).convert("RGB")
        original_size = image.size

        # BiRefNet expects 1024x1024 input
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
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


def remove_background(
    input_path: str,
    output_path: str,
    model_name: str = "ZhengPeng7/BiRefNet",
) -> Image.Image:
    """Convenience function for backward compatibility."""
    return BiRefNetRemover(model_name=model_name).remove(input_path, output_path)
