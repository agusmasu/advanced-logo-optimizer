import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torchvision.transforms.functional import normalize
from transformers import AutoModelForImageSegmentation

from .base import BackgroundRemover


class RMBGRemover(BackgroundRemover):
    """Background remover using RMBG 1.4 (Bria AI) with edge control."""

    def __init__(self, threshold: float = 0.5, erode_pixels: int = 1):
        self.threshold = threshold
        self.erode_pixels = erode_pixels
        self._model = None
        self._device = None

    def _load_model(self):
        if self._model is None:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self._model = AutoModelForImageSegmentation.from_pretrained(
                "briaai/RMBG-1.4",
                trust_remote_code=True,
            )
            self._model.to(self._device)
            self._model.eval()

    def remove(self, input_path: str, output_path: str) -> Image.Image:
        self._load_model()

        orig_image = Image.open(input_path).convert("RGB")
        orig_im = np.array(orig_image)
        orig_im_size = orig_im.shape[0:2]
        model_input_size = [1024, 1024]

        # RMBG 1.4 specific preprocessing
        im_tensor = torch.tensor(orig_im, dtype=torch.float32).permute(2, 0, 1)
        im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), size=model_input_size, mode='bilinear')
        image = torch.divide(im_tensor, 255.0)
        image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]).to(self._device)

        with torch.no_grad():
            result = self._model(image)

        # Post-process to get mask
        result_mask = torch.squeeze(
            F.interpolate(result[0][0], size=orig_im_size, mode='bicubic', align_corners=False), 0
        )
        ma, mi = torch.max(result_mask), torch.min(result_mask)
        result_mask = (result_mask - mi) / (ma - mi)

        # Apply threshold for hard binary edges
        result_mask = (result_mask > self.threshold).float()

        mask_array = result_mask.permute(1, 2, 0).cpu().data.numpy().squeeze().astype(np.uint8)

        # Apply morphological erosion to remove edge fringing
        if self.erode_pixels > 0:
            mask_array = ndimage.binary_erosion(mask_array, iterations=self.erode_pixels).astype(np.uint8)

        mask_image = Image.fromarray(mask_array * 255)

        rgba_image = orig_image.convert("RGBA")
        rgba_image.putalpha(mask_image)
        rgba_image.save(output_path, "PNG")

        return rgba_image


def remove_background(
    input_path: str,
    output_path: str,
    threshold: float = 0.5,
    erode_pixels: int = 1,
) -> Image.Image:
    """Convenience function for backward compatibility."""
    return RMBGRemover(threshold=threshold, erode_pixels=erode_pixels).remove(input_path, output_path)
