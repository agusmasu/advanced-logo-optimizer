"""
Image upscaling using Spandrel for universal model loading.

Spandrel provides automatic architecture detection and a unified interface
for running various upscaling models (Real-ESRGAN, HAT, SwinIR, etc.)
with seamless CUDA/CPU fallback.
"""

from PIL import Image
import numpy as np
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download

# Lazy-loaded model instance
_model = None
_device = None

# Model configuration
MODEL_REPO = "ai-forever/Real-ESRGAN"
MODEL_FILENAME = "RealESRGAN_x4.pth"


def _get_device() -> torch.device:
    """Get the best available device (CUDA or CPU)."""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _load_model():
    """Load the upscaling model using Spandrel."""
    global _model
    if _model is not None:
        return _model

    from spandrel import ModelLoader

    # Download model from HuggingFace Hub (cached after first download)
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)

    # Load model with Spandrel (auto-detects architecture)
    _model = ModelLoader().load_from_file(model_path)
    _model = _model.to(_get_device())
    _model.eval()

    return _model


def upscale(input_path: str, output_path: str, scale: int = 4) -> Image.Image:
    """
    Upscale an image using Real-ESRGAN AI model via Spandrel.

    Args:
        input_path: Path to the input image
        output_path: Path to save the upscaled image
        scale: Upscaling factor (default 4x, must match model)

    Returns:
        PIL Image object of the upscaled image
    """
    model = _load_model()
    device = _get_device()

    with Image.open(input_path) as img:
        # Convert to RGB (upscaler expects 3 channels)
        img_rgb = img.convert("RGB")

        # Convert to tensor: HWC -> CHW, normalize to [0, 1]
        img_array = np.array(img_rgb).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(device)

        # Run inference
        with torch.no_grad():
            # Use half precision on CUDA for better performance
            if device.type == "cuda":
                with torch.autocast(device_type="cuda"):
                    output_tensor = model(img_tensor)
            else:
                output_tensor = model(img_tensor)

        # Convert back to PIL Image: CHW -> HWC, denormalize
        output_array = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_array = np.clip(output_array * 255, 0, 255).astype(np.uint8)
        upscaled = Image.fromarray(output_array)

        # If original had alpha channel, handle transparency
        if img.mode == "RGBA":
            # Upscale alpha channel separately using bicubic
            alpha = img.split()[3]
            new_size = (upscaled.width, upscaled.height)
            alpha_upscaled = alpha.resize(new_size, Image.Resampling.BICUBIC)
            upscaled = upscaled.convert("RGBA")
            upscaled.putalpha(alpha_upscaled)

    upscaled.save(output_path, "PNG")
    return upscaled
