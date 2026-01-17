"""
Modal deployment for Logo Post-Processing API.

Deploy with: modal deploy modal_app.py
Local dev:   modal serve modal_app.py
"""

import os
import modal

MODAL_APP_NAME = os.getenv("MODAL_APP_NAME")
if not MODAL_APP_NAME:
    raise ValueError("MODAL_APP_NAME environment variable is required")

app = modal.App(MODAL_APP_NAME)

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    # System dependencies for image processing and vector export
    .apt_install(
        "git",
        "inkscape",
        "potrace",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        # Vulkan libraries for realesrgan-ncnn-py
        "libvulkan1",
        "libvulkan-dev",
        "mesa-vulkan-drivers",
    )
    # Python dependencies
    .pip_install(
        # Core image processing
        "pillow",
        "numpy",
        "opencv-python-headless",
        # Background removal
        "rembg",
        "onnxruntime",
        "torch",
        "torchvision",
        "transformers",
        "scipy",
        "einops",
        "kornia",
        "timm",
        # Upscaling (Spandrel - universal model loader, simpler than basicsr)
        "spandrel",
        "huggingface_hub",
        # Vectorization
        "vtracer",
        "scour",
        # Web API
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
        "resend",
    )
    # Pre-download model weights for warm starts
    .run_commands(
        # Download BiRefNet-HR model weights (best for fine details)
        "python -c \"from transformers import AutoModelForImageSegmentation; AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet_HR', trust_remote_code=True)\"",
        # Download Real-ESRGAN model weights for upscaling
        "python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ai-forever/Real-ESRGAN', filename='RealESRGAN_x4.pth')\"",
    )
    # Add local source code LAST (Modal adds these at container startup for faster rebuilds)
    .add_local_dir("src", remote_path="/root/src")
)


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    secrets=[modal.Secret.from_name("resend-api")],
)
@modal.asgi_app()
def fastapi_app():
    """Serve the FastAPI application."""
    import sys
    sys.path.insert(0, "/root")
    from src.api.server import app as fastapi_application

    return fastapi_application
