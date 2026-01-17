"""
Modal deployment for Logo Post-Processing API.

Deploy with: modal deploy modal_app.py
Local dev:   modal serve modal_app.py
"""

import modal

app = modal.App("aio")

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
    )
    # Python dependencies
    .pip_install(
        # Core image processing
        "pillow",
        "numpy",
        "opencv-python-headless",
        # Background removal (BiRefNet-HR via transformers)
        "torch",
        "torchvision",
        "transformers",
        "timm",  # Required by BiRefNet
        "kornia",  # Required by BiRefNet
        # Upscaling (Spandrel - universal model loader)
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


@app.cls(
    image=image,
    gpu="T4",
    timeout=600,
    secrets=[
        modal.Secret.from_name("resend-api"),
        modal.Secret.from_name("my-app-secrets"),
    ],
    scaledown_window=300,  # Keep container warm for 5 min after last request
)
class LogoProcessor:
    """
    Logo processing service with optimized cold starts.

    Uses concurrent model loading and extended scaledown window
    to reduce cold start impact.
    """

    @modal.enter()
    def load_models(self):
        """
        Load ML models at container startup.
        The snap=True flag captures memory state after loading,
        so future cold starts restore from snapshot instead of reloading.
        """
        import sys
        sys.path.insert(0, "/root")

        from concurrent.futures import ThreadPoolExecutor
        import torch
        from transformers import AutoModelForImageSegmentation
        from huggingface_hub import hf_hub_download
        from spandrel import ModelLoader

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def load_birefnet():
            """Load BiRefNet-HR for background removal."""
            model = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet_HR",
                trust_remote_code=True,
            )
            model.to(device)
            model.eval()
            return model

        def load_esrgan():
            """Load Real-ESRGAN for upscaling."""
            model_path = hf_hub_download(
                repo_id="ai-forever/Real-ESRGAN",
                filename="RealESRGAN_x4.pth"
            )
            model = ModelLoader().load_from_file(model_path)
            model = model.to(device)
            model.eval()
            return model

        # Load both models concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            birefnet_future = executor.submit(load_birefnet)
            esrgan_future = executor.submit(load_esrgan)

            self.birefnet_model = birefnet_future.result()
            self.esrgan_model = esrgan_future.result()

        self.device = device
        print(f"Models loaded on {device}")

    @modal.method()
    def process(self, image_data: bytes, options: dict = None) -> dict:
        """
        Process a logo image. Private method callable only via Modal SDK.

        Args:
            image_data: Raw image bytes
            options: Processing options dict

        Returns:
            dict with processed image data and metadata
        """
        import sys
        sys.path.insert(0, "/root")

        # Inject pre-loaded models into the processing modules
        from src.processors import upscaler
        from src.models import birefnet_hr_model

        print(f"options: {str(options)}")

        # Set the pre-loaded models so they don't reload
        upscaler._model = self.esrgan_model
        upscaler._device = self.device

        # For BiRefNet, we need to patch the class to use pre-loaded model
        birefnet_hr_model._preloaded_model = self.birefnet_model
        birefnet_hr_model._preloaded_device = self.device

        # Import and run your processing logic
        from src.api.server import process_image
        return process_image(image_data, options or {})
