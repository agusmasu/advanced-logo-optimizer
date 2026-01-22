import base64
import os
import tempfile
import zipfile
from pathlib import Path

import resend
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from resend import Emails

from src.config import EMAIL_FROM, RESEND_API_KEY
from src.models.birefnet_hr_model import BiRefNetHRRemover
from src.processors.cleanup import trim_transparent
from src.processors.exporter import export_to_eps, export_to_pdf
from src.processors.upscaler import upscale
from src.processors.pipeline_vectorizer import run_vectorization_pipeline

# Get the path to the email template
TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
EMAIL_TEMPLATE_PATH = TEMPLATE_DIR / "email_logo_ready.html"

app = FastAPI(title="Logo Post-Processing API")

resend.api_key = RESEND_API_KEY


def process_logo(input_path: str, output_dir: str) -> dict:
    """
    Process a logo image through the full pipeline.
    Returns a dictionary with paths to all generated files.

    Pipeline:
    1. Upscale (Real-ESRGAN 4x)
    2. Remove background (BiRefNet-HR)
    3. Trim transparent areas
    4. Vectorize (LAB-quantize + potrace + SVGO)
    5. Export to PDF/EPS
    """
    # Step 1: Upscale the image (Real-ESRGAN 4x)
    upscaled_path = os.path.join(output_dir, "0_upscaled.png")
    upscale(input_path, upscaled_path)

    # Step 2: Remove background (BiRefNet-HR)
    no_bg_path = os.path.join(output_dir, "1_no_bg.png")
    im = BiRefNetHRRemover().remove(upscaled_path, no_bg_path)

    # Step 3: Trim transparent areas
    im = trim_transparent(im, pad=24)
    clean_path = os.path.join(output_dir, "2_clean.png")
    im.save(clean_path, "PNG")

    # Step 4: Vectorize using LAB-quantize + potrace pipeline
    # This pipeline:
    # - Composites on white background
    # - Applies light denoising
    # - Quantizes to 6 colors using LAB K-Means
    # - Traces with potrace (smooth curves)
    # - Optimizes with SVGO
    svg_path = os.path.join(output_dir, "3_vectorized.svg")
    run_vectorization_pipeline(
        im,
        svg_path,
        n_colors=6,
        denoise=True,
        denoise_strength="light",
        use_svgo=True,
        alphamax=1.0,
        turdsize=10,
        opttolerance=0.2,
    )

    # Step 5: Export to PDF
    pdf_path = os.path.join(output_dir, "4_output.pdf")
    export_to_pdf(svg_path, pdf_path)

    # Step 6: Export to EPS
    eps_path = os.path.join(output_dir, "5_output.eps")
    export_to_eps(svg_path, eps_path)

    return {
        "upscaled": upscaled_path,
        "no_bg": no_bg_path,
        "clean": clean_path,
        "svg": svg_path,
        "pdf": pdf_path,
        "eps": eps_path,
    }


def _send_email(email: str, file_paths: dict) -> dict:
    """
    Send an email with all processed files as a zip attachment using Resend.
    Returns dict with email_id on success, raises Exception on failure.
    """
    if not resend.api_key:
        raise ValueError("RESEND_API_KEY environment variable is not set")

    # Create a zip file containing all processed files
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
        zip_path = tmp_zip.name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, path in file_paths.items():
            if os.path.exists(path):
                file_ext = Path(path).suffix
                zf.write(path, f"{name}{file_ext}")

    # Read and encode the zip file
    with open(zip_path, "rb") as f:
        zip_content = base64.b64encode(f.read()).decode()

    # Clean up temp zip file
    os.unlink(zip_path)

    attachments = [{
        "filename": "logos.zip",
        "content": zip_content,
    }]

    # Load the HTML template
    with open(EMAIL_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        html_content = f.read()

    params = {
        "from": f"Wer Studio <{EMAIL_FROM}>",
        "to": [email],
        "subject": "Tus Logos Están Listos — Wer Studio",
        "html": html_content,
        "attachments": attachments,
    }

    email_response = Emails.send(params)
    return email_response


def process_image(image_data: bytes, options: dict) -> dict:
    """
    Process a logo image from raw bytes. Called by Modal method.

    Args:
        image_data: Raw image bytes (PNG)
        options: Dict with optional keys:
            - email: str - Email address to send results to
            - send_email: bool - Whether to send email (default: True if email provided)

    Returns:
        dict with:
            - success: bool
            - message: str
            - email_id: str (if email was sent)
            - files: dict with base64-encoded file contents
    """
    email = options.get("email")
    send_email = options.get("send_email", bool(email))

    print(f"Send email is {send_email} and email is {email}")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Write input image to temp file
        input_path = os.path.join(temp_dir, "input.png")
        with open(input_path, "wb") as f:
            f.write(image_data)

        # Process the logo
        file_paths = process_logo(input_path, temp_dir)

        # Prepare response with base64-encoded files
        files = {}
        for name, path in file_paths.items():
            if os.path.exists(path):
                with open(path, "rb") as f:
                    files[name] = base64.b64encode(f.read()).decode()

        result = {
            "success": True,
            "message": "Logo processed successfully",
            "files": files,
            "files_generated": list(file_paths.keys()),
        }

        # Send email if requested
        if send_email and email:
            email_response = _send_email(email, file_paths)
            result["email_id"] = email_response.get("id") if email_response else None
            result["message"] = "Logo processed successfully and email sent"

        return result


@app.post("/process-logo")
async def process_logo_endpoint(
    logo: UploadFile = File(..., description="PNG logo file"),
    email: str = Form(..., description="Email address to send results to"),
):
    """
    Process a logo image and send the results via email.

    - **logo**: PNG image file
    - **email**: Email address where results will be sent
    """
    if not logo.content_type or not logo.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input.png")
        with open(input_path, "wb") as f:
            content = await logo.read()
            f.write(content)

        try:
            file_paths = process_logo(input_path, temp_dir)
            email_response = _send_email(email, file_paths)

            return JSONResponse(
                status_code=200,
                content={
                    "message": "Logo processed successfully and email sent",
                    "email_id": email_response.get("id") if email_response else None,
                    "files_generated": list(file_paths.keys()),
                }
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error: {str(e)}"
            )


@app.get("/")
async def root():
    return {"message": "Logo Post-Processing API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
