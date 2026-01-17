import base64
import os
import tempfile
from pathlib import Path

import resend
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from resend import Emails

from src.config import EMAIL_FROM, RESEND_API_KEY
from src.models.birefnet_hr_model import BiRefNetHRRemover
from src.processors.cleanup import quantize, trim_transparent
from src.processors.exporter import export_to_eps, export_to_pdf, optimize_svg
from src.processors.upscaler import upscale
from src.processors.vectorizer import vectorize_color

app = FastAPI(title="Logo Post-Processing API")

resend.api_key = RESEND_API_KEY


def process_logo(input_path: str, output_dir: str) -> dict:
    """
    Process a logo image through the full pipeline.
    Returns a dictionary with paths to all generated files.
    """
    # Step 1: Upscale the image (Real-ESRGAN 4x)
    upscaled_path = os.path.join(output_dir, "0_upscaled.png")
    im = upscale(input_path, upscaled_path)

    # Step 2: Remove background (BiRefNet-HR - best for fine details)
    no_bg_path = os.path.join(output_dir, "1_no_bg.png")
    remover = BiRefNetHRRemover()
    im = remover.remove(upscaled_path, no_bg_path)

    # Step 3: Trim transparent areas
    im = trim_transparent(im, pad=24)

    # Step 4: Quantize colors for cleaner vectors
    im = quantize(im, max_colors=12)
    clean_path = os.path.join(output_dir, "2_clean.png")
    im.save(clean_path, "PNG")

    # Step 5: Vectorize using vtracer (preserves colors)
    svg_path = os.path.join(output_dir, "3_vector.svg")
    vectorize_color(clean_path, svg_path)

    # Step 6: Optimize SVG using scour
    svg_optimized_path = os.path.join(output_dir, "4_optimized.svg")
    optimize_svg(svg_path, svg_optimized_path)

    # Step 7: Export to PDF
    pdf_path = os.path.join(output_dir, "5_output.pdf")
    export_to_pdf(svg_optimized_path, pdf_path)

    # Step 8: Export to EPS
    eps_path = os.path.join(output_dir, "6_output.eps")
    export_to_eps(svg_optimized_path, eps_path)

    return {
        "upscaled": upscaled_path,
        "no_bg": no_bg_path,
        "clean": clean_path,
        "svg": svg_path,
        "svg_optimized": svg_optimized_path,
        "pdf": pdf_path,
        "eps": eps_path,
    }


def send_email_with_attachments(email: str, file_paths: dict):
    """
    Send an email with all processed files as attachments using Resend.
    """
    if not resend.api_key:
        raise HTTPException(
            status_code=500,
            detail="RESEND_API_KEY environment variable is not set"
        )

    attachments = []
    for name, path in file_paths.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                file_content = f.read()
                encoded_content = base64.b64encode(file_content).decode()
                file_ext = Path(path).suffix
                attachments.append({
                    "filename": f"{name}{file_ext}",
                    "content": encoded_content,
                })

    try:
        params = {
            "from": EMAIL_FROM,
            "to": [email],
            "subject": "Your Logo Processing Results",
            "html": """
            <h2>Logo Processing Complete</h2>
            <p>Your logo has been processed successfully. Please find all the generated files attached.</p>
            <ul>
                <li>Background removed (PNG)</li>
                <li>Cleaned and quantized (PNG)</li>
                <li>Vector SVG (optimized)</li>
                <li>PDF export</li>
                <li>EPS export</li>
            </ul>
            <p>Thank you for using our service!</p>
            """,
            "attachments": attachments,
        }

        email_response = Emails.send(params)
        return email_response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send email: {str(e)}"
        )


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
            email_response = send_email_with_attachments(email, file_paths)

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
