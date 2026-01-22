"""
Illustrator-style vectorization that mimics Adobe Illustrator's Image Trace.

Adobe Illustrator's Image Trace (3-color mode) uses:
1. K-Means clustering for color quantization (not median cut)
2. Gaussian blur preprocessing for noise reduction
3. Layer separation - each color traced separately
4. Potrace-based Bezier curve fitting with corner detection
5. Stacked/layered output combining all colors

This produces cleaner results than simple quantization because K-Means
finds optimal color clusters rather than just splitting the color space.
"""

import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans


def vectorize_illustrator_3color(
    input_path: str,
    output_path: str,
    n_colors: int = 3,
    blur_sigma: float = 0.5,
    alphamax: float = 0.5,
    turdsize: int = 2,
    morphological_cleanup: bool = True,
) -> str:
    """
    Vectorize an image using Illustrator-style 3-color tracing.

    This mimics Adobe Illustrator's Image Trace with limited colors:
    - Uses K-Means clustering for better color quantization
    - Applies Gaussian blur to reduce noise
    - Traces each color layer with Potrace
    - Combines layers into a single SVG

    Args:
        input_path: Path to the input PNG image (with transparency)
        output_path: Path to save the output SVG
        n_colors: Number of colors to quantize to (default 3)
        blur_sigma: Gaussian blur sigma for preprocessing (0 = no blur)
        alphamax: Potrace corner threshold (0 = sharp, 1.33 = smooth)
        turdsize: Minimum area to trace (despeckle)
        morphological_cleanup: Apply morphological operations to clean masks

    Returns:
        Path to the output SVG file
    """
    # Load image
    img = Image.open(input_path).convert("RGBA")
    arr = np.array(img)

    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]
    mask = alpha > 0

    # Get visible pixels
    visible_pixels = rgb[mask]

    if len(visible_pixels) == 0:
        # Empty image
        svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"></svg>'
        with open(output_path, "w") as f:
            f.write(svg_content)
        return output_path

    # 1. Preprocess: Apply Gaussian blur for noise reduction
    if blur_sigma > 0:
        rgb_blurred = cv2.GaussianBlur(rgb, (0, 0), blur_sigma)
        # Only blur visible pixels
        rgb = np.where(mask[:, :, None], rgb_blurred, rgb)

    # 2. K-Means color quantization (Illustrator-style)
    # This is better than PIL's median cut for finding optimal color clusters
    kmeans = MiniBatchKMeans(
        n_clusters=n_colors,
        random_state=42,
        n_init=10,
        max_iter=300,
    )

    # Fit on visible pixels only
    visible_rgb = rgb[mask]
    labels = kmeans.fit_predict(visible_rgb)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # Reconstruct quantized image
    quantized_rgb = np.zeros_like(rgb)
    quantized_rgb[mask] = centers[labels]

    # 3. Trace each color layer with Potrace
    svg_paths = []

    # Sort colors by frequency (most common first = background-like)
    # This helps with proper layering
    color_counts = np.bincount(labels, minlength=n_colors)
    color_order = np.argsort(-color_counts)  # Descending

    with tempfile.TemporaryDirectory() as temp_dir:
        for color_idx in color_order:
            color = centers[color_idx]

            # Create binary mask for this color
            color_mask = np.all(quantized_rgb == color, axis=2) & mask

            if not color_mask.any():
                continue

            # Black where color exists, white elsewhere (potrace traces black)
            bw = np.where(color_mask, 0, 255).astype(np.uint8)

            # 4. Optional morphological cleanup (reduces noise in masks)
            if morphological_cleanup:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                # Opening removes small noise
                bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
                # Closing fills small holes
                bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

            # Save as PBM
            pbm_path = Path(temp_dir) / f"layer_{color_idx}.pbm"
            _save_pbm(bw, str(pbm_path))

            # Trace with Potrace
            svg_layer_path = Path(temp_dir) / f"layer_{color_idx}.svg"
            cmd = [
                "potrace",
                str(pbm_path),
                "-s",  # SVG output
                "-o", str(svg_layer_path),
                "-a", str(alphamax),  # Corner threshold
                "-t", str(turdsize),  # Despeckle
            ]

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                continue

            # Read SVG and extract paths
            with open(svg_layer_path, "r") as f:
                svg_content = f.read()

            # Extract path data and add color
            paths = re.findall(r'<path[^>]*d="([^"]*)"[^>]*/>', svg_content)
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

            for path_d in paths:
                svg_paths.append(f'<path d="{path_d}" fill="{hex_color}" stroke="none"/>')

    # 5. Combine all paths into final SVG
    final_svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
{chr(10).join(svg_paths)}
</svg>'''

    with open(output_path, "w") as f:
        f.write(final_svg)

    return output_path


def vectorize_illustrator_6color(
    input_path: str,
    output_path: str,
) -> str:
    """
    Vectorize with 6 colors - similar to Illustrator's "6 Colors" preset.
    """
    return vectorize_illustrator_3color(
        input_path,
        output_path,
        n_colors=6,
        blur_sigma=0.3,
        alphamax=0.8,
        turdsize=4,
    )


def vectorize_illustrator_16color(
    input_path: str,
    output_path: str,
) -> str:
    """
    Vectorize with 16 colors - similar to Illustrator's "16 Colors" preset.
    """
    return vectorize_illustrator_3color(
        input_path,
        output_path,
        n_colors=16,
        blur_sigma=0.2,
        alphamax=1.0,
        turdsize=4,
    )


def vectorize_illustrator_sharp(
    input_path: str,
    output_path: str,
    n_colors: int = 3,
) -> str:
    """
    Vectorize with maximum sharpness (no blur, sharp corners).
    Best for logos with geometric shapes.
    """
    return vectorize_illustrator_3color(
        input_path,
        output_path,
        n_colors=n_colors,
        blur_sigma=0.0,  # No blur
        alphamax=0.0,    # Maximum sharpness
        turdsize=0,      # Keep all details
        morphological_cleanup=False,
    )


def _save_pbm(bw_array: np.ndarray, output_path: str) -> str:
    """Save a binary array as PBM (portable bitmap) format."""
    h, w = bw_array.shape

    # Convert to 1-bit (0 = white, 1 = black for PBM)
    bits = (bw_array < 128).astype(np.uint8)

    # PBM P4 format (binary)
    header = f"P4\n{w} {h}\n".encode("ascii")

    # Pack bits into bytes (8 pixels per byte)
    packed = np.packbits(bits, axis=1)

    with open(output_path, "wb") as f:
        f.write(header + packed.tobytes())

    return output_path
