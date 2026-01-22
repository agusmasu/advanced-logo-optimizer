"""
High-quality vectorization pipeline using LAB-space quantization and potrace.

This module provides the production vectorization pipeline that:
1. Composites transparent images onto white background
2. Applies light denoising to clean upscaling artifacts
3. Quantizes colors using LAB-space K-Means for perceptually uniform clustering
4. Vectorizes with potrace using smooth curve settings
5. Optimizes with SVGO

Best for: AI-generated logos that need clean, professional vector output.
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans


def composite_on_white(image: np.ndarray) -> np.ndarray:
    """
    Composite an RGBA image onto a pure white background.

    This ensures a clean, uniform background without AI-generated
    color variations or artifacts.

    Args:
        image: RGBA numpy array (must have alpha channel)

    Returns:
        RGB numpy array composited on white
    """
    if image.shape[2] != 4:
        # No alpha channel, return as-is
        return image

    rgb = image[:, :, :3].astype(np.float32)
    alpha = image[:, :, 3].astype(np.float32) / 255.0

    # White background
    white = np.ones_like(rgb) * 255.0

    # Alpha compositing: result = foreground * alpha + background * (1 - alpha)
    alpha_3ch = alpha[:, :, np.newaxis]
    composited = rgb * alpha_3ch + white * (1 - alpha_3ch)

    return composited.astype(np.uint8)


def light_denoise(image: np.ndarray, strength: str = "light") -> np.ndarray:
    """
    Apply light denoising to clean upscaling artifacts while preserving edges.

    Args:
        image: RGB or RGBA numpy array
        strength: "light", "medium", or "strong"

    Returns:
        Denoised numpy array
    """
    # Handle RGBA by processing RGB and keeping alpha
    if image.shape[2] == 4:
        rgb = image[:, :, :3]
        alpha = image[:, :, 3]
    else:
        rgb = image
        alpha = None

    # Bilateral filter settings based on strength
    # Bilateral filter smooths while preserving edges
    settings = {
        "light": (5, 30, 30),      # d, sigmaColor, sigmaSpace
        "medium": (7, 50, 50),
        "strong": (9, 75, 75),
    }

    d, sigma_color, sigma_space = settings.get(strength, settings["light"])

    # Apply bilateral filter (preserves edges better than Gaussian)
    denoised = cv2.bilateralFilter(rgb, d, sigma_color, sigma_space)

    if alpha is not None:
        return np.dstack([denoised, alpha])
    return denoised


def quantize_colors_kmeans(
    image: np.ndarray,
    n_colors: int = 6,
    white_bg: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize image colors using K-Means clustering in LAB color space.

    Using LAB (perceptually uniform) instead of RGB ensures that colors
    which look similar to humans are grouped together, even if their
    RGB values differ significantly.

    Args:
        image: RGB or RGBA numpy array
        n_colors: Number of colors to quantize to
        white_bg: If True, treat white as background (exclude from clustering)

    Returns:
        Tuple of (quantized image, unique colors array in RGB)
    """
    # Handle RGBA
    if image.shape[2] == 4:
        rgb = image[:, :, :3]
        alpha = image[:, :, 3]
        mask = alpha > 0
    else:
        rgb = image
        alpha = None
        mask = np.ones(rgb.shape[:2], dtype=bool)

    # If white background, exclude near-white pixels from clustering
    if white_bg:
        # Pixels that are very bright (close to white) are background
        brightness = np.mean(rgb, axis=2)
        mask = mask & (brightness < 250)

    # Get visible pixels for clustering
    visible_pixels = rgb[mask]

    if len(visible_pixels) == 0:
        # Return original if no visible content
        return image, np.array([])

    # Convert RGB to LAB for perceptually uniform clustering
    # cv2.cvtColor needs shape (1, N, 3) for a list of pixels
    visible_rgb_reshaped = visible_pixels.reshape(1, -1, 3).astype(np.uint8)
    visible_lab = cv2.cvtColor(visible_rgb_reshaped, cv2.COLOR_RGB2LAB).reshape(-1, 3)

    # K-Means clustering in LAB space
    kmeans = MiniBatchKMeans(
        n_clusters=n_colors,
        random_state=42,
        n_init=10,
        max_iter=300,
    )

    labels = kmeans.fit_predict(visible_lab)
    lab_centers = kmeans.cluster_centers_

    # Convert LAB centers back to RGB
    lab_centers_reshaped = lab_centers.reshape(1, -1, 3).astype(np.uint8)
    rgb_centers_reshaped = cv2.cvtColor(lab_centers_reshaped, cv2.COLOR_LAB2RGB)
    centers = rgb_centers_reshaped.reshape(-1, 3)

    # Reconstruct quantized image
    quantized = np.full_like(rgb, 255)  # Start with white background
    quantized[mask] = centers[labels]

    if alpha is not None:
        return np.dstack([quantized, alpha]), centers
    return quantized, centers


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


def vectorize_with_potrace(
    quantized_image: np.ndarray,
    colors: np.ndarray,
    output_path: str,
    alphamax: float = 1.0,
    turdsize: int = 10,
    opttolerance: float = 0.2,
) -> str:
    """
    Vectorize a quantized image using potrace.

    Traces each color layer separately and combines into single SVG.

    Args:
        quantized_image: Quantized RGB/RGBA numpy array
        colors: Array of unique colors (from quantization)
        output_path: Output SVG path
        alphamax: Corner smoothness (1.0 = smooth curves)
        turdsize: Remove paths smaller than this (removes noise)
        opttolerance: Curve optimization tolerance (lower = more detail)

    Returns:
        Path to output SVG
    """
    # Handle RGBA
    if quantized_image.shape[2] == 4:
        rgb = quantized_image[:, :, :3]
        alpha = quantized_image[:, :, 3]
        mask = alpha > 0
    else:
        rgb = quantized_image
        mask = np.ones(rgb.shape[:2], dtype=bool)
        # Exclude white background from mask
        brightness = np.mean(rgb, axis=2)
        mask = brightness < 250

    h, w = rgb.shape[:2]

    # Sort colors by frequency (largest area first for proper layering)
    color_counts = []
    for color in colors:
        count = np.sum(np.all(rgb == color, axis=2) & mask)
        color_counts.append(count)

    color_order = np.argsort(color_counts)[::-1]  # Largest first

    svg_groups = []
    svg_header = None
    svg_transform = None

    with tempfile.TemporaryDirectory() as temp_dir:
        for idx, color_idx in enumerate(color_order):
            color = colors[color_idx]

            # Skip near-white colors (background) - don't trace them
            # Check if all RGB channels are > 240 (very close to white)
            if np.all(color > 240):
                continue

            # Create binary mask for this color
            color_mask = np.all(rgb == color, axis=2) & mask

            if not color_mask.any():
                continue

            # Black where color exists, white elsewhere (potrace traces black)
            bw = np.where(color_mask, 0, 255).astype(np.uint8)

            # Save as PBM
            pbm_path = Path(temp_dir) / f"layer_{idx}.pbm"
            _save_pbm(bw, str(pbm_path))

            # Trace with potrace
            svg_layer_path = Path(temp_dir) / f"layer_{idx}.svg"
            cmd = [
                "potrace",
                str(pbm_path),
                "-s",  # SVG output
                "-o", str(svg_layer_path),
                "-a", str(alphamax),       # Corner smoothness
                "-t", str(turdsize),       # Remove tiny noise paths
                "-O", str(opttolerance),   # Curve optimization
            ]

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                continue

            # Read SVG and extract paths
            with open(svg_layer_path, "r") as f:
                svg_content = f.read()

            # Extract header info from first layer
            if svg_header is None:
                svg_match = re.search(r'<svg[^>]*>', svg_content)
                if svg_match:
                    svg_header = svg_match.group(0)

                transform_match = re.search(r'<g transform="([^"]*)"', svg_content)
                if transform_match:
                    svg_transform = transform_match.group(1)

            # Extract paths and color them
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
            paths = re.findall(r'<path[^>]*d="([^"]*)"[^>]*/>', svg_content)

            if paths:
                path_elements = '\n'.join([f'<path d="{path_d}"/>' for path_d in paths])
                svg_groups.append(f'<g fill="{hex_color}" stroke="none">\n{path_elements}\n</g>')

    # Construct final SVG
    if svg_header and svg_transform:
        final_svg = f'''{svg_header}
<metadata>Created by potrace pipeline (quantized + smooth tracing)</metadata>
<g transform="{svg_transform}">
{chr(10).join(svg_groups)}
</g>
</svg>'''
    else:
        final_svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<metadata>Created by potrace pipeline (quantized + smooth tracing)</metadata>
{chr(10).join(svg_groups)}
</svg>'''

    with open(output_path, "w") as f:
        f.write(final_svg)

    return output_path


def optimize_with_svgo(input_svg: str, output_svg: str) -> bool:
    """
    Optimize SVG using SVGO.

    Returns True if optimization succeeded, False otherwise.
    """
    # Check if svgo is available
    if not shutil.which("svgo"):
        # Copy input to output unchanged
        shutil.copy(input_svg, output_svg)
        return False

    cmd = [
        "svgo",
        input_svg,
        "-o", output_svg,
        "--multipass",  # Multiple optimization passes
        "--precision=2",  # Reduce coordinate precision (smaller file)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        shutil.copy(input_svg, output_svg)
        return False

    return True


def run_vectorization_pipeline(
    input_image: Image.Image,
    output_svg_path: str,
    n_colors: int = 6,
    denoise: bool = True,
    denoise_strength: str = "light",
    use_svgo: bool = True,
    alphamax: float = 1.0,
    turdsize: int = 10,
    opttolerance: float = 0.2,
) -> str:
    """
    Run the full vectorization pipeline on a PIL Image.

    This is the main entry point for the production pipeline.

    Args:
        input_image: PIL Image (RGBA, from background removal)
        output_svg_path: Path to save the final SVG
        n_colors: Number of colors for quantization (default: 6)
        denoise: Whether to apply denoising (default: True)
        denoise_strength: Denoise strength ("light", "medium", "strong")
        use_svgo: Whether to optimize with SVGO (default: True)
        alphamax: Potrace corner smoothness (1.0 = smooth)
        turdsize: Potrace noise removal threshold
        opttolerance: Potrace curve optimization tolerance

    Returns:
        Path to the output SVG file
    """
    # Convert PIL Image to numpy array
    image = np.array(input_image.convert("RGBA"))

    # Step 1: Composite on white background
    image = composite_on_white(image)

    # Step 2: Light denoise
    if denoise:
        image = light_denoise(image, strength=denoise_strength)

    # Step 3: Color quantization (LAB-space K-Means)
    quantized, colors = quantize_colors_kmeans(image, n_colors=n_colors, white_bg=True)

    if len(colors) == 0:
        # Empty image, create minimal SVG
        h, w = image.shape[:2]
        with open(output_svg_path, "w") as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"></svg>')
        return output_svg_path

    # Step 4: Vectorize with potrace
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
        raw_svg_path = tmp.name

    vectorize_with_potrace(
        quantized,
        colors,
        raw_svg_path,
        alphamax=alphamax,
        turdsize=turdsize,
        opttolerance=opttolerance,
    )

    # Step 5: Optimize with SVGO
    if use_svgo:
        optimize_with_svgo(raw_svg_path, output_svg_path)
        # Clean up temp file
        Path(raw_svg_path).unlink(missing_ok=True)
    else:
        shutil.move(raw_svg_path, output_svg_path)

    return output_svg_path
