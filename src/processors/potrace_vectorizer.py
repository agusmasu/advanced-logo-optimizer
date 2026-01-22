"""
Potrace-based vectorization for pixel-perfect sharp edges.

Potrace uses an O(n²) global optimization algorithm that finds the best way
to trace a shape, making it ideal for logos and fonts where sharp corners matter.

Best for: Monochrome logos, text/fonts, and images where sharp corners are critical.
Limitation: Only supports monochrome - for color images, each color layer must be traced separately.
"""

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def vectorize_potrace_sharp(
    input_path: str,
    output_path: str,
    alphamax: float = 0.0,
    opticurve: bool = False,
    turdsize: int = 0,
) -> str:
    """
    Convert an image to SVG using potrace with sharp corner settings.

    This configuration produces pixel-perfect sharp corners, ideal for text and logos.

    Args:
        input_path: Path to the input PNG image (with transparency)
        output_path: Path to save the output SVG
        alphamax: Corner threshold (0.0 = sharp polygon corners, 1.33 = smooth curves)
        opticurve: Whether to optimize curves (False = preserve all details)
        turdsize: Minimum area to trace (0 = trace everything, including small details)

    Returns:
        Path to the output SVG file
    """
    # Convert to PBM (portable bitmap) format that potrace accepts
    pbm_path = _to_pbm(input_path)

    # Build potrace command
    cmd = [
        "potrace",
        pbm_path,
        "-s",  # Output SVG
        "-o", output_path,
        "-a", str(alphamax),  # Corner threshold
        "-t", str(turdsize),  # Turd size (despeckle)
    ]

    if not opticurve:
        cmd.append("-n")  # Disable curve optimization

    subprocess.run(cmd, check=True)

    # Clean up temp file
    Path(pbm_path).unlink(missing_ok=True)

    return output_path


def vectorize_potrace_smooth(
    input_path: str,
    output_path: str,
    alphamax: float = 1.0,
    opticurve: bool = True,
    opttolerance: float = 0.2,
    turdsize: int = 2,
) -> str:
    """
    Convert an image to SVG using potrace with smooth curve settings.

    This configuration produces smoother curves while still maintaining good quality.

    Args:
        input_path: Path to the input PNG image (with transparency)
        output_path: Path to save the output SVG
        alphamax: Corner threshold (higher = smoother curves)
        opticurve: Whether to optimize curves
        opttolerance: Curve optimization tolerance
        turdsize: Minimum area to trace

    Returns:
        Path to the output SVG file
    """
    pbm_path = _to_pbm(input_path)

    cmd = [
        "potrace",
        pbm_path,
        "-s",
        "-o", output_path,
        "-a", str(alphamax),
        "-t", str(turdsize),
    ]

    if opticurve:
        cmd.extend(["-O", str(opttolerance)])
    else:
        cmd.append("-n")

    subprocess.run(cmd, check=True)

    Path(pbm_path).unlink(missing_ok=True)

    return output_path


def vectorize_potrace_color(
    input_path: str,
    output_path: str,
    max_colors: int = 12,
    alphamax: float = 0.5,
    turdsize: int = 2,
) -> str:
    """
    Convert a color image to SVG by tracing each color layer separately with potrace.

    This approach traces each color as a separate layer, then combines them into one SVG.

    Args:
        input_path: Path to the input PNG image
        output_path: Path to save the output SVG
        max_colors: Maximum number of colors to extract
        alphamax: Corner threshold for potrace
        turdsize: Minimum area to trace

    Returns:
        Path to the output SVG file
    """
    # Load image
    image = Image.open(input_path).convert("RGBA")
    arr = np.array(image)

    # Get unique colors (ignoring fully transparent pixels)
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]

    # Mask out transparent pixels
    mask = alpha > 0
    visible_pixels = rgb[mask]

    if len(visible_pixels) == 0:
        # Empty image, create empty SVG
        h, w = arr.shape[:2]
        svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"></svg>'
        with open(output_path, "w") as f:
            f.write(svg_content)
        return output_path

    # Quantize colors
    from src.processors.cleanup import quantize
    quantized = quantize(image, max_colors=max_colors)
    q_arr = np.array(quantized)

    # Get unique colors from quantized image
    q_alpha = q_arr[:, :, 3]
    q_rgb = q_arr[:, :, :3]
    q_mask = q_alpha > 0

    visible_q_pixels = q_rgb[q_mask]
    unique_colors = np.unique(visible_q_pixels.reshape(-1, 3), axis=0)

    h, w = arr.shape[:2]
    svg_paths = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for i, color in enumerate(unique_colors):
            # Create binary mask for this color
            color_mask = np.all(q_rgb == color, axis=2) & q_mask
            bw = np.where(color_mask, 0, 255).astype(np.uint8)  # Black on white

            # Save as PBM
            pbm_path = Path(temp_dir) / f"layer_{i}.pbm"
            _save_pbm(bw, str(pbm_path))

            # Trace with potrace
            svg_layer_path = Path(temp_dir) / f"layer_{i}.svg"
            cmd = [
                "potrace",
                str(pbm_path),
                "-s",
                "-o", str(svg_layer_path),
                "-a", str(alphamax),
                "-t", str(turdsize),
            ]
            subprocess.run(cmd, check=True)

            # Read SVG and extract paths
            with open(svg_layer_path, "r") as f:
                svg_content = f.read()

            # Extract path data and add color
            import re
            paths = re.findall(r'<path[^>]*d="([^"]*)"[^>]*/>', svg_content)
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

            for path_d in paths:
                svg_paths.append(f'<path d="{path_d}" fill="{hex_color}" stroke="none"/>')

    # Combine all paths into final SVG
    final_svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
{chr(10).join(svg_paths)}
</svg>'''

    with open(output_path, "w") as f:
        f.write(final_svg)

    return output_path


def vectorize_potrace_smooth_color(
    input_path: str,
    output_path: str,
    max_colors: int = 3,
    alphamax: float = 1.0,
    opticurve: bool = True,
    opttolerance: float = 0.2,
    turdsize: int = 2,
) -> str:
    """
    Convert a color image to SVG using the same smooth settings as potrace_smooth.

    This applies the high-quality smooth B&W tracing to each color layer separately,
    producing results comparable to the excellent B&W smooth output but with colors.

    Args:
        input_path: Path to the input PNG image
        output_path: Path to save the output SVG
        max_colors: Number of colors to quantize to (default 3)
        alphamax: Corner threshold (1.0 = smooth curves, same as potrace_smooth)
        opticurve: Whether to optimize curves (True = smoother output)
        opttolerance: Curve optimization tolerance
        turdsize: Minimum area to trace (despeckle)

    Returns:
        Path to the output SVG file
    """
    from sklearn.cluster import MiniBatchKMeans

    # Load image
    image = Image.open(input_path).convert("RGBA")
    arr = np.array(image)

    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]
    mask = alpha > 0

    visible_pixels = rgb[mask]

    if len(visible_pixels) == 0:
        svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"></svg>'
        with open(output_path, "w") as f:
            f.write(svg_content)
        return output_path

    # K-Means color quantization (better than median cut)
    kmeans = MiniBatchKMeans(
        n_clusters=max_colors,
        random_state=42,
        n_init=10,
        max_iter=300,
    )

    visible_rgb = rgb[mask]
    labels = kmeans.fit_predict(visible_rgb)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # Reconstruct quantized image
    quantized_rgb = np.zeros_like(rgb)
    quantized_rgb[mask] = centers[labels]

    # Sort colors by frequency (largest area first = background)
    color_counts = np.bincount(labels, minlength=max_colors)
    color_order = np.argsort(-color_counts)

    svg_paths = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for color_idx in color_order:
            color = centers[color_idx]

            # Create binary mask for this color
            color_mask = np.all(quantized_rgb == color, axis=2) & mask

            if not color_mask.any():
                continue

            # Black where color exists, white elsewhere
            bw = np.where(color_mask, 0, 255).astype(np.uint8)

            # Save as PBM
            pbm_path = Path(temp_dir) / f"layer_{color_idx}.pbm"
            _save_pbm(bw, str(pbm_path))

            # Trace with potrace using SMOOTH settings (same as vectorize_potrace_smooth)
            svg_layer_path = Path(temp_dir) / f"layer_{color_idx}.svg"
            cmd = [
                "potrace",
                str(pbm_path),
                "-s",
                "-o", str(svg_layer_path),
                "-a", str(alphamax),
                "-t", str(turdsize),
            ]

            # Add curve optimization (key difference from potrace_color)
            if opticurve:
                cmd.extend(["-O", str(opttolerance)])

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                continue

            # Read SVG and extract paths
            with open(svg_layer_path, "r") as f:
                svg_content = f.read()

            import re
            paths = re.findall(r'<path[^>]*d="([^"]*)"[^>]*/>', svg_content)
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

            for path_d in paths:
                svg_paths.append(f'<path d="{path_d}" fill="{hex_color}" stroke="none"/>')

    # Combine all paths into final SVG
    # Use inches for reasonable size in Illustrator (default 10" on longest side)
    max_dim = max(w, h)
    target_inches = 10
    width_in = (w / max_dim) * target_inches
    height_in = (h / max_dim) * target_inches

    final_svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width_in}in" height="{height_in}in" viewBox="0 0 {w} {h}">
{chr(10).join(svg_paths)}
</svg>'''

    with open(output_path, "w") as f:
        f.write(final_svg)

    return output_path


def _to_pbm(input_path: str) -> str:
    """
    Convert an RGBA image to PBM format for potrace.

    Creates a binary black/white image from the alpha channel or luminance.
    """
    image = Image.open(input_path).convert("RGBA")
    arr = np.array(image)

    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    # Composite onto white background
    white = np.ones_like(rgb) * 255
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    comp = (rgb * a + white * (1 - a)).astype(np.uint8)

    # Convert to grayscale
    gray = cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY)

    # Threshold to binary
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Ensure "black shapes on white" (potrace traces black areas)
    if bw.mean() < 127:
        bw = 255 - bw

    # Save as PBM
    with tempfile.NamedTemporaryFile(suffix=".pbm", delete=False) as f:
        pbm_path = f.name

    _save_pbm(bw, pbm_path)

    return pbm_path


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
