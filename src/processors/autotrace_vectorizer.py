"""
AutoTrace-based vectorization.

AutoTrace is an alternative tracing algorithm that supports both outline
and centerline tracing, with color reduction capabilities.

Best for: Batch processing, images where centerline tracing is needed,
alternative algorithm when vtracer/potrace don't work well.

Note: Requires pyautotrace to be installed: pip install pyautotrace[standard]
"""

import numpy as np
from PIL import Image


def vectorize_autotrace(
    input_path: str,
    output_path: str,
    color_count: int = 16,
    corner_threshold: float = 100.0,
    corner_always_threshold: float = 60.0,
    corner_surround: int = 4,
    error_threshold: float = 2.0,
    filter_iterations: int = 4,
    line_threshold: float = 1.0,
    despeckle_level: int = 0,
    despeckle_tightness: float = 2.0,
) -> str:
    """
    Convert an image to SVG using AutoTrace.

    Args:
        input_path: Path to the input PNG image
        output_path: Path to save the output SVG
        color_count: Number of colors to reduce to (default 16)
        corner_threshold: Angle threshold for corners in degrees (default 100)
        corner_always_threshold: Angle always considered a corner (default 60)
        corner_surround: Pixels to consider for corner detection (default 4)
        error_threshold: Curve fitting error threshold (default 2.0)
        filter_iterations: Smoothing filter iterations (default 4)
        line_threshold: Lines straighter than this become straight (default 1.0)
        despeckle_level: Remove specks up to this size (default 0)
        despeckle_tightness: Despeckle tightness (default 2.0)

    Returns:
        Path to the output SVG file
    """
    try:
        from autotrace import Bitmap, VectorFormat
    except ImportError:
        raise ImportError(
            "pyautotrace is not installed. Install with: pip install pyautotrace[standard]"
        )

    # Load image and convert to RGB
    image = Image.open(input_path).convert("RGBA")

    # Composite onto white background for autotrace
    arr = np.array(image)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    white = np.ones_like(rgb) * 255
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    comp = (rgb * a + white * (1 - a)).astype(np.uint8)

    # Create bitmap
    bitmap = Bitmap(comp)

    # Trace with parameters
    vector = bitmap.trace(
        color_count=color_count,
        corner_threshold=corner_threshold,
        corner_always_threshold=corner_always_threshold,
        corner_surround=corner_surround,
        error_threshold=error_threshold,
        filter_iterations=filter_iterations,
        line_threshold=line_threshold,
        despeckle_level=despeckle_level,
        despeckle_tightness=despeckle_tightness,
    )

    # Save as SVG
    vector.save(output_path)

    return output_path


def vectorize_autotrace_sharp(
    input_path: str,
    output_path: str,
    color_count: int = 16,
) -> str:
    """
    Convert an image to SVG using AutoTrace with sharp corner settings.

    Args:
        input_path: Path to the input PNG image
        output_path: Path to save the output SVG
        color_count: Number of colors to reduce to

    Returns:
        Path to the output SVG file
    """
    return vectorize_autotrace(
        input_path,
        output_path,
        color_count=color_count,
        corner_threshold=60.0,  # Lower = more corners detected
        corner_always_threshold=45.0,  # Lower = more strict corner detection
        error_threshold=1.0,  # Lower = more accurate curves
        filter_iterations=2,  # Less smoothing
        despeckle_level=0,  # Keep all details
    )


def vectorize_autotrace_smooth(
    input_path: str,
    output_path: str,
    color_count: int = 16,
) -> str:
    """
    Convert an image to SVG using AutoTrace with smooth curve settings.

    Args:
        input_path: Path to the input PNG image
        output_path: Path to save the output SVG
        color_count: Number of colors to reduce to

    Returns:
        Path to the output SVG file
    """
    return vectorize_autotrace(
        input_path,
        output_path,
        color_count=color_count,
        corner_threshold=120.0,  # Higher = fewer corners, more curves
        corner_always_threshold=80.0,
        error_threshold=3.0,  # Higher = smoother curves
        filter_iterations=6,  # More smoothing
        despeckle_level=2,  # Remove small specks
    )


def vectorize_autotrace_centerline(
    input_path: str,
    output_path: str,
) -> str:
    """
    Convert an image to SVG using AutoTrace centerline tracing.

    Centerline tracing traces the center of strokes rather than their outlines,
    useful for thin lines and handwritten text.

    Note: This uses the command-line autotrace if available, as centerline
    may not be exposed in all Python bindings.

    Args:
        input_path: Path to the input PNG image
        output_path: Path to save the output SVG

    Returns:
        Path to the output SVG file
    """
    import subprocess
    import tempfile
    from pathlib import Path

    # Convert to BMP for autotrace CLI
    image = Image.open(input_path).convert("RGBA")
    arr = np.array(image)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    white = np.ones_like(rgb) * 255
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    comp = (rgb * a + white * (1 - a)).astype(np.uint8)

    with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as f:
        bmp_path = f.name

    Image.fromarray(comp).save(bmp_path, "BMP")

    try:
        # Try using autotrace CLI with centerline option
        subprocess.run(
            [
                "autotrace",
                "-centerline",
                "-output-format", "svg",
                "-output-file", output_path,
                bmp_path,
            ],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to regular tracing if centerline not available
        Path(bmp_path).unlink(missing_ok=True)
        return vectorize_autotrace(input_path, output_path)

    Path(bmp_path).unlink(missing_ok=True)

    return output_path
