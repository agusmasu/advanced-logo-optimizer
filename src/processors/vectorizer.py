import subprocess

import cv2
import numpy as np
import vtracer
from PIL import Image


def vectorize_color(
    input_path: str,
    output_path: str,
    filter_speckle: int = 4,
    color_precision: int = 6,
) -> str:
    """
    Convert an image to a color SVG using vtracer.

    Args:
        input_path: Path to the input PNG image
        output_path: Path to save the output SVG
        filter_speckle: Filter out small artifacts (default 4)
        color_precision: Color precision level (default 6)

    Returns:
        Path to the output SVG file
    """
    vtracer.convert_image_to_svg_py(
        input_path,
        output_path,
        colormode="color",
        filter_speckle=filter_speckle,
        color_precision=color_precision,
    )
    return output_path


def vectorize_bw(input_path: str, output_path: str) -> str:
    """
    Convert an image to a black & white SVG using vtracer.

    Args:
        input_path: Path to the input PNG image
        output_path: Path to save the output SVG

    Returns:
        Path to the output SVG file
    """
    vtracer.convert_image_to_svg_py(
        input_path,
        output_path,
        colormode="binary",
    )
    return output_path


def vectorize_potrace(pgm_path: str, output_path: str) -> str:
    """
    Convert a PGM image to SVG using potrace.

    Args:
        pgm_path: Path to the input PGM image
        output_path: Path to save the output SVG

    Returns:
        Path to the output SVG file
    """
    subprocess.run(["potrace", pgm_path, "-s", "-o", output_path], check=True)
    return output_path


def to_pgm(im: Image.Image, output_path: str) -> str:
    """
    Convert an RGBA image to PGM format for use with potrace.

    Args:
        im: PIL Image to convert
        output_path: Path to save the PGM file

    Returns:
        Path to the output PGM file
    """
    arr = np.array(im)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    # Composite onto white background
    white = np.ones_like(rgb) * 255
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    comp = (rgb * a + white * (1 - a)).astype(np.uint8)

    gray = cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY)

    # Threshold
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Ensure "black shapes on white"
    if bw.mean() < 127:
        bw = 255 - bw

    h, w = bw.shape
    header = f"P5\n{w} {h}\n255\n".encode("ascii")

    with open(output_path, "wb") as f:
        f.write(header + bw.tobytes())

    return output_path
