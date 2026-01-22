"""
Local test script for potrace vectorization methods.

Usage:
    python test_vectorization_potrace.py <input_image>
    python test_vectorization_potrace.py <input_image> --method potrace_color|potrace_smooth|potrace_smooth_color
    python test_vectorization_potrace.py <input_image> --all  # Test all methods

Example:
    python test_vectorization_potrace.py logo.png
    python test_vectorization_potrace.py logo.png --method potrace_smooth
    python test_vectorization_potrace.py logo.png --all
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.processors.cleanup import quantize
from src.processors.potrace_vectorizer import (
    vectorize_potrace_color,
    vectorize_potrace_smooth,
    _save_pbm,
)


def vectorize_potrace_smooth_color(
    input_path: str,
    output_path: str,
    max_colors: int = 12,
    alphamax: float = 1.0,
    opticurve: bool = True,
    opttolerance: float = 0.2,
    turdsize: int = 2,
) -> str:
    """
    Convert a color image to SVG using potrace with smooth curve settings.
    
    This combines the color layer approach with smooth curve optimization,
    preserving potrace's native coordinate system.
    
    Args:
        input_path: Path to the input PNG image
        output_path: Path to save the output SVG
        max_colors: Maximum number of colors to extract
        alphamax: Corner threshold (higher = smoother curves)
        opticurve: Whether to optimize curves
        opttolerance: Curve optimization tolerance
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
    quantized = quantize(image, max_colors=max_colors)
    q_arr = np.array(quantized)

    # Get unique colors from quantized image
    q_alpha = q_arr[:, :, 3]
    q_rgb = q_arr[:, :, :3]
    q_mask = q_alpha > 0

    visible_q_pixels = q_rgb[q_mask]
    unique_colors = np.unique(visible_q_pixels.reshape(-1, 3), axis=0)

    h, w = arr.shape[:2]
    
    # Extract SVG metadata from first layer to get proper viewBox and dimensions
    svg_header = None
    svg_transform = None
    svg_groups = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for i, color in enumerate(unique_colors):
            # Create binary mask for this color
            color_mask = np.all(q_rgb == color, axis=2) & q_mask
            bw = np.where(color_mask, 0, 255).astype(np.uint8)  # Black on white

            # Save as PBM
            pbm_path = Path(temp_dir) / f"layer_{i}.pbm"
            _save_pbm(bw, str(pbm_path))

            # Trace with potrace using smooth settings
            svg_layer_path = Path(temp_dir) / f"layer_{i}.svg"
            cmd = [
                "potrace",
                str(pbm_path),
                "-s",
                "-o", str(svg_layer_path),
                "-a", str(alphamax),
                "-t", str(turdsize),
            ]
            
            if opticurve:
                cmd.extend(["-O", str(opttolerance)])
            else:
                cmd.append("-n")
            
            subprocess.run(cmd, check=True)

            # Read SVG and extract header and paths
            with open(svg_layer_path, "r") as f:
                svg_content = f.read()

            # On first iteration, extract the SVG header and viewBox
            if i == 0:
                # Extract the svg tag attributes
                svg_match = re.search(r'<svg[^>]*>', svg_content)
                if svg_match:
                    svg_header = svg_match.group(0)
                
                # Extract the transform from the g tag
                transform_match = re.search(r'<g transform="([^"]*)"', svg_content)
                if transform_match:
                    svg_transform = transform_match.group(1)

            # Extract all path elements with their fill color
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
            
            # Find all paths and wrap them in a colored group
            paths = re.findall(r'<path[^>]*d="([^"]*)"[^>]*/>', svg_content)
            if paths:
                path_elements = '\n'.join([f'<path d="{path_d}"/>' for path_d in paths])
                svg_groups.append(f'<g fill="{hex_color}" stroke="none">\n{path_elements}\n</g>')

    # Construct final SVG with proper coordinate system
    if svg_header and svg_transform:
        # Use potrace's coordinate system
        final_svg = f'''{svg_header}
<metadata>
Created by potrace with smooth color layer tracing
</metadata>
<g transform="{svg_transform}">
{chr(10).join(svg_groups)}
</g>
</svg>'''
    else:
        # Fallback to simple format
        final_svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
{chr(10).join(svg_groups)}
</svg>'''

    with open(output_path, "w") as f:
        f.write(final_svg)

    return output_path


def test_vectorization(input_path: str, method: str, output_dir: str) -> str:
    """Test a specific vectorization method."""
    output_path = os.path.join(output_dir, f"{method}_result.svg")

    if method == "potrace_color":
        vectorize_potrace_color(input_path, output_path)

    elif method == "potrace_smooth":
        vectorize_potrace_smooth(input_path, output_path)

    elif method == "potrace_smooth_color":
        vectorize_potrace_smooth_color(input_path, output_path)

    else:
        raise ValueError(f"Unknown method: {method}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Test potrace vectorization locally")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument(
        "--method",
        choices=["potrace_color", "potrace_smooth", "potrace_smooth_color"],
        default="potrace_smooth",
        help="Vectorization method (default: potrace_smooth)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all methods and compare results",
    )
    parser.add_argument(
        "--output-dir",
        default="test_output",
        help="Output directory (default: test_output)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    input_name = Path(args.input).stem
    output_subdir = os.path.join(args.output_dir, input_name)
    os.makedirs(output_subdir, exist_ok=True)

    # Test vectorization methods
    if args.all:
        methods = ["potrace_color", "potrace_smooth", "potrace_smooth_color"]
        print(f"Testing all potrace vectorization methods:")

        for method in methods:
            try:
                print(f"  [{method}] Processing...", end=" ", flush=True)
                output_path = test_vectorization(args.input, method, output_subdir)
                print(f"Done -> {output_path}")
            except Exception as e:
                print(f"Failed: {e}")
                import traceback
                traceback.print_exc()

    else:
        print(f"Testing {args.method} vectorization...")
        try:
            output_path = test_vectorization(args.input, args.method, output_subdir)
            print(f"Output saved to: {output_path}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
