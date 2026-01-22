"""
Test script for the improved vectorization pipeline.

Pipeline:
1. Input: Upscaled image with white background
2. (Optional) Light denoise to clean upscaling artifacts
3. Color quantization (4-8 colors)
4. Potrace vectorization with smooth curve settings
5. SVGO optimization

Usage:
    python test_vectorization_pipeline.py <input_image>
    python test_vectorization_pipeline.py <input_image> --colors 6 --denoise
    python test_vectorization_pipeline.py <input_image> --no-svgo

Example:
    python test_vectorization_pipeline.py logo_upscaled.png --colors 4 --denoise
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
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

    h, w = rgb.shape[:2]

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


def save_pbm(bw_array: np.ndarray, output_path: str) -> str:
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
            save_pbm(bw, str(pbm_path))

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
                print(f"  Warning: potrace failed for layer {idx}")
                continue

            # Read SVG and extract paths
            with open(svg_layer_path, "r") as f:
                svg_content = f.read()

            # Extract header info from first layer
            if idx == 0:
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
        print("  Warning: SVGO not installed. Skipping optimization.")
        print("  Install with: npm install -g svgo")
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
        print(f"  Warning: SVGO failed: {result.stderr}")
        shutil.copy(input_svg, output_svg)
        return False

    return True


def run_pipeline(
    input_path: str,
    output_dir: str,
    n_colors: int = 6,
    denoise: bool = False,
    denoise_strength: str = "light",
    use_svgo: bool = True,
    alphamax: float = 1.0,
    turdsize: int = 10,
    opttolerance: float = 0.2,
    add_white_bg: bool = False,
) -> dict:
    """
    Run the full vectorization pipeline.

    Returns dict with paths to intermediate and final outputs.
    """
    results = {}

    input_name = Path(input_path).stem

    # Load image
    print(f"Loading image: {input_path}")
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not load image: {input_path}")

    # Convert BGR to RGB
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image.shape[:2]
    print(f"  Image size: {w}x{h}")
    results["input_size"] = (w, h)

    # Step 0: Composite on white background (for transparent PNGs)
    if add_white_bg and image.shape[2] == 4:
        print("Step 0: Compositing on white background...")
        image = composite_on_white(image)

        # Save composited image for inspection
        composited_path = os.path.join(output_dir, f"{input_name}_0_white_bg.png")
        composited_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(composited_path, composited_bgr)
        results["composited"] = composited_path
        print(f"  Saved: {composited_path}")

    # Step 1: Optional denoise
    if denoise:
        print(f"Step 1: Applying {denoise_strength} denoise...")
        image = light_denoise(image, strength=denoise_strength)

        # Save denoised image for inspection
        denoised_path = os.path.join(output_dir, f"{input_name}_1_denoised.png")
        denoised_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA if image.shape[2] == 4 else cv2.COLOR_RGB2BGR)
        cv2.imwrite(denoised_path, denoised_bgr)
        results["denoised"] = denoised_path
        print(f"  Saved: {denoised_path}")
    else:
        print("Step 1: Skipping denoise")

    # Step 2: Color quantization
    print(f"Step 2: Quantizing to {n_colors} colors...")
    quantized, colors = quantize_colors_kmeans(image, n_colors=n_colors, white_bg=True)

    # Save quantized image for inspection
    quantized_path = os.path.join(output_dir, f"{input_name}_2_quantized.png")
    quantized_bgr = cv2.cvtColor(quantized, cv2.COLOR_RGBA2BGRA if quantized.shape[2] == 4 else cv2.COLOR_RGB2BGR)
    cv2.imwrite(quantized_path, quantized_bgr)
    results["quantized"] = quantized_path
    print(f"  Found {len(colors)} colors")
    print(f"  Saved: {quantized_path}")

    # Step 3: Potrace vectorization
    print(f"Step 3: Vectorizing with potrace (alphamax={alphamax}, turdsize={turdsize}, opttolerance={opttolerance})...")
    raw_svg_path = os.path.join(output_dir, f"{input_name}_3_potrace.svg")
    vectorize_with_potrace(
        quantized,
        colors,
        raw_svg_path,
        alphamax=alphamax,
        turdsize=turdsize,
        opttolerance=opttolerance,
    )
    results["potrace_svg"] = raw_svg_path
    raw_size = os.path.getsize(raw_svg_path)
    print(f"  Saved: {raw_svg_path} ({raw_size:,} bytes)")

    # Step 4: SVGO optimization
    if use_svgo:
        print("Step 4: Optimizing with SVGO...")
        final_svg_path = os.path.join(output_dir, f"{input_name}_4_final.svg")
        success = optimize_with_svgo(raw_svg_path, final_svg_path)

        if success:
            final_size = os.path.getsize(final_svg_path)
            reduction = (1 - final_size / raw_size) * 100
            print(f"  Saved: {final_svg_path} ({final_size:,} bytes, {reduction:.1f}% reduction)")
        else:
            final_size = raw_size
            print(f"  Saved: {final_svg_path} (no optimization)")

        results["final_svg"] = final_svg_path
        results["final_size"] = final_size
    else:
        print("Step 4: Skipping SVGO optimization")
        # Copy raw svg as final
        final_svg_path = os.path.join(output_dir, f"{input_name}_final.svg")
        shutil.copy(raw_svg_path, final_svg_path)
        results["final_svg"] = final_svg_path
        results["final_size"] = raw_size

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Improved vectorization pipeline: denoise -> quantize -> potrace -> SVGO"
    )
    parser.add_argument("input", help="Path to input image (upscaled with white background)")
    parser.add_argument(
        "--colors", "-c",
        type=int,
        default=6,
        help="Number of colors for quantization (default: 6, range: 4-8 recommended)",
    )
    parser.add_argument(
        "--denoise", "-d",
        action="store_true",
        help="Apply light denoising before quantization",
    )
    parser.add_argument(
        "--denoise-strength",
        choices=["light", "medium", "strong"],
        default="light",
        help="Denoise strength (default: light)",
    )
    parser.add_argument(
        "--no-svgo",
        action="store_true",
        help="Skip SVGO optimization step",
    )
    parser.add_argument(
        "--alphamax",
        type=float,
        default=1.0,
        help="Potrace corner smoothness (default: 1.0, higher = smoother)",
    )
    parser.add_argument(
        "--turdsize",
        type=int,
        default=10,
        help="Potrace noise removal threshold (default: 10)",
    )
    parser.add_argument(
        "--opttolerance",
        type=float,
        default=0.2,
        help="Potrace curve simplification (default: 0.2, lower = more detail)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="test_output",
        help="Output directory (default: test_output)",
    )
    parser.add_argument(
        "--white-bg", "-w",
        action="store_true",
        help="Composite transparent PNG onto white background first (use with no-bg images)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Create output directory
    input_name = Path(args.input).stem
    output_subdir = os.path.join(args.output_dir, f"{input_name}_pipeline")
    os.makedirs(output_subdir, exist_ok=True)

    print("=" * 60)
    print("Vectorization Pipeline")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output directory: {output_subdir}")
    print(f"Settings:")
    print(f"  White background: {args.white_bg}")
    print(f"  Colors: {args.colors}")
    print(f"  Denoise: {args.denoise} ({args.denoise_strength})")
    print(f"  SVGO: {not args.no_svgo}")
    print(f"  Potrace: alphamax={args.alphamax}, turdsize={args.turdsize}, opttolerance={args.opttolerance}")
    print("=" * 60)

    try:
        results = run_pipeline(
            args.input,
            output_subdir,
            n_colors=args.colors,
            denoise=args.denoise,
            denoise_strength=args.denoise_strength,
            use_svgo=not args.no_svgo,
            alphamax=args.alphamax,
            turdsize=args.turdsize,
            opttolerance=args.opttolerance,
            add_white_bg=args.white_bg,
        )

        print("=" * 60)
        print("Pipeline complete!")
        print(f"Final SVG: {results['final_svg']}")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
