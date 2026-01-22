"""
Local test script for vectorization methods.

Usage:
    python test_vectorization.py <input_image>
    python test_vectorization.py <input_image> --method vtracer_default|potrace_sharp|...
    python test_vectorization.py <input_image> --all  # Test all methods
    python test_vectorization.py <input_image> --color  # Test only color methods

Example:
    python test_vectorization.py logo.png
    python test_vectorization.py logo.png --method potrace_sharp
    python test_vectorization.py logo.png --all
    python test_vectorization.py logo.png --color  # Compare color vectorization approaches

This script runs the full preprocessing pipeline (upscale -> bg removal -> quantize)
before vectorization, similar to the main production pipeline.

Color Vectorization Methods:
----------------------------
- color_separated_smooth: K-means + Potrace with smooth curves (best general-purpose)
- opencv_contours: OpenCV contour detection + polygon simplification (geometric)
- edge_preserving: Bilateral filter + K-means (preserves sharp edges)
- hierarchical: Hierarchical color clustering (good for gradients)
- clean_flat: Morphological cleanup + Potrace (simple logos)
- median_cut: PIL median-cut quantization (alternative to K-means)
- lab_quantize: LAB color space quantization (perceptually accurate colors)
"""

import argparse
import os
import sys
from pathlib import Path

from src.models.birefnet_hr_model import BiRefNetHRRemover
from src.processors.cleanup import quantize, trim_transparent
from src.processors.upscaler import upscale


def preprocess_image(input_path: str, output_dir: str, skip_upscale: bool = False) -> str:
    """
    Run the preprocessing pipeline on an image.

    Steps:
    1. Upscale (Real-ESRGAN 4x) - optional
    2. Remove background (BiRefNet-HR)
    3. Trim transparent areas
    4. Quantize colors

    Returns the path to the preprocessed clean image.
    """
    if skip_upscale:
        print("  [1/4] Upscaling... Skipped")
        upscaled_path = input_path
    else:
        # Step 1: Upscale
        print("  [1/4] Upscaling...", end=" ", flush=True)
        upscaled_path = os.path.join(output_dir, "0_upscaled.png")
        upscale(input_path, upscaled_path)
        print("Done")

    # Step 2: Remove background
    print("  [2/4] Removing background...", end=" ", flush=True)
    no_bg_path = os.path.join(output_dir, "1_no_bg.png")
    im = BiRefNetHRRemover().remove(upscaled_path, no_bg_path)
    print("Done")

    # Step 3: Trim transparent areas
    print("  [3/4] Trimming...", end=" ", flush=True)
    im = trim_transparent(im, pad=24)
    print("Done")

    # Step 4: Quantize colors
    print("  [4/4] Quantizing colors...", end=" ", flush=True)
    im = quantize(im, max_colors=12)
    clean_path = os.path.join(output_dir, "2_clean.png")
    im.save(clean_path, "PNG")
    print("Done")

    return clean_path


def preprocess_vectorize_only(input_path: str, output_dir: str, clean_for_vector: bool = False) -> str:
    """
    Minimal preprocessing for images that are already upscaled and have no background.

    Steps:
    1. Trim transparent areas
    2. Quantize colors
    3. (Optional) Clean for vectorization - removes anti-aliasing and cleans regions

    Returns the path to the preprocessed clean image.
    """
    from PIL import Image

    print("  [1/2] Trimming...", end=" ", flush=True)
    im = Image.open(input_path).convert("RGBA")
    im = trim_transparent(im, pad=24)
    print("Done")

    print("  [2/2] Quantizing colors...", end=" ", flush=True)
    im = quantize(im, max_colors=12)
    clean_path = os.path.join(output_dir, "2_clean.png")
    im.save(clean_path, "PNG")
    print("Done")

    # Optional: additional cleaning for vectorization
    if clean_for_vector:
        print("  [3/3] Cleaning for vectorization...", end=" ", flush=True)
        from src.processors.color_vectorizer import preprocess_for_vectorization
        vector_clean_path = os.path.join(output_dir, "3_vector_clean.png")
        preprocess_for_vectorization(
            clean_path,
            vector_clean_path,
            n_colors=12,
            remove_antialiasing=True,
            morphological_clean=True,
        )
        print("Done")
        return vector_clean_path

    return clean_path


def test_vectorization(input_path: str, method: str, output_dir: str) -> str:
    """Test a specific vectorization method."""
    output_path = os.path.join(output_dir, f"vec_{method}.svg")

    if method == "vtracer_default":
        from src.processors.vtracer_vectorizer import vectorize_vtracer_default
        vectorize_vtracer_default(input_path, output_path)

    elif method == "vtracer_polygon":
        from src.processors.vtracer_vectorizer import vectorize_vtracer_polygon
        vectorize_vtracer_polygon(input_path, output_path)

    elif method == "vtracer_sharp":
        from src.processors.vtracer_vectorizer import vectorize_vtracer_sharp
        vectorize_vtracer_sharp(input_path, output_path)

    elif method == "vtracer_detailed":
        from src.processors.vtracer_vectorizer import vectorize_vtracer_detailed
        vectorize_vtracer_detailed(input_path, output_path)

    elif method == "potrace_sharp":
        from src.processors.potrace_vectorizer import vectorize_potrace_sharp
        vectorize_potrace_sharp(input_path, output_path)

    elif method == "potrace_smooth":
        from src.processors.potrace_vectorizer import vectorize_potrace_smooth
        vectorize_potrace_smooth(input_path, output_path)

    elif method == "potrace_color":
        from src.processors.potrace_vectorizer import vectorize_potrace_color
        vectorize_potrace_color(input_path, output_path)

    elif method == "autotrace":
        from src.processors.autotrace_vectorizer import vectorize_autotrace
        vectorize_autotrace(input_path, output_path)

    elif method == "autotrace_sharp":
        from src.processors.autotrace_vectorizer import vectorize_autotrace_sharp
        vectorize_autotrace_sharp(input_path, output_path)

    elif method == "autotrace_centerline":
        from src.processors.autotrace_vectorizer import vectorize_autotrace_centerline
        vectorize_autotrace_centerline(input_path, output_path)

    elif method == "illustrator_3color":
        from src.processors.illustrator_vectorizer import vectorize_illustrator_3color
        vectorize_illustrator_3color(input_path, output_path, n_colors=3)

    elif method == "illustrator_6color":
        from src.processors.illustrator_vectorizer import vectorize_illustrator_6color
        vectorize_illustrator_6color(input_path, output_path)

    elif method == "illustrator_sharp":
        from src.processors.illustrator_vectorizer import vectorize_illustrator_sharp
        vectorize_illustrator_sharp(input_path, output_path, n_colors=3)

    elif method == "potrace_smooth_color":
        from src.processors.potrace_vectorizer import vectorize_potrace_smooth_color
        vectorize_potrace_smooth_color(input_path, output_path, max_colors=3)

    # === NEW COLOR VECTORIZATION METHODS ===

    elif method == "color_separated_smooth":
        from src.processors.color_vectorizer import vectorize_color_separated_smooth
        vectorize_color_separated_smooth(input_path, output_path, n_colors=8)

    elif method == "color_separated_8":
        from src.processors.color_vectorizer import vectorize_color_separated_smooth
        vectorize_color_separated_smooth(input_path, output_path, n_colors=8, alphamax=0.8)

    elif method == "color_separated_12":
        from src.processors.color_vectorizer import vectorize_color_separated_smooth
        vectorize_color_separated_smooth(input_path, output_path, n_colors=12)

    elif method == "opencv_contours":
        from src.processors.color_vectorizer import vectorize_opencv_contours
        vectorize_opencv_contours(input_path, output_path, n_colors=8)

    elif method == "opencv_contours_detailed":
        from src.processors.color_vectorizer import vectorize_opencv_contours
        vectorize_opencv_contours(input_path, output_path, n_colors=12, epsilon_factor=0.0005)

    elif method == "opencv_contours_simple":
        from src.processors.color_vectorizer import vectorize_opencv_contours
        vectorize_opencv_contours(input_path, output_path, n_colors=6, epsilon_factor=0.005)

    elif method == "edge_preserving":
        from src.processors.color_vectorizer import vectorize_edge_preserving
        vectorize_edge_preserving(input_path, output_path, n_colors=8)

    elif method == "edge_preserving_sharp":
        from src.processors.color_vectorizer import vectorize_edge_preserving
        vectorize_edge_preserving(input_path, output_path, n_colors=8, alphamax=0.2)

    elif method == "hierarchical":
        from src.processors.color_vectorizer import vectorize_hierarchical
        vectorize_hierarchical(input_path, output_path, n_colors=8)

    elif method == "hierarchical_simple":
        from src.processors.color_vectorizer import vectorize_hierarchical
        vectorize_hierarchical(input_path, output_path, n_colors=4)

    elif method == "clean_flat":
        from src.processors.color_vectorizer import vectorize_clean_flat
        vectorize_clean_flat(input_path, output_path, n_colors=6)

    elif method == "clean_flat_detailed":
        from src.processors.color_vectorizer import vectorize_clean_flat
        vectorize_clean_flat(input_path, output_path, n_colors=10, simplify_threshold=2)

    elif method == "median_cut":
        from src.processors.color_vectorizer import vectorize_median_cut
        vectorize_median_cut(input_path, output_path, n_colors=8)

    elif method == "lab_quantize":
        from src.processors.color_vectorizer import vectorize_lab_quantize
        vectorize_lab_quantize(input_path, output_path, n_colors=8)

    elif method == "lab_quantize_smooth":
        from src.processors.color_vectorizer import vectorize_lab_quantize
        vectorize_lab_quantize(input_path, output_path, n_colors=8, alphamax=1.0)

    # === HIGH-ACCURACY METHODS ===

    elif method == "pixel_perfect":
        from src.processors.color_vectorizer import vectorize_pixel_perfect
        vectorize_pixel_perfect(input_path, output_path, n_colors=8, simplify_tolerance=0.5)

    elif method == "pixel_perfect_detailed":
        from src.processors.color_vectorizer import vectorize_pixel_perfect
        vectorize_pixel_perfect(input_path, output_path, n_colors=12, simplify_tolerance=0.2)

    elif method == "potrace_accurate":
        from src.processors.color_vectorizer import vectorize_potrace_accurate
        vectorize_potrace_accurate(input_path, output_path, n_colors=8)

    elif method == "potrace_accurate_detailed":
        from src.processors.color_vectorizer import vectorize_potrace_accurate
        vectorize_potrace_accurate(input_path, output_path, n_colors=12)

    elif method == "exact_colors":
        from src.processors.color_vectorizer import vectorize_exact_colors
        vectorize_exact_colors(input_path, output_path, max_colors=16)

    elif method == "exact_colors_smooth":
        from src.processors.color_vectorizer import vectorize_exact_colors
        vectorize_exact_colors(input_path, output_path, max_colors=16, alphamax=0.5)

    # === CLEAN REGION METHODS (fixes broken vectors) ===

    elif method == "clean_regions":
        from src.processors.color_vectorizer import vectorize_clean_regions
        vectorize_clean_regions(input_path, output_path, n_colors=8)

    elif method == "clean_regions_12":
        from src.processors.color_vectorizer import vectorize_clean_regions
        vectorize_clean_regions(input_path, output_path, n_colors=12, grow_regions=2)

    elif method == "flood_fill":
        from src.processors.color_vectorizer import vectorize_flood_fill_regions
        vectorize_flood_fill_regions(input_path, output_path, n_colors=8)

    elif method == "flood_fill_detailed":
        from src.processors.color_vectorizer import vectorize_flood_fill_regions
        vectorize_flood_fill_regions(input_path, output_path, n_colors=12, turdsize=1)

    elif method == "no_antialiasing":
        from src.processors.color_vectorizer import vectorize_no_antialiasing
        vectorize_no_antialiasing(input_path, output_path, n_colors=8)

    elif method == "no_antialiasing_detailed":
        from src.processors.color_vectorizer import vectorize_no_antialiasing
        vectorize_no_antialiasing(input_path, output_path, n_colors=12)

    else:
        raise ValueError(f"Unknown method: {method}")

    return output_path


ALL_METHODS = [
    # Original methods
    "vtracer_default",
    "vtracer_polygon",
    "vtracer_sharp",
    "vtracer_detailed",
    "potrace_sharp",
    "potrace_smooth",
    "potrace_color",
    "autotrace",
    "autotrace_sharp",
    "illustrator_3color",
    "illustrator_6color",
    "illustrator_sharp",
    "potrace_smooth_color",
    # New color methods
    "color_separated_smooth",
    "color_separated_8",
    "color_separated_12",
    "opencv_contours",
    "opencv_contours_detailed",
    "opencv_contours_simple",
    "edge_preserving",
    "edge_preserving_sharp",
    "hierarchical",
    "hierarchical_simple",
    "clean_flat",
    "clean_flat_detailed",
    "median_cut",
    "lab_quantize",
    "lab_quantize_smooth",
    # High-accuracy methods
    "pixel_perfect",
    "pixel_perfect_detailed",
    "potrace_accurate",
    "potrace_accurate_detailed",
    "exact_colors",
    "exact_colors_smooth",
    # Clean region methods (fixes broken vectors)
    "clean_regions",
    "clean_regions_12",
    "flood_fill",
    "flood_fill_detailed",
    "no_antialiasing",
    "no_antialiasing_detailed",
]

# Color-specific methods for focused testing
COLOR_METHODS = [
    "color_separated_smooth",
    "color_separated_8",
    "color_separated_12",
    "opencv_contours",
    "opencv_contours_detailed",
    "opencv_contours_simple",
    "edge_preserving",
    "edge_preserving_sharp",
    "hierarchical",
    "hierarchical_simple",
    "clean_flat",
    "clean_flat_detailed",
    "median_cut",
    "lab_quantize",
    "lab_quantize_smooth",
    # High-accuracy methods
    "pixel_perfect",
    "pixel_perfect_detailed",
    "potrace_accurate",
    "potrace_accurate_detailed",
    "exact_colors",
    "exact_colors_smooth",
    # Clean region methods (fixes broken vectors)
    "clean_regions",
    "clean_regions_12",
    "flood_fill",
    "flood_fill_detailed",
    "no_antialiasing",
    "no_antialiasing_detailed",
    # Include existing color methods for comparison
    "vtracer_default",
    "vtracer_sharp",
    "potrace_color",
    "potrace_smooth_color",
    "illustrator_3color",
    "illustrator_6color",
]


def main():
    parser = argparse.ArgumentParser(description="Test vectorization methods locally")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument(
        "--method",
        choices=ALL_METHODS,
        default="vtracer_default",
        help="Vectorization method (default: vtracer_default)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all methods and compare results",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="Test only color vectorization methods (for comparing color approaches)",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing (use input image directly)",
    )
    parser.add_argument(
        "--skip-upscale",
        action="store_true",
        help="Skip upscaling step (image already upscaled)",
    )
    parser.add_argument(
        "--vectorize-only",
        action="store_true",
        help="Image is already upscaled and has no background - only trim and quantize",
    )
    parser.add_argument(
        "--clean-for-vector",
        action="store_true",
        help="Additional preprocessing: remove anti-aliasing and clean color regions",
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

    # Preprocess or use input directly
    if args.skip_preprocess:
        print(f"Skipping preprocessing, using input directly: {args.input}")
        clean_path = args.input
    elif args.vectorize_only:
        clean_mode = " + clean for vector" if args.clean_for_vector else ""
        print(f"Step 1: Preprocessing (trim + quantize{clean_mode})...")
        clean_path = preprocess_vectorize_only(args.input, output_subdir, clean_for_vector=args.clean_for_vector)
        print(f"  -> {clean_path}")
    else:
        print(f"Step 1: Preprocessing image...")
        clean_path = preprocess_image(args.input, output_subdir, skip_upscale=args.skip_upscale)
        print(f"  -> {clean_path}")

    # Run vectorization
    if args.all or args.color:
        methods = COLOR_METHODS if args.color else ALL_METHODS
        mode_name = "color" if args.color else "all"
        print(f"\nStep 2: Testing {mode_name} vectorization methods ({len(methods)} total):")

        results = []
        for method in methods:
            try:
                print(f"  [{method}] Processing...", end=" ", flush=True)
                output_path = test_vectorization(clean_path, method, output_subdir)
                # Get file size
                size_kb = os.path.getsize(output_path) / 1024
                print(f"Done -> {output_path} ({size_kb:.1f} KB)")
                results.append((method, size_kb, "success"))
            except Exception as e:
                print(f"Failed: {e}")
                results.append((method, 0, str(e)))

        # Print summary table
        print(f"\n{'=' * 60}")
        print("SUMMARY - Compare these SVGs to find the best approach:")
        print(f"{'=' * 60}")
        print(f"{'Method':<30} {'Size (KB)':<12} {'Status'}")
        print(f"{'-' * 60}")
        for method, size_kb, status in results:
            if status == "success":
                print(f"{method:<30} {size_kb:<12.1f} OK")
            else:
                print(f"{method:<30} {'-':<12} FAILED")

        print(f"\nAll outputs saved to: {output_subdir}")
        print("\nRecommendations for color logos:")
        print("  - Clean geometric logos: opencv_contours, clean_flat")
        print("  - Smooth curves: color_separated_smooth, lab_quantize_smooth")
        print("  - Complex gradients: hierarchical, edge_preserving")
        print("  - Print-ready: clean_flat, potrace_smooth_color")

    else:
        print(f"\nStep 2: Testing {args.method} vectorization...")
        try:
            output_path = test_vectorization(clean_path, args.method, output_subdir)
            size_kb = os.path.getsize(output_path) / 1024
            print(f"Output saved to: {output_path} ({size_kb:.1f} KB)")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
