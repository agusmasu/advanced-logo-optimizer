"""
Local test script for background removal models.

Usage:
    python test_bg_removal.py <input_image>
    python test_bg_removal.py <input_image> --method flood_fill|threshold|color_key|birefnet
    python test_bg_removal.py <input_image> --all  # Test all methods

Example:
    python test_bg_removal.py logo.png
    python test_bg_removal.py logo.png --method threshold
    python test_bg_removal.py logo.png --all
"""

import argparse
import os
import sys
from pathlib import Path

from src.processors.upscaler import upscale


def upscale_image(input_path: str, output_dir: str) -> str:
    """Upscale the input image first (same as main pipeline)."""
    output_path = os.path.join(output_dir, "0_upscaled.png")
    upscale(input_path, output_path)
    return output_path


def test_removal(input_path: str, method: str, output_dir: str) -> str:
    """Test a specific background removal method."""
    output_path = os.path.join(output_dir, f"{method}_result.png")

    if method == "flood_fill":
        from src.models.flood_fill_model import FloodFillRemover
        remover = FloodFillRemover()
        remover.remove(input_path, output_path)

    elif method == "threshold":
        from src.models.color_threshold_model import ColorThresholdRemover
        remover = ColorThresholdRemover(tolerance=20)
        remover.remove(input_path, output_path)

    elif method == "color_key":
        from src.models.color_key_model import ColorKeyRemover
        remover = ColorKeyRemover(tolerance=15.0)
        remover.remove(input_path, output_path)

    elif method == "birefnet":
        from src.models.birefnet_hr_model import BiRefNetHRRemover
        remover = BiRefNetHRRemover()
        remover.remove(input_path, output_path)

    else:
        raise ValueError(f"Unknown method: {method}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Test background removal locally")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument(
        "--method",
        choices=["flood_fill", "threshold", "color_key", "birefnet"],
        default="flood_fill",
        help="Background removal method (default: flood_fill)",
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

    # Step 1: Upscale the image first (same as main pipeline)
    print(f"Step 1: Upscaling image...", end=" ", flush=True)
    upscaled_path = upscale_image(args.input, output_subdir)
    print(f"Done -> {upscaled_path}")

    # Step 2: Run background removal on the upscaled image
    if args.all:
        methods = ["flood_fill", "threshold", "color_key", "birefnet"]
        print(f"\nStep 2: Testing all background removal methods:")

        for method in methods:
            try:
                print(f"  [{method}] Processing...", end=" ", flush=True)
                output_path = test_removal(upscaled_path, method, output_subdir)
                print(f"Done -> {output_path}")
            except Exception as e:
                print(f"Failed: {e}")

    else:
        print(f"\nStep 2: Testing {args.method} background removal...")
        try:
            output_path = test_removal(upscaled_path, args.method, output_subdir)
            print(f"Output saved to: {output_path}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
