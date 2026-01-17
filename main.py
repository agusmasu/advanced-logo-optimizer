#!/usr/bin/env python3
"""
Logo Post-Processing CLI Pipeline

Processes a logo image through: Upscale → Remove BG → Cleanup → Vectorize → Export
"""

import argparse
import sys
from pathlib import Path

from src.config import (
    DEFAULT_MAX_COLORS,
    DEFAULT_TRIM_PADDING,
    DEFAULT_VTRACER_COLOR_PRECISION,
    DEFAULT_VTRACER_FILTER_SPECKLE,
    OUTPUTS_DIR,
)
from src.models.rmbg_model import RMBGRemover
from src.processors.cleanup import quantize, trim_transparent
from src.processors.exporter import export_to_eps, export_to_pdf, optimize_svg
from src.processors.upscaler import upscale
from src.processors.vectorizer import vectorize_bw, vectorize_color


def process_logo(
    input_path: str,
    output_dir: str | None = None,
    max_colors: int = DEFAULT_MAX_COLORS,
    trim_pad: int = DEFAULT_TRIM_PADDING,
) -> dict:
    """
    Process a logo through the full pipeline.

    Args:
        input_path: Path to the input logo image
        output_dir: Directory to save outputs (default: data/outputs)
        max_colors: Maximum colors for quantization (0 to disable)
        trim_pad: Padding around content after trim

    Returns:
        Dictionary of output file paths
    """
    output_dir = Path(output_dir) if output_dir else OUTPUTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    # Step 1: Upscale the image
    upscaled_path = str(output_dir / "0_upscaled.png")
    im = upscale(input_path, upscaled_path)
    print(f"Wrote {upscaled_path}")
    outputs["upscaled"] = upscaled_path

    # Step 2: Remove background
    no_bg_path = str(output_dir / "1_no_bg.png")
    remover = RMBGRemover()
    im = remover.remove(upscaled_path, no_bg_path)
    print(f"Wrote {no_bg_path}")
    outputs["no_bg"] = no_bg_path

    # Step 3: Trim and quantize
    im = trim_transparent(im, pad=trim_pad)
    im = quantize(im, max_colors=max_colors)
    clean_path = str(output_dir / "2_clean.png")
    im.save(clean_path, "PNG")
    print(f"Wrote {clean_path}")
    outputs["clean"] = clean_path

    # Step 4: Generate COLOR vector SVG
    svg_color_path = str(output_dir / "3_trace_color.svg")
    vectorize_color(
        clean_path,
        svg_color_path,
        filter_speckle=DEFAULT_VTRACER_FILTER_SPECKLE,
        color_precision=DEFAULT_VTRACER_COLOR_PRECISION,
    )
    print(f"Wrote {svg_color_path}")
    outputs["svg_color"] = svg_color_path

    # Step 5: Generate B&W vector SVG
    svg_bw_path = str(output_dir / "3_trace_bw.svg")
    vectorize_bw(clean_path, svg_bw_path)
    print(f"Wrote {svg_bw_path}")
    outputs["svg_bw"] = svg_bw_path

    # Step 6: Optimize COLOR SVG
    svg_color_optimized = str(output_dir / "4_color_optimized.svg")
    optimize_svg(svg_color_path, svg_color_optimized)
    print(f"Wrote {svg_color_optimized}")
    outputs["svg_color_optimized"] = svg_color_optimized

    # Step 7: Optimize B&W SVG
    svg_bw_optimized = str(output_dir / "4_bw_optimized.svg")
    optimize_svg(svg_bw_path, svg_bw_optimized)
    print(f"Wrote {svg_bw_optimized}")
    outputs["svg_bw_optimized"] = svg_bw_optimized

    # Step 8: Export COLOR to PDF and EPS
    color_pdf = str(output_dir / "5_color.pdf")
    export_to_pdf(svg_color_optimized, color_pdf)
    print(f"Wrote {color_pdf}")
    outputs["color_pdf"] = color_pdf

    color_eps = str(output_dir / "5_color.eps")
    export_to_eps(svg_color_optimized, color_eps)
    print(f"Wrote {color_eps}")
    outputs["color_eps"] = color_eps

    # Step 9: Export B&W to PDF and EPS
    bw_pdf = str(output_dir / "6_bw.pdf")
    export_to_pdf(svg_bw_optimized, bw_pdf)
    print(f"Wrote {bw_pdf}")
    outputs["bw_pdf"] = bw_pdf

    bw_eps = str(output_dir / "6_bw.eps")
    export_to_eps(svg_bw_optimized, bw_eps)
    print(f"Wrote {bw_eps}")
    outputs["bw_eps"] = bw_eps

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Process a logo image through the post-processing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="data/inputs/logo.png",
        help="Path to the input logo image",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Directory to save output files (default: data/outputs)",
    )
    parser.add_argument(
        "--max-colors",
        type=int,
        default=DEFAULT_MAX_COLORS,
        help="Maximum colors for quantization (0 to disable)",
    )
    parser.add_argument(
        "--trim-pad",
        type=int,
        default=DEFAULT_TRIM_PADDING,
        help="Padding around content after trim",
    )

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing: {args.input}")
    outputs = process_logo(
        args.input,
        output_dir=args.output_dir,
        max_colors=args.max_colors,
        trim_pad=args.trim_pad,
    )
    print(f"\nDone! Generated {len(outputs)} files.")


if __name__ == "__main__":
    main()
