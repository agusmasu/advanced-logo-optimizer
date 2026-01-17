import subprocess


def optimize_svg(input_path: str, output_path: str) -> str:
    """
    Optimize an SVG file using scour.

    Args:
        input_path: Path to the input SVG file
        output_path: Path to save the optimized SVG

    Returns:
        Path to the output SVG file
    """
    subprocess.run([
        "scour",
        "-i", input_path,
        "-o", output_path,
        "--enable-viewboxing",
        "--enable-id-stripping",
        "--enable-comment-stripping",
        "--shorten-ids",
        "--indent=none",
    ], check=True)
    return output_path


def export_to_pdf(svg_path: str, output_path: str) -> str:
    """
    Export an SVG file to PDF using Inkscape.

    Args:
        svg_path: Path to the input SVG file
        output_path: Path to save the PDF

    Returns:
        Path to the output PDF file
    """
    subprocess.run([
        "inkscape",
        svg_path,
        "--export-filename", output_path,
        "--export-type=pdf",
    ], check=True)
    return output_path


def export_to_eps(svg_path: str, output_path: str) -> str:
    """
    Export an SVG file to EPS using Inkscape.

    Args:
        svg_path: Path to the input SVG file
        output_path: Path to save the EPS

    Returns:
        Path to the output EPS file
    """
    subprocess.run([
        "inkscape",
        svg_path,
        "--export-filename", output_path,
        "--export-type=eps",
    ], check=True)
    return output_path
