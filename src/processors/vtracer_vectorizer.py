"""
VTracer-based vectorization with different configurations.

VTracer is an O(n) algorithm that handles colored images well but can produce
artifacts on small details. These configurations aim to minimize artifacts
while preserving sharp edges.

Best for: Colored images, high-resolution scans, images with many colors.
"""

import vtracer


def vectorize_vtracer_polygon(
    input_path: str,
    output_path: str,
    filter_speckle: int = 4,
    color_precision: int = 6,
) -> str:
    """
    Convert an image to SVG using vtracer in polygon mode.

    Polygon mode produces hard edges instead of smooth curves,
    which can be better for logos with geometric shapes.

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
        mode="polygon",  # Hard edges instead of curves
        filter_speckle=filter_speckle,
        color_precision=color_precision,
    )
    return output_path


def vectorize_vtracer_sharp(
    input_path: str,
    output_path: str,
    corner_threshold: int = 30,
    splice_threshold: int = 25,
    filter_speckle: int = 2,
    color_precision: int = 8,
) -> str:
    """
    Convert an image to SVG using vtracer with sharp corner settings.

    Lower corner_threshold and splice_threshold values produce sharper corners
    and more accurate curves at the cost of slightly larger file size.

    Args:
        input_path: Path to the input PNG image
        output_path: Path to save the output SVG
        corner_threshold: Min angle (degrees) to be considered a corner (default 30, lower = sharper)
        splice_threshold: Min angle displacement to splice a spline (default 25, lower = more accurate)
        filter_speckle: Filter out small artifacts (default 2)
        color_precision: Color precision level (default 8, higher = more colors preserved)

    Returns:
        Path to the output SVG file
    """
    vtracer.convert_image_to_svg_py(
        input_path,
        output_path,
        colormode="color",
        mode="spline",
        corner_threshold=corner_threshold,
        splice_threshold=splice_threshold,
        filter_speckle=filter_speckle,
        color_precision=color_precision,
    )
    return output_path


def vectorize_vtracer_detailed(
    input_path: str,
    output_path: str,
    corner_threshold: int = 45,
    splice_threshold: int = 30,
    filter_speckle: int = 0,
    color_precision: int = 10,
    path_precision: int = 8,
) -> str:
    """
    Convert an image to SVG using vtracer with maximum detail preservation.

    This configuration minimizes filtering and maximizes precision,
    suitable for detailed logos where every edge matters.

    Args:
        input_path: Path to the input PNG image
        output_path: Path to save the output SVG
        corner_threshold: Min angle (degrees) for corners (default 45)
        splice_threshold: Min angle displacement for splicing (default 30)
        filter_speckle: Filter out small artifacts (0 = no filtering)
        color_precision: Color precision level (10 = very precise)
        path_precision: Decimal places in path coordinates (8 = high precision)

    Returns:
        Path to the output SVG file
    """
    vtracer.convert_image_to_svg_py(
        input_path,
        output_path,
        colormode="color",
        mode="spline",
        corner_threshold=corner_threshold,
        splice_threshold=splice_threshold,
        filter_speckle=filter_speckle,
        color_precision=color_precision,
        path_precision=path_precision,
    )
    return output_path


def vectorize_vtracer_default(
    input_path: str,
    output_path: str,
    filter_speckle: int = 4,
    color_precision: int = 6,
) -> str:
    """
    Convert an image to SVG using vtracer with default settings.

    This is the original configuration used in the pipeline.

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
