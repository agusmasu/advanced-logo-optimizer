"""
Advanced color vectorization methods for SVG logo creation.

This module implements industry-standard approaches for converting color images
to clean vector SVGs, particularly optimized for logos.

Key approaches:
1. Color-separated layer tracing (Potrace per color layer)
2. OpenCV contour detection with polygon simplification
3. Edge-preserving color quantization
4. Hierarchical color decomposition
5. Bitmap effect with clean edges
"""

import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans


def vectorize_color_separated_smooth(
    input_path: str,
    output_path: str,
    n_colors: int = 8,
    alphamax: float = 1.0,
    opttolerance: float = 0.2,
    turdsize: int = 2,
) -> str:
    """
    Color-separated vectorization with smooth curves.

    Traces each color layer separately with Potrace using optimal smooth settings.
    This is similar to Adobe Illustrator's approach.

    Best for: Logos with smooth curves and gradual transitions.
    """
    image = Image.open(input_path).convert("RGBA")
    arr = np.array(image)

    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]
    mask = alpha > 128

    visible_pixels = rgb[mask]

    if len(visible_pixels) == 0:
        _write_empty_svg(output_path, w, h)
        return output_path

    # K-means clustering for optimal color palette
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(visible_pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # Create label map for full image
    label_map = np.full((h, w), -1, dtype=np.int32)
    label_map[mask] = labels

    # Sort colors by area (largest first for proper layering)
    color_counts = np.bincount(labels, minlength=n_colors)
    color_order = np.argsort(-color_counts)

    svg_paths = _trace_color_layers(
        label_map, centers, color_order, mask, w, h,
        alphamax=alphamax, opttolerance=opttolerance, turdsize=turdsize
    )

    _write_svg(output_path, svg_paths, w, h)
    return output_path


def vectorize_opencv_contours(
    input_path: str,
    output_path: str,
    n_colors: int = 8,
    epsilon_factor: float = 0.001,
    min_area: int = 50,
) -> str:
    """
    OpenCV contour-based vectorization with polygon simplification.

    Uses cv2.findContours to extract shape boundaries and Douglas-Peucker
    algorithm for polygon simplification. Produces very clean, geometric results.

    Best for: Logos with geometric shapes and hard edges.

    Args:
        epsilon_factor: Higher = more simplification (0.001 = detailed, 0.01 = simplified)
        min_area: Minimum contour area to include
    """
    image = Image.open(input_path).convert("RGBA")
    arr = np.array(image)

    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]
    mask = alpha > 128

    visible_pixels = rgb[mask]

    if len(visible_pixels) == 0:
        _write_empty_svg(output_path, w, h)
        return output_path

    # Quantize colors
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(visible_pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # Create quantized image
    quantized = np.zeros_like(rgb)
    quantized[mask] = centers[labels]

    svg_paths = []

    # Sort by area for proper layering
    color_counts = np.bincount(labels, minlength=n_colors)
    color_order = np.argsort(-color_counts)

    for color_idx in color_order:
        color = centers[color_idx]

        # Create binary mask for this color
        color_mask = (np.all(quantized == color, axis=2) & mask).astype(np.uint8) * 255

        if color_mask.sum() == 0:
            continue

        # Find contours
        contours, hierarchy = cv2.findContours(
            color_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

        # Process each contour
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Simplify contour using Douglas-Peucker
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < 3:
                continue

            # Convert to SVG path
            path_d = _contour_to_path(approx)

            # Check if this is a hole (hierarchy[0][i][3] >= 0 means it has a parent)
            if hierarchy is not None and hierarchy[0][i][3] >= 0:
                # This is a hole, use evenodd fill rule
                svg_paths.append(f'<path d="{path_d}" fill="{hex_color}" fill-rule="evenodd" stroke="none"/>')
            else:
                svg_paths.append(f'<path d="{path_d}" fill="{hex_color}" stroke="none"/>')

    _write_svg(output_path, svg_paths, w, h)
    return output_path


def vectorize_edge_preserving(
    input_path: str,
    output_path: str,
    n_colors: int = 8,
    edge_sigma: float = 1.5,
    alphamax: float = 0.8,
    turdsize: int = 2,
) -> str:
    """
    Edge-preserving color vectorization.

    Uses bilateral filtering to smooth colors while preserving edges,
    then applies color quantization and Potrace tracing.

    Best for: Logos where edge sharpness is critical but color areas should be smooth.
    """
    image = Image.open(input_path).convert("RGBA")
    arr = np.array(image)

    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]
    mask = alpha > 128

    visible_pixels = rgb[mask]

    if len(visible_pixels) == 0:
        _write_empty_svg(output_path, w, h)
        return output_path

    # Apply bilateral filter (edge-preserving smoothing)
    rgb_filtered = cv2.bilateralFilter(rgb, d=9, sigmaColor=75, sigmaSpace=edge_sigma * 10)
    rgb_filtered = np.where(mask[:, :, None], rgb_filtered, rgb)

    # Quantize colors
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init=10)
    filtered_visible = rgb_filtered[mask]
    labels = kmeans.fit_predict(filtered_visible)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # Create label map
    label_map = np.full((h, w), -1, dtype=np.int32)
    label_map[mask] = labels

    color_counts = np.bincount(labels, minlength=n_colors)
    color_order = np.argsort(-color_counts)

    svg_paths = _trace_color_layers(
        label_map, centers, color_order, mask, w, h,
        alphamax=alphamax, opttolerance=0.2, turdsize=turdsize
    )

    _write_svg(output_path, svg_paths, w, h)
    return output_path


def vectorize_hierarchical(
    input_path: str,
    output_path: str,
    n_colors: int = 8,
    alphamax: float = 0.6,
    turdsize: int = 2,
) -> str:
    """
    Hierarchical color decomposition vectorization.

    Groups similar colors together and traces them as nested layers,
    which can produce cleaner results for logos with color variations.

    Best for: Logos with shading or color gradients that should be simplified.
    """
    image = Image.open(input_path).convert("RGBA")
    arr = np.array(image)

    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]
    mask = alpha > 128

    visible_pixels = rgb[mask]

    if len(visible_pixels) == 0:
        _write_empty_svg(output_path, w, h)
        return output_path

    # K-means with more clusters initially
    initial_colors = min(n_colors * 2, 32)
    kmeans = MiniBatchKMeans(n_clusters=initial_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(visible_pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # Hierarchical clustering to reduce to n_colors
    from scipy.cluster.hierarchy import linkage, fcluster

    linkage_matrix = linkage(centers, method='ward')
    cluster_labels = fcluster(linkage_matrix, n_colors, criterion='maxclust')

    # Map original labels to hierarchical clusters
    final_centers = np.zeros((n_colors, 3), dtype=np.uint8)
    for i in range(n_colors):
        cluster_mask = cluster_labels == (i + 1)
        if cluster_mask.any():
            # Average color of all centers in this cluster
            final_centers[i] = centers[cluster_mask].mean(axis=0).astype(np.uint8)

    # Remap labels
    new_labels = cluster_labels[labels] - 1  # Convert to 0-indexed

    # Create label map
    label_map = np.full((h, w), -1, dtype=np.int32)
    label_map[mask] = new_labels

    color_counts = np.bincount(new_labels, minlength=n_colors)
    color_order = np.argsort(-color_counts)

    svg_paths = _trace_color_layers(
        label_map, final_centers, color_order, mask, w, h,
        alphamax=alphamax, opttolerance=0.2, turdsize=turdsize
    )

    _write_svg(output_path, svg_paths, w, h)
    return output_path


def vectorize_clean_flat(
    input_path: str,
    output_path: str,
    n_colors: int = 6,
    simplify_threshold: int = 3,
    alphamax: float = 0.5,
) -> str:
    """
    Clean flat color vectorization optimized for simple logos.

    Uses morphological operations to clean up color regions before tracing,
    producing very clean, print-ready vectors.

    Best for: Simple logos with solid colors and clean edges.
    """
    image = Image.open(input_path).convert("RGBA")
    arr = np.array(image)

    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]
    mask = alpha > 128

    visible_pixels = rgb[mask]

    if len(visible_pixels) == 0:
        _write_empty_svg(output_path, w, h)
        return output_path

    # Quantize colors
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(visible_pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # Create quantized image
    quantized = np.zeros_like(rgb)
    quantized[mask] = centers[labels]

    svg_paths = []

    color_counts = np.bincount(labels, minlength=n_colors)
    color_order = np.argsort(-color_counts)

    with tempfile.TemporaryDirectory() as temp_dir:
        for color_idx in color_order:
            color = centers[color_idx]

            # Create binary mask
            color_mask = (np.all(quantized == color, axis=2) & mask).astype(np.uint8) * 255

            if color_mask.sum() == 0:
                continue

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (simplify_threshold, simplify_threshold))
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

            # Invert for potrace (traces black areas)
            bw = 255 - color_mask

            # Save as PBM and trace
            pbm_path = Path(temp_dir) / f"layer_{color_idx}.pbm"
            _save_pbm(bw, str(pbm_path))

            svg_layer_path = Path(temp_dir) / f"layer_{color_idx}.svg"
            cmd = [
                "potrace", str(pbm_path), "-s", "-o", str(svg_layer_path),
                "-a", str(alphamax), "-t", "2", "-O", "0.2"
            ]

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                continue

            with open(svg_layer_path, "r") as f:
                svg_content = f.read()

            paths = re.findall(r'<path[^>]*d="([^"]*)"[^>]*/>', svg_content)
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

            for path_d in paths:
                svg_paths.append(f'<path d="{path_d}" fill="{hex_color}" stroke="none"/>')

    _write_svg(output_path, svg_paths, w, h)
    return output_path


def vectorize_median_cut(
    input_path: str,
    output_path: str,
    n_colors: int = 8,
    alphamax: float = 0.8,
    turdsize: int = 2,
) -> str:
    """
    Median-cut color quantization + Potrace vectorization.

    Uses PIL's built-in median-cut algorithm for color quantization,
    which can produce different results than K-means.

    Best for: Comparing against K-means based approaches.
    """
    image = Image.open(input_path).convert("RGBA")
    arr = np.array(image)

    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    mask = alpha > 128

    # Use PIL's quantize with median cut
    rgb_image = image.convert("RGB")
    quantized_pil = rgb_image.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
    quantized_rgb = np.array(quantized_pil.convert("RGB"))

    # Get palette
    palette = quantized_pil.getpalette()[:n_colors * 3]
    colors = np.array(palette).reshape(-1, 3).astype(np.uint8)

    svg_paths = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for i, color in enumerate(colors):
            # Create binary mask for this color
            color_mask = (np.all(quantized_rgb == color, axis=2) & mask).astype(np.uint8) * 255

            if color_mask.sum() == 0:
                continue

            bw = 255 - color_mask

            pbm_path = Path(temp_dir) / f"layer_{i}.pbm"
            _save_pbm(bw, str(pbm_path))

            svg_layer_path = Path(temp_dir) / f"layer_{i}.svg"
            cmd = [
                "potrace", str(pbm_path), "-s", "-o", str(svg_layer_path),
                "-a", str(alphamax), "-t", str(turdsize), "-O", "0.2"
            ]

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                continue

            with open(svg_layer_path, "r") as f:
                svg_content = f.read()

            paths = re.findall(r'<path[^>]*d="([^"]*)"[^>]*/>', svg_content)
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

            for path_d in paths:
                svg_paths.append(f'<path d="{path_d}" fill="{hex_color}" stroke="none"/>')

    _write_svg(output_path, svg_paths, w, h)
    return output_path


def vectorize_lab_quantize(
    input_path: str,
    output_path: str,
    n_colors: int = 8,
    alphamax: float = 0.8,
    turdsize: int = 2,
) -> str:
    """
    LAB color space quantization + Potrace vectorization.

    Performs color quantization in LAB color space, which better matches
    human perception of color differences.

    Best for: Logos where perceptually similar colors should be grouped.
    """
    image = Image.open(input_path).convert("RGBA")
    arr = np.array(image)

    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]
    mask = alpha > 128

    visible_pixels = rgb[mask]

    if len(visible_pixels) == 0:
        _write_empty_svg(output_path, w, h)
        return output_path

    # Convert to LAB color space
    rgb_img = rgb.reshape(1, -1, 3).astype(np.uint8)
    lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    lab_pixels = lab_img.reshape(-1, 3)

    visible_lab = lab_pixels[mask.flatten()]

    # K-means in LAB space
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(visible_lab)
    lab_centers = kmeans.cluster_centers_

    # Convert centers back to RGB
    lab_centers_img = lab_centers.reshape(1, -1, 3).astype(np.uint8)
    rgb_centers_img = cv2.cvtColor(lab_centers_img, cv2.COLOR_LAB2RGB)
    centers = rgb_centers_img.reshape(-1, 3).astype(np.uint8)

    # Create label map
    label_map = np.full((h, w), -1, dtype=np.int32)
    label_map[mask] = labels

    color_counts = np.bincount(labels, minlength=n_colors)
    color_order = np.argsort(-color_counts)

    svg_paths = _trace_color_layers(
        label_map, centers, color_order, mask, w, h,
        alphamax=alphamax, opttolerance=0.2, turdsize=turdsize
    )

    _write_svg(output_path, svg_paths, w, h)
    return output_path


# ============================================================================
# Helper functions
# ============================================================================

def _trace_color_layers(
    label_map: np.ndarray,
    centers: np.ndarray,
    color_order: np.ndarray,
    mask: np.ndarray,
    w: int,
    h: int,
    alphamax: float = 1.0,
    opttolerance: float = 0.2,
    turdsize: int = 2,
) -> List[str]:
    """Trace each color layer with Potrace."""
    svg_paths = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for color_idx in color_order:
            color = centers[color_idx]

            # Create binary mask for this color
            color_mask = ((label_map == color_idx) & mask).astype(np.uint8) * 255

            if color_mask.sum() == 0:
                continue

            # Invert for potrace
            bw = 255 - color_mask

            # Save as PBM
            pbm_path = Path(temp_dir) / f"layer_{color_idx}.pbm"
            _save_pbm(bw, str(pbm_path))

            # Trace with Potrace
            svg_layer_path = Path(temp_dir) / f"layer_{color_idx}.svg"
            cmd = [
                "potrace", str(pbm_path), "-s", "-o", str(svg_layer_path),
                "-a", str(alphamax), "-t", str(turdsize), "-O", str(opttolerance)
            ]

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                continue

            with open(svg_layer_path, "r") as f:
                svg_content = f.read()

            paths = re.findall(r'<path[^>]*d="([^"]*)"[^>]*/>', svg_content)
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

            for path_d in paths:
                svg_paths.append(f'<path d="{path_d}" fill="{hex_color}" stroke="none"/>')

    return svg_paths


def _contour_to_path(contour: np.ndarray) -> str:
    """Convert an OpenCV contour to an SVG path string."""
    points = contour.reshape(-1, 2)

    if len(points) < 2:
        return ""

    path_parts = [f"M {points[0][0]} {points[0][1]}"]

    for point in points[1:]:
        path_parts.append(f"L {point[0]} {point[1]}")

    path_parts.append("Z")

    return " ".join(path_parts)


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


def _write_empty_svg(output_path: str, w: int, h: int) -> None:
    """Write an empty SVG file."""
    svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"></svg>'
    with open(output_path, "w") as f:
        f.write(svg_content)


def _write_svg(output_path: str, svg_paths: List[str], w: int, h: int) -> None:
    """Write SVG file with proper dimensions."""
    # Use inches for reasonable size in design software
    max_dim = max(w, h)
    target_inches = 10
    width_in = (w / max_dim) * target_inches
    height_in = (h / max_dim) * target_inches

    final_svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width_in:.2f}in" height="{height_in:.2f}in" viewBox="0 0 {w} {h}">
{chr(10).join(svg_paths)}
</svg>'''

    with open(output_path, "w") as f:
        f.write(final_svg)


# ============================================================================
# High-accuracy vectorization methods
# ============================================================================

def vectorize_pixel_perfect(
    input_path: str,
    output_path: str,
    n_colors: int = 8,
    simplify_tolerance: float = 0.5,
) -> str:
    """
    Pixel-perfect vectorization using marching squares.

    Creates contours that follow exact pixel boundaries, then applies
    minimal simplification. This preserves fine details and sharp corners
    at the cost of larger file sizes.

    Best for: Logos where accuracy is more important than file size.

    Args:
        simplify_tolerance: Lower = more accurate (0.5 = minimal, 2.0 = moderate)
    """
    from skimage import measure

    image = Image.open(input_path).convert("RGBA")
    arr = np.array(image)

    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]
    mask = alpha > 128

    visible_pixels = rgb[mask]

    if len(visible_pixels) == 0:
        _write_empty_svg(output_path, w, h)
        return output_path

    # K-means clustering
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(visible_pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # Create label map
    label_map = np.full((h, w), -1, dtype=np.int32)
    label_map[mask] = labels

    # Sort by area
    color_counts = np.bincount(labels, minlength=n_colors)
    color_order = np.argsort(-color_counts)

    svg_paths = []

    for color_idx in color_order:
        color = centers[color_idx]

        # Binary mask for this color
        color_mask = ((label_map == color_idx) & mask).astype(np.uint8)

        if color_mask.sum() == 0:
            continue

        # Use marching squares to find contours at pixel level
        # Level 0.5 finds the boundary between 0 and 1
        contours = measure.find_contours(color_mask, 0.5)

        hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

        for contour in contours:
            if len(contour) < 3:
                continue

            # Simplify contour slightly to reduce points
            simplified = _simplify_contour(contour, simplify_tolerance)

            if len(simplified) < 3:
                continue

            # Convert to SVG path
            path_d = _contour_array_to_path(simplified)
            svg_paths.append(f'<path d="{path_d}" fill="{hex_color}" stroke="none"/>')

    _write_svg(output_path, svg_paths, w, h)
    return output_path


def vectorize_potrace_accurate(
    input_path: str,
    output_path: str,
    n_colors: int = 8,
    alphamax: float = 0.0,  # Sharp corners
    opttolerance: float = 0.0,  # No curve optimization
    turdsize: int = 0,  # Keep all details
) -> str:
    """
    Potrace with maximum accuracy settings.

    Uses Potrace's strictest settings to minimize smoothing and
    preserve all details, including small elements.

    Best for: Logos with fine details that need to be preserved exactly.
    """
    image = Image.open(input_path).convert("RGBA")
    arr = np.array(image)

    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]
    mask = alpha > 128

    visible_pixels = rgb[mask]

    if len(visible_pixels) == 0:
        _write_empty_svg(output_path, w, h)
        return output_path

    # K-means clustering
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(visible_pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # Create label map
    label_map = np.full((h, w), -1, dtype=np.int32)
    label_map[mask] = labels

    color_counts = np.bincount(labels, minlength=n_colors)
    color_order = np.argsort(-color_counts)

    svg_paths = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for color_idx in color_order:
            color = centers[color_idx]

            color_mask = ((label_map == color_idx) & mask).astype(np.uint8) * 255

            if color_mask.sum() == 0:
                continue

            bw = 255 - color_mask

            pbm_path = Path(temp_dir) / f"layer_{color_idx}.pbm"
            _save_pbm(bw, str(pbm_path))

            svg_layer_path = Path(temp_dir) / f"layer_{color_idx}.svg"

            # Maximum accuracy settings:
            # -a 0: alphamax=0 means corners are always sharp
            # -n: disable curve optimization entirely
            # -t 0: turdsize=0 keeps all details, even 1-pixel elements
            cmd = [
                "potrace", str(pbm_path), "-s", "-o", str(svg_layer_path),
                "-a", str(alphamax),
                "-t", str(turdsize),
                "-n",  # Disable curve optimization
            ]

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                continue

            with open(svg_layer_path, "r") as f:
                svg_content = f.read()

            paths = re.findall(r'<path[^>]*d="([^"]*)"[^>]*/>', svg_content)
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

            for path_d in paths:
                svg_paths.append(f'<path d="{path_d}" fill="{hex_color}" stroke="none"/>')

    _write_svg(output_path, svg_paths, w, h)
    return output_path


def vectorize_exact_colors(
    input_path: str,
    output_path: str,
    max_colors: int = 16,
    alphamax: float = 0.0,
    turdsize: int = 0,
) -> str:
    """
    Vectorize using the exact colors present in the image.

    Instead of K-means clustering which can shift colors, this method
    extracts the actual unique colors from the (already quantized) image
    and traces each one.

    Best for: Images that have already been color-quantized where you
    want to preserve the exact colors.
    """
    image = Image.open(input_path).convert("RGBA")
    arr = np.array(image)

    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]
    mask = alpha > 128

    if not mask.any():
        _write_empty_svg(output_path, w, h)
        return output_path

    # Get unique colors from visible pixels
    visible_pixels = rgb[mask]
    unique_colors, counts = np.unique(visible_pixels, axis=0, return_counts=True)

    # If too many colors, use K-means to reduce
    if len(unique_colors) > max_colors:
        kmeans = MiniBatchKMeans(n_clusters=max_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(visible_pixels)
        unique_colors = kmeans.cluster_centers_.astype(np.uint8)
        counts = np.bincount(labels, minlength=max_colors)

    # Sort by count (largest areas first)
    color_order = np.argsort(-counts)

    svg_paths = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for i, idx in enumerate(color_order):
            if idx >= len(unique_colors):
                continue

            color = unique_colors[idx]

            # Create mask for this exact color
            if len(unique_colors) <= max_colors:
                # Exact color match
                color_mask = (np.all(rgb == color, axis=2) & mask).astype(np.uint8) * 255
            else:
                # K-means was used, match by cluster
                color_mask = np.zeros((h, w), dtype=np.uint8)
                label_map = np.full(visible_pixels.shape[0], -1, dtype=np.int32)
                label_map = kmeans.predict(visible_pixels)
                temp_mask = np.zeros((h, w), dtype=bool)
                temp_mask[mask] = (label_map == idx)
                color_mask = temp_mask.astype(np.uint8) * 255

            if color_mask.sum() == 0:
                continue

            bw = 255 - color_mask

            pbm_path = Path(temp_dir) / f"layer_{i}.pbm"
            _save_pbm(bw, str(pbm_path))

            svg_layer_path = Path(temp_dir) / f"layer_{i}.svg"
            cmd = [
                "potrace", str(pbm_path), "-s", "-o", str(svg_layer_path),
                "-a", str(alphamax),
                "-t", str(turdsize),
                "-n",
            ]

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                continue

            with open(svg_layer_path, "r") as f:
                svg_content = f.read()

            paths = re.findall(r'<path[^>]*d="([^"]*)"[^>]*/>', svg_content)
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

            for path_d in paths:
                svg_paths.append(f'<path d="{path_d}" fill="{hex_color}" stroke="none"/>')

    _write_svg(output_path, svg_paths, w, h)
    return output_path


def _simplify_contour(contour: np.ndarray, tolerance: float) -> np.ndarray:
    """Simplify contour using Douglas-Peucker algorithm."""
    if len(contour) < 3:
        return contour

    # Convert to format OpenCV expects
    contour_cv = contour.astype(np.float32).reshape(-1, 1, 2)

    # Apply Douglas-Peucker simplification
    simplified = cv2.approxPolyDP(contour_cv, tolerance, True)

    return simplified.reshape(-1, 2)


def _contour_array_to_path(contour: np.ndarray) -> str:
    """Convert a numpy contour array to SVG path string."""
    if len(contour) < 2:
        return ""

    # Note: marching squares returns (row, col) so we need to swap
    path_parts = [f"M {contour[0][1]:.1f} {contour[0][0]:.1f}"]

    for point in contour[1:]:
        path_parts.append(f"L {point[1]:.1f} {point[0]:.1f}")

    path_parts.append("Z")

    return " ".join(path_parts)


# ============================================================================
# Clean vectorization with proper preprocessing
# ============================================================================

def preprocess_for_vectorization(
    input_path: str,
    output_path: str,
    n_colors: int = 8,
    remove_antialiasing: bool = True,
    morphological_clean: bool = True,
    kernel_size: int = 3,
) -> str:
    """
    Preprocess an image to make it cleaner for vectorization.

    This creates a clean, flat-color image with hard edges that will
    vectorize much better than an anti-aliased image.

    Args:
        input_path: Input image path
        output_path: Output preprocessed image path
        n_colors: Number of colors to quantize to
        remove_antialiasing: Convert soft edges to hard edges
        morphological_clean: Apply morphological operations to clean regions
        kernel_size: Size of morphological kernel

    Returns:
        Path to preprocessed image
    """
    image = Image.open(input_path).convert("RGBA")
    arr = np.array(image)

    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]

    # Step 1: Hard threshold alpha (remove anti-aliasing on edges)
    if remove_antialiasing:
        alpha = np.where(alpha > 128, 255, 0).astype(np.uint8)

    mask = alpha > 128
    visible_pixels = rgb[mask]

    if len(visible_pixels) == 0:
        image.save(output_path)
        return output_path

    # Step 2: Quantize colors
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(visible_pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # Create label map
    label_map = np.full((h, w), -1, dtype=np.int32)
    label_map[mask] = labels

    # Step 3: Morphological cleanup on each color region
    if morphological_clean:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        cleaned_label_map = np.full((h, w), -1, dtype=np.int32)

        for color_idx in range(n_colors):
            color_mask = (label_map == color_idx).astype(np.uint8)

            if color_mask.sum() == 0:
                continue

            # Close then open to clean up boundaries
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

            # Assign back (later colors can overwrite earlier ones at boundaries)
            cleaned_label_map[color_mask > 0] = color_idx

        label_map = cleaned_label_map

    # Step 4: Reconstruct clean image
    clean_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    clean_alpha = np.zeros((h, w), dtype=np.uint8)

    for color_idx in range(n_colors):
        color_mask = label_map == color_idx
        if color_mask.any():
            clean_rgb[color_mask] = centers[color_idx]
            clean_alpha[color_mask] = 255

    # Combine into RGBA
    clean_arr = np.dstack([clean_rgb, clean_alpha])
    clean_image = Image.fromarray(clean_arr, mode="RGBA")
    clean_image.save(output_path)

    return output_path


def vectorize_clean_regions(
    input_path: str,
    output_path: str,
    n_colors: int = 8,
    alphamax: float = 0.5,
    turdsize: int = 2,
    grow_regions: int = 1,
) -> str:
    """
    Vectorize with clean, non-overlapping color regions.

    This method:
    1. Quantizes to exact colors
    2. Cleans up each color region with morphological ops
    3. Slightly grows each region to ensure overlap (no gaps)
    4. Traces in back-to-front order so overlaps are hidden

    Best for: Logos where you want clean, production-ready vectors.
    """
    image = Image.open(input_path).convert("RGBA")
    arr = np.array(image)

    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]

    # Hard threshold alpha
    alpha = np.where(alpha > 128, 255, 0).astype(np.uint8)
    mask = alpha > 128

    visible_pixels = rgb[mask]

    if len(visible_pixels) == 0:
        _write_empty_svg(output_path, w, h)
        return output_path

    # Quantize colors
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(visible_pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # Create label map
    label_map = np.full((h, w), -1, dtype=np.int32)
    label_map[mask] = labels

    # Sort by area (largest first = background)
    color_counts = np.bincount(labels, minlength=n_colors)
    color_order = np.argsort(-color_counts)

    svg_paths = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for color_idx in color_order:
            color = centers[color_idx]

            # Create mask for this color
            color_mask = (label_map == color_idx).astype(np.uint8) * 255

            if color_mask.sum() == 0:
                continue

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

            # Grow region slightly to ensure overlap with neighbors
            if grow_regions > 0:
                grow_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (grow_regions * 2 + 1, grow_regions * 2 + 1)
                )
                color_mask = cv2.dilate(color_mask, grow_kernel)

            # Invert for potrace
            bw = 255 - color_mask

            pbm_path = Path(temp_dir) / f"layer_{color_idx}.pbm"
            _save_pbm(bw, str(pbm_path))

            svg_layer_path = Path(temp_dir) / f"layer_{color_idx}.svg"
            cmd = [
                "potrace", str(pbm_path), "-s", "-o", str(svg_layer_path),
                "-a", str(alphamax),
                "-t", str(turdsize),
                "-O", "0.2",
            ]

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                continue

            with open(svg_layer_path, "r") as f:
                svg_content = f.read()

            paths = re.findall(r'<path[^>]*d="([^"]*)"[^>]*/>', svg_content)
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

            for path_d in paths:
                svg_paths.append(f'<path d="{path_d}" fill="{hex_color}" stroke="none"/>')

    _write_svg(output_path, svg_paths, w, h)
    return output_path


def vectorize_flood_fill_regions(
    input_path: str,
    output_path: str,
    n_colors: int = 8,
    alphamax: float = 0.5,
    turdsize: int = 2,
    tolerance: int = 10,
) -> str:
    """
    Vectorize using flood-fill style region detection.

    Instead of K-means clustering pixels independently, this identifies
    connected regions of similar colors, which produces cleaner boundaries.

    Best for: Logos with distinct color regions that should stay separate.
    """
    image = Image.open(input_path).convert("RGBA")
    arr = np.array(image)

    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]

    # Hard threshold alpha
    mask = alpha > 128

    visible_pixels = rgb[mask]

    if len(visible_pixels) == 0:
        _write_empty_svg(output_path, w, h)
        return output_path

    # First pass: quantize colors
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(visible_pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # Create quantized image
    quantized = np.zeros_like(rgb)
    quantized[mask] = centers[labels]

    # Use connected components to find regions
    # Convert to grayscale label image
    label_map = np.full((h, w), -1, dtype=np.int32)
    label_map[mask] = labels

    # For each color, find connected components
    all_regions = []

    for color_idx in range(n_colors):
        color = centers[color_idx]
        color_mask = (label_map == color_idx).astype(np.uint8)

        if color_mask.sum() == 0:
            continue

        # Find connected components
        num_labels, labels_cc = cv2.connectedComponents(color_mask)

        for region_id in range(1, num_labels):
            region_mask = (labels_cc == region_id).astype(np.uint8) * 255
            region_area = region_mask.sum() // 255

            if region_area < turdsize * turdsize:
                continue

            all_regions.append({
                'color': color,
                'mask': region_mask,
                'area': region_area,
            })

    # Sort by area (largest first)
    all_regions.sort(key=lambda x: -x['area'])

    svg_paths = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for i, region in enumerate(all_regions):
            color = region['color']
            region_mask = region['mask']

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel)

            # Invert for potrace
            bw = 255 - region_mask

            pbm_path = Path(temp_dir) / f"region_{i}.pbm"
            _save_pbm(bw, str(pbm_path))

            svg_layer_path = Path(temp_dir) / f"region_{i}.svg"
            cmd = [
                "potrace", str(pbm_path), "-s", "-o", str(svg_layer_path),
                "-a", str(alphamax),
                "-t", str(turdsize),
                "-O", "0.2",
            ]

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                continue

            with open(svg_layer_path, "r") as f:
                svg_content = f.read()

            paths = re.findall(r'<path[^>]*d="([^"]*)"[^>]*/>', svg_content)
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

            for path_d in paths:
                svg_paths.append(f'<path d="{path_d}" fill="{hex_color}" stroke="none"/>')

    _write_svg(output_path, svg_paths, w, h)
    return output_path


def vectorize_no_antialiasing(
    input_path: str,
    output_path: str,
    n_colors: int = 8,
    alphamax: float = 0.0,
    turdsize: int = 0,
) -> str:
    """
    Vectorize with anti-aliasing completely removed.

    Converts the image to hard-edged pixels before vectorizing,
    which can produce cleaner results for certain logo types.

    Best for: Pixel art style logos or when you need hard edges.
    """
    image = Image.open(input_path).convert("RGBA")
    arr = np.array(image)

    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]

    # Hard threshold alpha - no semi-transparent pixels
    alpha_hard = np.where(alpha > 128, 255, 0).astype(np.uint8)
    mask = alpha_hard > 0

    visible_pixels = rgb[mask]

    if len(visible_pixels) == 0:
        _write_empty_svg(output_path, w, h)
        return output_path

    # Quantize colors
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(visible_pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # Round centers to nearest "web-safe" style colors for cleaner look
    # This snaps to increments of 17 (0x11)
    centers = ((centers + 8) // 17 * 17).astype(np.uint8)
    centers = np.clip(centers, 0, 255)

    # Create label map
    label_map = np.full((h, w), -1, dtype=np.int32)
    label_map[mask] = labels

    color_counts = np.bincount(labels, minlength=n_colors)
    color_order = np.argsort(-color_counts)

    svg_paths = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for color_idx in color_order:
            color = centers[color_idx]

            color_mask = (label_map == color_idx).astype(np.uint8) * 255

            if color_mask.sum() == 0:
                continue

            bw = 255 - color_mask

            pbm_path = Path(temp_dir) / f"layer_{color_idx}.pbm"
            _save_pbm(bw, str(pbm_path))

            svg_layer_path = Path(temp_dir) / f"layer_{color_idx}.svg"

            # Maximum sharpness settings
            cmd = [
                "potrace", str(pbm_path), "-s", "-o", str(svg_layer_path),
                "-a", str(alphamax),  # 0 = sharp corners
                "-t", str(turdsize),  # 0 = keep all details
                "-n",  # No curve optimization
            ]

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                continue

            with open(svg_layer_path, "r") as f:
                svg_content = f.read()

            paths = re.findall(r'<path[^>]*d="([^"]*)"[^>]*/>', svg_content)
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

            for path_d in paths:
                svg_paths.append(f'<path d="{path_d}" fill="{hex_color}" stroke="none"/>')

    _write_svg(output_path, svg_paths, w, h)
    return output_path


# ============================================================================
# Professional-grade vectorization (shared boundary approach)
# ============================================================================

def vectorize_stacked_silhouettes(
    input_path: str,
    output_path: str,
    n_colors: int = 8,
    alphamax: float = 0.8,
    turdsize: int = 2,
) -> str:
    """
    Stacked silhouette approach - how Inkscape and many professional tools work.

    Instead of tracing each color separately, this creates "cumulative" masks:
    - Layer 1: Everything (full silhouette) in background color
    - Layer 2: Everything except background color
    - Layer 3: Everything except background and second color
    - etc.

    This ensures no gaps between colors because each layer completely covers
    the layers below it. This is the standard approach used by Inkscape's
    "Trace Bitmap" multi-color mode.

    Best for: Most logos - this is the industry standard approach.
    """
    image = Image.open(input_path).convert("RGBA")
    arr = np.array(image)

    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]

    # Hard threshold alpha
    mask = alpha > 128

    visible_pixels = rgb[mask]

    if len(visible_pixels) == 0:
        _write_empty_svg(output_path, w, h)
        return output_path

    # Quantize colors
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(visible_pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # Create label map
    label_map = np.full((h, w), -1, dtype=np.int32)
    label_map[mask] = labels

    # Sort by area (largest first = background, rendered first)
    color_counts = np.bincount(labels, minlength=n_colors)
    color_order = np.argsort(-color_counts)

    svg_paths = []

    with tempfile.TemporaryDirectory() as temp_dir:
        # Build cumulative masks - each layer includes itself + all smaller layers
        cumulative_mask = np.zeros((h, w), dtype=bool)

        for i, color_idx in enumerate(color_order):
            color = centers[color_idx]

            # This color's region
            this_color_mask = (label_map == color_idx)

            if not this_color_mask.any():
                continue

            # Cumulative: this color + all colors that come after (smaller areas)
            # We trace this cumulative shape, not just this color
            cumulative_mask = cumulative_mask | this_color_mask

            # For stacking, we trace the cumulative silhouette at each step
            # But we want the INVERSE order for SVG (background first)
            # So we'll collect and reverse at the end

            # Create the mask to trace (cumulative of remaining colors)
            remaining_mask = np.zeros((h, w), dtype=bool)
            for j in range(i, len(color_order)):
                remaining_mask = remaining_mask | (label_map == color_order[j])

            trace_mask = remaining_mask.astype(np.uint8) * 255

            if trace_mask.sum() == 0:
                continue

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            trace_mask = cv2.morphologyEx(trace_mask, cv2.MORPH_CLOSE, kernel)

            bw = 255 - trace_mask

            pbm_path = Path(temp_dir) / f"layer_{i}.pbm"
            _save_pbm(bw, str(pbm_path))

            svg_layer_path = Path(temp_dir) / f"layer_{i}.svg"
            cmd = [
                "potrace", str(pbm_path), "-s", "-o", str(svg_layer_path),
                "-a", str(alphamax),
                "-t", str(turdsize),
                "-O", "0.2",
            ]

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                continue

            with open(svg_layer_path, "r") as f:
                svg_content = f.read()

            paths = re.findall(r'<path[^>]*d="([^"]*)"[^>]*/>', svg_content)
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

            for path_d in paths:
                svg_paths.append(f'<path d="{path_d}" fill="{hex_color}" stroke="none"/>')

    _write_svg(output_path, svg_paths, w, h)
    return output_path


def vectorize_vtracer_layered(
    input_path: str,
    output_path: str,
    n_colors: int = 8,
    mode: str = "spline",
    corner_threshold: int = 60,
    filter_speckle: int = 4,
    color_precision: int = 6,
) -> str:
    """
    Use vtracer on a properly quantized image.

    VTracer is designed to handle multi-color images natively.
    The key is to give it a clean, quantized input with no anti-aliasing.

    This preprocesses the image to be clean, then lets vtracer do its job.
    """
    import vtracer

    # First, create a clean quantized version
    image = Image.open(input_path).convert("RGBA")
    arr = np.array(image)

    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]

    # Hard threshold alpha
    alpha_clean = np.where(alpha > 128, 255, 0).astype(np.uint8)
    mask = alpha_clean > 0

    visible_pixels = rgb[mask]

    if len(visible_pixels) == 0:
        _write_empty_svg(output_path, w, h)
        return output_path

    # Quantize colors
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(visible_pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # Create clean quantized image
    clean_rgb = np.zeros_like(rgb)
    clean_rgb[mask] = centers[labels]

    clean_arr = np.dstack([clean_rgb, alpha_clean])
    clean_image = Image.fromarray(clean_arr, mode="RGBA")

    # Save temp file for vtracer
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name
        clean_image.save(temp_path, "PNG")

    try:
        # Let vtracer handle the multi-color tracing
        vtracer.convert_image_to_svg_py(
            temp_path,
            output_path,
            colormode="color",
            mode=mode,
            corner_threshold=corner_threshold,
            filter_speckle=filter_speckle,
            color_precision=color_precision,
        )
    finally:
        Path(temp_path).unlink(missing_ok=True)

    return output_path


def vectorize_watershed_regions(
    input_path: str,
    output_path: str,
    n_colors: int = 8,
    alphamax: float = 0.6,
    turdsize: int = 2,
) -> str:
    """
    Watershed segmentation for clean region boundaries.

    Uses OpenCV's watershed algorithm to create clean, non-overlapping
    regions with well-defined boundaries. This is similar to how
    some professional tools segment images.

    Best for: Logos with distinct regions that should have clean separation.
    """
    image = Image.open(input_path).convert("RGBA")
    arr = np.array(image)

    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]

    mask = alpha > 128

    visible_pixels = rgb[mask]

    if len(visible_pixels) == 0:
        _write_empty_svg(output_path, w, h)
        return output_path

    # Quantize colors
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(visible_pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # Create initial label map
    label_map = np.full((h, w), -1, dtype=np.int32)
    label_map[mask] = labels

    # Create markers for watershed (erode each region to get sure foreground)
    markers = np.zeros((h, w), dtype=np.int32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for color_idx in range(n_colors):
        color_mask = (label_map == color_idx).astype(np.uint8)
        if color_mask.sum() == 0:
            continue

        # Erode to get "sure" region
        sure_fg = cv2.erode(color_mask, kernel, iterations=2)
        markers[sure_fg > 0] = color_idx + 1  # +1 because 0 is unknown

    # Unknown region (boundaries)
    unknown = mask.astype(np.uint8) - (markers > 0).astype(np.uint8)
    markers[unknown > 0] = 0

    # Prepare image for watershed (needs 3-channel)
    rgb_for_watershed = rgb.copy()

    # Apply watershed
    markers_result = cv2.watershed(rgb_for_watershed, markers.copy())

    # markers_result: -1 = boundary, 0 = background, 1+ = regions
    # Assign boundary pixels to nearest region
    boundary_mask = markers_result == -1
    if boundary_mask.any():
        # Simple: assign to the region with most neighbors
        for y, x in zip(*np.where(boundary_mask)):
            neighbors = []
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and markers_result[ny, nx] > 0:
                    neighbors.append(markers_result[ny, nx])
            if neighbors:
                markers_result[y, x] = max(set(neighbors), key=neighbors.count)

    # Now trace each region
    svg_paths = []
    color_counts = np.bincount(labels, minlength=n_colors)
    color_order = np.argsort(-color_counts)

    with tempfile.TemporaryDirectory() as temp_dir:
        for color_idx in color_order:
            color = centers[color_idx]
            region_id = color_idx + 1

            region_mask = ((markers_result == region_id) & mask).astype(np.uint8) * 255

            if region_mask.sum() == 0:
                continue

            # Small morphological cleanup
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel_small)

            bw = 255 - region_mask

            pbm_path = Path(temp_dir) / f"layer_{color_idx}.pbm"
            _save_pbm(bw, str(pbm_path))

            svg_layer_path = Path(temp_dir) / f"layer_{color_idx}.svg"
            cmd = [
                "potrace", str(pbm_path), "-s", "-o", str(svg_layer_path),
                "-a", str(alphamax),
                "-t", str(turdsize),
                "-O", "0.2",
            ]

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                continue

            with open(svg_layer_path, "r") as f:
                svg_content = f.read()

            paths = re.findall(r'<path[^>]*d="([^"]*)"[^>]*/>', svg_content)
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

            for path_d in paths:
                svg_paths.append(f'<path d="{path_d}" fill="{hex_color}" stroke="none"/>')

    _write_svg(output_path, svg_paths, w, h)
    return output_path


def vectorize_superpixel(
    input_path: str,
    output_path: str,
    n_colors: int = 8,
    n_segments: int = 500,
    compactness: float = 10.0,
    alphamax: float = 0.8,
    turdsize: int = 2,
) -> str:
    """
    SLIC superpixel segmentation + color quantization.

    Uses SLIC (Simple Linear Iterative Clustering) to create perceptually
    meaningful regions, then groups them by color. This creates very clean
    boundaries that follow image edges naturally.

    Best for: Complex logos with subtle color variations.
    """
    from skimage.segmentation import slic
    from skimage.color import label2rgb

    image = Image.open(input_path).convert("RGBA")
    arr = np.array(image)

    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]

    mask = alpha > 128

    if not mask.any():
        _write_empty_svg(output_path, w, h)
        return output_path

    # Run SLIC on the RGB image
    segments = slic(
        rgb,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        mask=mask,
    )

    # Get mean color of each superpixel
    unique_segments = np.unique(segments[mask])
    segment_colors = []

    for seg_id in unique_segments:
        seg_mask = segments == seg_id
        if seg_mask.any():
            mean_color = rgb[seg_mask].mean(axis=0)
            segment_colors.append((seg_id, mean_color))

    if not segment_colors:
        _write_empty_svg(output_path, w, h)
        return output_path

    # Cluster superpixel colors
    superpixel_colors = np.array([c[1] for c in segment_colors])
    kmeans = MiniBatchKMeans(n_clusters=min(n_colors, len(superpixel_colors)), random_state=42, n_init=10)
    color_labels = kmeans.fit_predict(superpixel_colors)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # Map segments to colors
    segment_to_color = {}
    for i, (seg_id, _) in enumerate(segment_colors):
        segment_to_color[seg_id] = color_labels[i]

    # Create final label map
    label_map = np.full((h, w), -1, dtype=np.int32)
    for seg_id, color_idx in segment_to_color.items():
        label_map[segments == seg_id] = color_idx

    # Trace each color
    svg_paths = []
    actual_n_colors = len(centers)
    color_counts = np.zeros(actual_n_colors, dtype=np.int64)
    for i in range(actual_n_colors):
        color_counts[i] = (label_map == i).sum()
    color_order = np.argsort(-color_counts)

    with tempfile.TemporaryDirectory() as temp_dir:
        for color_idx in color_order:
            color = centers[color_idx]

            color_mask = ((label_map == color_idx) & mask).astype(np.uint8) * 255

            if color_mask.sum() == 0:
                continue

            # Small cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

            bw = 255 - color_mask

            pbm_path = Path(temp_dir) / f"layer_{color_idx}.pbm"
            _save_pbm(bw, str(pbm_path))

            svg_layer_path = Path(temp_dir) / f"layer_{color_idx}.svg"
            cmd = [
                "potrace", str(pbm_path), "-s", "-o", str(svg_layer_path),
                "-a", str(alphamax),
                "-t", str(turdsize),
                "-O", "0.2",
            ]

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                continue

            with open(svg_layer_path, "r") as f:
                svg_content = f.read()

            paths = re.findall(r'<path[^>]*d="([^"]*)"[^>]*/>', svg_content)
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

            for path_d in paths:
                svg_paths.append(f'<path d="{path_d}" fill="{hex_color}" stroke="none"/>')

    _write_svg(output_path, svg_paths, w, h)
    return output_path
