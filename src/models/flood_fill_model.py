import cv2
import numpy as np
from PIL import Image

from .base import BackgroundRemover


class FloodFillRemover(BackgroundRemover):
    """
    Background remover using flood fill from image corners.

    Best for: Logos with solid color backgrounds where internal same-color elements
    should be preserved. This is the RECOMMENDED approach for most logo use cases.

    How it works:
    1. Auto-detects background color from image corners
    2. Flood fills from all four corners to find connected background regions
    3. Only removes the outer background, preserving internal elements

    Produces pixel-perfect hard edges (no anti-aliasing artifacts).
    """

    def __init__(self, tolerance: tuple = (15, 15, 15), auto_detect_color: bool = True):
        """
        Initialize the flood fill remover.

        Args:
            tolerance: (lo_diff, up_diff) or single int for color tolerance in flood fill.
                       Lower values = stricter matching, higher = more forgiving.
            auto_detect_color: If True, detect background color from corners.
                              If False, assume white (255, 255, 255).
        """
        if isinstance(tolerance, int):
            tolerance = (tolerance, tolerance, tolerance)
        self.tolerance = tolerance
        self.auto_detect_color = auto_detect_color

    def _detect_background_color(self, image: np.ndarray) -> tuple:
        """
        Detect the dominant background color by sampling the four corners.

        Args:
            image: BGR image array

        Returns:
            Tuple of (B, G, R) background color
        """
        h, w = image.shape[:2]
        corner_size = max(5, min(h, w) // 20)  # Sample 5% or at least 5 pixels

        # Sample corners
        corners = [
            image[0:corner_size, 0:corner_size],  # Top-left
            image[0:corner_size, w - corner_size:w],  # Top-right
            image[h - corner_size:h, 0:corner_size],  # Bottom-left
            image[h - corner_size:h, w - corner_size:w],  # Bottom-right
        ]

        # Calculate median color across all corner samples
        all_pixels = np.vstack([c.reshape(-1, 3) for c in corners])
        median_color = np.median(all_pixels, axis=0).astype(np.uint8)

        return tuple(median_color)

    def remove(self, input_path: str, output_path: str) -> Image.Image:
        """Remove background using flood fill from corners."""
        # Read image with OpenCV (BGR format)
        image_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Could not read image: {input_path}")

        h, w = image_bgr.shape[:2]

        # Detect or use default background color
        if self.auto_detect_color:
            bg_color = self._detect_background_color(image_bgr)
        else:
            bg_color = (255, 255, 255)  # White in BGR

        # Create a mask for flood fill (must be 2 pixels larger than image)
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        # Flood fill from all four corners
        # Use a temporary fill color to mark background regions
        fill_value = 128  # Temporary marker value
        lo_diff = self.tolerance
        up_diff = self.tolerance

        # Define corner seed points
        corners = [
            (0, 0),  # Top-left
            (w - 1, 0),  # Top-right
            (0, h - 1),  # Bottom-left
            (w - 1, h - 1),  # Bottom-right
        ]

        # Also add edge midpoints for better coverage
        edge_points = [
            (w // 2, 0),  # Top-middle
            (w // 2, h - 1),  # Bottom-middle
            (0, h // 2),  # Left-middle
            (w - 1, h // 2),  # Right-middle
        ]

        seed_points = corners + edge_points

        # Create a working copy for flood fill
        work_image = image_bgr.copy()

        # Flags for flood fill: 4-connectivity, fill mask only
        flags = 4 | (fill_value << 8) | cv2.FLOODFILL_MASK_ONLY

        for seed in seed_points:
            # Check if this pixel is close to background color
            pixel_color = image_bgr[seed[1], seed[0]]
            color_diff = np.abs(pixel_color.astype(np.int16) - np.array(bg_color, dtype=np.int16))

            if np.all(color_diff <= np.array(self.tolerance)):
                cv2.floodFill(
                    work_image,
                    mask,
                    seed,
                    (0, 0, 0),  # New value (not used with MASK_ONLY)
                    lo_diff,
                    up_diff,
                    flags,
                )

        # Extract the filled region from mask (excluding the 1-pixel border)
        filled_mask = mask[1:-1, 1:-1]

        # Create alpha channel: 255 for foreground, 0 for background
        alpha = np.where(filled_mask == fill_value, 0, 255).astype(np.uint8)

        # Convert BGR to RGB and create RGBA image
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = np.dstack((image_rgb, alpha))

        output_image = Image.fromarray(result, mode="RGBA")
        output_image.save(output_path, "PNG")

        return output_image


def remove_background(input_path: str, output_path: str, tolerance: int = 15) -> Image.Image:
    """Convenience function for flood fill background removal."""
    return FloodFillRemover(tolerance=(tolerance, tolerance, tolerance)).remove(
        input_path, output_path
    )
