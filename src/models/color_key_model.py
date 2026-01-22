import cv2
import numpy as np
from PIL import Image

from .base import BackgroundRemover


class ColorKeyRemover(BackgroundRemover):
    """
    Background remover using LAB color space for perceptually accurate color matching.

    Best for: Logos with solid backgrounds where color accuracy is critical,
    especially when the background has slight color variations or lighting differences.

    How it works:
    1. Converts image to LAB color space (perceptually uniform)
    2. Calculates Delta-E color distance from target background color
    3. Creates a mask based on perceptual color similarity

    LAB color space advantages:
    - Delta-E distance correlates with human perception of color difference
    - More accurate than RGB Euclidean distance for color matching
    - Handles slight lighting/color variations better
    """

    def __init__(
        self,
        target_color_rgb: tuple = (255, 255, 255),
        tolerance: float = 15.0,
        auto_detect_color: bool = True,
    ):
        """
        Initialize the LAB color key remover.

        Args:
            target_color_rgb: RGB tuple of background color (default: white)
            tolerance: Maximum Delta-E distance to consider as background.
                      Typical values: 5-10 (strict), 15-25 (moderate), 30+ (loose)
            auto_detect_color: If True, detect background color from corners.
        """
        self.target_color_rgb = target_color_rgb
        self.tolerance = tolerance
        self.auto_detect_color = auto_detect_color

    def _rgb_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB array to LAB color space."""
        # Ensure input is in correct format for OpenCV
        if rgb.ndim == 1:
            rgb = rgb.reshape(1, 1, 3)
        rgb_uint8 = rgb.astype(np.uint8)
        lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
        return lab.astype(np.float32)

    def _detect_background_color(self, image_rgb: np.ndarray) -> tuple:
        """
        Detect the dominant background color from image corners.

        Args:
            image_rgb: RGB image array

        Returns:
            Tuple of (R, G, B) background color
        """
        h, w = image_rgb.shape[:2]
        corner_size = max(5, min(h, w) // 20)

        corners = [
            image_rgb[0:corner_size, 0:corner_size],
            image_rgb[0:corner_size, w - corner_size:w],
            image_rgb[h - corner_size:h, 0:corner_size],
            image_rgb[h - corner_size:h, w - corner_size:w],
        ]

        all_pixels = np.vstack([c.reshape(-1, 3) for c in corners])
        median_color = np.median(all_pixels, axis=0).astype(np.uint8)

        return tuple(median_color)

    def _calculate_delta_e(self, lab_image: np.ndarray, lab_target: np.ndarray) -> np.ndarray:
        """
        Calculate Delta-E (CIE76) color distance for each pixel.

        Args:
            lab_image: LAB image array (H, W, 3)
            lab_target: LAB target color (1, 1, 3) or (3,)

        Returns:
            Array of Delta-E distances (H, W)
        """
        lab_target = lab_target.reshape(1, 1, 3).astype(np.float32)
        diff = lab_image.astype(np.float32) - lab_target
        delta_e = np.sqrt(np.sum(diff ** 2, axis=2))
        return delta_e

    def remove(self, input_path: str, output_path: str) -> Image.Image:
        """Remove background using LAB color space matching."""
        # Read image
        image = Image.open(input_path).convert("RGB")
        image_rgb = np.array(image)

        # Detect or use specified background color
        if self.auto_detect_color:
            bg_color_rgb = self._detect_background_color(image_rgb)
        else:
            bg_color_rgb = self.target_color_rgb

        # Convert both image and target to LAB
        image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_lab = self._rgb_to_lab(np.array(bg_color_rgb, dtype=np.uint8))

        # Calculate Delta-E distance for each pixel
        delta_e = self._calculate_delta_e(image_lab, target_lab)

        # Create binary mask: 0 for background, 255 for foreground
        mask = np.where(delta_e <= self.tolerance, 0, 255).astype(np.uint8)

        # Create RGBA output
        result = np.dstack((image_rgb, mask))

        output_image = Image.fromarray(result, mode="RGBA")
        output_image.save(output_path, "PNG")

        return output_image


def remove_background(
    input_path: str, output_path: str, tolerance: float = 15.0
) -> Image.Image:
    """Convenience function for LAB color key background removal."""
    return ColorKeyRemover(tolerance=tolerance).remove(input_path, output_path)
