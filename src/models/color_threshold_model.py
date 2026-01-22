import numpy as np
from PIL import Image

from .base import BackgroundRemover


class ColorThresholdRemover(BackgroundRemover):
    """
    Background remover using simple color threshold.

    Best for: Logos with solid white/light backgrounds where speed is priority.
    Removes all pixels within a color distance threshold from the target color.

    Note: This removes ALL matching pixels, including white elements inside the logo.
    Use FloodFillRemover if you need to preserve internal white areas.
    """

    def __init__(self, target_color: tuple = (255, 255, 255), tolerance: int = 20):
        """
        Initialize the threshold remover.

        Args:
            target_color: RGB tuple of the background color to remove (default: white)
            tolerance: Maximum color distance to consider as background (0-255)
        """
        self.target_color = np.array(target_color, dtype=np.float32)
        self.tolerance = tolerance

    def remove(self, input_path: str, output_path: str) -> Image.Image:
        """Remove background using color threshold."""
        image = Image.open(input_path).convert("RGBA")
        arr = np.array(image, dtype=np.float32)

        # Calculate Euclidean distance from target color for each pixel
        rgb = arr[:, :, :3]
        distance = np.sqrt(np.sum((rgb - self.target_color) ** 2, axis=2))

        # Create binary mask: 0 for background, 255 for foreground
        mask = np.where(distance <= self.tolerance, 0, 255).astype(np.uint8)

        # Apply mask as alpha channel
        result = arr.astype(np.uint8)
        result[:, :, 3] = mask

        output_image = Image.fromarray(result, mode="RGBA")
        output_image.save(output_path, "PNG")

        return output_image


def remove_background(input_path: str, output_path: str, tolerance: int = 20) -> Image.Image:
    """Convenience function for color threshold background removal."""
    return ColorThresholdRemover(tolerance=tolerance).remove(input_path, output_path)
