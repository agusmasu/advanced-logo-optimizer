from abc import ABC, abstractmethod
from PIL import Image


class BackgroundRemover(ABC):
    """Abstract base class for background removal models."""

    @abstractmethod
    def remove(self, input_path: str, output_path: str) -> Image.Image:
        """
        Remove background from an image.

        Args:
            input_path: Path to the input image
            output_path: Path to save the output image with transparent background

        Returns:
            PIL Image with transparent background
        """
        pass
