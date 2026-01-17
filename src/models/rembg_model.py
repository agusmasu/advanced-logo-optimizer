import io
import pathlib

from PIL import Image
from rembg import remove

from .base import BackgroundRemover


class RembgRemover(BackgroundRemover):
    """Background remover using the rembg library (fast, pre-trained model)."""

    def remove(self, input_path: str, output_path: str) -> Image.Image:
        inp = pathlib.Path(input_path).read_bytes()
        out = remove(inp)

        img = Image.open(io.BytesIO(out)).convert("RGBA")
        img.save(output_path, "PNG")
        return img


def remove_background(input_path: str, output_path: str) -> Image.Image:
    """Convenience function for backward compatibility."""
    return RembgRemover().remove(input_path, output_path)
