import numpy as np
from PIL import Image


def trim_transparent(im: Image.Image, pad: int = 24) -> Image.Image:
    """
    Crop transparent areas around the image content.

    Args:
        im: PIL Image with transparency
        pad: Padding to add around the content (default 24px)

    Returns:
        Cropped PIL Image
    """
    im = im.convert("RGBA")
    arr = np.array(im)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 0)

    if len(xs) == 0:
        return im

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(arr.shape[1] - 1, x1 + pad)
    y1 = min(arr.shape[0] - 1, y1 + pad)

    return im.crop((x0, y0, x1 + 1, y1 + 1))


def quantize(im: Image.Image, max_colors: int = 12) -> Image.Image:
    """
    Reduce the color palette of an image while preserving transparency.

    Args:
        im: PIL Image to quantize
        max_colors: Maximum number of colors in the palette (default 12)

    Returns:
        Quantized PIL Image with preserved alpha channel
    """
    if max_colors <= 0:
        return im

    im = im.convert("RGBA")
    rgb = im.convert("RGB")
    q = rgb.quantize(colors=max_colors).convert("RGB")
    return Image.merge("RGBA", (*q.split(), im.split()[3]))
