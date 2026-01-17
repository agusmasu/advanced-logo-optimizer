import pathlib

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .base import BackgroundRemover

SAM2_MODELS = {
    "large": {
        "checkpoint": "sam2_hiera_large.pt",
        "config": "sam2_hiera_l.yaml",
    },
    "base_plus": {
        "checkpoint": "sam2_hiera_base_plus.pt",
        "config": "sam2_hiera_b+.yaml",
    },
    "small": {
        "checkpoint": "sam2_hiera_small.pt",
        "config": "sam2_hiera_s.yaml",
    },
    "tiny": {
        "checkpoint": "sam2_hiera_tiny.pt",
        "config": "sam2_hiera_t.yaml",
    },
}


class SAM2Remover(BackgroundRemover):
    """Background remover using SAM2 (Segment Anything Model 2)."""

    def __init__(
        self,
        model_size: str = "large",
        checkpoint_dir: str = "./data/checkpoints",
    ):
        if model_size not in SAM2_MODELS:
            raise ValueError(f"model_size must be one of {list(SAM2_MODELS.keys())}")
        self.model_size = model_size
        self.checkpoint_dir = checkpoint_dir
        self._model = None
        self._predictor = None
        self._device = None

    def _load_model(self):
        if self._model is None:
            model_config = SAM2_MODELS[self.model_size]
            checkpoint_path = pathlib.Path(self.checkpoint_dir) / model_config["checkpoint"]
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model = build_sam2(
                model_config["config"],
                str(checkpoint_path),
                device=self._device,
            )
            self._predictor = SAM2ImagePredictor(self._model)

    def remove(self, input_path: str, output_path: str) -> Image.Image:
        self._load_model()

        image = Image.open(input_path).convert("RGB")
        image_np = np.array(image)

        self._predictor.set_image(image_np)

        # Use center point as prompt (assumes logo is roughly centered)
        h, w = image_np.shape[:2]
        center_point = np.array([[w // 2, h // 2]])
        center_label = np.array([1])  # 1 = foreground

        masks, scores, _ = self._predictor.predict(
            point_coords=center_point,
            point_labels=center_label,
            multimask_output=True,
        )

        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx]

        rgba_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
        rgb_image = image.convert("RGBA")

        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        rgba_image.paste(rgb_image, mask=mask_image)

        rgba_image.save(output_path, "PNG")
        return rgba_image

    def remove_with_box(
        self,
        input_path: str,
        output_path: str,
        box: tuple[int, int, int, int] | None = None,
    ) -> Image.Image:
        """Remove background using a bounding box prompt."""
        self._load_model()

        image = Image.open(input_path).convert("RGB")
        image_np = np.array(image)

        self._predictor.set_image(image_np)

        h, w = image_np.shape[:2]
        if box is None:
            margin = min(w, h) // 20
            box = (margin, margin, w - margin, h - margin)

        input_box = np.array([box])

        masks, scores, _ = self._predictor.predict(
            box=input_box,
            multimask_output=True,
        )

        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx]

        rgba_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
        rgb_image = image.convert("RGBA")

        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        rgba_image.paste(rgb_image, mask=mask_image)

        rgba_image.save(output_path, "PNG")
        return rgba_image

    def remove_auto(self, input_path: str, output_path: str) -> Image.Image:
        """Automatically segment all objects and keep the largest one."""
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        self._load_model()

        mask_generator = SAM2AutomaticMaskGenerator(
            model=self._model,
            points_per_side=32,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.9,
            min_mask_region_area=100,
        )

        image = Image.open(input_path).convert("RGB")
        image_np = np.array(image)

        masks = mask_generator.generate(image_np)

        if not masks:
            raise ValueError("No masks generated. The image may not contain detectable objects.")

        masks_sorted = sorted(masks, key=lambda x: x["area"], reverse=True)
        best_mask = masks_sorted[0]["segmentation"]

        rgba_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
        rgb_image = image.convert("RGBA")

        mask_image = Image.fromarray((best_mask * 255).astype(np.uint8))
        rgba_image.paste(rgb_image, mask=mask_image)

        rgba_image.save(output_path, "PNG")
        return rgba_image


def remove_background(
    input_path: str,
    output_path: str,
    model_size: str = "large",
    checkpoint_dir: str = "./data/checkpoints",
) -> Image.Image:
    """Convenience function for backward compatibility."""
    return SAM2Remover(model_size=model_size, checkpoint_dir=checkpoint_dir).remove(input_path, output_path)
