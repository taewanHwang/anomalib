"""Custom ImageVisualizer for WinCLIP HDMAP experiments.

Generates 4-column visualization:
1. Image (original)
2. Image + Anomaly Map (auto scale - per-image min-max)
3. Image + Anomaly Map (fixed scale - 0 to 1)
4. Image + Pred Mask
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image as PILImage

from anomalib.utils.path import generate_output_filename
from anomalib.visualization.base import Visualizer
from anomalib.visualization.image.item_visualizer import (
    DEFAULT_FIELDS_CONFIG,
    DEFAULT_OVERLAY_FIELDS_CONFIG,
    DEFAULT_TEXT_CONFIG,
)

if TYPE_CHECKING:
    from lightning.pytorch import Trainer
    from anomalib.data import ImageBatch
    from anomalib.models import AnomalibModule


def apply_colormap(arr: np.ndarray, normalize: bool = True, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    """Apply jet colormap to array.

    Args:
        arr: Input array (H, W) with values in [0, 1] or arbitrary range.
        normalize: If True, normalize to [0, 1] using min-max. If False, use vmin/vmax.
        vmin: Minimum value for fixed scale.
        vmax: Maximum value for fixed scale.

    Returns:
        RGB array (H, W, 3) with values in [0, 255].
    """
    import matplotlib.pyplot as plt

    if normalize:
        # Auto scale (per-image min-max)
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr_norm = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr_norm = np.zeros_like(arr)
    else:
        # Fixed scale [vmin, vmax] -> [0, 1]
        arr_norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1)

    # Apply colormap
    cmap = plt.cm.jet
    colored = cmap(arr_norm)[:, :, :3]  # Remove alpha channel
    return (colored * 255).astype(np.uint8)


def blend_images(base: np.ndarray, overlay: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend two RGB images.

    Args:
        base: Base image (H, W, 3).
        overlay: Overlay image (H, W, 3).
        alpha: Blend factor for overlay.

    Returns:
        Blended image (H, W, 3).
    """
    return ((1 - alpha) * base + alpha * overlay).astype(np.uint8)


class FourColumnVisualizer(Visualizer):
    """Four-column visualizer for anomaly detection.

    Generates visualization with 4 columns:
    1. Image (original)
    2. Image + Anomaly Map (auto scale)
    3. Image + Anomaly Map (fixed 0-1 scale)
    4. Image + Pred Mask

    Args:
        field_size: Size of each field (width, height).
        alpha: Blend alpha for overlay.
        output_dir: Directory to save visualizations.
    """

    def __init__(
        self,
        field_size: tuple[int, int] = (256, 256),
        alpha: float = 0.5,
        output_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.field_size = field_size
        self.alpha = alpha
        self.output_dir = Path(output_dir) if output_dir else None

    def on_test_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "AnomalibModule",
        outputs: Any,
        batch: "ImageBatch",
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Generate 4-column visualization for test batch."""
        self._visualize_batch(trainer, batch, "test")

    def on_predict_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "AnomalibModule",
        outputs: Any,
        batch: "ImageBatch",
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Generate 4-column visualization for predict batch."""
        self._visualize_batch(trainer, batch, "predict")

    def _visualize_batch(self, trainer: "Trainer", batch: "ImageBatch", stage: str) -> None:
        """Visualize batch and save images."""
        batch_size = batch.image.shape[0]

        for i in range(batch_size):
            # Get image path for filename
            if hasattr(batch, "image_path") and batch.image_path is not None:
                image_path = batch.image_path[i] if isinstance(batch.image_path, (list, tuple)) else batch.image_path
            else:
                image_path = f"sample_{i}"

            # Generate visualization
            vis_image = self._visualize_item(batch, i)

            if vis_image is not None:
                # Determine output path
                output_dir = self.output_dir or Path(trainer.default_root_dir) / "images"
                output_path = generate_output_filename(
                    input_path=image_path,
                    output_path=output_dir,
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                vis_image.save(output_path)

    def _visualize_item(self, batch: "ImageBatch", idx: int) -> PILImage.Image | None:
        """Create 4-column visualization for a single item.

        Args:
            batch: Image batch.
            idx: Index of item in batch.

        Returns:
            PIL Image with 4 columns or None if failed.
        """
        try:
            # Extract data for this item
            image = batch.image[idx].cpu().numpy()  # (C, H, W)
            image = np.transpose(image, (1, 2, 0))  # (H, W, C)

            # Denormalize if needed (assume ImageNet normalization)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image * std + mean
            image = np.clip(image, 0, 1)
            image_uint8 = (image * 255).astype(np.uint8)

            # Get anomaly map
            anomaly_map = None
            if hasattr(batch, "anomaly_map") and batch.anomaly_map is not None:
                amap = batch.anomaly_map[idx].cpu().numpy()
                if amap.ndim == 3:
                    amap = amap.squeeze(0)  # Remove channel dim
                anomaly_map = amap

            # Get pred mask
            pred_mask = None
            if hasattr(batch, "pred_mask") and batch.pred_mask is not None:
                pmask = batch.pred_mask[idx].cpu().numpy()
                if pmask.ndim == 3:
                    pmask = pmask.squeeze(0)
                pred_mask = pmask

            # Resize images to field_size
            w, h = self.field_size
            image_pil = PILImage.fromarray(image_uint8).resize((w, h), PILImage.Resampling.BILINEAR)
            image_resized = np.array(image_pil)

            # Create 4 columns
            columns = []

            # Column 1: Original Image
            columns.append(image_resized.copy())

            # Column 2: Image + Anomaly Map (Auto Scale)
            if anomaly_map is not None:
                amap_resized = np.array(PILImage.fromarray(anomaly_map).resize((w, h), PILImage.Resampling.BILINEAR))
                amap_auto = apply_colormap(amap_resized, normalize=True)
                col2 = blend_images(image_resized, amap_auto, self.alpha)
            else:
                col2 = image_resized.copy()
            columns.append(col2)

            # Column 3: Image + Anomaly Map (Fixed Scale 0-1)
            if anomaly_map is not None:
                amap_fixed = apply_colormap(amap_resized, normalize=False, vmin=0.0, vmax=1.0)
                col3 = blend_images(image_resized, amap_fixed, self.alpha)
            else:
                col3 = image_resized.copy()
            columns.append(col3)

            # Column 4: Image + Pred Mask
            if pred_mask is not None:
                mask_resized = np.array(PILImage.fromarray((pred_mask * 255).astype(np.uint8)).resize((w, h), PILImage.Resampling.NEAREST))
                mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
                mask_rgb[:, :, 0] = mask_resized  # Red channel
                col4 = blend_images(image_resized, mask_rgb, self.alpha)
            else:
                col4 = image_resized.copy()
            columns.append(col4)

            # Add labels
            labels = ["Image", "Anomaly (Auto)", "Anomaly (0-1)", "Pred Mask"]
            labeled_columns = []
            for col, label in zip(columns, labels):
                col_with_label = self._add_label(col, label)
                labeled_columns.append(col_with_label)

            # Concatenate horizontally
            combined = np.concatenate(labeled_columns, axis=1)
            return PILImage.fromarray(combined)

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to visualize item: {e}")
            return None

    def _add_label(self, image: np.ndarray, label: str, font_size: int = 16) -> np.ndarray:
        """Add text label to top of image.

        Args:
            image: Input image (H, W, 3).
            label: Text label.
            font_size: Font size.

        Returns:
            Image with label (H + label_height, W, 3).
        """
        from PIL import ImageDraw, ImageFont

        h, w = image.shape[:2]
        label_height = font_size + 10

        # Create canvas with label space
        canvas = np.ones((h + label_height, w, 3), dtype=np.uint8) * 255
        canvas[label_height:, :, :] = image

        # Draw label
        pil_img = PILImage.fromarray(canvas)
        draw = ImageDraw.Draw(pil_img)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        # Get text size and center it
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = (w - text_w) // 2
        text_y = 2

        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

        return np.array(pil_img)
