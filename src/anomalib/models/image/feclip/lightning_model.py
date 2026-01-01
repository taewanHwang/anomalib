"""FE-CLIP: Frequency Enhanced CLIP Model for Zero-Shot Anomaly Detection.

This module implements the FE-CLIP model for zero-shot anomaly detection using
CLIP embeddings enhanced with frequency-aware adapters (FFE and LFS).

The model can perform both anomaly classification and segmentation tasks by
injecting frequency information into CLIP's visual encoder.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.engine import Engine
    >>> from anomalib.models.image import FEClip

    >>> datamodule = MVTecAD(root="./datasets/MVTecAD")  # doctest: +SKIP
    >>> model = FEClip()  # doctest: +SKIP

    >>> # Training (fine-tuning adapters)
    >>> Engine.fit(model=model, datamodule=datamodule)  # doctest: +SKIP

    >>> # Testing
    >>> Engine.test(model=model, datamodule=datamodule)  # doctest: +SKIP

Paper:
    FE-CLIP: Frequency Enhanced CLIP Model for Zero-Shot Anomaly Detection
"""

import logging
from typing import Any

import torch
from torch import nn, optim
from torchvision.transforms.v2 import CenterCrop, Compose, InterpolationMode, Normalize, Resize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .losses import bce_loss, focal_loss, dice_loss
from .torch_model import FEClipModel

logger = logging.getLogger(__name__)

__all__ = ["FEClip"]

# Default image size for FE-CLIP (ViT-L-14-336)
IMAGE_SIZE = 336


class FEClip(AnomalibModule):
    """FE-CLIP Lightning model.

    This model implements the FE-CLIP algorithm for zero-shot anomaly detection using
    CLIP embeddings enhanced with frequency-aware adapters.

    Args:
        backbone: CLIP backbone model name. Default: "ViT-L-14-336".
        pretrained: Pretrained weights to use. Default: "openai".
        n_taps: Number of tap blocks for adapter injection. Default: 4.
        lambda_fuse: Fusion weight for adapter outputs. Default: 0.1.
        P: Window size for FFE adapter. Default: 3.
        Q: Window size for LFS adapter. Default: 3.
        lr: Learning rate for adapter training. Default: 5e-4.
        w_cls: Weight for classification loss. Default: 1.0.
            Paper uses L_total = L_cls + L_mask (w_cls=1, w_mask=1).
        w_mask: Weight for segmentation loss. Default: 1.0.
        init_fc_patch_from_clip: Initialize fc_patch from visual.proj. Default: False.
            WARNING: Setting True HURTS zero-shot pixel performance significantly.
        pre_processor: Pre-processor instance or flag to use default. Default: True.
        post_processor: Post-processor instance or flag to use default. Default: True.
        evaluator: Evaluator instance or flag to use default. Default: True.
        visualizer: Visualizer instance or flag to use default. Default: True.

    Example:
        >>> from anomalib.models.image import FEClip
        >>> # Default configuration
        >>> model = FEClip()  # doctest: +SKIP
        >>> # Custom learning rate
        >>> model = FEClip(lr=1e-4)  # doctest: +SKIP

    Notes:
        - Input image size is fixed at 336x336 for ViT-L-14-336 backbone
        - Uses CLIP-specific normalization
        - CLIP weights are frozen, only adapters and fc_patch are trainable
    """

    def __init__(
        self,
        backbone: str = "ViT-L-14-336",
        pretrained: str = "openai",
        n_taps: int = 4,
        tap_indices: list[int] | None = None,
        lambda_fuse: float = 0.1,
        P: int = 3,
        Q: int = 3,
        lfs_agg_mode: str = "mean",
        lr: float = 5e-4,
        w_cls: float = 1.0,
        w_mask: float = 1.0,
        use_clip_logit_scale: bool = False,
        freeze_fc_patch: bool = False,
        init_fc_patch_from_clip: bool = False,
        pre_processor: nn.Module | bool = True,
        post_processor: nn.Module | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.model = FEClipModel(
            backbone=backbone,
            pretrained=pretrained,
            n_taps=n_taps,
            tap_indices=tap_indices,
            lambda_fuse=lambda_fuse,
            P=P,
            Q=Q,
            lfs_agg_mode=lfs_agg_mode,
            use_clip_logit_scale=use_clip_logit_scale,
            freeze_fc_patch=freeze_fc_patch,
            init_fc_patch_from_clip=init_fc_patch_from_clip,
        )
        self.lr = lr
        self.w_cls = w_cls
        self.w_mask = w_mask
        self.is_setup = False

    def setup(self, stage: str) -> None:
        """Setup FE-CLIP model.

        This method initializes the text embeddings used for anomaly score computation.

        Args:
            stage: Current stage (fit, validate, test, predict).
        """
        del stage
        if self.is_setup:
            return

        # Initialize text embeddings
        self.model.setup_text()
        self.is_setup = True

    def configure_optimizers(self) -> optim.Optimizer:
        """Configure optimizers for adapter training.

        Only the adapters (FFE, LFS) and fc_patch are trainable.
        CLIP parameters remain frozen.
        Paper uses Adam optimizer (not AdamW).

        Returns:
            Adam optimizer for trainable parameters.
        """
        # Collect trainable parameters (adapters and fc_patch)
        params = [p for p in self.model.parameters() if p.requires_grad]
        return optim.Adam(params, lr=self.lr)

    def training_step(self, batch: Batch, *args, **kwargs) -> torch.Tensor:
        """Training step for FE-CLIP with per-tap loss (Eq.4/5 in paper).

        Computes the loss for adapter training:
        - L_cls: BCE loss averaged across N tap blocks (Eq.4)
        - L_mask: (Focal + Dice) loss averaged across N tap blocks (Eq.5)

        IMPORTANT: Paper computes loss on each tap separately, then averages.
        This is different from computing loss on averaged predictions:
        mean(BCE(s_n)) ≠ BCE(mean(s_n))

        Args:
            batch: Input batch containing image, gt_label, and optionally gt_mask.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Total loss value.
        """
        del args, kwargs

        # Forward pass - get per-tap scores and maps
        scores, maps = self.model.forward_tokens(batch.image)
        # scores: list of (B,) tensors, one per tap block
        # maps: list of (B, Ht, Wt) tensors, one per tap block

        # Prepare ground truth
        gt_label = batch.gt_label.float() if hasattr(batch, "gt_label") and batch.gt_label is not None else None
        gt_mask = None
        if hasattr(batch, "gt_mask") and batch.gt_mask is not None:
            gt_mask = batch.gt_mask.float()
            if gt_mask.ndim == 4:
                gt_mask = gt_mask.squeeze(1)

        if gt_label is None:
            return torch.tensor(0.0, device=batch.image.device, requires_grad=True)

        n_taps = len(scores)
        loss_dict = {}

        # Per-tap classification loss (Eq.4): L_cls = (1/N) * Σ BCE(S_a,n, S_gt)
        loss_cls = torch.tensor(0.0, device=batch.image.device)
        for s in scores:
            loss_cls = loss_cls + bce_loss(s, gt_label)
        loss_cls = loss_cls / n_taps
        loss_dict["loss_cls"] = loss_cls

        total_loss = self.w_cls * loss_cls

        # Per-tap mask loss (Eq.5): L_mask = (1/N) * Σ (Focal + Dice)(Up(M_a,n), M_gt)
        if gt_mask is not None:
            loss_focal_total = torch.tensor(0.0, device=batch.image.device)
            loss_dice_total = torch.tensor(0.0, device=batch.image.device)

            for m in maps:
                # Upsample each tap's map to full resolution
                m_up = torch.nn.functional.interpolate(
                    m.unsqueeze(1),
                    size=batch.image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)  # (B, H, W)

                loss_focal_total = loss_focal_total + focal_loss(m_up, gt_mask)
                loss_dice_total = loss_dice_total + dice_loss(m_up, gt_mask)

            loss_focal = loss_focal_total / n_taps
            loss_dice = loss_dice_total / n_taps
            loss_mask = loss_focal + loss_dice

            loss_dict["loss_focal"] = loss_focal
            loss_dict["loss_dice"] = loss_dice
            loss_dict["loss_mask"] = loss_mask
            total_loss = total_loss + self.w_mask * loss_mask

        loss_dict["loss_total"] = total_loss

        # Log losses
        for name, value in loss_dict.items():
            self.log(f"train_{name}", value, prog_bar=name == "loss_total")

        return total_loss

    def test_step(self, batch: Batch, batch_idx: int, *args, **kwargs) -> dict:
        """Test step for FE-CLIP with debug logging."""
        del args, kwargs, batch_idx

        # Debug: Log batch.image shape
        if not hasattr(self, "_logged_test_shape"):
            logger.info(f"[DEBUG test_step] batch.image shape: {batch.image.shape}")
            logger.info(f"[DEBUG test_step] batch.image min/max: {batch.image.min():.4f}/{batch.image.max():.4f}")
            self._logged_test_shape = True

        predictions = self.model(batch.image)

        # Debug: Log pred_score stats
        if not hasattr(self, "_logged_test_pred"):
            logger.info(f"[DEBUG test_step] pred_score: mean={predictions.pred_score.mean():.4f}, std={predictions.pred_score.std():.4f}")
            logger.info(f"[DEBUG test_step] pred_score values: {predictions.pred_score[:5]}")
            self._logged_test_pred = True

        return batch.update(**predictions._asdict())

    def validation_step(self, batch: Batch, *args, **kwargs) -> dict:
        """Validation step for FE-CLIP.

        Args:
            batch: Input batch.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Dictionary containing the batch updated with predictions.
        """
        del args, kwargs

        # Debug: Log batch.image shape to verify preprocessing
        if not hasattr(self, "_logged_shape"):
            logger.info(f"[DEBUG] validation_step batch.image shape: {batch.image.shape}")
            self._logged_shape = True

        predictions = self.model(batch.image)

        # Debug: Log pred_score stats on first batch
        if not hasattr(self, "_logged_pred"):
            logger.info(f"[DEBUG] pred_score: mean={predictions.pred_score.mean():.4f}, std={predictions.pred_score.std():.4f}")
            self._logged_pred = True

        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Get model-specific trainer arguments.

        Returns:
            Dictionary with trainer arguments.
        """
        return {}

    @property
    def learning_type(self) -> LearningType:
        """Get the learning type of the model.

        Returns:
            LearningType.ONE_CLASS as FE-CLIP trains on normal samples with adapter fine-tuning.
        """
        return LearningType.ONE_CLASS

    @classmethod
    def configure_pre_processor(cls, image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Configure the default pre-processor for FE-CLIP.

        Uses CLIP-style preprocessing which preserves aspect ratio:
        1. Resize shorter edge to IMAGE_SIZE (maintains aspect ratio)
        2. CenterCrop to (IMAGE_SIZE, IMAGE_SIZE)
        3. CLIP normalization

        Args:
            image_size: Not used as FE-CLIP has fixed input size (336x336).

        Returns:
            PreProcessor with CLIP-specific transforms.
        """
        if image_size is not None:
            logger.warning("Image size is not used in FE-CLIP. The input size is fixed at 336x336.")

        # CLIP-style: Resize shorter edge + CenterCrop (preserves aspect ratio)
        transform = Compose([
            Resize(IMAGE_SIZE, antialias=True, interpolation=InterpolationMode.BICUBIC),
            CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
            Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        return PreProcessor(transform=transform)

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        """Configure the default post-processor for FE-CLIP.

        Returns:
            Default PostProcessor instance.
        """
        return PostProcessor()
