"""
Dinomaly Lightning Model with APE-based Adaptive Bottleneck Dropout.

This module provides a Lightning wrapper for DinomalyModelAdaptive,
which uses Angular Power Entropy (APE)-based adaptive dropout.

APE measures directional concentration of power in the frequency domain:
- Low APE (strong directional pattern) → Higher overfit risk → Higher dropout
- High APE (isotropic/complex pattern) → Lower overfit risk → Lower dropout
"""

import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib.models.components import AnomalibModule
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model_adaptive import DinomalyModelAdaptive
from .lightning_model import Dinomaly, DEFAULT_IMAGE_SIZE

logger = logging.getLogger(__name__)


class DinomalyAdaptive(Dinomaly):
    """
    Dinomaly model with APE-based adaptive bottleneck dropout for anomaly detection.

    This variant uses Angular Power Entropy (APE)-based adaptive dropout to prevent
    overfitting to highly regular normal patterns during long training.

    Key features:
    - Computes APE from input images (frequency domain directional analysis)
    - Maps APE to per-sample dropout probability relative to normal_ape
    - APE < normal_ape (more directional) → Higher dropout
    - APE > normal_ape (more isotropic) → Lower dropout

    This is particularly effective for datasets with:
    - Strong periodic/regular normal patterns (e.g., HDMAP PNG)
    - Long training requirements
    - Risk of anomaly contrast collapse

    Args:
        encoder_name (str): Name of the DINOv2 encoder. Options include
            "dinov2reg_vit_small_14", "dinov2reg_vit_base_14",
            "dinov2reg_vit_large_14". Default: "dinov2reg_vit_base_14".
        base_dropout (float): Base dropout probability (at normal_ape).
            Default: 0.3.
        min_dropout (float): Minimum dropout probability for isotropic samples.
            Default: 0.1.
        max_dropout (float): Maximum dropout probability for directional samples.
            Default: 0.6.
        dropout_sensitivity (float): Sensitivity of APE-to-dropout mapping.
            Higher values make dropout more responsive to APE. Default: 4.0.
            When sensitivity=0, dropout always equals base_dropout.
        normal_ape (float): Reference APE from normal training samples.
            Domain-specific defaults from EDA:
            - domain_A: 0.777
            - domain_B: 0.713
            - domain_C: 0.866
            - domain_D: 0.816
            Default: 0.78 (average).
        use_adaptive_dropout (bool): If False, uses fixed base_dropout.
            Useful for ablation studies. Default: True.
        decoder_depth (int): Number of decoder transformer blocks. Default: 8.
        target_layers (list[int] | None): Encoder layers for feature extraction.
        fuse_layer_encoder (list[list[int]] | None): Layer fusion config.
        fuse_layer_decoder (list[list[int]] | None): Layer fusion config.
        remove_class_token (bool): Whether to remove class token. Default: False.
        pre_processor (PreProcessor | bool): Pre-processor config. Default: True.
        post_processor (PostProcessor | bool): Post-processor config. Default: True.
        evaluator (Evaluator | bool): Evaluator config. Default: True.
        visualizer (Visualizer | bool): Visualizer config. Default: True.

    Example:
        >>> from anomalib.models.image.dinomaly import DinomalyAdaptive
        >>> model = DinomalyAdaptive(
        ...     encoder_name="dinov2reg_vit_large_14",
        ...     base_dropout=0.3,
        ...     min_dropout=0.1,
        ...     max_dropout=0.6,
        ...     dropout_sensitivity=4.0,
        ...     normal_ape=0.777,  # domain_A specific
        ... )

    Note:
        APE is computed in the frequency domain using 2D FFT. It measures
        how concentrated the power spectrum is along specific angles.
    """

    def __init__(
        self,
        encoder_name: str = "dinov2reg_vit_base_14",
        base_dropout: float = 0.3,
        min_dropout: float = 0.1,
        max_dropout: float = 0.6,
        dropout_sensitivity: float = 4.0,
        normal_ape: float = 0.78,
        use_adaptive_dropout: bool = True,
        decoder_depth: int = 8,
        target_layers: list[int] | None = None,
        fuse_layer_encoder: list[list[int]] | None = None,
        fuse_layer_decoder: list[list[int]] | None = None,
        remove_class_token: bool = False,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        # Initialize parent's parent (AnomalibModule) directly
        AnomalibModule.__init__(
            self,
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        # Store adaptive dropout config
        self.base_dropout = base_dropout
        self.min_dropout = min_dropout
        self.max_dropout = max_dropout
        self.dropout_sensitivity = dropout_sensitivity
        self.normal_ape = normal_ape
        self.use_adaptive_dropout = use_adaptive_dropout

        # Create adaptive model
        self.model: DinomalyModelAdaptive = DinomalyModelAdaptive(
            encoder_name=encoder_name,
            base_dropout=base_dropout,
            min_dropout=min_dropout,
            max_dropout=max_dropout,
            dropout_sensitivity=dropout_sensitivity,
            normal_ape=normal_ape,
            use_adaptive_dropout=use_adaptive_dropout,
            decoder_depth=decoder_depth,
            target_layers=target_layers,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            remove_class_token=remove_class_token,
        )

        # Freeze encoder, unfreeze bottleneck and decoder
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.bottleneck.parameters():
            param.requires_grad = True
        for param in self.model.decoder.parameters():
            param.requires_grad = True

        self.trainable_modules = torch.nn.ModuleList([
            self.model.bottleneck,
            self.model.decoder,
        ])
        self._initialize_trainable_modules(self.trainable_modules)

        # Logging interval for dropout stats
        self._log_interval = 100
        self._step_count = 0

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Training step with adaptive dropout monitoring."""
        del args, kwargs

        loss = self.model(batch.image, global_step=self.global_step)

        # Log adaptive dropout statistics periodically
        self._step_count += 1
        if self._step_count % self._log_interval == 0:
            stats = self.model.get_dropout_stats()
            if stats:
                # APE statistics
                self.log("train/ape_mean", stats["ape_mean"], prog_bar=False)
                self.log("train/ape_p10", stats.get("ape_p10", 0), prog_bar=False)
                self.log("train/ape_p50", stats.get("ape_p50", 0), prog_bar=False)
                self.log("train/ape_p90", stats.get("ape_p90", 0), prog_bar=False)

                # Dropout statistics
                self.log("train/dropout_mean", stats["dropout_mean"], prog_bar=False)
                self.log("train/dropout_p10", stats.get("dropout_p10", 0), prog_bar=False)
                self.log("train/dropout_p50", stats.get("dropout_p50", 0), prog_bar=False)
                self.log("train/dropout_p90", stats.get("dropout_p90", 0), prog_bar=False)
                self.log("train/dropout_min", stats.get("dropout_min", 0), prog_bar=False)
                self.log("train/dropout_max", stats.get("dropout_max", 0), prog_bar=False)

                self.model.reset_dropout_stats()

        self.log("train/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        """Log epoch-level dropout statistics."""
        stats = self.model.get_dropout_stats()
        if stats:
            logger.info(
                f"Epoch {self.current_epoch} - "
                f"APE: mean={stats['ape_mean']:.4f}, p10={stats.get('ape_p10', 0):.4f}, p90={stats.get('ape_p90', 0):.4f} | "
                f"Dropout: mean={stats['dropout_mean']:.4f}, p10={stats.get('dropout_p10', 0):.4f}, p90={stats.get('dropout_p90', 0):.4f}"
            )
            self.model.reset_dropout_stats()

    def on_validation_epoch_end(self) -> None:
        """Log validation AUROC to console after each validation."""
        # Access metrics from trainer's callback_metrics
        metrics = self.trainer.callback_metrics
        val_auroc = metrics.get("val_image_AUROC", None)
        if val_auroc is not None:
            logger.info(
                f"Step {self.global_step} - val_image_AUROC: {val_auroc:.4f} ({val_auroc*100:.2f}%)"
            )

    def on_test_epoch_end(self) -> None:
        """Log test AUROC to console after test."""
        metrics = self.trainer.callback_metrics
        test_auroc = metrics.get("test_image_AUROC", None)
        if test_auroc is not None:
            logger.info(
                f"TEST COMPLETE - test_image_AUROC: {test_auroc:.4f} ({test_auroc*100:.2f}%)"
            )


# Convenience function to create model
def create_dinomaly_adaptive(
    encoder_name: str = "dinov2reg_vit_large_14",
    base_dropout: float = 0.3,
    min_dropout: float = 0.1,
    max_dropout: float = 0.6,
    dropout_sensitivity: float = 4.0,
    normal_ape: float = 0.78,
    use_adaptive_dropout: bool = True,
    evaluator: Evaluator | None = None,
) -> DinomalyAdaptive:
    """
    Create DinomalyAdaptive model with common settings.

    Args:
        encoder_name: DINOv2 encoder name.
        base_dropout: Base dropout probability (at normal_ape).
        min_dropout: Minimum dropout.
        max_dropout: Maximum dropout.
        dropout_sensitivity: APE-to-dropout sensitivity.
        normal_ape: Reference APE from normal samples.
            Domain-specific defaults:
            - domain_A: 0.777
            - domain_B: 0.713
            - domain_C: 0.866
            - domain_D: 0.816
        use_adaptive_dropout: Enable/disable adaptive dropout.
        evaluator: Optional evaluator instance.

    Returns:
        Configured DinomalyAdaptive model.
    """
    return DinomalyAdaptive(
        encoder_name=encoder_name,
        base_dropout=base_dropout,
        min_dropout=min_dropout,
        max_dropout=max_dropout,
        dropout_sensitivity=dropout_sensitivity,
        normal_ape=normal_ape,
        use_adaptive_dropout=use_adaptive_dropout,
        evaluator=evaluator if evaluator else True,
        pre_processor=True,
    )
