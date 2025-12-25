"""Dinomaly with GEM (Generalized Mean) Pooling.

This variant replaces the simple average pooling in anomaly map aggregation
with Generalized Mean Pooling (GEM), which can emphasize harder samples.

GEM formula: GEM(x) = (mean(x^p))^(1/p)
- p=1: equivalent to average pooling
- p→∞: approaches max pooling
- p=3: good balance (default)

Key feature (v2): GEM is now applied during TRAINING via CosineHardMiningGEMLoss,
not just during inference. This allows GEM to influence the learned representations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from anomalib.data import InferenceBatch
from anomalib.models.image.dinomaly import Dinomaly
from anomalib.models.image.dinomaly.torch_model import (
    DEFAULT_MAX_RATIO,
    DEFAULT_RESIZE_SIZE,
    DinomalyModel,
)
from anomalib.models.image.dinomaly_variants.gem_loss import CosineHardMiningGEMLoss


class DinomalyGEMModel(DinomalyModel):
    """Dinomaly model with GEM pooling for anomaly map aggregation.

    This model uses GEM pooling in two places:
    1. Training: CosineHardMiningGEMLoss aggregates scale-wise distances with GEM
    2. Inference: Anomaly maps from different scales are aggregated with GEM

    This ensures GEM influences both training and inference.

    Args:
        gem_p: Power parameter for GEM pooling. Higher values emphasize
            larger values (like max pooling). Default: 3.0
        **kwargs: Arguments passed to parent DinomalyModel.
    """

    def __init__(
        self,
        gem_p: float = 3.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Store gem_p as buffer for inference
        self.register_buffer("gem_p", torch.tensor(gem_p))

        # Override the default loss with GEM-aware loss
        self.loss_fn = CosineHardMiningGEMLoss(gem_p=gem_p)

    def gem_pool(self, x: torch.Tensor, p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Apply Generalized Mean Pooling along channel dimension.

        Args:
            x: Input tensor of shape (B, C, H, W)
            p: Power parameter for GEM
            eps: Small value for numerical stability

        Returns:
            GEM pooled tensor of shape (B, 1, H, W)
        """
        # Clamp to avoid negative values (anomaly maps should be >= 0)
        x_clamped = x.clamp(min=eps)
        # GEM: (mean(x^p))^(1/p)
        x_pow = x_clamped.pow(p)
        x_mean = x_pow.mean(dim=1, keepdim=True)
        return x_mean.pow(1.0 / p)

    def calculate_anomaly_maps_gem(
        self,
        source_feature_maps: list[torch.Tensor],
        target_feature_maps: list[torch.Tensor],
        out_size: int | tuple[int, int] = 392,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Calculate anomaly maps with GEM pooling aggregation.

        Args:
            source_feature_maps: Encoder feature maps
            target_feature_maps: Decoder feature maps
            out_size: Output size for anomaly maps

        Returns:
            Tuple of (aggregated_anomaly_map, list_of_scale_anomaly_maps)
        """
        if not isinstance(out_size, tuple):
            out_size = (out_size, out_size)

        anomaly_map_list = []
        for i in range(len(target_feature_maps)):
            fs = source_feature_maps[i]
            ft = target_feature_maps[i]
            a_map = 1 - F.cosine_similarity(fs, ft)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=out_size, mode="bilinear", align_corners=True)
            anomaly_map_list.append(a_map)

        # Stack and apply GEM pooling instead of mean
        stacked_maps = torch.cat(anomaly_map_list, dim=1)  # (B, num_scales, H, W)
        anomaly_map = self.gem_pool(stacked_maps, self.gem_p)  # (B, 1, H, W)

        return anomaly_map, anomaly_map_list

    def forward(self, batch: torch.Tensor, global_step: int | None = None) -> torch.Tensor | InferenceBatch:
        """Forward pass with GEM pooling for inference.

        Args:
            batch: Input image batch
            global_step: Current training step, used for loss computation

        Returns:
            Training: Loss tensor from CosineHardMiningGEMLoss
            Inference: InferenceBatch with predictions
        """
        image_size = batch.shape[-1]
        en, de = self.get_encoder_decoder_outputs(batch)

        # During training, compute and return the GEM loss
        if self.training:
            if global_step is None:
                error_msg = "global_step must be provided during training"
                raise ValueError(error_msg)
            return self.loss_fn(encoder_features=en, decoder_features=de, global_step=global_step)

        # Use GEM pooling for anomaly map aggregation
        anomaly_map, _ = self.calculate_anomaly_maps_gem(en, de, out_size=image_size)
        anomaly_map_resized = anomaly_map.clone()

        # Resize anomaly map for processing
        if DEFAULT_RESIZE_SIZE is not None:
            anomaly_map = F.interpolate(
                anomaly_map, size=DEFAULT_RESIZE_SIZE, mode="bilinear", align_corners=False
            )

        # Apply Gaussian smoothing
        anomaly_map = self.gaussian_blur(anomaly_map)

        # Calculate anomaly score
        if DEFAULT_MAX_RATIO == 0:
            sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
        else:
            anomaly_map_flat = anomaly_map.flatten(1)
            sp_score = torch.sort(anomaly_map_flat, dim=1, descending=True)[0][
                :,
                : int(anomaly_map_flat.shape[1] * DEFAULT_MAX_RATIO),
            ]
            sp_score = sp_score.mean(dim=1)
        pred_score = sp_score

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map_resized)


class DinomalyGEM(Dinomaly):
    """Lightning wrapper for DinomalyGEM model.

    This is a drop-in replacement for Dinomaly that uses GEM pooling
    for anomaly map aggregation. GEM is applied both during training
    (via CosineHardMiningGEMLoss) and inference (in anomaly map aggregation).

    Args:
        gem_p: Power parameter for GEM pooling. Default: 3.0
        gem_factor: Gradient reduction factor for easy points in hard mining.
            Default: 0.3. Lower = more aggressive down-weighting.
        **kwargs: Arguments passed to parent Dinomaly.
    """

    def __init__(
        self,
        encoder_name: str = "dinov2reg_vit_base_14",
        bottleneck_dropout: float = 0.2,
        decoder_depth: int = 8,
        target_layers: list[int] | None = None,
        fuse_layer_encoder: list[list[int]] | None = None,
        fuse_layer_decoder: list[list[int]] | None = None,
        remove_class_token: bool = False,
        gem_p: float = 3.0,
        gem_factor: float = 0.3,
        **kwargs,
    ) -> None:
        # Store parameters for _setup_gem_model
        self.encoder_name = encoder_name
        self.bottleneck_dropout = bottleneck_dropout
        self.decoder_depth = decoder_depth
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.gem_p = gem_p
        self.gem_factor = gem_factor

        # Call parent init (creates DinomalyModel)
        super().__init__(
            encoder_name=encoder_name,
            bottleneck_dropout=bottleneck_dropout,
            decoder_depth=decoder_depth,
            target_layers=target_layers,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            remove_class_token=remove_class_token,
            **kwargs,
        )

        # Replace DinomalyModel with DinomalyGEMModel
        self._setup_gem_model()

    def _setup_gem_model(self) -> None:
        """Replace the default DinomalyModel with DinomalyGEMModel.

        This is called after super().__init__() to replace the model
        with our GEM-enabled variant that uses CosineHardMiningGEMLoss.
        """
        self.model = DinomalyGEMModel(
            encoder_name=self.encoder_name,
            bottleneck_dropout=self.bottleneck_dropout,
            decoder_depth=self.decoder_depth,
            target_layers=self.target_layers,
            fuse_layer_encoder=self.fuse_layer_encoder,
            fuse_layer_decoder=self.fuse_layer_decoder,
            remove_class_token=self.remove_class_token,
            gem_p=self.gem_p,
        )
        # Override factor in the loss function
        self.model.loss_fn.factor = self.gem_factor
        # Use GEM-aware loss for training
        self.loss = self.model.loss_fn

        # Freeze encoder, unfreeze bottleneck and decoder (same as parent)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.bottleneck.parameters():
            param.requires_grad = True
        for param in self.model.decoder.parameters():
            param.requires_grad = True

        # Update trainable modules reference
        self.trainable_modules = torch.nn.ModuleList([self.model.bottleneck, self.model.decoder])

    def training_step(self, batch, *args, **kwargs):
        """Training step with GEM loss metrics logging.

        Extends parent training_step to log GEM-specific sanity metrics:
        - gem/easy_ratio: Should match p schedule
        - gem/gem_mean, gem_std, gem_min, gem_max: GEM map statistics
        - gem/thresh: Hard mining threshold
        - gem/p: Current hard mining proportion
        """
        # Call parent training_step
        result = super().training_step(batch, *args, **kwargs)

        # Log GEM loss metrics (every 50 steps to reduce overhead)
        if self.global_step % 50 == 0 and hasattr(self.model, "loss_fn"):
            metrics = self.model.loss_fn.metrics
            if metrics:
                for key, value in metrics.items():
                    self.log(f"gem/{key}", value, on_step=True, on_epoch=False)

        return result
