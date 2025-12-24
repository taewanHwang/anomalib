"""Dinomaly with GEM (Generalized Mean) Pooling.

This variant replaces the simple average pooling in anomaly map aggregation
with Generalized Mean Pooling (GEM), which can emphasize harder samples.

GEM formula: GEM(x) = (mean(x^p))^(1/p)
- p=1: equivalent to average pooling
- p→∞: approaches max pooling
- p=3: good balance (default)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from anomalib.data import InferenceBatch
from anomalib.models.image.dinomaly import Dinomaly
from anomalib.models.image.dinomaly.torch_model import (
    DEFAULT_GAUSSIAN_KERNEL_SIZE,
    DEFAULT_GAUSSIAN_SIGMA,
    DEFAULT_MAX_RATIO,
    DEFAULT_RESIZE_SIZE,
    DinomalyModel,
)


class DinomalyGEMModel(DinomalyModel):
    """Dinomaly model with GEM pooling for anomaly map aggregation.

    Instead of simple averaging across feature scales, this model uses
    Generalized Mean Pooling (GEM) which can better emphasize anomalous regions.

    Args:
        gem_p: Power parameter for GEM pooling. Higher values emphasize
            larger values (like max pooling). Default: 3.0
        learnable_gem_p: If True, gem_p becomes a learnable parameter.
        **kwargs: Arguments passed to parent DinomalyModel.
    """

    def __init__(
        self,
        gem_p: float = 3.0,
        learnable_gem_p: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.learnable_gem_p = learnable_gem_p

        if learnable_gem_p:
            # Initialize as learnable parameter
            self.gem_p = nn.Parameter(torch.tensor(gem_p))
        else:
            self.register_buffer("gem_p", torch.tensor(gem_p))

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

    def forward(self, batch: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """Forward pass with GEM pooling for inference.

        Args:
            batch: Input image batch

        Returns:
            Training: Tuple of (encoder_features, decoder_features)
            Inference: InferenceBatch with predictions
        """
        image_size = batch.shape[-1]
        en, de = self._encode_decode(batch)

        # During training, return features for loss calculation
        if self.training:
            return en, de

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
    for anomaly map aggregation.

    Args:
        gem_p: Power parameter for GEM pooling. Default: 3.0
        learnable_gem_p: If True, gem_p becomes learnable. Default: False
        **kwargs: Arguments passed to parent Dinomaly.
    """

    def __init__(
        self,
        gem_p: float = 3.0,
        learnable_gem_p: bool = False,
        **kwargs,
    ) -> None:
        self.gem_p = gem_p
        self.learnable_gem_p = learnable_gem_p
        super().__init__(**kwargs)

    def _setup(self) -> None:
        """Setup the GEM model instead of regular Dinomaly model."""
        self.model = DinomalyGEMModel(
            encoder_name=self.encoder_name,
            bottleneck_dropout=self.bottleneck_dropout,
            decoder_depth=self.decoder_depth,
            target_layers=self.target_layers,
            fuse_layer_encoder=self.fuse_layer_encoder,
            fuse_layer_decoder=self.fuse_layer_decoder,
            remove_class_token=self.remove_class_token,
            gem_p=self.gem_p,
            learnable_gem_p=self.learnable_gem_p,
        )
        self.loss = self.model.cosine_loss
