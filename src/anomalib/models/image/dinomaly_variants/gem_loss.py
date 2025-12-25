# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""GEM-aware Cosine Hard Mining Loss for training DinomalyGEM model.

This module implements a modified loss function that uses Generalized Mean (GEM)
pooling for scale aggregation during training. Unlike the original CosineHardMiningLoss
which processes scales independently and averages the losses, this version:

1. Computes distance maps for each scale
2. Aggregates them using GEM pooling (emphasizes harder scales)
3. Applies hard mining on the GEM-aggregated map

This allows the GEM aggregation to influence the training gradients.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineHardMiningGEMLoss(nn.Module):
    """Cosine similarity loss with GEM aggregation and hard mining.

    This loss function combines:
    1. Scale-wise distance map computation (1 - cosine_similarity)
    2. GEM aggregation over scales (instead of simple averaging)
    3. Hard mining to down-weight easy (well-reconstructed) points

    The GEM aggregation formula:
        GEM(x, p) = (mean(x^p))^(1/p)

    When p > 1, GEM emphasizes larger values, giving more weight to scales
    with higher reconstruction error.

    Args:
        gem_p: Power parameter for GEM pooling. Higher values emphasize
            larger values (like max pooling). Default: 3.0
        p_final: Final percentage of well-reconstructed points to down-weight.
            Default: 0.9 (down-weight 90% of easy points).
        p_schedule_steps: Number of steps to reach p_final. Default: 1000.
        factor: Gradient reduction factor for easy points. Default: 0.3.
            Note: 0.1 can be too aggressive for weak defects (domain_C).

    Attributes:
        metrics: Dictionary of last computed metrics for TensorBoard logging.
            Keys: easy_ratio, gem_mean, gem_std, gem_min, gem_max, thresh, p
    """

    def __init__(
        self,
        gem_p: float = 3.0,
        p_final: float = 0.9,
        p_schedule_steps: int = 1000,
        factor: float = 0.3,
    ) -> None:
        """Initialize the GEM loss."""
        super().__init__()

        self.gem_p = gem_p
        self.p_final = p_final
        self.p_schedule_steps = p_schedule_steps
        self.factor = factor
        self.p = 0.0  # Updated before loss calculation

        # Metrics for TensorBoard logging (updated each forward pass)
        self.metrics: dict[str, float] = {}

    def gem_aggregate(self, distance_maps: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Apply GEM aggregation over the scale dimension.

        Args:
            distance_maps: Tensor of shape (B, S, H, W) where S is the number of scales.
            eps: Small value for numerical stability.

        Returns:
            GEM-aggregated tensor of shape (B, 1, H, W).
        """
        # Clamp to avoid issues with negative values or zeros
        x = distance_maps.clamp(min=eps)
        # GEM: (mean(x^p))^(1/p)
        x_pow = x.pow(self.gem_p)
        x_mean = x_pow.mean(dim=1, keepdim=True)
        return x_mean.pow(1.0 / self.gem_p)

    def forward(
        self,
        encoder_features: list[torch.Tensor],
        decoder_features: list[torch.Tensor],
        global_step: int,
    ) -> torch.Tensor:
        """Forward pass of the GEM loss.

        Args:
            encoder_features: List of feature tensors from encoder layers.
                Each tensor has shape (B, C, H, W).
            decoder_features: List of corresponding decoder feature tensors.
            global_step: Current training step for p-schedule.

        Returns:
            Computed loss value.
        """
        # Update hard mining p value based on schedule
        self._update_p_schedule(global_step)

        # 1. Compute distance maps for each scale
        distance_maps = []
        for en, de in zip(encoder_features, decoder_features):
            # Detach encoder features to prevent gradient flow through encoder
            en_detached = en.detach()
            # Compute cosine distance: 1 - cosine_similarity
            # cosine_similarity returns (B, H, W), add channel dim -> (B, 1, H, W)
            cos_sim = F.cosine_similarity(en_detached, de, dim=1)  # (B, H, W)
            dist = 1.0 - cos_sim  # (B, H, W)
            distance_maps.append(dist.unsqueeze(1))  # (B, 1, H, W)

        # 2. Stack distance maps: (B, num_scales, H, W)
        stacked = torch.cat(distance_maps, dim=1)

        # 3. Apply GEM aggregation over scales
        gem_map = self.gem_aggregate(stacked)  # (B, 1, H, W)

        # 4. Compute hard mining threshold on GEM map (batch-global)
        # Keep top (1 - p) fraction as "hard" pixels, down-weight the rest (easy)
        with torch.no_grad():
            k = max(1, int(gem_map.numel() * (1 - self.p)))
            thresh = torch.topk(gem_map.reshape(-1), k=k)[0][-1]
            # easy_mask: points with low GEM distance (well-reconstructed)
            easy_mask = gem_map < thresh  # (B, 1, H, W)

            # Compute sanity metrics for TensorBoard logging
            easy_ratio = easy_mask.float().mean().item()
            self.metrics = {
                "easy_ratio": easy_ratio,
                "gem_mean": gem_map.mean().item(),
                "gem_std": gem_map.std().item(),
                "gem_min": gem_map.min().item(),
                "gem_max": gem_map.max().item(),
                "thresh": thresh.item(),
                "p": self.p,
            }

        # 5. Compute loss as mean of GEM map
        loss = gem_map.mean()

        # 6. Register gradient hook directly on gem_map (cleaner than stacked)
        # This down-weights gradients for easy pixels based on GEM-map threshold
        if gem_map.requires_grad:
            factor = self.factor  # Capture for closure

            def _hook_grad(grad: torch.Tensor) -> torch.Tensor:
                """Down-weight gradients for easy (low GEM distance) pixels."""
                if grad is None:
                    return grad
                g = grad.clone()
                g[easy_mask] = g[easy_mask] * factor
                return g

            gem_map.register_hook(_hook_grad)

        return loss

    def _update_p_schedule(self, global_step: int) -> None:
        """Update the p value based on training progress.

        Args:
            global_step: Current training step.
        """
        self.p = min(self.p_final * global_step / self.p_schedule_steps, self.p_final)
