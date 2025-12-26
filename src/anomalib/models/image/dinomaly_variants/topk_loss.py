# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Top-q% Focused Cosine Loss for training DinomalyTopK model.

This module implements a loss function that focuses on the top q% of distance values
during training. Unlike the original CosineHardMiningLoss which averages all pixels,
this version:

1. Computes distance maps for each scale
2. Averages them across scales
3. Selects only the top q% of distance values for loss computation

This allows the model to focus on reconstructing the hardest (highest distance) regions,
which is particularly useful for datasets with weak anomalies (like HDMAP domain_C).

The key insight is that the evaluation metric (TPR@FPR=1%) cares about the tail of
the score distribution, so aligning the training objective with the evaluation metric
can improve low-FPR performance.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineTopKLoss(nn.Module):
    """Cosine similarity loss focusing on top q% of distance values.

    This loss function:
    1. Computes scale-wise distance maps (1 - cosine_similarity)
    2. Averages them across scales
    3. Selects top q% of distance values
    4. Computes loss as mean of selected values

    By focusing on the top q% (hardest to reconstruct regions), the model
    learns to better separate anomalies from normal samples at low FPR.

    Args:
        q_percent: Percentage of top distance values to use (1-100).
            Default: 5.0 (top 5% of pixels).
        q_schedule: Whether to use a warmup schedule. Default: True.
            If True, starts with 100% and gradually decreases to q_percent.
        warmup_steps: Number of warmup steps before reaching target q.
            Default: 200.
        min_k: Minimum number of elements to select. Default: 100.

    Attributes:
        metrics: Dictionary of last computed metrics for TensorBoard logging.
            Keys: q_current, k_selected, dist_mean, dist_std, top_mean, top_std
    """

    def __init__(
        self,
        q_percent: float = 5.0,
        q_schedule: bool = True,
        warmup_steps: int = 200,
        min_k: int = 100,
    ) -> None:
        """Initialize the TopK loss."""
        super().__init__()

        self.q_percent = q_percent
        self.q_schedule = q_schedule
        self.warmup_steps = warmup_steps
        self.min_k = min_k

        # Metrics for TensorBoard logging (updated each forward pass)
        self.metrics: dict[str, float] = {}

    def forward(
        self,
        encoder_features: list[torch.Tensor],
        decoder_features: list[torch.Tensor],
        global_step: int,
    ) -> torch.Tensor:
        """Forward pass of the TopK loss.

        Args:
            encoder_features: List of feature tensors from encoder layers.
                Each tensor has shape (B, C, H, W).
            decoder_features: List of corresponding decoder feature tensors.
            global_step: Current training step for q-schedule.

        Returns:
            Computed loss value.
        """
        # 1. Compute distance maps for each scale
        distance_maps = []
        for en, de in zip(encoder_features, decoder_features):
            # Detach encoder features to prevent gradient flow through encoder
            en_detached = en.detach()
            # Compute cosine distance: 1 - cosine_similarity
            cos_sim = F.cosine_similarity(en_detached, de, dim=1)  # (B, H, W)
            dist = 1.0 - cos_sim  # (B, H, W)
            distance_maps.append(dist)

        # 2. Average distance maps across scales: (B, H, W)
        dist_map = torch.stack(distance_maps, dim=0).mean(dim=0)

        # 3. Get current q and compute k (number of elements to select)
        q_current = self._get_current_q(global_step)
        total_elements = dist_map.numel()
        k = max(self.min_k, int(total_elements * q_current / 100))

        # 4. Select top k elements (highest distance values)
        dist_flat = dist_map.reshape(-1)
        top_values, _ = torch.topk(dist_flat, k=k, largest=True, sorted=False)

        # 5. Compute loss as mean of top-k values
        loss = top_values.mean()

        # 6. Update metrics for logging
        with torch.no_grad():
            self.metrics = {
                "q_current": q_current,
                "k_selected": k,
                "k_ratio": k / total_elements * 100,
                "dist_mean": dist_map.mean().item(),
                "dist_std": dist_map.std().item(),
                "dist_max": dist_map.max().item(),
                "top_mean": top_values.mean().item(),
                "top_min": top_values.min().item(),
            }

        return loss

    def _get_current_q(self, global_step: int) -> float:
        """Get the current q percentage based on training progress.

        Args:
            global_step: Current training step.

        Returns:
            Current q percentage (1-100).
        """
        if not self.q_schedule:
            return self.q_percent

        # Warmup: 100% -> q_percent over warmup_steps
        if global_step < self.warmup_steps:
            # Linear interpolation from 100 to q_percent
            progress = global_step / self.warmup_steps
            return 100.0 - (100.0 - self.q_percent) * progress
        return self.q_percent
