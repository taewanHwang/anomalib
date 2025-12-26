# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dinomaly with Horizontal Segment Dropout + TopK Loss.

This variant combines:
1. HorizontalSegmentDropout in bottleneck (architectural constraint)
2. CosineTopKLoss for training (loss function constraint)

The hypothesis is that combining both approaches will be synergistic:
- Architectural constraint: Suppress horizontal reconstruction ability
- Loss constraint: Focus on tail distribution (hard examples)

This addresses the limitation of v3.0 where bottleneck-only dropout was
ineffective due to global attention in decoder layers.
"""

from __future__ import annotations

import torch
from torch import nn

from anomalib.models.image.dinomaly import Dinomaly
from anomalib.models.image.dinomaly_variants.horizontal_model import (
    DinomalyHorizontal,
    DinomalyHorizontalModel,
)
from anomalib.models.image.dinomaly_variants.topk_loss import CosineTopKLoss


class DinomalyHorizontalTopKModel(DinomalyHorizontalModel):
    """Dinomaly model with Horizontal Dropout + TopK Loss.

    Combines:
    - HorizontalSegmentMLP in bottleneck (from DinomalyHorizontalModel)
    - CosineTopKLoss for training (from DinomalyTopKModel)

    Args:
        q_percent: Percentage of top distance values to use (1-100). Default: 2.0
        q_schedule: Whether to use warmup schedule. Default: True
        warmup_steps: Number of warmup steps. Default: 200
        **kwargs: Arguments passed to DinomalyHorizontalModel.
    """

    def __init__(
        self,
        q_percent: float = 2.0,  # Best from v2 experiments
        q_schedule: bool = True,
        warmup_steps: int = 200,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Store TopK parameters
        self.q_percent = q_percent
        self.q_schedule = q_schedule
        self.warmup_steps = warmup_steps

        # Override loss function with TopK loss
        self.loss_fn = CosineTopKLoss(
            q_percent=q_percent,
            q_schedule=q_schedule,
            warmup_steps=warmup_steps,
        )


class DinomalyHorizontalTopK(DinomalyHorizontal):
    """Lightning wrapper for DinomalyHorizontalTopK model.

    Combines Horizontal Segment Dropout with TopK Loss for synergistic effect.

    v3.1 improvements over v3.0:
    - Uses TopK Loss (q=2%) which was proven effective in v2
    - Supports stronger dropout params (row_p, seg_len, seg_drop_p)
    - Combines architectural + loss constraints

    Args:
        q_percent: TopK loss q percentage. Default: 2.0 (best from v2)
        q_schedule: Whether to use warmup schedule. Default: True
        warmup_steps: Number of warmup steps. Default: 200
        elem_drop: Element-wise dropout probability. Default: 0.1
        row_p: Row selection probability for segment dropout. Default: 0.3 (increased from 0.2)
        seg_len: Segment length for segment dropout. Default: 3 (increased from 2)
        seg_drop_p: Drop probability within segments. Default: 0.8 (increased from 0.6)
        enable_segment_dropout: Whether to enable segment dropout. Default: True
        **kwargs: Additional arguments passed to parent.
    """

    def __init__(
        self,
        # TopK Loss params
        q_percent: float = 2.0,
        q_schedule: bool = True,
        warmup_steps: int = 200,
        # Horizontal Dropout params (v3.1 stronger defaults)
        elem_drop: float = 0.1,
        row_p: float = 0.3,  # v3.0: 0.2 -> v3.1: 0.3
        seg_len: int = 3,  # v3.0: 2 -> v3.1: 3
        seg_drop_p: float = 0.8,  # v3.0: 0.6 -> v3.1: 0.8
        enable_segment_dropout: bool = True,
        **kwargs,
    ) -> None:
        # Store TopK parameters
        self._q_percent = q_percent
        self._q_schedule = q_schedule
        self._warmup_steps = warmup_steps

        # Call parent init (DinomalyHorizontal)
        super().__init__(
            elem_drop=elem_drop,
            row_p=row_p,
            seg_len=seg_len,
            seg_drop_p=seg_drop_p,
            enable_segment_dropout=enable_segment_dropout,
            **kwargs,
        )

    def _setup_horizontal_model(self) -> None:
        """Replace with HorizontalTopK model that uses TopK loss."""
        self.model = DinomalyHorizontalTopKModel(
            encoder_name=self.encoder_name,
            elem_drop=self.elem_drop,
            row_p=self.row_p,
            seg_len=self.seg_len,
            seg_drop_p=self.seg_drop_p,
            enable_segment_dropout=self.enable_segment_dropout,
            decoder_depth=self.decoder_depth,
            target_layers=self.target_layers,
            fuse_layer_encoder=self.fuse_layer_encoder,
            fuse_layer_decoder=self.fuse_layer_decoder,
            remove_class_token=self.remove_class_token,
            image_size=self.image_size,
            # TopK params
            q_percent=self._q_percent,
            q_schedule=self._q_schedule,
            warmup_steps=self._warmup_steps,
        )

        # Freeze encoder, unfreeze bottleneck and decoder
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.bottleneck.parameters():
            param.requires_grad = True
        for param in self.model.decoder.parameters():
            param.requires_grad = True

        # Update trainable modules reference
        self.trainable_modules = torch.nn.ModuleList([self.model.bottleneck, self.model.decoder])

    def training_step(self, batch, *args, **kwargs):
        """Training step with TopK + Horizontal metrics logging."""
        result = super().training_step(batch, *args, **kwargs)

        # Log TopK config (every 100 steps)
        if self.global_step % 100 == 0:
            self.log("topk/q_percent", self._q_percent, on_step=True, on_epoch=False)

        return result

    @staticmethod
    def get_config_name(
        elem_drop: float,
        enable_segment: bool,
        row_p: float,
        seg_len: int,
        use_topk: bool = True,
    ) -> str:
        """Get configuration name for experiment tracking."""
        parts = []
        if use_topk:
            parts.append("TopK")
        if enable_segment:
            parts.append(f"Horiz_r{row_p}_s{seg_len}")
        else:
            parts.append(f"Elem{elem_drop}")
        return "_".join(parts)
