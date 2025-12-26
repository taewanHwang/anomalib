# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dinomaly with Top-q% Loss for Tail-Focused Learning.

This variant replaces the standard reconstruction loss with CosineTopKLoss,
which focuses on the top q% of distance values during training.

Key insight: The evaluation metric (TPR@FPR=1%) cares about the tail of the
score distribution. By training to focus on high-distance regions, we align
the training objective with the evaluation metric.

This is particularly useful for datasets like HDMAP domain_C where:
- Defects are weak (low contrast with normal patterns)
- Normal patterns are strong (high reconstruction for most pixels)
- The overlap between Good and Fault score distributions is in the tail
"""

from __future__ import annotations

import torch

from anomalib.models.image.dinomaly import Dinomaly
from anomalib.models.image.dinomaly.torch_model import DinomalyModel
from anomalib.models.image.dinomaly_variants.topk_loss import CosineTopKLoss


class DinomalyTopKModel(DinomalyModel):
    """Dinomaly model with Top-q% focused loss.

    This model uses CosineTopKLoss instead of the default loss:
    - Training: Only top q% of distance values contribute to loss
    - Inference: Unchanged from baseline (same anomaly map computation)

    Args:
        q_percent: Percentage of top distance values to use (1-100). Default: 5.0
        q_schedule: Whether to use warmup schedule. Default: True
        warmup_steps: Number of warmup steps. Default: 200
        **kwargs: Arguments passed to parent DinomalyModel.
    """

    def __init__(
        self,
        q_percent: float = 5.0,
        q_schedule: bool = True,
        warmup_steps: int = 200,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Store parameters
        self.q_percent = q_percent
        self.q_schedule = q_schedule
        self.warmup_steps = warmup_steps

        # Override the default loss with TopK loss
        self.loss_fn = CosineTopKLoss(
            q_percent=q_percent,
            q_schedule=q_schedule,
            warmup_steps=warmup_steps,
        )

    def forward(self, batch: torch.Tensor, global_step: int | None = None):
        """Forward pass.

        Args:
            batch: Input image batch
            global_step: Current training step, used for loss computation

        Returns:
            Training: Loss tensor from CosineTopKLoss
            Inference: InferenceBatch with predictions
        """
        # Get encoder and decoder outputs
        en, de = self.get_encoder_decoder_outputs(batch)

        # During training, compute and return the TopK loss
        if self.training:
            if global_step is None:
                error_msg = "global_step must be provided during training"
                raise ValueError(error_msg)
            return self.loss_fn(encoder_features=en, decoder_features=de, global_step=global_step)

        # Inference: use parent's anomaly map computation (unchanged)
        return super().forward(batch)


class DinomalyTopK(Dinomaly):
    """Lightning wrapper for DinomalyTopK model.

    This is a drop-in replacement for Dinomaly that uses Top-q% focused loss.
    Only the training loss is modified; inference is identical to baseline.

    Args:
        q_percent: Percentage of top distance values to use (1-100). Default: 5.0
        q_schedule: Whether to use warmup schedule. Default: True
        warmup_steps: Number of warmup steps. Default: 200
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
        q_percent: float = 5.0,
        q_schedule: bool = True,
        warmup_steps: int = 200,
        **kwargs,
    ) -> None:
        # Store parameters for _setup_topk_model
        self.encoder_name = encoder_name
        self.bottleneck_dropout = bottleneck_dropout
        self.decoder_depth = decoder_depth
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.q_percent = q_percent
        self.q_schedule = q_schedule
        self.warmup_steps = warmup_steps

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

        # Replace DinomalyModel with DinomalyTopKModel
        self._setup_topk_model()

    def _setup_topk_model(self) -> None:
        """Replace the default DinomalyModel with DinomalyTopKModel.

        This is called after super().__init__() to replace the model
        with our TopK-loss variant.
        """
        self.model = DinomalyTopKModel(
            encoder_name=self.encoder_name,
            bottleneck_dropout=self.bottleneck_dropout,
            decoder_depth=self.decoder_depth,
            target_layers=self.target_layers,
            fuse_layer_encoder=self.fuse_layer_encoder,
            fuse_layer_decoder=self.fuse_layer_decoder,
            remove_class_token=self.remove_class_token,
            q_percent=self.q_percent,
            q_schedule=self.q_schedule,
            warmup_steps=self.warmup_steps,
        )
        # Use TopK loss for training
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
        """Training step with TopK loss metrics logging.

        Extends parent training_step to log TopK-specific metrics:
        - topk/q_current: Current q percentage
        - topk/k_selected: Number of elements selected
        - topk/dist_mean: Mean of full distance map
        - topk/top_mean: Mean of selected top values
        """
        # Call parent training_step
        result = super().training_step(batch, *args, **kwargs)

        # Log TopK loss metrics (every 50 steps to reduce overhead)
        if self.global_step % 50 == 0 and hasattr(self.model, "loss_fn"):
            metrics = self.model.loss_fn.metrics
            if metrics:
                for key, value in metrics.items():
                    self.log(f"topk/{key}", value, on_step=True, on_epoch=False)

        return result
