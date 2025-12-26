# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dinomaly with Horizontal Segment Dropout for Direction-Aware Regularization.

This variant replaces the standard bottleneck MLP with HorizontalSegmentMLP,
which applies spatially-aware dropout that targets horizontal token segments.

Key insight: Domain C defects appear as horizontal patterns. Standard dropout
treats all tokens independently, allowing the decoder to use horizontal neighbors
to reconstruct (and thus hide) these patterns. By dropping consecutive horizontal
tokens, we force the decoder to be less proficient at horizontal reconstruction,
making horizontal defects more detectable.

Ablation configurations:
    A. Baseline: elem_p=0.2, no segment dropout (original Dinomaly)
    B. Element↓: elem_p=0.1, no segment dropout (reduced regularization)
    C. Segment Only: elem_p=0.0, segment dropout enabled
    D. Hybrid: elem_p=0.1, segment dropout enabled (recommended)
"""

from __future__ import annotations

from functools import partial

import torch
from torch import nn

from anomalib.models.image.dinomaly import Dinomaly
from anomalib.models.image.dinomaly.torch_model import (
    DEFAULT_FUSE_LAYERS,
    DINOV2_ARCHITECTURES,
    TRANSFORMER_CONFIG,
    DecoderViTBlock,
    DinomalyModel,
)
from anomalib.models.image.dinomaly.components import load as load_dinov2_model
from anomalib.models.image.dinomaly.components.layers import LinearAttention
from anomalib.models.image.dinomaly_variants.horizontal_dropout import HorizontalSegmentMLP


class DinomalyHorizontalModel(DinomalyModel):
    """Dinomaly model with Horizontal Segment Dropout in bottleneck.

    This model replaces the bottleneck MLP with HorizontalSegmentMLP that applies:
    1. Horizontal Segment Dropout: Drops consecutive tokens within rows
    2. Element Dropout: Standard element-wise dropout (at reduced rate)

    The segment dropout suppresses the decoder's ability to use horizontal
    neighbor information, making horizontal defect patterns more detectable.

    Args:
        encoder_name: Name of the encoder model
        elem_drop: Element-wise dropout probability (default: 0.1)
        row_p: Probability of applying segment dropout per row (default: 0.2)
        seg_len: Length of consecutive tokens to drop (default: 2)
        seg_drop_p: Probability of dropping each token in segment (default: 0.6)
        enable_segment_dropout: Whether to enable segment dropout (default: True)
        decoder_depth: Number of decoder layers (default: 8)
        target_layers: Encoder layers to extract features from
        fuse_layer_encoder: Layer groupings for encoder feature fusion
        fuse_layer_decoder: Layer groupings for decoder feature fusion
        remove_class_token: Whether to remove class token
    """

    def __init__(
        self,
        encoder_name: str = "dinov2reg_vit_base_14",
        elem_drop: float = 0.1,
        row_p: float = 0.2,
        seg_len: int = 2,
        seg_drop_p: float = 0.6,
        enable_segment_dropout: bool = True,
        decoder_depth: int = 8,
        target_layers: list[int] | None = None,
        fuse_layer_encoder: list[list[int]] | None = None,
        fuse_layer_decoder: list[list[int]] | None = None,
        remove_class_token: bool = True,  # Must be True for HorizontalSegmentDropout
        image_size: int = 392,  # Input image size for calculating token grid side
    ) -> None:
        # Initialize nn.Module directly (skip DinomalyModel.__init__)
        nn.Module.__init__(self)

        # Calculate side from image_size and patch_size (14 for ViT-Base/14)
        patch_size = 14
        self.side = image_size // patch_size  # e.g., 392/14 = 28

        # Store parameters
        self.elem_drop = elem_drop
        self.row_p = row_p
        self.seg_len = seg_len
        self.seg_drop_p = seg_drop_p
        self.enable_segment_dropout = enable_segment_dropout

        if target_layers is None:
            target_layers = [2, 3, 4, 5, 6, 7, 8, 9]

        if fuse_layer_encoder is None:
            fuse_layer_encoder = DEFAULT_FUSE_LAYERS
        if fuse_layer_decoder is None:
            fuse_layer_decoder = DEFAULT_FUSE_LAYERS

        # Load encoder
        encoder = load_dinov2_model(encoder_name)

        # Extract architecture config
        arch_config = self._get_architecture_config(encoder_name, target_layers)
        embed_dim = arch_config["embed_dim"]
        num_heads = arch_config["num_heads"]
        target_layers = arch_config["target_layers"]

        if decoder_depth <= 1:
            msg = f"decoder_depth must be greater than 1, got {decoder_depth}"
            raise ValueError(msg)

        # Create bottleneck with HorizontalSegmentMLP
        bottleneck = []
        bottle_neck_mlp = HorizontalSegmentMLP(
            in_features=embed_dim,
            hidden_features=embed_dim * 4,
            out_features=embed_dim,
            act_layer=nn.GELU,
            elem_drop=elem_drop,
            bias=False,
            side=self.side,  # Dynamically calculated: image_size/patch_size (e.g., 392/14=28)
            row_p=row_p,
            seg_len=seg_len,
            seg_drop_p=seg_drop_p,
            apply_segment_dropout=enable_segment_dropout,
        )
        bottleneck.append(bottle_neck_mlp)
        bottleneck = nn.ModuleList(bottleneck)

        # Create decoder (same as parent)
        decoder = []
        for _ in range(decoder_depth):
            mlp_ratio_val = TRANSFORMER_CONFIG["mlp_ratio"]
            assert isinstance(mlp_ratio_val, float)

            qkv_bias_val = TRANSFORMER_CONFIG["qkv_bias"]
            assert isinstance(qkv_bias_val, bool)

            layer_norm_eps_val = TRANSFORMER_CONFIG["layer_norm_eps"]
            assert isinstance(layer_norm_eps_val, float)

            attn_drop_val = TRANSFORMER_CONFIG["attn_drop"]
            assert isinstance(attn_drop_val, float)

            decoder_block = DecoderViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio_val,
                qkv_bias=qkv_bias_val,
                norm_layer=partial(nn.LayerNorm, eps=layer_norm_eps_val),  # type: ignore[arg-type]
                attn_drop=attn_drop_val,
                attn=LinearAttention,
            )
            decoder.append(decoder_block)
        decoder = nn.ModuleList(decoder)

        # Store components
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.embed_dim = embed_dim
        self.remove_class_token = remove_class_token
        self.encoder_name = encoder_name

        if not hasattr(self.encoder, "num_register_tokens"):
            self.encoder.num_register_tokens = 0

        # Initialize Gaussian blur for anomaly map smoothing (same as parent)
        from anomalib.models.components import GaussianBlur2d
        self.gaussian_blur = GaussianBlur2d(
            sigma=4,  # DEFAULT_GAUSSIAN_SIGMA
            channels=1,
            kernel_size=5,  # DEFAULT_GAUSSIAN_KERNEL_SIZE
        )

        # Loss function (same as parent)
        from anomalib.models.image.dinomaly.components import CosineHardMiningLoss
        self.loss_fn = CosineHardMiningLoss()

    @staticmethod
    def _get_architecture_config(
        encoder_name: str,
        target_layers: list[int],
    ) -> dict:
        """Get architecture configuration based on encoder name."""
        if "small" in encoder_name.lower():
            config = DINOV2_ARCHITECTURES["small"].copy()
        elif "large" in encoder_name.lower():
            config = DINOV2_ARCHITECTURES["large"].copy()
        else:  # Default to base
            config = DINOV2_ARCHITECTURES["base"].copy()

        # Override target_layers if provided
        config["target_layers"] = target_layers
        return config


class DinomalyHorizontal(Dinomaly):
    """Lightning wrapper for DinomalyHorizontal model.

    This is a drop-in replacement for Dinomaly that uses Horizontal Segment Dropout
    in the bottleneck MLP. Only the bottleneck is modified; the rest of the
    architecture is identical to baseline Dinomaly.

    Ablation configurations:
        A. Baseline: elem_p=0.2, enable_segment=False
        B. Element↓: elem_p=0.1, enable_segment=False
        C. Segment Only: elem_p=0.0, enable_segment=True
        D. Hybrid: elem_p=0.1, enable_segment=True (recommended)

    Args:
        encoder_name: Name of the encoder model
        elem_drop: Element-wise dropout probability (default: 0.1)
        row_p: Probability of applying segment dropout per row (default: 0.2)
        seg_len: Length of consecutive tokens to drop (default: 2)
        seg_drop_p: Probability of dropping each token in segment (default: 0.6)
        enable_segment_dropout: Whether to enable segment dropout (default: True)
        decoder_depth: Number of decoder layers (default: 8)
        **kwargs: Additional arguments passed to parent Dinomaly
    """

    def __init__(
        self,
        encoder_name: str = "dinov2reg_vit_base_14",
        elem_drop: float = 0.1,
        row_p: float = 0.2,
        seg_len: int = 2,
        seg_drop_p: float = 0.6,
        enable_segment_dropout: bool = True,
        decoder_depth: int = 8,
        target_layers: list[int] | None = None,
        fuse_layer_encoder: list[list[int]] | None = None,
        fuse_layer_decoder: list[list[int]] | None = None,
        remove_class_token: bool = True,  # Must be True for HorizontalSegmentDropout
        image_size: int = 392,  # Input image size for calculating token grid side
        **kwargs,
    ) -> None:
        # Store parameters for _setup_horizontal_model
        self.encoder_name = encoder_name
        self.elem_drop = elem_drop
        self.row_p = row_p
        self.seg_len = seg_len
        self.seg_drop_p = seg_drop_p
        self.enable_segment_dropout = enable_segment_dropout
        self.decoder_depth = decoder_depth
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.image_size = image_size

        # Call parent init with bottleneck_dropout set to elem_drop
        # (This creates a standard DinomalyModel which we'll replace)
        super().__init__(
            encoder_name=encoder_name,
            bottleneck_dropout=elem_drop,
            decoder_depth=decoder_depth,
            target_layers=target_layers,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            remove_class_token=remove_class_token,
            **kwargs,
        )

        # Replace with HorizontalModel
        self._setup_horizontal_model()

    def _setup_horizontal_model(self) -> None:
        """Replace the default DinomalyModel with DinomalyHorizontalModel."""
        self.model = DinomalyHorizontalModel(
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
        """Training step with horizontal dropout metrics logging.

        Logs segment dropout parameters every 100 steps.
        """
        result = super().training_step(batch, *args, **kwargs)

        # Log horizontal dropout config (every 100 steps)
        if self.global_step % 100 == 0:
            self.log("horizontal/elem_drop", self.elem_drop, on_step=True, on_epoch=False)
            self.log("horizontal/row_p", self.row_p, on_step=True, on_epoch=False)
            self.log("horizontal/seg_len", float(self.seg_len), on_step=True, on_epoch=False)
            self.log("horizontal/seg_drop_p", self.seg_drop_p, on_step=True, on_epoch=False)
            self.log("horizontal/segment_enabled", float(self.enable_segment_dropout), on_step=True, on_epoch=False)

        return result

    @staticmethod
    def get_config_name(elem_drop: float, enable_segment: bool) -> str:
        """Get configuration name for ablation study.

        Args:
            elem_drop: Element dropout probability
            enable_segment: Whether segment dropout is enabled

        Returns:
            Configuration name (A, B, C, or D)
        """
        if elem_drop >= 0.15 and not enable_segment:
            return "A_baseline"
        elif elem_drop < 0.15 and not enable_segment:
            return "B_elem_only"
        elif elem_drop < 0.05 and enable_segment:
            return "C_segment_only"
        else:
            return "D_hybrid"
