# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dinomaly with K/V Row-wise Dropout in Decoder.

This variant applies V (Value) row-wise segment dropout directly in the
decoder's LinearAttention layers, addressing the key limitation of v3.1:
bottleneck dropout being recovered by global attention in decoder layers.

Key insight from v3.1 failure analysis:
- Bottleneck dropout: "noisy tokens" → Decoder recovers with global context
- K/V dropout: "missing V tokens" → No reconstruction material available

This approach moves dropout from bottleneck (recoverable) to decoder K/V
(not recoverable), fundamentally changing the pseudo-anomaly generation.

Safety mechanisms (all ablatable):
1. V-only masking: Preserve K for attention stability
2. Head-wise dropout: Apply to subset of attention heads
3. Layer-wise scheduling: Apply only to deep decoder layers
4. Warmup schedule: Gradual dropout increase during training
5. Row-internal segment: Drop segments within rows, not full rows

Reference: MULTICLASS_DINOMALY_EXPERIMENTS_v4.md
"""

from __future__ import annotations

from functools import partial

import torch
from torch import Tensor, nn

from anomalib.models.image.dinomaly import Dinomaly
from anomalib.models.image.dinomaly.torch_model import (
    DEFAULT_FUSE_LAYERS,
    DINOV2_ARCHITECTURES,
    TRANSFORMER_CONFIG,
    DinomalyModel,
)
from anomalib.models.image.dinomaly.components import load as load_dinov2_model
from anomalib.models.image.dinomaly.components.layers import DinomalyMLP
from anomalib.models.image.dinomaly_variants.kv_dropout import LinearAttentionWithVDropout
from anomalib.models.image.dinomaly_variants.topk_loss import CosineTopKLoss


class DecoderViTBlockWithVDropout(nn.Module):
    """Decoder ViT Block with V dropout in attention.

    Modified from DecoderViTBlock to use LinearAttentionWithVDropout
    for applying row-wise segment dropout to V tensors.

    Args:
        dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        qkv_bias: Whether to add bias to qkv projection
        norm_layer: Normalization layer
        attn_drop: Attention dropout rate
        v_drop_p: V dropout probability
        row_p: Row selection probability
        band_width: Segment width
        head_ratio: Ratio of heads to apply dropout
        v_only: Only dropout V (not K)
        apply_dropout: Whether to apply V dropout
        side: Token grid side
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        attn_drop: float = 0.0,
        # V Dropout parameters
        v_drop_p: float = 0.05,
        row_p: float = 0.1,
        band_width: int = 2,
        head_ratio: float = 0.5,
        v_only: bool = True,
        apply_dropout: bool = True,
        side: int = 28,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LinearAttentionWithVDropout(
            input_dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            # V Dropout params
            v_drop_p=v_drop_p,
            row_p=row_p,
            band_width=band_width,
            head_ratio=head_ratio,
            v_only=v_only,
            apply_dropout=apply_dropout,
            side=side,
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DinomalyMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=0.0,
            bias=False,
            apply_input_dropout=False,
        )

    def forward(
        self,
        x: Tensor,
        return_attention: bool = False,
        attn_mask: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass with residual connections.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            return_attention: Whether to return attention weights
            attn_mask: Optional attention mask (ignored in LinearAttention)

        Returns:
            Output tensor, or tuple of (output, attention) if return_attention=True
        """
        # Note: LinearAttentionWithVDropout doesn't use attn_mask (linear attention)
        attn_out, attn = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))

        if return_attention:
            return x, attn
        return x

    def set_warmup_state(self, current_step: int, warmup_steps: int) -> None:
        """Set warmup state for V dropout scheduling."""
        self.attn.set_warmup_state(current_step, warmup_steps)


class DinomalyKVDropoutModel(DinomalyModel):
    """Dinomaly model with K/V Row-wise Dropout in decoder.

    Replaces standard LinearAttention with LinearAttentionWithVDropout
    in specified decoder layers.

    Args:
        encoder_name: DINOv2 encoder name
        v_drop_p: V dropout probability
        row_p: Row selection probability
        band_width: Segment width
        head_ratio: Ratio of heads to apply dropout
        v_only: Only dropout V (not K)
        apply_layers: List of decoder layer indices to apply V dropout
        warmup_steps: Warmup steps for dropout scheduling
        q_percent: TopK loss q percentage
        q_schedule: Whether to use TopK warmup schedule
        decoder_depth: Number of decoder layers
        target_layers: Encoder layers to extract features from
        fuse_layer_encoder: Encoder layer groupings for fusion
        fuse_layer_decoder: Decoder layer groupings for fusion
        remove_class_token: Whether to remove class token
        image_size: Input image size
    """

    def __init__(
        self,
        encoder_name: str = "dinov2reg_vit_base_14",
        # V Dropout parameters
        v_drop_p: float = 0.05,
        row_p: float = 0.1,
        band_width: int = 2,
        head_ratio: float = 0.5,
        v_only: bool = True,
        apply_layers: list[int] | None = None,
        warmup_steps: int = 200,
        # TopK Loss parameters
        q_percent: float = 2.0,
        q_schedule: bool = True,
        # Model architecture
        decoder_depth: int = 8,
        target_layers: list[int] | None = None,
        fuse_layer_encoder: list[list[int]] | None = None,
        fuse_layer_decoder: list[list[int]] | None = None,
        remove_class_token: bool = True,
        image_size: int = 392,
    ) -> None:
        # Initialize nn.Module directly
        nn.Module.__init__(self)

        # Default apply_layers: last 4 layers (indices 4, 5, 6, 7)
        if apply_layers is None:
            apply_layers = [4, 5, 6, 7]

        # Calculate side from image_size
        patch_size = 14
        self.side = image_size // patch_size

        # Store parameters
        self.v_drop_p = v_drop_p
        self.row_p = row_p
        self.band_width = band_width
        self.head_ratio = head_ratio
        self.v_only = v_only
        self.apply_layers = apply_layers
        self.warmup_steps = warmup_steps
        self.q_percent = q_percent
        self.q_schedule = q_schedule
        self._current_step = 0

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

        # Create standard bottleneck (no horizontal dropout)
        bottleneck = []
        bottle_neck_mlp = DinomalyMLP(
            in_features=embed_dim,
            hidden_features=embed_dim * 4,
            out_features=embed_dim,
            act_layer=nn.GELU,
            drop=0.2,  # Standard dropout
            bias=False,
            apply_input_dropout=True,
        )
        bottleneck.append(bottle_neck_mlp)
        bottleneck = nn.ModuleList(bottleneck)

        # Create decoder with V dropout in specified layers
        decoder = []
        for layer_idx in range(decoder_depth):
            mlp_ratio_val = TRANSFORMER_CONFIG["mlp_ratio"]
            assert isinstance(mlp_ratio_val, float)

            qkv_bias_val = TRANSFORMER_CONFIG["qkv_bias"]
            assert isinstance(qkv_bias_val, bool)

            layer_norm_eps_val = TRANSFORMER_CONFIG["layer_norm_eps"]
            assert isinstance(layer_norm_eps_val, float)

            attn_drop_val = TRANSFORMER_CONFIG["attn_drop"]
            assert isinstance(attn_drop_val, float)

            # Apply V dropout only to specified layers
            apply_dropout = layer_idx in apply_layers

            decoder_block = DecoderViTBlockWithVDropout(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio_val,
                qkv_bias=qkv_bias_val,
                norm_layer=partial(nn.LayerNorm, eps=layer_norm_eps_val),
                attn_drop=attn_drop_val,
                # V Dropout params
                v_drop_p=v_drop_p,
                row_p=row_p,
                band_width=band_width,
                head_ratio=head_ratio,
                v_only=v_only,
                apply_dropout=apply_dropout,
                side=self.side,
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

        # Initialize Gaussian blur for anomaly map smoothing
        from anomalib.models.components import GaussianBlur2d
        self.gaussian_blur = GaussianBlur2d(
            sigma=4,
            channels=1,
            kernel_size=5,
        )

        # TopK Loss function
        self.loss_fn = CosineTopKLoss(
            q_percent=q_percent,
            q_schedule=q_schedule,
            warmup_steps=warmup_steps,
        )

    def set_training_step(self, step: int) -> None:
        """Set current training step for warmup scheduling."""
        self._current_step = step
        # Update all decoder blocks with V dropout
        for layer_idx, block in enumerate(self.decoder):
            if hasattr(block, "set_warmup_state"):
                block.set_warmup_state(step, self.warmup_steps)

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
        else:
            config = DINOV2_ARCHITECTURES["base"].copy()
        config["target_layers"] = target_layers
        return config


class DinomalyKVDropout(Dinomaly):
    """Lightning wrapper for DinomalyKVDropout model.

    This model applies V row-wise segment dropout in decoder attention layers,
    combined with TopK loss for optimal anomaly detection performance.

    Key advantages over v3.1 (bottleneck dropout):
    1. Dropout applied at reconstruction site (V in attention)
    2. Cannot be recovered by subsequent layers
    3. Layer-wise control for fine-grained ablation

    Args:
        encoder_name: DINOv2 encoder name
        v_drop_p: V dropout probability
        row_p: Row selection probability
        band_width: Segment width
        head_ratio: Ratio of heads to apply dropout
        v_only: Only dropout V (not K)
        apply_layers: Decoder layers to apply V dropout
        warmup_steps: Warmup steps for dropout scheduling
        q_percent: TopK loss q percentage
        q_schedule: Whether to use TopK warmup schedule
        decoder_depth: Number of decoder layers
        target_layers: Encoder layers to extract
        fuse_layer_encoder: Encoder layer groupings
        fuse_layer_decoder: Decoder layer groupings
        remove_class_token: Whether to remove class token
        image_size: Input image size
        **kwargs: Additional arguments for parent Dinomaly
    """

    def __init__(
        self,
        encoder_name: str = "dinov2reg_vit_base_14",
        # V Dropout parameters
        v_drop_p: float = 0.05,
        row_p: float = 0.1,
        band_width: int = 2,
        head_ratio: float = 0.5,
        v_only: bool = True,
        apply_layers: list[int] | None = None,
        warmup_steps: int = 200,
        # TopK Loss parameters
        q_percent: float = 2.0,
        q_schedule: bool = True,
        # Model architecture
        decoder_depth: int = 8,
        target_layers: list[int] | None = None,
        fuse_layer_encoder: list[list[int]] | None = None,
        fuse_layer_decoder: list[list[int]] | None = None,
        remove_class_token: bool = True,
        image_size: int = 392,
        **kwargs,
    ) -> None:
        # Store parameters for model setup
        self._encoder_name = encoder_name
        self._v_drop_p = v_drop_p
        self._row_p = row_p
        self._band_width = band_width
        self._head_ratio = head_ratio
        self._v_only = v_only
        self._apply_layers = apply_layers if apply_layers is not None else [4, 5, 6, 7]
        self._warmup_steps = warmup_steps
        self._q_percent = q_percent
        self._q_schedule = q_schedule
        self._decoder_depth = decoder_depth
        self._target_layers = target_layers
        self._fuse_layer_encoder = fuse_layer_encoder
        self._fuse_layer_decoder = fuse_layer_decoder
        self._remove_class_token = remove_class_token
        self._image_size = image_size

        # Call parent init
        super().__init__(
            encoder_name=encoder_name,
            bottleneck_dropout=0.2,  # Standard dropout
            decoder_depth=decoder_depth,
            target_layers=target_layers,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            remove_class_token=remove_class_token,
            **kwargs,
        )

        # Replace with KVDropout model
        self._setup_kv_dropout_model()

    def _setup_kv_dropout_model(self) -> None:
        """Replace the default DinomalyModel with DinomalyKVDropoutModel."""
        self.model = DinomalyKVDropoutModel(
            encoder_name=self._encoder_name,
            v_drop_p=self._v_drop_p,
            row_p=self._row_p,
            band_width=self._band_width,
            head_ratio=self._head_ratio,
            v_only=self._v_only,
            apply_layers=self._apply_layers,
            warmup_steps=self._warmup_steps,
            q_percent=self._q_percent,
            q_schedule=self._q_schedule,
            decoder_depth=self._decoder_depth,
            target_layers=self._target_layers,
            fuse_layer_encoder=self._fuse_layer_encoder,
            fuse_layer_decoder=self._fuse_layer_decoder,
            remove_class_token=self._remove_class_token,
            image_size=self._image_size,
        )

        # Freeze encoder, unfreeze bottleneck and decoder
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.bottleneck.parameters():
            param.requires_grad = True
        for param in self.model.decoder.parameters():
            param.requires_grad = True

        # Update trainable modules reference
        self.trainable_modules = torch.nn.ModuleList([
            self.model.bottleneck,
            self.model.decoder,
        ])

    def training_step(self, batch, *args, **kwargs):
        """Training step with warmup scheduling and logging."""
        # Update warmup state
        self.model.set_training_step(self.global_step)

        result = super().training_step(batch, *args, **kwargs)

        # Log K/V dropout config (every 100 steps)
        if self.global_step % 100 == 0:
            self.log("kv_dropout/v_drop_p", self._v_drop_p, on_step=True, on_epoch=False)
            self.log("kv_dropout/row_p", self._row_p, on_step=True, on_epoch=False)
            self.log("kv_dropout/band_width", float(self._band_width), on_step=True, on_epoch=False)
            self.log("kv_dropout/head_ratio", self._head_ratio, on_step=True, on_epoch=False)
            self.log("kv_dropout/v_only", float(self._v_only), on_step=True, on_epoch=False)
            self.log("topk/q_percent", self._q_percent, on_step=True, on_epoch=False)

            # Calculate effective dropout based on warmup
            if self._warmup_steps > 0 and self.global_step < self._warmup_steps:
                effective_p = self._v_drop_p * (self.global_step / self._warmup_steps)
            else:
                effective_p = self._v_drop_p
            self.log("kv_dropout/effective_p", effective_p, on_step=True, on_epoch=False)

        return result

    @staticmethod
    def get_config_name(
        v_drop_p: float,
        head_ratio: float,
        apply_layers: list[int],
        v_only: bool = True,
    ) -> str:
        """Get configuration name for experiment tracking.

        Args:
            v_drop_p: V dropout probability
            head_ratio: Ratio of heads to apply dropout
            apply_layers: Layers where dropout is applied
            v_only: Whether only V is dropped

        Returns:
            Configuration name string
        """
        parts = ["KVDrop"]
        parts.append(f"p{v_drop_p}")
        parts.append(f"h{head_ratio}")
        parts.append(f"L{len(apply_layers)}")
        if not v_only:
            parts.append("KV")
        return "_".join(parts)
