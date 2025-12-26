# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""V Row-wise Segment Dropout for Decoder Attention.

This module implements row-wise segment dropout for the V (Value) tensor
in LinearAttention layers. Unlike bottleneck dropout which can be recovered
by decoder layers, K/V dropout removes reconstruction material directly,
making it impossible to recover via global attention.

Key insight from v3.1 failure analysis:
- Bottleneck dropout: "noisy tokens" → Decoder recovers with global context
- K/V dropout: "missing V tokens" → No reconstruction material available

Safety mechanisms (all ablatable):
1. V-only masking: Preserve K for attention stability
2. Head-wise dropout: Apply to subset of attention heads
3. Layer-wise control: Apply only to specified decoder layers
4. Warmup schedule: Gradual dropout increase
5. Row-internal segment: Drop segments within rows, not full rows

Reference: v3.1 post-mortem analysis in MULTICLASS_DINOMALY_EXPERIMENTS_v4.md
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class VRowSegmentDropout(nn.Module):
    """Row-wise segment dropout for V (Value) tensors in attention.

    Unlike HorizontalSegmentDropout for bottleneck, this operates on:
    - V tensor shape: (batch, num_heads, seq_len, head_dim)
    - Drops segments within rows across specified heads

    Key differences from bottleneck dropout:
    1. Applied inside attention (cannot be recovered by downstream layers)
    2. V-only preserves attention computation stability
    3. Head-wise control for gradual ablation

    Args:
        side: Token grid side (e.g., 28 for 392/14 image)
        row_p: Probability of selecting a row for dropout
        band_width: Width of segment to drop within row
        seg_drop_p: Probability of dropping tokens in selected segment
        head_ratio: Ratio of heads to apply dropout to
        training_only: Only apply during training
    """

    def __init__(
        self,
        side: int = 28,
        row_p: float = 0.1,
        band_width: int = 2,
        seg_drop_p: float = 0.8,
        head_ratio: float = 0.5,
        training_only: bool = True,
    ) -> None:
        super().__init__()
        self.side = side
        self.row_p = row_p
        self.band_width = band_width
        self.seg_drop_p = seg_drop_p
        self.head_ratio = head_ratio
        self.training_only = training_only

    def forward(
        self,
        v: Tensor,
        current_step: int = 0,
        warmup_steps: int = 0,
        effective_drop_p: float | None = None,
    ) -> Tensor:
        """Apply row-wise segment dropout to V tensor.

        Args:
            v: Value tensor of shape (batch, num_heads, seq_len, head_dim)
            current_step: Current training step for warmup calculation
            warmup_steps: Number of warmup steps (0 = no warmup)
            effective_drop_p: Override dropout probability (for external control)

        Returns:
            V tensor with row-wise segment dropout applied
        """
        if self.training_only and not self.training:
            return v

        batch_size, num_heads, seq_len, head_dim = v.shape
        expected_seq_len = self.side * self.side

        # Handle seq_len mismatch (e.g., with class token)
        if seq_len != expected_seq_len:
            # Assume first tokens are special (class token, registers)
            # Apply dropout only to spatial tokens
            num_special = seq_len - expected_seq_len
            if num_special > 0:
                special_tokens = v[:, :, :num_special, :]
                spatial_tokens = v[:, :, num_special:, :]
                spatial_tokens = self._apply_dropout(
                    spatial_tokens, current_step, warmup_steps, effective_drop_p
                )
                return torch.cat([special_tokens, spatial_tokens], dim=2)
            else:
                # seq_len < expected, shouldn't happen in normal usage
                return v

        return self._apply_dropout(v, current_step, warmup_steps, effective_drop_p)

    def _apply_dropout(
        self,
        v: Tensor,
        current_step: int,
        warmup_steps: int,
        effective_drop_p: float | None,
    ) -> Tensor:
        """Apply row-wise segment dropout to spatial V tokens."""
        batch_size, num_heads, seq_len, head_dim = v.shape
        device = v.device
        dtype = v.dtype

        # Calculate effective dropout probability with warmup
        if effective_drop_p is not None:
            drop_p = effective_drop_p
        else:
            drop_p = self.seg_drop_p
            if warmup_steps > 0 and current_step < warmup_steps:
                warmup_factor = current_step / warmup_steps
                drop_p = drop_p * warmup_factor

        # Determine which heads to apply dropout
        num_dropout_heads = max(1, int(num_heads * self.head_ratio))
        dropout_head_indices = torch.randperm(num_heads, device=device)[:num_dropout_heads]

        # Create base mask (all ones = keep all)
        mask = torch.ones(batch_size, num_heads, seq_len, 1, device=device, dtype=dtype)

        # Generate row-wise segment dropout mask
        for head_idx in dropout_head_indices:
            head_mask = self._generate_row_segment_mask(
                batch_size, seq_len, drop_p, device, dtype
            )
            mask[:, head_idx, :, 0] = head_mask

        return v * mask

    def _generate_row_segment_mask(
        self,
        batch_size: int,
        seq_len: int,
        drop_p: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Generate row-wise segment dropout mask.

        Returns:
            Mask tensor of shape (batch_size, seq_len)
        """
        # Initialize mask (all ones = keep all)
        mask = torch.ones(batch_size, self.side, self.side, device=device, dtype=dtype)

        # Select rows to apply segment dropout
        row_mask = torch.rand(batch_size, self.side, device=device) < self.row_p

        for b in range(batch_size):
            for row in range(self.side):
                if row_mask[b, row]:
                    # Select random start position for segment
                    max_start = max(0, self.side - self.band_width)
                    start = torch.randint(0, max_start + 1, (1,), device=device).item()
                    end = min(start + self.band_width, self.side)

                    # Apply segment dropout with probability
                    segment_mask = torch.rand(end - start, device=device) < drop_p
                    for col_offset, should_drop in enumerate(segment_mask):
                        if should_drop:
                            mask[b, row, start + col_offset] = 0.0

        return mask.view(batch_size, seq_len)


class LinearAttentionWithVDropout(nn.Module):
    """Linear Attention with V-only Row-wise Segment Dropout.

    This is a modified version of LinearAttention that applies row-wise
    segment dropout to the V (Value) tensor during forward pass.

    Key insight: Drop V tokens → remove reconstruction material
    Unlike bottleneck dropout, this cannot be recovered by global attention.

    Safety mechanisms (all ablatable):
    1. V-only masking (preserve K for attention stability)
    2. Head-wise selective application
    3. Layer index check (apply only to specified layers)
    4. Warmup schedule (gradual increase)
    5. Row-internal segment dropout

    Args:
        input_dim: Input feature dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to add bias to qkv projection
        qk_scale: Scale factor for qk (default: head_dim**-0.5)
        attn_drop: Attention dropout rate (for output)
        proj_drop: Projection dropout rate
        v_drop_p: V dropout probability (within segment)
        row_p: Probability of selecting a row for dropout
        band_width: Width of segment to drop within row
        head_ratio: Ratio of heads to apply dropout to
        v_only: Only dropout V (preserve K)
        apply_dropout: Whether to apply V dropout (layer control)
        side: Token grid side for spatial dropout
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
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
        self.num_heads = num_heads
        head_dim = input_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(input_dim, input_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(input_dim, input_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # V Dropout module
        self.v_drop_p = v_drop_p
        self.v_only = v_only
        self.apply_dropout = apply_dropout
        self.side = side

        self.v_dropout = VRowSegmentDropout(
            side=side,
            row_p=row_p,
            band_width=band_width,
            seg_drop_p=v_drop_p,
            head_ratio=head_ratio,
            training_only=True,
        )

        # External control for warmup
        self._current_step = 0
        self._warmup_steps = 0

    def set_warmup_state(self, current_step: int, warmup_steps: int) -> None:
        """Set warmup state for dropout scheduling."""
        self._current_step = current_step
        self._warmup_steps = warmup_steps

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass through linear attention with V dropout.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            A tuple containing:
                - Output tensor of shape (batch_size, seq_len, embed_dim)
                - Key-value interaction tensor for potential downstream use
        """
        batch_size, seq_len, embed_dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_size, seq_len, 3, self.num_heads, embed_dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply V dropout if enabled
        if self.apply_dropout and self.training:
            v = self.v_dropout(
                v,
                current_step=self._current_step,
                warmup_steps=self._warmup_steps,
            )

            # Optionally apply to K as well (v_only=False)
            if not self.v_only:
                # Use same dropout module but for K
                k = self.v_dropout(
                    k,
                    current_step=self._current_step,
                    warmup_steps=self._warmup_steps,
                )

        # Apply ELU activation for linear attention
        q = torch.nn.functional.elu(q) + 1.0
        k = torch.nn.functional.elu(k) + 1.0

        # Compute attention (linear complexity)
        kv = torch.matmul(k.transpose(-2, -1), v)
        k_sum = k.sum(dim=-2, keepdim=True)
        z = 1.0 / torch.sum(q * k_sum, dim=-1, keepdim=True)
        x = torch.matmul(q, kv) * z

        x = x.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, kv
