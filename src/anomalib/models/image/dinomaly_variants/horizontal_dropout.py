# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Horizontal Segment Dropout for Dinomaly.

This module implements a spatially-aware dropout mechanism that drops
consecutive tokens within the same row. This is designed to suppress
the decoder's ability to use horizontal neighbor information for reconstruction,
which is particularly useful for detecting horizontal defect patterns (e.g., Domain C).

Key idea: Instead of dropping tokens independently (standard dropout),
we drop consecutive segments within rows to prevent the decoder from
"filling in" horizontal patterns using neighboring context.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class HorizontalSegmentDropout(nn.Module):
    """Horizontal Segment Dropout for spatial-aware regularization.

    This dropout mechanism:
    1. Selects rows with probability `row_p`
    2. For selected rows, drops a consecutive segment of `seg_len` tokens
    3. Within the segment, each token is dropped with probability `seg_drop_p`

    Unlike standard dropout:
    - Tokens are treated as a 2D grid (side x side)
    - Consecutive horizontal tokens are dropped together
    - This suppresses horizontal neighbor reconstruction

    Note: Inverted dropout scaling is intentionally omitted because:
    - LayerNorm in Transformer will normalize the output
    - Row-wise dropout can cause variance issues with scaling

    Args:
        side: Side length of the token grid (default 14 for 14x14=196 tokens)
        row_p: Probability of applying segment dropout to each row
        seg_len: Length of consecutive tokens to drop
        seg_drop_p: Probability of dropping each token within the segment
        training_only: Whether to apply dropout only during training

    Example:
        >>> dropout = HorizontalSegmentDropout(side=14, row_p=0.2, seg_len=2, seg_drop_p=0.6)
        >>> x = torch.randn(4, 196, 768)  # (B, N, D)
        >>> out = dropout(x)  # Same shape, with horizontal segments dropped
    """

    def __init__(
        self,
        side: int = 14,
        row_p: float = 0.2,
        seg_len: int = 2,
        seg_drop_p: float = 0.6,
        training_only: bool = True,
    ) -> None:
        super().__init__()
        self.side = side
        self.row_p = row_p
        self.seg_len = seg_len
        self.seg_drop_p = seg_drop_p
        self.training_only = training_only

        # Validate parameters
        if seg_len > side:
            msg = f"seg_len ({seg_len}) cannot be greater than side ({side})"
            raise ValueError(msg)
        if not 0 <= row_p <= 1:
            msg = f"row_p must be in [0, 1], got {row_p}"
            raise ValueError(msg)
        if not 0 <= seg_drop_p <= 1:
            msg = f"seg_drop_p must be in [0, 1], got {seg_drop_p}"
            raise ValueError(msg)

    def forward(self, x: Tensor) -> Tensor:
        """Apply horizontal segment dropout.

        Args:
            x: Input tensor of shape (B, N, D) where N = side * side

        Returns:
            Tensor with horizontal segments dropped (same shape as input)
        """
        if not self.training and self.training_only:
            return x

        if self.row_p == 0 or self.seg_drop_p == 0:
            return x

        B, N, D = x.shape
        device = x.device
        dtype = x.dtype

        # Validate token count
        expected_N = self.side * self.side
        if N != expected_N:
            msg = f"Expected {expected_N} tokens (side={self.side}), got {N}"
            raise ValueError(msg)

        # Create mask using vectorized operations
        mask = self._create_segment_mask(B, device, dtype)

        # Apply mask (no scaling - LayerNorm will normalize)
        return x * mask

    def _create_segment_mask(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Create a vectorized segment dropout mask.

        This implementation avoids for-loops by:
        1. Sampling all row selection decisions at once
        2. Sampling all segment start positions at once
        3. Using arange + broadcasting to create segment masks

        Args:
            batch_size: Number of samples in the batch
            device: Device to create the mask on
            dtype: Data type for the mask

        Returns:
            Mask tensor of shape (B, N, 1) with 1s for kept tokens and 0s for dropped
        """
        side = self.side
        seg_len = self.seg_len

        # Initialize mask as all 1s: (B, side, side)
        mask = torch.ones(batch_size, side, side, device=device, dtype=dtype)

        # 1. Sample which rows will have segment dropout: (B, side)
        row_decisions = torch.rand(batch_size, side, device=device) < self.row_p

        # 2. Sample segment start positions for all rows: (B, side)
        #    Valid range: [0, side - seg_len]
        max_start = side - seg_len
        segment_starts = torch.randint(0, max_start + 1, (batch_size, side), device=device)

        # 3. Create column indices for each position: (side,)
        col_indices = torch.arange(side, device=device)

        # 4. Create segment membership masks using broadcasting
        #    For each (batch, row), check if each column is within the segment
        #    segment_starts: (B, side, 1), col_indices: (side,) -> (B, side, side)
        segment_starts_expanded = segment_starts.unsqueeze(-1)  # (B, side, 1)
        in_segment = (col_indices >= segment_starts_expanded) & (
            col_indices < segment_starts_expanded + seg_len
        )  # (B, side, side)

        # 5. Apply row decisions: only drop if row is selected AND column is in segment
        #    row_decisions: (B, side, 1)
        row_decisions_expanded = row_decisions.unsqueeze(-1)  # (B, side, 1)
        should_consider_drop = row_decisions_expanded & in_segment  # (B, side, side)

        # 6. For positions that should be considered, sample dropout
        #    This creates probabilistic dropout within the segment
        segment_keep_prob = 1.0 - self.seg_drop_p
        keep_decisions = torch.rand(batch_size, side, side, device=device) < segment_keep_prob

        # 7. Final mask: keep if not in drop region OR if randomly kept
        mask = (~should_consider_drop) | (should_consider_drop & keep_decisions)

        # Convert to float and reshape: (B, side, side) -> (B, N, 1)
        mask = mask.float().view(batch_size, side * side, 1)

        return mask

    def extra_repr(self) -> str:
        """Return string representation of module parameters."""
        return f"side={self.side}, row_p={self.row_p}, seg_len={self.seg_len}, seg_drop_p={self.seg_drop_p}"


class HorizontalSegmentMLP(nn.Module):
    """MLP with Horizontal Segment Dropout for Dinomaly bottleneck.

    This is a replacement for DinomalyMLP that uses HorizontalSegmentDropout
    in addition to (or instead of) standard element-wise dropout.

    The MLP applies:
    1. Horizontal Segment Dropout (optional, on input)
    2. Standard Element Dropout (with reduced rate)
    3. FC1 -> GELU -> Dropout -> FC2 -> Dropout

    Args:
        in_features: Number of input features
        hidden_features: Number of hidden features (default: in_features)
        out_features: Number of output features (default: in_features)
        act_layer: Activation layer class
        elem_drop: Element-wise dropout probability
        bias: Whether to use bias in linear layers
        side: Side length of token grid for segment dropout
        row_p: Row selection probability for segment dropout
        seg_len: Segment length for segment dropout
        seg_drop_p: Drop probability within segments
        apply_segment_dropout: Whether to apply segment dropout on input

    Example:
        >>> mlp = HorizontalSegmentMLP(
        ...     in_features=768,
        ...     hidden_features=768 * 4,
        ...     elem_drop=0.1,
        ...     row_p=0.2,
        ...     seg_len=2,
        ...     seg_drop_p=0.6,
        ... )
        >>> x = torch.randn(4, 196, 768)
        >>> out = mlp(x)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        elem_drop: float = 0.1,
        bias: bool = False,
        side: int = 14,
        row_p: float = 0.2,
        seg_len: int = 2,
        seg_drop_p: float = 0.6,
        apply_segment_dropout: bool = True,
    ) -> None:
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Linear layers
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

        # Dropouts
        self.elem_drop = nn.Dropout(elem_drop)
        self.apply_segment_dropout = apply_segment_dropout

        if apply_segment_dropout and row_p > 0:
            self.segment_dropout = HorizontalSegmentDropout(
                side=side,
                row_p=row_p,
                seg_len=seg_len,
                seg_drop_p=seg_drop_p,
            )
        else:
            self.segment_dropout = nn.Identity()

        # Store parameters for logging
        self.side = side
        self.row_p = row_p
        self.seg_len = seg_len
        self.seg_drop_p = seg_drop_p

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with horizontal segment dropout.

        Args:
            x: Input tensor of shape (B, N, D)

        Returns:
            Output tensor of shape (B, N, out_features)
        """
        # 1. Apply horizontal segment dropout on input
        x = self.segment_dropout(x)

        # 2. Apply element dropout on input (same as original DinomalyMLP)
        x = self.elem_drop(x)

        # 3. FC1 -> GELU -> Dropout
        x = self.fc1(x)
        x = self.act(x)
        x = self.elem_drop(x)

        # 4. FC2 -> Dropout
        x = self.fc2(x)
        return self.elem_drop(x)
