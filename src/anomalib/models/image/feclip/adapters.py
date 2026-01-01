"""Frequency-aware adapters for FE-CLIP model.

This module implements the FFE (Frequency-aware Feature Extraction) and
LFS (Local Frequency Statistics) adapters for the FE-CLIP model.

Paper: FE-CLIP: Frequency Enhanced CLIP Model for Zero-Shot Anomaly Detection
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def dct_basis(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate DCT basis matrix.

    Args:
        n: Size of the DCT basis matrix.
        device: Device to create tensor on.
        dtype: Data type of the tensor.

    Returns:
        DCT basis matrix of shape (n, n).
    """
    k = torch.arange(n, device=device, dtype=dtype).view(-1, 1)
    i = torch.arange(n, device=device, dtype=dtype).view(1, -1)
    C = torch.cos(math.pi / n * (i + 0.5) * k)
    C[0] *= 1.0 / math.sqrt(n)
    C[1:] *= math.sqrt(2.0 / n)
    return C


def dct2(x: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Apply 2D DCT transformation.

    Args:
        x: Input tensor of shape (..., n, n).
        C: DCT basis matrix of shape (n, n).

    Returns:
        DCT transformed tensor of shape (..., n, n).
    """
    y = torch.einsum("ab,...bc->...ac", C, x)
    y = torch.einsum("...ab,cb->...ac", y, C)
    return y


def idct2(x: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Apply 2D inverse DCT transformation.

    Args:
        x: Input tensor of shape (..., n, n).
        C: DCT basis matrix of shape (n, n).

    Returns:
        Inverse DCT transformed tensor of shape (..., n, n).
    """
    y = torch.einsum("ba,...bc->...ac", C, x)
    y = torch.einsum("...ab,bc->...ac", y, C)
    return y


class FFEAdapter(nn.Module):
    """Frequency-aware Feature Extraction (FFE) adapter.

    This adapter extracts frequency information using non-overlapping windows
    with DCT transformation, processes through MLP, and transforms back to
    spatial domain using inverse DCT.

    Args:
        d_model: Dimension of the input features.
        P: Window size for non-overlapping DCT. Default: 3.
    """

    def __init__(self, d_model: int, P: int = 3) -> None:
        super().__init__()
        self.P = P
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Forward pass of FFE adapter.

        Args:
            f: Input tensor of shape (B, H, W, D).

        Returns:
            Frequency-aware features of shape (B, H, W, D).
        """
        B, H, W, D = f.shape
        P = self.P

        # Store original size for later cropping
        orig_H, orig_W = H, W

        # Pad token-grid so H, W are divisible by P
        Hp = (H + P - 1) // P * P
        Wp = (W + P - 1) // P * P
        if Hp != H or Wp != W:
            pad_h = Hp - H
            pad_w = Wp - W
            f = F.pad(f, (0, 0, 0, pad_w, 0, pad_h), mode="replicate")

        B, H2, W2, D = f.shape
        C = dct_basis(P, device=f.device, dtype=f.dtype)

        # Reshape to (B, H/P, W/P, P, P, D)
        x = f.view(B, H2 // P, P, W2 // P, P, D).permute(0, 1, 3, 2, 4, 5)
        # Reshape to (B, Hp, Wp, D, P, P)
        x = x.permute(0, 1, 2, 5, 3, 4)

        # Apply DCT, MLP, and inverse DCT
        x = dct2(x, C)  # (B, Hp, Wp, D, P, P)
        x = x.permute(0, 1, 2, 4, 5, 3)  # (B, Hp, Wp, P, P, D)
        x = self.mlp(x)
        x = x.permute(0, 1, 2, 5, 3, 4)  # (B, Hp, Wp, D, P, P)
        x = idct2(x, C)  # (B, Hp, Wp, D, P, P)
        x = x.permute(0, 1, 2, 4, 5, 3)  # (B, Hp, Wp, P, P, D)

        # Reshape back to (B, H2, W2, D)
        out = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H2, W2, D)

        # Crop back to original size if padded
        return out[:, :orig_H, :orig_W, :]


class LFSAdapter(nn.Module):
    """Local Frequency Statistics (LFS) adapter.

    This adapter extracts local frequency statistics using sliding window DCT,
    aggregates statistics, and processes through Conv+GELU.

    Args:
        d_model: Dimension of the input features.
        Q: Window size for sliding window DCT. Default: 3.
        conv_kernel: Kernel size for convolution layer. Default: 3.
        pad_mode: Padding mode for feature map. Default: "replicate".
        agg_mode: Aggregation mode for DCT coefficients. Default: "mean".
            - "mean": signed mean (original, may cancel out)
            - "abs": absolute value mean
            - "power": squared mean (frequency energy)
    """

    def __init__(
        self,
        d_model: int,
        Q: int = 3,
        conv_kernel: int = 3,
        pad_mode: str = "replicate",
        agg_mode: str = "mean",
    ) -> None:
        super().__init__()
        self.Q = Q
        self.pad_mode = pad_mode
        self.agg_mode = agg_mode
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=conv_kernel, padding=conv_kernel // 2),
            nn.GELU(),
        )

    def forward(
        self,
        f: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of LFS adapter.

        Args:
            f: Input tensor of shape (B, H, W, D).
            valid_mask: Optional mask tensor of shape (B, H, W) where True/1
                indicates valid image regions (not letterbox padding).
                Used for masked mean computation.

        Returns:
            Local frequency statistics features of shape (B, H, W, D).
        """
        B, H, W, D = f.shape
        Q = self.Q
        C = dct_basis(Q, device=f.device, dtype=f.dtype)

        # (B, H, W, D) -> (B, D, H, W)
        x = f.permute(0, 3, 1, 2).contiguous()
        pad = Q // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode=self.pad_mode)

        # Extract patches using unfold
        patches = F.unfold(x_pad, kernel_size=Q, stride=1)  # (B, D*Q*Q, H*W)
        patches = patches.view(B, D, Q, Q, H, W).permute(0, 4, 5, 1, 2, 3)  # (B, H, W, D, Q, Q)

        # Apply sliding window DCT
        fd = dct2(patches, C)  # (B, H, W, D, Q, Q)

        # Apply aggregation mode
        if self.agg_mode == "abs":
            fd = fd.abs()
        elif self.agg_mode == "power":
            fd = fd ** 2

        # Aggregate: mean over (Q, Q) as per paper
        if valid_mask is None:
            stats = fd.mean(dim=(-1, -2))  # (B, H, W, D)
        else:
            # Masked mean: only consider valid neighbors (for letterbox padding)
            vm = valid_mask.to(fd.dtype)
            vm_pad = F.pad(vm, (pad, pad, pad, pad), mode="constant", value=0)
            vm_nb = F.unfold(vm_pad.unsqueeze(1), kernel_size=Q, stride=1)  # (B, Q*Q, H*W)
            vm_nb = vm_nb.view(B, Q, Q, H, W).permute(0, 3, 4, 1, 2)  # (B, H, W, Q, Q)
            denom = vm_nb.sum(dim=(-1, -2)).clamp(min=1.0)  # (B, H, W)
            stats = (fd * vm_nb.unsqueeze(3)).sum(dim=(-1, -2)) / denom.unsqueeze(-1)  # (B, H, W, D)

        # Apply Conv + GELU
        y = stats.permute(0, 3, 1, 2).contiguous()
        y = self.conv(y).permute(0, 2, 3, 1).contiguous()
        return y
