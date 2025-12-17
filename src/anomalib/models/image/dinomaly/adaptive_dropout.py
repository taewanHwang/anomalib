"""
Adaptive Bottleneck Dropout for Dinomaly using Angular Power Entropy (APE).

This module implements APE-based adaptive dropout to prevent overfitting
to highly regular normal patterns during long training.

Key idea (based on EDA findings):
- APE measures directional concentration of power in the frequency domain
- Low APE (strong directional pattern) → higher overfit risk → higher dropout
- High APE (isotropic/complex pattern) → lower overfit risk → lower dropout

APE showed strong separation (Cohen's d: 1.73~4.34) across all HDMAP domains
with consistent direction (Normal APE < Anomaly APE).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


def compute_angular_power_entropy_batch(
    images: torch.Tensor,
    num_angle_bins: int = 36,
    r_min_ratio: float = 0.05,
    r_max_ratio: float = 0.95,
    apply_hann_window: bool = True,
) -> torch.Tensor:
    """
    Compute Angular Power Entropy (APE) for a batch of images (GPU-friendly).

    APE measures how concentrated the power spectrum is along specific angles:
    - Low APE: Energy concentrated in specific directions (regular pattern)
    - High APE: Energy spread across all directions (isotropic/complex)

    Parameters
    ----------
    images : torch.Tensor
        Batch of images (B, C, H, W) normalized to [0, 1] or ImageNet normalized.
    num_angle_bins : int
        Number of angular bins (36 = 10° resolution).
    r_min_ratio : float
        Minimum radius ratio to exclude DC component (0.05 = 5% of max radius).
    r_max_ratio : float
        Maximum radius ratio to exclude corner high frequencies.
    apply_hann_window : bool
        Whether to apply Hann window to reduce spectral leakage.

    Returns
    -------
    torch.Tensor
        APE values of shape (B,) in [0, 1].
    """
    B, C, H, W = images.shape
    device = images.device
    dtype = images.dtype
    eps = 1e-12

    # Convert to grayscale if RGB
    if C == 3:
        # Denormalize from ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
        images_denorm = images * std + mean

        # ITU-R BT.601 luma weights
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device, dtype=dtype).view(1, 3, 1, 1)
        gray = (images_denorm * weights).sum(dim=1)  # (B, H, W)
    else:
        gray = images.squeeze(1)  # (B, H, W)

    # Remove DC (mean subtraction per sample)
    gray = gray - gray.mean(dim=(1, 2), keepdim=True)

    # Apply Hann window to reduce spectral leakage
    if apply_hann_window:
        hann_y = torch.hann_window(H, device=device, dtype=dtype).view(1, H, 1)
        hann_x = torch.hann_window(W, device=device, dtype=dtype).view(1, 1, W)
        window = hann_y * hann_x  # (1, H, W)
        gray = gray * window

    # 2D FFT → Power spectrum
    F_complex = torch.fft.fft2(gray)  # (B, H, W)
    P = torch.abs(F_complex) ** 2  # Power spectrum
    P = torch.fft.fftshift(P, dim=(1, 2))  # Shift DC to center

    # Create coordinate system (centered)
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    yy = torch.arange(H, device=device, dtype=dtype).view(1, H, 1) - cy
    xx = torch.arange(W, device=device, dtype=dtype).view(1, 1, W) - cx

    # Polar coordinates
    r = torch.sqrt(xx ** 2 + yy ** 2)  # (1, H, W)
    theta = torch.atan2(yy, xx)  # [-pi, pi], (1, H, W)
    theta = (theta + 2 * np.pi) % (2 * np.pi)  # [0, 2*pi)

    # Radial mask (exclude DC and corners)
    r_max = r.max()
    r_min = r_min_ratio * r_max
    r_upper = r_max_ratio * r_max
    radial_mask = (r >= r_min) & (r <= r_upper)  # (1, H, W)

    # Angular bin indices
    angle_bins = torch.linspace(0, 2 * np.pi, num_angle_bins + 1, device=device, dtype=dtype)
    bin_width = angle_bins[1] - angle_bins[0]
    angle_idx = ((theta - angle_bins[0]) / bin_width).long().clamp(0, num_angle_bins - 1)  # (1, H, W)

    # Compute APE per image
    ape_values = torch.zeros(B, device=device, dtype=dtype)

    # Flatten spatial dimensions for scatter_add
    P_flat = P.view(B, -1)  # (B, H*W)
    mask_flat = radial_mask.view(-1)  # (H*W,)
    angle_idx_flat = angle_idx.view(-1)  # (H*W,)

    for i in range(B):
        # Apply radial mask
        P_i = P_flat[i]  # (H*W,)
        P_masked = P_i[mask_flat]  # (num_valid,)
        idx_masked = angle_idx_flat[mask_flat]  # (num_valid,)

        # Aggregate power per angle bin using scatter_add
        angular_energy = torch.zeros(num_angle_bins, device=device, dtype=dtype)
        angular_energy.scatter_add_(0, idx_masked, P_masked)

        # Compute normalized Shannon entropy
        total_energy = angular_energy.sum()
        if total_energy <= eps:
            ape_values[i] = 0.0
            continue

        p = angular_energy / (total_energy + eps)
        H_theta = -torch.sum(p * torch.log(p + eps))
        H_max = np.log(num_angle_bins)
        ape = H_theta / H_max

        ape_values[i] = ape.clamp(0.0, 1.0)

    return ape_values


def ape_to_dropout_prob(
    ape: torch.Tensor,
    base_dropout: float = 0.3,
    min_dropout: float = 0.1,
    max_dropout: float = 0.6,
    sensitivity: float = 4.0,
    normal_ape: float = 0.78,
) -> torch.Tensor:
    """
    Map Angular Power Entropy (APE) to dropout probability.

    Low APE (more directional than normal) → high dropout (prevent overfitting)
    High APE (more isotropic than normal) → low dropout

    The formula is centered on normal_ape (average APE of normal samples):
    - When sensitivity=0: always returns base_dropout
    - When ape=normal_ape: returns base_dropout
    - When ape<normal_ape: dropout increases toward max_dropout
    - When ape>normal_ape: dropout decreases toward min_dropout

    Parameters
    ----------
    ape : torch.Tensor
        Angular Power Entropy values in [0, 1].
    base_dropout : float
        Base dropout probability (returned when sensitivity=0 or ape=normal_ape).
    min_dropout : float
        Minimum dropout probability (for high APE / isotropic samples).
    max_dropout : float
        Maximum dropout probability (for low APE / directional samples).
    sensitivity : float
        Sensitivity of mapping. When 0, dropout=base_dropout always.
    normal_ape : float
        Reference APE value from normal training samples.
        Domain-specific defaults from EDA:
        - domain_A: 0.777
        - domain_B: 0.713
        - domain_C: 0.866
        - domain_D: 0.816

    Returns
    -------
    torch.Tensor
        Dropout probabilities.
    """
    # Deviation from normal APE:
    # - positive (APE < normal): more directional → increase dropout
    # - negative (APE > normal): more isotropic → decrease dropout
    deviation = normal_ape - ape

    # Tanh mapping centered at base_dropout
    delta = deviation * sensitivity
    adjustment = torch.tanh(delta)  # Range: [-1, 1]

    # Asymmetric mapping centered at base_dropout:
    # - adjustment > 0 (more directional): dropout increases toward max_dropout
    # - adjustment < 0 (more isotropic): dropout decreases toward min_dropout
    # - adjustment = 0 (ape=normal_ape or sensitivity=0): dropout = base_dropout
    dropout = torch.where(
        adjustment >= 0,
        base_dropout + adjustment * (max_dropout - base_dropout),
        base_dropout + adjustment * (base_dropout - min_dropout),
    )

    return dropout.clamp(min_dropout, max_dropout)


class AdaptiveDropout(nn.Module):
    """
    Adaptive dropout module that adjusts dropout probability per sample
    based on input image Angular Power Entropy (APE).
    """

    def __init__(
        self,
        base_dropout: float = 0.3,
        min_dropout: float = 0.1,
        max_dropout: float = 0.6,
        sensitivity: float = 4.0,
        normal_ape: float = 0.78,
        use_adaptive: bool = True,
    ):
        """
        Initialize AdaptiveDropout.

        Parameters
        ----------
        base_dropout : float
            Base dropout probability (at normal_ape).
        min_dropout : float
            Minimum dropout probability.
        max_dropout : float
            Maximum dropout probability.
        sensitivity : float
            Sensitivity of APE-to-dropout mapping.
        normal_ape : float
            Reference APE from normal samples.
        use_adaptive : bool
            If False, uses fixed base_dropout (for comparison).
        """
        super().__init__()
        self.base_dropout = base_dropout
        self.min_dropout = min_dropout
        self.max_dropout = max_dropout
        self.sensitivity = sensitivity
        self.normal_ape = normal_ape
        self.use_adaptive = use_adaptive

        # Cache for current batch's dropout probs
        self._current_dropout_probs: Optional[torch.Tensor] = None

    def set_dropout_probs(self, dropout_probs: torch.Tensor) -> None:
        """Set per-sample dropout probabilities for current batch."""
        self._current_dropout_probs = dropout_probs

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply adaptive dropout.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, N, D) where B is batch size,
            N is sequence length, D is feature dimension.

        Returns
        -------
        Tensor
            Output tensor with dropout applied.
        """
        if not self.training:
            return x

        if not self.use_adaptive or self._current_dropout_probs is None:
            # Fallback to standard dropout
            return F.dropout(x, p=self.base_dropout, training=True)

        B = x.shape[0]
        dropout_probs = self._current_dropout_probs

        # Ensure dropout_probs matches batch size
        if len(dropout_probs) != B:
            return F.dropout(x, p=self.base_dropout, training=True)

        # Apply per-sample dropout
        output = torch.zeros_like(x)
        for i in range(B):
            p = float(dropout_probs[i])
            if p > 0:
                mask = torch.bernoulli(torch.full_like(x[i], 1 - p))
                output[i] = x[i] * mask / (1 - p + 1e-8)
            else:
                output[i] = x[i]

        return output


class AdaptiveMLP(nn.Module):
    """
    MLP with adaptive dropout based on input image APE.
    Replacement for DinomalyMLP in bottleneck.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type = nn.GELU,
        base_dropout: float = 0.3,
        min_dropout: float = 0.1,
        max_dropout: float = 0.6,
        sensitivity: float = 4.0,
        normal_ape: float = 0.78,
        bias: bool = False,
        apply_input_dropout: bool = True,
        use_adaptive: bool = True,
    ):
        """Initialize AdaptiveMLP."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

        self.adaptive_drop = AdaptiveDropout(
            base_dropout=base_dropout,
            min_dropout=min_dropout,
            max_dropout=max_dropout,
            sensitivity=sensitivity,
            normal_ape=normal_ape,
            use_adaptive=use_adaptive,
        )
        self.apply_input_dropout = apply_input_dropout

    def set_dropout_from_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute and set adaptive dropout probabilities from input images.

        Parameters
        ----------
        images : torch.Tensor
            Input images (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Computed dropout probabilities for logging/monitoring.
        """
        ape = compute_angular_power_entropy_batch(images)
        dropout_probs = ape_to_dropout_prob(
            ape,
            base_dropout=self.adaptive_drop.base_dropout,
            min_dropout=self.adaptive_drop.min_dropout,
            max_dropout=self.adaptive_drop.max_dropout,
            sensitivity=self.adaptive_drop.sensitivity,
            normal_ape=self.adaptive_drop.normal_ape,
        )
        self.adaptive_drop.set_dropout_probs(dropout_probs)
        return dropout_probs

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with adaptive dropout."""
        if self.apply_input_dropout:
            x = self.adaptive_drop(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.adaptive_drop(x)
        x = self.fc2(x)
        return self.adaptive_drop(x)


# For debugging and analysis
class AdaptiveDropoutStats:
    """Track adaptive dropout statistics during training."""

    def __init__(self):
        self.ape_values = []
        self.dropout_probs = []

    def log(self, ape: torch.Tensor, dropout_probs: torch.Tensor):
        """Log batch statistics."""
        self.ape_values.extend(ape.detach().cpu().numpy().tolist())
        self.dropout_probs.extend(dropout_probs.detach().cpu().numpy().tolist())

    def get_summary(self) -> dict:
        """Get summary statistics including percentiles."""
        if not self.ape_values:
            return {}

        ape_arr = np.array(self.ape_values)
        dropout_arr = np.array(self.dropout_probs)

        return {
            # APE statistics
            "ape_mean": float(np.mean(ape_arr)),
            "ape_std": float(np.std(ape_arr)),
            "ape_min": float(np.min(ape_arr)),
            "ape_max": float(np.max(ape_arr)),
            "ape_p10": float(np.percentile(ape_arr, 10)),
            "ape_p50": float(np.percentile(ape_arr, 50)),
            "ape_p90": float(np.percentile(ape_arr, 90)),
            # Dropout statistics
            "dropout_mean": float(np.mean(dropout_arr)),
            "dropout_std": float(np.std(dropout_arr)),
            "dropout_min": float(np.min(dropout_arr)),
            "dropout_max": float(np.max(dropout_arr)),
            "dropout_p10": float(np.percentile(dropout_arr, 10)),
            "dropout_p50": float(np.percentile(dropout_arr, 50)),
            "dropout_p90": float(np.percentile(dropout_arr, 90)),
            # Sample count
            "n_samples": len(self.ape_values),
        }

    def reset(self):
        """Reset statistics."""
        self.ape_values = []
        self.dropout_probs = []
