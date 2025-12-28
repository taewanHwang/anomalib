"""Per-image normalization transforms for HDMAP images.

This module provides per-image normalization to address the amplitude scale
difference between Cold Start and Warmed Up conditions in HDMAP data.

Background:
    - Cold Start images have ~1.4x lower mean intensity
    - Cold Start images have ~1.6x lower dynamic range
    - This causes Fault Cold and Normal Warm score distributions to overlap
    - Per-image normalization can remove this scale dependency
"""

import torch


class PerImageNormalize:
    """Per-image normalization with multiple methods.

    This transform normalizes each image independently to remove amplitude
    scale dependency that can confuse anomaly detection models.

    Methods:
        - minmax: (x - min) / (max - min) -> [0, 1] range
        - robust: (x - p5) / (p95 - p5) -> robust to outliers
        - robust_soft: (x - p1) / (p99 - p1) -> softer stretch, less noise amplification

    Args:
        method: Normalization method ("minmax", "robust", or "robust_soft")

    Example:
        >>> transform = PerImageNormalize(method="robust_soft")
        >>> normalized_img = transform(img)
    """

    def __init__(self, method: str = "minmax"):
        """Initialize the transform.

        Args:
            method: Normalization method. One of:
                - "minmax": Standard min-max normalization to [0, 1]
                - "robust": Percentile-based normalization (p5, p95)
                - "robust_soft": Softer percentile normalization (p1, p99)
        """
        if method not in ("minmax", "robust", "robust_soft"):
            raise ValueError(f"Unknown method: {method}. Choose 'minmax', 'robust', or 'robust_soft'.")
        self.method = method

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply per-image normalization.

        Args:
            img: Input tensor of shape (C, H, W) or (H, W)

        Returns:
            Normalized tensor with same shape
        """
        if self.method == "minmax":
            return self._minmax_normalize(img)
        elif self.method == "robust":
            return self._robust_normalize(img, p_low=0.05, p_high=0.95)
        elif self.method == "robust_soft":
            return self._robust_normalize(img, p_low=0.01, p_high=0.99)
        return img

    def _minmax_normalize(self, img: torch.Tensor) -> torch.Tensor:
        """Standard min-max normalization.

        Normalizes to [0, 1] range based on actual min/max values.
        """
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min > 1e-8:
            img = (img - img_min) / (img_max - img_min)
        return img

    def _robust_normalize(self, img: torch.Tensor, p_low: float = 0.05, p_high: float = 0.95) -> torch.Tensor:
        """Robust normalization using percentiles.

        Args:
            img: Input tensor
            p_low: Lower percentile (default 0.05 for p5)
            p_high: Upper percentile (default 0.95 for p95)

        Values outside [0, 1] are clipped.
        """
        flat = img.flatten()
        low_val = torch.quantile(flat.float(), p_low)
        high_val = torch.quantile(flat.float(), p_high)

        if high_val - low_val > 1e-8:
            img = (img - low_val) / (high_val - low_val)
            img = torch.clamp(img, 0.0, 1.0)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(method='{self.method}')"


def normalize_batch(batch: torch.Tensor, method: str = "minmax") -> torch.Tensor:
    """Normalize a batch of images independently.

    This function applies per-image normalization to each image in a batch.

    Args:
        batch: Input tensor of shape (B, C, H, W)
        method: Normalization method ("minmax" or "robust")

    Returns:
        Normalized batch tensor with same shape
    """
    transform = PerImageNormalize(method=method)
    normalized = []

    for img in batch:
        normalized.append(transform(img))

    return torch.stack(normalized)
