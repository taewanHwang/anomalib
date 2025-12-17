"""
Orientation Entropy Analysis for Adaptive Bottleneck Dropout.

This module provides functions to compute orientation entropy from images,
which can be used to guide adaptive dropout probability in Dinomaly.

Key insight:
- Low entropy (close to 0): Strong dominant direction → highly regular structure
- High entropy (close to 1): Diverse directions → complex/irregular/potentially anomalous

For industrial signal images (HDMap, 2D FFT), normal samples often have
strong regular structures. Adaptive dropout based on orientation entropy
can prevent overfitting to these dominant patterns.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional
import torch


def orientation_entropy_cv2(
    img: np.ndarray,
    num_bins: int = 36,
    magnitude_threshold: float = 1e-3,
    eps: float = 1e-12,
) -> float:
    """
    Compute normalized orientation entropy in [0, 1] using OpenCV (fast version).

    Parameters
    ----------
    img : np.ndarray
        Input image (H,W) grayscale or (H,W,3) BGR/RGB.
    num_bins : int
        Number of orientation bins over [0, pi).
    magnitude_threshold : float
        Ignore pixels whose gradient magnitude is below this threshold.
    eps : float
        Small number to avoid log(0).

    Returns
    -------
    float
        Normalized entropy (0: single dominant direction, 1: uniform directions).
    """
    # Convert to grayscale if needed
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Ensure float32
    if gray.dtype != np.float32:
        gray = gray.astype(np.float32)
        if gray.max() > 1.5:
            gray = gray / 255.0

    # Sobel gradients (faster than manual convolution)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Magnitude and orientation
    mag = np.sqrt(gx * gx + gy * gy)

    # Apply magnitude threshold
    mask = mag > magnitude_threshold
    if not np.any(mask):
        return 0.0

    # Orientation in [0, pi) - fold negative angles
    theta = np.arctan2(gy, gx)  # [-pi, pi]
    theta = np.mod(theta, np.pi)  # [0, pi)

    theta_m = theta[mask]
    mag_m = mag[mask]

    # Weighted histogram
    hist, _ = np.histogram(theta_m, bins=num_bins, range=(0.0, np.pi), weights=mag_m)
    total = hist.sum()
    if total <= 0:
        return 0.0

    p = hist / total

    # Shannon entropy normalized by max entropy
    H = -np.sum(p * np.log(p + eps))
    H_norm = H / np.log(num_bins)

    return float(np.clip(H_norm, 0.0, 1.0))


def orientation_entropy_batch(
    images: torch.Tensor,
    num_bins: int = 36,
    magnitude_threshold: float = 1e-3,
) -> torch.Tensor:
    """
    Compute orientation entropy for a batch of images (GPU-accelerated).

    Parameters
    ----------
    images : torch.Tensor
        Batch of images (B, C, H, W) or (B, 1, H, W).
    num_bins : int
        Number of orientation bins.
    magnitude_threshold : float
        Minimum gradient magnitude to consider.

    Returns
    -------
    torch.Tensor
        Entropy values of shape (B,) in [0, 1].
    """
    device = images.device
    B, C, H, W = images.shape

    # Convert to grayscale if RGB
    if C == 3:
        # ITU-R BT.601 luma coefficients
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device).view(1, 3, 1, 1)
        gray = (images * weights).sum(dim=1, keepdim=True)
    else:
        gray = images

    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)

    # Compute gradients
    gx = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
    gy = torch.nn.functional.conv2d(gray, sobel_y, padding=1)

    # Magnitude and orientation
    mag = torch.sqrt(gx ** 2 + gy ** 2)
    theta = torch.atan2(gy, gx)  # [-pi, pi]
    theta = torch.remainder(theta, np.pi)  # [0, pi)

    # Compute entropy per image
    entropies = []
    eps = 1e-12

    for i in range(B):
        mag_i = mag[i, 0].flatten()
        theta_i = theta[i, 0].flatten()

        # Apply threshold
        mask = mag_i > magnitude_threshold
        if mask.sum() == 0:
            entropies.append(0.0)
            continue

        theta_masked = theta_i[mask]
        mag_masked = mag_i[mask]

        # Compute weighted histogram using bincount (approximate)
        bin_indices = (theta_masked / np.pi * num_bins).long().clamp(0, num_bins - 1)
        hist = torch.zeros(num_bins, device=device)
        hist.scatter_add_(0, bin_indices, mag_masked)

        # Normalize
        total = hist.sum()
        if total <= 0:
            entropies.append(0.0)
            continue

        p = hist / total
        H = -torch.sum(p * torch.log(p + eps))
        H_norm = H / np.log(num_bins)
        entropies.append(float(H_norm.clamp(0.0, 1.0)))

    return torch.tensor(entropies, device=device)


def dominant_frequency_ratio(
    img: np.ndarray,
    top_k: int = 5,
) -> float:
    """
    Compute the energy ratio of top-k dominant frequency components.

    High ratio → energy concentrated in few frequencies → regular pattern
    Low ratio → distributed energy → complex/irregular pattern

    Parameters
    ----------
    img : np.ndarray
        Input image (H,W) or (H,W,3).
    top_k : int
        Number of top frequency components to consider.

    Returns
    -------
    float
        Ratio of top-k energy to total energy in [0, 1].
    """
    # Convert to grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if gray.dtype != np.float32:
        gray = gray.astype(np.float32)

    # 2D FFT
    f = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f)
    magnitude = np.abs(f_shift)

    # Exclude DC component
    h, w = magnitude.shape
    magnitude[h//2, w//2] = 0

    # Total energy
    total_energy = np.sum(magnitude ** 2)
    if total_energy <= 0:
        return 0.0

    # Top-k energy
    flat_mag = magnitude.flatten()
    top_k_indices = np.argpartition(flat_mag, -top_k)[-top_k:]
    top_k_energy = np.sum(flat_mag[top_k_indices] ** 2)

    return float(top_k_energy / total_energy)


def compute_regularity_score(
    img: np.ndarray,
    orientation_weight: float = 0.5,
    frequency_weight: float = 0.5,
) -> Tuple[float, dict]:
    """
    Compute a combined regularity score from orientation entropy and frequency ratio.

    High regularity score → sample is highly regular → needs more dropout
    Low regularity score → sample is complex/irregular → needs less dropout

    Parameters
    ----------
    img : np.ndarray
        Input image.
    orientation_weight : float
        Weight for (1 - orientation_entropy).
    frequency_weight : float
        Weight for dominant_frequency_ratio.

    Returns
    -------
    Tuple[float, dict]
        Combined regularity score in [0, 1] and component scores.
    """
    orient_entropy = orientation_entropy_cv2(img)
    freq_ratio = dominant_frequency_ratio(img)

    # Invert orientation entropy (low entropy = high regularity)
    orient_regularity = 1.0 - orient_entropy

    # Combine
    regularity = orientation_weight * orient_regularity + frequency_weight * freq_ratio

    return regularity, {
        "orientation_entropy": orient_entropy,
        "orientation_regularity": orient_regularity,
        "frequency_ratio": freq_ratio,
        "combined_regularity": regularity,
    }


def adaptive_dropout_prob(
    regularity_score: float,
    base_dropout: float = 0.3,
    min_dropout: float = 0.1,
    max_dropout: float = 0.7,
    sensitivity: float = 2.0,
) -> float:
    """
    Map regularity score to adaptive dropout probability.

    High regularity → higher dropout to prevent overfitting
    Low regularity → lower dropout to maintain training stability

    Parameters
    ----------
    regularity_score : float
        Combined regularity score in [0, 1].
    base_dropout : float
        Base dropout probability.
    min_dropout : float
        Minimum dropout probability.
    max_dropout : float
        Maximum dropout probability.
    sensitivity : float
        How sensitive dropout is to regularity changes.

    Returns
    -------
    float
        Adaptive dropout probability.
    """
    # Sigmoid-like mapping centered at base_dropout
    # regularity 0.5 → base_dropout
    # regularity 1.0 → max_dropout
    # regularity 0.0 → min_dropout

    delta = (regularity_score - 0.5) * sensitivity
    dropout = base_dropout + (max_dropout - base_dropout) * (1 / (1 + np.exp(-delta)) - 0.5) * 2

    return float(np.clip(dropout, min_dropout, max_dropout))


if __name__ == "__main__":
    # Quick test
    import time

    # Create test image
    img = (np.random.rand(256, 256) * 255).astype(np.uint8)

    # Test CPU version
    start = time.time()
    for _ in range(100):
        entropy = orientation_entropy_cv2(img)
    cpu_time = time.time() - start
    print(f"CPU (100 iterations): {cpu_time:.3f}s, entropy={entropy:.4f}")

    # Test regularity score
    reg_score, components = compute_regularity_score(img)
    print(f"Regularity score: {reg_score:.4f}")
    print(f"Components: {components}")

    # Test adaptive dropout
    dropout = adaptive_dropout_prob(reg_score)
    print(f"Adaptive dropout: {dropout:.4f}")
