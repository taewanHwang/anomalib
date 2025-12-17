"""2D FFT-based Frequency Filters for Anomaly Detection.

This module implements various frequency domain filters including:
- High-Pass Filters (HPF): Ideal, Gaussian, Butterworth
- Low-Pass Filters (LPF): Ideal, Gaussian, Butterworth

Based on FAIR paper insights: reconstruction-based anomaly detection trade-off
is a frequency domain problem, not a spatial domain problem.

Reference: FAIR - Frequency-aware Image Restoration for Industrial Anomaly Detection
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

import numpy as np
import torch


class FilterType(Enum):
    """Filter type enumeration."""
    IDEAL = "ideal"
    GAUSSIAN = "gaussian"
    BUTTERWORTH = "butterworth"


class FilterMode(Enum):
    """Filter mode enumeration."""
    HIGH_PASS = "hpf"
    LOW_PASS = "lpf"


def create_frequency_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Create frequency coordinate grid for filter construction.

    Args:
        height: Image height
        width: Image width

    Returns:
        Tuple of (u, v) frequency coordinate grids
    """
    # Create frequency coordinates centered at (0, 0)
    u = np.fft.fftfreq(height)
    v = np.fft.fftfreq(width)

    # Create 2D meshgrid
    V, U = np.meshgrid(v, u)

    return U, V


def create_distance_from_center(height: int, width: int) -> np.ndarray:
    """Create distance matrix from center of frequency domain.

    Args:
        height: Image height
        width: Image width

    Returns:
        Distance matrix where D[u, v] = sqrt(u^2 + v^2) normalized to [0, 1]
    """
    U, V = create_frequency_grid(height, width)

    # Distance from center (DC component)
    # Normalized so that maximum distance is approximately 0.5 (Nyquist)
    D = np.sqrt(U**2 + V**2)

    return D


def ideal_filter(
    height: int,
    width: int,
    cutoff: float,
    mode: FilterMode = FilterMode.LOW_PASS
) -> np.ndarray:
    """Create Ideal frequency filter.

    Ideal filter has sharp cutoff - passes or blocks all frequencies
    beyond the cutoff completely.

    Warning: Can cause ringing artifacts due to sharp transition.

    Args:
        height: Image height
        width: Image width
        cutoff: Cutoff frequency (normalized, 0-0.5 where 0.5 is Nyquist)
        mode: LOW_PASS or HIGH_PASS

    Returns:
        Filter mask of shape (height, width)
    """
    D = create_distance_from_center(height, width)

    if mode == FilterMode.LOW_PASS:
        # Pass frequencies below cutoff
        H = (D <= cutoff).astype(np.float32)
    else:
        # Pass frequencies above cutoff
        H = (D > cutoff).astype(np.float32)

    return H


def gaussian_filter(
    height: int,
    width: int,
    cutoff: float,
    mode: FilterMode = FilterMode.LOW_PASS
) -> np.ndarray:
    """Create Gaussian frequency filter.

    Gaussian filter has smooth transition - no ringing artifacts
    but may preserve too much of blocked frequencies.

    H_lpf(u,v) = exp(-D(u,v)^2 / (2 * cutoff^2))
    H_hpf(u,v) = 1 - H_lpf(u,v)

    Args:
        height: Image height
        width: Image width
        cutoff: Cutoff frequency (controls width of Gaussian)
        mode: LOW_PASS or HIGH_PASS

    Returns:
        Filter mask of shape (height, width)
    """
    D = create_distance_from_center(height, width)

    # Gaussian low-pass
    # Avoid division by zero
    if cutoff <= 0:
        cutoff = 1e-6

    H_lpf = np.exp(-(D**2) / (2 * cutoff**2))

    if mode == FilterMode.LOW_PASS:
        return H_lpf.astype(np.float32)
    else:
        return (1.0 - H_lpf).astype(np.float32)


def butterworth_filter(
    height: int,
    width: int,
    cutoff: float,
    order: int = 2,
    mode: FilterMode = FilterMode.LOW_PASS
) -> np.ndarray:
    """Create Butterworth frequency filter.

    Butterworth filter provides a good compromise between Ideal and Gaussian:
    - Smoother transition than Ideal (less ringing)
    - Sharper cutoff than Gaussian (better frequency separation)

    H_lpf(u,v) = 1 / (1 + (D(u,v) / cutoff)^(2*order))
    H_hpf(u,v) = 1 - H_lpf(u,v)

    Higher order = sharper transition (closer to Ideal)

    FAIR paper finding: 2nd-order Butterworth HPF (2-BHPF) is optimal.

    Args:
        height: Image height
        width: Image width
        cutoff: Cutoff frequency (normalized, 0-0.5)
        order: Filter order (1, 2, 3, ...). Higher = sharper transition.
        mode: LOW_PASS or HIGH_PASS

    Returns:
        Filter mask of shape (height, width)
    """
    D = create_distance_from_center(height, width)

    # Avoid division by zero
    if cutoff <= 0:
        cutoff = 1e-6

    # Butterworth low-pass
    # Add small epsilon to avoid numerical issues
    H_lpf = 1.0 / (1.0 + (D / cutoff + 1e-10)**(2 * order))

    if mode == FilterMode.LOW_PASS:
        return H_lpf.astype(np.float32)
    else:
        return (1.0 - H_lpf).astype(np.float32)


def create_filter(
    height: int,
    width: int,
    filter_type: FilterType | str,
    cutoff: float,
    mode: FilterMode | str = FilterMode.LOW_PASS,
    order: int = 2
) -> np.ndarray:
    """Factory function to create frequency filter.

    Args:
        height: Image height
        width: Image width
        filter_type: Type of filter (ideal, gaussian, butterworth)
        cutoff: Cutoff frequency (normalized, 0-0.5)
        mode: LOW_PASS (lpf) or HIGH_PASS (hpf)
        order: Filter order (only for Butterworth)

    Returns:
        Filter mask of shape (height, width)
    """
    # Convert string to enum if needed
    if isinstance(filter_type, str):
        filter_type = FilterType(filter_type.lower())
    if isinstance(mode, str):
        mode = FilterMode(mode.lower())

    if filter_type == FilterType.IDEAL:
        return ideal_filter(height, width, cutoff, mode)
    elif filter_type == FilterType.GAUSSIAN:
        return gaussian_filter(height, width, cutoff, mode)
    elif filter_type == FilterType.BUTTERWORTH:
        return butterworth_filter(height, width, cutoff, order, mode)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def apply_frequency_filter(
    image: np.ndarray | torch.Tensor,
    filter_mask: np.ndarray
) -> np.ndarray:
    """Apply frequency filter to image using 2D FFT.

    Pipeline: image -> FFT -> shift -> filter -> shift -> iFFT -> filtered_image

    Args:
        image: Input image (H, W) or (C, H, W) or (B, C, H, W)
        filter_mask: Frequency domain filter mask (H, W)

    Returns:
        Filtered image (same shape as input)
    """
    # Convert torch tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        was_tensor = True
        device = image.device
        dtype = image.dtype
        image = image.detach().cpu().numpy()
    else:
        was_tensor = False

    original_shape = image.shape

    # Handle different input dimensions
    if image.ndim == 2:
        # Single image (H, W)
        filtered = _apply_filter_2d(image, filter_mask)
    elif image.ndim == 3:
        # (C, H, W) - apply to each channel
        filtered = np.stack([
            _apply_filter_2d(image[c], filter_mask)
            for c in range(image.shape[0])
        ], axis=0)
    elif image.ndim == 4:
        # (B, C, H, W) - apply to each batch and channel
        filtered = np.stack([
            np.stack([
                _apply_filter_2d(image[b, c], filter_mask)
                for c in range(image.shape[1])
            ], axis=0)
            for b in range(image.shape[0])
        ], axis=0)
    else:
        raise ValueError(f"Unsupported image shape: {original_shape}")

    # Convert back to torch if input was tensor
    if was_tensor:
        filtered = torch.from_numpy(filtered).to(device=device, dtype=dtype)

    return filtered


def _apply_filter_2d(image: np.ndarray, filter_mask: np.ndarray) -> np.ndarray:
    """Apply frequency filter to single 2D image.

    Args:
        image: Input image (H, W)
        filter_mask: Filter mask (H, W)

    Returns:
        Filtered image (H, W)
    """
    # 2D FFT
    f_transform = np.fft.fft2(image)

    # Shift zero frequency to center
    f_shift = np.fft.fftshift(f_transform)

    # Apply filter
    f_filtered = f_shift * np.fft.fftshift(filter_mask)

    # Shift back
    f_ishift = np.fft.ifftshift(f_filtered)

    # Inverse FFT
    filtered = np.fft.ifft2(f_ishift)

    # Return real part (imaginary should be ~0 for real input)
    return np.real(filtered).astype(np.float32)


def get_magnitude_spectrum(
    image: np.ndarray | torch.Tensor,
    log_scale: bool = True,
    shift: bool = True
) -> np.ndarray:
    """Compute magnitude spectrum of image.

    Args:
        image: Input image (H, W) or (C, H, W)
        log_scale: Apply log scaling (log(1 + magnitude))
        shift: Shift zero frequency to center

    Returns:
        Magnitude spectrum (same spatial shape as input)
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if image.ndim == 3:
        # Use first channel
        image = image[0]

    # 2D FFT
    f_transform = np.fft.fft2(image)

    if shift:
        f_transform = np.fft.fftshift(f_transform)

    # Magnitude
    magnitude = np.abs(f_transform)

    if log_scale:
        magnitude = np.log1p(magnitude)

    return magnitude.astype(np.float32)


def get_phase_spectrum(
    image: np.ndarray | torch.Tensor,
    shift: bool = True
) -> np.ndarray:
    """Compute phase spectrum of image.

    Args:
        image: Input image (H, W) or (C, H, W)
        shift: Shift zero frequency to center

    Returns:
        Phase spectrum in radians [-pi, pi]
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if image.ndim == 3:
        image = image[0]

    f_transform = np.fft.fft2(image)

    if shift:
        f_transform = np.fft.fftshift(f_transform)

    phase = np.angle(f_transform)

    return phase.astype(np.float32)


def get_frequency_band_energy(
    image: np.ndarray | torch.Tensor,
    n_bands: int = 5
) -> dict[str, float]:
    """Compute energy distribution across frequency bands.

    Divides frequency domain into radial bands and computes energy in each.

    Args:
        image: Input image (H, W) or (C, H, W)
        n_bands: Number of frequency bands

    Returns:
        Dictionary with band names and their energy ratios
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if image.ndim == 3:
        image = image[0]

    height, width = image.shape

    # Get magnitude spectrum (not log-scaled for energy calculation)
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    # Distance from center
    D = create_distance_from_center(height, width)
    D_shifted = np.fft.fftshift(D)

    # Total energy
    total_energy = np.sum(magnitude**2)

    # Divide into bands
    max_dist = 0.5  # Nyquist frequency
    band_width = max_dist / n_bands

    band_energies = {}
    band_names = ["very_low", "low", "mid", "high", "very_high"][:n_bands]

    for i, name in enumerate(band_names):
        low_bound = i * band_width
        high_bound = (i + 1) * band_width

        # Create band mask
        if i == n_bands - 1:
            # Last band includes everything above
            band_mask = D_shifted >= low_bound
        else:
            band_mask = (D_shifted >= low_bound) & (D_shifted < high_bound)

        # Compute band energy
        band_energy = np.sum(magnitude[band_mask]**2)

        # Normalize by total energy
        if total_energy > 0:
            band_energies[name] = float(band_energy / total_energy)
        else:
            band_energies[name] = 0.0

    return band_energies


class FrequencyFilter:
    """Convenience class for frequency filtering operations.

    Example:
        >>> filter = FrequencyFilter(
        ...     filter_type="butterworth",
        ...     cutoff=0.1,
        ...     mode="hpf",
        ...     order=2
        ... )
        >>> filtered_image = filter(image)
    """

    def __init__(
        self,
        filter_type: FilterType | str = "butterworth",
        cutoff: float = 0.1,
        mode: FilterMode | str = "lpf",
        order: int = 2
    ):
        """Initialize frequency filter.

        Args:
            filter_type: Type of filter (ideal, gaussian, butterworth)
            cutoff: Cutoff frequency (normalized, 0-0.5)
            mode: Filter mode (lpf or hpf)
            order: Filter order (only for Butterworth)
        """
        self.filter_type = FilterType(filter_type) if isinstance(filter_type, str) else filter_type
        self.cutoff = cutoff
        self.mode = FilterMode(mode) if isinstance(mode, str) else mode
        self.order = order

        # Cache for filter masks
        self._filter_cache: dict[tuple[int, int], np.ndarray] = {}

    def get_filter_mask(self, height: int, width: int) -> np.ndarray:
        """Get or create filter mask for given dimensions.

        Args:
            height: Image height
            width: Image width

        Returns:
            Filter mask
        """
        key = (height, width)

        if key not in self._filter_cache:
            self._filter_cache[key] = create_filter(
                height, width,
                self.filter_type,
                self.cutoff,
                self.mode,
                self.order
            )

        return self._filter_cache[key]

    def __call__(self, image: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Apply filter to image.

        Args:
            image: Input image

        Returns:
            Filtered image
        """
        # Get image dimensions
        if isinstance(image, torch.Tensor):
            if image.ndim == 4:
                height, width = image.shape[2], image.shape[3]
            elif image.ndim == 3:
                height, width = image.shape[1], image.shape[2]
            else:
                height, width = image.shape[0], image.shape[1]
        else:
            if image.ndim == 4:
                height, width = image.shape[2], image.shape[3]
            elif image.ndim == 3:
                height, width = image.shape[1], image.shape[2]
            else:
                height, width = image.shape[0], image.shape[1]

        filter_mask = self.get_filter_mask(height, width)

        return apply_frequency_filter(image, filter_mask)

    def __repr__(self) -> str:
        return (
            f"FrequencyFilter("
            f"type={self.filter_type.value}, "
            f"cutoff={self.cutoff}, "
            f"mode={self.mode.value}, "
            f"order={self.order})"
        )


# Convenience functions for common filter configurations
def apply_butterworth_hpf(
    image: np.ndarray | torch.Tensor,
    cutoff: float = 0.1,
    order: int = 2
) -> np.ndarray | torch.Tensor:
    """Apply 2nd-order Butterworth High-Pass Filter (recommended by FAIR).

    Args:
        image: Input image
        cutoff: Cutoff frequency (normalized, 0-0.5)
        order: Filter order (default: 2)

    Returns:
        High-pass filtered image
    """
    filter_obj = FrequencyFilter(
        filter_type="butterworth",
        cutoff=cutoff,
        mode="hpf",
        order=order
    )
    return filter_obj(image)


def apply_butterworth_lpf(
    image: np.ndarray | torch.Tensor,
    cutoff: float = 0.1,
    order: int = 2
) -> np.ndarray | torch.Tensor:
    """Apply Butterworth Low-Pass Filter.

    Args:
        image: Input image
        cutoff: Cutoff frequency (normalized, 0-0.5)
        order: Filter order (default: 2)

    Returns:
        Low-pass filtered image
    """
    filter_obj = FrequencyFilter(
        filter_type="butterworth",
        cutoff=cutoff,
        mode="lpf",
        order=order
    )
    return filter_obj(image)
