"""Visualization utilities for frequency analysis EDA.

This module provides visualization functions for:
- Magnitude spectrum visualization
- Filter response visualization
- Normal vs Anomaly frequency comparison
- CutPaste effect visualization
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

try:
    from .frequency_filters import (
        FilterMode,
        FilterType,
        create_filter,
        get_frequency_band_energy,
        get_magnitude_spectrum,
    )
except ImportError:
    from frequency_filters import (
        FilterMode,
        FilterType,
        create_filter,
        get_frequency_band_energy,
        get_magnitude_spectrum,
    )


def setup_plot_style():
    """Setup matplotlib style for consistent visualization."""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'figure.dpi': 100,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'font.family': 'DejaVu Sans',
    })


def plot_image_with_spectrum(
    image: np.ndarray,
    title: str = "Image",
    save_path: Path | str | None = None,
    figsize: tuple[int, int] = (12, 4)
) -> plt.Figure:
    """Plot image alongside its magnitude spectrum.

    Args:
        image: Input image (H, W) or (C, H, W)
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if image.ndim == 3:
        image = image[0]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Original image
    axes[0].imshow(image, cmap='viridis', aspect='auto')
    axes[0].set_title(f"{title} - Original")
    axes[0].set_xlabel("Width")
    axes[0].set_ylabel("Height")

    # Magnitude spectrum (log scale)
    magnitude = get_magnitude_spectrum(image, log_scale=True)
    im1 = axes[1].imshow(magnitude, cmap='hot', aspect='auto')
    axes[1].set_title(f"{title} - Log Magnitude Spectrum")
    axes[1].set_xlabel("Frequency (u)")
    axes[1].set_ylabel("Frequency (v)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Magnitude spectrum (linear scale)
    magnitude_linear = get_magnitude_spectrum(image, log_scale=False)
    im2 = axes[2].imshow(magnitude_linear, cmap='hot', aspect='auto')
    axes[2].set_title(f"{title} - Magnitude Spectrum")
    axes[2].set_xlabel("Frequency (u)")
    axes[2].set_ylabel("Frequency (v)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_filter_comparison(
    height: int,
    width: int,
    cutoffs: Sequence[float] = (0.05, 0.1, 0.15, 0.2),
    save_path: Path | str | None = None
) -> plt.Figure:
    """Plot comparison of different filter types and cutoffs.

    Args:
        height: Image height
        width: Image width
        cutoffs: List of cutoff frequencies to compare
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    filter_types = [FilterType.IDEAL, FilterType.GAUSSIAN, FilterType.BUTTERWORTH]
    n_cutoffs = len(cutoffs)
    n_types = len(filter_types)

    fig, axes = plt.subplots(n_types * 2, n_cutoffs, figsize=(4 * n_cutoffs, 3 * n_types * 2))

    for i, ftype in enumerate(filter_types):
        for j, cutoff in enumerate(cutoffs):
            # Low-pass filter
            lpf = create_filter(height, width, ftype, cutoff, FilterMode.LOW_PASS, order=2)
            axes[i * 2, j].imshow(np.fft.fftshift(lpf), cmap='gray', aspect='auto', vmin=0, vmax=1)
            axes[i * 2, j].set_title(f"{ftype.value.title()} LPF\ncutoff={cutoff}")
            axes[i * 2, j].axis('off')

            # High-pass filter
            hpf = create_filter(height, width, ftype, cutoff, FilterMode.HIGH_PASS, order=2)
            axes[i * 2 + 1, j].imshow(np.fft.fftshift(hpf), cmap='gray', aspect='auto', vmin=0, vmax=1)
            axes[i * 2 + 1, j].set_title(f"{ftype.value.title()} HPF\ncutoff={cutoff}")
            axes[i * 2 + 1, j].axis('off')

    plt.suptitle(f"Filter Comparison (Image size: {width}x{height})", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_butterworth_orders(
    height: int,
    width: int,
    cutoff: float = 0.1,
    orders: Sequence[int] = (1, 2, 3, 4, 5),
    save_path: Path | str | None = None
) -> plt.Figure:
    """Plot comparison of different Butterworth filter orders.

    Args:
        height: Image height
        width: Image width
        cutoff: Cutoff frequency
        orders: List of orders to compare
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    n_orders = len(orders)

    fig, axes = plt.subplots(2, n_orders, figsize=(3 * n_orders, 6))

    for j, order in enumerate(orders):
        # Low-pass
        lpf = create_filter(height, width, FilterType.BUTTERWORTH, cutoff, FilterMode.LOW_PASS, order=order)
        axes[0, j].imshow(np.fft.fftshift(lpf), cmap='gray', aspect='auto', vmin=0, vmax=1)
        axes[0, j].set_title(f"Order {order} LPF")
        axes[0, j].axis('off')

        # High-pass
        hpf = create_filter(height, width, FilterType.BUTTERWORTH, cutoff, FilterMode.HIGH_PASS, order=order)
        axes[1, j].imshow(np.fft.fftshift(hpf), cmap='gray', aspect='auto', vmin=0, vmax=1)
        axes[1, j].set_title(f"Order {order} HPF")
        axes[1, j].axis('off')

    plt.suptitle(f"Butterworth Filter Orders (cutoff={cutoff})", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_filter_effect_on_image(
    image: np.ndarray,
    cutoffs: Sequence[float] = (0.05, 0.1, 0.15, 0.2),
    filter_type: FilterType | str = FilterType.BUTTERWORTH,
    order: int = 2,
    save_path: Path | str | None = None
) -> plt.Figure:
    """Plot the effect of HPF and LPF on an image.

    Args:
        image: Input image (H, W) or (C, H, W)
        cutoffs: List of cutoff frequencies
        filter_type: Type of filter
        order: Butterworth order
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    try:
        from .frequency_filters import apply_frequency_filter
    except ImportError:
        from frequency_filters import apply_frequency_filter

    if isinstance(filter_type, str):
        filter_type = FilterType(filter_type.lower())

    if image.ndim == 3:
        image = image[0]

    height, width = image.shape
    n_cutoffs = len(cutoffs)

    fig, axes = plt.subplots(3, n_cutoffs + 1, figsize=(3 * (n_cutoffs + 1), 9))

    # Original image column
    axes[0, 0].imshow(image, cmap='viridis', aspect='auto')
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')

    orig_spectrum = get_magnitude_spectrum(image, log_scale=True)
    axes[1, 0].imshow(orig_spectrum, cmap='hot', aspect='auto')
    axes[1, 0].set_title("Original Spectrum")
    axes[1, 0].axis('off')

    axes[2, 0].axis('off')

    # Filtered images
    for j, cutoff in enumerate(cutoffs):
        # HPF
        hpf_mask = create_filter(height, width, filter_type, cutoff, FilterMode.HIGH_PASS, order)
        hpf_image = apply_frequency_filter(image, hpf_mask)

        axes[0, j + 1].imshow(hpf_image, cmap='viridis', aspect='auto')
        axes[0, j + 1].set_title(f"HPF (cutoff={cutoff})")
        axes[0, j + 1].axis('off')

        hpf_spectrum = get_magnitude_spectrum(hpf_image, log_scale=True)
        axes[1, j + 1].imshow(hpf_spectrum, cmap='hot', aspect='auto')
        axes[1, j + 1].set_title(f"HPF Spectrum")
        axes[1, j + 1].axis('off')

        # LPF
        lpf_mask = create_filter(height, width, filter_type, cutoff, FilterMode.LOW_PASS, order)
        lpf_image = apply_frequency_filter(image, lpf_mask)

        axes[2, j + 1].imshow(lpf_image, cmap='viridis', aspect='auto')
        axes[2, j + 1].set_title(f"LPF (cutoff={cutoff})")
        axes[2, j + 1].axis('off')

    plt.suptitle(f"{filter_type.value.title()} Filter Effects (order={order})", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_frequency_band_comparison(
    normal_images: Sequence[np.ndarray],
    anomaly_images: Sequence[np.ndarray],
    n_bands: int = 5,
    save_path: Path | str | None = None
) -> plt.Figure:
    """Plot frequency band energy comparison between normal and anomaly images.

    Args:
        normal_images: List of normal images
        anomaly_images: List of anomaly images
        n_bands: Number of frequency bands
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    # Compute band energies for all images
    normal_energies = []
    anomaly_energies = []

    for img in normal_images:
        if img.ndim == 3:
            img = img[0]
        normal_energies.append(get_frequency_band_energy(img, n_bands))

    for img in anomaly_images:
        if img.ndim == 3:
            img = img[0]
        anomaly_energies.append(get_frequency_band_energy(img, n_bands))

    # Average across images
    band_names = list(normal_energies[0].keys())

    normal_means = {
        band: np.mean([e[band] for e in normal_energies]) for band in band_names
    }
    normal_stds = {
        band: np.std([e[band] for e in normal_energies]) for band in band_names
    }

    anomaly_means = {
        band: np.mean([e[band] for e in anomaly_energies]) for band in band_names
    }
    anomaly_stds = {
        band: np.std([e[band] for e in anomaly_energies]) for band in band_names
    }

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(band_names))
    width = 0.35

    # Bar plot
    normal_vals = [normal_means[b] for b in band_names]
    normal_errs = [normal_stds[b] for b in band_names]
    anomaly_vals = [anomaly_means[b] for b in band_names]
    anomaly_errs = [anomaly_stds[b] for b in band_names]

    bars1 = axes[0].bar(x - width/2, normal_vals, width, yerr=normal_errs,
                        label='Normal', color='blue', alpha=0.7, capsize=3)
    bars2 = axes[0].bar(x + width/2, anomaly_vals, width, yerr=anomaly_errs,
                        label='Anomaly', color='red', alpha=0.7, capsize=3)

    axes[0].set_xlabel('Frequency Band')
    axes[0].set_ylabel('Energy Ratio')
    axes[0].set_title('Frequency Band Energy Distribution')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([b.replace('_', ' ').title() for b in band_names])
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Difference plot
    diff_vals = [anomaly_means[b] - normal_means[b] for b in band_names]
    colors = ['green' if d > 0 else 'purple' for d in diff_vals]

    axes[1].bar(x, diff_vals, color=colors, alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel('Frequency Band')
    axes[1].set_ylabel('Energy Difference (Anomaly - Normal)')
    axes[1].set_title('Frequency Band Energy Difference')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([b.replace('_', ' ').title() for b in band_names])
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_average_spectrum_comparison(
    normal_images: Sequence[np.ndarray],
    anomaly_images: Sequence[np.ndarray],
    save_path: Path | str | None = None
) -> plt.Figure:
    """Plot average magnitude spectrum comparison.

    Args:
        normal_images: List of normal images
        anomaly_images: List of anomaly images
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    # Compute average spectra
    normal_spectra = []
    anomaly_spectra = []

    for img in normal_images:
        if img.ndim == 3:
            img = img[0]
        normal_spectra.append(get_magnitude_spectrum(img, log_scale=True))

    for img in anomaly_images:
        if img.ndim == 3:
            img = img[0]
        anomaly_spectra.append(get_magnitude_spectrum(img, log_scale=True))

    avg_normal = np.mean(normal_spectra, axis=0)
    avg_anomaly = np.mean(anomaly_spectra, axis=0)
    diff_spectrum = avg_anomaly - avg_normal

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Normal average spectrum
    im0 = axes[0].imshow(avg_normal, cmap='hot', aspect='auto')
    axes[0].set_title(f'Average Normal Spectrum\n(n={len(normal_images)})')
    axes[0].set_xlabel('Frequency (u)')
    axes[0].set_ylabel('Frequency (v)')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Anomaly average spectrum
    im1 = axes[1].imshow(avg_anomaly, cmap='hot', aspect='auto')
    axes[1].set_title(f'Average Anomaly Spectrum\n(n={len(anomaly_images)})')
    axes[1].set_xlabel('Frequency (u)')
    axes[1].set_ylabel('Frequency (v)')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Difference spectrum
    vmax = max(abs(diff_spectrum.min()), abs(diff_spectrum.max()))
    im2 = axes[2].imshow(diff_spectrum, cmap='RdBu_r', aspect='auto',
                         vmin=-vmax, vmax=vmax)
    axes[2].set_title('Difference (Anomaly - Normal)')
    axes[2].set_xlabel('Frequency (u)')
    axes[2].set_ylabel('Frequency (v)')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    # Radial average
    height, width = avg_normal.shape
    center_y, center_x = height // 2, width // 2
    Y, X = np.ogrid[:height, :width]
    r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    r = r.astype(int)

    max_r = min(center_x, center_y)
    radial_normal = np.zeros(max_r)
    radial_anomaly = np.zeros(max_r)

    for i in range(max_r):
        mask = r == i
        if np.any(mask):
            radial_normal[i] = avg_normal[mask].mean()
            radial_anomaly[i] = avg_anomaly[mask].mean()

    freq_axis = np.arange(max_r) / max_r * 0.5  # Normalize to [0, 0.5]

    axes[3].plot(freq_axis, radial_normal, 'b-', label='Normal', linewidth=2)
    axes[3].plot(freq_axis, radial_anomaly, 'r-', label='Anomaly', linewidth=2)
    axes[3].fill_between(freq_axis, radial_normal, radial_anomaly,
                         alpha=0.3, color='gray')
    axes[3].set_xlabel('Normalized Frequency')
    axes[3].set_ylabel('Average Log Magnitude')
    axes[3].set_title('Radial Average Spectrum')
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_cutpaste_frequency_analysis(
    original_image: np.ndarray,
    cutpaste_image: np.ndarray,
    fault_mask: np.ndarray | None = None,
    patch_info: dict | None = None,
    save_path: Path | str | None = None
) -> plt.Figure:
    """Analyze frequency characteristics of CutPaste augmentation.

    Args:
        original_image: Original image before CutPaste
        cutpaste_image: Image after CutPaste augmentation
        fault_mask: Optional fault mask
        patch_info: Optional patch information dictionary
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    if original_image.ndim == 3:
        original_image = original_image[0]
    if cutpaste_image.ndim == 3:
        cutpaste_image = cutpaste_image[0]

    # Compute difference
    diff_image = cutpaste_image - original_image

    # Get spectra
    orig_spectrum = get_magnitude_spectrum(original_image, log_scale=True)
    cutpaste_spectrum = get_magnitude_spectrum(cutpaste_image, log_scale=True)
    diff_spectrum = get_magnitude_spectrum(diff_image, log_scale=True)

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 0.8])

    # Row 1: Original
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.imshow(original_image, cmap='viridis', aspect='auto')
    ax00.set_title('Original Image')
    ax00.axis('off')

    ax01 = fig.add_subplot(gs[0, 1])
    im01 = ax01.imshow(orig_spectrum, cmap='hot', aspect='auto')
    ax01.set_title('Original Spectrum')
    ax01.axis('off')
    plt.colorbar(im01, ax=ax01, fraction=0.046)

    # Row 1: CutPaste
    ax02 = fig.add_subplot(gs[0, 2])
    ax02.imshow(cutpaste_image, cmap='viridis', aspect='auto')
    ax02.set_title('CutPaste Image')
    ax02.axis('off')

    ax03 = fig.add_subplot(gs[0, 3])
    im03 = ax03.imshow(cutpaste_spectrum, cmap='hot', aspect='auto')
    ax03.set_title('CutPaste Spectrum')
    ax03.axis('off')
    plt.colorbar(im03, ax=ax03, fraction=0.046)

    # Row 2: Difference
    ax10 = fig.add_subplot(gs[1, 0])
    vmax_diff = max(abs(diff_image.min()), abs(diff_image.max()))
    if vmax_diff > 0:
        im10 = ax10.imshow(diff_image, cmap='RdBu_r', aspect='auto',
                          vmin=-vmax_diff, vmax=vmax_diff)
    else:
        im10 = ax10.imshow(diff_image, cmap='RdBu_r', aspect='auto')
    ax10.set_title('Difference Image')
    ax10.axis('off')
    plt.colorbar(im10, ax=ax10, fraction=0.046)

    ax11 = fig.add_subplot(gs[1, 1])
    im11 = ax11.imshow(diff_spectrum, cmap='hot', aspect='auto')
    ax11.set_title('Difference Spectrum')
    ax11.axis('off')
    plt.colorbar(im11, ax=ax11, fraction=0.046)

    # Fault mask if available
    if fault_mask is not None:
        if fault_mask.ndim > 2:
            fault_mask = fault_mask.squeeze()
        ax12 = fig.add_subplot(gs[1, 2])
        ax12.imshow(fault_mask, cmap='gray', aspect='auto')
        ax12.set_title('Fault Mask')
        ax12.axis('off')

    # Spectrum difference
    ax13 = fig.add_subplot(gs[1, 3])
    spec_diff = cutpaste_spectrum - orig_spectrum
    vmax_spec = max(abs(spec_diff.min()), abs(spec_diff.max()))
    if vmax_spec > 0:
        im13 = ax13.imshow(spec_diff, cmap='RdBu_r', aspect='auto',
                          vmin=-vmax_spec, vmax=vmax_spec)
    else:
        im13 = ax13.imshow(spec_diff, cmap='RdBu_r', aspect='auto')
    ax13.set_title('Spectrum Difference')
    ax13.axis('off')
    plt.colorbar(im13, ax=ax13, fraction=0.046)

    # Row 3: Band energy comparison
    ax20 = fig.add_subplot(gs[2, :2])
    orig_bands = get_frequency_band_energy(original_image, n_bands=5)
    cutpaste_bands = get_frequency_band_energy(cutpaste_image, n_bands=5)

    band_names = list(orig_bands.keys())
    x = np.arange(len(band_names))
    width = 0.35

    ax20.bar(x - width/2, [orig_bands[b] for b in band_names], width,
             label='Original', color='blue', alpha=0.7)
    ax20.bar(x + width/2, [cutpaste_bands[b] for b in band_names], width,
             label='CutPaste', color='red', alpha=0.7)
    ax20.set_xlabel('Frequency Band')
    ax20.set_ylabel('Energy Ratio')
    ax20.set_title('Frequency Band Energy')
    ax20.set_xticks(x)
    ax20.set_xticklabels([b.replace('_', ' ').title() for b in band_names])
    ax20.legend()
    ax20.grid(axis='y', alpha=0.3)

    # Patch info text
    ax21 = fig.add_subplot(gs[2, 2:])
    ax21.axis('off')

    if patch_info:
        info_text = "CutPaste Parameters:\n"
        info_text += f"  Patch size: {patch_info.get('cut_w', 'N/A')} x {patch_info.get('cut_h', 'N/A')}\n"
        info_text += f"  From: ({patch_info.get('from_location_w', 'N/A')}, {patch_info.get('from_location_h', 'N/A')})\n"
        info_text += f"  To: ({patch_info.get('to_location_w', 'N/A')}, {patch_info.get('to_location_h', 'N/A')})\n"
        info_text += f"  Amplitude: {patch_info.get('a_fault', 'N/A'):.3f}\n"
        info_text += f"  Coverage: {patch_info.get('coverage_percentage', 'N/A'):.2f}%"
        ax21.text(0.1, 0.5, info_text, transform=ax21.transAxes,
                  fontsize=10, verticalalignment='center',
                  fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_multiple_samples_spectrum(
    images: Sequence[np.ndarray],
    titles: Sequence[str] | None = None,
    category: str = "Samples",
    max_samples: int = 8,
    save_path: Path | str | None = None
) -> plt.Figure:
    """Plot multiple image samples with their spectra.

    Args:
        images: List of images
        titles: Optional list of titles
        category: Category name for suptitle
        max_samples: Maximum number of samples to show
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    n_samples = min(len(images), max_samples)
    images = images[:n_samples]

    if titles is None:
        titles = [f"Sample {i+1}" for i in range(n_samples)]
    else:
        titles = titles[:n_samples]

    fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))

    if n_samples == 1:
        axes = axes.reshape(2, 1)

    for i, (img, title) in enumerate(zip(images, titles)):
        if img.ndim == 3:
            img = img[0]

        # Image
        axes[0, i].imshow(img, cmap='viridis', aspect='auto')
        axes[0, i].set_title(title)
        axes[0, i].axis('off')

        # Spectrum
        spectrum = get_magnitude_spectrum(img, log_scale=True)
        axes[1, i].imshow(spectrum, cmap='hot', aspect='auto')
        axes[1, i].set_title('Spectrum')
        axes[1, i].axis('off')

    plt.suptitle(f"{category} - Images and Magnitude Spectra", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
