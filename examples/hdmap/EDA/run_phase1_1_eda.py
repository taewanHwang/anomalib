#!/usr/bin/env python
"""Phase 1-1 EDA: Normal vs Anomaly Frequency Analysis for HDMAP.

This script performs comprehensive frequency analysis:
- Compare frequency characteristics between normal and anomaly images
- Visualize frequency bands on actual 2D FFT images
- Analyze fault data split (first 500 vs last 500 have different characteristics)
- Run analysis on all domains (A, B, C, D)

Usage:
    python run_phase1_1_eda.py --domain domain_A
    python run_phase1_1_eda.py --all_domains
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Wedge
from PIL import Image
from tqdm import tqdm

# Default number of workers for parallel processing
DEFAULT_NUM_WORKERS = 16

# Local imports
from frequency_filters import (
    create_distance_from_center,
    get_frequency_band_energy,
    get_magnitude_spectrum,
)


# Default paths
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "HDMAP" / "1000_tiff_minmax"
DOMAINS = ["domain_A", "domain_B", "domain_C", "domain_D"]


def _load_and_compute_band_energy(args: tuple) -> dict[str, float]:
    """Load image and compute band energy (combined for efficiency).

    Args:
        args: Tuple of (img_path, n_bands)

    Returns:
        Dictionary with band names and energy ratios
    """
    img_path, n_bands = args
    img = Image.open(img_path)
    img_array = np.array(img, dtype=np.float32)
    if img_array.ndim == 3:
        img_array = img_array[0]
    return get_frequency_band_energy(img_array, n_bands)


def compute_stats_from_paths_parallel(
    image_paths: list[Path],
    desc: str,
    n_bands: int = 5,
    n_workers: int = DEFAULT_NUM_WORKERS,
) -> tuple[dict, dict, list]:
    """Compute frequency band statistics in parallel directly from file paths.

    This is more efficient as it avoids transferring image data between processes.

    Args:
        image_paths: List of image file paths
        desc: Description for progress bar
        n_bands: Number of frequency bands
        n_workers: Number of parallel workers

    Returns:
        Tuple of (means, stds, band_names)
    """
    n_images = len(image_paths)

    if n_images == 0:
        raise ValueError("No image paths provided")

    # Prepare arguments
    args = [(path, n_bands) for path in image_paths]

    # Use ProcessPoolExecutor - workers load and process images directly
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        band_energies = list(tqdm(
            executor.map(_load_and_compute_band_energy, args),
            total=n_images,
            desc=desc
        ))

    band_names = list(band_energies[0].keys())
    means = {b: float(np.mean([e[b] for e in band_energies])) for b in band_names}
    stds = {b: float(np.std([e[b] for e in band_energies])) for b in band_names}

    return means, stds, band_names


def visualize_frequency_bands_on_fft(
    sample_image: np.ndarray,
    n_bands: int = 5,
    save_path: Path | None = None
) -> plt.Figure:
    """Visualize frequency bands overlaid on actual FFT magnitude spectrum.

    Args:
        sample_image: Sample image to visualize
        n_bands: Number of frequency bands
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    if sample_image.ndim == 3:
        sample_image = sample_image[0]

    height, width = sample_image.shape

    # Get magnitude spectrum
    f_transform = np.fft.fft2(sample_image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.log1p(np.abs(f_shift))

    # Get distance matrix (shifted to match fftshift)
    D = create_distance_from_center(height, width)
    D_shifted = np.fft.fftshift(D)

    # Band definitions
    max_dist = 0.5
    band_width = max_dist / n_bands
    band_names = ["very_low", "low", "mid", "high", "very_high"][:n_bands]
    band_colors = ['red', 'orange', 'yellow', 'green', 'blue'][:n_bands]

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Original image
    axes[0, 0].imshow(sample_image, cmap='viridis', aspect='auto')
    axes[0, 0].set_title(f'Original Image ({width}x{height})')
    axes[0, 0].set_xlabel('Width')
    axes[0, 0].set_ylabel('Height')

    # 2. Full magnitude spectrum with band overlay
    im1 = axes[0, 1].imshow(magnitude, cmap='hot', aspect='auto')
    axes[0, 1].set_title('Magnitude Spectrum with Band Boundaries')

    # Draw band boundaries as contours
    center_y, center_x = height // 2, width // 2

    # Calculate pixel distances for each band boundary
    # D is normalized [0, 0.5], we need to map to pixel coordinates
    max_pixel_dist = np.sqrt((height/2)**2 + (width/2)**2)

    for i in range(n_bands):
        boundary_freq = (i + 1) * band_width
        # Convert normalized frequency to approximate pixel radius
        # This is approximate due to rectangular image
        radius_y = boundary_freq * height
        radius_x = boundary_freq * width

        # Draw ellipse for band boundary
        theta = np.linspace(0, 2*np.pi, 100)
        x_ellipse = center_x + radius_x * np.cos(theta)
        y_ellipse = center_y + radius_y * np.sin(theta)

        axes[0, 1].plot(x_ellipse, y_ellipse, color=band_colors[i],
                       linestyle='--', linewidth=1.5, label=f'{band_names[i]} boundary')

    axes[0, 1].legend(loc='upper right', fontsize=8)
    axes[0, 1].set_xlabel('Frequency u')
    axes[0, 1].set_ylabel('Frequency v')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # 3. Distance matrix visualization
    im2 = axes[0, 2].imshow(D_shifted, cmap='viridis', aspect='auto')
    axes[0, 2].set_title('Normalized Distance from DC\n(used for band classification)')
    axes[0, 2].set_xlabel('Frequency u')
    axes[0, 2].set_ylabel('Frequency v')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, label='Normalized freq')

    # 4-8. Individual band masks and their contribution
    for i, (name, color) in enumerate(zip(band_names, band_colors)):
        if i >= 3:
            ax = axes[1, i - 3 + 1] if i < 5 else None
        else:
            ax = axes[1, i]

        if ax is None:
            continue

        low_bound = i * band_width
        high_bound = (i + 1) * band_width

        # Create band mask
        if i == n_bands - 1:
            band_mask = D_shifted >= low_bound
        else:
            band_mask = (D_shifted >= low_bound) & (D_shifted < high_bound)

        # Show magnitude only in this band
        band_magnitude = np.zeros_like(magnitude)
        band_magnitude[band_mask] = magnitude[band_mask]

        # Calculate energy ratio
        full_energy = np.sum(np.abs(f_shift)**2)
        band_energy = np.sum(np.abs(f_shift[band_mask])**2)
        energy_ratio = band_energy / full_energy * 100

        ax.imshow(band_magnitude, cmap='hot', aspect='auto',
                 vmin=0, vmax=magnitude.max())
        ax.set_title(f'{name.upper()}\nfreq: [{low_bound:.2f}, {high_bound:.2f})\nEnergy: {energy_ratio:.1f}%')
        ax.axis('off')

    # Hide unused subplot if n_bands < 5
    if n_bands < 5:
        for i in range(n_bands, 3):
            axes[1, i].axis('off')

    plt.suptitle('Frequency Band Visualization on 2D FFT', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_band_comparison_detailed(
    normal_images: list[np.ndarray],
    anomaly_images: list[np.ndarray],
    anomaly_first_half: list[np.ndarray],
    anomaly_second_half: list[np.ndarray],
    n_bands: int = 5,
    save_path: Path | None = None
) -> plt.Figure:
    """Detailed band comparison with fault split analysis.

    Args:
        normal_images: List of normal images
        anomaly_images: All anomaly images
        anomaly_first_half: First 500 anomaly images
        anomaly_second_half: Last 500 anomaly images
        n_bands: Number of frequency bands
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    def compute_band_stats(images):
        band_energies = []
        for img in images:
            if img.ndim == 3:
                img = img[0]
            band_energies.append(get_frequency_band_energy(img, n_bands))

        band_names = list(band_energies[0].keys())
        means = {b: np.mean([e[b] for e in band_energies]) for b in band_names}
        stds = {b: np.std([e[b] for e in band_energies]) for b in band_names}
        return means, stds, band_names

    normal_means, normal_stds, band_names = compute_band_stats(normal_images)
    anomaly_means, anomaly_stds, _ = compute_band_stats(anomaly_images)
    first_half_means, first_half_stds, _ = compute_band_stats(anomaly_first_half)
    second_half_means, second_half_stds, _ = compute_band_stats(anomaly_second_half)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.arange(len(band_names))
    width = 0.2

    # 1. Normal vs All Anomaly
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width/2, [normal_means[b] for b in band_names], width,
                    yerr=[normal_stds[b] for b in band_names],
                    label='Normal', color='blue', alpha=0.7, capsize=3)
    bars2 = ax1.bar(x + width/2, [anomaly_means[b] for b in band_names], width,
                    yerr=[anomaly_stds[b] for b in band_names],
                    label='Anomaly (All)', color='red', alpha=0.7, capsize=3)

    ax1.set_xlabel('Frequency Band')
    ax1.set_ylabel('Energy Ratio')
    ax1.set_title(f'Normal vs Anomaly (All)\nn_normal={len(normal_images)}, n_anomaly={len(anomaly_images)}')
    ax1.set_xticks(x)
    ax1.set_xticklabels([b.replace('_', '\n') for b in band_names])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 2. Anomaly First Half vs Second Half
    ax2 = axes[0, 1]
    bars3 = ax2.bar(x - width/2, [first_half_means[b] for b in band_names], width,
                    yerr=[first_half_stds[b] for b in band_names],
                    label='Fault First 500', color='orange', alpha=0.7, capsize=3)
    bars4 = ax2.bar(x + width/2, [second_half_means[b] for b in band_names], width,
                    yerr=[second_half_stds[b] for b in band_names],
                    label='Fault Last 500', color='purple', alpha=0.7, capsize=3)

    ax2.set_xlabel('Frequency Band')
    ax2.set_ylabel('Energy Ratio')
    ax2.set_title(f'Fault First 500 vs Last 500\n(Different fault characteristics)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([b.replace('_', '\n') for b in band_names])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 3. All groups comparison
    ax3 = axes[1, 0]
    width3 = 0.18
    positions = [x - 1.5*width3, x - 0.5*width3, x + 0.5*width3, x + 1.5*width3]

    ax3.bar(positions[0], [normal_means[b] for b in band_names], width3,
            label='Normal', color='blue', alpha=0.7)
    ax3.bar(positions[1], [first_half_means[b] for b in band_names], width3,
            label='Fault 1st 500', color='orange', alpha=0.7)
    ax3.bar(positions[2], [second_half_means[b] for b in band_names], width3,
            label='Fault 2nd 500', color='purple', alpha=0.7)
    ax3.bar(positions[3], [anomaly_means[b] for b in band_names], width3,
            label='Fault All', color='red', alpha=0.7)

    ax3.set_xlabel('Frequency Band')
    ax3.set_ylabel('Energy Ratio')
    ax3.set_title('All Groups Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels([b.replace('_', '\n') for b in band_names])
    ax3.legend(loc='upper right')
    ax3.grid(axis='y', alpha=0.3)

    # 4. Difference from Normal
    ax4 = axes[1, 1]
    diff_all = [anomaly_means[b] - normal_means[b] for b in band_names]
    diff_first = [first_half_means[b] - normal_means[b] for b in band_names]
    diff_second = [second_half_means[b] - normal_means[b] for b in band_names]

    ax4.bar(x - width, diff_first, width, label='Fault 1st - Normal', color='orange', alpha=0.7)
    ax4.bar(x, diff_second, width, label='Fault 2nd - Normal', color='purple', alpha=0.7)
    ax4.bar(x + width, diff_all, width, label='Fault All - Normal', color='red', alpha=0.7)

    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Frequency Band')
    ax4.set_ylabel('Energy Difference')
    ax4.set_title('Energy Difference from Normal')
    ax4.set_xticks(x)
    ax4.set_xticklabels([b.replace('_', '\n') for b in band_names])
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_average_spectrum_with_bands(
    normal_images: list[np.ndarray],
    anomaly_first_half: list[np.ndarray],
    anomaly_second_half: list[np.ndarray],
    n_bands: int = 5,
    save_path: Path | None = None
) -> plt.Figure:
    """Visualize average spectra with band overlays.

    Args:
        normal_images: Normal images
        anomaly_first_half: First half anomaly images
        anomaly_second_half: Second half anomaly images
        n_bands: Number of bands
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    def compute_avg_spectrum(images):
        spectra = []
        for img in images:
            if img.ndim == 3:
                img = img[0]
            spectra.append(get_magnitude_spectrum(img, log_scale=True))
        return np.mean(spectra, axis=0)

    avg_normal = compute_avg_spectrum(normal_images)
    avg_first = compute_avg_spectrum(anomaly_first_half)
    avg_second = compute_avg_spectrum(anomaly_second_half)

    # Compute differences
    diff_first = avg_first - avg_normal
    diff_second = avg_second - avg_normal
    diff_between = avg_second - avg_first

    # Get image shape for band boundaries
    height, width = avg_normal.shape
    D = create_distance_from_center(height, width)
    D_shifted = np.fft.fftshift(D)

    max_dist = 0.5
    band_width = max_dist / n_bands
    band_names = ["very_low", "low", "mid", "high", "very_high"][:n_bands]
    band_colors = ['red', 'orange', 'yellow', 'green', 'cyan'][:n_bands]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    def add_band_boundaries(ax):
        center_y, center_x = height // 2, width // 2
        for i in range(n_bands):
            boundary_freq = (i + 1) * band_width
            radius_y = boundary_freq * height
            radius_x = boundary_freq * width
            theta = np.linspace(0, 2*np.pi, 100)
            x_ellipse = center_x + radius_x * np.cos(theta)
            y_ellipse = center_y + radius_y * np.sin(theta)
            ax.plot(x_ellipse, y_ellipse, color=band_colors[i],
                   linestyle='--', linewidth=1, alpha=0.7)

    # Row 1: Average spectra
    im0 = axes[0, 0].imshow(avg_normal, cmap='hot', aspect='auto')
    add_band_boundaries(axes[0, 0])
    axes[0, 0].set_title(f'Normal Average Spectrum\n(n={len(normal_images)})')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(avg_first, cmap='hot', aspect='auto')
    add_band_boundaries(axes[0, 1])
    axes[0, 1].set_title(f'Fault First 500 Average Spectrum\n(n={len(anomaly_first_half)})')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(avg_second, cmap='hot', aspect='auto')
    add_band_boundaries(axes[0, 2])
    axes[0, 2].set_title(f'Fault Last 500 Average Spectrum\n(n={len(anomaly_second_half)})')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # Row 2: Differences
    vmax1 = max(abs(diff_first.min()), abs(diff_first.max()))
    im3 = axes[1, 0].imshow(diff_first, cmap='RdBu_r', aspect='auto',
                            vmin=-vmax1, vmax=vmax1)
    add_band_boundaries(axes[1, 0])
    axes[1, 0].set_title('Difference: Fault 1st - Normal')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    vmax2 = max(abs(diff_second.min()), abs(diff_second.max()))
    im4 = axes[1, 1].imshow(diff_second, cmap='RdBu_r', aspect='auto',
                            vmin=-vmax2, vmax=vmax2)
    add_band_boundaries(axes[1, 1])
    axes[1, 1].set_title('Difference: Fault 2nd - Normal')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    vmax3 = max(abs(diff_between.min()), abs(diff_between.max()))
    im5 = axes[1, 2].imshow(diff_between, cmap='RdBu_r', aspect='auto',
                            vmin=-vmax3, vmax=vmax3)
    add_band_boundaries(axes[1, 2])
    axes[1, 2].set_title('Difference: Fault 2nd - Fault 1st')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

    # Add band legend
    legend_elements = [plt.Line2D([0], [0], color=c, linestyle='--', label=n)
                       for n, c in zip(band_names, band_colors)]
    fig.legend(handles=legend_elements, loc='upper center', ncol=5,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_radial_profile(
    normal_images: list[np.ndarray],
    anomaly_first_half: list[np.ndarray],
    anomaly_second_half: list[np.ndarray],
    n_bands: int = 5,
    save_path: Path | None = None
) -> plt.Figure:
    """Visualize radial profile of frequency energy with band markers.

    Args:
        normal_images: Normal images
        anomaly_first_half: First half anomaly
        anomaly_second_half: Second half anomaly
        n_bands: Number of bands
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    def compute_radial_profile(images):
        spectra = []
        for img in images:
            if img.ndim == 3:
                img = img[0]
            f = np.fft.fft2(img)
            f_shift = np.fft.fftshift(f)
            spectra.append(np.abs(f_shift))

        avg_magnitude = np.mean(spectra, axis=0)
        height, width = avg_magnitude.shape

        D = create_distance_from_center(height, width)
        D_shifted = np.fft.fftshift(D)

        # Compute radial average
        max_r = 50  # number of radial bins
        radial_profile = np.zeros(max_r)
        radial_counts = np.zeros(max_r)

        for i in range(height):
            for j in range(width):
                r_idx = int(D_shifted[i, j] * (max_r * 2))  # Scale to bins
                if r_idx < max_r:
                    radial_profile[r_idx] += avg_magnitude[i, j]
                    radial_counts[r_idx] += 1

        radial_counts[radial_counts == 0] = 1
        radial_profile /= radial_counts

        return radial_profile

    normal_profile = compute_radial_profile(normal_images)
    first_profile = compute_radial_profile(anomaly_first_half)
    second_profile = compute_radial_profile(anomaly_second_half)

    max_r = len(normal_profile)
    freq_axis = np.arange(max_r) / (max_r * 2)  # Normalized frequency [0, 0.5]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Band boundaries
    max_dist = 0.5
    band_width = max_dist / n_bands
    band_names = ["very_low", "low", "mid", "high", "very_high"][:n_bands]
    band_colors = ['red', 'orange', 'yellow', 'green', 'cyan'][:n_bands]

    # Plot 1: Absolute profiles
    ax1 = axes[0]
    ax1.plot(freq_axis, normal_profile, 'b-', label='Normal', linewidth=2)
    ax1.plot(freq_axis, first_profile, 'orange', label='Fault 1st 500', linewidth=2, linestyle='--')
    ax1.plot(freq_axis, second_profile, 'purple', label='Fault 2nd 500', linewidth=2, linestyle=':')

    # Add band boundaries
    for i in range(n_bands):
        boundary = (i + 1) * band_width
        if boundary <= freq_axis.max():
            ax1.axvline(x=boundary, color=band_colors[i], linestyle='--', alpha=0.5)
            ax1.text(boundary, ax1.get_ylim()[1] * 0.95, band_names[i],
                    rotation=90, va='top', ha='right', fontsize=8, color=band_colors[i])

    ax1.set_xlabel('Normalized Frequency')
    ax1.set_ylabel('Average Magnitude')
    ax1.set_title('Radial Frequency Profile')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 0.5])

    # Plot 2: Log scale
    ax2 = axes[1]
    ax2.semilogy(freq_axis, normal_profile + 1, 'b-', label='Normal', linewidth=2)
    ax2.semilogy(freq_axis, first_profile + 1, 'orange', label='Fault 1st 500', linewidth=2, linestyle='--')
    ax2.semilogy(freq_axis, second_profile + 1, 'purple', label='Fault 2nd 500', linewidth=2, linestyle=':')

    for i in range(n_bands):
        boundary = (i + 1) * band_width
        if boundary <= freq_axis.max():
            ax2.axvline(x=boundary, color=band_colors[i], linestyle='--', alpha=0.5)

    ax2.set_xlabel('Normalized Frequency')
    ax2.set_ylabel('Average Magnitude (log scale)')
    ax2.set_title('Radial Frequency Profile (Log Scale)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xlim([0, 0.5])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def _get_image_paths(data_dir: Path, category: str, split: str) -> list[Path]:
    """Get sorted list of image paths from directory.

    Args:
        data_dir: Path to domain directory
        category: Image category ("good" or "fault")
        split: Data split ("train" or "test")

    Returns:
        Sorted list of image paths
    """
    image_dir = data_dir / split / category
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    image_files = sorted(list(image_dir.glob("*.tiff")) + list(image_dir.glob("*.png")))
    return image_files


def run_phase1_1_analysis(
    data_dir: Path,
    output_dir: Path,
    n_bands: int = 5,
    parallel: bool = True,
    n_workers: int = DEFAULT_NUM_WORKERS,
) -> dict:
    """Run complete Phase 1-1 analysis for a single domain.

    Args:
        data_dir: Path to domain data directory
        output_dir: Output directory
        n_bands: Number of frequency bands
        parallel: Use parallel processing for FFT calculations
        n_workers: Number of parallel workers

    Returns:
        Dictionary with all results
    """
    print(f"\n{'='*60}")
    print(f"PHASE 1-1: {data_dir.name}")
    print(f"{'='*60}")
    if parallel:
        print(f"Parallel processing: ON (workers={n_workers})")
    else:
        print("Parallel processing: OFF")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get image paths (fast operation, no loading)
    print("\n[1/5] Scanning image files...")
    normal_train_paths = _get_image_paths(data_dir, "good", "train")
    normal_test_paths = _get_image_paths(data_dir, "good", "test")
    fault_all_paths = _get_image_paths(data_dir, "fault", "test")

    # Split fault data
    n_fault = len(fault_all_paths)
    split_idx = n_fault // 2
    fault_first_paths = fault_all_paths[:split_idx]
    fault_second_paths = fault_all_paths[split_idx:]

    # Combine all normal paths
    normal_all_paths = normal_train_paths + normal_test_paths

    print(f"  Normal (train): {len(normal_train_paths)}")
    print(f"  Normal (test): {len(normal_test_paths)}")
    print(f"  Normal (all): {len(normal_all_paths)}")
    print(f"  Fault (all): {len(fault_all_paths)}")
    print(f"  Fault (first {split_idx}): {len(fault_first_paths)}")
    print(f"  Fault (last {n_fault - split_idx}): {len(fault_second_paths)}")

    # Compute band energy statistics
    print("\n[2/5] Computing frequency band statistics...")

    if parallel:
        # Use path-based parallel processing (more efficient - no IPC overhead for image data)
        normal_means, normal_stds, band_names = compute_stats_from_paths_parallel(
            normal_all_paths, "Normal", n_bands, n_workers
        )
        fault_all_means, fault_all_stds, _ = compute_stats_from_paths_parallel(
            fault_all_paths, "Fault All", n_bands, n_workers
        )
        fault_first_means, fault_first_stds, _ = compute_stats_from_paths_parallel(
            fault_first_paths, "Fault 1st", n_bands, n_workers
        )
        fault_second_means, fault_second_stds, _ = compute_stats_from_paths_parallel(
            fault_second_paths, "Fault 2nd", n_bands, n_workers
        )
    else:
        def compute_stats_from_paths(image_paths, desc):
            band_energies = []
            for img_path in tqdm(image_paths, desc=desc):
                img = Image.open(img_path)
                img_array = np.array(img, dtype=np.float32)
                if img_array.ndim == 3:
                    img_array = img_array[0]
                band_energies.append(get_frequency_band_energy(img_array, n_bands))

            band_names = list(band_energies[0].keys())
            means = {b: float(np.mean([e[b] for e in band_energies])) for b in band_names}
            stds = {b: float(np.std([e[b] for e in band_energies])) for b in band_names}
            return means, stds, band_names

        normal_means, normal_stds, band_names = compute_stats_from_paths(normal_all_paths, "Normal")
        fault_all_means, fault_all_stds, _ = compute_stats_from_paths(fault_all_paths, "Fault All")
        fault_first_means, fault_first_stds, _ = compute_stats_from_paths(fault_first_paths, "Fault 1st")
        fault_second_means, fault_second_stds, _ = compute_stats_from_paths(fault_second_paths, "Fault 2nd")

    # Load sample images for visualization (only a few needed)
    print("\n[3/5] Loading sample images for visualization...")
    normal_all = []
    fault_first_half = []
    fault_second_half = []

    # Load a subset for visualization (not all images)
    n_viz_samples = min(100, len(normal_all_paths))
    for path in tqdm(normal_all_paths[:n_viz_samples], desc="Loading normal samples"):
        img = Image.open(path)
        normal_all.append(np.array(img, dtype=np.float32))

    for path in tqdm(fault_first_paths[:min(50, len(fault_first_paths))], desc="Loading fault 1st samples"):
        img = Image.open(path)
        fault_first_half.append(np.array(img, dtype=np.float32))

    for path in tqdm(fault_second_paths[:min(50, len(fault_second_paths))], desc="Loading fault 2nd samples"):
        img = Image.open(path)
        fault_second_half.append(np.array(img, dtype=np.float32))

    # Print summary
    print("\n" + "-"*60)
    print("FREQUENCY BAND ENERGY SUMMARY")
    print("-"*60)

    print(f"\n{'Band':<12} {'Normal':>12} {'Fault All':>12} {'Fault 1st':>12} {'Fault 2nd':>12}")
    print("-"*60)
    for band in band_names:
        print(f"{band:<12} {normal_means[band]:>12.4f} {fault_all_means[band]:>12.4f} "
              f"{fault_first_means[band]:>12.4f} {fault_second_means[band]:>12.4f}")

    print(f"\n{'Band':<12} {'All-Normal':>12} {'1st-Normal':>12} {'2nd-Normal':>12} {'2nd-1st':>12}")
    print("-"*60)
    for band in band_names:
        diff_all = fault_all_means[band] - normal_means[band]
        diff_first = fault_first_means[band] - normal_means[band]
        diff_second = fault_second_means[band] - normal_means[band]
        diff_between = fault_second_means[band] - fault_first_means[band]
        print(f"{band:<12} {diff_all:>+12.4f} {diff_first:>+12.4f} "
              f"{diff_second:>+12.4f} {diff_between:>+12.4f}")

    # Combine fault samples for visualization
    fault_all = fault_first_half + fault_second_half

    # Generate visualizations
    print("\n[4/5] Generating frequency band visualization on FFT...")
    sample_normal = normal_all[0]
    visualize_frequency_bands_on_fft(
        sample_normal, n_bands,
        save_path=output_dir / "frequency_bands_on_fft.png"
    )
    plt.close()

    print("[5/5] Generating band comparison and spectrum charts...")
    visualize_band_comparison_detailed(
        normal_all, fault_all, fault_first_half, fault_second_half, n_bands,
        save_path=output_dir / "band_comparison_detailed.png"
    )
    plt.close()

    visualize_average_spectrum_with_bands(
        normal_all, fault_first_half, fault_second_half, n_bands,
        save_path=output_dir / "average_spectrum_with_bands.png"
    )
    plt.close()

    visualize_radial_profile(
        normal_all, fault_first_half, fault_second_half, n_bands,
        save_path=output_dir / "radial_frequency_profile.png"
    )
    plt.close()

    # Save results to JSON
    results = {
        "domain": data_dir.name,
        "n_normal_train": len(normal_train_paths),
        "n_normal_test": len(normal_test_paths),
        "n_normal_all": len(normal_all_paths),
        "n_fault_all": len(fault_all_paths),
        "n_fault_first_half": len(fault_first_paths),
        "n_fault_second_half": len(fault_second_paths),
        "band_names": band_names,
        "normal": {"means": normal_means, "stds": normal_stds},
        "fault_all": {"means": fault_all_means, "stds": fault_all_stds},
        "fault_first_half": {"means": fault_first_means, "stds": fault_first_stds},
        "fault_second_half": {"means": fault_second_means, "stds": fault_second_stds},
        "differences": {
            "fault_all_minus_normal": {b: fault_all_means[b] - normal_means[b] for b in band_names},
            "fault_first_minus_normal": {b: fault_first_means[b] - normal_means[b] for b in band_names},
            "fault_second_minus_normal": {b: fault_second_means[b] - normal_means[b] for b in band_names},
            "fault_second_minus_first": {b: fault_second_means[b] - fault_first_means[b] for b in band_names},
        }
    }

    with open(output_dir / "phase1_1_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1-1 EDA: Normal vs Anomaly Frequency Analysis"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="domain_A",
        choices=DOMAINS,
        help="Domain to analyze"
    )
    parser.add_argument(
        "--all_domains",
        action="store_true",
        help="Run analysis on all domains"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Path to HDMAP data root"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--n_bands",
        type=int,
        default=5,
        help="Number of frequency bands"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Use parallel processing (default: True)"
    )
    parser.add_argument(
        "--no-parallel",
        dest="parallel",
        action="store_false",
        help="Disable parallel processing"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_NUM_WORKERS})"
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Error: Data root not found: {data_root}")
        sys.exit(1)

    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        output_root = Path(__file__).parent / "results" / "phase1_1"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    domains = DOMAINS if args.all_domains else [args.domain]

    print("="*60)
    print("HDMAP Phase 1-1: Normal vs Anomaly Frequency Analysis")
    print("="*60)
    print(f"Data root: {data_root}")
    print(f"Domains: {domains}")
    print(f"Output: {output_root}")
    print(f"Parallel: {args.parallel} (workers={args.n_workers})")
    print("="*60)

    all_results = {}

    for domain in domains:
        domain_data_dir = data_root / domain
        domain_output_dir = output_root / f"{timestamp}_{domain}"

        if not domain_data_dir.exists():
            print(f"Warning: {domain_data_dir} not found, skipping...")
            continue

        results = run_phase1_1_analysis(
            domain_data_dir,
            domain_output_dir,
            n_bands=args.n_bands,
            parallel=args.parallel,
            n_workers=args.n_workers,
        )
        all_results[domain] = results

    # Save combined results
    if len(all_results) > 1:
        combined_path = output_root / f"{timestamp}_all_domains_summary.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nCombined results saved to: {combined_path}")

    print("\n" + "="*60)
    print("Phase 1-1 Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
