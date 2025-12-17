"""Frequency Analysis Functions for Phase 1 EDA.

This module implements the core analysis functions for:
- Phase 1-1: Normal vs Anomaly frequency spectrum comparison
- Phase 1-2: CutPaste augmentation frequency analysis

Based on FAIR paper hypothesis:
- Normal reconstruction error → biased to high-frequency
- Anomaly reconstruction error → distributed across all frequencies
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    from .frequency_filters import (
        FilterMode,
        FilterType,
        FrequencyFilter,
        apply_frequency_filter,
        create_filter,
        get_frequency_band_energy,
        get_magnitude_spectrum,
    )
except ImportError:
    from frequency_filters import (
        FilterMode,
        FilterType,
        FrequencyFilter,
        apply_frequency_filter,
        create_filter,
        get_frequency_band_energy,
        get_magnitude_spectrum,
    )


@dataclass
class FrequencyAnalysisResult:
    """Results from frequency analysis."""
    # Image info
    n_normal: int
    n_anomaly: int
    image_shape: tuple[int, int]

    # Band energy statistics
    normal_band_means: dict[str, float]
    normal_band_stds: dict[str, float]
    anomaly_band_means: dict[str, float]
    anomaly_band_stds: dict[str, float]

    # Spectrum statistics
    normal_spectrum_mean: np.ndarray
    normal_spectrum_std: np.ndarray
    anomaly_spectrum_mean: np.ndarray
    anomaly_spectrum_std: np.ndarray

    # Difference analysis
    band_differences: dict[str, float]  # anomaly - normal
    spectrum_difference: np.ndarray


@dataclass
class CutPasteAnalysisResult:
    """Results from CutPaste frequency analysis."""
    # Original vs CutPaste comparison
    original_band_energy: dict[str, float]
    cutpaste_band_energy: dict[str, float]
    band_energy_change: dict[str, float]

    # Spectrum changes
    original_spectrum: np.ndarray
    cutpaste_spectrum: np.ndarray
    spectrum_change: np.ndarray

    # Patch characteristics
    patch_info: dict


def load_hdmap_images(
    data_dir: Path | str,
    category: str = "good",
    split: str = "train",
    max_samples: int | None = None,
    seed: int = 42
) -> list[np.ndarray]:
    """Load HDMAP images from specified directory.

    Args:
        data_dir: Path to domain directory (e.g., datasets/HDMAP/100000_tiff_minmax/domain_A)
        category: Image category ("good" or "fault")
        split: Data split ("train" or "test")
        max_samples: Maximum number of samples to load (None for all)
        seed: Random seed for sampling

    Returns:
        List of numpy arrays (each image as float32)
    """
    data_dir = Path(data_dir)
    image_dir = data_dir / split / category

    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    # Get all image files
    image_files = sorted(list(image_dir.glob("*.tiff")) + list(image_dir.glob("*.png")))

    if not image_files:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    # Sample if needed
    if max_samples is not None and max_samples < len(image_files):
        random.seed(seed)
        image_files = random.sample(image_files, max_samples)

    # Load images
    images = []
    for img_path in tqdm(image_files, desc=f"Loading {category} images"):
        img = Image.open(img_path)
        img_array = np.array(img, dtype=np.float32)
        images.append(img_array)

    return images


def analyze_normal_vs_anomaly(
    normal_images: Sequence[np.ndarray],
    anomaly_images: Sequence[np.ndarray],
    n_bands: int = 5
) -> FrequencyAnalysisResult:
    """Analyze frequency characteristics: Normal vs Anomaly.

    This implements Phase 1-1 of the EDA plan:
    - Compare frequency spectra between normal and anomaly images
    - Compute band energy distributions
    - Test FAIR hypothesis: anomaly has energy across all frequency bands

    Args:
        normal_images: List of normal images
        anomaly_images: List of anomaly images
        n_bands: Number of frequency bands for energy analysis

    Returns:
        FrequencyAnalysisResult with all statistics
    """
    # Get image shape from first image
    sample_img = normal_images[0]
    if sample_img.ndim == 3:
        sample_img = sample_img[0]
    image_shape = sample_img.shape

    # Collect band energies
    normal_band_energies = []
    anomaly_band_energies = []

    # Collect spectra
    normal_spectra = []
    anomaly_spectra = []

    print("Analyzing normal images...")
    for img in tqdm(normal_images, desc="Normal"):
        if img.ndim == 3:
            img = img[0]
        normal_band_energies.append(get_frequency_band_energy(img, n_bands))
        normal_spectra.append(get_magnitude_spectrum(img, log_scale=True))

    print("Analyzing anomaly images...")
    for img in tqdm(anomaly_images, desc="Anomaly"):
        if img.ndim == 3:
            img = img[0]
        anomaly_band_energies.append(get_frequency_band_energy(img, n_bands))
        anomaly_spectra.append(get_magnitude_spectrum(img, log_scale=True))

    # Compute statistics
    band_names = list(normal_band_energies[0].keys())

    normal_band_means = {
        band: float(np.mean([e[band] for e in normal_band_energies]))
        for band in band_names
    }
    normal_band_stds = {
        band: float(np.std([e[band] for e in normal_band_energies]))
        for band in band_names
    }

    anomaly_band_means = {
        band: float(np.mean([e[band] for e in anomaly_band_energies]))
        for band in band_names
    }
    anomaly_band_stds = {
        band: float(np.std([e[band] for e in anomaly_band_energies]))
        for band in band_names
    }

    band_differences = {
        band: anomaly_band_means[band] - normal_band_means[band]
        for band in band_names
    }

    # Spectrum statistics
    normal_spectrum_mean = np.mean(normal_spectra, axis=0)
    normal_spectrum_std = np.std(normal_spectra, axis=0)
    anomaly_spectrum_mean = np.mean(anomaly_spectra, axis=0)
    anomaly_spectrum_std = np.std(anomaly_spectra, axis=0)
    spectrum_difference = anomaly_spectrum_mean - normal_spectrum_mean

    return FrequencyAnalysisResult(
        n_normal=len(normal_images),
        n_anomaly=len(anomaly_images),
        image_shape=image_shape,
        normal_band_means=normal_band_means,
        normal_band_stds=normal_band_stds,
        anomaly_band_means=anomaly_band_means,
        anomaly_band_stds=anomaly_band_stds,
        normal_spectrum_mean=normal_spectrum_mean,
        normal_spectrum_std=normal_spectrum_std,
        anomaly_spectrum_mean=anomaly_spectrum_mean,
        anomaly_spectrum_std=anomaly_spectrum_std,
        band_differences=band_differences,
        spectrum_difference=spectrum_difference,
    )


def analyze_cutpaste_frequency(
    image: np.ndarray,
    cut_w_range: tuple[int, int] = (10, 80),
    cut_h_range: tuple[int, int] = (1, 2),
    a_fault_start: float = 1.0,
    a_fault_range_end: float = 10.0,
    n_bands: int = 5
) -> CutPasteAnalysisResult:
    """Analyze frequency characteristics of CutPaste augmentation.

    This implements Phase 1-2 of the EDA plan:
    - Apply CutPaste to an image
    - Analyze frequency changes introduced by CutPaste
    - Identify which frequency bands are affected

    Args:
        image: Input image (H, W) or (C, H, W)
        cut_w_range: Range of patch widths
        cut_h_range: Range of patch heights
        a_fault_start: Minimum fault amplitude
        a_fault_range_end: Maximum fault amplitude
        n_bands: Number of frequency bands

    Returns:
        CutPasteAnalysisResult with analysis
    """
    # Import CutPaste generator
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
    from anomalib.models.image.draem_cutpaste_clf.synthetic_generator_v2 import CutPasteSyntheticGenerator

    if image.ndim == 2:
        image = image[np.newaxis, ...]  # Add channel dimension

    # Convert to tensor
    image_tensor = torch.from_numpy(image).unsqueeze(0)  # (1, C, H, W)

    # Create generator
    generator = CutPasteSyntheticGenerator(
        cut_w_range=cut_w_range,
        cut_h_range=cut_h_range,
        a_fault_start=a_fault_start,
        a_fault_range_end=a_fault_range_end,
        probability=1.0,  # Always apply
    )

    # Generate synthetic fault
    synthetic_image, fault_mask, severity, patch_info = generator(
        image_tensor,
        return_patch_info=True
    )

    # Convert back to numpy
    original_np = image[0] if image.ndim == 3 else image
    synthetic_np = synthetic_image.squeeze().numpy()
    if synthetic_np.ndim == 3:
        synthetic_np = synthetic_np[0]

    # Analyze frequency bands
    original_bands = get_frequency_band_energy(original_np, n_bands)
    cutpaste_bands = get_frequency_band_energy(synthetic_np, n_bands)

    band_change = {
        band: cutpaste_bands[band] - original_bands[band]
        for band in original_bands.keys()
    }

    # Analyze spectra
    original_spectrum = get_magnitude_spectrum(original_np, log_scale=True)
    cutpaste_spectrum = get_magnitude_spectrum(synthetic_np, log_scale=True)
    spectrum_change = cutpaste_spectrum - original_spectrum

    return CutPasteAnalysisResult(
        original_band_energy=original_bands,
        cutpaste_band_energy=cutpaste_bands,
        band_energy_change=band_change,
        original_spectrum=original_spectrum,
        cutpaste_spectrum=cutpaste_spectrum,
        spectrum_change=spectrum_change,
        patch_info=patch_info,
    )


def analyze_filter_effect(
    image: np.ndarray,
    filter_type: FilterType | str = FilterType.BUTTERWORTH,
    cutoffs: Sequence[float] = (0.05, 0.1, 0.15, 0.2),
    order: int = 2,
    n_bands: int = 5
) -> dict:
    """Analyze the effect of frequency filters on an image.

    Args:
        image: Input image
        filter_type: Type of filter
        cutoffs: List of cutoff frequencies to test
        order: Butterworth filter order
        n_bands: Number of frequency bands

    Returns:
        Dictionary with analysis results for each cutoff
    """
    if isinstance(filter_type, str):
        filter_type = FilterType(filter_type.lower())

    if image.ndim == 3:
        image = image[0]

    height, width = image.shape

    results = {
        "original": {
            "band_energy": get_frequency_band_energy(image, n_bands),
            "image_stats": {
                "mean": float(image.mean()),
                "std": float(image.std()),
                "min": float(image.min()),
                "max": float(image.max()),
            }
        },
        "hpf": {},
        "lpf": {},
    }

    for cutoff in cutoffs:
        # High-pass filter
        hpf_mask = create_filter(height, width, filter_type, cutoff, FilterMode.HIGH_PASS, order)
        hpf_image = apply_frequency_filter(image, hpf_mask)

        results["hpf"][cutoff] = {
            "band_energy": get_frequency_band_energy(hpf_image, n_bands),
            "image_stats": {
                "mean": float(hpf_image.mean()),
                "std": float(hpf_image.std()),
                "min": float(hpf_image.min()),
                "max": float(hpf_image.max()),
            },
            "energy_preserved": float((hpf_image**2).sum() / (image**2).sum()),
        }

        # Low-pass filter
        lpf_mask = create_filter(height, width, filter_type, cutoff, FilterMode.LOW_PASS, order)
        lpf_image = apply_frequency_filter(image, lpf_mask)

        results["lpf"][cutoff] = {
            "band_energy": get_frequency_band_energy(lpf_image, n_bands),
            "image_stats": {
                "mean": float(lpf_image.mean()),
                "std": float(lpf_image.std()),
                "min": float(lpf_image.min()),
                "max": float(lpf_image.max()),
            },
            "energy_preserved": float((lpf_image**2).sum() / (image**2).sum()),
        }

    return results


def compute_reconstruction_frequency_error(
    original: np.ndarray,
    reconstruction: np.ndarray,
    n_bands: int = 5
) -> dict:
    """Compute frequency-domain reconstruction error.

    This is the key analysis for FAIR hypothesis validation:
    - Normal reconstruction error should be biased to high-frequency
    - Anomaly reconstruction error should be distributed across all frequencies

    Args:
        original: Original image
        reconstruction: Reconstructed image
        n_bands: Number of frequency bands

    Returns:
        Dictionary with frequency error analysis
    """
    if original.ndim == 3:
        original = original[0]
    if reconstruction.ndim == 3:
        reconstruction = reconstruction[0]

    # Spatial domain error
    spatial_error = reconstruction - original

    # Frequency domain error analysis
    error_spectrum = get_magnitude_spectrum(spatial_error, log_scale=False)

    # Band-wise error
    error_band_energy = get_frequency_band_energy(spatial_error, n_bands)

    # Original and reconstruction spectra
    orig_spectrum = get_magnitude_spectrum(original, log_scale=False)
    recon_spectrum = get_magnitude_spectrum(reconstruction, log_scale=False)

    # Frequency-domain error (difference in magnitude spectra)
    freq_error = np.abs(recon_spectrum - orig_spectrum)

    return {
        "spatial_error_stats": {
            "mean": float(spatial_error.mean()),
            "std": float(spatial_error.std()),
            "mse": float((spatial_error**2).mean()),
            "mae": float(np.abs(spatial_error).mean()),
        },
        "error_band_energy": error_band_energy,
        "error_spectrum": error_spectrum,
        "freq_error": freq_error,
        "freq_error_total": float(freq_error.sum()),
    }


def batch_analyze_cutpaste(
    images: Sequence[np.ndarray],
    n_samples: int = 10,
    cut_w_range: tuple[int, int] = (10, 80),
    cut_h_range: tuple[int, int] = (1, 2),
    a_fault_start: float = 1.0,
    a_fault_range_end: float = 10.0,
    n_bands: int = 5,
    seed: int = 42
) -> dict:
    """Batch analyze CutPaste effects on multiple images.

    Args:
        images: List of input images
        n_samples: Number of samples to analyze
        cut_w_range: Patch width range
        cut_h_range: Patch height range
        a_fault_start: Minimum amplitude
        a_fault_range_end: Maximum amplitude
        n_bands: Number of frequency bands
        seed: Random seed

    Returns:
        Dictionary with aggregated statistics
    """
    random.seed(seed)

    if len(images) > n_samples:
        sample_indices = random.sample(range(len(images)), n_samples)
        images = [images[i] for i in sample_indices]

    results = []

    print(f"Analyzing CutPaste on {len(images)} images...")
    for img in tqdm(images, desc="CutPaste analysis"):
        result = analyze_cutpaste_frequency(
            img,
            cut_w_range=cut_w_range,
            cut_h_range=cut_h_range,
            a_fault_start=a_fault_start,
            a_fault_range_end=a_fault_range_end,
            n_bands=n_bands,
        )
        results.append(result)

    # Aggregate statistics
    band_names = list(results[0].original_band_energy.keys())

    aggregated = {
        "n_samples": len(results),
        "original_band_mean": {
            band: float(np.mean([r.original_band_energy[band] for r in results]))
            for band in band_names
        },
        "cutpaste_band_mean": {
            band: float(np.mean([r.cutpaste_band_energy[band] for r in results]))
            for band in band_names
        },
        "band_change_mean": {
            band: float(np.mean([r.band_energy_change[band] for r in results]))
            for band in band_names
        },
        "band_change_std": {
            band: float(np.std([r.band_energy_change[band] for r in results]))
            for band in band_names
        },
        "patch_info_stats": {
            "cut_w_mean": float(np.mean([r.patch_info["cut_w"] for r in results])),
            "cut_h_mean": float(np.mean([r.patch_info["cut_h"] for r in results])),
            "a_fault_mean": float(np.mean([r.patch_info["a_fault"] for r in results])),
            "coverage_mean": float(np.mean([r.patch_info["coverage_percentage"] for r in results])),
        },
    }

    return aggregated


def print_analysis_summary(result: FrequencyAnalysisResult) -> None:
    """Print a summary of frequency analysis results.

    Args:
        result: FrequencyAnalysisResult to summarize
    """
    print("\n" + "="*60)
    print("FREQUENCY ANALYSIS SUMMARY")
    print("="*60)

    print(f"\nDataset Info:")
    print(f"  Normal samples: {result.n_normal}")
    print(f"  Anomaly samples: {result.n_anomaly}")
    print(f"  Image shape: {result.image_shape}")

    print(f"\nBand Energy Distribution (Normal):")
    for band, mean in result.normal_band_means.items():
        std = result.normal_band_stds[band]
        print(f"  {band:12s}: {mean:.4f} ± {std:.4f}")

    print(f"\nBand Energy Distribution (Anomaly):")
    for band, mean in result.anomaly_band_means.items():
        std = result.anomaly_band_stds[band]
        print(f"  {band:12s}: {mean:.4f} ± {std:.4f}")

    print(f"\nBand Energy Difference (Anomaly - Normal):")
    for band, diff in result.band_differences.items():
        sign = "+" if diff > 0 else ""
        print(f"  {band:12s}: {sign}{diff:.4f}")

    # FAIR hypothesis check
    print("\n" + "-"*60)
    print("FAIR Hypothesis Check:")
    print("-"*60)

    low_freq_bands = ["very_low", "low"]
    high_freq_bands = ["high", "very_high"]

    low_diff = sum(result.band_differences.get(b, 0) for b in low_freq_bands)
    high_diff = sum(result.band_differences.get(b, 0) for b in high_freq_bands)

    print(f"  Low frequency difference sum: {low_diff:+.4f}")
    print(f"  High frequency difference sum: {high_diff:+.4f}")

    if low_diff > high_diff:
        print("\n  → Anomaly has MORE energy in LOW frequencies")
        print("    (Consistent with FAIR hypothesis)")
    else:
        print("\n  → Anomaly has MORE energy in HIGH frequencies")
        print("    (Different from FAIR hypothesis)")

    print("="*60 + "\n")
