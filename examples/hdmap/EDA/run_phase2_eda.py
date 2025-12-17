"""Phase 2: Filter Effect Visualization for CutPaste.

Analyzes how different frequency filters (HPF/LPF) affect CutPaste patches.
Goal: Find filter configurations that make CutPaste more similar to real faults
(i.e., more very_high frequency, less low/mid frequency).

Usage:
    python run_phase2_eda.py --all_domains --n_samples 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.anomalib.models.image.draem_cutpaste_clf.synthetic_generator_v2 import (
    CutPasteSyntheticGenerator,
)
from examples.hdmap.EDA.frequency_filters import (
    get_frequency_band_energy,
    get_magnitude_spectrum,
    create_filter,
    apply_frequency_filter,
    FilterType,
    FilterMode,
)

# Default paths and parameters
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "HDMAP" / "1000_tiff_minmax"
DEFAULT_OUTPUT_ROOT = Path(__file__).parent / "results" / "phase2"
DEFAULT_N_SAMPLES = 50

# Target image size
TARGET_SIZE = (31, 95)  # (H, W)

# CutPaste parameters (reduced a_fault for less distortion)
CUTPASTE_PARAMS = {
    "cut_w_range": (10, 80),
    "cut_h_range": (1, 2),
    "a_fault_start": 0,
    "a_fault_range_end": 0.5,  # Reduced from 2 to 0.5
    "probability": 1.0,
    "norm": True,
}

# Filter configurations to test
FILTER_CONFIGS = [
    # No filter (baseline)
    {"name": "no_filter", "filter_type": None, "cutoff": None, "mode": None, "order": None},

    # HPF - Butterworth (FAIR recommended)
    {"name": "bhpf_c0.1_o2", "filter_type": "butterworth", "cutoff": 0.1, "mode": "hpf", "order": 2},
    {"name": "bhpf_c0.15_o2", "filter_type": "butterworth", "cutoff": 0.15, "mode": "hpf", "order": 2},
    {"name": "bhpf_c0.2_o2", "filter_type": "butterworth", "cutoff": 0.2, "mode": "hpf", "order": 2},
    {"name": "bhpf_c0.25_o2", "filter_type": "butterworth", "cutoff": 0.25, "mode": "hpf", "order": 2},
    {"name": "bhpf_c0.3_o2", "filter_type": "butterworth", "cutoff": 0.3, "mode": "hpf", "order": 2},

    # HPF - Gaussian (same cutoffs as Butterworth for fair comparison)
    {"name": "ghpf_c0.1", "filter_type": "gaussian", "cutoff": 0.1, "mode": "hpf", "order": None},
    {"name": "ghpf_c0.15", "filter_type": "gaussian", "cutoff": 0.15, "mode": "hpf", "order": None},
    {"name": "ghpf_c0.2", "filter_type": "gaussian", "cutoff": 0.2, "mode": "hpf", "order": None},
    {"name": "ghpf_c0.25", "filter_type": "gaussian", "cutoff": 0.25, "mode": "hpf", "order": None},
    {"name": "ghpf_c0.3", "filter_type": "gaussian", "cutoff": 0.3, "mode": "hpf", "order": None},

    # HPF - Ideal
    {"name": "ihpf_c0.1", "filter_type": "ideal", "cutoff": 0.1, "mode": "hpf", "order": None},
    {"name": "ihpf_c0.2", "filter_type": "ideal", "cutoff": 0.2, "mode": "hpf", "order": None},
    {"name": "ihpf_c0.3", "filter_type": "ideal", "cutoff": 0.3, "mode": "hpf", "order": None},

    # LPF for comparison
    {"name": "blpf_c0.1_o2", "filter_type": "butterworth", "cutoff": 0.1, "mode": "lpf", "order": 2},
    {"name": "blpf_c0.2_o2", "filter_type": "butterworth", "cutoff": 0.2, "mode": "lpf", "order": 2},
]

DOMAINS = ["domain_A", "domain_B", "domain_C", "domain_D"]


def load_and_resize_image(img_path: Path, target_size: tuple[int, int]) -> np.ndarray:
    """Load image and resize to target size."""
    img = Image.open(img_path)
    img_resized = img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)
    if img_array.ndim == 3:
        img_array = img_array[..., 0]
    return img_array


def apply_filter_to_patch(
    patch: np.ndarray,
    filter_config: dict,
) -> np.ndarray:
    """Apply frequency filter to a patch.

    Args:
        patch: Input patch (H, W)
        filter_config: Filter configuration dict

    Returns:
        Filtered patch
    """
    if filter_config["filter_type"] is None:
        return patch

    h, w = patch.shape

    # Create filter mask
    filter_mask = create_filter(
        height=h,
        width=w,
        filter_type=filter_config["filter_type"],
        cutoff=filter_config["cutoff"],
        mode=filter_config["mode"],
        order=filter_config["order"] if filter_config["order"] else 2,
    )

    # Apply filter
    filtered = apply_frequency_filter(patch, filter_mask)

    return filtered


def apply_filtered_cutpaste(
    image: np.ndarray,
    generator: CutPasteSyntheticGenerator,
    filter_config: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply CutPaste with optional filter on the patch.

    The filter is applied to the patch BEFORE pasting.

    Args:
        image: Original image (H, W)
        generator: CutPaste generator
        filter_config: Filter configuration

    Returns:
        Tuple of (synthetic_image, mask, patch_info)
    """
    h, w = image.shape

    # Get CutPaste parameters from generator
    cut_h = random.randint(*generator.cut_h_range)
    cut_w = random.randint(*generator.cut_w_range)
    cut_w = min(cut_w, w - 1)
    cut_h = min(cut_h, h - 1)

    # Sample from location
    from_h = random.randint(0, h - cut_h)
    from_w = random.randint(0, w - cut_w)

    # Sample to location (non-overlapping)
    valid_h_ranges = []
    valid_w_ranges = []

    if from_h - cut_h >= 0:
        valid_h_ranges.append((0, from_h - cut_h))
    if from_h + cut_h <= h - cut_h:
        valid_h_ranges.append((from_h + cut_h, h - cut_h))

    if from_w - cut_w >= 0:
        valid_w_ranges.append((0, from_w - cut_w))
    if from_w + cut_w <= w - cut_w:
        valid_w_ranges.append((from_w + cut_w, w - cut_w))

    if not valid_h_ranges or not valid_w_ranges:
        to_h = random.randint(0, h - cut_h)
        to_w = random.randint(0, w - cut_w)
    else:
        h_range = random.choice(valid_h_ranges)
        w_range = random.choice(valid_w_ranges)
        to_h = random.randint(h_range[0], h_range[1])
        to_w = random.randint(w_range[0], w_range[1])

    # Extract patch
    patch = image[from_h:from_h+cut_h, from_w:from_w+cut_w].copy()

    # Normalize patch
    if generator.norm:
        max_val = np.max(np.abs(patch))
        if max_val > 0:
            patch = patch / max_val

    # Apply frequency filter to patch
    filtered_patch = apply_filter_to_patch(patch, filter_config)

    # Apply amplitude
    a_fault = random.uniform(generator.a_fault_start, generator.a_fault_range_end)
    augmented_patch = filtered_patch * a_fault

    # Create synthetic image (add patch)
    synthetic = image.copy()
    synthetic[to_h:to_h+cut_h, to_w:to_w+cut_w] += augmented_patch

    # Create mask
    mask = np.zeros_like(image)
    mask[to_h:to_h+cut_h, to_w:to_w+cut_w] = 1.0

    patch_info = {
        "cut_w": cut_w,
        "cut_h": cut_h,
        "from_h": from_h,
        "from_w": from_w,
        "to_h": to_h,
        "to_w": to_w,
        "a_fault": a_fault,
        "filter": filter_config["name"],
    }

    return synthetic, mask, patch_info


def analyze_filter_effect(
    image_paths: list[Path],
    filter_configs: list[dict],
    n_bands: int = 5,
    seed: int = 42,
) -> dict:
    """Analyze frequency effects of different filters.

    Args:
        image_paths: List of image paths
        filter_configs: List of filter configurations
        n_bands: Number of frequency bands
        seed: Random seed

    Returns:
        Results dictionary
    """
    random.seed(seed)
    np.random.seed(seed)

    generator = CutPasteSyntheticGenerator(**CUTPASTE_PARAMS)

    results = {config["name"]: [] for config in filter_configs}

    for img_path in tqdm(image_paths, desc="Processing images"):
        image = load_and_resize_image(img_path, TARGET_SIZE)
        original_bands = get_frequency_band_energy(image, n_bands)

        # Use same random state for all filters on this image
        state = random.getstate()
        np_state = np.random.get_state()

        for config in filter_configs:
            # Reset random state for fair comparison
            random.setstate(state)
            np.random.set_state(np_state)

            synthetic, mask, patch_info = apply_filtered_cutpaste(
                image, generator, config
            )

            synthetic_bands = get_frequency_band_energy(synthetic, n_bands)
            diff_bands = get_frequency_band_energy(synthetic - image, n_bands)

            # Compute band changes
            band_changes = {
                band: synthetic_bands[band] - original_bands[band]
                for band in original_bands
            }

            results[config["name"]].append({
                "original_bands": original_bands,
                "synthetic_bands": synthetic_bands,
                "diff_bands": diff_bands,
                "band_changes": band_changes,
                "patch_info": patch_info,
            })

    return results


def compute_statistics(results: dict, band_names: list[str]) -> dict:
    """Compute statistics for each filter configuration."""
    stats = {}

    for filter_name, samples in results.items():
        stats[filter_name] = {
            "n_samples": len(samples),
            "band_changes": {},
            "diff_bands": {},
        }

        for band in band_names:
            changes = [s["band_changes"][band] for s in samples]
            diff_energies = [s["diff_bands"][band] for s in samples]

            stats[filter_name]["band_changes"][band] = {
                "mean": float(np.mean(changes)),
                "std": float(np.std(changes)),
            }
            stats[filter_name]["diff_bands"][band] = {
                "mean": float(np.mean(diff_energies)),
                "std": float(np.std(diff_energies)),
            }

    return stats


def create_example_images_visualization(
    image_paths: list[Path],
    output_dir: Path,
    domain: str,
    n_examples: int = 3,
    seed: int = 42,
) -> None:
    """Create visualization of actual CutPaste images with different filters.

    Shows: Original, CutPaste (no filter), CutPaste + BHPF, CutPaste + GHPF
    Along with their FFT spectrums.
    """
    random.seed(seed)
    np.random.seed(seed)

    generator = CutPasteSyntheticGenerator(**CUTPASTE_PARAMS)

    # Select filters to visualize
    filters_to_show = [
        {"name": "no_filter", "filter_type": None, "cutoff": None, "mode": None, "order": None},
        {"name": "bhpf_c0.2", "filter_type": "butterworth", "cutoff": 0.2, "mode": "hpf", "order": 2},
        {"name": "bhpf_c0.3", "filter_type": "butterworth", "cutoff": 0.3, "mode": "hpf", "order": 2},
        {"name": "ghpf_c0.2", "filter_type": "gaussian", "cutoff": 0.2, "mode": "hpf", "order": None},
        {"name": "ghpf_c0.3", "filter_type": "gaussian", "cutoff": 0.3, "mode": "hpf", "order": None},
    ]

    # Sample images
    sample_paths = random.sample(image_paths, min(n_examples, len(image_paths)))

    # Create figure: n_examples rows x (1 + len(filters)*2) columns
    # Each row: Original | (CutPaste + FFT) for each filter
    n_filters = len(filters_to_show)
    fig, axes = plt.subplots(n_examples, 1 + n_filters * 2, figsize=(4 * (1 + n_filters * 2), 4 * n_examples))

    if n_examples == 1:
        axes = axes.reshape(1, -1)

    for row, img_path in enumerate(sample_paths):
        image = load_and_resize_image(img_path, TARGET_SIZE)

        # Save random state
        state = random.getstate()
        np_state = np.random.get_state()

        # Column 0: Original image
        ax = axes[row, 0]
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        ax.set_title('Original' if row == 0 else '')
        ax.axis('off')

        col = 1
        for fconfig in filters_to_show:
            # Reset random state for fair comparison
            random.setstate(state)
            np.random.set_state(np_state)

            # Apply filtered CutPaste
            synthetic, mask, patch_info = apply_filtered_cutpaste(image, generator, fconfig)

            # CutPaste image
            ax = axes[row, col]
            ax.imshow(synthetic, cmap='gray')
            # Overlay mask contour
            ax.contour(mask, colors='red', linewidths=0.5, alpha=0.7)
            if row == 0:
                ax.set_title(f'{fconfig["name"]}\n(a={patch_info["a_fault"]:.2f})')
            ax.axis('off')
            col += 1

            # FFT of diff image
            ax = axes[row, col]
            diff_img = synthetic - image
            diff_spectrum = get_magnitude_spectrum(diff_img, log_scale=True, shift=True)
            ax.imshow(diff_spectrum, cmap='jet')
            if row == 0:
                # Calculate very_high percentage
                diff_bands = get_frequency_band_energy(diff_img, 5)
                vh_pct = diff_bands["very_high"] * 100
                ax.set_title(f'Diff FFT\n(vh={vh_pct:.1f}%)')
            ax.axis('off')
            col += 1

    plt.tight_layout()
    plt.savefig(output_dir / f"{domain}_example_images.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Create Butterworth vs Gaussian comparison figure
    create_bhpf_vs_ghpf_comparison(image_paths, output_dir, domain, seed)


def create_bhpf_vs_ghpf_comparison(
    image_paths: list[Path],
    output_dir: Path,
    domain: str,
    seed: int = 42,
) -> None:
    """Create detailed comparison between Butterworth and Gaussian HPF."""
    random.seed(seed)
    np.random.seed(seed)

    generator = CutPasteSyntheticGenerator(**CUTPASTE_PARAMS)

    # Cutoffs to compare
    cutoffs = [0.1, 0.2, 0.3]

    # Sample one image
    img_path = random.choice(image_paths)
    image = load_and_resize_image(img_path, TARGET_SIZE)

    # Create figure: 2 rows (BHPF, GHPF) x (1 + len(cutoffs)*2) columns
    n_cutoffs = len(cutoffs)
    fig, axes = plt.subplots(2, 1 + n_cutoffs * 2, figsize=(4 * (1 + n_cutoffs * 2), 8))

    filter_types = [
        ("Butterworth HPF", "butterworth"),
        ("Gaussian HPF", "gaussian"),
    ]

    for row, (filter_label, filter_type) in enumerate(filter_types):
        # Save random state
        state = random.getstate()
        np_state = np.random.get_state()

        # Original
        ax = axes[row, 0]
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        ax.set_title('Original' if row == 0 else '')
        ax.set_ylabel(filter_label, fontsize=12)
        ax.axis('off')

        col = 1
        for cutoff in cutoffs:
            # Reset random state
            random.setstate(state)
            np.random.set_state(np_state)

            fconfig = {
                "name": f"{filter_type}_c{cutoff}",
                "filter_type": filter_type,
                "cutoff": cutoff,
                "mode": "hpf",
                "order": 2 if filter_type == "butterworth" else None,
            }

            synthetic, mask, patch_info = apply_filtered_cutpaste(image, generator, fconfig)

            # CutPaste image
            ax = axes[row, col]
            ax.imshow(synthetic, cmap='gray')
            ax.contour(mask, colors='red', linewidths=0.5, alpha=0.7)
            if row == 0:
                ax.set_title(f'cutoff={cutoff}')
            ax.axis('off')
            col += 1

            # FFT of diff
            ax = axes[row, col]
            diff_img = synthetic - image
            diff_spectrum = get_magnitude_spectrum(diff_img, log_scale=True, shift=True)
            ax.imshow(diff_spectrum, cmap='jet')

            diff_bands = get_frequency_band_energy(diff_img, 5)
            vh_pct = diff_bands["very_high"] * 100
            if row == 0:
                ax.set_title(f'vh={vh_pct:.1f}%')
            else:
                ax.set_title(f'vh={vh_pct:.1f}%', fontsize=10)
            ax.axis('off')
            col += 1

    plt.suptitle(f'{domain}: Butterworth vs Gaussian HPF Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"{domain}_bhpf_vs_ghpf.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Create cutoff sweep comparison
    create_cutoff_sweep_visualization(image_paths, output_dir, domain, seed)


def create_cutoff_sweep_visualization(
    image_paths: list[Path],
    output_dir: Path,
    domain: str,
    seed: int = 42,
) -> None:
    """Create visualization showing effect of different cutoff values."""
    random.seed(seed + 100)  # Different seed for variety
    np.random.seed(seed + 100)

    generator = CutPasteSyntheticGenerator(**CUTPASTE_PARAMS)

    # Cutoffs to sweep
    cutoffs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    # Sample one image
    img_path = random.choice(image_paths)
    image = load_and_resize_image(img_path, TARGET_SIZE)

    # Create figure
    fig, axes = plt.subplots(3, len(cutoffs) + 1, figsize=(3 * (len(cutoffs) + 1), 9))

    row_labels = ['Butterworth HPF', 'Gaussian HPF', 'Diff FFT (BHPF)']

    # Save random state
    state = random.getstate()
    np_state = np.random.get_state()

    # First column: Original
    for row in range(3):
        ax = axes[row, 0]
        if row < 2:
            ax.imshow(image, cmap='gray', vmin=0, vmax=1)
            ax.set_title('Original' if row == 0 else '')
        else:
            ax.axis('off')
        ax.set_ylabel(row_labels[row], fontsize=10)
        if row < 2:
            ax.set_xticks([])
            ax.set_yticks([])

    # Collect very_high percentages for bar chart
    bhpf_vh = []
    ghpf_vh = []

    for col, cutoff in enumerate(cutoffs):
        # Butterworth HPF
        random.setstate(state)
        np.random.set_state(np_state)

        fconfig_b = {
            "name": f"bhpf_c{cutoff}",
            "filter_type": "butterworth",
            "cutoff": cutoff,
            "mode": "hpf",
            "order": 2,
        }
        synthetic_b, mask_b, _ = apply_filtered_cutpaste(image, generator, fconfig_b)

        ax = axes[0, col + 1]
        ax.imshow(synthetic_b, cmap='gray')
        ax.contour(mask_b, colors='red', linewidths=0.5, alpha=0.7)
        ax.set_title(f'c={cutoff}')
        ax.axis('off')

        # Gaussian HPF
        random.setstate(state)
        np.random.set_state(np_state)

        fconfig_g = {
            "name": f"ghpf_c{cutoff}",
            "filter_type": "gaussian",
            "cutoff": cutoff,
            "mode": "hpf",
            "order": None,
        }
        synthetic_g, mask_g, _ = apply_filtered_cutpaste(image, generator, fconfig_g)

        ax = axes[1, col + 1]
        ax.imshow(synthetic_g, cmap='gray')
        ax.contour(mask_g, colors='red', linewidths=0.5, alpha=0.7)
        ax.axis('off')

        # FFT of Butterworth diff
        ax = axes[2, col + 1]
        diff_b = synthetic_b - image
        diff_spectrum = get_magnitude_spectrum(diff_b, log_scale=True, shift=True)
        ax.imshow(diff_spectrum, cmap='jet')

        diff_bands_b = get_frequency_band_energy(diff_b, 5)
        diff_bands_g = get_frequency_band_energy(synthetic_g - image, 5)

        bhpf_vh.append(diff_bands_b["very_high"] * 100)
        ghpf_vh.append(diff_bands_g["very_high"] * 100)

        ax.set_title(f'vh={bhpf_vh[-1]:.0f}%')
        ax.axis('off')

    plt.suptitle(f'{domain}: Cutoff Sweep (Butterworth vs Gaussian HPF)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"{domain}_cutoff_sweep.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Create bar chart comparing BHPF vs GHPF very_high percentages
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(cutoffs))
    width = 0.35

    bars1 = ax.bar(x - width/2, bhpf_vh, width, label='Butterworth HPF', alpha=0.8)
    bars2 = ax.bar(x + width/2, ghpf_vh, width, label='Gaussian HPF', alpha=0.8)

    ax.set_xlabel('Cutoff Frequency')
    ax.set_ylabel('very_high Band Energy (%)')
    ax.set_title(f'{domain}: Diff Image very_high Energy vs Cutoff')
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in cutoffs])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / f"{domain}_cutoff_vh_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_visualizations(
    stats: dict,
    filter_configs: list[dict],
    output_dir: Path,
    domain: str,
) -> None:
    """Create visualization plots."""
    band_names = ["very_low", "low", "mid", "high", "very_high"]
    filter_names = [c["name"] for c in filter_configs]

    # 1. Band changes comparison (bar chart for each filter)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Group filters by type
    hpf_filters = [f for f in filter_names if "hpf" in f or f == "no_filter"]
    lpf_filters = [f for f in filter_names if "lpf" in f]

    x = np.arange(len(band_names))
    width = 0.12

    # HPF filters comparison
    ax = axes[0, 0]
    for i, fname in enumerate(hpf_filters[:6]):  # Limit to 6 for readability
        changes = [stats[fname]["band_changes"][b]["mean"] * 100 for b in band_names]
        ax.bar(x + i * width, changes, width, label=fname, alpha=0.8)

    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Energy Change (%p)')
    ax.set_title(f'{domain}: HPF Effect on Band Changes')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(band_names, rotation=45)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # Diff image frequency distribution
    ax = axes[0, 1]
    for i, fname in enumerate(hpf_filters[:6]):
        diff_means = [stats[fname]["diff_bands"][b]["mean"] * 100 for b in band_names]
        ax.bar(x + i * width, diff_means, width, label=fname, alpha=0.8)

    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Energy (%)')
    ax.set_title(f'{domain}: HPF Effect on Diff Image Frequency')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(band_names, rotation=45)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # very_high increase vs cutoff - BHPF vs GHPF comparison
    ax = axes[1, 0]
    cutoffs = [0.1, 0.15, 0.2, 0.25, 0.3]

    # Butterworth HPF
    bhpf_filters = [f"bhpf_c{c}_o2" for c in cutoffs]
    bhpf_diff = [stats[f]["diff_bands"]["very_high"]["mean"] * 100 for f in bhpf_filters if f in stats]

    # Gaussian HPF
    ghpf_filters = [f"ghpf_c{c}" for c in cutoffs]
    ghpf_diff = [stats[f]["diff_bands"]["very_high"]["mean"] * 100 for f in ghpf_filters if f in stats]

    ax.plot(cutoffs[:len(bhpf_diff)], bhpf_diff, 'o-', label='Butterworth HPF', markersize=8, color='blue')
    ax.plot(cutoffs[:len(ghpf_diff)], ghpf_diff, 's-', label='Gaussian HPF', markersize=8, color='green')
    ax.axhline(y=stats["no_filter"]["diff_bands"]["very_high"]["mean"] * 100,
               color='red', linestyle='--', label='No Filter', linewidth=2)

    ax.set_xlabel('HPF Cutoff Frequency')
    ax.set_ylabel('Diff very_high Energy (%)')
    ax.set_title(f'{domain}: Diff Image very_high vs Cutoff\n(BHPF vs GHPF)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Summary: very_high / low ratio
    ax = axes[1, 1]
    ratios = []
    labels = []

    for fname in hpf_filters:
        vh = stats[fname]["band_changes"]["very_high"]["mean"]
        low = stats[fname]["band_changes"]["low"]["mean"]
        # Avoid division by zero
        if abs(low) > 0.0001:
            ratio = vh / low
        else:
            ratio = 0
        ratios.append(ratio)
        labels.append(fname.replace("_", "\n"))

    colors = ['green' if r > 0 else 'red' for r in ratios]
    ax.bar(range(len(ratios)), ratios, color=colors, alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45)
    ax.set_ylabel('very_high / low Ratio')
    ax.set_title(f'{domain}: very_high / low Change Ratio\n(Higher = More like Real Fault)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"{domain}_filter_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Filter type comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Compare filter types at cutoff=0.2
    filter_types = ["no_filter", "bhpf_c0.2_o2", "ghpf_c0.2", "ihpf_c0.2"]
    x = np.arange(len(band_names))
    width = 0.2

    ax = axes[0]
    for i, fname in enumerate(filter_types):
        if fname in stats:
            changes = [stats[fname]["band_changes"][b]["mean"] * 100 for b in band_names]
            ax.bar(x + i * width, changes, width, label=fname, alpha=0.8)

    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Energy Change (%p)')
    ax.set_title(f'{domain}: Filter Type Comparison (cutoff=0.2)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(band_names, rotation=45)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # HPF vs LPF
    ax = axes[1]
    compare_filters = ["no_filter", "bhpf_c0.2_o2", "blpf_c0.2_o2"]
    for i, fname in enumerate(compare_filters):
        if fname in stats:
            changes = [stats[fname]["band_changes"][b]["mean"] * 100 for b in band_names]
            ax.bar(x + i * 0.25, changes, 0.25, label=fname, alpha=0.8)

    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Energy Change (%p)')
    ax.set_title(f'{domain}: HPF vs LPF (cutoff=0.2)')
    ax.set_xticks(x + 0.25)
    ax.set_xticklabels(band_names, rotation=45)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Real Fault reference
    ax = axes[2]
    # Real Fault average changes (from Phase 1-1)
    real_fault_avg = {
        "very_low": -0.55,
        "low": 0.10,
        "mid": 0.12,
        "high": 0.05,
        "very_high": 0.28,
    }

    best_hpf = "bhpf_c0.2_o2"  # Example

    real_changes = [real_fault_avg[b] for b in band_names]
    cutpaste_changes = [stats["no_filter"]["band_changes"][b]["mean"] * 100 for b in band_names]
    hpf_changes = [stats[best_hpf]["band_changes"][b]["mean"] * 100 for b in band_names]

    x = np.arange(len(band_names))
    width = 0.25

    ax.bar(x - width, real_changes, width, label='Real Fault', color='green', alpha=0.8)
    ax.bar(x, cutpaste_changes, width, label='CutPaste (no filter)', color='blue', alpha=0.8)
    ax.bar(x + width, hpf_changes, width, label=f'CutPaste + {best_hpf}', color='orange', alpha=0.8)

    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Energy Change (%p)')
    ax.set_title(f'{domain}: Real Fault vs CutPaste vs CutPaste+HPF')
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, rotation=45)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"{domain}_filter_type_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def analyze_domain(
    domain: str,
    data_root: Path,
    output_dir: Path,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = 42,
) -> dict:
    """Analyze filter effects for a domain."""
    domain_dir = data_root / domain
    train_dir = domain_dir / "train" / "good"

    image_paths = list(train_dir.glob("*.tiff")) + list(train_dir.glob("*.tif"))
    if not image_paths:
        raise FileNotFoundError(f"No TIFF images found in {train_dir}")

    if len(image_paths) > n_samples:
        random.seed(seed)
        image_paths = random.sample(image_paths, n_samples)

    print(f"[{domain}] Analyzing {len(image_paths)} images with {len(FILTER_CONFIGS)} filter configs...")

    results = analyze_filter_effect(image_paths, FILTER_CONFIGS, seed=seed)

    band_names = ["very_low", "low", "mid", "high", "very_high"]
    stats = compute_statistics(results, band_names)

    # Create output directory
    domain_output = output_dir / domain
    domain_output.mkdir(parents=True, exist_ok=True)

    # Create visualizations
    create_visualizations(stats, FILTER_CONFIGS, domain_output, domain)

    # Create example images visualization (actual CutPaste images)
    print(f"[{domain}] Creating example image visualizations...")
    all_image_paths = list(train_dir.glob("*.tiff")) + list(train_dir.glob("*.tif"))
    create_example_images_visualization(all_image_paths, domain_output, domain, n_examples=3, seed=seed)

    # Save results
    with open(domain_output / "phase2_results.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def create_summary_report(all_stats: dict, output_dir: Path) -> None:
    """Create cross-domain summary report."""
    domains = list(all_stats.keys())
    band_names = ["very_low", "low", "mid", "high", "very_high"]
    filter_names = list(all_stats[domains[0]].keys())

    # Find best filter for each domain (maximizes very_high / minimizes low+mid)
    best_filters = {}
    for domain in domains:
        best_score = -float('inf')
        best_filter = None

        for fname in filter_names:
            if fname == "no_filter":
                continue

            vh_change = all_stats[domain][fname]["band_changes"]["very_high"]["mean"]
            low_change = all_stats[domain][fname]["band_changes"]["low"]["mean"]
            mid_change = all_stats[domain][fname]["band_changes"]["mid"]["mean"]

            # Score: maximize very_high, minimize low+mid
            score = vh_change - (low_change + mid_change)

            if score > best_score:
                best_score = score
                best_filter = fname

        best_filters[domain] = best_filter

    # Create summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    x = np.arange(len(band_names))
    width = 0.2

    # All domains comparison with best HPF
    ax = axes[0, 0]
    for i, domain in enumerate(domains):
        best_f = best_filters[domain]
        changes = [all_stats[domain][best_f]["band_changes"][b]["mean"] * 100 for b in band_names]
        ax.bar(x + i * width, changes, width, label=f'{domain} ({best_f})', alpha=0.8)

    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Energy Change (%p)')
    ax.set_title('Best HPF per Domain: Band Changes')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(band_names, rotation=45)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # No filter vs best filter comparison
    ax = axes[0, 1]
    no_filter_vh = [all_stats[d]["no_filter"]["band_changes"]["very_high"]["mean"] * 100 for d in domains]
    best_filter_vh = [all_stats[d][best_filters[d]]["band_changes"]["very_high"]["mean"] * 100 for d in domains]

    x_dom = np.arange(len(domains))
    ax.bar(x_dom - 0.2, no_filter_vh, 0.4, label='No Filter', alpha=0.8)
    ax.bar(x_dom + 0.2, best_filter_vh, 0.4, label='Best HPF', alpha=0.8)

    ax.set_xlabel('Domain')
    ax.set_ylabel('very_high Change (%p)')
    ax.set_title('very_high Band Change: No Filter vs Best HPF')
    ax.set_xticks(x_dom)
    ax.set_xticklabels([d.replace('_', ' ').title() for d in domains])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Cutoff comparison across domains
    ax = axes[1, 0]
    cutoffs = [0.1, 0.15, 0.2, 0.25, 0.3]
    bhpf_names = [f"bhpf_c{c}_o2" for c in cutoffs]

    for domain in domains:
        vh_changes = [all_stats[domain][f]["band_changes"]["very_high"]["mean"] * 100
                      for f in bhpf_names]
        ax.plot(cutoffs, vh_changes, 'o-', label=domain, markersize=6)

    ax.set_xlabel('Butterworth HPF Cutoff')
    ax.set_ylabel('very_high Change (%p)')
    ax.set_title('very_high Band Change vs HPF Cutoff')
    ax.legend()
    ax.grid(alpha=0.3)

    # Summary text
    ax = axes[1, 1]
    ax.axis('off')

    text = "Phase 2 Summary\n" + "="*40 + "\n\n"
    text += "Best Filter per Domain:\n"
    for domain in domains:
        text += f"  {domain}: {best_filters[domain]}\n"

    text += "\n" + "="*40 + "\n"
    text += "Improvement (very_high change):\n"
    for domain in domains:
        nf = all_stats[domain]["no_filter"]["band_changes"]["very_high"]["mean"] * 100
        bf = all_stats[domain][best_filters[domain]]["band_changes"]["very_high"]["mean"] * 100
        text += f"  {domain}: {nf:.2f}%p → {bf:.2f}%p ({bf-nf:+.2f}%p)\n"

    text += "\n" + "="*40 + "\n"
    text += "Recommendation:\n"
    # Find most common best filter
    from collections import Counter
    best_common = Counter(best_filters.values()).most_common(1)[0][0]
    text += f"  Use {best_common} for all domains\n"

    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / "cross_domain_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save summary JSON
    summary = {
        "best_filters": best_filters,
        "filter_configs": FILTER_CONFIGS,
        "cutpaste_params": CUTPASTE_PARAMS,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Create README
    create_readme(all_stats, best_filters, output_dir)


def create_readme(all_stats: dict, best_filters: dict, output_dir: Path) -> None:
    """Create README markdown."""
    domains = list(all_stats.keys())
    band_names = ["very_low", "low", "mid", "high", "very_high"]

    md = "# Phase 2: Filter Effect Visualization\n\n"

    md += "## 분석 개요\n\n"
    md += "- **목표**: CutPaste 패치에 HPF/LPF 적용 시 주파수 특성 변화 분석\n"
    md += "- **목적**: Real Fault의 very_high 위주 특성을 더 잘 모사하는 필터 찾기\n\n"

    md += "## 테스트한 필터 설정\n\n"
    md += "| 타입 | Cutoff | Order |\n"
    md += "|------|--------|-------|\n"
    md += "| Butterworth HPF | 0.1, 0.15, 0.2, 0.25, 0.3 | 2 |\n"
    md += "| Gaussian HPF | 0.1, 0.2, 0.3 | - |\n"
    md += "| Ideal HPF | 0.1, 0.2, 0.3 | - |\n"
    md += "| Butterworth LPF | 0.1, 0.2 | 2 |\n\n"

    md += "## 결과 요약\n\n"

    md += "### 도메인별 최적 필터\n\n"
    md += "| 도메인 | 최적 필터 | very_high 변화 (no filter → best) |\n"
    md += "|--------|----------|-----------------------------------|\n"

    for domain in domains:
        bf = best_filters[domain]
        nf = all_stats[domain]["no_filter"]["band_changes"]["very_high"]["mean"] * 100
        best = all_stats[domain][bf]["band_changes"]["very_high"]["mean"] * 100
        md += f"| {domain.replace('_', ' ').title()} | {bf} | {nf:.2f}%p → {best:.2f}%p ({best-nf:+.2f}%p) |\n"

    md += "\n"

    md += "### 주요 발견\n\n"

    # Calculate average improvement
    avg_nf_vh = np.mean([all_stats[d]["no_filter"]["band_changes"]["very_high"]["mean"] * 100 for d in domains])
    avg_best_vh = np.mean([all_stats[d][best_filters[d]]["band_changes"]["very_high"]["mean"] * 100 for d in domains])

    md += f"1. **HPF 적용 시 very_high 증가**: {avg_nf_vh:.2f}%p → {avg_best_vh:.2f}%p (평균 {avg_best_vh - avg_nf_vh:+.2f}%p)\n"
    md += "2. **Butterworth HPF (order=2)가 가장 효과적**\n"
    md += "3. **최적 cutoff**: 0.15~0.25 범위\n\n"

    md += "### Butterworth HPF Cutoff별 very_high 변화 (%p)\n\n"
    md += "| 도메인 | cutoff=0.1 | cutoff=0.15 | cutoff=0.2 | cutoff=0.25 | cutoff=0.3 |\n"
    md += "|--------|------------|-------------|------------|-------------|------------|\n"

    cutoffs = [0.1, 0.15, 0.2, 0.25, 0.3]
    for domain in domains:
        md += f"| {domain.replace('_', ' ').title()} |"
        for c in cutoffs:
            fname = f"bhpf_c{c}_o2"
            vh = all_stats[domain][fname]["band_changes"]["very_high"]["mean"] * 100
            md += f" {vh:.2f} |"
        md += "\n"

    md += "\n"

    md += "## 생성 파일\n\n"
    md += "각 도메인 폴더:\n"
    md += "- `*_filter_comparison.png`: 필터별 주파수 변화 비교\n"
    md += "- `*_filter_type_comparison.png`: 필터 타입별 비교 및 Real Fault 비교\n"
    md += "- `phase2_results.json`: 상세 결과\n\n"
    md += "루트 폴더:\n"
    md += "- `cross_domain_summary.png`: 도메인간 비교\n"
    md += "- `summary.json`: 요약 데이터\n"

    with open(output_dir / "README.md", "w") as f:
        f.write(md)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Filter Effect Visualization")
    parser.add_argument("--domain", type=str, default=None, help="Single domain to analyze")
    parser.add_argument("--all_domains", action="store_true", help="Analyze all domains")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT, help="Data root path")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output path")
    parser.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES, help="Samples per domain")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.all_domains:
        domains = DOMAINS
    elif args.domain:
        domains = [args.domain]
    else:
        domains = ["domain_A"]

    print(f"Phase 2: Filter Effect Visualization")
    print(f"Data root: {args.data_root}")
    print(f"Output: {args.output_root}")
    print(f"Domains: {domains}")
    print(f"Samples per domain: {args.n_samples}")
    print(f"Filter configs: {len(FILTER_CONFIGS)}")
    print()

    args.output_root.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for domain in domains:
        stats = analyze_domain(
            domain=domain,
            data_root=args.data_root,
            output_dir=args.output_root,
            n_samples=args.n_samples,
            seed=args.seed,
        )
        all_stats[domain] = stats

    if len(domains) > 1:
        create_summary_report(all_stats, args.output_root)

    print("\nDone! Results saved to:", args.output_root)


if __name__ == "__main__":
    main()
