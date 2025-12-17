"""Phase 3: Full Image Filtering - Normal vs CutPaste Separability Analysis.

Analyzes whether applying HPF to the full image increases the separability
between Normal and CutPaste (Synthetic Fault) images in frequency domain.

This tests "Option 2" from EDA_frequency_cutpaste.md:
    Scaled CutPaste → Frequency Filter → Model

Usage:
    python run_phase3_eda.py --all_domains --n_samples 50
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
    apply_frequency_filter,
    FilterType,
    FilterMode,
)

# Default paths and parameters
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "HDMAP" / "1000_tiff_minmax"
DEFAULT_OUTPUT_ROOT = Path(__file__).parent / "results" / "phase3"
DEFAULT_N_SAMPLES = 50

# Target image size
TARGET_SIZE = (31, 95)  # (H, W)

# CutPaste parameters
CUTPASTE_PARAMS = {
    "cut_w_range": (10, 80),
    "cut_h_range": (1, 2),
    "a_fault_start": 0,
    "a_fault_range_end": 0.5,
    "probability": 1.0,
    "norm": True,
}

# Filter configurations for full image filtering
FILTER_CONFIGS = [
    {"name": "no_filter", "filter_type": None, "cutoff": None, "mode": None},
    {"name": "ghpf_c0.1", "filter_type": "gaussian", "cutoff": 0.1, "mode": "hpf"},
    {"name": "ghpf_c0.15", "filter_type": "gaussian", "cutoff": 0.15, "mode": "hpf"},
    {"name": "ghpf_c0.2", "filter_type": "gaussian", "cutoff": 0.2, "mode": "hpf"},
    {"name": "ghpf_c0.25", "filter_type": "gaussian", "cutoff": 0.25, "mode": "hpf"},
    {"name": "ghpf_c0.3", "filter_type": "gaussian", "cutoff": 0.3, "mode": "hpf"},
]

DOMAINS = ["domain_A", "domain_B", "domain_C", "domain_D"]
BANDS = ["very_low", "low", "mid", "high", "very_high"]


def load_and_resize_image(img_path: Path, target_size: tuple[int, int]) -> np.ndarray:
    """Load image and resize to target size."""
    img = Image.open(img_path)
    img_resized = img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)
    if img_array.ndim == 3:
        img_array = img_array[..., 0]
    return img_array


def apply_full_image_filter(
    image: np.ndarray,
    filter_config: dict,
) -> np.ndarray:
    """Apply frequency filter to the full image."""
    if filter_config["filter_type"] is None:
        return image

    filter_type = filter_config["filter_type"]
    filter_mode = filter_config["mode"]
    cutoff = filter_config["cutoff"]

    # Create filter mask
    h, w = image.shape[:2]
    from examples.hdmap.EDA.frequency_filters import create_filter
    filter_mask = create_filter(h, w, filter_type, cutoff, filter_mode, order=2)

    # Apply filter
    filtered = apply_frequency_filter(image, filter_mask)
    return filtered


def compute_separability(
    normal_bands: dict[str, list[float]],
    fault_bands: dict[str, list[float]],
) -> dict[str, float]:
    """Compute separability metrics between Normal and Fault distributions.

    Separability = |mean_normal - mean_fault| / (std_normal + std_fault + eps)
    Higher is better (more separable).
    """
    separability = {}
    eps = 1e-8

    for band in BANDS:
        normal_vals = np.array(normal_bands[band])
        fault_vals = np.array(fault_bands[band])

        mean_diff = abs(np.mean(normal_vals) - np.mean(fault_vals))
        std_sum = np.std(normal_vals) + np.std(fault_vals) + eps

        separability[band] = mean_diff / std_sum

    # Overall separability (average across bands)
    separability["overall"] = np.mean([separability[b] for b in BANDS])

    return separability


def analyze_domain(
    data_root: Path,
    output_dir: Path,
    domain: str,
    n_samples: int,
    seed: int = 42,
) -> dict:
    """Analyze Normal vs CutPaste separability for a domain."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Find normal images
    normal_dir = data_root / domain / "train" / "good"
    image_paths = sorted(normal_dir.glob("*.tiff"))[:n_samples]

    if not image_paths:
        print(f"  [WARN] No images found in {normal_dir}")
        return {}

    # Initialize CutPaste generator
    generator = CutPasteSyntheticGenerator(
        cut_w_range=CUTPASTE_PARAMS["cut_w_range"],
        cut_h_range=CUTPASTE_PARAMS["cut_h_range"],
        a_fault_start=CUTPASTE_PARAMS["a_fault_start"],
        a_fault_range_end=CUTPASTE_PARAMS["a_fault_range_end"],
        probability=CUTPASTE_PARAMS["probability"],
        norm=CUTPASTE_PARAMS["norm"],
    )

    results = {config["name"]: {} for config in FILTER_CONFIGS}

    for fconfig in FILTER_CONFIGS:
        fname = fconfig["name"]

        # Collect frequency bands for Normal and Fault
        normal_bands = {band: [] for band in BANDS}
        fault_bands = {band: [] for band in BANDS}

        for img_path in tqdm(image_paths, desc=f"  {fname}", leave=False):
            # Load and resize image
            image = load_and_resize_image(img_path, TARGET_SIZE)

            # Create CutPaste version
            img_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
            fault_tensor, mask, _ = generator(img_tensor)  # Returns (image, mask, severity)
            fault_image = fault_tensor.squeeze().numpy()

            # Apply full image filter to BOTH Normal and Fault
            normal_filtered = apply_full_image_filter(image, fconfig)
            fault_filtered = apply_full_image_filter(fault_image, fconfig)

            # Get frequency band energies
            normal_energy = get_frequency_band_energy(normal_filtered)
            fault_energy = get_frequency_band_energy(fault_filtered)

            for band in BANDS:
                normal_bands[band].append(normal_energy[band])
                fault_bands[band].append(fault_energy[band])

        # Compute statistics
        normal_stats = {
            band: {"mean": float(np.mean(normal_bands[band])), "std": float(np.std(normal_bands[band]))}
            for band in BANDS
        }
        fault_stats = {
            band: {"mean": float(np.mean(fault_bands[band])), "std": float(np.std(fault_bands[band]))}
            for band in BANDS
        }

        # Compute separability
        separability = compute_separability(normal_bands, fault_bands)

        results[fname] = {
            "normal": normal_stats,
            "fault": fault_stats,
            "separability": {k: float(v) for k, v in separability.items()},
            "n_samples": len(image_paths),
        }

    return results


def create_separability_comparison(
    results: dict,
    output_path: Path,
    domain: str,
) -> None:
    """Create visualization comparing separability across filter configs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    filter_names = [c["name"] for c in FILTER_CONFIGS]

    # Plot 1: Overall separability by filter
    ax = axes[0]
    overall_sep = [results[f]["separability"]["overall"] for f in filter_names]
    colors = ['gray' if f == 'no_filter' else 'steelblue' for f in filter_names]
    bars = ax.bar(range(len(filter_names)), overall_sep, color=colors)
    ax.set_xticks(range(len(filter_names)))
    ax.set_xticklabels([f.replace("ghpf_", "").replace("no_filter", "None") for f in filter_names], rotation=45)
    ax.set_ylabel("Separability (higher = better)")
    ax.set_title(f"{domain}: Overall Separability\n(Normal vs CutPaste)")
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, overall_sep):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Band-wise separability heatmap
    ax = axes[1]
    sep_matrix = np.array([
        [results[f]["separability"][b] for b in BANDS]
        for f in filter_names
    ])
    im = ax.imshow(sep_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels(BANDS, rotation=45)
    ax.set_yticks(range(len(filter_names)))
    ax.set_yticklabels([f.replace("ghpf_", "").replace("no_filter", "None") for f in filter_names])
    ax.set_title("Band-wise Separability")
    plt.colorbar(im, ax=ax, label="Separability")

    # Add text annotations
    for i in range(len(filter_names)):
        for j in range(len(BANDS)):
            ax.text(j, i, f'{sep_matrix[i, j]:.2f}', ha='center', va='center', fontsize=8)

    # Plot 3: very_high band comparison (Normal vs Fault mean)
    ax = axes[2]
    x = np.arange(len(filter_names))
    width = 0.35

    normal_vh = [results[f]["normal"]["very_high"]["mean"] * 100 for f in filter_names]
    fault_vh = [results[f]["fault"]["very_high"]["mean"] * 100 for f in filter_names]

    bars1 = ax.bar(x - width/2, normal_vh, width, label='Normal', color='forestgreen', alpha=0.7)
    bars2 = ax.bar(x + width/2, fault_vh, width, label='CutPaste', color='crimson', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("ghpf_", "").replace("no_filter", "None") for f in filter_names], rotation=45)
    ax.set_ylabel("very_high Energy (%)")
    ax.set_title("very_high Band: Normal vs CutPaste")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f"Phase 3: Full Image HPF - Separability Analysis ({domain})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_image_visualization(
    data_root: Path,
    output_path: Path,
    domain: str,
    n_examples: int = 2,
    seed: int = 42,
) -> None:
    """Create visualization of Normal vs CutPaste with full image filtering."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Find normal images
    normal_dir = data_root / domain / "train" / "good"
    image_paths = sorted(normal_dir.glob("*.tiff"))[:n_examples]

    if not image_paths:
        return

    # Initialize generator
    generator = CutPasteSyntheticGenerator(
        cut_w_range=CUTPASTE_PARAMS["cut_w_range"],
        cut_h_range=CUTPASTE_PARAMS["cut_h_range"],
        a_fault_start=CUTPASTE_PARAMS["a_fault_start"],
        a_fault_range_end=CUTPASTE_PARAMS["a_fault_range_end"],
        probability=CUTPASTE_PARAMS["probability"],
        norm=CUTPASTE_PARAMS["norm"],
    )

    # Select filters to visualize
    vis_filters = [
        {"name": "no_filter", "filter_type": None, "cutoff": None, "mode": None},
        {"name": "ghpf_c0.15", "filter_type": "gaussian", "cutoff": 0.15, "mode": "hpf"},
        {"name": "ghpf_c0.25", "filter_type": "gaussian", "cutoff": 0.25, "mode": "hpf"},
    ]

    n_filters = len(vis_filters)
    fig, axes = plt.subplots(n_examples, n_filters * 4, figsize=(n_filters * 8, n_examples * 2.5))

    if n_examples == 1:
        axes = axes.reshape(1, -1)

    for row, img_path in enumerate(image_paths):
        image = load_and_resize_image(img_path, TARGET_SIZE)

        # Create CutPaste
        img_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        fault_tensor, mask, _ = generator(img_tensor)  # Returns (image, mask, severity)
        fault_image = fault_tensor.squeeze().numpy()
        mask_np = mask.squeeze().numpy()

        col = 0
        for fconfig in vis_filters:
            fname = fconfig["name"]

            # Apply filter to both
            normal_filtered = apply_full_image_filter(image, fconfig)
            fault_filtered = apply_full_image_filter(fault_image, fconfig)

            # Get FFT spectrums
            normal_fft = get_magnitude_spectrum(normal_filtered)
            fault_fft = get_magnitude_spectrum(fault_filtered)

            # Plot Normal filtered
            ax = axes[row, col]
            ax.imshow(normal_filtered, cmap='gray', aspect='auto')
            if row == 0:
                ax.set_title(f"Normal\n({fname})", fontsize=9)
            ax.axis('off')
            col += 1

            # Plot Normal FFT
            ax = axes[row, col]
            ax.imshow(np.log1p(np.fft.fftshift(normal_fft)), cmap='jet', aspect='auto')
            if row == 0:
                ax.set_title("Normal FFT", fontsize=9)
            ax.axis('off')
            col += 1

            # Plot Fault filtered
            ax = axes[row, col]
            ax.imshow(fault_filtered, cmap='gray', aspect='auto')
            ax.contour(mask_np, colors='red', linewidths=0.5, alpha=0.7)
            if row == 0:
                ax.set_title(f"CutPaste\n({fname})", fontsize=9)
            ax.axis('off')
            col += 1

            # Plot Fault FFT
            ax = axes[row, col]
            ax.imshow(np.log1p(np.fft.fftshift(fault_fft)), cmap='jet', aspect='auto')
            if row == 0:
                ax.set_title("CutPaste FFT", fontsize=9)
            ax.axis('off')
            col += 1

    plt.suptitle(f"{domain}: Full Image HPF Applied to Normal vs CutPaste", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_cross_domain_summary(
    all_results: dict,
    output_dir: Path,
) -> None:
    """Create cross-domain summary visualization."""
    domains = list(all_results.keys())
    filter_names = [c["name"] for c in FILTER_CONFIGS]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Overall separability by domain and filter
    ax = axes[0]
    x = np.arange(len(filter_names))
    width = 0.2

    for i, domain in enumerate(domains):
        overall_sep = [all_results[domain][f]["separability"]["overall"] for f in filter_names]
        ax.bar(x + i * width, overall_sep, width, label=domain, alpha=0.8)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f.replace("ghpf_", "").replace("no_filter", "None") for f in filter_names], rotation=45)
    ax.set_ylabel("Overall Separability")
    ax.set_title("Separability by Domain and Filter")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Separability improvement over no_filter
    ax = axes[1]

    for i, domain in enumerate(domains):
        baseline = all_results[domain]["no_filter"]["separability"]["overall"]
        improvements = [
            (all_results[domain][f]["separability"]["overall"] - baseline) / baseline * 100
            for f in filter_names[1:]  # Skip no_filter
        ]
        ax.plot(range(len(improvements)), improvements, 'o-', label=domain, linewidth=2, markersize=8)

    ax.set_xticks(range(len(filter_names) - 1))
    ax.set_xticklabels([f.replace("ghpf_", "") for f in filter_names[1:]], rotation=45)
    ax.set_ylabel("Improvement over No Filter (%)")
    ax.set_title("Separability Improvement by HPF")
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle("Phase 3: Cross-Domain Summary - Full Image HPF Effect on Separability", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "cross_domain_summary.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Full Image HPF Separability Analysis")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--domains", nargs="+", default=["domain_A"])
    parser.add_argument("--all_domains", action="store_true")
    parser.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.all_domains:
        args.domains = DOMAINS

    args.output_root.mkdir(parents=True, exist_ok=True)

    print("Phase 3: Full Image HPF - Normal vs CutPaste Separability")
    print(f"Data root: {args.data_root}")
    print(f"Output: {args.output_root}")
    print(f"Domains: {args.domains}")
    print(f"Samples per domain: {args.n_samples}")
    print(f"Filter configs: {len(FILTER_CONFIGS)}")
    print()

    all_results = {}

    for domain in args.domains:
        print(f"[{domain}] Analyzing separability...")

        domain_output = args.output_root / domain
        domain_output.mkdir(parents=True, exist_ok=True)

        # Run analysis
        results = analyze_domain(
            args.data_root,
            domain_output,
            domain,
            args.n_samples,
            args.seed,
        )

        if not results:
            continue

        all_results[domain] = results

        # Save results
        with open(domain_output / "phase3_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Create visualizations
        print(f"[{domain}] Creating visualizations...")
        create_separability_comparison(results, domain_output / f"{domain}_separability.png", domain)
        create_image_visualization(args.data_root, domain_output / f"{domain}_examples.png", domain)

    # Cross-domain summary
    if len(all_results) > 1:
        print("Creating cross-domain summary...")
        create_cross_domain_summary(all_results, args.output_root)

        # Save summary
        summary = {
            domain: {
                fname: {
                    "separability_overall": results[fname]["separability"]["overall"],
                    "separability_very_high": results[fname]["separability"]["very_high"],
                }
                for fname in results
            }
            for domain, results in all_results.items()
        }
        with open(args.output_root / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\nDone! Results saved to: {args.output_root}")


if __name__ == "__main__":
    main()
