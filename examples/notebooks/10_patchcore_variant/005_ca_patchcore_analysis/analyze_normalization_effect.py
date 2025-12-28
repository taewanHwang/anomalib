#!/usr/bin/env python3
"""Analysis 2: Preprocessing Normalization Investigation.

Determine if ImageNet normalization erases the intensity difference
between cold/warm conditions.

Key Question: Does (x - mean) / std transformation make cold and warm
images indistinguishable in the normalized space?

Usage:
    python analyze_normalization_effect.py --domain domain_C --n-samples 100

Outputs:
    - Intensity distribution plots (before/after normalization)
    - Separability metrics (mean gap, overlap %, Bhattacharyya distance)
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results" / "normalization"
ANOMALIB_ROOT = SCRIPT_DIR.parent.parent.parent.parent
DATASET_ROOT = ANOMALIB_ROOT / "datasets" / "HDMAP" / "1000_tiff_minmax"

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def load_tiff_image(image_path: Path) -> np.ndarray:
    """Load TIFF image as float32 array.

    Returns:
        Array in range [0, 1] with shape (H, W)
    """
    img = tifffile.imread(str(image_path)).astype(np.float32)
    return img


def apply_imagenet_normalization(image: np.ndarray) -> np.ndarray:
    """Apply ImageNet normalization.

    Input: image in [0, 1] range, shape (H, W) grayscale
    Output: normalized image, shape (H, W)
    """
    # For grayscale, all channels have same value, so use average of means
    mean = IMAGENET_MEAN.mean()  # 0.449
    std = IMAGENET_STD.mean()    # 0.226

    return (image - mean) / std


def compute_intensity_stats(images: List[np.ndarray]) -> Dict:
    """Compute intensity distribution statistics."""
    all_pixels = np.concatenate([img.flatten() for img in images])

    return {
        "mean": float(np.mean(all_pixels)),
        "std": float(np.std(all_pixels)),
        "min": float(np.min(all_pixels)),
        "max": float(np.max(all_pixels)),
        "p10": float(np.percentile(all_pixels, 10)),
        "p50": float(np.percentile(all_pixels, 50)),
        "p90": float(np.percentile(all_pixels, 90)),
        "histogram": np.histogram(all_pixels, bins=100, density=True),
    }


def compute_separability_metrics(
    cold_stats: Dict,
    warm_stats: Dict,
    cold_images: List[np.ndarray],
    warm_images: List[np.ndarray]
) -> Dict:
    """Compute metrics for cold/warm separability."""
    # Mean gap
    mean_gap = warm_stats["mean"] - cold_stats["mean"]

    # Cohen's d effect size
    pooled_std = np.sqrt((cold_stats["std"]**2 + warm_stats["std"]**2) / 2)
    cohens_d = mean_gap / pooled_std if pooled_std > 0 else 0

    # Histogram overlap
    cold_hist, cold_bins = cold_stats["histogram"]
    warm_hist, warm_bins = warm_stats["histogram"]

    # Resample to same bins for overlap calculation
    combined_min = min(cold_stats["min"], warm_stats["min"])
    combined_max = max(cold_stats["max"], warm_stats["max"])
    bins = np.linspace(combined_min, combined_max, 101)

    cold_pixels = np.concatenate([img.flatten() for img in cold_images])
    warm_pixels = np.concatenate([img.flatten() for img in warm_images])

    cold_hist, _ = np.histogram(cold_pixels, bins=bins, density=True)
    warm_hist, _ = np.histogram(warm_pixels, bins=bins, density=True)

    # Overlap coefficient (intersection)
    bin_width = bins[1] - bins[0]
    overlap = np.sum(np.minimum(cold_hist, warm_hist)) * bin_width

    # Bhattacharyya coefficient and distance
    cold_hist_norm = cold_hist / (cold_hist.sum() + 1e-10)
    warm_hist_norm = warm_hist / (warm_hist.sum() + 1e-10)
    bc = np.sum(np.sqrt(cold_hist_norm * warm_hist_norm))
    bhattacharyya_dist = -np.log(bc + 1e-10)

    # KL divergence (symmetric)
    eps = 1e-10
    kl_cw = np.sum(cold_hist_norm * np.log((cold_hist_norm + eps) / (warm_hist_norm + eps)))
    kl_wc = np.sum(warm_hist_norm * np.log((warm_hist_norm + eps) / (cold_hist_norm + eps)))
    symmetric_kl = (kl_cw + kl_wc) / 2

    return {
        "mean_gap": mean_gap,
        "cohens_d": cohens_d,
        "overlap_coefficient": overlap,
        "bhattacharyya_dist": bhattacharyya_dist,
        "symmetric_kl": symmetric_kl,
        "cold_mean": cold_stats["mean"],
        "warm_mean": warm_stats["mean"],
        "cold_std": cold_stats["std"],
        "warm_std": warm_stats["std"],
    }


def load_samples(domain_path: Path, n_samples: int) -> Tuple[List, List, List, List]:
    """Load cold and warm samples (good and fault).

    Returns:
        cold_good, warm_good, cold_fault, warm_fault
    """
    good_path = domain_path / "test" / "good"
    fault_path = domain_path / "test" / "fault"

    good_files = sorted(good_path.glob("*.tiff"))
    fault_files = sorted(fault_path.glob("*.tiff"))

    # Separate by file index (0-499: cold, 500-999: warm)
    cold_good = []
    warm_good = []
    cold_fault = []
    warm_fault = []

    for f in good_files:
        idx = int(f.stem)
        img = load_tiff_image(f)
        if idx < 500:
            cold_good.append(img)
        else:
            warm_good.append(img)

    for f in fault_files:
        idx = int(f.stem)
        img = load_tiff_image(f)
        if idx < 500:
            cold_fault.append(img)
        else:
            warm_fault.append(img)

    # Subsample if needed
    if len(cold_good) > n_samples:
        step = len(cold_good) // n_samples
        cold_good = cold_good[::step][:n_samples]
    if len(warm_good) > n_samples:
        step = len(warm_good) // n_samples
        warm_good = warm_good[::step][:n_samples]
    if len(cold_fault) > n_samples:
        step = len(cold_fault) // n_samples
        cold_fault = cold_fault[::step][:n_samples]
    if len(warm_fault) > n_samples:
        step = len(warm_fault) // n_samples
        warm_fault = warm_fault[::step][:n_samples]

    return cold_good, warm_good, cold_fault, warm_fault


def visualize_distributions(
    raw_cold: Dict, raw_warm: Dict,
    norm_cold: Dict, norm_warm: Dict,
    raw_sep: Dict, norm_sep: Dict,
    output_path: Path
):
    """Create visualization comparing raw and normalized distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Raw intensity distributions
    ax = axes[0, 0]
    cold_hist, cold_bins = raw_cold["histogram"]
    warm_hist, warm_bins = raw_warm["histogram"]
    bin_centers_cold = (cold_bins[:-1] + cold_bins[1:]) / 2
    bin_centers_warm = (warm_bins[:-1] + warm_bins[1:]) / 2
    ax.plot(bin_centers_cold, cold_hist, 'b-', label=f'Cold (mean={raw_cold["mean"]:.4f})', alpha=0.7)
    ax.plot(bin_centers_warm, warm_hist, 'r-', label=f'Warm (mean={raw_warm["mean"]:.4f})', alpha=0.7)
    ax.axvline(raw_cold["mean"], color='blue', linestyle='--', alpha=0.5)
    ax.axvline(raw_warm["mean"], color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Density")
    ax.set_title(f"Raw Intensity Distribution\nGap={raw_sep['mean_gap']:.4f}, Cohen's d={raw_sep['cohens_d']:.2f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Normalized intensity distributions
    ax = axes[0, 1]
    cold_hist, cold_bins = norm_cold["histogram"]
    warm_hist, warm_bins = norm_warm["histogram"]
    bin_centers_cold = (cold_bins[:-1] + cold_bins[1:]) / 2
    bin_centers_warm = (warm_bins[:-1] + warm_bins[1:]) / 2
    ax.plot(bin_centers_cold, cold_hist, 'b-', label=f'Cold (mean={norm_cold["mean"]:.4f})', alpha=0.7)
    ax.plot(bin_centers_warm, warm_hist, 'r-', label=f'Warm (mean={norm_warm["mean"]:.4f})', alpha=0.7)
    ax.axvline(norm_cold["mean"], color='blue', linestyle='--', alpha=0.5)
    ax.axvline(norm_warm["mean"], color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel("Normalized Intensity")
    ax.set_ylabel("Density")
    ax.set_title(f"ImageNet Normalized Distribution\nGap={norm_sep['mean_gap']:.4f}, Cohen's d={norm_sep['cohens_d']:.2f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Separability metrics comparison
    ax = axes[1, 0]
    metrics = ['Cohen\'s d', 'Overlap', 'Bhatt. Dist']
    raw_values = [raw_sep['cohens_d'], raw_sep['overlap_coefficient'], raw_sep['bhattacharyya_dist']]
    norm_values = [norm_sep['cohens_d'], norm_sep['overlap_coefficient'], norm_sep['bhattacharyya_dist']]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, raw_values, width, label='Raw', color='steelblue')
    bars2 = ax.bar(x + width/2, norm_values, width, label='Normalized', color='coral')

    ax.set_ylabel('Value')
    ax.set_title('Separability Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('off')

    table_data = [
        ['Metric', 'Raw', 'Normalized', 'Change'],
        ['Cold Mean', f'{raw_sep["cold_mean"]:.4f}', f'{norm_sep["cold_mean"]:.4f}', ''],
        ['Warm Mean', f'{raw_sep["warm_mean"]:.4f}', f'{norm_sep["warm_mean"]:.4f}', ''],
        ['Mean Gap', f'{raw_sep["mean_gap"]:.4f}', f'{norm_sep["mean_gap"]:.4f}',
         f'{(norm_sep["mean_gap"]/raw_sep["mean_gap"]-1)*100:+.1f}%' if raw_sep["mean_gap"] != 0 else 'N/A'],
        ['Cohen\'s d', f'{raw_sep["cohens_d"]:.3f}', f'{norm_sep["cohens_d"]:.3f}',
         f'{(norm_sep["cohens_d"]/raw_sep["cohens_d"]-1)*100:+.1f}%' if raw_sep["cohens_d"] != 0 else 'N/A'],
        ['Overlap', f'{raw_sep["overlap_coefficient"]:.3f}', f'{norm_sep["overlap_coefficient"]:.3f}',
         f'{(norm_sep["overlap_coefficient"]/raw_sep["overlap_coefficient"]-1)*100:+.1f}%' if raw_sep["overlap_coefficient"] != 0 else 'N/A'],
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.3, 0.2, 0.25, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    # Style header row
    for j in range(4):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze normalization effect on cold/warm separability")
    parser.add_argument("--domain", type=str, required=True,
                        choices=["domain_A", "domain_B", "domain_C", "domain_D"])
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of samples per condition to analyze")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / args.domain / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Analysis 2: Preprocessing Normalization Investigation")
    print(f"Domain: {args.domain}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Load samples
    domain_path = DATASET_ROOT / args.domain
    print(f"Loading samples from: {domain_path}")

    cold_good, warm_good, cold_fault, warm_fault = load_samples(domain_path, args.n_samples)
    print(f"Loaded: {len(cold_good)} cold good, {len(warm_good)} warm good")
    print(f"        {len(cold_fault)} cold fault, {len(warm_fault)} warm fault")

    # Combine all cold and warm samples
    cold_all = cold_good + cold_fault
    warm_all = warm_good + warm_fault

    # Compute raw intensity stats
    print("\nComputing raw intensity statistics...")
    raw_cold_stats = compute_intensity_stats(cold_all)
    raw_warm_stats = compute_intensity_stats(warm_all)
    raw_separability = compute_separability_metrics(
        raw_cold_stats, raw_warm_stats, cold_all, warm_all
    )

    # Apply normalization and compute stats
    print("Applying ImageNet normalization...")
    norm_cold_all = [apply_imagenet_normalization(img) for img in cold_all]
    norm_warm_all = [apply_imagenet_normalization(img) for img in warm_all]

    norm_cold_stats = compute_intensity_stats(norm_cold_all)
    norm_warm_stats = compute_intensity_stats(norm_warm_all)
    norm_separability = compute_separability_metrics(
        norm_cold_stats, norm_warm_stats, norm_cold_all, norm_warm_all
    )

    # Create visualization
    print("\nCreating visualization...")
    visualize_distributions(
        raw_cold_stats, raw_warm_stats,
        norm_cold_stats, norm_warm_stats,
        raw_separability, norm_separability,
        output_dir / "normalization_comparison.png"
    )

    # Save results
    results = {
        "domain": args.domain,
        "n_samples": {
            "cold": len(cold_all),
            "warm": len(warm_all),
        },
        "imagenet_normalization": {
            "mean": IMAGENET_MEAN.mean(),
            "std": IMAGENET_STD.mean(),
        },
        "raw": {
            "cold": {k: v for k, v in raw_cold_stats.items() if k != "histogram"},
            "warm": {k: v for k, v in raw_warm_stats.items() if k != "histogram"},
            "separability": raw_separability,
        },
        "normalized": {
            "cold": {k: v for k, v in norm_cold_stats.items() if k != "histogram"},
            "warm": {k: v for k, v in norm_warm_stats.items() if k != "histogram"},
            "separability": norm_separability,
        },
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"\nRaw Intensity:")
    print(f"  Cold mean: {raw_cold_stats['mean']:.4f} (p90: {raw_cold_stats['p90']:.4f})")
    print(f"  Warm mean: {raw_warm_stats['mean']:.4f} (p90: {raw_warm_stats['p90']:.4f})")
    print(f"  Mean gap: {raw_separability['mean_gap']:.4f}")
    print(f"  Cohen's d: {raw_separability['cohens_d']:.3f}")

    print(f"\nAfter ImageNet Normalization:")
    print(f"  Cold mean: {norm_cold_stats['mean']:.4f}")
    print(f"  Warm mean: {norm_warm_stats['mean']:.4f}")
    print(f"  Mean gap: {norm_separability['mean_gap']:.4f}")
    print(f"  Cohen's d: {norm_separability['cohens_d']:.3f}")

    # Interpretation
    gap_preserved = abs(norm_separability['mean_gap'] / raw_separability['mean_gap'] - 1) < 0.5
    cohen_preserved = norm_separability['cohens_d'] / raw_separability['cohens_d'] > 0.5

    print(f"\n*** INTERPRETATION ***")
    if gap_preserved and cohen_preserved:
        print("ImageNet normalization PRESERVES cold/warm separability.")
        print("The intensity difference is NOT erased by normalization.")
        print("This means DINOv2 should see the cold/warm difference.")
    else:
        print("ImageNet normalization REDUCES cold/warm separability.")
        print("The intensity difference IS affected by normalization.")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
