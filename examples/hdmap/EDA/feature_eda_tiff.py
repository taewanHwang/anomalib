#!/usr/bin/env python3
"""
Comprehensive Feature EDA for HDMAP TIFF Dataset.

Uses ORIGINAL TIFF files to preserve absolute amplitude values.
(PNG conversion applies per-image min-max normalization which loses this info)

Features analyzed:
1. Intensity Mean (amplitude)
2. Intensity Std (contrast)
3. Intensity Max (peak value)
4. Intensity Range (dynamic range)
5. GMM (Gradient Magnitude Mean)
6. LE (Global Entropy)
7. APE (Angular Power Entropy) - frequency domain
8. OE (Orientational Entropy) - spatial domain gradient
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Dataset path - TIFF original!
HDMAP_TIFF_ROOT = "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_tiff_minmax"
DOMAINS = ["domain_A", "domain_B", "domain_C", "domain_D"]


def load_tiff_images(domain_path: Path, split: str, label: str, max_samples: int = None) -> List[np.ndarray]:
    """Load TIFF images from a directory (preserving original values)."""
    img_dir = domain_path / split / label
    if not img_dir.exists():
        return []

    img_files = sorted(img_dir.glob("*.tiff")) + sorted(img_dir.glob("*.tif"))
    if max_samples:
        img_files = img_files[:max_samples]

    images = []
    for img_path in img_files:
        img = Image.open(img_path)
        # TIFF mode 'F' = 32-bit float, preserve original values!
        arr = np.array(img, dtype=np.float32)
        images.append(arr)
    return images


# ============================================================
# Feature Computations
# ============================================================
def compute_intensity_stats(image: np.ndarray) -> Dict[str, float]:
    """Compute intensity statistics (on original TIFF values)."""
    return {
        "mean": float(np.mean(image)),
        "std": float(np.std(image)),
        "min": float(np.min(image)),
        "max": float(np.max(image)),
        "range": float(np.max(image) - np.min(image)),
        "p10": float(np.percentile(image, 10)),
        "p90": float(np.percentile(image, 90)),
    }


def compute_ape(image: np.ndarray, num_angle_bins: int = 36) -> float:
    """
    Compute Angular Power Entropy (APE) - frequency domain feature.

    APE measures how concentrated the power spectrum is along specific angles:
    - Low APE: Energy concentrated in specific directions (regular pattern)
    - High APE: Energy spread across all directions (isotropic/complex)

    Args:
        image: 2D grayscale image (H, W)
        num_angle_bins: Number of angular bins

    Returns:
        APE value in [0, 1]
    """
    eps = 1e-12
    H, W = image.shape

    # Normalize image to 0-1 range for consistent processing
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        img_norm = (image - img_min) / (img_max - img_min)
    else:
        return 0.5  # No variation

    # Remove DC (mean subtraction)
    img_centered = img_norm - img_norm.mean()

    # Apply Hann window to reduce spectral leakage
    hann_y = np.hanning(H).reshape(H, 1)
    hann_x = np.hanning(W).reshape(1, W)
    img_windowed = img_centered * hann_y * hann_x

    # 2D FFT
    fft = np.fft.fft2(img_windowed)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2

    # Create coordinate grids
    cy, cx = H // 2, W // 2
    y_coords = np.arange(H) - cy
    x_coords = np.arange(W) - cx
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Radius and angle
    r = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)  # [-pi, pi]
    theta = np.mod(theta, np.pi)  # [0, pi) - fold to half circle

    # Radius mask (exclude DC and corners)
    r_max = min(cy, cx)
    r_min = r_max * 0.05
    r_max_limit = r_max * 0.95
    mask = (r >= r_min) & (r <= r_max_limit)

    if not np.any(mask):
        return 0.5

    # Extract values
    theta_m = theta[mask]
    power_m = power[mask]

    # Weighted histogram by power
    hist, _ = np.histogram(theta_m, bins=num_angle_bins, range=(0, np.pi), weights=power_m)
    total = hist.sum()

    if total <= 0:
        return 0.5

    p = hist / total

    # Shannon entropy normalized
    H_entropy = -np.sum(p * np.log(p + eps))
    H_max = np.log(num_angle_bins)
    ape = H_entropy / H_max

    return float(np.clip(ape, 0.0, 1.0))


def compute_oe(image: np.ndarray, num_bins: int = 36, magnitude_threshold: float = 1e-3) -> float:
    """
    Compute Orientational Entropy (OE) - spatial domain gradient feature.

    OE measures how diverse the gradient directions are:
    - Low OE: Strong dominant direction (regular structure)
    - High OE: Diverse directions (complex/irregular)

    Args:
        image: 2D grayscale image (H, W)
        num_bins: Number of orientation bins over [0, pi)
        magnitude_threshold: Ignore pixels below this gradient magnitude

    Returns:
        OE value in [0, 1]
    """
    eps = 1e-12

    # Normalize to 0-1 for consistent gradient computation
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        img_norm = ((image - img_min) / (img_max - img_min)).astype(np.float32)
    else:
        return 0.5

    # Sobel gradients
    gx = cv2.Sobel(img_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_norm, cv2.CV_32F, 0, 1, ksize=3)

    # Magnitude and orientation
    mag = np.sqrt(gx**2 + gy**2)

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

    # Shannon entropy normalized
    H_entropy = -np.sum(p * np.log(p + eps))
    H_max = np.log(num_bins)
    oe = H_entropy / H_max

    return float(np.clip(oe, 0.0, 1.0))


def compute_gmm(image: np.ndarray) -> float:
    """Compute Gradient Magnitude Mean."""
    gx = ndimage.sobel(image, axis=1)
    gy = ndimage.sobel(image, axis=0)
    magnitude = np.sqrt(gx**2 + gy**2)
    return float(np.mean(magnitude))


def compute_le(image: np.ndarray, num_bins: int = 256) -> float:
    """
    Compute Global Entropy (histogram-based).

    For TIFF with variable ranges, we use global histogram entropy
    instead of local entropy which requires uint8 input.
    """
    # Normalize to 0-1 for histogram
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        img_normalized = (image - img_min) / (img_max - img_min)
    else:
        return 0.0

    # Compute histogram
    hist, _ = np.histogram(img_normalized.flatten(), bins=num_bins, range=(0, 1), density=True)
    hist = hist[hist > 0]  # Remove zeros

    # Shannon entropy
    from scipy.stats import entropy
    return float(entropy(hist, base=2))


def compute_all_features(images: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute all features for a list of images."""
    intensity_means = []
    intensity_stds = []
    intensity_maxs = []
    intensity_ranges = []
    gmm_values = []
    le_values = []
    ape_values = []
    oe_values = []

    for img in images:
        stats = compute_intensity_stats(img)
        intensity_means.append(stats["mean"])
        intensity_stds.append(stats["std"])
        intensity_maxs.append(stats["max"])
        intensity_ranges.append(stats["range"])
        gmm_values.append(compute_gmm(img))
        le_values.append(compute_le(img))
        ape_values.append(compute_ape(img))
        oe_values.append(compute_oe(img))

    return {
        "intensity_mean": np.array(intensity_means),
        "intensity_std": np.array(intensity_stds),
        "intensity_max": np.array(intensity_maxs),
        "intensity_range": np.array(intensity_ranges),
        "gmm": np.array(gmm_values),
        "le": np.array(le_values),
        "ape": np.array(ape_values),
        "oe": np.array(oe_values),
    }


def compute_statistics(values: np.ndarray) -> Dict:
    """Compute summary statistics."""
    if len(values) == 0:
        return {}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def run_eda_for_domain(domain: str, max_samples: int = None) -> Dict:
    """Run EDA for a single domain."""
    domain_path = Path(HDMAP_TIFF_ROOT) / domain

    print(f"\n{'='*60}")
    print(f"Processing {domain} (TIFF original)")
    print(f"{'='*60}")

    results = {"domain": domain, "splits": {}}

    splits = [("train", "good"), ("test", "good"), ("test", "fault")]

    for split, label in splits:
        split_key = f"{split}_{label}"
        print(f"\n  Loading {split_key}...")

        images = load_tiff_images(domain_path, split, label, max_samples)
        if not images:
            print(f"    No images found")
            continue

        print(f"    Loaded {len(images)} TIFF images")
        print(f"    Computing features...")

        features = compute_all_features(images)

        results["splits"][split_key] = {
            "stats": {name: compute_statistics(values) for name, values in features.items()},
            "raw": {name: values.tolist() for name, values in features.items()},
        }

    return results


def compute_separability(all_results: List[Dict]) -> Dict:
    """Compute separability between normal and fault."""
    separability = {}
    feature_names = ["intensity_mean", "intensity_std", "intensity_max", "intensity_range", "gmm", "le", "ape", "oe"]

    for domain_result in all_results:
        domain = domain_result["domain"]
        separability[domain] = {}

        splits = domain_result["splits"]
        if "test_good" not in splits or "test_fault" not in splits:
            continue

        for feature in feature_names:
            normal_raw = splits["test_good"].get("raw", {}).get(feature, [])
            fault_raw = splits["test_fault"].get("raw", {}).get(feature, [])

            if not normal_raw or not fault_raw:
                continue

            normal = np.array(normal_raw)
            fault = np.array(fault_raw)

            # Cohen's d
            pooled_std = np.sqrt((np.var(normal) + np.var(fault)) / 2)
            if pooled_std > 1e-10:
                cohens_d = (np.mean(fault) - np.mean(normal)) / pooled_std
            else:
                cohens_d = 0.0

            separability[domain][feature] = {
                "cohens_d": float(cohens_d),
                "normal_mean": float(np.mean(normal)),
                "fault_mean": float(np.mean(fault)),
                "direction": "fault_higher" if np.mean(fault) > np.mean(normal) else "fault_lower",
                "diff_percent": float((np.mean(fault) - np.mean(normal)) / np.mean(normal) * 100) if np.mean(normal) != 0 else 0,
            }

    return separability


def print_separability(separability: Dict):
    """Print separability analysis."""
    print("\n" + "="*120)
    print("SEPARABILITY ANALYSIS (TIFF Original) - Normal vs Fault")
    print("="*120)

    cohens_label = "Cohen's d"
    print(f"{'Domain':<12} | {'Feature':<18} | {cohens_label:>12} | {'Normal Mean':>12} | {'Fault Mean':>12} | {'Diff %':>10} | {'Direction':<15}")
    print("-"*120)

    for domain in sorted(separability.keys()):
        features = separability[domain]
        for feature, metrics in sorted(features.items()):
            d = metrics["cohens_d"]
            normal_mean = metrics["normal_mean"]
            fault_mean = metrics["fault_mean"]
            diff_pct = metrics["diff_percent"]
            direction = metrics["direction"]

            # Effect size interpretation
            abs_d = abs(d)
            if abs_d >= 0.8:
                marker = " ***"
            elif abs_d >= 0.5:
                marker = " **"
            else:
                marker = ""

            print(f"{domain:<12} | {feature:<18} | {d:>12.4f} | {normal_mean:>12.4f} | {fault_mean:>12.4f} | {diff_pct:>+9.1f}% | {direction:<15}{marker}")


def create_comparison_plot(all_results: List[Dict], output_dir: Path):
    """Create comparison violin plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = ["intensity_mean", "intensity_std", "intensity_max", "intensity_range", "gmm", "le", "ape", "oe"]
    feature_labels = {
        "intensity_mean": "Intensity Mean\n(Amplitude)",
        "intensity_std": "Intensity Std\n(Contrast)",
        "intensity_max": "Intensity Max\n(Peak)",
        "intensity_range": "Intensity Range\n(Dynamic Range)",
        "gmm": "GMM\n(Edge Density)",
        "le": "Global Entropy\n(Complexity)",
        "ape": "APE\n(Freq. Angular Entropy)",
        "oe": "OE\n(Spatial Orient. Entropy)",
    }

    for feature in feature_names:
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        fig.suptitle(f"{feature_labels[feature]} Distribution (TIFF Original)", fontsize=14, fontweight='bold')

        for idx, domain_result in enumerate(all_results):
            ax = axes[idx]
            domain = domain_result["domain"]

            data = []
            labels = []
            colors = []

            for split_key in ["train_good", "test_good", "test_fault"]:
                if split_key in domain_result["splits"]:
                    raw_data = domain_result["splits"][split_key].get("raw", {}).get(feature, [])
                    if raw_data:
                        data.append(raw_data)
                        labels.append(split_key.replace("_", "\n"))
                        colors.append("red" if "fault" in split_key else "blue")

            if data:
                parts = ax.violinplot(data, positions=range(len(data)), showmeans=True, showmedians=True)
                ax.set_xticks(range(len(data)))
                ax.set_xticklabels(labels)
                ax.set_title(domain)
                ax.set_ylabel(feature_labels[feature])

                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(colors[i] if i < len(colors) else 'gray')
                    pc.set_alpha(0.7)

        plt.tight_layout()
        plt.savefig(output_dir / f"tiff_{feature}_violin.png", dpi=150)
        plt.close()
        print(f"  Saved: {output_dir / f'tiff_{feature}_violin.png'}")


def main():
    parser = argparse.ArgumentParser(description="Feature EDA on TIFF Original Data")
    parser.add_argument("--output-dir", type=str, default="results/feature_eda_tiff",
                        help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per split")
    args = parser.parse_args()

    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("HDMAP Feature EDA on TIFF ORIGINAL DATA")
    print("(Preserves absolute amplitude - no per-image normalization)")
    print("="*70)

    # Run EDA
    all_results = []
    for domain in DOMAINS:
        result = run_eda_for_domain(domain, max_samples=args.max_samples)
        all_results.append(result)

    # Separability analysis
    separability = compute_separability(all_results)
    print_separability(separability)

    # Create visualizations
    print("\nCreating visualizations...")
    create_comparison_plot(all_results, output_dir)

    # Save results
    results_file = output_dir / "feature_eda_tiff_results.json"
    with open(results_file, "w") as f:
        results_for_json = []
        for r in all_results:
            r_copy = {"domain": r["domain"], "splits": {}}
            for split_key, split_data in r["splits"].items():
                r_copy["splits"][split_key] = {"stats": split_data.get("stats", {})}
            results_for_json.append(r_copy)
        json.dump({"results": results_for_json, "separability": separability}, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Print Domain C summary
    print("\n" + "="*80)
    print("DOMAIN C SUMMARY (TIFF Original)")
    print("="*80)
    if "domain_C" in separability:
        sorted_features = sorted(separability["domain_C"].items(),
                                key=lambda x: abs(x[1]["cohens_d"]), reverse=True)

        cohens_header = "|Cohen's d|"
        print(f"{'Rank':<6} | {'Feature':<20} | {cohens_header:>12} | {'Direction':<15} | {'Diff %':>10}")
        print("-"*80)

        for rank, (feature, metrics) in enumerate(sorted_features, 1):
            abs_d = abs(metrics["cohens_d"])
            direction = metrics["direction"]
            diff_pct = metrics["diff_percent"]
            print(f"{rank:<6} | {feature:<20} | {abs_d:>12.4f} | {direction:<15} | {diff_pct:>+9.1f}%")

    print("\nEDA Complete!")


if __name__ == "__main__":
    main()
