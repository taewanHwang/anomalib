#!/usr/bin/env python3
"""
Direction-Agnostic Feature EDA for HDMAP Dataset.

Computes and analyzes three direction-independent features for adaptive discarding:
1. SF (Spectral Flatness): How flat the FFT spectrum is
2. GMM (Gradient Magnitude Mean): Edge density measure
3. LE (Local Entropy): Information content / complexity

These features are designed to work across different industrial domains,
not just HDMAP vertical stripes.

Usage:
    python feature_eda_direction_agnostic.py --output-dir results/feature_eda_v2
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage
from scipy.stats import entropy as scipy_entropy
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Dataset path
HDMAP_ROOT = "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png"
DOMAINS = ["domain_A", "domain_B", "domain_C", "domain_D"]


def load_images(domain_path: Path, split: str, label: str, max_samples: int = None) -> List[np.ndarray]:
    """Load images from a directory."""
    img_dir = domain_path / split / label
    if not img_dir.exists():
        return []

    img_files = sorted(img_dir.glob("*.png"))
    if max_samples:
        img_files = img_files[:max_samples]

    images = []
    for img_path in img_files:
        img = np.array(Image.open(img_path).convert("L"))  # Grayscale
        images.append(img)
    return images


# ============================================================
# Feature 1: Spectral Flatness (SF)
# ============================================================
def compute_spectral_flatness(image: np.ndarray) -> float:
    """
    Compute Spectral Flatness from FFT magnitude spectrum.

    SF = geometric_mean(spectrum) / arithmetic_mean(spectrum)

    - SF close to 1: Flat spectrum (noise-like, complex pattern)
    - SF close to 0: Peaked spectrum (regular, structured pattern)

    This is direction-agnostic - considers all frequencies equally.

    Returns:
        SF value in (0, 1]
    """
    # 2D FFT
    f = np.fft.fft2(image.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # Exclude DC component
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    magnitude[cy, cx] = 0

    # Flatten and remove zeros
    mag_flat = magnitude.flatten()
    mag_flat = mag_flat[mag_flat > 1e-10]

    if len(mag_flat) == 0:
        return 0.0

    # Spectral flatness = geometric mean / arithmetic mean
    log_mag = np.log(mag_flat + 1e-10)
    geometric_mean = np.exp(np.mean(log_mag))
    arithmetic_mean = np.mean(mag_flat)

    if arithmetic_mean < 1e-10:
        return 0.0

    sf = geometric_mean / arithmetic_mean
    return float(np.clip(sf, 0.0, 1.0))


def compute_sf_batch(images: List[np.ndarray]) -> np.ndarray:
    """Compute Spectral Flatness for a batch of images."""
    return np.array([compute_spectral_flatness(img) for img in images])


# ============================================================
# Feature 2: Gradient Magnitude Mean (GMM)
# ============================================================
def compute_gradient_magnitude_mean(image: np.ndarray) -> float:
    """
    Compute mean gradient magnitude (edge density).

    GMM = mean(sqrt(Gx^2 + Gy^2))

    - High GMM: Many edges (complex pattern)
    - Low GMM: Few edges (simple/uniform pattern)

    Direction-agnostic - only considers magnitude, not direction.

    Returns:
        GMM value (normalized by max possible gradient)
    """
    img_float = image.astype(np.float32) / 255.0

    # Sobel gradients
    gx = ndimage.sobel(img_float, axis=1)  # Horizontal gradient
    gy = ndimage.sobel(img_float, axis=0)  # Vertical gradient

    # Gradient magnitude
    magnitude = np.sqrt(gx**2 + gy**2)

    # Mean gradient magnitude
    gmm = np.mean(magnitude)

    # Normalize (max Sobel response is ~4 for 0-1 image)
    gmm_normalized = gmm / 4.0

    return float(np.clip(gmm_normalized, 0.0, 1.0))


def compute_gmm_batch(images: List[np.ndarray]) -> np.ndarray:
    """Compute Gradient Magnitude Mean for a batch of images."""
    return np.array([compute_gradient_magnitude_mean(img) for img in images])


# ============================================================
# Feature 3: Local Entropy (LE)
# ============================================================
def compute_local_entropy(image: np.ndarray, disk_radius: int = 5) -> float:
    """
    Compute mean local entropy using sliding window.

    LE = mean of local Shannon entropy over image patches

    - High LE: High information content (complex/irregular)
    - Low LE: Low information content (simple/regular)

    Direction-agnostic - based on local histogram statistics.

    Args:
        image: Grayscale image
        disk_radius: Radius of local neighborhood

    Returns:
        Mean local entropy (normalized to [0, 1])
    """
    from skimage.filters.rank import entropy as skimage_entropy
    from skimage.morphology import disk

    # Ensure uint8
    if image.dtype != np.uint8:
        img_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    else:
        img_uint8 = image

    # Compute local entropy
    try:
        local_ent = skimage_entropy(img_uint8, disk(disk_radius))
        # Normalize by max possible entropy (log2(256) = 8 for 8-bit image)
        mean_entropy = np.mean(local_ent) / 8.0
        return float(np.clip(mean_entropy, 0.0, 1.0))
    except Exception as e:
        # Fallback to global histogram entropy
        hist, _ = np.histogram(img_uint8.flatten(), bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        global_entropy = scipy_entropy(hist, base=2) / 8.0
        return float(np.clip(global_entropy, 0.0, 1.0))


def compute_le_batch(images: List[np.ndarray]) -> np.ndarray:
    """Compute Local Entropy for a batch of images."""
    return np.array([compute_local_entropy(img) for img in images])


# ============================================================
# Additional: Intensity Stats (for comparison)
# ============================================================
def compute_intensity_stats(image: np.ndarray) -> Tuple[float, float]:
    """Compute intensity mean and std."""
    img_float = image.astype(np.float32) / 255.0
    return float(np.mean(img_float)), float(np.std(img_float))


def compute_intensity_batch(images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute intensity stats for a batch of images."""
    means = []
    stds = []
    for img in images:
        m, s = compute_intensity_stats(img)
        means.append(m)
        stds.append(s)
    return np.array(means), np.array(stds)


# ============================================================
# EDA Functions
# ============================================================
def compute_statistics(values: np.ndarray) -> Dict:
    """Compute comprehensive statistics for a feature array."""
    if len(values) == 0:
        return {}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def run_eda_for_domain(domain: str, max_samples: int = None) -> Dict:
    """Run comprehensive EDA for a single domain."""
    domain_path = Path(HDMAP_ROOT) / domain

    print(f"\n{'='*60}")
    print(f"Processing {domain}")
    print(f"{'='*60}")

    results = {
        "domain": domain,
        "splits": {}
    }

    splits = [
        ("train", "good"),
        ("test", "good"),
        ("test", "fault"),
    ]

    for split, label in splits:
        split_key = f"{split}_{label}"
        print(f"\n  Loading {split_key}...")

        images = load_images(domain_path, split, label, max_samples)
        if not images:
            print(f"    No images found for {split_key}")
            continue

        print(f"    Loaded {len(images)} images")

        # Compute features
        print(f"    Computing Spectral Flatness (SF)...")
        sf_values = compute_sf_batch(images)

        print(f"    Computing Gradient Magnitude Mean (GMM)...")
        gmm_values = compute_gmm_batch(images)

        print(f"    Computing Local Entropy (LE)...")
        le_values = compute_le_batch(images)

        print(f"    Computing Intensity Stats...")
        intensity_means, intensity_stds = compute_intensity_batch(images)

        # Store results
        results["splits"][split_key] = {
            "sf": compute_statistics(sf_values),
            "gmm": compute_statistics(gmm_values),
            "le": compute_statistics(le_values),
            "intensity_mean": compute_statistics(intensity_means),
            "intensity_std": compute_statistics(intensity_stds),
            "raw": {
                "sf": sf_values.tolist(),
                "gmm": gmm_values.tolist(),
                "le": le_values.tolist(),
                "intensity_mean": intensity_means.tolist(),
                "intensity_std": intensity_stds.tolist(),
            }
        }

    return results


def print_summary(all_results: List[Dict]):
    """Print summary table of results."""
    print("\n" + "="*100)
    print("DIRECTION-AGNOSTIC FEATURE EDA SUMMARY")
    print("="*100)

    features = ["sf", "gmm", "le", "intensity_mean", "intensity_std"]
    feature_labels = {
        "sf": "Spectral Flatness",
        "gmm": "Gradient Mag Mean",
        "le": "Local Entropy",
        "intensity_mean": "Intensity Mean",
        "intensity_std": "Intensity Std",
    }

    for feature in features:
        print(f"\n{'='*100}")
        print(f"Feature: {feature_labels[feature]}")
        print(f"{'='*100}")
        print(f"{'Domain':<12} | {'Split':<12} | {'Mean':>10} | {'Std':>10} | {'P10':>10} | {'P50':>10} | {'P90':>10}")
        print("-"*100)

        for domain_result in all_results:
            domain = domain_result["domain"]
            for split_key, split_data in domain_result["splits"].items():
                if feature in split_data:
                    stats = split_data[feature]
                    print(f"{domain:<12} | {split_key:<12} | {stats['mean']:>10.4f} | {stats['std']:>10.4f} | {stats['p10']:>10.4f} | {stats['p50']:>10.4f} | {stats['p90']:>10.4f}")


def create_violin_plots(all_results: List[Dict], output_dir: Path):
    """Create violin plots for each feature across domains."""
    output_dir.mkdir(parents=True, exist_ok=True)

    features = ["sf", "gmm", "le", "intensity_mean", "intensity_std"]
    feature_labels = {
        "sf": "Spectral Flatness (SF)",
        "gmm": "Gradient Magnitude Mean (GMM)",
        "le": "Local Entropy (LE)",
        "intensity_mean": "Intensity Mean",
        "intensity_std": "Intensity Std",
    }

    for feature in features:
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        fig.suptitle(f"{feature_labels[feature]} Distribution by Domain", fontsize=14)

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
                        if "fault" in split_key:
                            colors.append("red")
                        else:
                            colors.append("blue")

            if data:
                parts = ax.violinplot(data, positions=range(len(data)), showmeans=True, showmedians=True)
                ax.set_xticks(range(len(data)))
                ax.set_xticklabels(labels)
                ax.set_title(domain)
                ax.set_ylabel(feature_labels[feature])

                # Color the violins
                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(colors[i] if i < len(colors) else 'gray')
                    pc.set_alpha(0.7)

        plt.tight_layout()
        plt.savefig(output_dir / f"{feature}_violin.png", dpi=150)
        plt.close()
        print(f"  Saved: {output_dir / f'{feature}_violin.png'}")


def compute_separability(all_results: List[Dict]) -> Dict:
    """Compute separability metrics between normal and fault."""
    separability = {}
    features = ["sf", "gmm", "le", "intensity_mean", "intensity_std"]

    for domain_result in all_results:
        domain = domain_result["domain"]
        separability[domain] = {}

        splits = domain_result["splits"]
        if "test_good" not in splits or "test_fault" not in splits:
            continue

        for feature in features:
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

            # AUC approximation
            from scipy import stats
            try:
                statistic, pvalue = stats.mannwhitneyu(normal, fault, alternative='two-sided')
                auc = statistic / (len(normal) * len(fault))
            except:
                auc = 0.5

            separability[domain][feature] = {
                "cohens_d": float(cohens_d),
                "auc": float(auc),
                "normal_mean": float(np.mean(normal)),
                "fault_mean": float(np.mean(fault)),
                "direction": "fault_higher" if np.mean(fault) > np.mean(normal) else "fault_lower",
            }

    return separability


def print_separability(separability: Dict):
    """Print separability analysis."""
    print("\n" + "="*110)
    print("SEPARABILITY ANALYSIS (Normal vs Fault) - Direction-Agnostic Features")
    print("="*110)
    cohens_label = "Cohen's d"
    print(f"{'Domain':<12} | {'Feature':<18} | {cohens_label:>12} | {'AUC':>10} | {'Direction':<15} | {'Effect Size':<12}")
    print("-"*110)

    for domain in sorted(separability.keys()):
        features = separability[domain]
        for feature, metrics in sorted(features.items()):
            d = metrics["cohens_d"]
            auc = metrics["auc"]
            direction = metrics["direction"]

            # Interpret effect size
            abs_d = abs(d)
            if abs_d < 0.2:
                interp = "Negligible"
            elif abs_d < 0.5:
                interp = "Small"
            elif abs_d < 0.8:
                interp = "Medium"
            else:
                interp = "Large"

            # Highlight good separability
            marker = " ***" if abs_d >= 0.8 else (" **" if abs_d >= 0.5 else "")

            print(f"{domain:<12} | {feature:<18} | {d:>12.4f} | {auc:>10.4f} | {direction:<15} | {interp:<12}{marker}")


def create_comparison_table(separability: Dict, output_dir: Path):
    """Create a comparison table for Domain C features."""
    domain_c = separability.get("domain_C", {})

    print("\n" + "="*80)
    print("DOMAIN C - FEATURE COMPARISON FOR ADAPTIVE DISCARDING")
    print("="*80)

    # Sort by absolute Cohen's d
    sorted_features = sorted(domain_c.items(), key=lambda x: abs(x[1]["cohens_d"]), reverse=True)

    cohens_header = "|Cohen's d|"
    print(f"{'Rank':<6} | {'Feature':<20} | {cohens_header:>12} | {'Direction':<15} | {'Recommendation':<15}")
    print("-"*80)

    for rank, (feature, metrics) in enumerate(sorted_features, 1):
        abs_d = abs(metrics["cohens_d"])
        direction = metrics["direction"]

        if abs_d >= 0.8:
            rec = "Highly Recommended"
        elif abs_d >= 0.5:
            rec = "Recommended"
        elif abs_d >= 0.2:
            rec = "Consider"
        else:
            rec = "Not Recommended"

        print(f"{rank:<6} | {feature:<20} | {abs_d:>12.4f} | {direction:<15} | {rec:<15}")


def main():
    parser = argparse.ArgumentParser(description="Direction-Agnostic Feature EDA for HDMAP")
    parser.add_argument("--output-dir", type=str, default="results/feature_eda_v2",
                        help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per split (for quick testing)")
    args = parser.parse_args()

    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("HDMAP Direction-Agnostic Feature EDA")
    print("Features: Spectral Flatness, Gradient Mag Mean, Local Entropy")
    print("="*70)
    print(f"Output directory: {output_dir}")

    # Run EDA for all domains
    all_results = []
    for domain in DOMAINS:
        result = run_eda_for_domain(domain, max_samples=args.max_samples)
        all_results.append(result)

    # Print summary
    print_summary(all_results)

    # Compute and print separability
    separability = compute_separability(all_results)
    print_separability(separability)

    # Domain C comparison
    create_comparison_table(separability, output_dir)

    # Create visualizations
    print("\nCreating visualizations...")
    create_violin_plots(all_results, output_dir)

    # Save results to JSON
    results_file = output_dir / "feature_eda_v2_results.json"
    with open(results_file, "w") as f:
        results_for_json = []
        for r in all_results:
            r_copy = {"domain": r["domain"], "splits": {}}
            for split_key, split_data in r["splits"].items():
                r_copy["splits"][split_key] = {k: v for k, v in split_data.items() if k != "raw"}
            results_for_json.append(r_copy)

        json.dump({
            "results": results_for_json,
            "separability": separability,
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Create summary markdown
    summary_file = output_dir / "FEATURE_EDA_V2_SUMMARY.md"
    with open(summary_file, "w") as f:
        f.write("# Direction-Agnostic Feature EDA Summary\n\n")
        f.write("## Features Analyzed\n\n")
        f.write("1. **SF (Spectral Flatness)**: FFT spectrum flatness (direction-agnostic APE alternative)\n")
        f.write("2. **GMM (Gradient Magnitude Mean)**: Edge density (direction-agnostic)\n")
        f.write("3. **LE (Local Entropy)**: Information content / complexity\n")
        f.write("4. **Intensity Mean/Std**: Brightness statistics (baseline)\n\n")

        f.write("## Key Properties\n\n")
        f.write("| Feature | Direction Independent | Computation | Interpretation |\n")
        f.write("|---------|----------------------|-------------|----------------|\n")
        f.write("| SF | Yes | FFT-based | Low=Regular, High=Complex |\n")
        f.write("| GMM | Yes | Gradient-based | Low=Uniform, High=Edgy |\n")
        f.write("| LE | Yes | Histogram-based | Low=Simple, High=Complex |\n\n")

        f.write("## Domain C Separability\n\n")
        if "domain_C" in separability:
            f.write("| Feature | Cohen's d | Direction | Recommendation |\n")
            f.write("|---------|-----------|-----------|----------------|\n")
            sorted_features = sorted(separability["domain_C"].items(),
                                    key=lambda x: abs(x[1]["cohens_d"]), reverse=True)
            for feature, metrics in sorted_features:
                d = metrics["cohens_d"]
                direction = metrics["direction"]
                abs_d = abs(d)
                if abs_d >= 0.8:
                    rec = "Highly Recommended"
                elif abs_d >= 0.5:
                    rec = "Recommended"
                else:
                    rec = "Consider"
                f.write(f"| {feature} | {d:.4f} | {direction} | {rec} |\n")

    print(f"Summary saved to: {summary_file}")
    print("\nEDA Complete!")


if __name__ == "__main__":
    main()
