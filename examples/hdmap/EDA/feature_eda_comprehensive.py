#!/usr/bin/env python3
"""
Comprehensive Feature EDA for HDMAP Dataset.

Computes and analyzes three new features for adaptive discarding:
1. VPR (Vertical Power Ratio): FFT-based vertical structure measure
2. Intensity Stats (Mean/Std): Simple brightness statistics
3. SRI (Stripe Regularity Index): Column-wise autocorrelation

Usage:
    python feature_eda_comprehensive.py --output-dir results/feature_eda
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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
# Feature 1: VPR (Vertical Power Ratio)
# ============================================================
def compute_vpr(image: np.ndarray) -> float:
    """
    Compute Vertical Power Ratio from FFT.

    VPR = vertical_power / horizontal_power

    - High VPR: Strong vertical structure (typical for normal HDMAP)
    - Low VPR: Weak vertical structure (potential anomaly)

    Returns:
        VPR value (typically > 1 for vertical-dominant images)
    """
    # 2D FFT
    f = np.fft.fft2(image.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    # Define vertical and horizontal bands (excluding DC)
    band_width = min(h, w) // 8  # Width of the band

    # Vertical band: narrow horizontal strip through center (captures horizontal frequencies = vertical patterns)
    vertical_mask = np.zeros_like(magnitude, dtype=bool)
    vertical_mask[cy-band_width:cy+band_width, :] = True
    vertical_mask[cy-2:cy+2, :] = False  # Exclude DC

    # Horizontal band: narrow vertical strip through center (captures vertical frequencies = horizontal patterns)
    horizontal_mask = np.zeros_like(magnitude, dtype=bool)
    horizontal_mask[:, cx-band_width:cx+band_width] = True
    horizontal_mask[:, cx-2:cx+2] = False  # Exclude DC

    # Note: In FFT, horizontal frequencies correspond to vertical patterns in spatial domain
    # So we swap the naming for intuitive interpretation
    vertical_power = np.sum(magnitude[vertical_mask])  # Power from vertical patterns
    horizontal_power = np.sum(magnitude[horizontal_mask])  # Power from horizontal patterns

    # Avoid division by zero
    if horizontal_power < 1e-10:
        return 10.0  # Cap at high value

    vpr = vertical_power / horizontal_power
    return float(vpr)


def compute_vpr_batch(images: List[np.ndarray]) -> np.ndarray:
    """Compute VPR for a batch of images."""
    return np.array([compute_vpr(img) for img in images])


# ============================================================
# Feature 2: Intensity Statistics (Mean, Std)
# ============================================================
def compute_intensity_stats(image: np.ndarray) -> Tuple[float, float]:
    """
    Compute intensity mean and std.

    Returns:
        (mean, std) tuple
    """
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
# Feature 3: SRI (Stripe Regularity Index)
# ============================================================
def compute_sri(image: np.ndarray, lag_range: int = 10) -> float:
    """
    Compute Stripe Regularity Index using column-wise autocorrelation.

    SRI measures how regular/periodic the vertical stripe pattern is.
    - High SRI: Regular, periodic stripes (normal)
    - Low SRI: Irregular, broken stripes (anomaly)

    Args:
        image: Grayscale image
        lag_range: Range of lags to compute autocorrelation

    Returns:
        SRI value in [0, 1]
    """
    img_float = image.astype(np.float32)
    h, w = img_float.shape

    # Compute column means (vertical profile)
    col_profile = np.mean(img_float, axis=0)  # Shape: (w,)

    # Normalize
    col_profile = col_profile - np.mean(col_profile)

    # Compute autocorrelation for various lags
    autocorr_values = []
    norm = np.sum(col_profile ** 2)

    if norm < 1e-10:
        return 0.0

    for lag in range(1, min(lag_range + 1, w)):
        corr = np.sum(col_profile[:-lag] * col_profile[lag:]) / norm
        autocorr_values.append(corr)

    # SRI = mean of positive autocorrelation values
    autocorr_values = np.array(autocorr_values)
    positive_autocorr = autocorr_values[autocorr_values > 0]

    if len(positive_autocorr) == 0:
        return 0.0

    sri = float(np.mean(positive_autocorr))
    return max(0.0, min(1.0, sri))  # Clamp to [0, 1]


def compute_sri_batch(images: List[np.ndarray]) -> np.ndarray:
    """Compute SRI for a batch of images."""
    return np.array([compute_sri(img) for img in images])


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
        print(f"    Computing VPR...")
        vpr_values = compute_vpr_batch(images)

        print(f"    Computing Intensity Stats...")
        intensity_means, intensity_stds = compute_intensity_batch(images)

        print(f"    Computing SRI...")
        sri_values = compute_sri_batch(images)

        # Store results
        results["splits"][split_key] = {
            "vpr": compute_statistics(vpr_values),
            "intensity_mean": compute_statistics(intensity_means),
            "intensity_std": compute_statistics(intensity_stds),
            "sri": compute_statistics(sri_values),
            "raw": {
                "vpr": vpr_values.tolist(),
                "intensity_mean": intensity_means.tolist(),
                "intensity_std": intensity_stds.tolist(),
                "sri": sri_values.tolist(),
            }
        }

    return results


def print_summary(all_results: List[Dict]):
    """Print summary table of results."""
    print("\n" + "="*100)
    print("FEATURE EDA SUMMARY")
    print("="*100)

    features = ["vpr", "intensity_mean", "intensity_std", "sri"]

    for feature in features:
        print(f"\n{'='*100}")
        print(f"Feature: {feature.upper()}")
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

    features = ["vpr", "intensity_mean", "intensity_std", "sri"]
    feature_labels = {
        "vpr": "VPR (Vertical Power Ratio)",
        "intensity_mean": "Intensity Mean",
        "intensity_std": "Intensity Std",
        "sri": "SRI (Stripe Regularity Index)",
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
    """
    Compute separability metrics between normal and fault.

    Uses Cohen's d effect size:
    d = (mean_fault - mean_normal) / pooled_std
    """
    separability = {}
    features = ["vpr", "intensity_mean", "intensity_std", "sri"]

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

            # AUC approximation using rank-biserial correlation
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
    print("\n" + "="*100)
    print("SEPARABILITY ANALYSIS (Normal vs Fault)")
    print("="*100)
    cohens_label = "Cohen's d"
    print(f"{'Domain':<12} | {'Feature':<18} | {cohens_label:>12} | {'AUC':>10} | {'Direction':<15} | {'Interpretation':<20}")
    print("-"*100)

    for domain, features in separability.items():
        for feature, metrics in features.items():
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

            print(f"{domain:<12} | {feature:<18} | {d:>12.4f} | {auc:>10.4f} | {direction:<15} | {interp:<20}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Feature EDA for HDMAP")
    parser.add_argument("--output-dir", type=str, default="results/feature_eda",
                        help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per split (for quick testing)")
    args = parser.parse_args()

    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("HDMAP Feature EDA: VPR, Intensity, SRI")
    print("="*60)
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

    # Create visualizations
    print("\nCreating visualizations...")
    create_violin_plots(all_results, output_dir)

    # Save results to JSON
    results_file = output_dir / "feature_eda_results.json"
    with open(results_file, "w") as f:
        # Remove raw data for JSON (too large)
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
    summary_file = output_dir / "FEATURE_EDA_SUMMARY.md"
    with open(summary_file, "w") as f:
        f.write("# Feature EDA Summary\n\n")
        f.write("## Features Analyzed\n\n")
        f.write("1. **VPR (Vertical Power Ratio)**: FFT-based vertical structure measure\n")
        f.write("2. **Intensity Mean**: Average brightness\n")
        f.write("3. **Intensity Std**: Brightness variation\n")
        f.write("4. **SRI (Stripe Regularity Index)**: Column-wise autocorrelation\n\n")

        f.write("## Separability Analysis (Domain C Focus)\n\n")
        if "domain_C" in separability:
            f.write("| Feature | Cohen's d | AUC | Direction | Interpretation |\n")
            f.write("|---------|-----------|-----|-----------|----------------|\n")
            for feature, metrics in separability["domain_C"].items():
                d = metrics["cohens_d"]
                auc = metrics["auc"]
                direction = metrics["direction"]
                abs_d = abs(d)
                if abs_d < 0.2:
                    interp = "Negligible"
                elif abs_d < 0.5:
                    interp = "Small"
                elif abs_d < 0.8:
                    interp = "Medium"
                else:
                    interp = "Large"
                f.write(f"| {feature} | {d:.4f} | {auc:.4f} | {direction} | {interp} |\n")

        f.write("\n## Recommendations\n\n")
        f.write("Based on the separability analysis, features with |Cohen's d| > 0.5 or AUC > 0.6 (or < 0.4) ")
        f.write("are good candidates for adaptive discarding.\n")

    print(f"Summary saved to: {summary_file}")
    print("\nEDA Complete!")


if __name__ == "__main__":
    main()
