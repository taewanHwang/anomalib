#!/usr/bin/env python3
"""
Angular Power Entropy (APE) EDA for HDMAP Dataset.

APE measures directional concentration of power in the frequency domain:
- Low APE → Strong directional pattern (energy concentrated in specific angles)
- High APE → Isotropic/noisy pattern (energy spread across all angles)

This is conceptually similar to Orientation Entropy but computed in frequency domain,
making it more robust for industrial signal images.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Dataset paths
HDMAP_PNG_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png")
OUTPUT_DIR = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/examples/hdmap/EDA/results/ape")


def angular_power_entropy_2d(
    x,
    *,
    num_angle_bins: int = 36,
    eps: float = 1e-12,
    remove_dc: bool = True,
    apply_hann_window: bool = True,
    r_min_ratio: float = 0.05,
    r_max_ratio: float = 0.95,
) -> float:
    """
    Compute Angular Power Entropy (APE) from a 2D input.

    Args:
        x: 2D array (H, W) or 3D (C, H, W) or (H, W, C). If 3D, averaged over channels.
        num_angle_bins: Number of angular bins (e.g., 36 → 10° resolution).
        eps: Numerical stability constant.
        remove_dc: Remove spatial mean before FFT.
        apply_hann_window: Apply 2D Hann window before FFT.
        r_min_ratio: Ignore very low frequencies (center), ratio of max radius.
        r_max_ratio: Ignore extreme high frequencies, ratio of max radius.

    Returns:
        ape (float): Angular Power Entropy normalized to [0, 1].
    """
    arr = np.asarray(x)

    # Handle shape: (H, W), (C, H, W), or (H, W, C)
    if arr.ndim == 3:
        if arr.shape[2] in [1, 3, 4]:  # (H, W, C)
            arr = arr.mean(axis=2)
        else:  # (C, H, W)
            arr = arr.mean(axis=0)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D or 3D input, got shape {arr.shape}")

    arr = arr.astype(np.float64, copy=False)
    H, W = arr.shape

    # Remove DC
    if remove_dc:
        arr = arr - arr.mean()

    # Apply Hann window
    if apply_hann_window:
        wy = np.hanning(H).reshape(H, 1)
        wx = np.hanning(W).reshape(1, W)
        arr = arr * (wy * wx)

    # FFT power spectrum
    F = np.fft.fft2(arr)
    P = np.abs(F) ** 2
    P = np.fft.fftshift(P)

    # Coordinate system
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dy = yy - cy
    dx = xx - cx

    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)  # [-pi, pi]

    # Radial mask (ignore DC & extreme corners)
    r_max = r.max()
    r_min = r_min_ratio * r_max
    r_upper = r_max_ratio * r_max
    radial_mask = (r >= r_min) & (r <= r_upper)

    # Angular binning
    theta_masked = theta[radial_mask]
    power_masked = P[radial_mask]

    # Map angle to [0, 2pi)
    theta_masked = (theta_masked + 2 * np.pi) % (2 * np.pi)

    angle_bins = np.linspace(0.0, 2 * np.pi, num_angle_bins + 1)
    angle_idx = np.digitize(theta_masked, angle_bins) - 1
    angle_idx = np.clip(angle_idx, 0, num_angle_bins - 1)

    # Aggregate power per angle bin
    angular_energy = np.zeros(num_angle_bins, dtype=np.float64)
    np.add.at(angular_energy, angle_idx, power_masked)

    total_energy = angular_energy.sum()
    if total_energy <= eps:
        return 0.0

    p = angular_energy / (total_energy + eps)

    # Shannon entropy (normalized)
    H_theta = -np.sum(p * np.log(p + eps))
    H_max = np.log(num_angle_bins)
    ape = H_theta / H_max

    # Clamp
    ape = float(np.clip(ape, 0.0, 1.0))
    return ape


def load_image(path: Path) -> np.ndarray:
    """Load image as numpy array."""
    img = Image.open(path)
    return np.array(img)


def compute_ape_for_directory(
    dir_path: Path,
    num_angle_bins: int = 36,
    limit: Optional[int] = None,
) -> list[dict]:
    """Compute APE for all images in a directory."""
    results = []
    image_files = sorted(dir_path.glob("*.png"))

    if limit:
        image_files = image_files[:limit]

    for img_path in tqdm(image_files, desc=f"Processing {dir_path.name}"):
        try:
            img = load_image(img_path)
            ape = angular_power_entropy_2d(img, num_angle_bins=num_angle_bins)
            results.append({
                "file": img_path.name,
                "ape": ape,
            })
        except Exception as e:
            logger.warning(f"Failed to process {img_path}: {e}")

    return results


def analyze_domain(
    domain: str,
    num_angle_bins: int = 36,
    limit: Optional[int] = None,
) -> dict:
    """Analyze APE for a single domain."""
    domain_path = HDMAP_PNG_ROOT / domain

    # Paths
    train_good = domain_path / "train" / "good"
    test_good = domain_path / "test" / "good"
    test_fault = domain_path / "test" / "fault"

    results = {
        "domain": domain,
        "num_angle_bins": num_angle_bins,
        "train_good": [],
        "test_good": [],
        "test_fault": [],
    }

    # Compute APE for each category
    if train_good.exists():
        results["train_good"] = compute_ape_for_directory(train_good, num_angle_bins, limit)
    if test_good.exists():
        results["test_good"] = compute_ape_for_directory(test_good, num_angle_bins, limit)
    if test_fault.exists():
        results["test_fault"] = compute_ape_for_directory(test_fault, num_angle_bins, limit)

    return results


def compute_statistics(values: list[float]) -> dict:
    """Compute statistics for a list of values."""
    if not values:
        return {}
    arr = np.array(values)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def plot_ape_distributions(all_results: dict, output_path: Path):
    """Plot APE distributions for all domains."""
    domains = list(all_results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    colors = {"train_good": "blue", "test_good": "green", "test_fault": "red"}
    labels = {"train_good": "Train Normal", "test_good": "Test Normal", "test_fault": "Test Anomaly"}

    for idx, domain in enumerate(domains):
        ax = axes[idx]
        data = all_results[domain]

        for category in ["train_good", "test_good", "test_fault"]:
            if data.get(category):
                ape_values = [r["ape"] for r in data[category]]
                if ape_values:
                    ax.hist(
                        ape_values,
                        bins=50,
                        alpha=0.6,
                        color=colors[category],
                        label=f"{labels[category]} (n={len(ape_values)})",
                        density=True,
                    )

        ax.set_xlabel("APE (Angular Power Entropy)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{domain}", fontsize=13, fontweight="bold")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

    plt.suptitle("APE Distribution by Domain (Normal vs Anomaly)\nLow APE = Strong directional pattern, High APE = Isotropic",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved distribution plot to {output_path}")


def plot_ape_boxplot(all_results: dict, output_path: Path):
    """Plot APE boxplot comparison."""
    fig, ax = plt.subplots(figsize=(14, 8))

    data_to_plot = []
    labels = []
    colors_list = []

    for domain in all_results.keys():
        data = all_results[domain]

        # Normal (train + test)
        normal_ape = []
        if data.get("train_good"):
            normal_ape.extend([r["ape"] for r in data["train_good"]])
        if data.get("test_good"):
            normal_ape.extend([r["ape"] for r in data["test_good"]])

        # Anomaly
        anomaly_ape = []
        if data.get("test_fault"):
            anomaly_ape.extend([r["ape"] for r in data["test_fault"]])

        if normal_ape:
            data_to_plot.append(normal_ape)
            labels.append(f"{domain}\nNormal")
            colors_list.append("steelblue")

        if anomaly_ape:
            data_to_plot.append(anomaly_ape)
            labels.append(f"{domain}\nAnomaly")
            colors_list.append("indianred")

    bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)

    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("APE (Angular Power Entropy)", fontsize=12)
    ax.set_title("APE Comparison: Normal vs Anomaly by Domain\n(Low = Directional, High = Isotropic)",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved boxplot to {output_path}")


def print_summary_table(all_results: dict):
    """Print summary statistics table."""
    print("\n" + "=" * 100)
    print("APE (Angular Power Entropy) EDA Summary Statistics")
    print("=" * 100)
    print(f"{'Domain':<12} | {'Category':<12} | {'Count':>6} | {'Mean':>8} | {'Std':>8} | {'P10':>8} | {'P50':>8} | {'P90':>8}")
    print("-" * 100)

    for domain in all_results.keys():
        data = all_results[domain]

        for category in ["train_good", "test_good", "test_fault"]:
            if data.get(category):
                ape_values = [r["ape"] for r in data[category]]
                stats = compute_statistics(ape_values)

                cat_label = {"train_good": "Train Norm", "test_good": "Test Norm", "test_fault": "Test Anom"}[category]
                print(
                    f"{domain:<12} | {cat_label:<12} | {stats['count']:>6} | "
                    f"{stats['mean']:>8.4f} | {stats['std']:>8.4f} | "
                    f"{stats['p10']:>8.4f} | {stats['p50']:>8.4f} | {stats['p90']:>8.4f}"
                )
        print("-" * 100)

    # Print separation analysis
    print("\n" + "=" * 100)
    print("Normal vs Anomaly Separation Analysis")
    print("=" * 100)
    print(f"{'Domain':<12} | {'Normal Mean':>12} | {'Anomaly Mean':>12} | {'Diff':>10} | {'Cohen d':>10} | {'Judgment':>12}")
    print("-" * 100)

    for domain in all_results.keys():
        data = all_results[domain]

        # Combine normal
        normal_ape = []
        if data.get("train_good"):
            normal_ape.extend([r["ape"] for r in data["train_good"]])
        if data.get("test_good"):
            normal_ape.extend([r["ape"] for r in data["test_good"]])

        # Anomaly
        anomaly_ape = []
        if data.get("test_fault"):
            anomaly_ape.extend([r["ape"] for r in data["test_fault"]])

        if normal_ape and anomaly_ape:
            normal_mean = np.mean(normal_ape)
            anomaly_mean = np.mean(anomaly_ape)
            normal_std = np.std(normal_ape)
            anomaly_std = np.std(anomaly_ape)

            diff = anomaly_mean - normal_mean
            # Cohen's d for separation
            pooled_std = np.sqrt((normal_std**2 + anomaly_std**2) / 2)
            cohens_d = diff / pooled_std if pooled_std > 0 else 0

            if abs(cohens_d) > 0.8:
                sep_label = "Strong"
            elif abs(cohens_d) > 0.5:
                sep_label = "Medium"
            else:
                sep_label = "Weak"

            print(
                f"{domain:<12} | {normal_mean:>12.4f} | {anomaly_mean:>12.4f} | "
                f"{diff:>+10.4f} | {cohens_d:>+10.2f} | {sep_label:>12}"
            )

    print("=" * 100)
    print("\nInterpretation:")
    print("  - Low APE: Strong directional pattern (overfit risk for adaptive dropout)")
    print("  - High APE: Isotropic/complex pattern (less overfit risk)")
    print("  - Positive diff: Anomaly has higher APE (more isotropic than normal)")
    print("  - Negative diff: Anomaly has lower APE (more directional than normal)")


def main():
    parser = argparse.ArgumentParser(description="APE EDA for HDMAP Dataset")
    parser.add_argument(
        "--domains",
        type=str,
        default="domain_A,domain_B,domain_C,domain_D",
        help="Comma-separated list of domains to analyze",
    )
    parser.add_argument(
        "--num-angle-bins",
        type=int,
        default=36,
        help="Number of angular bins (36 = 10° resolution)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images per category (for quick testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for plots and results",
    )

    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Analyzing domains: {domains}")
    logger.info(f"Number of angle bins: {args.num_angle_bins}")
    logger.info(f"Output dir: {output_dir}")

    # Analyze all domains
    all_results = {}
    for domain in domains:
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing {domain}")
        logger.info(f"{'='*60}")

        results = analyze_domain(domain, num_angle_bins=args.num_angle_bins, limit=args.limit)
        all_results[domain] = results

    # Print summary
    print_summary_table(all_results)

    # Save results
    results_file = output_dir / "ape_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved results to {results_file}")

    # Generate plots
    plot_ape_distributions(all_results, output_dir / "ape_distributions.png")
    plot_ape_boxplot(all_results, output_dir / "ape_boxplot.png")

    logger.info(f"\nEDA complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
