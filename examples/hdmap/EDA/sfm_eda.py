#!/usr/bin/env python3
"""
Spectral Flatness Measure (SFM) EDA for HDMAP Dataset.

This script analyzes the SFM distribution across all 4 domains (A, B, C, D)
to evaluate if SFM is a suitable feature for adaptive dropout.

Key idea:
- Regular patterns (tonal peaks) → SFM ↓ (low)
- Noise/defects → SFM ↑ (high)
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
OUTPUT_DIR = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/examples/hdmap/EDA/sfm_results")


def spectral_flatness_2d(
    x,
    *,
    eps: float = 1e-12,
    remove_dc: bool = True,
    apply_hann_window: bool = True,
    use_log_power: bool = False,
    crop_center: Optional[float] = None,
) -> float:
    """
    Compute 2D Spectral Flatness Measure (SFM) from a 2D input.

    Args:
        x: 2D array-like (H, W) or 3D (C, H, W). If 3D, it will be converted to grayscale by mean over C.
        eps: Small constant to avoid log(0) and division by zero.
        remove_dc: If True, subtract mean from spatial domain input.
        apply_hann_window: If True, apply separable Hann window in spatial domain to reduce spectral leakage.
        use_log_power: If True, compute SFM on log-power (less sensitive to huge peaks). Usually False is standard SFM.
        crop_center: Optional. If set (0<crop_center<=1), keeps only a centered square region of the spectrum
                     (e.g., 0.5 keeps the central 50% in both axes). Useful to ignore extreme high-freq corners.

    Returns:
        sfm (float): Spectral flatness in [0, 1] approximately.
    """
    arr = np.asarray(x)

    # Handle shapes: (H, W) or (C, H, W)
    if arr.ndim == 3:
        # Convert to grayscale-like by averaging channels
        arr = arr.mean(axis=0)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D (H,W) or 3D (C,H,W) input, got shape {arr.shape}")

    arr = arr.astype(np.float64, copy=False)

    H, W = arr.shape

    # Remove DC (mean)
    if remove_dc:
        arr = arr - arr.mean()

    # Apply Hann window (spatial)
    if apply_hann_window:
        wy = np.hanning(H).reshape(H, 1)
        wx = np.hanning(W).reshape(1, W)
        window = wy * wx
        arr = arr * window

    # 2D FFT and power spectrum
    F = np.fft.fft2(arr)
    P = np.abs(F) ** 2  # power

    # Shift to center (optional cropping convenience)
    P = np.fft.fftshift(P)

    # Optional: crop centered region to focus on mid/low frequencies
    if crop_center is not None:
        if not (0.0 < crop_center <= 1.0):
            raise ValueError("crop_center must be in (0, 1].")
        ch = int(round(H * crop_center))
        cw = int(round(W * crop_center))
        ch = max(1, min(ch, H))
        cw = max(1, min(cw, W))
        y0 = (H - ch) // 2
        x0 = (W - cw) // 2
        P = P[y0:y0 + ch, x0:x0 + cw]

    # Flatten
    P_flat = P.reshape(-1)

    # Avoid zeros
    P_flat = np.maximum(P_flat, eps)

    # Optionally use log-power (robust variant)
    if use_log_power:
        P_flat = np.log(P_flat + eps)

    # SFM = geometric_mean / arithmetic_mean
    gm = np.exp(np.mean(np.log(P_flat + eps)))
    am = np.mean(P_flat)

    sfm = float(gm / (am + eps))

    # Numerical safety: clamp
    if sfm < 0.0:
        sfm = 0.0
    # Sometimes >1 can occur due to log-power option or numerical quirks; clamp gently
    if sfm > 1.0 and not use_log_power:
        sfm = 1.0

    return sfm


def load_image(path: Path) -> np.ndarray:
    """Load image as numpy array (H, W, C) or (H, W)."""
    img = Image.open(path)
    return np.array(img)


def compute_sfm_for_directory(
    dir_path: Path,
    crop_center: Optional[float] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """Compute SFM for all images in a directory."""
    results = []
    image_files = sorted(dir_path.glob("*.png"))

    if limit:
        image_files = image_files[:limit]

    for img_path in tqdm(image_files, desc=f"Processing {dir_path.name}"):
        try:
            img = load_image(img_path)
            sfm = spectral_flatness_2d(img, crop_center=crop_center)
            results.append({
                "file": img_path.name,
                "sfm": sfm,
            })
        except Exception as e:
            logger.warning(f"Failed to process {img_path}: {e}")

    return results


def analyze_domain(
    domain: str,
    crop_center: Optional[float] = None,
    limit: Optional[int] = None,
) -> dict:
    """Analyze SFM for a single domain."""
    domain_path = HDMAP_PNG_ROOT / domain

    # Paths
    train_good = domain_path / "train" / "good"
    test_good = domain_path / "test" / "good"
    test_fault = domain_path / "test" / "fault"

    results = {
        "domain": domain,
        "crop_center": crop_center,
        "train_good": [],
        "test_good": [],
        "test_fault": [],
    }

    # Compute SFM for each category
    if train_good.exists():
        results["train_good"] = compute_sfm_for_directory(train_good, crop_center, limit)
    if test_good.exists():
        results["test_good"] = compute_sfm_for_directory(test_good, crop_center, limit)
    if test_fault.exists():
        results["test_fault"] = compute_sfm_for_directory(test_fault, crop_center, limit)

    return results


def compute_statistics(sfm_values: list[float]) -> dict:
    """Compute statistics for a list of SFM values."""
    if not sfm_values:
        return {}
    arr = np.array(sfm_values)
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


def plot_sfm_distributions(all_results: dict, output_path: Path):
    """Plot SFM distributions for all domains."""
    domains = list(all_results.keys())
    n_domains = len(domains)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    colors = {"train_good": "blue", "test_good": "green", "test_fault": "red"}
    labels = {"train_good": "Train Normal", "test_good": "Test Normal", "test_fault": "Test Anomaly"}

    for idx, domain in enumerate(domains):
        ax = axes[idx]
        data = all_results[domain]

        for category in ["train_good", "test_good", "test_fault"]:
            if data.get(category):
                sfm_values = [r["sfm"] for r in data[category]]
                if sfm_values:
                    ax.hist(
                        sfm_values,
                        bins=50,
                        alpha=0.6,
                        color=colors[category],
                        label=f"{labels[category]} (n={len(sfm_values)})",
                        density=True,
                    )

        ax.set_xlabel("SFM (Spectral Flatness)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{domain}", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.suptitle("SFM Distribution by Domain (Normal vs Anomaly)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved distribution plot to {output_path}")


def plot_sfm_boxplot(all_results: dict, output_path: Path):
    """Plot SFM boxplot comparison."""
    fig, ax = plt.subplots(figsize=(14, 8))

    data_to_plot = []
    labels = []
    colors_list = []

    for domain in all_results.keys():
        data = all_results[domain]

        # Normal (train + test)
        normal_sfm = []
        if data.get("train_good"):
            normal_sfm.extend([r["sfm"] for r in data["train_good"]])
        if data.get("test_good"):
            normal_sfm.extend([r["sfm"] for r in data["test_good"]])

        # Anomaly
        anomaly_sfm = []
        if data.get("test_fault"):
            anomaly_sfm.extend([r["sfm"] for r in data["test_fault"]])

        if normal_sfm:
            data_to_plot.append(normal_sfm)
            labels.append(f"{domain}\nNormal")
            colors_list.append("steelblue")

        if anomaly_sfm:
            data_to_plot.append(anomaly_sfm)
            labels.append(f"{domain}\nAnomaly")
            colors_list.append("indianred")

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("SFM (Spectral Flatness)", fontsize=12)
    ax.set_title("SFM Comparison: Normal vs Anomaly by Domain", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Add horizontal line at typical threshold
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Reference (0.5)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved boxplot to {output_path}")


def print_summary_table(all_results: dict):
    """Print summary statistics table."""
    print("\n" + "=" * 100)
    print("SFM EDA Summary Statistics")
    print("=" * 100)
    print(f"{'Domain':<12} | {'Category':<12} | {'Count':>6} | {'Mean':>8} | {'Std':>8} | {'P10':>8} | {'P50':>8} | {'P90':>8}")
    print("-" * 100)

    for domain in all_results.keys():
        data = all_results[domain]

        for category in ["train_good", "test_good", "test_fault"]:
            if data.get(category):
                sfm_values = [r["sfm"] for r in data[category]]
                stats = compute_statistics(sfm_values)

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
    print(f"{'Domain':<12} | {'Normal Mean':>12} | {'Anomaly Mean':>12} | {'Diff':>10} | {'Separation':>12}")
    print("-" * 100)

    for domain in all_results.keys():
        data = all_results[domain]

        # Combine normal
        normal_sfm = []
        if data.get("train_good"):
            normal_sfm.extend([r["sfm"] for r in data["train_good"]])
        if data.get("test_good"):
            normal_sfm.extend([r["sfm"] for r in data["test_good"]])

        # Anomaly
        anomaly_sfm = []
        if data.get("test_fault"):
            anomaly_sfm.extend([r["sfm"] for r in data["test_fault"]])

        if normal_sfm and anomaly_sfm:
            normal_mean = np.mean(normal_sfm)
            anomaly_mean = np.mean(anomaly_sfm)
            normal_std = np.std(normal_sfm)
            anomaly_std = np.std(anomaly_sfm)

            diff = anomaly_mean - normal_mean
            # Cohen's d for separation
            pooled_std = np.sqrt((normal_std**2 + anomaly_std**2) / 2)
            cohens_d = diff / pooled_std if pooled_std > 0 else 0

            sep_label = "Strong" if abs(cohens_d) > 0.8 else "Medium" if abs(cohens_d) > 0.5 else "Weak"

            print(
                f"{domain:<12} | {normal_mean:>12.4f} | {anomaly_mean:>12.4f} | "
                f"{diff:>+10.4f} | {cohens_d:>+8.2f} ({sep_label})"
            )

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="SFM EDA for HDMAP Dataset")
    parser.add_argument(
        "--domains",
        type=str,
        default="domain_A,domain_B,domain_C,domain_D",
        help="Comma-separated list of domains to analyze",
    )
    parser.add_argument(
        "--crop-center",
        type=float,
        default=None,
        help="Crop center region of spectrum (e.g., 0.7 for 70%)",
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
    logger.info(f"Crop center: {args.crop_center}")
    logger.info(f"Output dir: {output_dir}")

    # Analyze all domains
    all_results = {}
    for domain in domains:
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing {domain}")
        logger.info(f"{'='*60}")

        results = analyze_domain(domain, crop_center=args.crop_center, limit=args.limit)
        all_results[domain] = results

    # Print summary
    print_summary_table(all_results)

    # Save results
    results_file = output_dir / "sfm_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved results to {results_file}")

    # Generate plots
    plot_sfm_distributions(all_results, output_dir / "sfm_distributions.png")
    plot_sfm_boxplot(all_results, output_dir / "sfm_boxplot.png")

    logger.info(f"\nEDA complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
