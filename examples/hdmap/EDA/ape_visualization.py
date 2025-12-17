#!/usr/bin/env python3
"""
APE (Angular Power Entropy) Processing Visualization for HDMAP Dataset.

This script visualizes the step-by-step APE computation process:
1. Original image
2. Grayscale + DC removal
3. Hann windowed
4. 2D FFT Power Spectrum
5. Angular power distribution (polar plot)
6. Final APE value

Shows side-by-side comparison of Normal vs Anomaly samples.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.gridspec import GridSpec

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Dataset paths
HDMAP_PNG_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png")
OUTPUT_DIR = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/examples/hdmap/EDA/results/ape")


def angular_power_entropy_2d_with_intermediates(
    x,
    *,
    num_angle_bins: int = 36,
    eps: float = 1e-12,
    remove_dc: bool = True,
    apply_hann_window: bool = True,
    r_min_ratio: float = 0.05,
    r_max_ratio: float = 0.95,
) -> dict:
    """
    Compute APE and return all intermediate results for visualization.
    """
    arr = np.asarray(x)

    # Handle shape
    if arr.ndim == 3:
        if arr.shape[2] in [1, 3, 4]:  # (H, W, C)
            arr = arr.mean(axis=2)
        else:  # (C, H, W)
            arr = arr.mean(axis=0)

    original = arr.copy()
    arr = arr.astype(np.float64, copy=False)
    H, W = arr.shape

    # Step 1: Grayscale
    grayscale = arr.copy()
    dc_value = arr.mean()

    # Step 2: Remove DC
    if remove_dc:
        arr = arr - dc_value
    dc_removed = arr.copy()

    # Step 3: Apply Hann window
    if apply_hann_window:
        wy = np.hanning(H).reshape(H, 1)
        wx = np.hanning(W).reshape(1, W)
        window = wy * wx
        arr = arr * window
    else:
        window = np.ones((H, W))
    windowed = arr.copy()

    # Step 4: FFT power spectrum
    F = np.fft.fft2(arr)
    P = np.abs(F) ** 2
    P_shifted = np.fft.fftshift(P)
    P_log = np.log10(P_shifted + eps)

    # Coordinate system
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dy = yy - cy
    dx = xx - cx

    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    # Radial mask
    r_max = r.max()
    r_min = r_min_ratio * r_max
    r_upper = r_max_ratio * r_max
    radial_mask = (r >= r_min) & (r <= r_upper)

    # Angular binning
    theta_masked = theta[radial_mask]
    power_masked = P_shifted[radial_mask]

    # Map angle to [0, 2pi)
    theta_mapped = (theta_masked + 2 * np.pi) % (2 * np.pi)

    angle_bins = np.linspace(0.0, 2 * np.pi, num_angle_bins + 1)
    angle_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
    angle_idx = np.digitize(theta_mapped, angle_bins) - 1
    angle_idx = np.clip(angle_idx, 0, num_angle_bins - 1)

    # Aggregate power per angle bin
    angular_energy = np.zeros(num_angle_bins, dtype=np.float64)
    np.add.at(angular_energy, angle_idx, power_masked)

    total_energy = angular_energy.sum()
    if total_energy > eps:
        p = angular_energy / total_energy
        H_theta = -np.sum(p * np.log(p + eps))
        H_max = np.log(num_angle_bins)
        ape = H_theta / H_max
    else:
        p = np.zeros(num_angle_bins)
        ape = 0.0

    ape = float(np.clip(ape, 0.0, 1.0))

    return {
        "original": original,
        "grayscale": grayscale,
        "dc_value": dc_value,
        "dc_removed": dc_removed,
        "window": window,
        "windowed": windowed,
        "power_spectrum": P_shifted,
        "power_spectrum_log": P_log,
        "radial_mask": radial_mask,
        "angle_centers": angle_centers,
        "angular_energy": angular_energy,
        "angular_prob": p,
        "ape": ape,
        "num_angle_bins": num_angle_bins,
    }


def load_image(path: Path) -> np.ndarray:
    """Load image as numpy array."""
    img = Image.open(path)
    return np.array(img)


def visualize_ape_process(
    image_path: Path,
    output_path: Path,
    num_angle_bins: int = 36,
    title_prefix: str = "",
):
    """Visualize the complete APE computation process for a single image."""
    img = load_image(image_path)
    results = angular_power_entropy_2d_with_intermediates(img, num_angle_bins=num_angle_bins)

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    # Row 1: Spatial domain processing
    ax1 = fig.add_subplot(gs[0, 0])
    if len(results["original"].shape) == 2:
        ax1.imshow(results["original"], cmap="gray")
    else:
        ax1.imshow(results["original"])
    ax1.set_title("1. Original Image", fontsize=11, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(results["dc_removed"], cmap="RdBu_r")
    ax2.set_title(f"2. DC Removed\n(mean={results['dc_value']:.2f})", fontsize=11, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(results["windowed"], cmap="gray")
    ax3.set_title("3. Hann Windowed", fontsize=11, fontweight="bold")
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(results["power_spectrum_log"], cmap="hot")
    ax4.set_title("4. Power Spectrum (log10)", fontsize=11, fontweight="bold")
    ax4.axis("off")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # Row 2: Angular analysis
    ax5 = fig.add_subplot(gs[1, 0])
    # Show radial mask on power spectrum
    masked_power = results["power_spectrum_log"].copy()
    masked_power[~results["radial_mask"]] = masked_power.min()
    ax5.imshow(masked_power, cmap="hot")
    ax5.set_title("5. Radial Mask Applied\n(ignore DC & corners)", fontsize=11, fontweight="bold")
    ax5.axis("off")

    # Polar plot of angular energy
    ax6 = fig.add_subplot(gs[1, 1], projection="polar")
    angles = results["angle_centers"]
    energy = results["angular_energy"]
    energy_norm = energy / (energy.max() + 1e-12)
    ax6.bar(angles, energy_norm, width=2*np.pi/len(angles), alpha=0.7, color="steelblue", edgecolor="navy")
    ax6.set_title("6. Angular Power\n(polar view)", fontsize=11, fontweight="bold")
    ax6.set_ylim(0, 1.1)

    # Linear plot of angular probability
    ax7 = fig.add_subplot(gs[1, 2])
    angles_deg = np.degrees(results["angle_centers"])
    ax7.bar(angles_deg, results["angular_prob"], width=360/len(angles_deg), alpha=0.7, color="steelblue", edgecolor="navy")
    ax7.axhline(1/len(angles_deg), color="red", linestyle="--", label=f"Uniform ({1/len(angles_deg):.3f})")
    ax7.set_xlabel("Angle (degrees)", fontsize=10)
    ax7.set_ylabel("Probability", fontsize=10)
    ax7.set_title("7. Angular Probability\n(linear view)", fontsize=11, fontweight="bold")
    ax7.legend(fontsize=8)
    ax7.set_xlim(0, 360)

    # Entropy calculation
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis("off")

    p = results["angular_prob"]
    eps = 1e-12
    individual_entropy = -p * np.log(p + eps)

    # Show top contributing angles
    top_idx = np.argsort(individual_entropy)[-5:][::-1]
    entropy_text = "Entropy Contribution by Angle:\n"
    for idx in top_idx:
        entropy_text += f"  {np.degrees(results['angle_centers'][idx]):.0f}°: {individual_entropy[idx]:.4f}\n"

    ax8.text(0.1, 0.5, entropy_text, fontsize=11, fontfamily="monospace",
             verticalalignment="center", transform=ax8.transAxes,
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax8.set_title("8. Entropy Details", fontsize=11, fontweight="bold")

    # Row 3: Results summary
    ax9 = fig.add_subplot(gs[2, :2])
    ax9.axis("off")

    H_theta = -np.sum(p * np.log(p + eps))
    H_max = np.log(len(p))

    summary_text = f"""
    APE Computation Summary
    {'='*45}

    Number of angle bins: {results['num_angle_bins']}
    Angular Entropy (H): {H_theta:.4f}
    Max Entropy (H_max): {H_max:.4f} (= ln({results['num_angle_bins']}))

    APE = H / H_max = {results['ape']:.4f}

    Interpretation:
    - APE close to 0: Strong directional pattern
      (energy concentrated in few angles → overfit risk)
    - APE close to 1: Isotropic pattern
      (energy spread uniformly → less overfit risk)
    """
    ax9.text(0.05, 0.5, summary_text, fontsize=11, fontfamily="monospace",
             verticalalignment="center", transform=ax9.transAxes,
             bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5))

    # APE gauge
    ax10 = fig.add_subplot(gs[2, 2:])
    ape_value = results["ape"]

    # Create a horizontal bar gauge
    ax10.barh(0, ape_value, height=0.5, color="steelblue", alpha=0.8)
    ax10.barh(0, 1 - ape_value, left=ape_value, height=0.5, color="lightgray", alpha=0.5)
    ax10.axvline(ape_value, color="red", linewidth=3)
    ax10.set_xlim(0, 1)
    ax10.set_ylim(-0.5, 0.5)
    ax10.set_xlabel("APE Value", fontsize=12)
    ax10.set_title(f"APE = {ape_value:.4f}", fontsize=14, fontweight="bold", color="red")

    # Add labels
    ax10.text(0.05, -0.3, "Directional\n(Overfit Risk)", fontsize=10, ha="center")
    ax10.text(0.95, -0.3, "Isotropic\n(Less Risk)", fontsize=10, ha="center")
    ax10.set_yticks([])

    # Main title
    file_name = image_path.name
    plt.suptitle(f"{title_prefix}APE Processing: {file_name}", fontsize=14, fontweight="bold", y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved visualization to {output_path}")

    return results["ape"]


def visualize_comparison(
    normal_path: Path,
    anomaly_path: Path,
    output_path: Path,
    domain: str,
    num_angle_bins: int = 36,
):
    """Visualize side-by-side comparison of Normal vs Anomaly."""
    normal_img = load_image(normal_path)
    anomaly_img = load_image(anomaly_path)

    normal_results = angular_power_entropy_2d_with_intermediates(normal_img, num_angle_bins=num_angle_bins)
    anomaly_results = angular_power_entropy_2d_with_intermediates(anomaly_img, num_angle_bins=num_angle_bins)

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 5, figure=fig, hspace=0.3, wspace=0.3)

    # Normal row
    ax_n1 = fig.add_subplot(gs[0, 0])
    ax_n1.imshow(normal_results["original"], cmap="gray" if normal_results["original"].ndim == 2 else None)
    ax_n1.set_title("Original", fontsize=10)
    ax_n1.axis("off")

    ax_n2 = fig.add_subplot(gs[0, 1])
    ax_n2.imshow(normal_results["power_spectrum_log"], cmap="hot")
    ax_n2.set_title("Power Spectrum", fontsize=10)
    ax_n2.axis("off")

    ax_n3 = fig.add_subplot(gs[0, 2], projection="polar")
    angles = normal_results["angle_centers"]
    energy_norm = normal_results["angular_energy"] / (normal_results["angular_energy"].max() + 1e-12)
    ax_n3.bar(angles, energy_norm, width=2*np.pi/len(angles), alpha=0.7, color="steelblue")
    ax_n3.set_title("Angular Power", fontsize=10)
    ax_n3.set_ylim(0, 1.1)

    ax_n4 = fig.add_subplot(gs[0, 3])
    ax_n4.bar(np.degrees(angles), normal_results["angular_prob"], width=10, alpha=0.7, color="steelblue")
    ax_n4.axhline(1/len(angles), color="red", linestyle="--", alpha=0.5)
    ax_n4.set_xlabel("Angle (°)", fontsize=9)
    ax_n4.set_title("Probability", fontsize=10)
    ax_n4.set_xlim(0, 360)

    ax_n5 = fig.add_subplot(gs[0, 4])
    ax_n5.text(0.5, 0.5, f"APE = {normal_results['ape']:.4f}",
               fontsize=18, ha="center", va="center", fontweight="bold", color="blue",
               transform=ax_n5.transAxes)
    ax_n5.set_title("Result", fontsize=10)
    ax_n5.axis("off")
    ax_n5.set_facecolor("lightblue")

    # Anomaly row
    ax_a1 = fig.add_subplot(gs[1, 0])
    ax_a1.imshow(anomaly_results["original"], cmap="gray" if anomaly_results["original"].ndim == 2 else None)
    ax_a1.set_title("Original", fontsize=10)
    ax_a1.axis("off")

    ax_a2 = fig.add_subplot(gs[1, 1])
    ax_a2.imshow(anomaly_results["power_spectrum_log"], cmap="hot")
    ax_a2.set_title("Power Spectrum", fontsize=10)
    ax_a2.axis("off")

    ax_a3 = fig.add_subplot(gs[1, 2], projection="polar")
    energy_norm_a = anomaly_results["angular_energy"] / (anomaly_results["angular_energy"].max() + 1e-12)
    ax_a3.bar(angles, energy_norm_a, width=2*np.pi/len(angles), alpha=0.7, color="indianred")
    ax_a3.set_title("Angular Power", fontsize=10)
    ax_a3.set_ylim(0, 1.1)

    ax_a4 = fig.add_subplot(gs[1, 3])
    ax_a4.bar(np.degrees(angles), anomaly_results["angular_prob"], width=10, alpha=0.7, color="indianred")
    ax_a4.axhline(1/len(angles), color="red", linestyle="--", alpha=0.5)
    ax_a4.set_xlabel("Angle (°)", fontsize=9)
    ax_a4.set_title("Probability", fontsize=10)
    ax_a4.set_xlim(0, 360)

    ax_a5 = fig.add_subplot(gs[1, 4])
    ax_a5.text(0.5, 0.5, f"APE = {anomaly_results['ape']:.4f}",
               fontsize=18, ha="center", va="center", fontweight="bold", color="red",
               transform=ax_a5.transAxes)
    ax_a5.set_title("Result", fontsize=10)
    ax_a5.axis("off")
    ax_a5.set_facecolor("lightyellow")

    # Row labels
    fig.text(0.02, 0.72, "NORMAL", fontsize=12, fontweight="bold", color="blue", rotation=90, va="center")
    fig.text(0.02, 0.28, "ANOMALY", fontsize=12, fontweight="bold", color="red", rotation=90, va="center")

    # Difference
    ape_diff = anomaly_results["ape"] - normal_results["ape"]
    if ape_diff > 0:
        diff_text = f"APE Difference: {ape_diff:+.4f} (Anomaly more isotropic)"
    else:
        diff_text = f"APE Difference: {ape_diff:+.4f} (Anomaly more directional)"

    plt.suptitle(f"{domain}: Normal vs Anomaly APE Comparison\n{diff_text}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0.03, 0, 1, 0.94])

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved comparison to {output_path}")


def get_sample_images(domain: str, n_samples: int = 3) -> dict:
    """Get sample image paths from a domain."""
    domain_path = HDMAP_PNG_ROOT / domain

    train_good = sorted((domain_path / "train" / "good").glob("*.png"))[:n_samples]
    test_good = sorted((domain_path / "test" / "good").glob("*.png"))[:n_samples]
    test_fault = sorted((domain_path / "test" / "fault").glob("*.png"))[:n_samples]

    return {
        "train_good": train_good,
        "test_good": test_good,
        "test_fault": test_fault,
    }


def main():
    parser = argparse.ArgumentParser(description="APE Process Visualization")
    parser.add_argument(
        "--domains",
        type=str,
        default="domain_A,domain_B,domain_C,domain_D",
        help="Comma-separated list of domains",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=3,
        help="Number of sample images per category",
    )
    parser.add_argument(
        "--num-angle-bins",
        type=int,
        default=36,
        help="Number of angular bins",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory",
    )

    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for domain in domains:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {domain}")
        logger.info(f"{'='*60}")

        domain_output = output_dir / domain
        domain_output.mkdir(parents=True, exist_ok=True)

        samples = get_sample_images(domain, args.n_samples)

        # Visualize individual samples
        for category, paths in samples.items():
            for i, path in enumerate(paths):
                output_path = domain_output / f"{category}_{i+1}_process.png"
                ape = visualize_ape_process(
                    path, output_path,
                    num_angle_bins=args.num_angle_bins,
                    title_prefix=f"[{domain}] {category}: "
                )
                logger.info(f"  {category}_{i+1}: APE = {ape:.4f}")

        # Visualize comparisons
        if samples["train_good"] and samples["test_fault"]:
            for i in range(min(len(samples["train_good"]), len(samples["test_fault"]), args.n_samples)):
                output_path = domain_output / f"comparison_{i+1}.png"
                visualize_comparison(
                    samples["train_good"][i],
                    samples["test_fault"][i],
                    output_path,
                    domain,
                    num_angle_bins=args.num_angle_bins,
                )

    logger.info(f"\nVisualization complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
