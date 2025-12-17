#!/usr/bin/env python3
"""
SFM Processing Visualization for HDMAP Dataset.

This script visualizes the step-by-step SFM computation process:
1. Original image
2. Grayscale conversion
3. DC removal (mean subtraction)
4. Hann window application
5. 2D FFT
6. Power spectrum
7. Final SFM value

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
OUTPUT_DIR = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/examples/hdmap/EDA/sfm_results")


def spectral_flatness_2d_with_intermediates(
    x,
    *,
    eps: float = 1e-12,
    remove_dc: bool = True,
    apply_hann_window: bool = True,
    crop_center: Optional[float] = None,
) -> dict:
    """
    Compute 2D SFM and return all intermediate results for visualization.
    """
    arr = np.asarray(x)

    # Handle shapes: (H, W) or (C, H, W) or (H, W, C)
    if arr.ndim == 3:
        if arr.shape[2] in [1, 3, 4]:  # (H, W, C) format
            arr = arr.mean(axis=2)
        else:  # (C, H, W) format
            arr = arr.mean(axis=0)

    original = arr.copy()
    arr = arr.astype(np.float64, copy=False)
    H, W = arr.shape

    # Step 1: Grayscale (already done)
    grayscale = arr.copy()

    # Step 2: Remove DC (mean)
    dc_value = arr.mean()
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

    # Step 4: 2D FFT
    F = np.fft.fft2(arr)
    F_shifted = np.fft.fftshift(F)

    # Step 5: Power spectrum
    P = np.abs(F_shifted) ** 2
    P_log = np.log10(P + eps)  # For visualization

    # Step 6: Optional crop
    P_for_sfm = P.copy()
    if crop_center is not None:
        ch = int(round(H * crop_center))
        cw = int(round(W * crop_center))
        ch = max(1, min(ch, H))
        cw = max(1, min(cw, W))
        y0 = (H - ch) // 2
        x0 = (W - cw) // 2
        P_for_sfm = P[y0:y0 + ch, x0:x0 + cw]

    # Step 7: Compute SFM
    P_flat = P_for_sfm.reshape(-1)
    P_flat = np.maximum(P_flat, eps)

    gm = np.exp(np.mean(np.log(P_flat + eps)))
    am = np.mean(P_flat)
    sfm = float(gm / (am + eps))
    sfm = max(0.0, min(1.0, sfm))

    return {
        "original": original,
        "grayscale": grayscale,
        "dc_value": dc_value,
        "dc_removed": dc_removed,
        "window": window,
        "windowed": windowed,
        "fft_magnitude": np.abs(F_shifted),
        "power_spectrum": P,
        "power_spectrum_log": P_log,
        "geometric_mean": gm,
        "arithmetic_mean": am,
        "sfm": sfm,
    }


def load_image(path: Path) -> np.ndarray:
    """Load image as numpy array."""
    img = Image.open(path)
    return np.array(img)


def visualize_sfm_process(
    image_path: Path,
    output_path: Path,
    crop_center: Optional[float] = None,
    title_prefix: str = "",
):
    """Visualize the complete SFM computation process for a single image."""
    img = load_image(image_path)
    results = spectral_flatness_2d_with_intermediates(img, crop_center=crop_center)

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Row 1: Spatial domain processing
    ax1 = fig.add_subplot(gs[0, 0])
    if len(results["original"].shape) == 2:
        ax1.imshow(results["original"], cmap="gray")
    else:
        ax1.imshow(results["original"])
    ax1.set_title("1. Original Image", fontsize=11, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(results["grayscale"], cmap="gray")
    ax2.set_title(f"2. Grayscale\n(mean={results['dc_value']:.2f})", fontsize=11, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(results["dc_removed"], cmap="RdBu_r", vmin=-128, vmax=128)
    ax3.set_title("3. DC Removed\n(mean-subtracted)", fontsize=11, fontweight="bold")
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(results["window"], cmap="gray")
    ax4.set_title("4. Hann Window", fontsize=11, fontweight="bold")
    ax4.axis("off")

    # Row 2: Frequency domain
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(results["windowed"], cmap="gray")
    ax5.set_title("5. Windowed Image", fontsize=11, fontweight="bold")
    ax5.axis("off")

    ax6 = fig.add_subplot(gs[1, 1])
    fft_mag_log = np.log10(results["fft_magnitude"] + 1e-10)
    ax6.imshow(fft_mag_log, cmap="viridis")
    ax6.set_title("6. FFT Magnitude (log)", fontsize=11, fontweight="bold")
    ax6.axis("off")

    ax7 = fig.add_subplot(gs[1, 2])
    im7 = ax7.imshow(results["power_spectrum_log"], cmap="hot")
    ax7.set_title("7. Power Spectrum (log10)", fontsize=11, fontweight="bold")
    ax7.axis("off")
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

    # Row 2, Col 4: Power spectrum histogram
    ax8 = fig.add_subplot(gs[1, 3])
    power_flat = results["power_spectrum"].flatten()
    power_flat = power_flat[power_flat > 0]
    ax8.hist(np.log10(power_flat + 1e-12), bins=100, color="steelblue", alpha=0.7, density=True)
    ax8.axvline(np.log10(results["geometric_mean"]), color="red", linestyle="--", linewidth=2, label=f"GM={results['geometric_mean']:.2e}")
    ax8.axvline(np.log10(results["arithmetic_mean"]), color="green", linestyle="--", linewidth=2, label=f"AM={results['arithmetic_mean']:.2e}")
    ax8.set_xlabel("log10(Power)", fontsize=10)
    ax8.set_ylabel("Density", fontsize=10)
    ax8.set_title("8. Power Distribution", fontsize=11, fontweight="bold")
    ax8.legend(fontsize=8)

    # Row 3: Results summary
    ax9 = fig.add_subplot(gs[2, :2])
    ax9.axis("off")
    summary_text = f"""
    SFM Computation Summary
    {'='*40}

    Geometric Mean (GM): {results['geometric_mean']:.6e}
    Arithmetic Mean (AM): {results['arithmetic_mean']:.6e}

    SFM = GM / AM = {results['sfm']:.6f}

    Interpretation:
    - SFM close to 0: Strong peaks (regular/periodic pattern)
    - SFM close to 1: Flat spectrum (noise-like/irregular)
    """
    ax9.text(0.1, 0.5, summary_text, fontsize=12, fontfamily="monospace",
             verticalalignment="center", transform=ax9.transAxes,
             bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5))

    # SFM gauge
    ax10 = fig.add_subplot(gs[2, 2:])
    sfm_value = results["sfm"]

    # Create a horizontal bar gauge
    ax10.barh(0, sfm_value, height=0.5, color="steelblue", alpha=0.8)
    ax10.barh(0, 1 - sfm_value, left=sfm_value, height=0.5, color="lightgray", alpha=0.5)
    ax10.axvline(sfm_value, color="red", linewidth=3)
    ax10.set_xlim(0, 1)
    ax10.set_ylim(-0.5, 0.5)
    ax10.set_xlabel("SFM Value", fontsize=12)
    ax10.set_title(f"SFM = {sfm_value:.4f}", fontsize=14, fontweight="bold", color="red")

    # Add labels
    ax10.text(0.05, -0.3, "Peaky\n(Regular)", fontsize=10, ha="center")
    ax10.text(0.95, -0.3, "Flat\n(Noisy)", fontsize=10, ha="center")
    ax10.set_yticks([])

    # Main title
    file_name = image_path.name
    plt.suptitle(f"{title_prefix}SFM Processing: {file_name}", fontsize=14, fontweight="bold", y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved visualization to {output_path}")

    return results["sfm"]


def visualize_comparison(
    normal_path: Path,
    anomaly_path: Path,
    output_path: Path,
    domain: str,
    crop_center: Optional[float] = None,
):
    """Visualize side-by-side comparison of Normal vs Anomaly."""
    normal_img = load_image(normal_path)
    anomaly_img = load_image(anomaly_path)

    normal_results = spectral_flatness_2d_with_intermediates(normal_img, crop_center=crop_center)
    anomaly_results = spectral_flatness_2d_with_intermediates(anomaly_img, crop_center=crop_center)

    fig, axes = plt.subplots(2, 5, figsize=(20, 9))

    # Normal row
    axes[0, 0].imshow(normal_results["original"], cmap="gray" if normal_results["original"].ndim == 2 else None)
    axes[0, 0].set_title("Original", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(normal_results["dc_removed"], cmap="RdBu_r")
    axes[0, 1].set_title("DC Removed", fontsize=11)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(normal_results["windowed"], cmap="gray")
    axes[0, 2].set_title("Windowed", fontsize=11)
    axes[0, 2].axis("off")

    im_n = axes[0, 3].imshow(normal_results["power_spectrum_log"], cmap="hot")
    axes[0, 3].set_title("Power Spectrum (log)", fontsize=11)
    axes[0, 3].axis("off")

    axes[0, 4].text(0.5, 0.5, f"SFM = {normal_results['sfm']:.4f}",
                    fontsize=20, ha="center", va="center", fontweight="bold", color="blue",
                    transform=axes[0, 4].transAxes)
    axes[0, 4].set_title("Result", fontsize=11)
    axes[0, 4].axis("off")
    axes[0, 4].set_facecolor("lightblue")

    # Anomaly row
    axes[1, 0].imshow(anomaly_results["original"], cmap="gray" if anomaly_results["original"].ndim == 2 else None)
    axes[1, 0].set_title("Original", fontsize=11)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(anomaly_results["dc_removed"], cmap="RdBu_r")
    axes[1, 1].set_title("DC Removed", fontsize=11)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(anomaly_results["windowed"], cmap="gray")
    axes[1, 2].set_title("Windowed", fontsize=11)
    axes[1, 2].axis("off")

    im_a = axes[1, 3].imshow(anomaly_results["power_spectrum_log"], cmap="hot")
    axes[1, 3].set_title("Power Spectrum (log)", fontsize=11)
    axes[1, 3].axis("off")

    axes[1, 4].text(0.5, 0.5, f"SFM = {anomaly_results['sfm']:.4f}",
                    fontsize=20, ha="center", va="center", fontweight="bold", color="red",
                    transform=axes[1, 4].transAxes)
    axes[1, 4].set_title("Result", fontsize=11)
    axes[1, 4].axis("off")
    axes[1, 4].set_facecolor("lightyellow")

    # Row labels
    fig.text(0.02, 0.72, "NORMAL", fontsize=14, fontweight="bold", color="blue", rotation=90, va="center")
    fig.text(0.02, 0.28, "ANOMALY", fontsize=14, fontweight="bold", color="red", rotation=90, va="center")

    # Difference
    sfm_diff = anomaly_results["sfm"] - normal_results["sfm"]
    diff_text = f"SFM Difference: {sfm_diff:+.4f}"
    if sfm_diff > 0:
        diff_text += " (Anomaly more flat/noisy)"
    else:
        diff_text += " (Anomaly more peaky/regular)"

    plt.suptitle(f"{domain}: Normal vs Anomaly Comparison\n{diff_text}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0.03, 0, 1, 0.95])

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
    parser = argparse.ArgumentParser(description="SFM Process Visualization")
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
        "--crop-center",
        type=float,
        default=None,
        help="Crop center region of spectrum",
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
                sfm = visualize_sfm_process(
                    path, output_path,
                    crop_center=args.crop_center,
                    title_prefix=f"[{domain}] {category}: "
                )
                logger.info(f"  {category}_{i+1}: SFM = {sfm:.4f}")

        # Visualize comparisons
        if samples["train_good"] and samples["test_fault"]:
            for i in range(min(len(samples["train_good"]), len(samples["test_fault"]), args.n_samples)):
                output_path = domain_output / f"comparison_{i+1}.png"
                visualize_comparison(
                    samples["train_good"][i],
                    samples["test_fault"][i],
                    output_path,
                    domain,
                    crop_center=args.crop_center,
                )

    logger.info(f"\nVisualization complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
