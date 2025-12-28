#!/usr/bin/env python3
"""Visualize noise amplification effect of per-image normalization.

This script shows why per-image normalization degrades Cold image performance:
- Cold images have low intensity AND low signal
- Stretching to [0,1] amplifies noise along with signal
"""

import argparse
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_tiff_raw(path: Path) -> np.ndarray:
    img = Image.open(path)
    return np.array(img, dtype=np.float32)


def robust_normalize(img: np.ndarray) -> np.ndarray:
    p5 = np.percentile(img, 5)
    p95 = np.percentile(img, 95)
    if p95 - p5 > 1e-8:
        normalized = (img - p5) / (p95 - p5)
        return np.clip(normalized, 0, 1)
    return img


def compute_local_std(img: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Compute local standard deviation as noise estimate."""
    from scipy.ndimage import uniform_filter

    mean = uniform_filter(img, window_size)
    sqr_mean = uniform_filter(img**2, window_size)
    variance = sqr_mean - mean**2
    variance = np.maximum(variance, 0)  # Handle numerical errors
    return np.sqrt(variance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./datasets/HDMAP/1000_tiff_minmax")
    parser.add_argument("--domain", type=str, default="C")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select specific Cold and Warm fault images
    cold_idx = 327  # Cold fault
    warm_idx = 625  # Warm fault

    cold_path = root / f"domain_{args.domain}" / "test" / "fault" / f"{cold_idx:06d}.tiff"
    warm_path = root / f"domain_{args.domain}" / "test" / "fault" / f"{warm_idx:06d}.tiff"

    cold_img = load_tiff_raw(cold_path)
    warm_img = load_tiff_raw(warm_path)

    cold_norm = robust_normalize(cold_img.copy())
    warm_norm = robust_normalize(warm_img.copy())

    # Compute local noise (std)
    cold_noise_orig = compute_local_std(cold_img)
    cold_noise_norm = compute_local_std(cold_norm)
    warm_noise_orig = compute_local_std(warm_img)
    warm_noise_norm = compute_local_std(warm_norm)

    # Create figure
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))

    # Row 0: Original images
    axes[0, 0].imshow(cold_img, cmap='gray', vmin=0, vmax=0.5)
    axes[0, 0].set_title(f'Cold Original\nmean={cold_img.mean():.3f}', fontsize=10)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cold_norm, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Cold Normalized\nmean={cold_norm.mean():.3f}', fontsize=10)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(warm_img, cmap='gray', vmin=0, vmax=0.5)
    axes[0, 2].set_title(f'Warm Original\nmean={warm_img.mean():.3f}', fontsize=10)
    axes[0, 2].axis('off')

    axes[0, 3].imshow(warm_norm, cmap='gray', vmin=0, vmax=1)
    axes[0, 3].set_title(f'Warm Normalized\nmean={warm_norm.mean():.3f}', fontsize=10)
    axes[0, 3].axis('off')

    # Row 1: Local noise maps
    noise_vmax = max(cold_noise_norm.max(), warm_noise_norm.max())

    im1 = axes[1, 0].imshow(cold_noise_orig, cmap='hot', vmin=0, vmax=0.05)
    axes[1, 0].set_title(f'Cold Noise (orig)\nmean={cold_noise_orig.mean():.4f}', fontsize=10)
    axes[1, 0].axis('off')

    im2 = axes[1, 1].imshow(cold_noise_norm, cmap='hot', vmin=0, vmax=0.15)
    axes[1, 1].set_title(f'Cold Noise (norm)\nmean={cold_noise_norm.mean():.4f}\n↑{cold_noise_norm.mean()/cold_noise_orig.mean():.1f}x', fontsize=10)
    axes[1, 1].axis('off')

    im3 = axes[1, 2].imshow(warm_noise_orig, cmap='hot', vmin=0, vmax=0.05)
    axes[1, 2].set_title(f'Warm Noise (orig)\nmean={warm_noise_orig.mean():.4f}', fontsize=10)
    axes[1, 2].axis('off')

    im4 = axes[1, 3].imshow(warm_noise_norm, cmap='hot', vmin=0, vmax=0.15)
    axes[1, 3].set_title(f'Warm Noise (norm)\nmean={warm_noise_norm.mean():.4f}\n↑{warm_noise_norm.mean()/warm_noise_orig.mean():.1f}x', fontsize=10)
    axes[1, 3].axis('off')

    # Row 2: Histograms
    axes[2, 0].hist(cold_img.flatten(), bins=50, alpha=0.7, color='blue', density=True)
    axes[2, 0].set_title('Cold Original Distribution', fontsize=10)
    axes[2, 0].set_xlim(0, 0.6)

    axes[2, 1].hist(cold_norm.flatten(), bins=50, alpha=0.7, color='blue', density=True)
    axes[2, 1].set_title('Cold Normalized Distribution', fontsize=10)
    axes[2, 1].set_xlim(0, 1)

    axes[2, 2].hist(warm_img.flatten(), bins=50, alpha=0.7, color='red', density=True)
    axes[2, 2].set_title('Warm Original Distribution', fontsize=10)
    axes[2, 2].set_xlim(0, 0.6)

    axes[2, 3].hist(warm_norm.flatten(), bins=50, alpha=0.7, color='red', density=True)
    axes[2, 3].set_title('Warm Normalized Distribution', fontsize=10)
    axes[2, 3].set_xlim(0, 1)

    # Row 3: Line profiles (horizontal slice through center)
    center_row = cold_img.shape[0] // 2

    axes[3, 0].plot(cold_img[center_row, :], 'b-', alpha=0.7, label='Original')
    axes[3, 0].set_title('Cold Center Row Profile', fontsize=10)
    axes[3, 0].set_ylim(0, 0.5)
    axes[3, 0].legend(fontsize=8)

    axes[3, 1].plot(cold_norm[center_row, :], 'b-', alpha=0.7, label='Normalized')
    axes[3, 1].set_title('Cold Center Row (Norm)', fontsize=10)
    axes[3, 1].set_ylim(0, 1)
    axes[3, 1].legend(fontsize=8)

    axes[3, 2].plot(warm_img[center_row, :], 'r-', alpha=0.7, label='Original')
    axes[3, 2].set_title('Warm Center Row Profile', fontsize=10)
    axes[3, 2].set_ylim(0, 0.5)
    axes[3, 2].legend(fontsize=8)

    axes[3, 3].plot(warm_norm[center_row, :], 'r-', alpha=0.7, label='Normalized')
    axes[3, 3].set_title('Warm Center Row (Norm)', fontsize=10)
    axes[3, 3].set_ylim(0, 1)
    axes[3, 3].legend(fontsize=8)

    plt.suptitle(f'Domain {args.domain} - Per-Image Normalization Effect Analysis\n'
                 f'Cold images have higher noise amplification after normalization',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"noise_amplification_domain{args.domain}_{timestamp}.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"Saved: {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Noise Amplification Analysis")
    print(f"{'='*60}")

    cold_amp = cold_noise_norm.mean() / cold_noise_orig.mean()
    warm_amp = warm_noise_norm.mean() / warm_noise_orig.mean()

    print(f"\nCold (idx={cold_idx}):")
    print(f"  Original noise: {cold_noise_orig.mean():.4f}")
    print(f"  Normalized noise: {cold_noise_norm.mean():.4f}")
    print(f"  Amplification: {cold_amp:.2f}x")

    print(f"\nWarm (idx={warm_idx}):")
    print(f"  Original noise: {warm_noise_orig.mean():.4f}")
    print(f"  Normalized noise: {warm_noise_norm.mean():.4f}")
    print(f"  Amplification: {warm_amp:.2f}x")

    print(f"\nCold/Warm amplification ratio: {cold_amp/warm_amp:.2f}x")
    print(f"(Cold noise is amplified {cold_amp/warm_amp:.2f}x more than Warm)")


if __name__ == "__main__":
    main()
