#!/usr/bin/env python3
"""Visualize the effect of per-image normalization on HDMAP images.

This script creates a comparison visualization showing:
- Row 1: Original images (Cold 0-499 vs Warm 500-999)
- Row 2: MinMax normalized images
- Row 3: Robust normalized images

The goal is to verify that per-image normalization removes the
intensity difference between Cold and Warm conditions.
"""

import argparse
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def load_tiff_raw(path: Path) -> np.ndarray:
    """Load TIFF image as numpy array."""
    img = Image.open(path)
    data = np.array(img, dtype=np.float32)
    return data


def minmax_normalize(img: np.ndarray) -> np.ndarray:
    """Per-image min-max normalization to [0, 1]."""
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min > 1e-8:
        return (img - img_min) / (img_max - img_min)
    return img


def robust_normalize(img: np.ndarray) -> np.ndarray:
    """Per-image robust normalization using p5/p95."""
    p5 = np.percentile(img, 5)
    p95 = np.percentile(img, 95)
    if p95 - p5 > 1e-8:
        normalized = (img - p5) / (p95 - p5)
        return np.clip(normalized, 0, 1)
    return img


def get_sample_images(root: Path, domain: str, split: str, label: str,
                      cold_indices: list, warm_indices: list) -> dict:
    """Get sample images for Cold and Warm conditions.

    Returns:
        Dictionary with 'cold' and 'warm' lists of (path, index) tuples
    """
    label_dir = root / f"domain_{domain}" / split / label

    cold_samples = []
    warm_samples = []

    for idx in cold_indices:
        path = label_dir / f"{idx:06d}.tiff"
        if path.exists():
            cold_samples.append((path, idx))

    for idx in warm_indices:
        path = label_dir / f"{idx:06d}.tiff"
        if path.exists():
            warm_samples.append((path, idx))

    return {'cold': cold_samples, 'warm': warm_samples}


def create_comparison_figure(
    samples: dict,
    domain: str,
    split: str,
    label: str,
    output_dir: Path,
    n_samples: int = 5,
) -> None:
    """Create comparison figure showing Original vs MinMax vs Robust normalization.

    Layout:
        - Columns: Cold samples (n_samples) | Warm samples (n_samples)
        - Rows: Original | MinMax | Robust
    """
    cold_samples = samples['cold'][:n_samples]
    warm_samples = samples['warm'][:n_samples]

    total_cols = n_samples * 2  # Cold + Warm
    n_rows = 3  # Original, MinMax, Robust

    fig, axes = plt.subplots(n_rows, total_cols, figsize=(total_cols * 2.5, n_rows * 3))

    row_titles = ['Original', 'MinMax Norm', 'Robust Norm (p5/p95)']

    # Collect intensity stats for summary
    stats = {
        'original': {'cold': [], 'warm': []},
        'minmax': {'cold': [], 'warm': []},
        'robust': {'cold': [], 'warm': []},
    }

    for col_idx, (sample_path, file_idx) in enumerate(cold_samples + warm_samples):
        is_cold = col_idx < n_samples
        condition = 'cold' if is_cold else 'warm'

        # Load image
        img = load_tiff_raw(sample_path)

        # Apply normalizations
        img_minmax = minmax_normalize(img.copy())
        img_robust = robust_normalize(img.copy())

        # Collect stats
        stats['original'][condition].append(img.mean())
        stats['minmax'][condition].append(img_minmax.mean())
        stats['robust'][condition].append(img_robust.mean())

        images = [img, img_minmax, img_robust]

        for row_idx, (norm_img, title) in enumerate(zip(images, row_titles)):
            ax = axes[row_idx, col_idx]

            # Use fixed vmin/vmax for original to show intensity difference
            # Use 0-1 for normalized to show they're now similar
            if row_idx == 0:  # Original
                im = ax.imshow(norm_img, cmap='gray', vmin=0, vmax=1)
            else:  # Normalized
                im = ax.imshow(norm_img, cmap='gray', vmin=0, vmax=1)

            ax.axis('off')

            # Title with file index and mean intensity
            mean_val = norm_img.mean()
            if row_idx == 0:
                cond_str = 'Cold' if is_cold else 'Warm'
                ax.set_title(f'{file_idx:06d}\n({cond_str})\nmean={mean_val:.3f}', fontsize=8)
            else:
                ax.set_title(f'mean={mean_val:.3f}', fontsize=8)

    # Add row labels
    for row_idx, title in enumerate(row_titles):
        axes[row_idx, 0].set_ylabel(title, fontsize=12, rotation=90, labelpad=10)

    # Add column group labels
    fig.text(0.25, 0.98, 'Cold Condition (0-499)', ha='center', fontsize=12, fontweight='bold')
    fig.text(0.75, 0.98, 'Warm Condition (500-999)', ha='center', fontsize=12, fontweight='bold')

    # Add vertical separator
    fig.patches.append(plt.Rectangle((0.5, 0.02), 0.002, 0.94,
                                      transform=fig.transFigure,
                                      facecolor='black', alpha=0.3))

    plt.suptitle(f'HDMAP Domain {domain} ({split}/{label}) - Normalization Effect\n'
                 f'Per-image normalization removes Cold/Warm intensity difference',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"normalization_effect_domain{domain}_{label}_{timestamp}.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"Saved: {output_path}")

    # Print stats summary
    print(f"\n{'='*60}")
    print(f"Mean Intensity Statistics - Domain {domain} ({split}/{label})")
    print(f"{'='*60}")

    for norm_type in ['original', 'minmax', 'robust']:
        cold_mean = np.mean(stats[norm_type]['cold'])
        warm_mean = np.mean(stats[norm_type]['warm'])
        ratio = warm_mean / cold_mean if cold_mean > 0 else 0

        print(f"\n{norm_type.upper()}:")
        print(f"  Cold mean: {cold_mean:.4f}")
        print(f"  Warm mean: {warm_mean:.4f}")
        print(f"  Warm/Cold ratio: {ratio:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Visualize normalization effect on HDMAP")
    parser.add_argument("--root", type=str,
                        default="./datasets/HDMAP/1000_tiff_minmax",
                        help="Root directory of HDMAP dataset")
    parser.add_argument("--domain", type=str, default="C",
                        help="Domain to visualize (A, B, C, D)")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of samples per condition")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    root = Path(args.root)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"HDMAP Normalization Effect Visualization")
    print(f"Root: {root}")
    print(f"Domain: {args.domain}")
    print(f"Output: {output_dir}")
    print("-" * 50)

    # Sample indices
    cold_indices = random.sample(range(0, 500), args.n_samples)
    warm_indices = random.sample(range(500, 1000), args.n_samples)

    print(f"Cold indices: {cold_indices}")
    print(f"Warm indices: {warm_indices}")

    # Process test/fault
    print(f"\n--- Test Fault ---")
    fault_samples = get_sample_images(root, args.domain, 'test', 'fault',
                                       cold_indices, warm_indices)
    create_comparison_figure(fault_samples, args.domain, 'test', 'fault',
                            output_dir, args.n_samples)

    # Process test/good
    print(f"\n--- Test Good ---")
    good_samples = get_sample_images(root, args.domain, 'test', 'good',
                                      cold_indices, warm_indices)
    create_comparison_figure(good_samples, args.domain, 'test', 'good',
                            output_dir, args.n_samples)

    print("\n" + "=" * 50)
    print("Visualization complete!")


if __name__ == "__main__":
    main()
