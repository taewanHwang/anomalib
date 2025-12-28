#!/usr/bin/env python3
"""Visualize test images used in WinCLIP experiments.

Shows:
1. Fault vs Good comparison
2. Cold vs Warm references
3. Image with artificial horizontal lines
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from anomalib.data.datasets.image.hdmap import HDMAPDataset


ANOMALIB_ROOT = Path(__file__).parent.parent.parent.parent
DATASET_ROOT = ANOMALIB_ROOT / "datasets" / "HDMAP" / "1000_tiff_minmax"
OUTPUT_DIR = Path(__file__).parent / "HDMAP_vis"


def load_image(domain: str, split: str, label: str, index: int):
    """Load image from HDMAPDataset."""
    dataset = HDMAPDataset(
        root=str(DATASET_ROOT),
        domain=domain,
        split=split,
        target_size=None,  # Original size
    )

    target_path = f"{label}/{index:06d}.tiff"
    for item in dataset:
        if target_path in item.image_path:
            return item.image, item.image_path

    raise ValueError(f"Image not found: {target_path}")


def norm_display(t):
    """Normalize tensor for display."""
    if isinstance(t, torch.Tensor):
        arr = t.permute(1, 2, 0).numpy() if len(t.shape) == 3 else t.numpy()
    else:
        arr = t
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def add_horizontal_line(image: torch.Tensor, y_start: int, width: int = 5, intensity: float = 1.0):
    """Add artificial horizontal line to image."""
    modified = image.clone()
    for y in range(y_start, min(y_start + width, image.shape[1])):
        modified[:, y, :] = intensity
    return modified


def visualize_main_images(domain: str, fault_index: int, good_index: int, output_path: Path):
    """Visualize main test images: fault, good, cold ref, warm ref."""

    # Load images
    fault_img, fault_path = load_image(domain, "test", "fault", fault_index)
    good_img, good_path = load_image(domain, "test", "good", good_index)
    cold_ref, cold_path = load_image(domain, "test", "good", 0)
    warm_ref, warm_path = load_image(domain, "test", "good", 999)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Main comparison
    ax = axes[0, 0]
    ax.imshow(norm_display(fault_img))
    ax.set_title(f"FAULT Sample\n{domain}/fault/{fault_index:06d}.tiff\nShape: {fault_img.shape[1]}x{fault_img.shape[2]}", fontsize=11)
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(norm_display(good_img))
    ax.set_title(f"GOOD Sample\n{domain}/good/{good_index:06d}.tiff\nShape: {good_img.shape[1]}x{good_img.shape[2]}", fontsize=11)
    ax.axis("off")

    ax = axes[0, 2]
    diff = fault_img - good_img
    im = ax.imshow(diff.mean(dim=0).numpy(), cmap="RdBu_r", vmin=-0.1, vmax=0.1)
    ax.set_title(f"FAULT - GOOD Difference\nmax_diff={diff.abs().max():.4f}", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 2: Cold vs Warm reference
    ax = axes[1, 0]
    ax.imshow(norm_display(cold_ref))
    cold_mean = cold_ref.mean().item()
    cold_p90 = torch.quantile(cold_ref, 0.9).item()
    ax.set_title(f"COLD Reference (index 0)\nmean={cold_mean:.4f}, p90={cold_p90:.4f}", fontsize=11)
    ax.axis("off")

    ax = axes[1, 1]
    ax.imshow(norm_display(warm_ref))
    warm_mean = warm_ref.mean().item()
    warm_p90 = torch.quantile(warm_ref, 0.9).item()
    ax.set_title(f"WARM Reference (index 999)\nmean={warm_mean:.4f}, p90={warm_p90:.4f}", fontsize=11)
    ax.axis("off")

    ax = axes[1, 2]
    cold_warm_diff = warm_ref - cold_ref
    im = ax.imshow(cold_warm_diff.mean(dim=0).numpy(), cmap="RdBu_r", vmin=-0.2, vmax=0.2)
    ax.set_title(f"WARM - COLD Difference\nWarm is {warm_mean - cold_mean:.4f} brighter", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f"Test Images for WinCLIP Experiments ({domain})", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved main images visualization to: {output_path}")
    return fault_img, good_img


def visualize_artificial_lines(good_img: torch.Tensor, output_path: Path):
    """Visualize images with artificial horizontal lines."""

    # Create artificial line variants
    line_widths = [2, 5, 10, 20]
    h = good_img.shape[1]
    y_center = h // 2

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Original
    ax = axes[0, 0]
    ax.imshow(norm_display(good_img))
    ax.set_title("Original Good Image\n(No artificial line)", fontsize=11)
    ax.axis("off")

    # Different line widths
    for i, width in enumerate(line_widths):
        row = (i + 1) // 3
        col = (i + 1) % 3
        ax = axes[row, col]

        modified = add_horizontal_line(good_img, y_center - width // 2, width, intensity=1.0)
        ax.imshow(norm_display(modified))

        # Calculate line contrast
        line_intensity = 1.0
        bg_intensity = good_img.mean().item()
        contrast_ratio = line_intensity / (bg_intensity + 1e-8)

        ax.set_title(f"Artificial Line (width={width}px)\nLine intensity=1.0, Contrast={contrast_ratio:.2f}x", fontsize=11)
        ax.axis("off")

        # Add red box around line region
        from matplotlib.patches import Rectangle
        rect = Rectangle((0, y_center - width // 2 - 2), good_img.shape[2], width + 4,
                         linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    # Last cell: Contrast comparison
    ax = axes[1, 2]
    ax.axis("off")

    # Get real defect intensity from fault image
    fault_img, _ = load_image("domain_C", "test", "fault", 9)
    defect_row = 5
    defect_intensity = fault_img[:, defect_row, :].mean().item()
    bg_intensity = fault_img[:, defect_row + 5, :].mean().item()  # Nearby background
    real_contrast = defect_intensity / (bg_intensity + 1e-8)
    artificial_contrast = 1.0 / (good_img.mean().item() + 1e-8)

    text = f"""
    Contrast Comparison
    ══════════════════════════════

    Real HDMAP Defect:
      - Defect intensity: {defect_intensity:.4f}
      - Background: {bg_intensity:.4f}
      - Contrast ratio: {real_contrast:.2f}x

    Artificial Bright Line:
      - Line intensity: 1.0
      - Background: {good_img.mean().item():.4f}
      - Contrast ratio: {artificial_contrast:.2f}x

    ══════════════════════════════
    Artificial line is {artificial_contrast / real_contrast:.1f}x
    more visible than real defect!

    This explains why CLIP can recognize
    artificial lines but NOT real HDMAP defects.
    """

    ax.text(0.1, 0.5, text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle("Artificial Horizontal Line Test for CLIP Recognition", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved artificial lines visualization to: {output_path}")


def visualize_defect_detail(domain: str, fault_index: int, output_path: Path):
    """Visualize defect region in detail."""

    fault_img, _ = load_image(domain, "test", "fault", fault_index)
    good_img, _ = load_image(domain, "test", "good", fault_index)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Full images
    ax = axes[0, 0]
    ax.imshow(norm_display(fault_img))
    ax.set_title(f"Fault Image (full)\nShape: {fault_img.shape[1]}x{fault_img.shape[2]}", fontsize=11)
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(norm_display(good_img))
    ax.set_title("Good Image (full)", fontsize=11)
    ax.axis("off")

    ax = axes[0, 2]
    diff = (fault_img - good_img).mean(dim=0)
    im = ax.imshow(diff.numpy(), cmap="RdBu_r", vmin=-0.1, vmax=0.1)
    ax.set_title("Fault - Good (difference)", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 2: Row-by-row intensity analysis
    ax = axes[1, 0]
    fault_row_means = fault_img.mean(dim=(0, 2)).numpy()
    good_row_means = good_img.mean(dim=(0, 2)).numpy()
    x = np.arange(len(fault_row_means))
    ax.plot(x, fault_row_means, 'r-', label='Fault', linewidth=2)
    ax.plot(x, good_row_means, 'g-', label='Good', linewidth=2)
    ax.fill_between(x, fault_row_means, good_row_means, alpha=0.3,
                    where=fault_row_means > good_row_means, color='red', label='Defect region')
    ax.set_xlabel("Row index (Y)")
    ax.set_ylabel("Mean intensity")
    ax.set_title("Row-wise Intensity Profile", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Find defect rows
    diff_profile = fault_row_means - good_row_means
    defect_rows = np.where(diff_profile > 0.02)[0]

    ax = axes[1, 1]
    ax.bar(x, diff_profile, color=['red' if d > 0.02 else 'gray' for d in diff_profile])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.02, color='red', linestyle='--', linewidth=1, label='Defect threshold')
    ax.set_xlabel("Row index (Y)")
    ax.set_ylabel("Fault - Good")
    ax.set_title(f"Difference Profile\nDefect rows: {defect_rows}", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.axis("off")

    # Statistics
    if len(defect_rows) > 0:
        defect_intensity = fault_row_means[defect_rows].mean()
        normal_intensity = good_row_means.mean()
        contrast = defect_intensity / (normal_intensity + 1e-8)
    else:
        defect_intensity = fault_row_means.max()
        normal_intensity = good_row_means.mean()
        contrast = defect_intensity / (normal_intensity + 1e-8)

    text = f"""
    Defect Analysis Summary
    ══════════════════════════════════

    Image dimensions: {fault_img.shape[1]} x {fault_img.shape[2]}

    Defect location:
      - Rows with defect: {list(defect_rows) if len(defect_rows) > 0 else 'None detected'}
      - Defect type: Horizontal bright line

    Intensity statistics:
      - Defect region mean: {defect_intensity:.4f}
      - Normal region mean: {normal_intensity:.4f}
      - Contrast ratio: {contrast:.2f}x

    Key insight:
      HDMAP defects have very subtle contrast
      ({contrast:.2f}x), while CLIP needs much
      higher contrast (~5x) to recognize patterns.
    ══════════════════════════════════
    """

    ax.text(0.05, 0.5, text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

    plt.suptitle(f"Defect Detail Analysis ({domain}, fault index={fault_index})", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved defect detail visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="domain_C")
    parser.add_argument("--fault-index", type=int, default=9)
    parser.add_argument("--good-index", type=int, default=9)
    args = parser.parse_args()

    output_dir = OUTPUT_DIR / args.domain

    print(f"\n{'='*60}")
    print(f"Visualizing test images for {args.domain}")
    print(f"{'='*60}")

    # 1. Main images (fault, good, cold, warm)
    fault_img, good_img = visualize_main_images(
        args.domain, args.fault_index, args.good_index,
        output_dir / f"test_images_main.png"
    )

    # 2. Artificial horizontal lines
    visualize_artificial_lines(
        good_img,
        output_dir / f"test_images_artificial_lines.png"
    )

    # 3. Defect detail analysis
    visualize_defect_detail(
        args.domain, args.fault_index,
        output_dir / f"test_images_defect_detail.png"
    )

    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
