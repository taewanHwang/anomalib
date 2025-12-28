"""Visualize Cold samples (fault/cold and good/cold) from HDMAP dataset.

This script creates grid visualizations to compare:
- fault/cold: anomaly images in cold condition
- good/cold: normal images in cold condition

Usage:
    python visualize_cold_samples.py --gpu 0 --domain domain_C
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parents[4]))

from anomalib.data.datasets.image.hdmap import HDMAPDataset


def load_dataset(dataset_root: str, domain: str):
    """Load HDMAP test dataset."""
    dataset = HDMAPDataset(
        root=dataset_root,
        domain=domain,
        split="test",
        target_size=(240, 240),
        resize_method="resize",
    )
    return dataset


def tensor_to_numpy(img_tensor):
    """Convert tensor image to numpy for visualization.

    Preserves original TIFF values (0-1) without per-image scaling.
    """
    # (C, H, W) -> (H, W, C)
    img = img_tensor.permute(1, 2, 0).numpy()
    # Clip to [0, 1] to ensure valid display range (no per-image scaling)
    img = np.clip(img, 0, 1)
    return img


def visualize_grid(dataset, indices, title, save_path, cols=5, rows=4):
    """Visualize a grid of images.

    Args:
        dataset: HDMAP dataset
        indices: List of dataset indices to visualize
        title: Figure title
        save_path: Path to save the figure
        cols: Number of columns in grid
        rows: Number of rows in grid
    """
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for i, idx in enumerate(indices):
        if i >= rows * cols:
            break

        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]

        # Load sample
        sample = dataset[idx]
        img = tensor_to_numpy(sample.image)

        # Get file index from path
        file_idx = int(Path(sample.image_path).stem)
        label = "fault" if sample.gt_label == 1 else "good"

        # Compute image statistics
        mean_val = sample.image.mean().item()

        ax.imshow(img)
        ax.set_title(f"idx={idx}\nfile={file_idx:04d}\n{label}, mean={mean_val:.3f}", fontsize=8)
        ax.axis('off')

    # Hide empty subplots
    for i in range(len(indices), rows * cols):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--domain", type=str, default="domain_C")
    parser.add_argument("--interval", type=int, default=10, help="Sample interval")
    parser.add_argument("--per-figure", type=int, default=20, help="Images per figure")
    args = parser.parse_args()

    # Setup
    dataset_root = "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_tiff_minmax"
    output_dir = Path(__file__).parent / args.domain
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Domain: {args.domain}")
    print(f"Interval: {args.interval}")
    print(f"Images per figure: {args.per_figure}")
    print(f"Output: {output_dir}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(dataset_root, args.domain)
    print(f"Dataset size: {len(dataset)}")

    # Dataset index structure:
    #   0-499:     fault/cold (file 0-499)
    #   500-999:   fault/warm (file 500-999)
    #   1000-1499: good/cold (file 0-499)
    #   1500-1999: good/warm (file 500-999)

    # Generate indices with interval
    fault_cold_indices = list(range(0, 500, args.interval))      # 0, 10, 20, ..., 490
    good_cold_indices = list(range(1000, 1500, args.interval))   # 1000, 1010, ..., 1490

    print(f"\nFault/cold samples: {len(fault_cold_indices)} (indices 0-499, interval={args.interval})")
    print(f"Good/cold samples: {len(good_cold_indices)} (indices 1000-1499, interval={args.interval})")

    # Visualize fault/cold
    print("\n" + "=" * 60)
    print("Visualizing FAULT/COLD samples...")
    print("=" * 60)

    for fig_idx, start in enumerate(range(0, len(fault_cold_indices), args.per_figure)):
        chunk_indices = fault_cold_indices[start:start + args.per_figure]
        idx_range = f"{chunk_indices[0]:04d}-{chunk_indices[-1]:04d}"
        save_path = output_dir / f"fault_cold_{fig_idx:02d}_idx{idx_range}.png"
        title = f"FAULT/COLD - {args.domain} (indices {idx_range}, interval={args.interval})"
        visualize_grid(dataset, chunk_indices, title, save_path)

    # Visualize good/cold
    print("\n" + "=" * 60)
    print("Visualizing GOOD/COLD samples...")
    print("=" * 60)

    for fig_idx, start in enumerate(range(0, len(good_cold_indices), args.per_figure)):
        chunk_indices = good_cold_indices[start:start + args.per_figure]
        idx_range = f"{chunk_indices[0]:04d}-{chunk_indices[-1]:04d}"
        save_path = output_dir / f"good_cold_{fig_idx:02d}_idx{idx_range}.png"
        title = f"GOOD/COLD - {args.domain} (indices {idx_range}, interval={args.interval})"
        visualize_grid(dataset, chunk_indices, title, save_path)

    # Also create a side-by-side comparison figure
    print("\n" + "=" * 60)
    print("Creating comparison figure (fault vs good)...")
    print("=" * 60)

    # Select some representative samples for comparison
    comparison_fault_indices = list(range(0, 200, 20))[:10]  # 10 fault/cold samples
    comparison_good_indices = list(range(1000, 1200, 20))[:10]  # 10 good/cold samples

    fig, axes = plt.subplots(2, 10, figsize=(25, 6))
    fig.suptitle(f"COLD Comparison: Fault (top) vs Good (bottom) - {args.domain}", fontsize=14, fontweight='bold')

    # Top row: fault/cold
    for i, idx in enumerate(comparison_fault_indices):
        sample = dataset[idx]
        img = tensor_to_numpy(sample.image)
        file_idx = int(Path(sample.image_path).stem)
        mean_val = sample.image.mean().item()

        axes[0, i].imshow(img)
        axes[0, i].set_title(f"F:{file_idx:04d}\nm={mean_val:.3f}", fontsize=8)
        axes[0, i].axis('off')

    # Bottom row: good/cold
    for i, idx in enumerate(comparison_good_indices):
        sample = dataset[idx]
        img = tensor_to_numpy(sample.image)
        file_idx = int(Path(sample.image_path).stem)
        mean_val = sample.image.mean().item()

        axes[1, i].imshow(img)
        axes[1, i].set_title(f"G:{file_idx:04d}\nm={mean_val:.3f}", fontsize=8)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel("Fault/Cold", fontsize=12)
    axes[1, 0].set_ylabel("Good/Cold", fontsize=12)

    plt.tight_layout()
    comparison_path = output_dir / "comparison_fault_vs_good_cold.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {comparison_path}")
    plt.close()

    print(f"\nAll visualizations saved to: {output_dir}")
    print(f"Total figures: {len(list(output_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
