#!/usr/bin/env python3
"""HDMAP Dataset Visualization Script.

Generates visualization images for each HDMAP domain showing:
- 5 training samples (normal/good)
- 5 test normal samples (good)
- 5 test fault samples

Raw TIFF values are used without per-image scaling.
Values > 1 are clamped to 1 for visualization.

Usage:
    python visualize_hdmap_samples.py --root ./datasets/HDMAP/1000_tiff_minmax
    python visualize_hdmap_samples.py --root ./datasets/HDMAP/1000_tiff_minmax --domains A B
"""

import argparse
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def get_hdmap_samples(root: Path, domain: str) -> dict:
    """Get sample paths from HDMAP dataset structure.

    HDMAP structure:
        root/domain_X/
        ├── train/
        │   └── good/
        └── test/
            ├── good/
            └── fault/

    Args:
        root: Root directory of HDMAP dataset
        domain: Domain name (A, B, C, D)

    Returns:
        Dictionary with train_normal, test_normal, test_fault paths
    """
    domain_path = root / f"domain_{domain}"

    # Training samples (all normal/good)
    train_path = domain_path / "train" / "good"
    train_samples = list(train_path.glob("*.tiff")) + list(train_path.glob("*.tif"))

    # Test normal samples (good)
    test_good_path = domain_path / "test" / "good"
    test_normal = list(test_good_path.glob("*.tiff")) + list(test_good_path.glob("*.tif"))

    # Test fault samples
    test_fault_path = domain_path / "test" / "fault"
    test_fault = list(test_fault_path.glob("*.tiff")) + list(test_fault_path.glob("*.tif"))

    return {
        "train_normal": sorted(train_samples),
        "test_normal": sorted(test_normal),
        "test_fault": sorted(test_fault),
    }


def load_tiff_raw(path: Path) -> np.ndarray:
    """Load TIFF image without per-image scaling.

    Values > 1 are clamped to 1 for visualization.

    Args:
        path: Path to TIFF file

    Returns:
        numpy array with values in [0, 1] range
    """
    img = Image.open(path)
    data = np.array(img, dtype=np.float32)

    # Clamp values > 1 to 1 (no per-image scaling)
    data = np.clip(data, 0, 1)

    return data


def create_visualization(
    samples: dict,
    domain: str,
    n_samples: int = 5,
) -> plt.Figure:
    """Create visualization figure for a domain.

    Args:
        samples: Dictionary with train_normal, test_normal, test_fault paths
        domain: Domain name for title
        n_samples: Number of samples per row

    Returns:
        Matplotlib figure
    """
    # Random sample selection
    train_selected = random.sample(
        samples["train_normal"],
        min(n_samples, len(samples["train_normal"]))
    )
    test_normal_selected = random.sample(
        samples["test_normal"],
        min(n_samples, len(samples["test_normal"]))
    )
    test_fault_selected = random.sample(
        samples["test_fault"],
        min(n_samples, len(samples["test_fault"]))
    )

    # Create figure: 3 rows x n_samples columns
    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 3, 9))

    row_titles = [
        f"Train Good ({len(samples['train_normal'])} total)",
        f"Test Good ({len(samples['test_normal'])} total)",
        f"Test Fault ({len(samples['test_fault'])} total)",
    ]

    all_selected = [train_selected, test_normal_selected, test_fault_selected]

    for row_idx, (selected, title) in enumerate(zip(all_selected, row_titles)):
        for col_idx in range(n_samples):
            ax = axes[row_idx, col_idx]

            if col_idx < len(selected):
                # Load TIFF with raw values (clamped to [0,1])
                img = load_tiff_raw(selected[col_idx])

                # Display grayscale with fixed vmin/vmax
                im = ax.imshow(img, cmap='gray', vmin=0, vmax=1)

                # Show filename
                fname = selected[col_idx].name
                if len(fname) > 15:
                    fname = fname[:12] + "..."
                ax.set_title(fname, fontsize=8)
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=12)

            ax.axis('off')

            # Row label on first column
            if col_idx == 0:
                ax.set_ylabel(title, fontsize=10, rotation=0, labelpad=80, va='center')

    fig.suptitle(
        f"HDMAP Domain {domain} - Random Samples (Raw values, clipped to [0,1])",
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()

    return fig


def print_value_statistics(samples: dict, domain: str, n_check: int = 5):
    """Print value statistics for a few samples to verify raw loading."""
    print(f"\n  Value statistics (first {n_check} samples):")

    for category, paths in samples.items():
        if not paths:
            continue
        sample_paths = paths[:n_check]
        all_mins, all_maxs = [], []

        for p in sample_paths:
            img = Image.open(p)
            data = np.array(img, dtype=np.float32)
            all_mins.append(data.min())
            all_maxs.append(data.max())

        print(f"    {category}: min=[{min(all_mins):.4f}, {max(all_mins):.4f}], "
              f"max=[{min(all_maxs):.4f}, {max(all_maxs):.4f}]")


def main():
    parser = argparse.ArgumentParser(description="Visualize HDMAP dataset samples")
    parser.add_argument(
        "--root",
        type=str,
        default="./datasets/HDMAP/1000_tiff_minmax",
        help="Root directory of HDMAP dataset",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["A", "B", "C", "D"],
        help="Domains to visualize (default: A B C D)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of samples per row (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as script location/results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None for random)",
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Show value statistics for verification",
    )
    args = parser.parse_args()

    # Set seed if provided
    if args.seed is not None:
        random.seed(args.seed)

    root = Path(args.root)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"HDMAP Dataset Visualization")
    print(f"Root: {root}")
    print(f"Output: {output_dir}")
    print(f"Timestamp: {timestamp}")
    print(f"Note: Using raw TIFF values, clamping >1 to 1")
    print("-" * 50)

    for domain in args.domains:
        print(f"\nProcessing Domain {domain}...")

        # Check if domain exists
        domain_path = root / f"domain_{domain}"
        if not domain_path.exists():
            print(f"  WARNING: Domain path not found: {domain_path}")
            print(f"  Skipping...")
            continue

        # Get samples
        samples = get_hdmap_samples(root, domain)

        print(f"  Train good:  {len(samples['train_normal'])} images")
        print(f"  Test good:   {len(samples['test_normal'])} images")
        print(f"  Test fault:  {len(samples['test_fault'])} images")

        # Show statistics if requested
        if args.show_stats:
            print_value_statistics(samples, domain)

        # Check if enough samples
        if not samples['train_normal']:
            print(f"  WARNING: No training samples found")
            continue

        # Create visualization
        fig = create_visualization(samples, domain, args.n_samples)

        # Save
        output_path = output_dir / f"hdmap_domain{domain}_{timestamp}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        print(f"  Saved: {output_path}")

    print("\n" + "=" * 50)
    print("Visualization complete!")


if __name__ == "__main__":
    main()
