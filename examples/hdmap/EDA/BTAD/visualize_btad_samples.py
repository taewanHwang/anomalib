#!/usr/bin/env python3
"""BTAD Dataset Visualization Script.

Generates visualization images for each BTAD category showing:
- 5 training samples (normal)
- 5 test normal samples
- 5 test defect samples

Each run produces different random samples. Output filenames include timestamp.

Usage:
    python visualize_btad_samples.py --root ./datasets/BTech
    python visualize_btad_samples.py --root ./datasets/BTech --categories 01 02
"""

import argparse
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def get_btad_samples(root: Path, category: str) -> dict:
    """Get sample paths from BTAD dataset structure.

    BTAD structure:
        root/category/
        ├── train/
        │   └── ok/
        ├── test/
        │   ├── ok/
        │   └── <defect_types>/
        └── ground_truth/

    Args:
        root: Root directory of BTAD dataset
        category: Category name (01, 02, 03)

    Returns:
        Dictionary with train_normal, test_normal, test_defect paths
    """
    category_path = root / category

    # Training samples (all normal/ok)
    train_path = category_path / "train" / "ok"
    train_samples = list(train_path.glob("*.png")) + list(train_path.glob("*.bmp"))

    # Test normal samples
    test_ok_path = category_path / "test" / "ok"
    test_normal = list(test_ok_path.glob("*.png")) + list(test_ok_path.glob("*.bmp"))

    # Test defect samples (all non-ok folders in test)
    test_path = category_path / "test"
    test_defect = []
    for subfolder in test_path.iterdir():
        if subfolder.is_dir() and subfolder.name != "ok":
            test_defect.extend(list(subfolder.glob("*.png")))
            test_defect.extend(list(subfolder.glob("*.bmp")))

    return {
        "train_normal": train_samples,
        "test_normal": test_normal,
        "test_defect": test_defect,
    }


def load_image(path: Path) -> np.ndarray:
    """Load image as numpy array."""
    img = Image.open(path)
    return np.array(img)


def create_visualization(
    samples: dict,
    category: str,
    n_samples: int = 5,
) -> plt.Figure:
    """Create visualization figure for a category.

    Args:
        samples: Dictionary with train_normal, test_normal, test_defect paths
        category: Category name for title
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
    test_defect_selected = random.sample(
        samples["test_defect"],
        min(n_samples, len(samples["test_defect"]))
    )

    # Create figure: 3 rows x n_samples columns
    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 3, 9))

    row_titles = [
        f"Train Normal ({len(samples['train_normal'])} total)",
        f"Test Normal ({len(samples['test_normal'])} total)",
        f"Test Defect ({len(samples['test_defect'])} total)",
    ]

    all_selected = [train_selected, test_normal_selected, test_defect_selected]

    for row_idx, (selected, title) in enumerate(zip(all_selected, row_titles)):
        for col_idx in range(n_samples):
            ax = axes[row_idx, col_idx]

            if col_idx < len(selected):
                img = load_image(selected[col_idx])
                ax.imshow(img)
                # Show filename (truncated if too long)
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

    fig.suptitle(f"BTAD Category {category} - Random Samples", fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize BTAD dataset samples")
    parser.add_argument(
        "--root",
        type=str,
        default="./datasets/BTech",
        help="Root directory of BTAD/BTech dataset",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["01", "02", "03"],
        help="Categories to visualize (default: 01 02 03)",
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
        help="Output directory (default: same as script location)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None for random)",
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

    print(f"BTAD Dataset Visualization")
    print(f"Root: {root}")
    print(f"Output: {output_dir}")
    print(f"Timestamp: {timestamp}")
    print("-" * 50)

    for category in args.categories:
        print(f"\nProcessing Category {category}...")

        # Check if category exists
        category_path = root / category
        if not category_path.exists():
            print(f"  WARNING: Category path not found: {category_path}")
            print(f"  Skipping...")
            continue

        # Get samples
        samples = get_btad_samples(root, category)

        print(f"  Train normal: {len(samples['train_normal'])} images")
        print(f"  Test normal:  {len(samples['test_normal'])} images")
        print(f"  Test defect:  {len(samples['test_defect'])} images")

        # Check if enough samples
        if not samples['train_normal']:
            print(f"  WARNING: No training samples found")
            continue

        # Create visualization
        fig = create_visualization(samples, category, args.n_samples)

        # Save
        output_path = output_dir / f"btad_category{category}_{timestamp}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        print(f"  Saved: {output_path}")

    print("\n" + "=" * 50)
    print("Visualization complete!")


if __name__ == "__main__":
    main()
