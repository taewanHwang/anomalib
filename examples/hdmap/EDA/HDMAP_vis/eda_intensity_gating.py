"""EDA: Mean Intensity-based Gating for Cold/Warm Condition Selection.

Analyze whether simple mean image intensity can distinguish cold vs warm conditions.

Key observation from previous EDA:
- Cold images: mean intensity ~0.19
- Warm images: mean intensity ~0.28

This could be a very simple and effective gating method!

Usage:
    python eda_intensity_gating.py --gpu 0 --domain domain_C --num-samples 100
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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


def compute_intensity_stats(dataset, indices: List[int]) -> List[float]:
    """Compute mean intensity for each sample."""
    means = []
    for idx in indices:
        sample = dataset[idx]
        mean_intensity = sample.image.mean().item()
        means.append(mean_intensity)
    return means


def find_optimal_threshold(
    cold_means: List[float],
    warm_means: List[float],
) -> Tuple[float, float, Dict]:
    """Find optimal threshold to separate cold and warm.

    Returns:
        (optimal_threshold, accuracy, details)
    """
    all_means = cold_means + warm_means
    min_val, max_val = min(all_means), max(all_means)

    # Try different thresholds
    thresholds = np.linspace(min_val, max_val, 1000)
    best_threshold = 0
    best_accuracy = 0
    best_details = {}

    for thresh in thresholds:
        # Cold should be below threshold, warm should be above
        cold_correct = sum(1 for m in cold_means if m < thresh)
        warm_correct = sum(1 for m in warm_means if m >= thresh)

        total_correct = cold_correct + warm_correct
        total = len(cold_means) + len(warm_means)
        accuracy = total_correct / total

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = thresh
            best_details = {
                'cold_correct': cold_correct,
                'cold_total': len(cold_means),
                'warm_correct': warm_correct,
                'warm_total': len(warm_means),
            }

    return best_threshold, best_accuracy, best_details


def analyze_intensity_gating(
    dataset,
    test_indices: Dict[str, List[int]],
) -> Dict:
    """Analyze intensity-based gating for all groups."""

    # Collect mean intensities for each group
    group_means = {}
    for group_name, indices in test_indices.items():
        group_means[group_name] = compute_intensity_stats(dataset, indices)

    # Combine cold and warm groups
    cold_means = group_means['fault/cold'] + group_means['good/cold']
    warm_means = group_means['fault/warm'] + group_means['good/warm']

    # Find optimal threshold
    threshold, accuracy, details = find_optimal_threshold(cold_means, warm_means)

    # Per-group accuracy with optimal threshold
    per_group_results = {}
    for group_name, means in group_means.items():
        gt = 'cold' if 'cold' in group_name else 'warm'

        if gt == 'cold':
            correct = sum(1 for m in means if m < threshold)
        else:
            correct = sum(1 for m in means if m >= threshold)

        per_group_results[group_name] = {
            'accuracy': correct / len(means) * 100,
            'correct': correct,
            'total': len(means),
            'means': means,
            'gt': gt,
        }

    return {
        'threshold': threshold,
        'overall_accuracy': accuracy * 100,
        'details': details,
        'per_group': per_group_results,
        'cold_means': cold_means,
        'warm_means': warm_means,
    }


def visualize_intensity_distribution(results: Dict, output_dir: Path):
    """Visualize intensity distribution and threshold."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Histogram of all samples
    ax1 = axes[0]
    ax1.hist(results['cold_means'], bins=50, alpha=0.7, label='Cold (GT)', color='blue', density=True)
    ax1.hist(results['warm_means'], bins=50, alpha=0.7, label='Warm (GT)', color='red', density=True)
    ax1.axvline(results['threshold'], color='green', linestyle='--', linewidth=2,
                label=f"Threshold: {results['threshold']:.4f}")
    ax1.set_xlabel('Mean Intensity')
    ax1.set_ylabel('Density')
    ax1.set_title(f"Intensity Distribution (Overall Acc: {results['overall_accuracy']:.1f}%)")
    ax1.legend()

    # Plot 2: Per-group box plot
    ax2 = axes[1]
    groups = ['fault/cold', 'good/cold', 'fault/warm', 'good/warm']
    colors = ['lightblue', 'blue', 'lightsalmon', 'red']

    box_data = [results['per_group'][g]['means'] for g in groups]
    bp = ax2.boxplot(box_data, labels=groups, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax2.axhline(results['threshold'], color='green', linestyle='--', linewidth=2,
                label=f"Threshold: {results['threshold']:.4f}")
    ax2.set_ylabel('Mean Intensity')
    ax2.set_title('Per-Group Intensity Distribution')
    ax2.legend()

    # Add accuracy annotations
    for i, group in enumerate(groups):
        acc = results['per_group'][group]['accuracy']
        ax2.annotate(f"{acc:.0f}%", xy=(i+1, results['threshold']),
                    xytext=(i+1, results['threshold'] + 0.02),
                    ha='center', fontsize=9, fontweight='bold',
                    color='green' if acc >= 90 else 'orange' if acc >= 70 else 'red')

    plt.tight_layout()
    save_path = output_dir / "intensity_gating_distribution.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def visualize_per_group_accuracy(results: Dict, output_dir: Path):
    """Visualize per-group gating accuracy."""

    groups = ['fault/cold', 'fault/warm', 'good/cold', 'good/warm']
    accuracies = [results['per_group'][g]['accuracy'] for g in groups]
    colors = ['blue' if 'cold' in g else 'red' for g in groups]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(groups, accuracies, color=colors, alpha=0.7, edgecolor='black')

    ax.axhline(90, color='green', linestyle='--', linewidth=2, label='90% Target')
    ax.axhline(results['overall_accuracy'], color='purple', linestyle='-', linewidth=2,
               label=f"Overall: {results['overall_accuracy']:.1f}%")

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.annotate(f'{acc:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=12,
                   fontweight='bold')

    ax.set_ylabel('Gating Accuracy (%)')
    ax.set_title(f'Mean Intensity Gating Accuracy (Threshold: {results["threshold"]:.4f})')
    ax.set_ylim(0, 105)
    ax.legend()

    plt.tight_layout()
    save_path = output_dir / "intensity_gating_accuracy.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--domain", type=str, default="domain_C")
    parser.add_argument("--num-samples", type=int, default=100, help="Samples per group (max 500)")
    args = parser.parse_args()

    # Setup
    dataset_root = "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_tiff_minmax"
    output_dir = Path(__file__).parent / args.domain / "intensity_gating"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Domain: {args.domain}")
    print(f"Samples per group: {args.num_samples}")
    print(f"Output: {output_dir}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(dataset_root, args.domain)
    print(f"Dataset size: {len(dataset)}")

    # Test indices
    n_samples = min(args.num_samples, 500)
    interval = max(1, 500 // n_samples)

    test_indices = {
        'fault/cold': list(range(0, 500, interval))[:n_samples],
        'fault/warm': list(range(500, 1000, interval))[:n_samples],
        'good/cold': list(range(1000, 1500, interval))[:n_samples],
        'good/warm': list(range(1500, 2000, interval))[:n_samples],
    }

    for group, indices in test_indices.items():
        print(f"  {group}: {len(indices)} samples")

    # Analyze intensity-based gating
    print("\n" + "=" * 60)
    print("Analyzing Mean Intensity-based Gating...")
    print("=" * 60)

    results = analyze_intensity_gating(dataset, test_indices)

    # Print results
    print(f"\nOptimal Threshold: {results['threshold']:.4f}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.1f}%")

    print("\n" + "-" * 60)
    print("Per-Group Results:")
    print("-" * 60)
    print(f"{'Group':<15} {'Accuracy':<12} {'Mean Intensity':<20}")
    print("-" * 60)

    for group in ['fault/cold', 'fault/warm', 'good/cold', 'good/warm']:
        r = results['per_group'][group]
        mean_val = np.mean(r['means'])
        std_val = np.std(r['means'])
        print(f"{group:<15} {r['accuracy']:>8.1f}%    {mean_val:.4f} +/- {std_val:.4f}")

    print("-" * 60)
    print(f"{'OVERALL':<15} {results['overall_accuracy']:>8.1f}%")
    print("=" * 60)

    # Compare with other methods
    print("\n" + "-" * 60)
    print("Comparison with Other Gating Methods:")
    print("-" * 60)
    print(f"  Mean Intensity: {results['overall_accuracy']:.1f}%")
    print(f"  CLIP Confidence: 88.8%")
    print(f"  CLIP Global: 87.5%")
    print(f"  FFT Best: 64.8%")

    if results['overall_accuracy'] > 88.8:
        print(f"\n  *** MEAN INTENSITY IS THE BEST! ***")
    elif results['overall_accuracy'] > 87.5:
        print(f"\n  => Mean Intensity is competitive with CLIP!")
    else:
        print(f"\n  => CLIP-based gating is still better")

    # Visualizations
    print("\n" + "=" * 60)
    print("Creating Visualizations...")
    print("=" * 60)

    visualize_intensity_distribution(results, output_dir)
    visualize_per_group_accuracy(results, output_dir)

    # Save results
    summary_path = output_dir / "INTENSITY_GATING_RESULTS.md"
    with open(summary_path, 'w') as f:
        f.write("# Mean Intensity-based Gating Analysis\n\n")
        f.write(f"## Domain: {args.domain}\n")
        f.write(f"## Samples per group: {args.num_samples}\n\n")

        f.write(f"## Optimal Threshold: {results['threshold']:.4f}\n\n")

        f.write("## Per-Group Results\n\n")
        f.write("| Group | Accuracy | Mean Intensity |\n")
        f.write("|-------|----------|----------------|\n")

        for group in ['fault/cold', 'fault/warm', 'good/cold', 'good/warm']:
            r = results['per_group'][group]
            mean_val = np.mean(r['means'])
            f.write(f"| {group} | {r['accuracy']:.1f}% | {mean_val:.4f} |\n")

        f.write(f"| **OVERALL** | **{results['overall_accuracy']:.1f}%** | - |\n\n")

        f.write("## Comparison\n\n")
        f.write("| Method | Accuracy |\n")
        f.write("|--------|----------|\n")
        f.write(f"| **Mean Intensity** | **{results['overall_accuracy']:.1f}%** |\n")
        f.write("| CLIP Confidence | 88.8% |\n")
        f.write("| CLIP Global | 87.5% |\n")
        f.write("| FFT Best | 64.8% |\n")

    print(f"\nSummary saved to: {summary_path}")
    print(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
