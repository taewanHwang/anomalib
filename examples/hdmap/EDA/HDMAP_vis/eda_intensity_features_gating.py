"""EDA: Intensity Feature-based Gating for Cold/Warm Condition Selection.

Test multiple intensity features:
- mean, median
- percentiles: p10, p25, p75, p90
- std, min, max

Usage:
    python eda_intensity_features_gating.py --domain domain_C
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Callable

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


# Feature extraction functions
def get_feature_functions() -> Dict[str, Callable]:
    """Return dict of feature name -> extraction function."""
    return {
        'mean': lambda x: x.mean().item(),
        'median': lambda x: x.median().item(),
        'std': lambda x: x.std().item(),
        'min': lambda x: x.min().item(),
        'max': lambda x: x.max().item(),
        'p10': lambda x: np.percentile(x.numpy(), 10),
        'p25': lambda x: np.percentile(x.numpy(), 25),
        'p75': lambda x: np.percentile(x.numpy(), 75),
        'p90': lambda x: np.percentile(x.numpy(), 90),
        'range': lambda x: (x.max() - x.min()).item(),
        'iqr': lambda x: np.percentile(x.numpy(), 75) - np.percentile(x.numpy(), 25),
    }


def compute_features(dataset, indices: List[int], feature_funcs: Dict) -> Dict[str, List[float]]:
    """Compute all features for each sample."""
    results = {name: [] for name in feature_funcs}

    for idx in indices:
        sample = dataset[idx]
        img = sample.image

        for name, func in feature_funcs.items():
            value = func(img)
            results[name].append(value)

    return results


def find_optimal_threshold(
    cold_values: List[float],
    warm_values: List[float],
) -> Tuple[float, float, str]:
    """Find optimal threshold and direction.

    Returns:
        (threshold, accuracy, direction)
        direction: 'cold_below' if cold < threshold, 'cold_above' if cold > threshold
    """
    all_values = cold_values + warm_values
    min_val, max_val = min(all_values), max(all_values)

    thresholds = np.linspace(min_val, max_val, 1000)

    best_threshold = 0
    best_accuracy = 0
    best_direction = 'cold_below'

    for thresh in thresholds:
        # Try cold_below (cold < threshold < warm)
        cold_correct_below = sum(1 for v in cold_values if v < thresh)
        warm_correct_below = sum(1 for v in warm_values if v >= thresh)
        acc_below = (cold_correct_below + warm_correct_below) / len(all_values)

        # Try cold_above (warm < threshold < cold)
        cold_correct_above = sum(1 for v in cold_values if v >= thresh)
        warm_correct_above = sum(1 for v in warm_values if v < thresh)
        acc_above = (cold_correct_above + warm_correct_above) / len(all_values)

        if acc_below > best_accuracy:
            best_accuracy = acc_below
            best_threshold = thresh
            best_direction = 'cold_below'

        if acc_above > best_accuracy:
            best_accuracy = acc_above
            best_threshold = thresh
            best_direction = 'cold_above'

    return best_threshold, best_accuracy, best_direction


def evaluate_feature(
    feature_name: str,
    cold_values: List[float],
    warm_values: List[float],
    per_group_values: Dict[str, List[float]],
) -> Dict:
    """Evaluate a single feature for gating."""

    threshold, overall_acc, direction = find_optimal_threshold(cold_values, warm_values)

    # Per-group accuracy
    per_group_acc = {}
    for group_name, values in per_group_values.items():
        gt = 'cold' if 'cold' in group_name else 'warm'

        if direction == 'cold_below':
            if gt == 'cold':
                correct = sum(1 for v in values if v < threshold)
            else:
                correct = sum(1 for v in values if v >= threshold)
        else:  # cold_above
            if gt == 'cold':
                correct = sum(1 for v in values if v >= threshold)
            else:
                correct = sum(1 for v in values if v < threshold)

        per_group_acc[group_name] = correct / len(values) * 100

    return {
        'feature': feature_name,
        'threshold': threshold,
        'direction': direction,
        'overall_acc': overall_acc * 100,
        'per_group_acc': per_group_acc,
        'cold_mean': np.mean(cold_values),
        'cold_std': np.std(cold_values),
        'warm_mean': np.mean(warm_values),
        'warm_std': np.std(warm_values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="domain_C")
    args = parser.parse_args()

    # Setup
    dataset_root = "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_tiff_minmax"
    output_dir = Path(__file__).parent / args.domain / "intensity_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Domain: {args.domain}")
    print(f"Output: {output_dir}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(dataset_root, args.domain)
    print(f"Dataset size: {len(dataset)}")

    # Use ALL samples for accurate evaluation
    test_indices = {
        'fault/cold': list(range(0, 500)),
        'fault/warm': list(range(500, 1000)),
        'good/cold': list(range(1000, 1500)),
        'good/warm': list(range(1500, 2000)),
    }

    for group, indices in test_indices.items():
        print(f"  {group}: {len(indices)} samples")

    # Get feature functions
    feature_funcs = get_feature_functions()
    print(f"\nTesting {len(feature_funcs)} features: {list(feature_funcs.keys())}")

    # Compute features for all groups
    print("\nComputing features...")
    all_features = {}
    for group_name, indices in test_indices.items():
        print(f"  Processing {group_name}...")
        all_features[group_name] = compute_features(dataset, indices, feature_funcs)

    # Evaluate each feature
    print("\n" + "=" * 100)
    print("EVALUATING INTENSITY FEATURES FOR GATING")
    print("=" * 100)

    results = []
    for feature_name in feature_funcs.keys():
        # Combine cold and warm
        cold_values = all_features['fault/cold'][feature_name] + all_features['good/cold'][feature_name]
        warm_values = all_features['fault/warm'][feature_name] + all_features['good/warm'][feature_name]

        per_group_values = {
            group: all_features[group][feature_name]
            for group in test_indices.keys()
        }

        result = evaluate_feature(feature_name, cold_values, warm_values, per_group_values)
        results.append(result)

    # Sort by overall accuracy
    results.sort(key=lambda x: x['overall_acc'], reverse=True)

    # Print results table
    print("\n" + "-" * 100)
    print(f"{'Rank':<5} {'Feature':<10} {'Overall':<10} {'fault/cold':<12} {'fault/warm':<12} {'good/cold':<12} {'good/warm':<12} {'Threshold':<12}")
    print("-" * 100)

    for i, r in enumerate(results):
        print(f"{i+1:<5} {r['feature']:<10} {r['overall_acc']:>7.1f}%   "
              f"{r['per_group_acc']['fault/cold']:>9.1f}%   "
              f"{r['per_group_acc']['fault/warm']:>9.1f}%   "
              f"{r['per_group_acc']['good/cold']:>9.1f}%   "
              f"{r['per_group_acc']['good/warm']:>9.1f}%   "
              f"{r['threshold']:>10.4f}")

    print("-" * 100)

    # Best feature
    best = results[0]
    print(f"\n*** BEST FEATURE: {best['feature']} with {best['overall_acc']:.1f}% accuracy ***")
    print(f"    Threshold: {best['threshold']:.4f} ({best['direction']})")
    print(f"    Cold: {best['cold_mean']:.4f} +/- {best['cold_std']:.4f}")
    print(f"    Warm: {best['warm_mean']:.4f} +/- {best['warm_std']:.4f}")

    # Comparison with CLIP
    print("\n" + "-" * 60)
    print("Comparison with CLIP-based Gating:")
    print("-" * 60)
    print(f"  Best Intensity ({best['feature']}): {best['overall_acc']:.1f}%")
    print(f"  CLIP Confidence: 88.8%")
    print(f"  Improvement: +{best['overall_acc'] - 88.8:.1f}%p")

    # Visualize top features
    print("\n" + "=" * 60)
    print("Creating Visualizations...")
    print("=" * 60)

    # Bar chart of all features
    fig, ax = plt.subplots(figsize=(14, 6))

    features = [r['feature'] for r in results]
    accuracies = [r['overall_acc'] for r in results]
    colors = ['green' if a > 95 else 'blue' if a > 90 else 'orange' if a > 85 else 'red' for a in accuracies]

    bars = ax.bar(features, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(88.8, color='purple', linestyle='--', linewidth=2, label='CLIP Confidence (88.8%)')
    ax.axhline(95, color='green', linestyle=':', linewidth=2, label='95% Target')

    for bar, acc in zip(bars, accuracies):
        ax.annotate(f'{acc:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

    ax.set_ylabel('Gating Accuracy (%)')
    ax.set_title(f'Intensity Feature Comparison for Gating - {args.domain}')
    ax.set_ylim(0, 105)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = output_dir / "intensity_features_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    # Per-group heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    groups = ['fault/cold', 'fault/warm', 'good/cold', 'good/warm']
    data = np.array([[r['per_group_acc'][g] for g in groups] for r in results])

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=50, vmax=100)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([r['feature'] for r in results])

    # Add text annotations
    for i in range(len(results)):
        for j in range(len(groups)):
            text = ax.text(j, i, f'{data[i, j]:.1f}%', ha='center', va='center',
                          color='white' if data[i, j] < 70 else 'black', fontsize=9)

    ax.set_title(f'Per-Group Gating Accuracy by Feature - {args.domain}')
    plt.colorbar(im, ax=ax, label='Accuracy (%)')
    plt.tight_layout()

    save_path = output_dir / "intensity_features_heatmap.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    # Save summary
    summary_path = output_dir / "INTENSITY_FEATURES_RESULTS.md"
    with open(summary_path, 'w') as f:
        f.write(f"# Intensity Feature-based Gating Analysis\n\n")
        f.write(f"## Domain: {args.domain}\n\n")

        f.write("## Results (sorted by overall accuracy)\n\n")
        f.write("| Rank | Feature | Overall | fault/cold | fault/warm | good/cold | good/warm | Threshold |\n")
        f.write("|------|---------|---------|------------|------------|-----------|-----------|----------|\n")

        for i, r in enumerate(results):
            f.write(f"| {i+1} | {r['feature']} | {r['overall_acc']:.1f}% | "
                   f"{r['per_group_acc']['fault/cold']:.1f}% | "
                   f"{r['per_group_acc']['fault/warm']:.1f}% | "
                   f"{r['per_group_acc']['good/cold']:.1f}% | "
                   f"{r['per_group_acc']['good/warm']:.1f}% | "
                   f"{r['threshold']:.4f} |\n")

        f.write(f"\n## Best Feature: {best['feature']}\n")
        f.write(f"- Accuracy: {best['overall_acc']:.1f}%\n")
        f.write(f"- Threshold: {best['threshold']:.4f}\n")
        f.write(f"- Direction: {best['direction']}\n")
        f.write(f"- vs CLIP Confidence: +{best['overall_acc'] - 88.8:.1f}%p\n")

    print(f"\nSummary saved to: {summary_path}")
    print(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
