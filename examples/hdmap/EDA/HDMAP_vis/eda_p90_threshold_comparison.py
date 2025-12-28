"""Compare two threshold strategies for p90-based gating.

Strategy 1: midpoint = (max(good/cold) + min(good/warm)) / 2
Strategy 2: mean+3std = mean(good/cold) + 3 * std(good/cold)

Assumption: Normal data has cold/warm labels.

Usage:
    python eda_p90_threshold_comparison.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[4]))

from anomalib.data.datasets.image.hdmap import HDMAPDataset


def compute_p90(dataset, indices):
    """Compute p90 for each sample."""
    values = []
    for idx in indices:
        sample = dataset[idx]
        p90 = np.percentile(sample.image.numpy(), 90)
        values.append(p90)
    return values


def evaluate_threshold(p90_values, threshold, groups):
    """Evaluate gating accuracy for a threshold."""
    results = {}
    total_correct = 0
    total = 0

    for group in groups:
        gt = 'cold' if 'cold' in group else 'warm'
        vals = p90_values[group]

        if gt == 'cold':
            correct = sum(1 for v in vals if v <= threshold)
        else:
            correct = sum(1 for v in vals if v > threshold)

        results[group] = {
            'correct': correct,
            'total': len(vals),
            'accuracy': correct / len(vals) * 100
        }
        total_correct += correct
        total += len(vals)

    results['overall'] = {
        'correct': total_correct,
        'total': total,
        'accuracy': total_correct / total * 100
    }
    return results


def main():
    dataset_root = "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_tiff_minmax"
    domains = ['domain_A', 'domain_B', 'domain_C', 'domain_D']
    groups = ['fault/cold', 'fault/warm', 'good/cold', 'good/warm']

    all_results = {}

    for domain in domains:
        dataset = HDMAPDataset(
            root=dataset_root,
            domain=domain,
            split="test",
            target_size=(240, 240),
            resize_method="resize",
        )

        # Compute p90 for all groups
        indices = {
            'fault/cold': list(range(0, 500)),
            'fault/warm': list(range(500, 1000)),
            'good/cold': list(range(1000, 1500)),
            'good/warm': list(range(1500, 2000)),
        }

        p90_values = {group: compute_p90(dataset, idx) for group, idx in indices.items()}

        # Compute thresholds from good data only
        good_cold_vals = p90_values['good/cold']
        good_warm_vals = p90_values['good/warm']

        thresh_midpoint = (max(good_cold_vals) + min(good_warm_vals)) / 2
        thresh_mean3std = np.mean(good_cold_vals) + 3 * np.std(good_cold_vals)

        # Evaluate both thresholds
        results_midpoint = evaluate_threshold(p90_values, thresh_midpoint, groups)
        results_mean3std = evaluate_threshold(p90_values, thresh_mean3std, groups)

        all_results[domain] = {
            'midpoint': {
                'threshold': thresh_midpoint,
                'results': results_midpoint,
            },
            'mean3std': {
                'threshold': thresh_mean3std,
                'results': results_mean3std,
            },
            'p90_stats': {
                'good_cold_max': max(good_cold_vals),
                'good_warm_min': min(good_warm_vals),
                'good_cold_mean': np.mean(good_cold_vals),
                'good_cold_std': np.std(good_cold_vals),
            }
        }

    # Print results
    print("=" * 100)
    print("P90 THRESHOLD COMPARISON: midpoint vs mean+3std")
    print("=" * 100)
    print("\nAssumption: Normal (good) data has cold/warm labels")
    print("  - midpoint = (max(good/cold) + min(good/warm)) / 2")
    print("  - mean+3std = mean(good/cold) + 3 * std(good/cold)")

    # Table 1: Per-domain thresholds
    print("\n" + "-" * 80)
    print("THRESHOLDS")
    print("-" * 80)
    print(f"{'Domain':<12} {'max(g/c)':<12} {'min(g/w)':<12} {'midpoint':<12} {'mean+3std':<12}")
    print("-" * 80)
    for domain in domains:
        r = all_results[domain]
        print(f"{domain:<12} "
              f"{r['p90_stats']['good_cold_max']:<12.4f} "
              f"{r['p90_stats']['good_warm_min']:<12.4f} "
              f"{r['midpoint']['threshold']:<12.4f} "
              f"{r['mean3std']['threshold']:<12.4f}")

    # Table 2: Per-group accuracy comparison
    print("\n" + "-" * 100)
    print("PER-GROUP ACCURACY: midpoint")
    print("-" * 100)
    print(f"{'Domain':<12} {'fault/cold':<12} {'fault/warm':<12} {'good/cold':<12} {'good/warm':<12} {'OVERALL':<12}")
    print("-" * 100)
    for domain in domains:
        r = all_results[domain]['midpoint']['results']
        print(f"{domain:<12} "
              f"{r['fault/cold']['accuracy']:>9.1f}%   "
              f"{r['fault/warm']['accuracy']:>9.1f}%   "
              f"{r['good/cold']['accuracy']:>9.1f}%   "
              f"{r['good/warm']['accuracy']:>9.1f}%   "
              f"{r['overall']['accuracy']:>9.1f}%")

    print("\n" + "-" * 100)
    print("PER-GROUP ACCURACY: mean+3std")
    print("-" * 100)
    print(f"{'Domain':<12} {'fault/cold':<12} {'fault/warm':<12} {'good/cold':<12} {'good/warm':<12} {'OVERALL':<12}")
    print("-" * 100)
    for domain in domains:
        r = all_results[domain]['mean3std']['results']
        print(f"{domain:<12} "
              f"{r['fault/cold']['accuracy']:>9.1f}%   "
              f"{r['fault/warm']['accuracy']:>9.1f}%   "
              f"{r['good/cold']['accuracy']:>9.1f}%   "
              f"{r['good/warm']['accuracy']:>9.1f}%   "
              f"{r['overall']['accuracy']:>9.1f}%")

    # Table 3: Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY: Overall Accuracy Comparison")
    print("=" * 60)
    print(f"{'Domain':<12} {'midpoint':<15} {'mean+3std':<15} {'Better':<15}")
    print("-" * 60)

    midpoint_wins = 0
    mean3std_wins = 0

    for domain in domains:
        mid_acc = all_results[domain]['midpoint']['results']['overall']['accuracy']
        m3s_acc = all_results[domain]['mean3std']['results']['overall']['accuracy']

        if mid_acc > m3s_acc:
            better = "midpoint"
            midpoint_wins += 1
        elif m3s_acc > mid_acc:
            better = "mean+3std"
            mean3std_wins += 1
        else:
            better = "tie"

        print(f"{domain:<12} {mid_acc:>12.1f}%   {m3s_acc:>12.1f}%   {better:<15}")

    print("-" * 60)

    # Average
    avg_mid = np.mean([all_results[d]['midpoint']['results']['overall']['accuracy'] for d in domains])
    avg_m3s = np.mean([all_results[d]['mean3std']['results']['overall']['accuracy'] for d in domains])

    print(f"{'AVERAGE':<12} {avg_mid:>12.1f}%   {avg_m3s:>12.1f}%")
    print("=" * 60)

    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    if avg_mid > avg_m3s:
        print(f"  => midpoint is better on average ({avg_mid:.1f}% vs {avg_m3s:.1f}%)")
        print(f"  => Wins: midpoint {midpoint_wins}, mean+3std {mean3std_wins}")
    else:
        print(f"  => mean+3std is better on average ({avg_m3s:.1f}% vs {avg_mid:.1f}%)")
        print(f"  => Wins: mean+3std {mean3std_wins}, midpoint {midpoint_wins}")

    # fault/cold focus (the hardest group)
    print("\n" + "-" * 60)
    print("FOCUS: fault/cold (hardest group)")
    print("-" * 60)
    print(f"{'Domain':<12} {'midpoint':<15} {'mean+3std':<15}")
    for domain in domains:
        mid_fc = all_results[domain]['midpoint']['results']['fault/cold']['accuracy']
        m3s_fc = all_results[domain]['mean3std']['results']['fault/cold']['accuracy']
        print(f"{domain:<12} {mid_fc:>12.1f}%   {m3s_fc:>12.1f}%")

    avg_mid_fc = np.mean([all_results[d]['midpoint']['results']['fault/cold']['accuracy'] for d in domains])
    avg_m3s_fc = np.mean([all_results[d]['mean3std']['results']['fault/cold']['accuracy'] for d in domains])
    print(f"{'AVERAGE':<12} {avg_mid_fc:>12.1f}%   {avg_m3s_fc:>12.1f}%")


if __name__ == "__main__":
    main()
