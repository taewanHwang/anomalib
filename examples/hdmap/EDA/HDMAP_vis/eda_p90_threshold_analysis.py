"""Analyze p90 threshold using only normal (good) data.

Question: Can we set threshold = max(p90 of good/cold)?
This assumes we only have access to normal training data.

Usage:
    python eda_p90_threshold_analysis.py
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


def analyze_threshold_strategy(domain: str):
    """Analyze threshold strategy for a domain."""
    dataset_root = "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_tiff_minmax"
    dataset = HDMAPDataset(
        root=dataset_root,
        domain=domain,
        split="test",
        target_size=(240, 240),
        resize_method="resize",
    )

    # Compute p90 for all groups
    groups = {
        'fault/cold': list(range(0, 500)),
        'fault/warm': list(range(500, 1000)),
        'good/cold': list(range(1000, 1500)),
        'good/warm': list(range(1500, 2000)),
    }

    p90_values = {}
    for group, indices in groups.items():
        p90_values[group] = compute_p90(dataset, indices)

    # Statistics
    print(f"\n{'='*70}")
    print(f"Domain: {domain}")
    print(f"{'='*70}")

    print(f"\n{'Group':<15} {'Min':<10} {'Max':<10} {'Mean':<10} {'Std':<10}")
    print("-" * 55)
    for group in groups:
        vals = p90_values[group]
        print(f"{group:<15} {min(vals):.4f}    {max(vals):.4f}    {np.mean(vals):.4f}    {np.std(vals):.4f}")

    # Strategy 1: threshold = max(good/cold p90)
    thresh_max_cold = max(p90_values['good/cold'])

    # Strategy 2: threshold = mean(good/cold p90) + 3*std
    mean_cold = np.mean(p90_values['good/cold'])
    std_cold = np.std(p90_values['good/cold'])
    thresh_3sigma = mean_cold + 3 * std_cold

    # Strategy 3: threshold = (max(good/cold) + min(good/warm)) / 2
    thresh_midpoint = (max(p90_values['good/cold']) + min(p90_values['good/warm'])) / 2

    print(f"\n{'Threshold Strategies (using only good data)':}")
    print("-" * 55)
    print(f"  1. max(good/cold):           {thresh_max_cold:.4f}")
    print(f"  2. mean + 3*std (good/cold): {thresh_3sigma:.4f}")
    print(f"  3. midpoint (good/cold, good/warm): {thresh_midpoint:.4f}")

    # Gap analysis
    gap = min(p90_values['good/warm']) - max(p90_values['good/cold'])
    print(f"\n  Gap between good/cold max and good/warm min: {gap:.4f}")
    if gap > 0:
        print(f"  => Clean separation possible!")
    else:
        overlap_count = sum(1 for v in p90_values['good/cold'] if v > min(p90_values['good/warm']))
        print(f"  => OVERLAP! {overlap_count} good/cold samples exceed min(good/warm)")

    # Evaluate each threshold
    print(f"\n{'Gating Accuracy by Threshold':}")
    print("-" * 70)
    print(f"{'Threshold':<25} {'fault/cold':<12} {'fault/warm':<12} {'good/cold':<12} {'good/warm':<12} {'Overall':<10}")
    print("-" * 70)

    for name, thresh in [
        ('max(good/cold)', thresh_max_cold),
        ('mean+3std(good/cold)', thresh_3sigma),
        ('midpoint', thresh_midpoint),
    ]:
        accs = {}
        total_correct = 0
        total = 0

        for group in groups:
            gt = 'cold' if 'cold' in group else 'warm'
            vals = p90_values[group]

            if gt == 'cold':
                correct = sum(1 for v in vals if v <= thresh)
            else:
                correct = sum(1 for v in vals if v > thresh)

            accs[group] = correct / len(vals) * 100
            total_correct += correct
            total += len(vals)

        overall = total_correct / total * 100
        print(f"{name:<25} {accs['fault/cold']:>9.1f}%   {accs['fault/warm']:>9.1f}%   "
              f"{accs['good/cold']:>9.1f}%   {accs['good/warm']:>9.1f}%   {overall:>7.1f}%")

    return {
        'domain': domain,
        'thresh_max_cold': thresh_max_cold,
        'thresh_3sigma': thresh_3sigma,
        'thresh_midpoint': thresh_midpoint,
        'gap': gap,
        'p90_values': p90_values,
    }


def main():
    print("=" * 70)
    print("P90 Threshold Analysis: Using Only Normal (Good) Data")
    print("=" * 70)
    print("\nQuestion: Can we use max(p90 of good/cold) as threshold?")
    print("Assumption: We only have access to normal training data.")

    results = {}
    for domain in ['domain_A', 'domain_B', 'domain_C', 'domain_D']:
        results[domain] = analyze_threshold_strategy(domain)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Recommended Thresholds per Domain")
    print("=" * 70)
    print(f"\n{'Domain':<12} {'max(good/cold)':<18} {'Gap':<12} {'Recommended':<15}")
    print("-" * 60)

    for domain in ['domain_A', 'domain_B', 'domain_C', 'domain_D']:
        r = results[domain]
        recommended = r['thresh_max_cold'] if r['gap'] > 0 else r['thresh_midpoint']
        gap_status = "Clean" if r['gap'] > 0 else f"Overlap"
        print(f"{domain:<12} {r['thresh_max_cold']:<18.4f} {gap_status:<12} {recommended:.4f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    all_clean = all(results[d]['gap'] > 0 for d in results)
    if all_clean:
        print("All domains have clean separation between good/cold and good/warm.")
        print("=> max(good/cold p90) is a safe threshold for all domains!")
    else:
        overlapping = [d for d in results if results[d]['gap'] <= 0]
        print(f"Domains with overlap: {overlapping}")
        print("=> Consider using midpoint or mean+3std for these domains.")


if __name__ == "__main__":
    main()
