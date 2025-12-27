#!/usr/bin/env python3
"""
Analyze correlation between anomaly score and image intensity.

This script merges score data with intensity data to verify the hypothesis
that anomaly score is correlated with image amplitude.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import argparse


def load_and_merge_data(score_csv: Path, intensity_csv: Path) -> pd.DataFrame:
    """Load and merge score and intensity data."""
    scores = pd.read_csv(score_csv)
    intensity = pd.read_csv(intensity_csv)

    # Extract file index from image_name
    scores['file_idx'] = scores['image_name'].astype(int)

    # Merge on file_idx and label
    # Map gt_label_str to label
    scores['label'] = scores['gt_label_str'].replace({'fault': 'fault', 'good': 'good'})

    merged = scores.merge(
        intensity[['file_idx', 'label', 'mean', 'std', 'dynamic_range', 'full_condition']],
        on=['file_idx', 'label'],
        how='left'
    )

    # Add display condition
    merged['condition'] = merged['full_condition'].replace({
        'good_cold': 'normal_cold',
        'good_warm': 'normal_warm'
    })

    return merged


def plot_score_vs_intensity(df: pd.DataFrame, output_path: Path, title: str = ""):
    """Plot anomaly score vs image intensity."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    colors = {
        'fault_cold': '#ff6b6b',
        'fault_warm': '#c92a2a',
        'normal_cold': '#74b9ff',
        'normal_warm': '#0984e3'
    }

    # 1. Score vs Mean Intensity (scatter)
    ax1 = axes[0, 0]
    for cond in ['fault_cold', 'fault_warm', 'normal_cold', 'normal_warm']:
        subset = df[df['condition'] == cond]
        ax1.scatter(subset['mean'], subset['pred_score'],
                   c=colors.get(cond, 'gray'), alpha=0.5, s=20, label=cond)

    # Overall correlation
    corr, pval = stats.pearsonr(df['mean'].dropna(), df['pred_score'].dropna())
    ax1.set_xlabel('Image Mean Intensity')
    ax1.set_ylabel('Anomaly Score')
    ax1.set_title(f'{title} - Score vs Intensity\n(Pearson r={corr:.3f}, p={pval:.2e})')
    ax1.legend()

    # 2. Score vs Dynamic Range
    ax2 = axes[0, 1]
    for cond in ['fault_cold', 'fault_warm', 'normal_cold', 'normal_warm']:
        subset = df[df['condition'] == cond]
        ax2.scatter(subset['dynamic_range'], subset['pred_score'],
                   c=colors.get(cond, 'gray'), alpha=0.5, s=20, label=cond)

    corr_dr, pval_dr = stats.pearsonr(df['dynamic_range'].dropna(), df['pred_score'].dropna())
    ax2.set_xlabel('Image Dynamic Range (P95-P5)')
    ax2.set_ylabel('Anomaly Score')
    ax2.set_title(f'{title} - Score vs Dynamic Range\n(Pearson r={corr_dr:.3f}, p={pval_dr:.2e})')
    ax2.legend()

    # 3. Correlation by condition
    ax3 = axes[1, 0]
    conditions = ['fault_cold', 'fault_warm', 'normal_cold', 'normal_warm']
    corrs = []
    for cond in conditions:
        subset = df[df['condition'] == cond]
        if len(subset) > 10:
            r, _ = stats.pearsonr(subset['mean'].dropna(), subset['pred_score'].dropna())
            corrs.append(r)
        else:
            corrs.append(0)

    bars = ax3.bar(range(len(conditions)), corrs, color=[colors[c] for c in conditions])
    ax3.set_xticks(range(len(conditions)))
    ax3.set_xticklabels([c.replace('_', '\n') for c in conditions])
    ax3.set_ylabel('Pearson Correlation (r)')
    ax3.set_title('Correlation (Score vs Mean Intensity) by Condition')
    ax3.axhline(y=0, color='gray', linestyle='--')

    # Add correlation values on bars
    for bar, corr_val in zip(bars, corrs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{corr_val:.3f}', ha='center', va='bottom', fontsize=10)

    # 4. Mean intensity vs score by label (regression lines)
    ax4 = axes[1, 1]

    for label, marker, alpha in [('fault', 'o', 0.4), ('normal', 's', 0.4)]:
        label_map = 'fault' if label == 'fault' else 'good'
        subset = df[df['label'] == label_map]

        # Cold
        cold = subset[subset['condition'].str.contains('cold')]
        if len(cold) > 0:
            ax4.scatter(cold['mean'], cold['pred_score'],
                       c='red' if label == 'fault' else 'blue',
                       alpha=alpha, s=15, marker=marker, label=f'{label}_cold')

        # Warm
        warm = subset[subset['condition'].str.contains('warm')]
        if len(warm) > 0:
            ax4.scatter(warm['mean'], warm['pred_score'],
                       c='darkred' if label == 'fault' else 'darkblue',
                       alpha=alpha, s=15, marker=marker, label=f'{label}_warm')

    # Add regression line for all data
    valid = df.dropna(subset=['mean', 'pred_score'])
    z = np.polyfit(valid['mean'], valid['pred_score'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid['mean'].min(), valid['mean'].max(), 100)
    ax4.plot(x_line, p(x_line), 'k--', linewidth=2, label=f'Linear fit (slope={z[0]:.2f})')

    ax4.set_xlabel('Image Mean Intensity')
    ax4.set_ylabel('Anomaly Score')
    ax4.set_title('Score vs Intensity with Linear Fit')
    ax4.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    return corr, corr_dr


def print_correlation_analysis(df: pd.DataFrame):
    """Print detailed correlation analysis."""
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS: Anomaly Score vs Image Intensity")
    print("="*80)

    # Overall correlation
    corr_mean, p_mean = stats.pearsonr(df['mean'].dropna(), df['pred_score'].dropna())
    corr_dr, p_dr = stats.pearsonr(df['dynamic_range'].dropna(), df['pred_score'].dropna())

    print(f"\nOverall Correlation:")
    print(f"  Score vs Mean Intensity:   r = {corr_mean:.4f} (p = {p_mean:.2e})")
    print(f"  Score vs Dynamic Range:    r = {corr_dr:.4f} (p = {p_dr:.2e})")

    # Correlation by condition
    print(f"\nCorrelation by Condition (Score vs Mean Intensity):")
    for cond in ['fault_cold', 'fault_warm', 'normal_cold', 'normal_warm']:
        subset = df[df['condition'] == cond]
        if len(subset) > 10:
            r, p = stats.pearsonr(subset['mean'].dropna(), subset['pred_score'].dropna())
            print(f"  {cond:15s}: r = {r:.4f} (p = {p:.2e})")

    # Key insight: what happens if we normalize by intensity?
    print("\n" + "="*80)
    print("INSIGHT: Score Normalized by Intensity")
    print("="*80)

    df['score_normalized'] = df['pred_score'] / df['mean']

    print("\nMean Normalized Score by Condition:")
    for cond in ['fault_cold', 'fault_warm', 'normal_cold', 'normal_warm']:
        subset = df[df['condition'] == cond]
        orig_score = subset['pred_score'].mean()
        norm_score = subset['score_normalized'].mean()
        print(f"  {cond:15s}: Original={orig_score:.3f}, Normalized={norm_score:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-csv", type=Path, required=True)
    parser.add_argument("--intensity-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                       default=Path("examples/hdmap/EDA/winclip_baseline_post_analysis"))
    parser.add_argument("--title", type=str, default="")
    args = parser.parse_args()

    df = load_and_merge_data(args.score_csv, args.intensity_csv)

    print_correlation_analysis(df)

    output_name = args.score_csv.stem + "_vs_intensity"
    plot_score_vs_intensity(df, args.output_dir / f"{output_name}.png", args.title or output_name)


if __name__ == "__main__":
    main()
