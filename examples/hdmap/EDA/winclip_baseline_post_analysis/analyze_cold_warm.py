#!/usr/bin/env python3
"""
WinCLIP Cold Start vs Warmed Up Analysis.

Hypothesis:
- Test data has 2000 samples: first 1000 fault, last 1000 normal
- Within each 1000: first 500 = cold start, last 500 = warmed up
- Cold start vs cold start: good discrimination
- Warmed up vs warmed up: good discrimination
- Mixed: poor discrimination (cold start fault overlaps with warmed up normal)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
import argparse


def load_scores(csv_path: Path) -> pd.DataFrame:
    """Load score CSV and add condition labels."""
    df = pd.read_csv(csv_path)

    # Sort by image path to ensure correct order
    df = df.sort_values('image_path').reset_index(drop=True)

    # Add row index
    df['row_idx'] = range(len(df))

    # Determine condition based on file index
    def get_condition(row):
        # Extract file index from image_name (e.g., "000123" -> 123)
        file_idx = int(row['image_name'])

        if row['gt_label_str'] == 'fault':
            if file_idx < 500:
                return 'fault_cold'
            else:
                return 'fault_warm'
        else:  # good/normal
            if file_idx < 500:
                return 'normal_cold'
            else:
                return 'normal_warm'

    df['condition'] = df.apply(get_condition, axis=1)

    return df


def compute_auroc_by_condition(df: pd.DataFrame) -> dict:
    """Compute AUROC for different condition combinations."""
    results = {}

    # All data
    results['all'] = roc_auc_score(df['gt_label'], df['pred_score'])

    # Cold start only (fault_cold vs normal_cold)
    cold_df = df[df['condition'].isin(['fault_cold', 'normal_cold'])]
    if len(cold_df) > 0:
        results['cold_only'] = roc_auc_score(cold_df['gt_label'], cold_df['pred_score'])

    # Warmed up only (fault_warm vs normal_warm)
    warm_df = df[df['condition'].isin(['fault_warm', 'normal_warm'])]
    if len(warm_df) > 0:
        results['warm_only'] = roc_auc_score(warm_df['gt_label'], warm_df['pred_score'])

    # Cross condition: cold fault vs warm normal (expected: poor)
    cross1_df = df[df['condition'].isin(['fault_cold', 'normal_warm'])]
    if len(cross1_df) > 0:
        results['cold_fault_vs_warm_normal'] = roc_auc_score(cross1_df['gt_label'], cross1_df['pred_score'])

    # Cross condition: warm fault vs cold normal (expected: good)
    cross2_df = df[df['condition'].isin(['fault_warm', 'normal_cold'])]
    if len(cross2_df) > 0:
        results['warm_fault_vs_cold_normal'] = roc_auc_score(cross2_df['gt_label'], cross2_df['pred_score'])

    return results


def plot_score_by_condition(df: pd.DataFrame, output_path: Path, title: str = ""):
    """Plot score distribution with condition annotations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Time series plot with condition regions
    ax1 = axes[0, 0]
    colors = {
        'fault_cold': 'red',
        'fault_warm': 'darkred',
        'normal_cold': 'blue',
        'normal_warm': 'darkblue'
    }

    for condition in ['fault_cold', 'fault_warm', 'normal_cold', 'normal_warm']:
        mask = df['condition'] == condition
        ax1.scatter(df.loc[mask, 'row_idx'], df.loc[mask, 'pred_score'],
                   c=colors[condition], alpha=0.5, s=5, label=condition)

    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Anomaly Score')
    ax1.set_title(f'{title} - Score by Sample Index')
    ax1.legend()
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # 2. Box plot by condition
    ax2 = axes[0, 1]
    conditions = ['fault_cold', 'fault_warm', 'normal_cold', 'normal_warm']
    box_data = [df[df['condition'] == c]['pred_score'].values for c in conditions]
    bp = ax2.boxplot(box_data, labels=['Fault\nCold', 'Fault\nWarm', 'Normal\nCold', 'Normal\nWarm'],
                     patch_artist=True)
    box_colors = ['#ff6b6b', '#c92a2a', '#74b9ff', '#0984e3']
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('Anomaly Score')
    ax2.set_title(f'{title} - Score Distribution by Condition')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # 3. Histogram comparison: Same condition (Cold vs Cold, Warm vs Warm)
    ax3 = axes[1, 0]
    fault_cold = df[df['condition'] == 'fault_cold']['pred_score']
    normal_cold = df[df['condition'] == 'normal_cold']['pred_score']
    fault_warm = df[df['condition'] == 'fault_warm']['pred_score']
    normal_warm = df[df['condition'] == 'normal_warm']['pred_score']

    ax3.hist(fault_cold, bins=30, alpha=0.5, label=f'Fault Cold (n={len(fault_cold)})', color='red')
    ax3.hist(normal_cold, bins=30, alpha=0.5, label=f'Normal Cold (n={len(normal_cold)})', color='blue')
    ax3.set_xlabel('Anomaly Score')
    ax3.set_ylabel('Count')
    ax3.set_title('Cold Start Only - Good Separation Expected')
    ax3.legend()

    ax4 = axes[1, 1]
    ax4.hist(fault_warm, bins=30, alpha=0.5, label=f'Fault Warm (n={len(fault_warm)})', color='darkred')
    ax4.hist(normal_warm, bins=30, alpha=0.5, label=f'Normal Warm (n={len(normal_warm)})', color='darkblue')
    ax4.set_xlabel('Anomaly Score')
    ax4.set_ylabel('Count')
    ax4.set_title('Warmed Up Only - Good Separation Expected')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_cross_condition_analysis(df: pd.DataFrame, output_path: Path, title: str = ""):
    """Plot cross-condition overlap analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fault_cold = df[df['condition'] == 'fault_cold']['pred_score']
    normal_warm = df[df['condition'] == 'normal_warm']['pred_score']
    fault_warm = df[df['condition'] == 'fault_warm']['pred_score']
    normal_cold = df[df['condition'] == 'normal_cold']['pred_score']

    # Cross 1: Cold Fault vs Warm Normal (problematic)
    ax1 = axes[0]
    ax1.hist(fault_cold, bins=30, alpha=0.6, label=f'Fault Cold (n={len(fault_cold)})', color='red')
    ax1.hist(normal_warm, bins=30, alpha=0.6, label=f'Normal Warm (n={len(normal_warm)})', color='darkblue')
    ax1.set_xlabel('Anomaly Score')
    ax1.set_ylabel('Count')
    ax1.set_title('Cold Fault vs Warm Normal - OVERLAP EXPECTED')
    ax1.legend()

    # Cross 2: Warm Fault vs Cold Normal (good)
    ax2 = axes[1]
    ax2.hist(fault_warm, bins=30, alpha=0.6, label=f'Fault Warm (n={len(fault_warm)})', color='darkred')
    ax2.hist(normal_cold, bins=30, alpha=0.6, label=f'Normal Cold (n={len(normal_cold)})', color='blue')
    ax2.set_xlabel('Anomaly Score')
    ax2.set_ylabel('Count')
    ax2.set_title('Warm Fault vs Cold Normal - Good Separation Expected')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_roc_comparison(df: pd.DataFrame, output_path: Path, title: str = ""):
    """Plot ROC curves for different condition combinations."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # All data
    fpr, tpr, _ = roc_curve(df['gt_label'], df['pred_score'])
    auc_all = roc_auc_score(df['gt_label'], df['pred_score'])
    ax.plot(fpr, tpr, label=f'All Data (AUC={auc_all:.3f})', linewidth=2)

    # Cold only
    cold_df = df[df['condition'].isin(['fault_cold', 'normal_cold'])]
    if len(cold_df) > 0:
        fpr, tpr, _ = roc_curve(cold_df['gt_label'], cold_df['pred_score'])
        auc = roc_auc_score(cold_df['gt_label'], cold_df['pred_score'])
        ax.plot(fpr, tpr, label=f'Cold Only (AUC={auc:.3f})', linewidth=2)

    # Warm only
    warm_df = df[df['condition'].isin(['fault_warm', 'normal_warm'])]
    if len(warm_df) > 0:
        fpr, tpr, _ = roc_curve(warm_df['gt_label'], warm_df['pred_score'])
        auc = roc_auc_score(warm_df['gt_label'], warm_df['pred_score'])
        ax.plot(fpr, tpr, label=f'Warm Only (AUC={auc:.3f})', linewidth=2)

    # Cross: cold fault vs warm normal
    cross1_df = df[df['condition'].isin(['fault_cold', 'normal_warm'])]
    if len(cross1_df) > 0:
        fpr, tpr, _ = roc_curve(cross1_df['gt_label'], cross1_df['pred_score'])
        auc = roc_auc_score(cross1_df['gt_label'], cross1_df['pred_score'])
        ax.plot(fpr, tpr, label=f'Cold Fault vs Warm Normal (AUC={auc:.3f})', linewidth=2, linestyle='--')

    # Cross: warm fault vs cold normal
    cross2_df = df[df['condition'].isin(['fault_warm', 'normal_cold'])]
    if len(cross2_df) > 0:
        fpr, tpr, _ = roc_curve(cross2_df['gt_label'], cross2_df['pred_score'])
        auc = roc_auc_score(cross2_df['gt_label'], cross2_df['pred_score'])
        ax.plot(fpr, tpr, label=f'Warm Fault vs Cold Normal (AUC={auc:.3f})', linewidth=2, linestyle='--')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{title} - ROC Curve Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def analyze_single_experiment(csv_path: Path, output_dir: Path, exp_name: str):
    """Analyze a single experiment CSV."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {exp_name}")
    print(f"{'='*60}")

    df = load_scores(csv_path)

    # Print statistics
    print(f"\nData Summary:")
    print(f"  Total samples: {len(df)}")
    for cond in ['fault_cold', 'fault_warm', 'normal_cold', 'normal_warm']:
        n = len(df[df['condition'] == cond])
        mean_score = df[df['condition'] == cond]['pred_score'].mean()
        std_score = df[df['condition'] == cond]['pred_score'].std()
        print(f"  {cond}: n={n}, score={mean_score:.3f} ± {std_score:.3f}")

    # Compute AUROC by condition
    auroc_results = compute_auroc_by_condition(df)
    print(f"\nAUROC by Condition:")
    for key, value in auroc_results.items():
        print(f"  {key}: {value*100:.2f}%")

    # Generate plots
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_score_by_condition(df, output_dir / f"{exp_name}_score_by_condition.png", exp_name)
    plot_cross_condition_analysis(df, output_dir / f"{exp_name}_cross_condition.png", exp_name)
    plot_roc_comparison(df, output_dir / f"{exp_name}_roc_comparison.png", exp_name)

    return auroc_results


def main():
    parser = argparse.ArgumentParser(description="Analyze WinCLIP results by cold/warm condition")
    parser.add_argument("--csv", type=Path, help="Single CSV file to analyze")
    parser.add_argument("--result-dir", type=Path, help="Result directory containing scores/")
    parser.add_argument("--output-dir", type=Path,
                       default=Path("examples/hdmap/EDA/winclip_baseline_post_analysis"),
                       help="Output directory for visualizations")
    args = parser.parse_args()

    output_dir = args.output_dir

    if args.csv:
        exp_name = args.csv.stem
        analyze_single_experiment(args.csv, output_dir, exp_name)

    elif args.result_dir:
        # Find all score CSVs
        scores_dir = args.result_dir / "scores"
        if not scores_dir.exists():
            # Try to find scores in subdirectories
            for subdir in args.result_dir.iterdir():
                if subdir.is_dir():
                    scores_subdir = subdir / "scores"
                    if scores_subdir.exists():
                        scores_dir = scores_subdir
                        break

        if scores_dir.exists():
            for csv_file in sorted(scores_dir.glob("*.csv")):
                exp_name = csv_file.stem
                analyze_single_experiment(csv_file, output_dir, exp_name)
        else:
            print(f"No scores directory found in {args.result_dir}")

    else:
        print("Please provide --csv or --result-dir")


if __name__ == "__main__":
    main()
