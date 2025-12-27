#!/usr/bin/env python3
"""
Analyze HDMAP TIFF image intensity/amplitude by condition (cold/warm, fault/normal).

Hypothesis: Cold start images have lower amplitude, causing lower anomaly scores
despite the fault pattern being detected.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile
from tqdm import tqdm
import argparse


def load_tiff_stats(tiff_path: Path) -> dict:
    """Load TIFF and compute intensity statistics."""
    img = tifffile.imread(tiff_path)

    # Handle different formats
    if img.ndim == 3:
        if img.shape[0] == 3:  # (C, H, W)
            img = img.transpose(1, 2, 0)  # -> (H, W, C)
        # Use mean across channels or just first channel
        img_gray = img.mean(axis=-1) if img.ndim == 3 else img
    else:
        img_gray = img

    return {
        'mean': float(np.mean(img_gray)),
        'std': float(np.std(img_gray)),
        'min': float(np.min(img_gray)),
        'max': float(np.max(img_gray)),
        'range': float(np.max(img_gray) - np.min(img_gray)),
        'p5': float(np.percentile(img_gray, 5)),
        'p95': float(np.percentile(img_gray, 95)),
        'dynamic_range': float(np.percentile(img_gray, 95) - np.percentile(img_gray, 5)),
    }


def analyze_domain_intensity(domain_path: Path, domain_name: str) -> pd.DataFrame:
    """Analyze intensity for all images in a domain."""
    results = []

    for label in ['fault', 'good']:
        label_dir = domain_path / 'test' / label
        if not label_dir.exists():
            continue

        tiff_files = sorted(label_dir.glob('*.tiff'))

        for tiff_path in tqdm(tiff_files, desc=f"{domain_name}/{label}"):
            file_idx = int(tiff_path.stem)

            # Determine condition
            if file_idx < 500:
                condition = 'cold'
            else:
                condition = 'warm'

            full_condition = f"{label}_{condition}"

            stats = load_tiff_stats(tiff_path)
            stats['domain'] = domain_name
            stats['label'] = label
            stats['condition'] = condition
            stats['full_condition'] = full_condition
            stats['file_idx'] = file_idx
            stats['file_path'] = str(tiff_path)

            results.append(stats)

    return pd.DataFrame(results)


def plot_intensity_by_condition(df: pd.DataFrame, output_path: Path, metric: str = 'mean'):
    """Plot intensity distribution by condition."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    conditions = ['fault_cold', 'fault_warm', 'normal_cold', 'normal_warm']
    # Map 'good' to 'normal' for display
    df['display_condition'] = df['full_condition'].replace({
        'good_cold': 'normal_cold',
        'good_warm': 'normal_warm'
    })

    colors = {
        'fault_cold': '#ff6b6b',
        'fault_warm': '#c92a2a',
        'normal_cold': '#74b9ff',
        'normal_warm': '#0984e3'
    }

    # 1. Box plot by condition
    ax1 = axes[0, 0]
    box_data = []
    labels = []
    for cond in ['fault_cold', 'fault_warm', 'normal_cold', 'normal_warm']:
        data = df[df['display_condition'] == cond][metric].values
        if len(data) > 0:
            box_data.append(data)
            labels.append(cond.replace('_', '\n'))

    bp = ax1.boxplot(box_data, tick_labels=labels, patch_artist=True)
    for i, (patch, cond) in enumerate(zip(bp['boxes'], ['fault_cold', 'fault_warm', 'normal_cold', 'normal_warm'])):
        patch.set_facecolor(colors.get(cond, 'gray'))
        patch.set_alpha(0.7)
    ax1.set_ylabel(f'Image {metric.capitalize()}')
    ax1.set_title(f'Image {metric.capitalize()} by Condition')

    # 2. Time series of mean intensity
    ax2 = axes[0, 1]
    for cond in ['fault_cold', 'fault_warm', 'normal_cold', 'normal_warm']:
        subset = df[df['display_condition'] == cond].sort_values('file_idx')
        if len(subset) > 0:
            ax2.scatter(subset['file_idx'], subset[metric],
                       c=colors.get(cond, 'gray'), alpha=0.5, s=10, label=cond)
    ax2.set_xlabel('File Index')
    ax2.set_ylabel(f'Image {metric.capitalize()}')
    ax2.set_title(f'Image {metric.capitalize()} vs File Index')
    ax2.legend()

    # 3. Dynamic range comparison
    ax3 = axes[1, 0]
    box_data_dr = []
    for cond in ['fault_cold', 'fault_warm', 'normal_cold', 'normal_warm']:
        data = df[df['display_condition'] == cond]['dynamic_range'].values
        if len(data) > 0:
            box_data_dr.append(data)

    bp2 = ax3.boxplot(box_data_dr, tick_labels=labels, patch_artist=True)
    for i, (patch, cond) in enumerate(zip(bp2['boxes'], ['fault_cold', 'fault_warm', 'normal_cold', 'normal_warm'])):
        patch.set_facecolor(colors.get(cond, 'gray'))
        patch.set_alpha(0.7)
    ax3.set_ylabel('Dynamic Range (P95-P5)')
    ax3.set_title('Image Dynamic Range by Condition')

    # 4. Histogram of mean intensity
    ax4 = axes[1, 1]
    for cond in ['fault_cold', 'fault_warm', 'normal_cold', 'normal_warm']:
        data = df[df['display_condition'] == cond][metric].values
        if len(data) > 0:
            ax4.hist(data, bins=30, alpha=0.5, label=cond, color=colors.get(cond, 'gray'))
    ax4.set_xlabel(f'Image {metric.capitalize()}')
    ax4.set_ylabel('Count')
    ax4.set_title(f'Distribution of Image {metric.capitalize()}')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_statistics(df: pd.DataFrame):
    """Print summary statistics by condition."""
    df['display_condition'] = df['full_condition'].replace({
        'good_cold': 'normal_cold',
        'good_warm': 'normal_warm'
    })

    print("\n" + "="*80)
    print("IMAGE INTENSITY STATISTICS BY CONDITION")
    print("="*80)

    metrics = ['mean', 'std', 'range', 'dynamic_range']

    for metric in metrics:
        print(f"\n{metric.upper()}:")
        print("-"*60)
        for cond in ['fault_cold', 'fault_warm', 'normal_cold', 'normal_warm']:
            data = df[df['display_condition'] == cond][metric]
            if len(data) > 0:
                print(f"  {cond:15s}: {data.mean():.4f} ± {data.std():.4f} "
                      f"(min={data.min():.4f}, max={data.max():.4f})")

    # Compute ratios
    print("\n" + "="*80)
    print("COLD vs WARM RATIO (Warm / Cold)")
    print("="*80)

    for label in ['fault', 'normal']:
        cold_cond = f"{label}_cold"
        warm_cond = f"{label}_warm"

        cold_mean = df[df['display_condition'] == cold_cond]['mean'].mean()
        warm_mean = df[df['display_condition'] == warm_cond]['mean'].mean()

        cold_dr = df[df['display_condition'] == cold_cond]['dynamic_range'].mean()
        warm_dr = df[df['display_condition'] == warm_cond]['dynamic_range'].mean()

        print(f"\n{label.upper()}:")
        print(f"  Mean intensity: Warm/Cold = {warm_mean:.4f}/{cold_mean:.4f} = {warm_mean/cold_mean:.2f}x")
        print(f"  Dynamic range:  Warm/Cold = {warm_dr:.4f}/{cold_dr:.4f} = {warm_dr/cold_dr:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Analyze HDMAP image intensity by condition")
    parser.add_argument("--domain", type=str, default="domain_C", help="Domain to analyze")
    parser.add_argument("--dataset-root", type=Path,
                       default=Path("datasets/HDMAP/1000_tiff_minmax"),
                       help="Dataset root path")
    parser.add_argument("--output-dir", type=Path,
                       default=Path("examples/hdmap/EDA/winclip_baseline_post_analysis"),
                       help="Output directory")
    args = parser.parse_args()

    domain_path = args.dataset_root / args.domain

    print(f"Analyzing {args.domain} from {domain_path}")

    df = analyze_domain_intensity(domain_path, args.domain)

    # Save raw data
    output_csv = args.output_dir / f"{args.domain}_intensity_stats.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

    # Print statistics
    print_statistics(df)

    # Plot
    output_plot = args.output_dir / f"{args.domain}_intensity_by_condition.png"
    plot_intensity_by_condition(df, output_plot, metric='mean')


if __name__ == "__main__":
    main()
