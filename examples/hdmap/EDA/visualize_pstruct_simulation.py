#!/usr/bin/env python3
"""
p_struct Simulation and Visualization for Unified Information-Control Dropout.

Visualizes the relationship between APE (Angular Power Entropy) and p_struct
using the formula:
    p_struct = sigmoid(sensitivity * (normal_ape - APE))

Author: Claude Code
Date: 2024-12-19
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm

# Output directory
OUTPUT_DIR = Path(__file__).parent / "results" / "pstruct_simulation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# APE statistics from EDA (from EDA_SUMMARY.md and ape_results.json)
DOMAIN_APE_STATS = {
    "domain_A": {"normal_mean": 0.777, "normal_std": 0.030, "anomaly_mean": 0.850, "anomaly_std": 0.030, "cohens_d": 2.43},
    "domain_B": {"normal_mean": 0.713, "normal_std": 0.033, "anomaly_mean": 0.855, "anomaly_std": 0.033, "cohens_d": 4.34},
    "domain_C": {"normal_mean": 0.866, "normal_std": 0.013, "anomaly_mean": 0.889, "anomaly_std": 0.013, "cohens_d": 1.73},
    "domain_D": {"normal_mean": 0.816, "normal_std": 0.028, "anomaly_mean": 0.887, "anomaly_std": 0.028, "cohens_d": 2.54},
}

# Sensitivities to test (from experiment design)
SENSITIVITIES = [0, 2, 4, 8, 15, 30]


def sigmoid(x):
    """Sigmoid function with numerical stability."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def compute_pstruct(ape, sensitivity, normal_ape):
    """
    Compute p_struct = sigmoid(sensitivity * (normal_ape - APE)).

    - Low APE (structured/normal-like) → high p_struct → high dropout
    - High APE (isotropic/anomaly-like) → low p_struct → low dropout
    """
    if sensitivity == 0:
        return np.ones_like(ape)  # Ablation mode
    return sigmoid(sensitivity * (normal_ape - ape))


def plot_pstruct_vs_ape_basic():
    """Plot 1: Basic p_struct vs APE curves for different sensitivities."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # APE range (theoretical: 0-1, practical: 0.6-0.95 for HDMAP)
    ape = np.linspace(0.5, 1.0, 500)

    # Use Domain C as reference (normal_ape = 0.866)
    normal_ape = 0.866

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(SENSITIVITIES)))

    for sens, color in zip(SENSITIVITIES, colors):
        p_struct = compute_pstruct(ape, sens, normal_ape)
        label = f"sensitivity={sens}" if sens > 0 else f"sensitivity={sens} (ablation)"
        linestyle = '--' if sens == 0 else '-'
        ax.plot(ape, p_struct, color=color, linewidth=2.5, label=label, linestyle=linestyle)

    # Reference lines
    ax.axvline(x=normal_ape, color='red', linestyle='--', alpha=0.7, linewidth=2,
               label=f'normal_ape={normal_ape} (Domain C)')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    # Mark Normal and Anomaly regions
    ax.axvspan(0.85, 0.87, alpha=0.2, color='green', label='Normal APE range')
    ax.axvspan(0.88, 0.90, alpha=0.2, color='red', label='Anomaly APE range')

    ax.set_xlabel('APE (Angular Power Entropy)', fontsize=14)
    ax.set_ylabel('p_struct', fontsize=14)
    ax.set_title('p_struct = sigmoid(sensitivity × (normal_ape - APE))\nDomain C Reference (normal_ape = 0.866)',
                 fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Add annotations
    ax.annotate('Low APE\n(Structured/Normal)\n→ High p_struct\n→ High Dropout',
                xy=(0.65, 0.85), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax.annotate('High APE\n(Isotropic/Anomaly)\n→ Low p_struct\n→ Low Dropout',
                xy=(0.95, 0.15), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    return fig


def plot_pstruct_all_domains():
    """Plot 2: p_struct curves for all 4 domains with domain-specific normal_ape."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    sensitivities_main = [4, 15]  # Main sensitivities for experiments

    for ax, (domain, domain_stats) in zip(axes.flat, DOMAIN_APE_STATS.items()):
        ape = np.linspace(0.5, 1.0, 500)
        normal_ape = domain_stats["normal_mean"]

        for sens, color in zip(sensitivities_main, ['blue', 'green']):
            p_struct = compute_pstruct(ape, sens, normal_ape)
            ax.plot(ape, p_struct, color=color, linewidth=2.5, label=f'sensitivity={sens}')

        # Mark normal_ape
        ax.axvline(x=normal_ape, color='red', linestyle='--', alpha=0.8, linewidth=2,
                   label=f'normal_ape={normal_ape:.3f}')

        # Draw Normal and Anomaly distributions
        normal_x = np.linspace(normal_ape - 3*domain_stats["normal_std"],
                               normal_ape + 3*domain_stats["normal_std"], 100)
        normal_pdf = norm.pdf(normal_x, normal_ape, domain_stats["normal_std"])
        normal_pdf = normal_pdf / normal_pdf.max() * 0.3  # Scale for visibility

        anomaly_x = np.linspace(domain_stats["anomaly_mean"] - 3*domain_stats["anomaly_std"],
                                domain_stats["anomaly_mean"] + 3*domain_stats["anomaly_std"], 100)
        anomaly_pdf = norm.pdf(anomaly_x, domain_stats["anomaly_mean"], domain_stats["anomaly_std"])
        anomaly_pdf = anomaly_pdf / anomaly_pdf.max() * 0.3

        ax.fill_between(normal_x, 0, normal_pdf, alpha=0.3, color='green', label='Normal dist.')
        ax.fill_between(anomaly_x, 0, anomaly_pdf, alpha=0.3, color='red', label='Anomaly dist.')

        # Calculate p_struct for Normal and Anomaly means
        for sens in sensitivities_main:
            p_normal = compute_pstruct(np.array([normal_ape]), sens, normal_ape)[0]
            p_anomaly = compute_pstruct(np.array([domain_stats["anomaly_mean"]]), sens, normal_ape)[0]
            ax.scatter([normal_ape], [p_normal], color='green', s=80, zorder=5, edgecolor='black')
            ax.scatter([domain_stats["anomaly_mean"]], [p_anomaly], color='red', s=80, zorder=5, edgecolor='black')

        ax.set_xlabel('APE', fontsize=12)
        ax.set_ylabel('p_struct', fontsize=12)
        ax.set_title(f"{domain}\nNormal APE={normal_ape:.3f}, Anomaly APE={domain_stats['anomaly_mean']:.3f}\n"
                     f"Cohen's d={domain_stats['cohens_d']:.2f}", fontsize=11)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(0.6, 1.0)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    plt.suptitle('p_struct vs APE for All HDMAP Domains\n(with Normal/Anomaly distributions)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_pstruct_difference_analysis():
    """Plot 3: Analyze p_struct difference between Normal and Anomaly samples."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sensitivities_range = np.linspace(0, 50, 200)

    # Left: p_struct difference vs sensitivity for each domain
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 0.4, 4))

    for (domain, stats), color in zip(DOMAIN_APE_STATS.items(), colors):
        normal_ape = stats["normal_mean"]
        anomaly_ape = stats["anomaly_mean"]

        differences = []
        for sens in sensitivities_range:
            p_normal = compute_pstruct(np.array([normal_ape]), sens, normal_ape)[0]
            p_anomaly = compute_pstruct(np.array([anomaly_ape]), sens, normal_ape)[0]
            differences.append(p_normal - p_anomaly)

        ax1.plot(sensitivities_range, differences, color=color, linewidth=2.5, label=domain)

    # Mark tested sensitivities
    for sens in [4, 15]:
        ax1.axvline(x=sens, color='gray', linestyle='--', alpha=0.5)
        ax1.text(sens, 0.55, f's={sens}', fontsize=10, ha='center')

    ax1.set_xlabel('Sensitivity', fontsize=12)
    ax1.set_ylabel('p_struct(Normal) - p_struct(Anomaly)', fontsize=12)
    ax1.set_title('p_struct Difference vs Sensitivity\n(Higher = More Discrimination)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 50)
    ax1.set_ylim(0, 0.6)
    ax1.grid(True, alpha=0.3)

    # Right: Bar chart for sensitivity=4 and 15
    ax2 = axes[1]
    domains = list(DOMAIN_APE_STATS.keys())
    x = np.arange(len(domains))
    width = 0.35

    diffs_s4 = []
    diffs_s15 = []

    for domain, stats in DOMAIN_APE_STATS.items():
        normal_ape = stats["normal_mean"]
        anomaly_ape = stats["anomaly_mean"]

        p_normal_4 = compute_pstruct(np.array([normal_ape]), 4, normal_ape)[0]
        p_anomaly_4 = compute_pstruct(np.array([anomaly_ape]), 4, normal_ape)[0]
        diffs_s4.append(p_normal_4 - p_anomaly_4)

        p_normal_15 = compute_pstruct(np.array([normal_ape]), 15, normal_ape)[0]
        p_anomaly_15 = compute_pstruct(np.array([anomaly_ape]), 15, normal_ape)[0]
        diffs_s15.append(p_normal_15 - p_anomaly_15)

    bars1 = ax2.bar(x - width/2, diffs_s4, width, label='sensitivity=4', color='blue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, diffs_s15, width, label='sensitivity=15', color='green', alpha=0.7)

    # Add value labels
    for bar, val in zip(bars1, diffs_s4):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=9)
    for bar, val in zip(bars2, diffs_s15):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=9)

    ax2.set_xlabel('Domain', fontsize=12)
    ax2.set_ylabel('p_struct Difference', fontsize=12)
    ax2.set_title('p_struct(Normal) - p_struct(Anomaly)\nby Domain and Sensitivity', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.replace('domain_', '') for d in domains])
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 0.55)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_final_dropout_simulation():
    """Plot 4: Full dropout simulation including p_time."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Parameters
    p_min = 0.0
    p_max_values = [0.2, 0.4]
    t_warmup = 1000
    max_steps = 5000

    # Domain C reference
    normal_ape = 0.866

    # Simulate training steps
    steps = np.linspace(0, max_steps, 500)

    # Sample APE values
    ape_normal = 0.866  # Normal sample
    ape_anomaly = 0.889  # Anomaly sample
    ape_very_structured = 0.75  # Very structured (low APE)
    ape_very_isotropic = 0.95  # Very isotropic (high APE)

    sample_apes = {
        'Very Structured (APE=0.75)': 0.75,
        'Normal Mean (APE=0.866)': 0.866,
        'Anomaly Mean (APE=0.889)': 0.889,
        'Very Isotropic (APE=0.95)': 0.95,
    }

    colors = ['green', 'blue', 'orange', 'red']

    for row, p_max in enumerate(p_max_values):
        for col, sensitivity in enumerate([4, 15]):
            ax = axes[row, col]

            for (label, ape), color in zip(sample_apes.items(), colors):
                # Compute p_time over training
                p_time = np.minimum(1.0, steps / t_warmup)

                # Compute p_struct (constant for each sample)
                p_struct = compute_pstruct(np.array([ape]), sensitivity, normal_ape)[0]

                # Final dropout
                dropout = p_min + (p_max - p_min) * p_time * p_struct

                ax.plot(steps, dropout, color=color, linewidth=2, label=f'{label}')

            # Reference lines
            ax.axvline(x=t_warmup, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(t_warmup + 50, p_max * 0.9, 't_warmup', fontsize=9, alpha=0.7)

            ax.set_xlabel('Training Step', fontsize=11)
            ax.set_ylabel('Dropout Probability', fontsize=11)
            ax.set_title(f'p_max={p_max}, sensitivity={sensitivity}', fontsize=12)
            ax.legend(fontsize=8, loc='lower right')
            ax.set_xlim(0, max_steps)
            ax.set_ylim(0, p_max * 1.1)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Dropout Probability Over Training\n'
                 'dropout = p_min + (p_max - p_min) × p_time × p_struct\n'
                 f'Domain C (normal_ape={normal_ape}), t_warmup={t_warmup}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_ape_histogram_simulation():
    """Plot 5: Simulated APE distribution histogram with p_struct overlay."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    np.random.seed(42)

    for ax, (domain, stats) in zip(axes.flat, DOMAIN_APE_STATS.items()):
        normal_ape = stats["normal_mean"]

        # Simulate samples
        n_normal = 200
        n_anomaly = 50

        normal_samples = np.random.normal(normal_ape, stats["normal_std"], n_normal)
        anomaly_samples = np.random.normal(stats["anomaly_mean"], stats["anomaly_std"], n_anomaly)

        # Clip to valid range
        normal_samples = np.clip(normal_samples, 0.5, 1.0)
        anomaly_samples = np.clip(anomaly_samples, 0.5, 1.0)

        # Plot histograms
        bins = np.linspace(0.6, 1.0, 40)
        ax.hist(normal_samples, bins=bins, alpha=0.5, color='green', label='Normal', density=True)
        ax.hist(anomaly_samples, bins=bins, alpha=0.5, color='red', label='Anomaly', density=True)

        # Overlay p_struct curves (secondary y-axis)
        ax2 = ax.twinx()
        ape_range = np.linspace(0.6, 1.0, 200)

        for sens, color, style in [(4, 'blue', '-'), (15, 'purple', '--')]:
            p_struct = compute_pstruct(ape_range, sens, normal_ape)
            ax2.plot(ape_range, p_struct, color=color, linewidth=2, linestyle=style,
                     label=f'p_struct (s={sens})')

        ax.axvline(x=normal_ape, color='black', linestyle='--', alpha=0.8, linewidth=2)

        ax.set_xlabel('APE', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax2.set_ylabel('p_struct', fontsize=11, color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_ylim(0, 1.1)

        ax.set_title(f'{domain}\nnormal_ape={normal_ape:.3f}', fontsize=11)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

        ax.set_xlim(0.6, 1.0)
        ax.grid(True, alpha=0.3)

    plt.suptitle('APE Distribution with p_struct Mapping\n'
                 '(Simulated Normal/Anomaly samples based on EDA statistics)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def print_summary_table():
    """Print numerical summary of p_struct values."""
    print("=" * 90)
    print("p_struct SIMULATION SUMMARY")
    print("Formula: p_struct = sigmoid(sensitivity × (normal_ape - APE))")
    print("=" * 90)

    print(f"\n{'Domain':<12} | {'Normal APE':<12} | {'Anomaly APE':<12} | "
          f"{'Sens':<6} | {'p_struct(N)':<12} | {'p_struct(A)':<12} | {'Diff':<8}")
    print("-" * 90)

    for domain, stats in DOMAIN_APE_STATS.items():
        normal_ape = stats["normal_mean"]
        anomaly_ape = stats["anomaly_mean"]

        for sens in [4, 15]:
            p_normal = compute_pstruct(np.array([normal_ape]), sens, normal_ape)[0]
            p_anomaly = compute_pstruct(np.array([anomaly_ape]), sens, normal_ape)[0]
            diff = p_normal - p_anomaly

            if sens == 4:
                print(f"{domain:<12} | {normal_ape:<12.4f} | {anomaly_ape:<12.4f} | "
                      f"{sens:<6} | {p_normal:<12.4f} | {p_anomaly:<12.4f} | {diff:<8.4f}")
            else:
                print(f"{'':<12} | {'':<12} | {'':<12} | "
                      f"{sens:<6} | {p_normal:<12.4f} | {p_anomaly:<12.4f} | {diff:<8.4f}")
        print("-" * 90)

    print("\n" + "=" * 90)
    print("KEY INSIGHTS:")
    print("1. At normal_ape, p_struct = 0.5 (sigmoid(0) = 0.5)")
    print("2. APE < normal_ape → p_struct > 0.5 → Higher dropout (prevent overfitting)")
    print("3. APE > normal_ape → p_struct < 0.5 → Lower dropout (preserve info)")
    print("4. Higher sensitivity → Stronger discrimination between Normal and Anomaly")
    print("=" * 90)


def main():
    print("Generating p_struct simulation visualizations...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Print summary
    print_summary_table()

    # Generate plots
    print("\nGenerating plots...")

    fig1 = plot_pstruct_vs_ape_basic()
    fig1.savefig(OUTPUT_DIR / "01_pstruct_vs_ape_basic.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: 01_pstruct_vs_ape_basic.png")

    fig2 = plot_pstruct_all_domains()
    fig2.savefig(OUTPUT_DIR / "02_pstruct_all_domains.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: 02_pstruct_all_domains.png")

    fig3 = plot_pstruct_difference_analysis()
    fig3.savefig(OUTPUT_DIR / "03_pstruct_difference_analysis.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: 03_pstruct_difference_analysis.png")

    fig4 = plot_final_dropout_simulation()
    fig4.savefig(OUTPUT_DIR / "04_dropout_over_training.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: 04_dropout_over_training.png")

    fig5 = plot_ape_histogram_simulation()
    fig5.savefig(OUTPUT_DIR / "05_ape_histogram_with_pstruct.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: 05_ape_histogram_with_pstruct.png")

    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")

    # Save summary as JSON
    summary = {
        "formula": "p_struct = sigmoid(sensitivity * (normal_ape - APE))",
        "dropout_formula": "dropout = p_min + (p_max - p_min) * p_time * p_struct",
        "domain_stats": DOMAIN_APE_STATS,
        "key_insights": [
            "At normal_ape, p_struct = 0.5 (sigmoid(0) = 0.5)",
            "APE < normal_ape → p_struct > 0.5 → Higher dropout",
            "APE > normal_ape → p_struct < 0.5 → Lower dropout",
            "Higher sensitivity → Stronger discrimination"
        ]
    }

    with open(OUTPUT_DIR / "simulation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: simulation_summary.json")

    plt.show()


if __name__ == "__main__":
    main()
