"""
Dropout Mapping Analysis for HDMAP with Dinomaly preprocessing.

Visualizes how orientation entropy maps to dropout probability
with current settings (normal_entropy=0.53).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Current settings (from Dinomaly preprocessing analysis)
NORMAL_ENTROPY = 0.53  # Updated from 0.43
BASE_DROPOUT = 0.3
MIN_DROPOUT = 0.1
MAX_DROPOUT = 0.6

# HDMAP statistics (from violin plot analysis with Dinomaly preprocessing)
HDMAP_STATS = {
    "domain_A": {"normal": 0.535, "abnormal": 0.554},
    "domain_B": {"normal": 0.520, "abnormal": 0.558},
    "domain_C": {"normal": 0.544, "abnormal": 0.550},
    "domain_D": {"normal": 0.531, "abnormal": 0.554},
    "overall": {"normal": 0.5324, "abnormal": 0.5540},
}

OUTPUT_DIR = Path(__file__).parent / "results" / "dropout_mapping_analysis"


def entropy_to_dropout(entropy, sensitivity, normal_entropy=NORMAL_ENTROPY):
    """Tanh-based dropout mapping centered on normal_entropy."""
    deviation = normal_entropy - entropy
    delta = deviation * sensitivity
    adjustment = np.tanh(delta)

    dropout = np.where(
        adjustment >= 0,
        BASE_DROPOUT + adjustment * (MAX_DROPOUT - BASE_DROPOUT),
        BASE_DROPOUT + adjustment * (BASE_DROPOUT - MIN_DROPOUT),
    )
    return np.clip(dropout, MIN_DROPOUT, MAX_DROPOUT)


def create_dropout_mapping_plot():
    """Create comprehensive dropout mapping visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Entropy range for plotting
    entropy = np.linspace(0.35, 0.65, 500)

    # HDMAP entropy range (from violin plots)
    hdmap_min = 0.51
    hdmap_max = 0.57

    sensitivities = [0, 4, 10, 15, 20, 30]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sensitivities)))

    # ============================================================
    # Plot 1: Dropout vs Entropy for different sensitivities
    # ============================================================
    ax1 = axes[0, 0]
    for sens, color in zip(sensitivities, colors):
        dropout = entropy_to_dropout(entropy, sens)
        ax1.plot(entropy, dropout, color=color, linewidth=2, label=f"sensitivity={sens}")

    # Reference lines
    ax1.axhline(y=BASE_DROPOUT, color='gray', linestyle='--', alpha=0.7, label=f"base_dropout={BASE_DROPOUT}")
    ax1.axvline(x=NORMAL_ENTROPY, color='red', linestyle='--', alpha=0.7, label=f"normal_entropy={NORMAL_ENTROPY}")
    ax1.axvline(x=HDMAP_STATS["overall"]["normal"], color='green', linestyle=':', alpha=0.7, label=f"HDMAP Normal={HDMAP_STATS['overall']['normal']:.3f}")
    ax1.axvline(x=HDMAP_STATS["overall"]["abnormal"], color='orange', linestyle=':', alpha=0.7, label=f"HDMAP Abnormal={HDMAP_STATS['overall']['abnormal']:.3f}")

    # Shade HDMAP range
    ax1.axvspan(hdmap_min, hdmap_max, alpha=0.15, color='blue', label="HDMAP entropy range")

    ax1.set_xlabel("Orientation Entropy", fontsize=12)
    ax1.set_ylabel("Dropout Probability", fontsize=12)
    ax1.set_title("Dropout vs Entropy (tanh-based, centered on normal_entropy=0.53)", fontsize=11)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlim(0.35, 0.65)
    ax1.set_ylim(0.05, 0.65)
    ax1.grid(True, alpha=0.3)

    # ============================================================
    # Plot 2: Verification - sensitivity=0 always gives base_dropout
    # ============================================================
    ax2 = axes[0, 1]
    entropy_full = np.linspace(0, 1, 100)
    dropout_sens0 = entropy_to_dropout(entropy_full, 0)
    ax2.plot(entropy_full, dropout_sens0, 'b-', linewidth=2, label="sensitivity=0")
    ax2.axhline(y=BASE_DROPOUT, color='red', linestyle='--', linewidth=2, label=f"base_dropout={BASE_DROPOUT}")
    ax2.set_xlabel("Entropy", fontsize=12)
    ax2.set_ylabel("Dropout", fontsize=12)
    ax2.set_title("Verification: sensitivity=0 → dropout=base_dropout", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 0.7)
    ax2.grid(True, alpha=0.3)

    # ============================================================
    # Plot 3: HDMAP Analysis with sensitivity=4 and 15
    # ============================================================
    ax3 = axes[1, 0]

    for sens, color, style in [(4, 'blue', '-'), (15, 'green', '-')]:
        dropout = entropy_to_dropout(entropy, sens)
        ax3.plot(entropy, dropout, color=color, linestyle=style, linewidth=2, label=f"sensitivity={sens}")

        # Mark Normal and Abnormal points
        normal_e = HDMAP_STATS["overall"]["normal"]
        abnormal_e = HDMAP_STATS["overall"]["abnormal"]
        normal_d = entropy_to_dropout(np.array([normal_e]), sens)[0]
        abnormal_d = entropy_to_dropout(np.array([abnormal_e]), sens)[0]

        ax3.scatter([normal_e], [normal_d], color=color, s=100, marker='o', edgecolor='black', zorder=5)
        ax3.scatter([abnormal_e], [abnormal_d], color=color, s=100, marker='s', edgecolor='black', zorder=5)

    ax3.axhline(y=BASE_DROPOUT, color='gray', linestyle='--', alpha=0.7, label=f"base_dropout={BASE_DROPOUT}")
    ax3.axvline(x=NORMAL_ENTROPY, color='purple', linestyle='--', alpha=0.7, label=f"normal_entropy={NORMAL_ENTROPY}")
    ax3.axvspan(hdmap_min, hdmap_max, alpha=0.15, color='blue', label="HDMAP range")

    ax3.set_xlabel("Orientation Entropy", fontsize=12)
    ax3.set_ylabel("Dropout Probability", fontsize=12)
    ax3.set_title(f"HDMAP Analysis (normal_entropy={NORMAL_ENTROPY})", fontsize=11)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_xlim(0.45, 0.60)
    ax3.set_ylim(0.15, 0.45)
    ax3.grid(True, alpha=0.3)

    # Add text annotations
    for sens in [4, 15]:
        normal_d = entropy_to_dropout(np.array([HDMAP_STATS["overall"]["normal"]]), sens)[0]
        abnormal_d = entropy_to_dropout(np.array([HDMAP_STATS["overall"]["abnormal"]]), sens)[0]
        diff = normal_d - abnormal_d
        ax3.text(0.46, 0.42 - (sens-4)*0.03,
                f"sens={sens}: Normal→{normal_d:.3f}, Abnormal→{abnormal_d:.3f}, diff={diff:.3f}",
                fontsize=9, color='blue' if sens==4 else 'green')

    # ============================================================
    # Plot 4: Dropout Difference (Normal - Abnormal) vs Sensitivity
    # ============================================================
    ax4 = axes[1, 1]

    sensitivities_range = np.linspace(0, 50, 200)
    dropout_diffs = []

    for sens in sensitivities_range:
        normal_d = entropy_to_dropout(np.array([HDMAP_STATS["overall"]["normal"]]), sens)[0]
        abnormal_d = entropy_to_dropout(np.array([HDMAP_STATS["overall"]["abnormal"]]), sens)[0]
        dropout_diffs.append(normal_d - abnormal_d)

    ax4.plot(sensitivities_range, dropout_diffs, 'b-', linewidth=2)

    # Mark tested sensitivities
    for sens in [4, 15]:
        normal_d = entropy_to_dropout(np.array([HDMAP_STATS["overall"]["normal"]]), sens)[0]
        abnormal_d = entropy_to_dropout(np.array([HDMAP_STATS["overall"]["abnormal"]]), sens)[0]
        diff = normal_d - abnormal_d
        ax4.scatter([sens], [diff], color='red', s=100, zorder=5)
        ax4.annotate(f"sens={sens}\ndiff={diff:.3f}", (sens, diff),
                    textcoords="offset points", xytext=(10, 5), fontsize=9)

    ax4.axhline(y=0.05, color='green', linestyle='--', alpha=0.7, label="Target diff=0.05")
    ax4.axhline(y=0.03, color='orange', linestyle='--', alpha=0.7, label="Minimum meaningful diff=0.03")

    ax4.set_xlabel("Sensitivity", fontsize=12)
    ax4.set_ylabel("Dropout Difference (Normal - Abnormal)", fontsize=12)
    ax4.set_title("Dropout Difference vs Sensitivity", fontsize=11)
    ax4.legend(fontsize=9)
    ax4.set_xlim(0, 50)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f"Dropout Mapping Analysis (normal_entropy={NORMAL_ENTROPY}, HDMAP with Dinomaly preprocessing)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    return fig


def create_domain_specific_plot():
    """Create domain-specific dropout analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    domains = ["domain_A", "domain_B", "domain_C", "domain_D"]
    entropy = np.linspace(0.45, 0.60, 200)

    for ax, domain in zip(axes.flat, domains):
        stats = HDMAP_STATS[domain]

        for sens, color in [(4, 'blue'), (15, 'green')]:
            dropout = entropy_to_dropout(entropy, sens)
            ax.plot(entropy, dropout, color=color, linewidth=2, label=f"sens={sens}")

            # Mark points
            normal_d = entropy_to_dropout(np.array([stats["normal"]]), sens)[0]
            abnormal_d = entropy_to_dropout(np.array([stats["abnormal"]]), sens)[0]

            ax.scatter([stats["normal"]], [normal_d], color='green', s=80, marker='o',
                      edgecolor='black', zorder=5, label=f"Normal ({stats['normal']:.3f})" if sens==4 else "")
            ax.scatter([stats["abnormal"]], [abnormal_d], color='red', s=80, marker='s',
                      edgecolor='black', zorder=5, label=f"Abnormal ({stats['abnormal']:.3f})" if sens==4 else "")

        ax.axhline(y=BASE_DROPOUT, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(x=NORMAL_ENTROPY, color='purple', linestyle='--', alpha=0.5)

        # Calculate and display differences
        diff_4 = entropy_to_dropout(np.array([stats["normal"]]), 4)[0] - entropy_to_dropout(np.array([stats["abnormal"]]), 4)[0]
        diff_15 = entropy_to_dropout(np.array([stats["normal"]]), 15)[0] - entropy_to_dropout(np.array([stats["abnormal"]]), 15)[0]

        ax.set_title(f"{domain}\nEntropy gap: {stats['abnormal']-stats['normal']:.4f}, "
                    f"Dropout diff: sens4={diff_4:.3f}, sens15={diff_15:.3f}", fontsize=10)
        ax.set_xlabel("Entropy")
        ax.set_ylabel("Dropout")
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.48, 0.58)
        ax.set_ylim(0.20, 0.40)

    plt.suptitle(f"Domain-Specific Dropout Analysis (normal_entropy={NORMAL_ENTROPY})", fontsize=12, fontweight='bold')
    plt.tight_layout()

    return fig


def print_summary():
    """Print numerical summary."""
    print("=" * 70)
    print("DROPOUT MAPPING SUMMARY (normal_entropy=0.53)")
    print("=" * 70)
    print(f"\nSettings:")
    print(f"  normal_entropy = {NORMAL_ENTROPY}")
    print(f"  base_dropout = {BASE_DROPOUT}")
    print(f"  min_dropout = {MIN_DROPOUT}")
    print(f"  max_dropout = {MAX_DROPOUT}")

    print(f"\n{'Domain':<12} | {'Normal E':<10} | {'Abnormal E':<10} | {'E Gap':<8} | {'Sens':<6} | {'N Drop':<8} | {'A Drop':<8} | {'D Diff':<8}")
    print("-" * 90)

    for domain in ["domain_A", "domain_B", "domain_C", "domain_D", "overall"]:
        stats = HDMAP_STATS[domain]
        e_gap = stats["abnormal"] - stats["normal"]

        for sens in [4, 15]:
            normal_d = entropy_to_dropout(np.array([stats["normal"]]), sens)[0]
            abnormal_d = entropy_to_dropout(np.array([stats["abnormal"]]), sens)[0]
            d_diff = normal_d - abnormal_d

            if sens == 4:
                print(f"{domain:<12} | {stats['normal']:<10.4f} | {stats['abnormal']:<10.4f} | {e_gap:<8.4f} | {sens:<6} | {normal_d:<8.4f} | {abnormal_d:<8.4f} | {d_diff:<8.4f}")
            else:
                print(f"{'':<12} | {'':<10} | {'':<10} | {'':<8} | {sens:<6} | {normal_d:<8.4f} | {abnormal_d:<8.4f} | {d_diff:<8.4f}")
        print("-" * 90)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_summary()

    # Create main analysis plot
    fig1 = create_dropout_mapping_plot()
    fig1.savefig(OUTPUT_DIR / "dropout_mapping_analysis.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_DIR / 'dropout_mapping_analysis.png'}")

    # Create domain-specific plot
    fig2 = create_domain_specific_plot()
    fig2.savefig(OUTPUT_DIR / "dropout_mapping_by_domain.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'dropout_mapping_by_domain.png'}")

    plt.show()


if __name__ == "__main__":
    main()
