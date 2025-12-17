"""
Visualize Orientation Entropy Distribution per Domain using Violin Plots.

Creates 4 figures (domain A~D), each showing violin plots for:
- Normal Train (train/good)
- Normal Test (test/good)
- Abnormal Test (test/fault)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Import orientation entropy function
from orientation_entropy import orientation_entropy_cv2


# Dataset path
HDMAP_PNG_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png")
DOMAINS = ["domain_A", "domain_B", "domain_C", "domain_D"]

# Output directory
OUTPUT_DIR = Path(__file__).parent / "results" / "entropy_violin_plots"


def compute_entropy_for_folder(folder: Path) -> list[float]:
    """Compute orientation entropy for all images in a folder."""
    entropies = []

    if not folder.exists():
        print(f"Warning: {folder} does not exist")
        return entropies

    image_files = sorted(folder.glob("*.png"))

    for img_path in tqdm(image_files, desc=f"  {folder.name}", leave=False):
        img = cv2.imread(str(img_path))
        if img is not None:
            entropy = orientation_entropy_cv2(img)
            entropies.append(entropy)

    return entropies


def analyze_domain(domain: str) -> pd.DataFrame:
    """Analyze orientation entropy for a single domain."""
    domain_path = HDMAP_PNG_ROOT / domain

    data = []

    # Normal Train
    print(f"  Processing train/good...")
    train_good_entropies = compute_entropy_for_folder(domain_path / "train" / "good")
    for e in train_good_entropies:
        data.append({"category": "Normal Train", "entropy": e})

    # Normal Test
    print(f"  Processing test/good...")
    test_good_entropies = compute_entropy_for_folder(domain_path / "test" / "good")
    for e in test_good_entropies:
        data.append({"category": "Normal Test", "entropy": e})

    # Abnormal Test
    print(f"  Processing test/fault...")
    test_fault_entropies = compute_entropy_for_folder(domain_path / "test" / "fault")
    for e in test_fault_entropies:
        data.append({"category": "Abnormal Test", "entropy": e})

    return pd.DataFrame(data)


def create_violin_plot(df: pd.DataFrame, domain: str, output_path: Path):
    """Create violin plot for a domain."""
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["Normal Train", "Normal Test", "Abnormal Test"]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]  # green, blue, red

    # Prepare data for violin plot
    data_list = []
    positions = []
    for i, cat in enumerate(categories):
        cat_data = df[df["category"] == cat]["entropy"].values
        if len(cat_data) > 0:
            data_list.append(cat_data)
            positions.append(i)

    # Create violin plot
    parts = ax.violinplot(data_list, positions=positions, showmeans=True, showmedians=True)

    # Customize colors
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    # Customize lines
    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('white')
    parts['cmedians'].set_linewidth(2)

    # Add statistics annotation
    stats_text = []
    for i, cat in enumerate(categories):
        cat_data = df[df["category"] == cat]["entropy"]
        if len(cat_data) > 0:
            stats_text.append(
                f"{cat}:\n"
                f"  n={len(cat_data)}\n"
                f"  mean={cat_data.mean():.4f}\n"
                f"  std={cat_data.std():.4f}"
            )

    # Set labels and title
    ax.set_xticks(positions)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel("Orientation Entropy", fontsize=12)
    ax.set_title(f"Orientation Entropy Distribution - {domain}", fontsize=14, fontweight='bold')

    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Add statistics box
    textstr = "\n\n".join(stats_text)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(1.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_combined_figure(all_data: dict, output_path: Path):
    """Create a combined 2x2 figure with all domains."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    categories = ["Normal Train", "Normal Test", "Abnormal Test"]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    for idx, domain in enumerate(DOMAINS):
        ax = axes[idx]
        df = all_data[domain]

        # Prepare data
        data_list = []
        positions = []
        for i, cat in enumerate(categories):
            cat_data = df[df["category"] == cat]["entropy"].values
            if len(cat_data) > 0:
                data_list.append(cat_data)
                positions.append(i)

        # Create violin plot
        parts = ax.violinplot(data_list, positions=positions, showmeans=True, showmedians=True)

        # Customize colors
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        parts['cmeans'].set_color('black')
        parts['cmeans'].set_linewidth(1.5)
        parts['cmedians'].set_color('white')
        parts['cmedians'].set_linewidth(1.5)

        # Labels
        ax.set_xticks(positions)
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylabel("Orientation Entropy", fontsize=10)
        ax.set_title(f"{domain}", fontsize=12, fontweight='bold')
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # Add mean values as text
        for i, cat in enumerate(categories):
            cat_data = df[df["category"] == cat]["entropy"]
            if len(cat_data) > 0:
                ax.text(i, cat_data.mean() + 0.005, f"{cat_data.mean():.3f}",
                       ha='center', va='bottom', fontsize=8, color='black')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], alpha=0.7, label='Normal Train'),
        Patch(facecolor=colors[1], alpha=0.7, label='Normal Test'),
        Patch(facecolor=colors[2], alpha=0.7, label='Abnormal Test'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, 1.02))

    plt.suptitle("Orientation Entropy Distribution by Domain (HDMAP PNG)",
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined figure: {output_path}")


def print_summary_table(all_data: dict):
    """Print summary statistics table."""
    print("\n" + "=" * 80)
    print("SUMMARY: Orientation Entropy Statistics")
    print("=" * 80)

    categories = ["Normal Train", "Normal Test", "Abnormal Test"]

    # Header
    print(f"\n{'Domain':<12} | {'Category':<15} | {'Count':>6} | {'Mean':>8} | {'Std':>8} | {'Median':>8}")
    print("-" * 80)

    for domain in DOMAINS:
        df = all_data[domain]
        for cat in categories:
            cat_data = df[df["category"] == cat]["entropy"]
            if len(cat_data) > 0:
                print(f"{domain:<12} | {cat:<15} | {len(cat_data):>6} | {cat_data.mean():>8.4f} | {cat_data.std():>8.4f} | {cat_data.median():>8.4f}")
        print("-" * 80)

    # Normal vs Abnormal comparison
    print("\n" + "=" * 80)
    print("COMPARISON: Normal (Train+Test) vs Abnormal")
    print("=" * 80)
    print(f"\n{'Domain':<12} | {'Normal Mean':>12} | {'Abnormal Mean':>14} | {'Difference':>12} | {'Interpretation'}")
    print("-" * 80)

    for domain in DOMAINS:
        df = all_data[domain]
        normal_data = df[df["category"].isin(["Normal Train", "Normal Test"])]["entropy"]
        abnormal_data = df[df["category"] == "Abnormal Test"]["entropy"]

        if len(normal_data) > 0 and len(abnormal_data) > 0:
            diff = abnormal_data.mean() - normal_data.mean()
            interp = "Abnormal more irregular" if diff > 0 else "Normal more irregular"
            print(f"{domain:<12} | {normal_data.mean():>12.4f} | {abnormal_data.mean():>14.4f} | {diff:>+12.4f} | {interp}")

    print("=" * 80)


def main():
    """Main function."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_data = {}

    # Analyze each domain
    for domain in DOMAINS:
        print(f"\n{'='*60}")
        print(f"Analyzing {domain}")
        print("=" * 60)

        df = analyze_domain(domain)
        all_data[domain] = df

        # Save individual violin plot
        create_violin_plot(df, domain, OUTPUT_DIR / f"entropy_violin_{domain}.png")

        # Save data to CSV
        df.to_csv(OUTPUT_DIR / f"entropy_data_{domain}.csv", index=False)

    # Create combined figure
    create_combined_figure(all_data, OUTPUT_DIR / "entropy_violin_all_domains.png")

    # Print summary
    print_summary_table(all_data)

    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
