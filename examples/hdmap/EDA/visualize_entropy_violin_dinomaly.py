"""
Visualize Orientation Entropy Distribution per Domain using Violin Plots.
Uses EXACT same preprocessing as Dinomaly (448x448 resize + ImageNet normalize).

Creates 4 figures (domain A~D), each showing violin plots for:
- Normal Train (train/good)
- Normal Test (test/good)
- Abnormal Test (test/fault)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torchvision.transforms as T

# Import the EXACT same function used in Dinomaly
from anomalib.models.image.dinomaly.adaptive_dropout import compute_orientation_entropy_batch


# Dataset path
HDMAP_PNG_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png")
DOMAINS = ["domain_A", "domain_B", "domain_C", "domain_D"]

# Output directory
OUTPUT_DIR = Path(__file__).parent / "results" / "entropy_violin_plots_dinomaly"

# EXACT same transform as Dinomaly
DINOMALY_TRANSFORM = T.Compose([
    T.Resize((448, 448)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def compute_entropy_for_folder(folder: Path, batch_size: int = 64, device: str = "cuda") -> list[float]:
    """Compute orientation entropy for all images in a folder using Dinomaly preprocessing."""
    entropies = []

    if not folder.exists():
        print(f"Warning: {folder} does not exist")
        return entropies

    image_files = sorted(folder.glob("*.png"))

    # Process in batches
    for i in tqdm(range(0, len(image_files), batch_size), desc=f"  {folder.name}", leave=False):
        batch_files = image_files[i:i+batch_size]

        # Load and transform images
        batch_images = []
        for img_path in batch_files:
            img = Image.open(img_path).convert("RGB")
            img_tensor = DINOMALY_TRANSFORM(img)
            batch_images.append(img_tensor)

        if batch_images:
            batch_tensor = torch.stack(batch_images).to(device)  # GPU로 전송

            # Compute entropy using the exact same function as Dinomaly
            batch_entropy = compute_orientation_entropy_batch(batch_tensor)
            entropies.extend(batch_entropy.cpu().numpy().tolist())

    return entropies


def analyze_domain(domain: str, device: str = "cuda") -> pd.DataFrame:
    """Analyze orientation entropy for a single domain."""
    domain_path = HDMAP_PNG_ROOT / domain

    data = []

    # Normal Train
    print(f"  Processing train/good...")
    train_good_entropies = compute_entropy_for_folder(domain_path / "train" / "good", device=device)
    for e in train_good_entropies:
        data.append({"category": "Normal Train", "entropy": e})

    # Normal Test
    print(f"  Processing test/good...")
    test_good_entropies = compute_entropy_for_folder(domain_path / "test" / "good", device=device)
    for e in test_good_entropies:
        data.append({"category": "Normal Test", "entropy": e})

    # Abnormal Test
    print(f"  Processing test/fault...")
    test_fault_entropies = compute_entropy_for_folder(domain_path / "test" / "fault", device=device)
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
    ax.set_title(f"Orientation Entropy Distribution - {domain}\n(Dinomaly preprocessing: 448x448 + ImageNet norm)",
                 fontsize=14, fontweight='bold')

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

    plt.suptitle("Orientation Entropy Distribution by Domain (HDMAP PNG)\n"
                 "Dinomaly preprocessing: 448x448 resize + ImageNet normalize",
                 fontsize=14, fontweight='bold', y=1.06)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined figure: {output_path}")


def print_summary_table(all_data: dict):
    """Print summary statistics table."""
    print("\n" + "=" * 80)
    print("SUMMARY: Orientation Entropy Statistics (Dinomaly preprocessing)")
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
    print(f"\n{'Domain':<12} | {'Normal Mean':>12} | {'Abnormal Mean':>14} | {'Difference':>12}")
    print("-" * 80)

    all_normal_means = []
    all_abnormal_means = []

    for domain in DOMAINS:
        df = all_data[domain]
        normal_data = df[df["category"].isin(["Normal Train", "Normal Test"])]["entropy"]
        abnormal_data = df[df["category"] == "Abnormal Test"]["entropy"]

        if len(normal_data) > 0 and len(abnormal_data) > 0:
            diff = abnormal_data.mean() - normal_data.mean()
            all_normal_means.append(normal_data.mean())
            all_abnormal_means.append(abnormal_data.mean())
            print(f"{domain:<12} | {normal_data.mean():>12.4f} | {abnormal_data.mean():>14.4f} | {diff:>+12.4f}")

    print("-" * 80)
    overall_normal = np.mean(all_normal_means)
    overall_abnormal = np.mean(all_abnormal_means)
    print(f"{'OVERALL':<12} | {overall_normal:>12.4f} | {overall_abnormal:>14.4f} | {overall_abnormal - overall_normal:>+12.4f}")
    print("=" * 80)

    print(f"\n>>> RECOMMENDED normal_entropy for adaptive dropout: {overall_normal:.4f}")


def main():
    """Main function."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_data = {}

    # Analyze each domain
    for domain in DOMAINS:
        print(f"\n{'='*60}")
        print(f"Analyzing {domain}")
        print("=" * 60)

        df = analyze_domain(domain, device=device)
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
