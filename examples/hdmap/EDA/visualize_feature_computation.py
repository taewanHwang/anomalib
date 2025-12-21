#!/usr/bin/env python3
"""
Visualize Feature Computation Process.

Shows intermediate computation steps for:
1. GMM (Gradient Magnitude Mean): Gx, Gy, Magnitude
2. Local Entropy: Local entropy map
3. Intensity Mean: Histogram distribution

Compares normal vs fault samples side by side.
"""

import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Dataset path
HDMAP_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png")
OUTPUT_DIR = Path(__file__).parent / "results/feature_visualization"


def load_sample_images(domain: str = "domain_C"):
    """Load one normal and one fault sample."""
    domain_path = HDMAP_ROOT / domain

    # Load normal sample
    normal_path = domain_path / "test/good/000000.png"
    normal_img = np.array(Image.open(normal_path).convert("L"))

    # Load fault sample (choose one with clear difference)
    fault_path = domain_path / "test/fault/000500.png"  # This was notably darker
    fault_img = np.array(Image.open(fault_path).convert("L"))

    return normal_img, fault_img


def visualize_gmm_computation(normal_img: np.ndarray, fault_img: np.ndarray, output_path: Path):
    """
    Visualize GMM (Gradient Magnitude Mean) computation.

    Steps:
    1. Original image
    2. Gradient X (horizontal edges)
    3. Gradient Y (vertical edges)
    4. Gradient Magnitude = sqrt(Gx² + Gy²)
    5. Final GMM value
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("GMM (Gradient Magnitude Mean) Computation Process", fontsize=16, fontweight='bold')

    for row, (img, label) in enumerate([(normal_img, "Normal"), (fault_img, "Fault")]):
        img_float = img.astype(np.float32) / 255.0

        # Compute gradients
        gx = ndimage.sobel(img_float, axis=1)  # Horizontal gradient
        gy = ndimage.sobel(img_float, axis=0)  # Vertical gradient
        magnitude = np.sqrt(gx**2 + gy**2)

        # GMM value
        gmm = np.mean(magnitude)
        gmm_normalized = gmm / 4.0  # Normalize

        # Plot
        axes[row, 0].imshow(img, cmap='gray')
        axes[row, 0].set_title(f"{label}: Original")
        axes[row, 0].axis('off')

        axes[row, 1].imshow(gx, cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[row, 1].set_title("Gradient X (Sobel)")
        axes[row, 1].axis('off')

        axes[row, 2].imshow(gy, cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[row, 2].set_title("Gradient Y (Sobel)")
        axes[row, 2].axis('off')

        axes[row, 3].imshow(magnitude, cmap='hot')
        axes[row, 3].set_title("Magnitude = √(Gx² + Gy²)")
        axes[row, 3].axis('off')

        # GMM value as bar chart
        ax = axes[row, 4]
        colors = ['green' if label == "Normal" else 'red']
        ax.bar([label], [gmm_normalized], color=colors, width=0.5)
        ax.set_ylim(0, 0.05)
        ax.set_title(f"GMM = {gmm_normalized:.4f}")
        ax.set_ylabel("GMM Value")

        # Add text annotation
        ax.text(0, gmm_normalized + 0.002, f"{gmm_normalized:.4f}", ha='center', fontsize=12, fontweight='bold')

    # Add explanation
    fig.text(0.5, 0.02,
             "GMM = mean(Magnitude) / 4.0  |  Higher GMM = More edges (complex)  |  Lower GMM = Fewer edges (simple/uniform)",
             ha='center', fontsize=11, style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def visualize_le_computation(normal_img: np.ndarray, fault_img: np.ndarray, output_path: Path):
    """
    Visualize Local Entropy (LE) computation.

    Steps:
    1. Original image
    2. Local entropy map (sliding window histogram entropy)
    3. Final LE value (mean of map)
    """
    from skimage.filters.rank import entropy as skimage_entropy
    from skimage.morphology import disk

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Local Entropy (LE) Computation Process", fontsize=16, fontweight='bold')

    for row, (img, label) in enumerate([(normal_img, "Normal"), (fault_img, "Fault")]):
        # Compute local entropy
        local_ent = skimage_entropy(img.astype(np.uint8), disk(5))
        le_normalized = np.mean(local_ent) / 8.0  # Normalize by max entropy (log2(256)=8)

        # Plot
        axes[row, 0].imshow(img, cmap='gray')
        axes[row, 0].set_title(f"{label}: Original")
        axes[row, 0].axis('off')

        # Show disk kernel
        kernel_vis = np.zeros((11, 11))
        d = disk(5)
        kernel_vis[:d.shape[0], :d.shape[1]] = d
        axes[row, 1].imshow(kernel_vis, cmap='Blues')
        axes[row, 1].set_title("Sliding Window\n(disk radius=5)")
        axes[row, 1].axis('off')

        # Local entropy map
        im = axes[row, 2].imshow(local_ent, cmap='viridis')
        axes[row, 2].set_title("Local Entropy Map")
        axes[row, 2].axis('off')
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046)

        # LE value as bar chart
        ax = axes[row, 3]
        colors = ['green' if label == "Normal" else 'red']
        ax.bar([label], [le_normalized], color=colors, width=0.5)
        ax.set_ylim(0, 0.5)
        ax.set_title(f"LE = {le_normalized:.4f}")
        ax.set_ylabel("LE Value")
        ax.text(0, le_normalized + 0.01, f"{le_normalized:.4f}", ha='center', fontsize=12, fontweight='bold')

    # Add explanation
    fig.text(0.5, 0.02,
             "LE = mean(LocalEntropy) / 8.0  |  Higher LE = More information (complex)  |  Lower LE = Less information (simple)",
             ha='center', fontsize=11, style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def visualize_intensity_computation(normal_img: np.ndarray, fault_img: np.ndarray, output_path: Path):
    """
    Visualize Intensity Mean computation.

    Steps:
    1. Original image
    2. Histogram distribution
    3. Final Intensity Mean value
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Intensity Mean Computation Process", fontsize=16, fontweight='bold')

    for row, (img, label) in enumerate([(normal_img, "Normal"), (fault_img, "Fault")]):
        img_float = img.astype(np.float32) / 255.0
        intensity_mean = np.mean(img_float)
        intensity_std = np.std(img_float)

        # Plot original
        axes[row, 0].imshow(img, cmap='gray')
        axes[row, 0].set_title(f"{label}: Original")
        axes[row, 0].axis('off')

        # Histogram
        ax = axes[row, 1]
        hist, bins = np.histogram(img_float.flatten(), bins=50, range=(0, 1))
        ax.bar(bins[:-1], hist, width=0.02, color='blue', alpha=0.7)
        ax.axvline(intensity_mean, color='red', linestyle='--', linewidth=2, label=f'Mean={intensity_mean:.3f}')
        ax.axvline(intensity_mean - intensity_std, color='orange', linestyle=':', linewidth=1.5)
        ax.axvline(intensity_mean + intensity_std, color='orange', linestyle=':', linewidth=1.5, label=f'Std={intensity_std:.3f}')
        ax.set_xlim(0, 1)
        ax.set_xlabel("Pixel Intensity (0-1)")
        ax.set_ylabel("Count")
        ax.set_title("Intensity Histogram")
        ax.legend(loc='upper right')

        # Values as bar chart
        ax = axes[row, 2]
        color = 'green' if label == "Normal" else 'red'
        x = np.arange(2)
        bars = ax.bar(x, [intensity_mean, intensity_std], color=[color, color], width=0.4, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(['Mean', 'Std'])
        ax.set_ylim(0, 0.4)
        ax.set_title(f"Mean={intensity_mean:.4f}, Std={intensity_std:.4f}")
        ax.set_ylabel("Value")

        # Add text annotations
        for bar, val in zip(bars, [intensity_mean, intensity_std]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f"{val:.4f}", ha='center', fontsize=11, fontweight='bold')

    # Add explanation
    fig.text(0.5, 0.02,
             "Intensity Mean = mean(pixels/255)  |  Higher = Brighter image  |  Lower = Darker image (potential fault)",
             ha='center', fontsize=11, style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def visualize_all_features_comparison(normal_img: np.ndarray, fault_img: np.ndarray, output_path: Path):
    """
    Create a summary comparison of all features.
    """
    from skimage.filters.rank import entropy as skimage_entropy
    from skimage.morphology import disk

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("Feature Comparison: Normal vs Fault (Domain C)", fontsize=16, fontweight='bold')

    # Compute all features for both images
    features = {}
    for img, label in [(normal_img, "Normal"), (fault_img, "Fault")]:
        img_float = img.astype(np.float32) / 255.0

        # GMM
        gx = ndimage.sobel(img_float, axis=1)
        gy = ndimage.sobel(img_float, axis=0)
        magnitude = np.sqrt(gx**2 + gy**2)
        gmm = np.mean(magnitude) / 4.0

        # LE
        local_ent = skimage_entropy(img.astype(np.uint8), disk(5))
        le = np.mean(local_ent) / 8.0

        # Intensity
        intensity_mean = np.mean(img_float)
        intensity_std = np.std(img_float)

        features[label] = {
            'gmm': gmm,
            'gmm_map': magnitude,
            'le': le,
            'le_map': local_ent,
            'intensity_mean': intensity_mean,
            'intensity_std': intensity_std,
        }

    # Row 0: Original images
    axes[0, 0].imshow(normal_img, cmap='gray')
    axes[0, 0].set_title("Normal: Original")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(fault_img, cmap='gray')
    axes[0, 1].set_title("Fault: Original")
    axes[0, 1].axis('off')

    # Feature comparison bar chart
    ax = axes[0, 2]
    x = np.arange(3)
    width = 0.35
    normal_vals = [features['Normal']['gmm'], features['Normal']['le'], features['Normal']['intensity_mean']]
    fault_vals = [features['Fault']['gmm'], features['Fault']['le'], features['Fault']['intensity_mean']]
    bars1 = ax.bar(x - width/2, normal_vals, width, label='Normal', color='green', alpha=0.8)
    bars2 = ax.bar(x + width/2, fault_vals, width, label='Fault', color='red', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(['GMM', 'LE', 'Intensity\nMean'])
    ax.set_ylabel("Feature Value")
    ax.set_title("Feature Values Comparison")
    ax.legend()
    ax.set_ylim(0, 0.4)

    # Difference chart
    ax = axes[0, 3]
    diffs = [(f - n) for n, f in zip(normal_vals, fault_vals)]
    colors = ['red' if d < 0 else 'blue' for d in diffs]
    bars = ax.bar(['GMM', 'LE', 'Intensity\nMean'], diffs, color=colors, alpha=0.8)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel("Fault - Normal")
    ax.set_title("Feature Difference\n(Fault - Normal)")
    for bar, val in zip(bars, diffs):
        ax.text(bar.get_x() + bar.get_width()/2,
               bar.get_height() + (0.005 if val >= 0 else -0.015),
               f"{val:.4f}", ha='center', fontsize=10, fontweight='bold')

    # Row 1: GMM visualization
    axes[1, 0].imshow(features['Normal']['gmm_map'], cmap='hot')
    axes[1, 0].set_title(f"Normal: Gradient Mag\nGMM={features['Normal']['gmm']:.4f}")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(features['Fault']['gmm_map'], cmap='hot')
    axes[1, 1].set_title(f"Fault: Gradient Mag\nGMM={features['Fault']['gmm']:.4f}")
    axes[1, 1].axis('off')

    # GMM difference map
    gmm_diff = features['Normal']['gmm_map'] - features['Fault']['gmm_map']
    axes[1, 2].imshow(gmm_diff, cmap='RdBu', vmin=-0.2, vmax=0.2)
    axes[1, 2].set_title("GMM Difference\n(Normal - Fault)")
    axes[1, 2].axis('off')

    # GMM histogram
    ax = axes[1, 3]
    ax.hist(features['Normal']['gmm_map'].flatten(), bins=50, alpha=0.7, label='Normal', color='green', density=True)
    ax.hist(features['Fault']['gmm_map'].flatten(), bins=50, alpha=0.7, label='Fault', color='red', density=True)
    ax.set_xlabel("Gradient Magnitude")
    ax.set_ylabel("Density")
    ax.set_title("GMM Distribution")
    ax.legend()

    # Row 2: LE visualization
    axes[2, 0].imshow(features['Normal']['le_map'], cmap='viridis')
    axes[2, 0].set_title(f"Normal: Local Entropy\nLE={features['Normal']['le']:.4f}")
    axes[2, 0].axis('off')

    axes[2, 1].imshow(features['Fault']['le_map'], cmap='viridis')
    axes[2, 1].set_title(f"Fault: Local Entropy\nLE={features['Fault']['le']:.4f}")
    axes[2, 1].axis('off')

    # LE difference map
    le_diff = features['Normal']['le_map'] - features['Fault']['le_map']
    im = axes[2, 2].imshow(le_diff, cmap='RdBu', vmin=-1, vmax=1)
    axes[2, 2].set_title("LE Difference\n(Normal - Fault)")
    axes[2, 2].axis('off')

    # LE histogram
    ax = axes[2, 3]
    ax.hist(features['Normal']['le_map'].flatten(), bins=50, alpha=0.7, label='Normal', color='green', density=True)
    ax.hist(features['Fault']['le_map'].flatten(), bins=50, alpha=0.7, label='Fault', color='red', density=True)
    ax.set_xlabel("Local Entropy")
    ax.set_ylabel("Density")
    ax.set_title("LE Distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Feature Computation Visualization")
    print("="*60)

    # Load samples
    print("\nLoading Domain C samples...")
    normal_img, fault_img = load_sample_images("domain_C")
    print(f"  Normal shape: {normal_img.shape}")
    print(f"  Fault shape: {fault_img.shape}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    print("\n1. GMM (Gradient Magnitude Mean)...")
    visualize_gmm_computation(normal_img, fault_img, OUTPUT_DIR / "gmm_computation.png")

    print("\n2. Local Entropy (LE)...")
    visualize_le_computation(normal_img, fault_img, OUTPUT_DIR / "le_computation.png")

    print("\n3. Intensity Mean...")
    visualize_intensity_computation(normal_img, fault_img, OUTPUT_DIR / "intensity_computation.png")

    print("\n4. All Features Comparison...")
    visualize_all_features_comparison(normal_img, fault_img, OUTPUT_DIR / "all_features_comparison.png")

    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
