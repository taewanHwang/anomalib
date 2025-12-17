"""
Analyze orientation entropy distribution for HDMAP dataset.

This script:
1. Loads sample images from PNG and FFT datasets
2. Computes orientation entropy and regularity scores
3. Compares distributions between normal and abnormal samples
4. Visualizes results to validate the hypothesis
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from typing import List, Dict, Tuple
import pandas as pd

from orientation_entropy import (
    orientation_entropy_cv2,
    dominant_frequency_ratio,
    compute_regularity_score,
    adaptive_dropout_prob,
)


# Dataset paths
HDMAP_PNG_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png")
HDMAP_FFT_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png_2dfft")

DOMAINS = ["domain_A", "domain_B", "domain_C", "domain_D"]


def load_images_from_folder(folder: Path, max_samples: int = 100) -> List[np.ndarray]:
    """Load images from a folder."""
    images = []
    if not folder.exists():
        return images

    image_files = sorted(folder.glob("*.png"))[:max_samples]
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)
    return images


def analyze_dataset(
    dataset_root: Path,
    domains: List[str],
    max_samples_per_class: int = 100,
) -> pd.DataFrame:
    """Analyze orientation entropy for a dataset."""
    results = []

    for domain in tqdm(domains, desc="Analyzing domains"):
        domain_path = dataset_root / domain

        # Normal samples (train/good + test/good)
        normal_folders = [
            domain_path / "train" / "good",
            domain_path / "test" / "good",
        ]
        for folder in normal_folders:
            images = load_images_from_folder(folder, max_samples_per_class // 2)
            for img in images:
                reg_score, components = compute_regularity_score(img)
                results.append({
                    "domain": domain,
                    "label": "normal",
                    "folder": folder.name,
                    **components,
                    "adaptive_dropout": adaptive_dropout_prob(reg_score),
                })

        # Abnormal samples (test/fault)
        abnormal_folder = domain_path / "test" / "fault"
        images = load_images_from_folder(abnormal_folder, max_samples_per_class)
        for img in images:
            reg_score, components = compute_regularity_score(img)
            results.append({
                "domain": domain,
                "label": "abnormal",
                "folder": "fault",
                **components,
                "adaptive_dropout": adaptive_dropout_prob(reg_score),
            })

    return pd.DataFrame(results)


def plot_entropy_distribution(
    df: pd.DataFrame,
    dataset_name: str,
    save_path: Path,
):
    """Plot entropy distribution comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Orientation Entropy Analysis - {dataset_name}", fontsize=14)

    # 1. Overall orientation entropy distribution
    ax = axes[0, 0]
    for label in ["normal", "abnormal"]:
        data = df[df["label"] == label]["orientation_entropy"]
        ax.hist(data, bins=30, alpha=0.6, label=f"{label} (n={len(data)})", density=True)
    ax.set_xlabel("Orientation Entropy")
    ax.set_ylabel("Density")
    ax.set_title("Orientation Entropy Distribution")
    ax.legend()
    ax.axvline(df["orientation_entropy"].median(), color="gray", linestyle="--", alpha=0.5)

    # 2. Overall frequency ratio distribution
    ax = axes[0, 1]
    for label in ["normal", "abnormal"]:
        data = df[df["label"] == label]["frequency_ratio"]
        ax.hist(data, bins=30, alpha=0.6, label=f"{label}", density=True)
    ax.set_xlabel("Dominant Frequency Ratio")
    ax.set_ylabel("Density")
    ax.set_title("Frequency Ratio Distribution")
    ax.legend()

    # 3. Combined regularity score
    ax = axes[0, 2]
    for label in ["normal", "abnormal"]:
        data = df[df["label"] == label]["combined_regularity"]
        ax.hist(data, bins=30, alpha=0.6, label=f"{label}", density=True)
    ax.set_xlabel("Combined Regularity Score")
    ax.set_ylabel("Density")
    ax.set_title("Combined Regularity Distribution")
    ax.legend()

    # 4. Per-domain orientation entropy
    ax = axes[1, 0]
    domain_data = []
    labels_list = []
    for domain in DOMAINS:
        for label in ["normal", "abnormal"]:
            subset = df[(df["domain"] == domain) & (df["label"] == label)]
            if len(subset) > 0:
                domain_data.append(subset["orientation_entropy"].values)
                labels_list.append(f"{domain[-1]}_{label[0]}")

    if domain_data:
        bp = ax.boxplot(domain_data, labels=labels_list, patch_artist=True)
        colors = ["lightgreen" if "n" in l else "lightcoral" for l in labels_list]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
    ax.set_xlabel("Domain_Label")
    ax.set_ylabel("Orientation Entropy")
    ax.set_title("Entropy by Domain")
    ax.tick_params(axis="x", rotation=45)

    # 5. Adaptive dropout distribution
    ax = axes[1, 1]
    for label in ["normal", "abnormal"]:
        data = df[df["label"] == label]["adaptive_dropout"]
        ax.hist(data, bins=30, alpha=0.6, label=f"{label}", density=True)
    ax.set_xlabel("Adaptive Dropout Probability")
    ax.set_ylabel("Density")
    ax.set_title("Suggested Dropout Distribution")
    ax.legend()

    # 6. Scatter: Orientation Entropy vs Frequency Ratio
    ax = axes[1, 2]
    for label, color, marker in [("normal", "green", "o"), ("abnormal", "red", "x")]:
        subset = df[df["label"] == label]
        ax.scatter(
            subset["orientation_entropy"],
            subset["frequency_ratio"],
            c=color,
            marker=marker,
            alpha=0.3,
            label=label,
            s=20,
        )
    ax.set_xlabel("Orientation Entropy")
    ax.set_ylabel("Frequency Ratio")
    ax.set_title("Entropy vs Frequency Ratio")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def print_statistics(df: pd.DataFrame, dataset_name: str):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print(f"Statistics for {dataset_name}")
    print("=" * 60)

    for label in ["normal", "abnormal"]:
        subset = df[df["label"] == label]
        print(f"\n{label.upper()} samples (n={len(subset)}):")
        for col in ["orientation_entropy", "frequency_ratio", "combined_regularity", "adaptive_dropout"]:
            print(f"  {col}: mean={subset[col].mean():.4f}, std={subset[col].std():.4f}, "
                  f"median={subset[col].median():.4f}")

    # Per domain
    print("\nPer-domain orientation entropy (mean):")
    pivot = df.pivot_table(
        values="orientation_entropy",
        index="domain",
        columns="label",
        aggfunc="mean"
    )
    print(pivot.to_string())

    # Separation analysis
    print("\nNormal vs Abnormal separation:")
    normal_mean = df[df["label"] == "normal"]["orientation_entropy"].mean()
    abnormal_mean = df[df["label"] == "abnormal"]["orientation_entropy"].mean()
    print(f"  Orientation Entropy: normal={normal_mean:.4f}, abnormal={abnormal_mean:.4f}, "
          f"diff={abnormal_mean - normal_mean:.4f}")


def visualize_sample_images(
    dataset_root: Path,
    df: pd.DataFrame,
    dataset_name: str,
    save_path: Path,
    num_samples: int = 4,
):
    """Visualize sample images with their entropy values."""
    fig, axes = plt.subplots(2, num_samples * 2, figsize=(4 * num_samples, 8))
    fig.suptitle(f"Sample Images with Orientation Entropy - {dataset_name}", fontsize=14)

    # Get low and high entropy samples
    df_sorted = df.sort_values("orientation_entropy")
    low_entropy_samples = df_sorted.head(num_samples * 2)
    high_entropy_samples = df_sorted.tail(num_samples * 2)

    # Low entropy (high regularity) - top row
    for i, (_, row) in enumerate(low_entropy_samples.head(num_samples).iterrows()):
        domain = row["domain"]
        label = row["label"]
        folder = "train/good" if row["folder"] == "good" else f"test/{row['folder']}"
        img_folder = dataset_root / domain / folder.replace("/", "/")

        # Find an image (just take first one for visualization)
        img_files = list(img_folder.glob("*.png"))
        if img_files:
            img = cv2.imread(str(img_files[i % len(img_files)]))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[0, i].imshow(img_rgb)
            axes[0, i].set_title(f"Low Entropy: {row['orientation_entropy']:.3f}\n"
                                f"{domain[-1]}, {label}")
            axes[0, i].axis("off")

    # High entropy (low regularity) - top row continued
    for i, (_, row) in enumerate(high_entropy_samples.tail(num_samples).iterrows()):
        domain = row["domain"]
        label = row["label"]
        folder = "train/good" if row["folder"] == "good" else f"test/{row['folder']}"
        img_folder = dataset_root / domain / folder.replace("/", "/")

        img_files = list(img_folder.glob("*.png"))
        if img_files:
            img = cv2.imread(str(img_files[i % len(img_files)]))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[0, num_samples + i].imshow(img_rgb)
            axes[0, num_samples + i].set_title(f"High Entropy: {row['orientation_entropy']:.3f}\n"
                                               f"{domain[-1]}, {label}")
            axes[0, num_samples + i].axis("off")

    # Bottom row: gradient visualization for comparison
    for i in range(num_samples * 2):
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Main analysis function."""
    output_dir = Path(__file__).parent / "results" / "orientation_entropy"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze PNG dataset
    print("\n" + "=" * 60)
    print("Analyzing PNG Dataset")
    print("=" * 60)
    df_png = analyze_dataset(HDMAP_PNG_ROOT, DOMAINS, max_samples_per_class=200)
    df_png.to_csv(output_dir / "png_entropy_analysis.csv", index=False)
    print_statistics(df_png, "PNG Dataset")
    plot_entropy_distribution(df_png, "PNG Dataset", output_dir / "png_entropy_distribution.png")

    # Analyze FFT dataset
    print("\n" + "=" * 60)
    print("Analyzing FFT Dataset")
    print("=" * 60)
    df_fft = analyze_dataset(HDMAP_FFT_ROOT, DOMAINS, max_samples_per_class=200)
    df_fft.to_csv(output_dir / "fft_entropy_analysis.csv", index=False)
    print_statistics(df_fft, "FFT Dataset")
    plot_entropy_distribution(df_fft, "FFT Dataset", output_dir / "fft_entropy_distribution.png")

    # Compare PNG vs FFT
    print("\n" + "=" * 60)
    print("PNG vs FFT Comparison")
    print("=" * 60)
    print(f"PNG - Normal entropy mean: {df_png[df_png['label'] == 'normal']['orientation_entropy'].mean():.4f}")
    print(f"FFT - Normal entropy mean: {df_fft[df_fft['label'] == 'normal']['orientation_entropy'].mean():.4f}")
    print(f"PNG - Abnormal entropy mean: {df_png[df_png['label'] == 'abnormal']['orientation_entropy'].mean():.4f}")
    print(f"FFT - Abnormal entropy mean: {df_fft[df_fft['label'] == 'abnormal']['orientation_entropy'].mean():.4f}")

    # Summary JSON
    summary = {
        "png": {
            "normal_entropy_mean": float(df_png[df_png["label"] == "normal"]["orientation_entropy"].mean()),
            "abnormal_entropy_mean": float(df_png[df_png["label"] == "abnormal"]["orientation_entropy"].mean()),
            "normal_regularity_mean": float(df_png[df_png["label"] == "normal"]["combined_regularity"].mean()),
            "abnormal_regularity_mean": float(df_png[df_png["label"] == "abnormal"]["combined_regularity"].mean()),
        },
        "fft": {
            "normal_entropy_mean": float(df_fft[df_fft["label"] == "normal"]["orientation_entropy"].mean()),
            "abnormal_entropy_mean": float(df_fft[df_fft["label"] == "abnormal"]["orientation_entropy"].mean()),
            "normal_regularity_mean": float(df_fft[df_fft["label"] == "normal"]["combined_regularity"].mean()),
            "abnormal_regularity_mean": float(df_fft[df_fft["label"] == "abnormal"]["combined_regularity"].mean()),
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
