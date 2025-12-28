"""Quantitative analysis of preprocessing effects and FFT analysis.

1. Compute which preprocessing method maximizes fault vs good difference
2. 2D FFT analysis to compare fault/cold vs good/cold

Usage:
    python analyze_preprocessing_difference.py --domain domain_C
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parents[4]))

from anomalib.data.datasets.image.hdmap import HDMAPDataset


def load_dataset(dataset_root: str, domain: str):
    """Load HDMAP test dataset."""
    dataset = HDMAPDataset(
        root=dataset_root,
        domain=domain,
        split="test",
        target_size=(240, 240),
        resize_method="resize",
    )
    return dataset


def tensor_to_numpy(img_tensor):
    """Convert tensor image to numpy (H, W, C) format."""
    img = img_tensor.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    return img


def to_uint8(img):
    """Convert [0,1] float to [0,255] uint8."""
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def to_gray(img):
    """Convert to grayscale."""
    if len(img.shape) == 3:
        return cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return img


# ============================================================================
# Preprocessing Functions
# ============================================================================

def preprocess_original(img):
    return to_gray(img)


def preprocess_sobel_h(img):
    gray = (to_gray(img) * 255).astype(np.uint8)
    sobel_h = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_h = np.abs(sobel_h)
    return sobel_h / (sobel_h.max() + 1e-8)


def preprocess_sobel_v(img):
    gray = (to_gray(img) * 255).astype(np.uint8)
    sobel_v = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_v = np.abs(sobel_v)
    return sobel_v / (sobel_v.max() + 1e-8)


def preprocess_clahe(img):
    img_uint8 = to_uint8(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(img_uint8.shape) == 3:
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_uint8
    return clahe.apply(gray).astype(np.float32) / 255.0


def preprocess_contrast(img):
    gray = to_gray(img)
    p2, p98 = np.percentile(gray, (2, 98))
    return np.clip((gray - p2) / (p98 - p2 + 1e-8), 0, 1)


def preprocess_horizontal_hp(img):
    gray = (to_gray(img) * 255).astype(np.uint8)
    kernel = np.array([[ 1,  2,  1],
                       [ 0,  0,  0],
                       [-1, -2, -1]], dtype=np.float32)
    filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    filtered = np.abs(filtered)
    return filtered / (filtered.max() + 1e-8)


def preprocess_row_diff(img):
    gray = to_gray(img)
    row_diff = np.abs(np.diff(gray, axis=0))
    row_diff = np.vstack([row_diff, row_diff[-1:, :]])
    return row_diff / (row_diff.max() + 1e-8)


def preprocess_laplacian(img):
    gray = (to_gray(img) * 255).astype(np.uint8)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.abs(laplacian)
    return laplacian / (laplacian.max() + 1e-8)


PREPROCESSING_METHODS = {
    "Original": preprocess_original,
    "Sobel_H": preprocess_sobel_h,
    "Sobel_V": preprocess_sobel_v,
    "CLAHE": preprocess_clahe,
    "Contrast": preprocess_contrast,
    "Horizontal_HP": preprocess_horizontal_hp,
    "Row_Diff": preprocess_row_diff,
    "Laplacian": preprocess_laplacian,
}


# ============================================================================
# Analysis Functions
# ============================================================================

def compute_statistics(images):
    """Compute statistics for a list of images."""
    means = [img.mean() for img in images]
    stds = [img.std() for img in images]
    maxs = [img.max() for img in images]
    return {
        'mean': np.mean(means),
        'std': np.mean(stds),
        'max': np.mean(maxs),
        'means': means,
        'stds': stds,
    }


def analyze_preprocessing_difference(dataset, fault_indices, good_indices):
    """Analyze which preprocessing maximizes fault vs good difference."""

    results = {}

    for method_name, method_func in PREPROCESSING_METHODS.items():
        # Process all fault images
        fault_processed = []
        for idx in fault_indices:
            sample = dataset[idx]
            img = tensor_to_numpy(sample.image)
            processed = method_func(img)
            fault_processed.append(processed)

        # Process all good images
        good_processed = []
        for idx in good_indices:
            sample = dataset[idx]
            img = tensor_to_numpy(sample.image)
            processed = method_func(img)
            good_processed.append(processed)

        # Compute statistics
        fault_stats = compute_statistics(fault_processed)
        good_stats = compute_statistics(good_processed)

        # Compute separability metrics
        # 1. Mean difference
        mean_diff = abs(fault_stats['mean'] - good_stats['mean'])

        # 2. T-test
        t_stat, p_value = stats.ttest_ind(fault_stats['means'], good_stats['means'])

        # 3. Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(fault_stats['means'])**2 + np.std(good_stats['means'])**2) / 2)
        cohens_d = abs(fault_stats['mean'] - good_stats['mean']) / (pooled_std + 1e-8)

        # 4. Overlap coefficient (lower is better separation)
        fault_min, fault_max = min(fault_stats['means']), max(fault_stats['means'])
        good_min, good_max = min(good_stats['means']), max(good_stats['means'])
        overlap = max(0, min(fault_max, good_max) - max(fault_min, good_min))
        total_range = max(fault_max, good_max) - min(fault_min, good_min)
        overlap_ratio = overlap / (total_range + 1e-8)

        results[method_name] = {
            'fault_mean': fault_stats['mean'],
            'good_mean': good_stats['mean'],
            'mean_diff': mean_diff,
            't_stat': abs(t_stat),
            'p_value': p_value,
            'cohens_d': cohens_d,
            'overlap_ratio': overlap_ratio,
            'fault_means': fault_stats['means'],
            'good_means': good_stats['means'],
        }

    return results


def visualize_analysis_results(results, output_dir):
    """Visualize the analysis results."""

    # Sort by Cohen's d (effect size)
    sorted_methods = sorted(results.keys(), key=lambda x: results[x]['cohens_d'], reverse=True)

    # 1. Bar chart of effect sizes
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Cohen's d
    ax = axes[0, 0]
    cohens_ds = [results[m]['cohens_d'] for m in sorted_methods]
    colors = ['green' if d > 0.8 else 'orange' if d > 0.5 else 'red' for d in cohens_ds]
    ax.barh(sorted_methods, cohens_ds, color=colors)
    ax.set_xlabel("Cohen's d (Effect Size)")
    ax.set_title("Effect Size: Fault vs Good Difference\n(>0.8: Large, 0.5-0.8: Medium, <0.5: Small)")
    ax.axvline(0.8, color='green', linestyle='--', alpha=0.5, label='Large effect')
    ax.axvline(0.5, color='orange', linestyle='--', alpha=0.5, label='Medium effect')

    # T-statistic
    ax = axes[0, 1]
    t_stats = [results[m]['t_stat'] for m in sorted_methods]
    ax.barh(sorted_methods, t_stats, color='steelblue')
    ax.set_xlabel("T-statistic (absolute)")
    ax.set_title("T-test Statistic\n(Higher = More Significant Difference)")

    # Mean difference
    ax = axes[1, 0]
    mean_diffs = [results[m]['mean_diff'] for m in sorted_methods]
    ax.barh(sorted_methods, mean_diffs, color='purple')
    ax.set_xlabel("Mean Difference")
    ax.set_title("Absolute Mean Difference")

    # Overlap ratio (lower is better)
    ax = axes[1, 1]
    overlaps = [results[m]['overlap_ratio'] for m in sorted_methods]
    colors = ['green' if o < 0.3 else 'orange' if o < 0.6 else 'red' for o in overlaps]
    ax.barh(sorted_methods, overlaps, color=colors)
    ax.set_xlabel("Overlap Ratio")
    ax.set_title("Distribution Overlap\n(Lower = Better Separation)")

    plt.tight_layout()
    save_path = output_dir / "preprocessing_comparison_metrics.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    # 2. Distribution plots for top methods
    top_methods = sorted_methods[:4]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, method in enumerate(top_methods):
        ax = axes[i // 2, i % 2]
        fault_means = results[method]['fault_means']
        good_means = results[method]['good_means']

        ax.hist(fault_means, bins=15, alpha=0.6, label=f'Fault (n={len(fault_means)})', color='red')
        ax.hist(good_means, bins=15, alpha=0.6, label=f'Good (n={len(good_means)})', color='blue')
        ax.axvline(np.mean(fault_means), color='darkred', linestyle='--', linewidth=2)
        ax.axvline(np.mean(good_means), color='darkblue', linestyle='--', linewidth=2)

        d = results[method]['cohens_d']
        p = results[method]['p_value']
        ax.set_title(f"{method}\nCohen's d={d:.3f}, p={p:.2e}")
        ax.set_xlabel("Mean Intensity")
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    save_path = output_dir / "preprocessing_top_distributions.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    return sorted_methods


# ============================================================================
# FFT Analysis
# ============================================================================

def compute_2d_fft(img):
    """Compute 2D FFT and return magnitude spectrum."""
    gray = to_gray(img)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    # Log scale for better visualization
    magnitude_log = np.log1p(magnitude)
    return magnitude_log, fshift


def analyze_fft(dataset, fault_indices, good_indices, output_dir):
    """Perform 2D FFT analysis comparing fault vs good."""

    print("\nComputing FFT for fault/cold samples...")
    fault_ffts = []
    fault_mags = []
    for idx in fault_indices:
        sample = dataset[idx]
        img = tensor_to_numpy(sample.image)
        mag, fshift = compute_2d_fft(img)
        fault_ffts.append(fshift)
        fault_mags.append(mag)

    print("Computing FFT for good/cold samples...")
    good_ffts = []
    good_mags = []
    for idx in good_indices:
        sample = dataset[idx]
        img = tensor_to_numpy(sample.image)
        mag, fshift = compute_2d_fft(img)
        good_ffts.append(fshift)
        good_mags.append(mag)

    # Average magnitude spectra
    fault_avg_mag = np.mean(fault_mags, axis=0)
    good_avg_mag = np.mean(good_mags, axis=0)
    diff_mag = fault_avg_mag - good_avg_mag

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Average spectra
    ax = axes[0, 0]
    im = ax.imshow(fault_avg_mag, cmap='hot')
    ax.set_title(f"Fault/Cold Average FFT\n(n={len(fault_indices)})")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    im = ax.imshow(good_avg_mag, cmap='hot')
    ax.set_title(f"Good/Cold Average FFT\n(n={len(good_indices)})")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 2]
    im = ax.imshow(diff_mag, cmap='RdBu_r', vmin=-np.abs(diff_mag).max(), vmax=np.abs(diff_mag).max())
    ax.set_title("Difference (Fault - Good)")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 2: Frequency band analysis
    h, w = fault_avg_mag.shape
    center_h, center_w = h // 2, w // 2

    # Horizontal frequency profile (center row)
    ax = axes[1, 0]
    fault_h_profile = fault_avg_mag[center_h, :]
    good_h_profile = good_avg_mag[center_h, :]
    ax.plot(fault_h_profile, label='Fault', color='red', alpha=0.7)
    ax.plot(good_h_profile, label='Good', color='blue', alpha=0.7)
    ax.set_title("Horizontal Frequency Profile\n(Center Row)")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude (log)")
    ax.legend()

    # Vertical frequency profile (center column) - This shows horizontal patterns
    ax = axes[1, 1]
    fault_v_profile = fault_avg_mag[:, center_w]
    good_v_profile = good_avg_mag[:, center_w]
    ax.plot(fault_v_profile, label='Fault', color='red', alpha=0.7)
    ax.plot(good_v_profile, label='Good', color='blue', alpha=0.7)
    ax.set_title("Vertical Frequency Profile\n(Center Column - Shows Horizontal Patterns)")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude (log)")
    ax.legend()

    # Difference profile
    ax = axes[1, 2]
    diff_h = fault_h_profile - good_h_profile
    diff_v = fault_v_profile - good_v_profile
    ax.plot(diff_h, label='Horizontal diff', color='purple', alpha=0.7)
    ax.plot(diff_v, label='Vertical diff', color='green', alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_title("Frequency Difference\n(Fault - Good)")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude Difference")
    ax.legend()

    plt.tight_layout()
    save_path = output_dir / "fft_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    # Individual sample FFT comparison
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    fig.suptitle("Individual Sample FFT Comparison (Fault vs Good)", fontsize=12)

    for i in range(3):
        # Fault sample
        fault_idx = fault_indices[i * 10]  # Sample at different positions
        fault_sample = dataset[fault_idx]
        fault_img = tensor_to_numpy(fault_sample.image)
        fault_mag, _ = compute_2d_fft(fault_img)
        fault_file_idx = int(Path(fault_sample.image_path).stem)

        # Good sample
        good_idx = good_indices[i * 10]
        good_sample = dataset[good_idx]
        good_img = tensor_to_numpy(good_sample.image)
        good_mag, _ = compute_2d_fft(good_img)
        good_file_idx = int(Path(good_sample.image_path).stem)

        # Fault: original
        ax = axes[i, 0]
        ax.imshow(to_gray(fault_img), cmap='gray')
        ax.set_title(f"Fault {fault_file_idx:04d}", fontsize=9)
        ax.axis('off')

        # Fault: FFT
        ax = axes[i, 1]
        ax.imshow(fault_mag, cmap='hot')
        ax.set_title("FFT", fontsize=9)
        ax.axis('off')

        # Good: original
        ax = axes[i, 2]
        ax.imshow(to_gray(good_img), cmap='gray')
        ax.set_title(f"Good {good_file_idx:04d}", fontsize=9)
        ax.axis('off')

        # Good: FFT
        ax = axes[i, 3]
        ax.imshow(good_mag, cmap='hot')
        ax.set_title("FFT", fontsize=9)
        ax.axis('off')

        # Difference image
        ax = axes[i, 4]
        diff_img = to_gray(fault_img) - to_gray(good_img)
        ax.imshow(diff_img, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
        ax.set_title("Diff (F-G)", fontsize=9)
        ax.axis('off')

        # Difference FFT
        ax = axes[i, 5]
        diff_fft = fault_mag - good_mag
        ax.imshow(diff_fft, cmap='RdBu_r', vmin=-2, vmax=2)
        ax.set_title("FFT Diff", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    save_path = output_dir / "fft_individual_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    return fault_avg_mag, good_avg_mag, diff_mag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="domain_C")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples per group")
    args = parser.parse_args()

    # Setup
    dataset_root = "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_tiff_minmax"
    output_dir = Path(__file__).parent / args.domain / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Domain: {args.domain}")
    print(f"Samples per group: {args.num_samples}")
    print(f"Output: {output_dir}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(dataset_root, args.domain)
    print(f"Dataset size: {len(dataset)}")

    # Select samples (evenly spaced)
    interval = 500 // args.num_samples
    fault_cold_indices = list(range(0, 500, interval))[:args.num_samples]
    good_cold_indices = list(range(1000, 1500, interval))[:args.num_samples]

    print(f"\nFault/cold: {len(fault_cold_indices)} samples")
    print(f"Good/cold: {len(good_cold_indices)} samples")

    # ========================================
    # 1. Preprocessing Analysis
    # ========================================
    print("\n" + "=" * 60)
    print("1. PREPROCESSING ANALYSIS")
    print("=" * 60)

    results = analyze_preprocessing_difference(dataset, fault_cold_indices, good_cold_indices)
    sorted_methods = visualize_analysis_results(results, output_dir)

    # Print ranking
    print("\n" + "-" * 60)
    print("RANKING by Cohen's d (Effect Size):")
    print("-" * 60)
    cohens_label = "Cohen's d"
    print(f"{'Rank':<6} {'Method':<15} {cohens_label:<12} {'p-value':<12} {'Overlap':<10}")
    print("-" * 60)
    for i, method in enumerate(sorted_methods):
        r = results[method]
        print(f"{i+1:<6} {method:<15} {r['cohens_d']:<12.4f} {r['p_value']:<12.2e} {r['overlap_ratio']:<10.4f}")

    # ========================================
    # 2. FFT Analysis
    # ========================================
    print("\n" + "=" * 60)
    print("2. FFT ANALYSIS")
    print("=" * 60)

    fault_fft, good_fft, diff_fft = analyze_fft(dataset, fault_cold_indices, good_cold_indices, output_dir)

    # Compute horizontal vs vertical energy difference
    h, w = diff_fft.shape
    center_h, center_w = h // 2, w // 2

    # Horizontal band (rows around center) - captures vertical patterns
    h_band = diff_fft[center_h-10:center_h+10, :].mean()
    # Vertical band (columns around center) - captures horizontal patterns
    v_band = diff_fft[:, center_w-10:center_w+10].mean()

    print(f"\nFFT Band Analysis:")
    print(f"  Horizontal band mean diff: {h_band:.4f}")
    print(f"  Vertical band mean diff: {v_band:.4f}")
    print(f"  -> {'Horizontal patterns differ more' if abs(v_band) > abs(h_band) else 'Vertical patterns differ more'}")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
