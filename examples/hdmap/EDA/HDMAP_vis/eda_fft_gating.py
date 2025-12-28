"""EDA: FFT-based Gating for Cold/Warm Condition Selection.

Analyze whether 2D FFT features can be used to distinguish
cold vs warm conditions for CA-WinCLIP gating.

Approach:
1. Compute 2D FFT of test images
2. Compare to cold/warm reference FFT
3. Select condition based on FFT similarity
4. Evaluate gating accuracy

Usage:
    python eda_fft_gating.py --gpu 0 --domain domain_C --num-samples 50
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
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


def tensor_to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor image to numpy grayscale."""
    # (C, H, W) -> (H, W)
    img = img_tensor.mean(dim=0).numpy()  # Average RGB to grayscale
    return img


def compute_fft_features(img: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute various FFT features from an image.

    Returns:
        Dict with:
        - magnitude: Full FFT magnitude spectrum (H, W)
        - log_magnitude: Log-scaled magnitude
        - radial_profile: Radial average profile
        - horizontal_band: Horizontal frequency band (middle rows)
        - vertical_band: Vertical frequency band (middle cols)
    """
    # Compute 2D FFT
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    log_magnitude = np.log1p(magnitude)

    H, W = magnitude.shape
    center_h, center_w = H // 2, W // 2

    # Radial profile (average magnitude at each distance from center)
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((x - center_w)**2 + (y - center_h)**2).astype(int)
    max_r = min(center_h, center_w)
    radial_profile = np.zeros(max_r)
    for i in range(max_r):
        mask = r == i
        if mask.sum() > 0:
            radial_profile[i] = magnitude[mask].mean()

    # Horizontal band (low vertical freq, all horizontal freq)
    band_width = H // 8
    horizontal_band = magnitude[center_h - band_width:center_h + band_width, :].mean(axis=0)

    # Vertical band (all vertical freq, low horizontal freq)
    vertical_band = magnitude[:, center_w - band_width:center_w + band_width].mean(axis=1)

    return {
        'magnitude': magnitude,
        'log_magnitude': log_magnitude,
        'radial_profile': radial_profile,
        'horizontal_band': horizontal_band,
        'vertical_band': vertical_band,
    }


def compute_fft_similarity(fft1: Dict, fft2: Dict, method: str = 'cosine') -> float:
    """Compute similarity between two FFT feature sets.

    Methods:
    - cosine: Cosine similarity of flattened log magnitudes
    - correlation: Pearson correlation of log magnitudes
    - radial_cosine: Cosine similarity of radial profiles
    - band_cosine: Cosine similarity of combined H/V bands
    """
    if method == 'cosine':
        v1 = fft1['log_magnitude'].flatten()
        v2 = fft2['log_magnitude'].flatten()
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

    elif method == 'correlation':
        v1 = fft1['log_magnitude'].flatten()
        v2 = fft2['log_magnitude'].flatten()
        return stats.pearsonr(v1, v2)[0]

    elif method == 'radial_cosine':
        v1 = fft1['radial_profile']
        v2 = fft2['radial_profile']
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

    elif method == 'band_cosine':
        v1 = np.concatenate([fft1['horizontal_band'], fft1['vertical_band']])
        v2 = np.concatenate([fft2['horizontal_band'], fft2['vertical_band']])
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

    elif method == 'l2_distance':
        v1 = fft1['log_magnitude'].flatten()
        v2 = fft2['log_magnitude'].flatten()
        # Normalize to [0, 1] range for comparability
        v1 = v1 / (np.linalg.norm(v1) + 1e-8)
        v2 = v2 / (np.linalg.norm(v2) + 1e-8)
        # Return negative distance (higher = more similar)
        return -np.linalg.norm(v1 - v2)

    else:
        raise ValueError(f"Unknown method: {method}")


def analyze_fft_gating(
    dataset,
    cold_ref_idx: int,
    warm_ref_idx: int,
    test_indices: Dict[str, List[int]],
    similarity_methods: List[str],
) -> Dict[str, Dict[str, Dict]]:
    """Analyze FFT-based gating accuracy for all groups.

    Returns:
        {method: {group: {accuracy, predictions, gt_conditions}}}
    """
    # Load reference images
    cold_ref = dataset[cold_ref_idx]
    warm_ref = dataset[warm_ref_idx]

    cold_ref_gray = tensor_to_numpy(cold_ref.image)
    warm_ref_gray = tensor_to_numpy(warm_ref.image)

    cold_ref_fft = compute_fft_features(cold_ref_gray)
    warm_ref_fft = compute_fft_features(warm_ref_gray)

    results = {method: {} for method in similarity_methods}

    for group_name, indices in test_indices.items():
        gt_condition = 'cold' if 'cold' in group_name else 'warm'

        for method in similarity_methods:
            predictions = []
            sim_cold_list = []
            sim_warm_list = []

            for idx in indices:
                sample = dataset[idx]
                test_gray = tensor_to_numpy(sample.image)
                test_fft = compute_fft_features(test_gray)

                sim_cold = compute_fft_similarity(test_fft, cold_ref_fft, method)
                sim_warm = compute_fft_similarity(test_fft, warm_ref_fft, method)

                sim_cold_list.append(sim_cold)
                sim_warm_list.append(sim_warm)

                predicted = 'cold' if sim_cold > sim_warm else 'warm'
                predictions.append(predicted)

            correct = sum(1 for p in predictions if p == gt_condition)
            accuracy = correct / len(predictions) * 100

            results[method][group_name] = {
                'accuracy': accuracy,
                'predictions': predictions,
                'gt_condition': gt_condition,
                'sim_cold': sim_cold_list,
                'sim_warm': sim_warm_list,
                'correct': correct,
                'total': len(predictions),
            }

    return results


def print_results_table(results: Dict, methods: List[str], groups: List[str]):
    """Print formatted results table."""
    # Header
    print("\n" + "=" * 100)
    print("FFT GATING ACCURACY COMPARISON")
    print("=" * 100)

    header = f"{'Group':<15}"
    for method in methods:
        header += f" | {method:<15}"
    print(header)
    print("-" * 100)

    # Per-group results
    overall = {method: {'correct': 0, 'total': 0} for method in methods}

    for group in groups:
        row = f"{group:<15}"
        for method in methods:
            acc = results[method][group]['accuracy']
            row += f" | {acc:>13.1f}%"
            overall[method]['correct'] += results[method][group]['correct']
            overall[method]['total'] += results[method][group]['total']
        print(row)

    # Overall
    print("-" * 100)
    row = f"{'OVERALL':<15}"
    for method in methods:
        acc = overall[method]['correct'] / overall[method]['total'] * 100
        row += f" | {acc:>13.1f}%"
    print(row)
    print("=" * 100)


def visualize_fft_comparison(
    dataset,
    cold_ref_idx: int,
    warm_ref_idx: int,
    sample_indices: Dict[str, int],
    output_dir: Path,
):
    """Visualize FFT comparison for sample images."""
    # Load references
    cold_ref = tensor_to_numpy(dataset[cold_ref_idx].image)
    warm_ref = tensor_to_numpy(dataset[warm_ref_idx].image)

    cold_fft = compute_fft_features(cold_ref)
    warm_fft = compute_fft_features(warm_ref)

    fig, axes = plt.subplots(len(sample_indices) + 2, 4, figsize=(16, 4 * (len(sample_indices) + 2)))
    fig.suptitle("FFT Comparison: Reference vs Test Samples", fontsize=14, fontweight='bold')

    # Column titles
    for j, title in enumerate(['Image', 'FFT Magnitude', 'Radial Profile', 'H/V Bands']):
        axes[0, j].set_title(title, fontsize=10)

    # Row 0: Cold reference
    axes[0, 0].imshow(cold_ref, cmap='gray')
    axes[0, 0].set_ylabel("Cold Ref\n(1000)", fontsize=9)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cold_fft['log_magnitude'], cmap='viridis')
    axes[0, 1].axis('off')

    axes[0, 2].plot(cold_fft['radial_profile'], 'b-', label='Cold')
    axes[0, 2].legend(fontsize=8)

    axes[0, 3].plot(cold_fft['horizontal_band'], 'b-', alpha=0.7, label='H-band')
    axes[0, 3].plot(cold_fft['vertical_band'], 'b--', alpha=0.7, label='V-band')
    axes[0, 3].legend(fontsize=8)

    # Row 1: Warm reference
    axes[1, 0].imshow(warm_ref, cmap='gray')
    axes[1, 0].set_ylabel("Warm Ref\n(1999)", fontsize=9)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(warm_fft['log_magnitude'], cmap='viridis')
    axes[1, 1].axis('off')

    axes[1, 2].plot(warm_fft['radial_profile'], 'r-', label='Warm')
    axes[1, 2].legend(fontsize=8)

    axes[1, 3].plot(warm_fft['horizontal_band'], 'r-', alpha=0.7, label='H-band')
    axes[1, 3].plot(warm_fft['vertical_band'], 'r--', alpha=0.7, label='V-band')
    axes[1, 3].legend(fontsize=8)

    # Test samples
    for i, (group_name, idx) in enumerate(sample_indices.items()):
        row = i + 2
        sample = dataset[idx]
        test_gray = tensor_to_numpy(sample.image)
        test_fft = compute_fft_features(test_gray)

        # Compute similarities
        sim_cold = compute_fft_similarity(test_fft, cold_fft, 'cosine')
        sim_warm = compute_fft_similarity(test_fft, warm_fft, 'cosine')
        predicted = 'cold' if sim_cold > sim_warm else 'warm'
        gt = 'cold' if 'cold' in group_name else 'warm'
        correct = predicted == gt

        axes[row, 0].imshow(test_gray, cmap='gray')
        status = "OK" if correct else "WRONG"
        axes[row, 0].set_ylabel(f"{group_name}\n[{status}]", fontsize=9,
                                color='green' if correct else 'red')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(test_fft['log_magnitude'], cmap='viridis')
        axes[row, 1].axis('off')

        axes[row, 2].plot(cold_fft['radial_profile'], 'b-', alpha=0.5, label='Cold Ref')
        axes[row, 2].plot(warm_fft['radial_profile'], 'r-', alpha=0.5, label='Warm Ref')
        axes[row, 2].plot(test_fft['radial_profile'], 'g-', linewidth=2, label='Test')
        axes[row, 2].legend(fontsize=7)
        axes[row, 2].set_title(f"C:{sim_cold:.3f} W:{sim_warm:.3f}", fontsize=8)

        axes[row, 3].plot(cold_fft['horizontal_band'], 'b-', alpha=0.3)
        axes[row, 3].plot(warm_fft['horizontal_band'], 'r-', alpha=0.3)
        axes[row, 3].plot(test_fft['horizontal_band'], 'g-', linewidth=2)

    plt.tight_layout()
    save_path = output_dir / "fft_gating_visualization.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def visualize_similarity_distributions(
    results: Dict,
    methods: List[str],
    groups: List[str],
    output_dir: Path,
):
    """Visualize similarity score distributions for each method."""
    n_methods = len(methods)
    fig, axes = plt.subplots(n_methods, 2, figsize=(12, 4 * n_methods))
    fig.suptitle("FFT Similarity Distributions by Condition", fontsize=14, fontweight='bold')

    for i, method in enumerate(methods):
        ax_cold = axes[i, 0] if n_methods > 1 else axes[0]
        ax_warm = axes[i, 1] if n_methods > 1 else axes[1]

        # Collect similarities for cold and warm GT
        cold_gt_sims = {'cold': [], 'warm': []}
        warm_gt_sims = {'cold': [], 'warm': []}

        for group in groups:
            gt = results[method][group]['gt_condition']
            if gt == 'cold':
                cold_gt_sims['cold'].extend(results[method][group]['sim_cold'])
                cold_gt_sims['warm'].extend(results[method][group]['sim_warm'])
            else:
                warm_gt_sims['cold'].extend(results[method][group]['sim_cold'])
                warm_gt_sims['warm'].extend(results[method][group]['sim_warm'])

        # Plot for Cold GT samples
        ax_cold.hist(cold_gt_sims['cold'], bins=20, alpha=0.7, label='Sim to Cold Ref', color='blue')
        ax_cold.hist(cold_gt_sims['warm'], bins=20, alpha=0.7, label='Sim to Warm Ref', color='red')
        ax_cold.axvline(np.mean(cold_gt_sims['cold']), color='blue', linestyle='--', linewidth=2)
        ax_cold.axvline(np.mean(cold_gt_sims['warm']), color='red', linestyle='--', linewidth=2)
        ax_cold.set_title(f"{method}: Cold GT Samples", fontsize=10)
        ax_cold.legend(fontsize=8)
        ax_cold.set_xlabel("Similarity")

        # Plot for Warm GT samples
        ax_warm.hist(warm_gt_sims['cold'], bins=20, alpha=0.7, label='Sim to Cold Ref', color='blue')
        ax_warm.hist(warm_gt_sims['warm'], bins=20, alpha=0.7, label='Sim to Warm Ref', color='red')
        ax_warm.axvline(np.mean(warm_gt_sims['cold']), color='blue', linestyle='--', linewidth=2)
        ax_warm.axvline(np.mean(warm_gt_sims['warm']), color='red', linestyle='--', linewidth=2)
        ax_warm.set_title(f"{method}: Warm GT Samples", fontsize=10)
        ax_warm.legend(fontsize=8)
        ax_warm.set_xlabel("Similarity")

    plt.tight_layout()
    save_path = output_dir / "fft_similarity_distributions.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--domain", type=str, default="domain_C")
    parser.add_argument("--num-samples", type=int, default=50, help="Samples per group")
    args = parser.parse_args()

    # Setup
    dataset_root = "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_tiff_minmax"
    output_dir = Path(__file__).parent / args.domain / "fft_gating"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Domain: {args.domain}")
    print(f"Samples per group: {args.num_samples}")
    print(f"Output: {output_dir}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(dataset_root, args.domain)
    print(f"Dataset size: {len(dataset)}")

    # Reference indices (from test/good)
    # Cold ref: index 1000 (file 000000.tiff, most cold)
    # Warm ref: index 1999 (file 000999.tiff, most warm)
    cold_ref_idx = 1000
    warm_ref_idx = 1999

    print(f"\nCold reference: index {cold_ref_idx}")
    print(f"Warm reference: index {warm_ref_idx}")

    # Test indices (exclude reference images)
    # fault/cold: 0-499, fault/warm: 500-999
    # good/cold: 1000-1499, good/warm: 1500-1999
    n_samples = args.num_samples
    interval = max(1, 500 // n_samples)

    def get_indices(start, end, exclude=None):
        indices = list(range(start, end, interval))[:n_samples]
        if exclude:
            indices = [i for i in indices if i not in exclude]
        return indices

    test_indices = {
        'fault/cold': get_indices(0, 500),
        'fault/warm': get_indices(500, 1000),
        'good/cold': get_indices(1000, 1500, exclude={cold_ref_idx}),
        'good/warm': get_indices(1500, 2000, exclude={warm_ref_idx}),
    }

    for group, indices in test_indices.items():
        print(f"  {group}: {len(indices)} samples")

    # Similarity methods to test
    similarity_methods = [
        'cosine',           # Cosine similarity of full log FFT magnitude
        'correlation',      # Pearson correlation
        'radial_cosine',    # Cosine similarity of radial profile
        'band_cosine',      # Cosine similarity of H/V band features
        'l2_distance',      # Negative L2 distance (higher = more similar)
    ]

    # Analyze FFT gating
    print("\n" + "=" * 60)
    print("Analyzing FFT-based Gating...")
    print("=" * 60)

    results = analyze_fft_gating(
        dataset, cold_ref_idx, warm_ref_idx, test_indices, similarity_methods
    )

    # Print results
    groups = ['fault/cold', 'fault/warm', 'good/cold', 'good/warm']
    print_results_table(results, similarity_methods, groups)

    # Find best method
    overall_acc = {}
    for method in similarity_methods:
        correct = sum(results[method][g]['correct'] for g in groups)
        total = sum(results[method][g]['total'] for g in groups)
        overall_acc[method] = correct / total * 100

    best_method = max(overall_acc, key=overall_acc.get)
    print(f"\nBest FFT method: {best_method} ({overall_acc[best_method]:.1f}%)")

    # Compare with CLIP-based gating (from previous EDA)
    print("\n" + "-" * 60)
    print("Comparison with CLIP-based Gating (from previous EDA):")
    print("-" * 60)
    print("  CLIP Global: 87.5%")
    print("  CLIP Confidence: 88.8%")
    print(f"  FFT Best ({best_method}): {overall_acc[best_method]:.1f}%")

    if overall_acc[best_method] > 88.8:
        print("  => FFT-based gating is BETTER than CLIP!")
    else:
        print("  => CLIP-based gating is still better")

    # Visualizations
    print("\n" + "=" * 60)
    print("Creating Visualizations...")
    print("=" * 60)

    # Sample one from each group for visualization
    sample_indices = {
        'fault/cold': test_indices['fault/cold'][0],
        'fault/warm': test_indices['fault/warm'][0],
        'good/cold': test_indices['good/cold'][0],
        'good/warm': test_indices['good/warm'][0],
    }

    visualize_fft_comparison(dataset, cold_ref_idx, warm_ref_idx, sample_indices, output_dir)
    visualize_similarity_distributions(results, similarity_methods, groups, output_dir)

    # Save results summary
    summary_path = output_dir / "FFT_GATING_RESULTS.md"
    with open(summary_path, 'w') as f:
        f.write("# FFT-based Gating Analysis Results\n\n")
        f.write(f"## Domain: {args.domain}\n")
        f.write(f"## Samples per group: {args.num_samples}\n\n")

        f.write("## Results\n\n")
        f.write("| Group | " + " | ".join(similarity_methods) + " |\n")
        f.write("|-------|" + "|".join(["-----" for _ in similarity_methods]) + "|\n")

        for group in groups:
            row = f"| {group} |"
            for method in similarity_methods:
                acc = results[method][group]['accuracy']
                row += f" {acc:.1f}% |"
            f.write(row + "\n")

        f.write("| **OVERALL** |")
        for method in similarity_methods:
            f.write(f" **{overall_acc[method]:.1f}%** |")
        f.write("\n\n")

        f.write(f"## Best Method: {best_method} ({overall_acc[best_method]:.1f}%)\n\n")
        f.write("## Comparison with CLIP-based Gating\n")
        f.write("- CLIP Global: 87.5%\n")
        f.write("- CLIP Confidence: 88.8%\n")
        f.write(f"- FFT Best ({best_method}): {overall_acc[best_method]:.1f}%\n")

    print(f"\nSummary saved to: {summary_path}")
    print(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
