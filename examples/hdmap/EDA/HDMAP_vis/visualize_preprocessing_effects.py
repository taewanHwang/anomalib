"""Visualize preprocessing effects to enhance horizontal defects.

This script compares various preprocessing techniques:
1. Sobel Filter (Horizontal) - Enhance horizontal edges/defects
2. Sobel Filter (Vertical) - For comparison
3. CLAHE - Contrast Limited Adaptive Histogram Equalization
4. Contrast Enhancement - Simple contrast stretching
5. Horizontal High-Pass Filter - Custom filter for horizontal patterns

Usage:
    python visualize_preprocessing_effects.py --gpu 0 --domain domain_C
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

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


def to_float(img):
    """Convert [0,255] uint8 to [0,1] float."""
    return img.astype(np.float32) / 255.0


# ============================================================================
# Preprocessing Functions
# ============================================================================

def apply_sobel_horizontal(img):
    """Apply Sobel filter to detect horizontal edges."""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2GRAY)
    else:
        gray = to_uint8(img)

    # Sobel horizontal (detects horizontal edges)
    sobel_h = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_h = np.abs(sobel_h)
    sobel_h = (sobel_h / sobel_h.max() * 255).astype(np.uint8) if sobel_h.max() > 0 else sobel_h.astype(np.uint8)

    return sobel_h


def apply_sobel_vertical(img):
    """Apply Sobel filter to detect vertical edges."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2GRAY)
    else:
        gray = to_uint8(img)

    # Sobel vertical (detects vertical edges)
    sobel_v = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_v = np.abs(sobel_v)
    sobel_v = (sobel_v / sobel_v.max() * 255).astype(np.uint8) if sobel_v.max() > 0 else sobel_v.astype(np.uint8)

    return sobel_v


def apply_clahe(img, clip_limit=2.0, tile_size=8):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    img_uint8 = to_uint8(img)

    # Apply CLAHE to each channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

    if len(img_uint8.shape) == 3:
        # Apply to each channel
        result = np.zeros_like(img_uint8)
        for i in range(3):
            result[:, :, i] = clahe.apply(img_uint8[:, :, i])
    else:
        result = clahe.apply(img_uint8)

    return result


def apply_contrast_stretch(img, percentile_low=2, percentile_high=98):
    """Apply contrast stretching based on percentiles."""
    img_uint8 = to_uint8(img)

    if len(img_uint8.shape) == 3:
        result = np.zeros_like(img_uint8)
        for i in range(3):
            channel = img_uint8[:, :, i]
            p_low = np.percentile(channel, percentile_low)
            p_high = np.percentile(channel, percentile_high)
            result[:, :, i] = np.clip((channel - p_low) * 255.0 / (p_high - p_low + 1e-8), 0, 255).astype(np.uint8)
    else:
        p_low = np.percentile(img_uint8, percentile_low)
        p_high = np.percentile(img_uint8, percentile_high)
        result = np.clip((img_uint8 - p_low) * 255.0 / (p_high - p_low + 1e-8), 0, 255).astype(np.uint8)

    return result


def apply_horizontal_highpass(img):
    """Apply horizontal high-pass filter to enhance horizontal patterns."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2GRAY)
    else:
        gray = to_uint8(img)

    # Horizontal high-pass kernel (enhances horizontal variations)
    kernel = np.array([[ 1,  2,  1],
                       [ 0,  0,  0],
                       [-1, -2, -1]], dtype=np.float32)

    filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    filtered = np.abs(filtered)
    filtered = (filtered / filtered.max() * 255).astype(np.uint8) if filtered.max() > 0 else filtered.astype(np.uint8)

    return filtered


def apply_row_difference(img):
    """Compute row-to-row difference to highlight horizontal anomalies."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = to_uint8(img).astype(np.float32)

    # Compute difference between adjacent rows
    row_diff = np.abs(np.diff(gray, axis=0))
    # Pad to maintain size
    row_diff = np.vstack([row_diff, row_diff[-1:, :]])
    row_diff = (row_diff / row_diff.max() * 255).astype(np.uint8) if row_diff.max() > 0 else row_diff.astype(np.uint8)

    return row_diff


def apply_combined_enhancement(img):
    """Combined: CLAHE + Sobel Horizontal overlay."""
    # First apply CLAHE
    clahe_result = apply_clahe(img, clip_limit=3.0)

    # Then compute horizontal sobel
    sobel_h = apply_sobel_horizontal(to_float(clahe_result))

    # Overlay sobel on CLAHE result
    if len(clahe_result.shape) == 3:
        gray_clahe = cv2.cvtColor(clahe_result, cv2.COLOR_RGB2GRAY)
    else:
        gray_clahe = clahe_result

    # Blend: original + enhanced edges
    combined = cv2.addWeighted(gray_clahe, 0.7, sobel_h, 0.3, 0)

    return combined


# ============================================================================
# Visualization
# ============================================================================

def visualize_preprocessing_comparison(dataset, fault_indices, good_indices, output_dir):
    """Create comparison figures for each preprocessing method."""

    preprocessing_methods = [
        ("Original", lambda x: to_uint8(x)),
        ("Sobel Horizontal", apply_sobel_horizontal),
        ("Sobel Vertical", apply_sobel_vertical),
        ("CLAHE", apply_clahe),
        ("Contrast Stretch", apply_contrast_stretch),
        ("Horizontal High-Pass", apply_horizontal_highpass),
        ("Row Difference", apply_row_difference),
        ("CLAHE + Sobel H", apply_combined_enhancement),
    ]

    n_samples = min(len(fault_indices), len(good_indices), 5)  # Show 5 samples each

    for method_name, method_func in preprocessing_methods:
        print(f"\nProcessing: {method_name}...")

        fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 7))
        fig.suptitle(f"Preprocessing: {method_name}\nTop: Fault/Cold | Bottom: Good/Cold", fontsize=12, fontweight='bold')

        # Top row: Fault/Cold
        for i, idx in enumerate(fault_indices[:n_samples]):
            sample = dataset[idx]
            img = tensor_to_numpy(sample.image)
            processed = method_func(img)

            ax = axes[0, i]
            if len(processed.shape) == 2:
                ax.imshow(processed, cmap='gray')
            else:
                ax.imshow(processed)

            file_idx = int(Path(sample.image_path).stem)
            ax.set_title(f"F:{file_idx:04d}", fontsize=9)
            ax.axis('off')

        # Bottom row: Good/Cold
        for i, idx in enumerate(good_indices[:n_samples]):
            sample = dataset[idx]
            img = tensor_to_numpy(sample.image)
            processed = method_func(img)

            ax = axes[1, i]
            if len(processed.shape) == 2:
                ax.imshow(processed, cmap='gray')
            else:
                ax.imshow(processed)

            file_idx = int(Path(sample.image_path).stem)
            ax.set_title(f"G:{file_idx:04d}", fontsize=9)
            ax.axis('off')

        axes[0, 0].set_ylabel("Fault/Cold", fontsize=10)
        axes[1, 0].set_ylabel("Good/Cold", fontsize=10)

        plt.tight_layout()
        safe_name = method_name.lower().replace(" ", "_").replace("+", "_")
        save_path = output_dir / f"preprocess_{safe_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()


def visualize_all_methods_single_sample(dataset, fault_idx, good_idx, output_dir):
    """Show all preprocessing methods for a single fault and good sample side by side."""

    preprocessing_methods = [
        ("Original", lambda x: to_uint8(x)),
        ("Sobel H", apply_sobel_horizontal),
        ("Sobel V", apply_sobel_vertical),
        ("CLAHE", apply_clahe),
        ("Contrast", apply_contrast_stretch),
        ("H-HighPass", apply_horizontal_highpass),
        ("Row Diff", apply_row_difference),
        ("CLAHE+Sobel", apply_combined_enhancement),
    ]

    n_methods = len(preprocessing_methods)

    fig, axes = plt.subplots(2, n_methods, figsize=(n_methods * 2.5, 6))
    fig.suptitle(f"All Preprocessing Methods\nTop: Fault/Cold | Bottom: Good/Cold", fontsize=12, fontweight='bold')

    # Load samples
    fault_sample = dataset[fault_idx]
    good_sample = dataset[good_idx]
    fault_img = tensor_to_numpy(fault_sample.image)
    good_img = tensor_to_numpy(good_sample.image)

    fault_file_idx = int(Path(fault_sample.image_path).stem)
    good_file_idx = int(Path(good_sample.image_path).stem)

    for i, (method_name, method_func) in enumerate(preprocessing_methods):
        # Fault
        processed_fault = method_func(fault_img)
        ax = axes[0, i]
        if len(processed_fault.shape) == 2:
            ax.imshow(processed_fault, cmap='gray')
        else:
            ax.imshow(processed_fault)
        if i == 0:
            ax.set_ylabel(f"Fault\n{fault_file_idx:04d}", fontsize=9)
        ax.set_title(method_name, fontsize=8)
        ax.axis('off')

        # Good
        processed_good = method_func(good_img)
        ax = axes[1, i]
        if len(processed_good.shape) == 2:
            ax.imshow(processed_good, cmap='gray')
        else:
            ax.imshow(processed_good)
        if i == 0:
            ax.set_ylabel(f"Good\n{good_file_idx:04d}", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    save_path = output_dir / f"all_methods_comparison_f{fault_file_idx:04d}_g{good_file_idx:04d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--domain", type=str, default="domain_C")
    args = parser.parse_args()

    # Setup
    dataset_root = "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_tiff_minmax"
    output_dir = Path(__file__).parent / args.domain / "preprocessing"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Domain: {args.domain}")
    print(f"Output: {output_dir}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(dataset_root, args.domain)
    print(f"Dataset size: {len(dataset)}")

    # Select samples
    # fault/cold: indices 0-499
    # good/cold: indices 1000-1499
    fault_cold_indices = [0, 10, 20, 30, 40, 50, 100, 150, 200, 250]
    good_cold_indices = [1000, 1010, 1020, 1030, 1040, 1050, 1100, 1150, 1200, 1250]

    print(f"\nFault/cold samples: {fault_cold_indices[:5]}...")
    print(f"Good/cold samples: {good_cold_indices[:5]}...")

    # Create comparison figures for each method
    print("\n" + "=" * 60)
    print("Creating per-method comparison figures...")
    print("=" * 60)
    visualize_preprocessing_comparison(dataset, fault_cold_indices, good_cold_indices, output_dir)

    # Create all-methods comparison for a few sample pairs
    print("\n" + "=" * 60)
    print("Creating all-methods comparison figures...")
    print("=" * 60)

    for fault_idx, good_idx in [(0, 1000), (50, 1050), (100, 1100)]:
        visualize_all_methods_single_sample(dataset, fault_idx, good_idx, output_dir)

    print(f"\nAll visualizations saved to: {output_dir}")
    print(f"Total figures: {len(list(output_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
