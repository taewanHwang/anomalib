#!/usr/bin/env python3
"""Patch Analysis with 2D FFT Transformed Images.

Applies 2D FFT to images before WinCLIP patch analysis.
Compares different resize methods on FFT-transformed images.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from anomalib.data.datasets.image.hdmap import HDMAPDataset
from anomalib.models.image.winclip.torch_model import WinClipModel
from anomalib.models.image.winclip.utils import cosine_similarity


ANOMALIB_ROOT = Path(__file__).parent.parent.parent.parent
DATASET_ROOT = ANOMALIB_ROOT / "datasets" / "HDMAP" / "1000_tiff_minmax"
OUTPUT_DIR = Path(__file__).parent / "patch_analysis"


def apply_2d_fft(image: torch.Tensor) -> torch.Tensor:
    """Apply 2D FFT to image and return magnitude spectrum.

    Args:
        image: Input image tensor (C, H, W)

    Returns:
        FFT magnitude spectrum as 3-channel image (C, H, W)
    """
    # Convert to grayscale if needed (use first channel or average)
    if image.shape[0] == 3:
        # Use average of channels
        gray = image.mean(dim=0)  # (H, W)
    else:
        gray = image[0]

    # Apply 2D FFT
    fft = torch.fft.fft2(gray)
    fft_shift = torch.fft.fftshift(fft)

    # Get magnitude spectrum (log scale for better visualization)
    magnitude = torch.abs(fft_shift)
    magnitude_log = torch.log1p(magnitude)  # log(1 + x) to avoid log(0)

    # Normalize to [0, 1] range
    mag_min = magnitude_log.min()
    mag_max = magnitude_log.max()
    magnitude_norm = (magnitude_log - mag_min) / (mag_max - mag_min + 1e-8)

    # Convert to 3-channel
    fft_image = magnitude_norm.unsqueeze(0).repeat(3, 1, 1)

    return fft_image


def load_image_with_fft(domain: str, split: str, label: str, index: int, resize_method: str):
    """Load image, apply FFT, then resize."""

    # First load original image without resize
    dataset_orig = HDMAPDataset(
        root=str(DATASET_ROOT),
        domain=domain,
        split=split,
        target_size=None,  # No resize first
    )

    target_path = f"{label}/{index:06d}.tiff"
    orig_image = None
    image_path = None

    for item in dataset_orig:
        if target_path in item.image_path:
            orig_image = item.image
            image_path = item.image_path
            break

    if orig_image is None:
        raise ValueError(f"Image not found: {target_path}")

    # Apply 2D FFT
    fft_image = apply_2d_fft(orig_image)  # (3, H, W)

    # Now resize the FFT image to 240x240
    fft_image = fft_image.unsqueeze(0)  # (1, 3, H, W)

    if resize_method == "resize":
        resized = F.interpolate(fft_image, size=(240, 240), mode='nearest')
    elif resize_method == "resize_bilinear":
        resized = F.interpolate(fft_image, size=(240, 240), mode='bilinear', align_corners=False)
    elif resize_method == "resize_aspect_padding":
        # Aspect ratio preserving resize
        _, _, h, w = fft_image.shape
        scale = min(240 / h, 240 / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = F.interpolate(fft_image, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # Pad to 240x240
        pad_h = 240 - new_h
        pad_w = 240 - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        resized = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)
    else:
        raise ValueError(f"Unknown resize method: {resize_method}")

    return resized.squeeze(0), orig_image, image_path  # (3, 240, 240), original, path


def analyze_fft_with_resize_method(
    resize_method: str,
    model: WinClipModel,
    device: torch.device,
    domain: str,
    test_label: str,
    test_index: int,
):
    """Analyze FFT patch matching with specific resize method."""

    print(f"\n{'='*60}")
    print(f"FFT + Resize Method: {resize_method}")
    print(f"{'='*60}")

    # Load images with FFT
    test_fft, test_orig, _ = load_image_with_fft(domain, "test", test_label, test_index, resize_method)
    cold_fft, cold_orig, _ = load_image_with_fft(domain, "test", "good", 0, resize_method)
    warm_fft, warm_orig, _ = load_image_with_fft(domain, "test", "good", 999, resize_method)

    # Encode FFT images
    test_batch = test_fft.unsqueeze(0).to(device)
    cold_batch = cold_fft.unsqueeze(0).to(device)
    warm_batch = warm_fft.unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, test_patch_emb = model.encode_image(test_batch)
        _, _, cold_patch_emb = model.encode_image(cold_batch)
        _, _, warm_patch_emb = model.encode_image(warm_batch)

    grid_size = model.grid_size  # (15, 15)

    # Compute similarities
    test_emb = test_patch_emb.squeeze(0)
    cold_emb = cold_patch_emb.squeeze(0)
    warm_emb = warm_patch_emb.squeeze(0)

    sim_to_cold = cosine_similarity(test_emb, cold_emb).squeeze(0).cpu().numpy()
    sim_to_warm = cosine_similarity(test_emb, warm_emb).squeeze(0).cpu().numpy()

    max_sim_cold = sim_to_cold.max(axis=1)
    max_sim_warm = sim_to_warm.max(axis=1)
    max_sim_combined = np.maximum(max_sim_cold, max_sim_warm)
    anomaly_scores = (1 - max_sim_combined) / 2

    print(f"Grid size: {grid_size[0]}x{grid_size[1]} = {grid_size[0]*grid_size[1]} patches")
    print(f"Max Sim to Cold: mean={max_sim_cold.mean():.4f}, min={max_sim_cold.min():.4f}, max={max_sim_cold.max():.4f}")
    print(f"Max Sim to Warm: mean={max_sim_warm.mean():.4f}, min={max_sim_warm.min():.4f}, max={max_sim_warm.max():.4f}")
    print(f"Anomaly Score:   mean={anomaly_scores.mean():.4f}, min={anomaly_scores.min():.4f}, max={anomaly_scores.max():.4f}")

    return {
        "resize_method": resize_method,
        "grid_size": grid_size,
        "test_fft": test_fft,
        "test_orig": test_orig,
        "cold_fft": cold_fft,
        "cold_orig": cold_orig,
        "warm_fft": warm_fft,
        "warm_orig": warm_orig,
        "max_sim_cold": max_sim_cold.reshape(grid_size),
        "max_sim_warm": max_sim_warm.reshape(grid_size),
        "anomaly_scores": anomaly_scores.reshape(grid_size),
    }


def visualize_fft_comparison(results_list, output_path, sample_type="fault"):
    """Visualize FFT analysis comparison."""

    n_methods = len(results_list)
    fig, axes = plt.subplots(n_methods, 6, figsize=(24, 4 * n_methods))

    def norm_display(t):
        arr = t.permute(1, 2, 0).numpy() if len(t.shape) == 3 else t.numpy()
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    for row, results in enumerate(results_list):
        method = results["resize_method"]

        # Column 1: Original image
        ax = axes[row, 0]
        ax.imshow(norm_display(results["test_orig"]))
        ax.set_title(f"{method}\nOriginal ({sample_type})", fontsize=10)
        ax.axis("off")

        # Column 2: FFT image
        ax = axes[row, 1]
        ax.imshow(norm_display(results["test_fft"]))
        ax.set_title(f"2D FFT\n(log magnitude)", fontsize=10)
        ax.axis("off")

        # Column 3: Cold ref FFT
        ax = axes[row, 2]
        ax.imshow(norm_display(results["cold_fft"]))
        ax.set_title(f"Cold Ref FFT", fontsize=10)
        ax.axis("off")

        # Column 4: Sim to Cold
        ax = axes[row, 3]
        im = ax.imshow(results["max_sim_cold"], cmap="Blues", vmin=0.5, vmax=1.0)
        ax.set_title(f"Sim to Cold\nmean={results['max_sim_cold'].mean():.3f}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Column 5: Sim to Warm
        ax = axes[row, 4]
        im = ax.imshow(results["max_sim_warm"], cmap="Reds", vmin=0.5, vmax=1.0)
        ax.set_title(f"Sim to Warm\nmean={results['max_sim_warm'].mean():.3f}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Column 6: Anomaly score
        ax = axes[row, 5]
        im = ax.imshow(results["anomaly_scores"], cmap="hot", vmin=0, vmax=0.25)
        ax.set_title(f"Anomaly Score\nmean={results['anomaly_scores'].mean():.3f}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f"2D FFT Patch Analysis: Resize Methods Comparison ({sample_type.upper()} Sample)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved to: {output_path}")


def print_summary(fault_results, good_results, title="FFT"):
    """Print summary comparison table."""

    print(f"\n{'='*80}")
    print(f"SUMMARY ({title}): Fault vs Good Discrimination")
    print("=" * 80)
    print(f"{'Method':<25} {'Fault Anomaly':<15} {'Good Anomaly':<15} {'Diff':<10} {'Discrimination'}")
    print("-" * 80)

    for fault_r, good_r in zip(fault_results, good_results):
        method = fault_r["resize_method"]
        fault_score = fault_r["anomaly_scores"].mean()
        good_score = good_r["anomaly_scores"].mean()
        diff = fault_score - good_score

        if diff > 0.02:
            rating = "Good"
        elif diff > 0.01:
            rating = "Moderate"
        elif diff > 0:
            rating = "Weak"
        else:
            rating = "FAILED"

        print(f"{method:<25} {fault_score:<15.4f} {good_score:<15.4f} {diff:<+10.4f} {rating}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="domain_C")
    parser.add_argument("--fault-index", type=int, default=9)
    parser.add_argument("--good-index", type=int, default=9)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model
    model = WinClipModel(class_name="industrial sensor data")
    model = model.to(device)
    model.eval()

    resize_methods = ["resize", "resize_bilinear", "resize_aspect_padding"]

    # Analyze fault sample with FFT
    print("\n" + "#" * 60)
    print("# FFT FAULT SAMPLE ANALYSIS")
    print("#" * 60)

    fault_results = []
    for method in resize_methods:
        results = analyze_fft_with_resize_method(
            method, model, device, args.domain, "fault", args.fault_index
        )
        fault_results.append(results)

    # Analyze good sample with FFT
    print("\n" + "#" * 60)
    print("# FFT GOOD SAMPLE ANALYSIS")
    print("#" * 60)

    good_results = []
    for method in resize_methods:
        results = analyze_fft_with_resize_method(
            method, model, device, args.domain, "good", args.good_index
        )
        good_results.append(results)

    # Print summary
    print_summary(fault_results, good_results, "2D FFT")

    # Visualize
    output_fault = OUTPUT_DIR / args.domain / f"fft_resize_methods_fault_{args.fault_index:06d}.png"
    output_good = OUTPUT_DIR / args.domain / f"fft_resize_methods_good_{args.good_index:06d}.png"

    visualize_fft_comparison(fault_results, output_fault, "fault")
    visualize_fft_comparison(good_results, output_good, "good")


if __name__ == "__main__":
    main()
