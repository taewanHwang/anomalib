#!/usr/bin/env python3
"""Patch Analysis with Morphological Dilation.

Applies morphological dilation to enhance thin defects before WinCLIP analysis.
"""

import argparse
from pathlib import Path

import cv2
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


def apply_dilation(image: torch.Tensor, kernel_size: int = 3, iterations: int = 1) -> torch.Tensor:
    """Apply morphological dilation to image.

    Args:
        image: Input image tensor (C, H, W)
        kernel_size: Size of dilation kernel
        iterations: Number of dilation iterations

    Returns:
        Dilated image tensor (C, H, W)
    """
    # Convert to numpy for OpenCV
    img_np = image.permute(1, 2, 0).numpy()  # (H, W, C)

    # Create kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Apply dilation to each channel
    dilated_channels = []
    for c in range(img_np.shape[2]):
        channel = img_np[:, :, c]
        dilated = cv2.dilate(channel, kernel, iterations=iterations)
        dilated_channels.append(dilated)

    dilated_np = np.stack(dilated_channels, axis=2)  # (H, W, C)

    # Convert back to tensor
    return torch.from_numpy(dilated_np).permute(2, 0, 1).float()  # (C, H, W)


def apply_erosion(image: torch.Tensor, kernel_size: int = 3, iterations: int = 1) -> torch.Tensor:
    """Apply morphological erosion to image (for dark defects)."""
    img_np = image.permute(1, 2, 0).numpy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    eroded_channels = []
    for c in range(img_np.shape[2]):
        channel = img_np[:, :, c]
        eroded = cv2.erode(channel, kernel, iterations=iterations)
        eroded_channels.append(eroded)

    eroded_np = np.stack(eroded_channels, axis=2)
    return torch.from_numpy(eroded_np).permute(2, 0, 1).float()


def load_and_preprocess(domain: str, split: str, label: str, index: int,
                        preprocess: str = "none", kernel_size: int = 3, iterations: int = 1):
    """Load image and apply preprocessing."""

    # Load original image (no resize yet)
    dataset = HDMAPDataset(
        root=str(DATASET_ROOT),
        domain=domain,
        split=split,
        target_size=None,
    )

    target_path = f"{label}/{index:06d}.tiff"
    orig_image = None
    image_path = None

    for item in dataset:
        if target_path in item.image_path:
            orig_image = item.image
            image_path = item.image_path
            break

    if orig_image is None:
        raise ValueError(f"Image not found: {target_path}")

    # Apply preprocessing
    if preprocess == "none":
        processed = orig_image
    elif preprocess == "dilation":
        processed = apply_dilation(orig_image, kernel_size, iterations)
    elif preprocess == "erosion":
        processed = apply_erosion(orig_image, kernel_size, iterations)
    else:
        raise ValueError(f"Unknown preprocess: {preprocess}")

    # Resize to 240x240
    processed = processed.unsqueeze(0)  # (1, C, H, W)
    resized = F.interpolate(processed, size=(240, 240), mode='bilinear', align_corners=False)

    return resized.squeeze(0), orig_image, image_path


def analyze_with_preprocess(
    preprocess: str,
    kernel_size: int,
    iterations: int,
    model: WinClipModel,
    device: torch.device,
    domain: str,
    test_label: str,
    test_index: int,
):
    """Analyze patch matching with specific preprocessing."""

    label = f"{preprocess}" if preprocess == "none" else f"{preprocess}_k{kernel_size}_i{iterations}"
    print(f"\n{'='*60}")
    print(f"Preprocess: {label}")
    print(f"{'='*60}")

    # Load and preprocess images
    test_img, test_orig, _ = load_and_preprocess(
        domain, "test", test_label, test_index, preprocess, kernel_size, iterations
    )
    cold_img, cold_orig, _ = load_and_preprocess(
        domain, "test", "good", 0, preprocess, kernel_size, iterations
    )
    warm_img, warm_orig, _ = load_and_preprocess(
        domain, "test", "good", 999, preprocess, kernel_size, iterations
    )

    # Encode images
    test_batch = test_img.unsqueeze(0).to(device)
    cold_batch = cold_img.unsqueeze(0).to(device)
    warm_batch = warm_img.unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, test_patch_emb = model.encode_image(test_batch)
        _, _, cold_patch_emb = model.encode_image(cold_batch)
        _, _, warm_patch_emb = model.encode_image(warm_batch)

    grid_size = model.grid_size

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

    print(f"Anomaly Score: mean={anomaly_scores.mean():.4f}, min={anomaly_scores.min():.4f}, max={anomaly_scores.max():.4f}")

    return {
        "label": label,
        "preprocess": preprocess,
        "kernel_size": kernel_size,
        "iterations": iterations,
        "grid_size": grid_size,
        "test_img": test_img,
        "test_orig": test_orig,
        "cold_img": cold_img,
        "warm_img": warm_img,
        "max_sim_cold": max_sim_cold.reshape(grid_size),
        "max_sim_warm": max_sim_warm.reshape(grid_size),
        "anomaly_scores": anomaly_scores.reshape(grid_size),
    }


def visualize_dilation_effect(fault_results, good_results, output_path):
    """Visualize dilation effect on fault detection."""

    n_configs = len(fault_results)
    fig, axes = plt.subplots(n_configs, 6, figsize=(24, 4 * n_configs))

    def norm_display(t):
        arr = t.permute(1, 2, 0).numpy() if len(t.shape) == 3 else t.numpy()
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    for row, (fault_r, good_r) in enumerate(zip(fault_results, good_results)):
        label = fault_r["label"]

        # Column 1: Original fault
        ax = axes[row, 0]
        ax.imshow(norm_display(fault_r["test_orig"]))
        ax.set_title(f"{label}\nOriginal Fault", fontsize=10)
        ax.axis("off")

        # Column 2: Processed fault (resized)
        ax = axes[row, 1]
        ax.imshow(norm_display(fault_r["test_img"]))
        ax.set_title(f"Processed\n(240x240)", fontsize=10)
        ax.axis("off")

        # Column 3: Fault anomaly map
        ax = axes[row, 2]
        im = ax.imshow(fault_r["anomaly_scores"], cmap="hot", vmin=0, vmax=0.2)
        ax.set_title(f"Fault Anomaly\nmean={fault_r['anomaly_scores'].mean():.4f}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Column 4: Good anomaly map
        ax = axes[row, 3]
        im = ax.imshow(good_r["anomaly_scores"], cmap="hot", vmin=0, vmax=0.2)
        ax.set_title(f"Good Anomaly\nmean={good_r['anomaly_scores'].mean():.4f}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Column 5: Difference (Fault - Good)
        diff = fault_r["anomaly_scores"] - good_r["anomaly_scores"]
        ax = axes[row, 4]
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-0.1, vmax=0.1)
        ax.set_title(f"Fault - Good\nmean={diff.mean():.4f}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Column 6: Stats text
        ax = axes[row, 5]
        ax.axis("off")

        fault_mean = fault_r["anomaly_scores"].mean()
        good_mean = good_r["anomaly_scores"].mean()
        discrimination = fault_mean - good_mean

        if discrimination > 0.02:
            rating = "GOOD"
            color = "green"
        elif discrimination > 0.01:
            rating = "Moderate"
            color = "orange"
        elif discrimination > 0:
            rating = "Weak"
            color = "darkorange"
        else:
            rating = "FAILED"
            color = "red"

        stats_text = f"""
Configuration: {label}

Fault Anomaly: {fault_mean:.4f}
Good Anomaly:  {good_mean:.4f}
─────────────────────
Difference:    {discrimination:+.4f}

Discrimination: {rating}
        """
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle("Morphological Dilation Effect on Fault Detection (Domain C)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved to: {output_path}")


def print_summary(fault_results, good_results):
    """Print summary table."""

    print("\n" + "=" * 80)
    print("SUMMARY: Dilation Effect on Fault vs Good Discrimination")
    print("=" * 80)
    print(f"{'Config':<25} {'Fault':<12} {'Good':<12} {'Diff':<12} {'Rating'}")
    print("-" * 80)

    for fault_r, good_r in zip(fault_results, good_results):
        label = fault_r["label"]
        fault_score = fault_r["anomaly_scores"].mean()
        good_score = good_r["anomaly_scores"].mean()
        diff = fault_score - good_score

        if diff > 0.02:
            rating = "GOOD"
        elif diff > 0.01:
            rating = "Moderate"
        elif diff > 0:
            rating = "Weak"
        else:
            rating = "FAILED"

        print(f"{label:<25} {fault_score:<12.4f} {good_score:<12.4f} {diff:<+12.4f} {rating}")

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

    # Test configurations: (preprocess, kernel_size, iterations)
    configs = [
        ("none", 0, 0),           # Baseline
        ("dilation", 3, 1),       # Small dilation
        ("dilation", 5, 1),       # Medium dilation
        ("dilation", 3, 2),       # Double iteration
        ("dilation", 5, 2),       # Medium + double
        ("dilation", 7, 1),       # Large kernel
        ("erosion", 3, 1),        # Try erosion (for dark defects)
    ]

    # Analyze fault samples
    print("\n" + "#" * 60)
    print(f"# FAULT SAMPLE ANALYSIS (index={args.fault_index})")
    print("#" * 60)

    fault_results = []
    for preprocess, kernel_size, iterations in configs:
        results = analyze_with_preprocess(
            preprocess, kernel_size, iterations,
            model, device, args.domain, "fault", args.fault_index
        )
        fault_results.append(results)

    # Analyze good samples
    print("\n" + "#" * 60)
    print(f"# GOOD SAMPLE ANALYSIS (index={args.good_index})")
    print("#" * 60)

    good_results = []
    for preprocess, kernel_size, iterations in configs:
        results = analyze_with_preprocess(
            preprocess, kernel_size, iterations,
            model, device, args.domain, "good", args.good_index
        )
        good_results.append(results)

    # Print summary
    print_summary(fault_results, good_results)

    # Visualize
    output_path = OUTPUT_DIR / args.domain / f"dilation_effect_fault{args.fault_index:06d}_good{args.good_index:06d}.png"
    visualize_dilation_effect(fault_results, good_results, output_path)


if __name__ == "__main__":
    main()
