#!/usr/bin/env python3
"""Patch Analysis with Different Resize Methods.

Compares patch-level anomaly detection performance across:
- resize (nearest neighbor) - baseline
- resize_bilinear (smooth interpolation)
- resize_aspect_padding (aspect ratio preserved)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from anomalib.data.datasets.image.hdmap import HDMAPDataset
from anomalib.models.image.winclip.torch_model import WinClipModel
from anomalib.models.image.winclip.utils import cosine_similarity


ANOMALIB_ROOT = Path(__file__).parent.parent.parent.parent
DATASET_ROOT = ANOMALIB_ROOT / "datasets" / "HDMAP" / "1000_tiff_minmax"
OUTPUT_DIR = Path(__file__).parent / "patch_analysis"


def load_image(domain: str, split: str, label: str, index: int, resize_method: str):
    """Load image from HDMAPDataset with specified resize method."""
    dataset = HDMAPDataset(
        root=str(DATASET_ROOT),
        domain=domain,
        split=split,
        target_size=(240, 240),
        resize_method=resize_method,
    )

    target_path = f"{label}/{index:06d}.tiff"
    for item in dataset:
        if target_path in item.image_path:
            return item.image, item.image_path

    raise ValueError(f"Image not found: {target_path}")


def analyze_with_resize_method(
    resize_method: str,
    model: WinClipModel,
    device: torch.device,
    domain: str,
    test_label: str,
    test_index: int,
):
    """Analyze patch matching with specific resize method."""

    print(f"\n{'='*60}")
    print(f"Resize Method: {resize_method}")
    print(f"{'='*60}")

    # Load images
    test_image, _ = load_image(domain, "test", test_label, test_index, resize_method)
    cold_ref, _ = load_image(domain, "test", "good", 0, resize_method)
    warm_ref, _ = load_image(domain, "test", "good", 999, resize_method)

    # Encode images
    test_batch = test_image.unsqueeze(0).to(device)
    cold_batch = cold_ref.unsqueeze(0).to(device)
    warm_batch = warm_ref.unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, test_patch_emb = model.encode_image(test_batch)
        _, _, cold_patch_emb = model.encode_image(cold_batch)
        _, _, warm_patch_emb = model.encode_image(warm_batch)

    grid_size = model.grid_size  # (15, 15)

    # Compute similarities
    test_emb = test_patch_emb.squeeze(0)  # (225, 896)
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
        "test_image": test_image,
        "cold_ref": cold_ref,
        "warm_ref": warm_ref,
        "max_sim_cold": max_sim_cold.reshape(grid_size),
        "max_sim_warm": max_sim_warm.reshape(grid_size),
        "anomaly_scores": anomaly_scores.reshape(grid_size),
    }


def visualize_comparison(results_list, output_path):
    """Visualize comparison of different resize methods."""

    n_methods = len(results_list)
    fig, axes = plt.subplots(n_methods, 5, figsize=(20, 4 * n_methods))

    def norm_display(t):
        arr = t.permute(1, 2, 0).numpy()
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    for row, results in enumerate(results_list):
        method = results["resize_method"]
        grid_size = results["grid_size"]

        # Column 1: Test image
        ax = axes[row, 0]
        ax.imshow(norm_display(results["test_image"]))
        ax.set_title(f"{method}\nTest Image", fontsize=10)
        ax.axis("off")

        # Column 2: Cold reference
        ax = axes[row, 1]
        ax.imshow(norm_display(results["cold_ref"]))
        ax.set_title(f"Cold Ref (good/000000)", fontsize=10)
        ax.axis("off")

        # Column 3: Max sim to cold
        ax = axes[row, 2]
        im = ax.imshow(results["max_sim_cold"], cmap="Blues", vmin=0.5, vmax=1.0)
        ax.set_title(f"Sim to Cold\nmean={results['max_sim_cold'].mean():.3f}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Column 4: Max sim to warm
        ax = axes[row, 3]
        im = ax.imshow(results["max_sim_warm"], cmap="Reds", vmin=0.5, vmax=1.0)
        ax.set_title(f"Sim to Warm\nmean={results['max_sim_warm'].mean():.3f}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Column 5: Anomaly score
        ax = axes[row, 4]
        im = ax.imshow(results["anomaly_scores"], cmap="hot", vmin=0, vmax=0.25)
        ax.set_title(f"Anomaly Score\nmean={results['anomaly_scores'].mean():.3f}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("Patch Analysis: Resize Methods Comparison (Fault Sample)", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved to: {output_path}")


def visualize_good_comparison(results_list, output_path):
    """Visualize comparison for good sample."""

    n_methods = len(results_list)
    fig, axes = plt.subplots(n_methods, 5, figsize=(20, 4 * n_methods))

    def norm_display(t):
        arr = t.permute(1, 2, 0).numpy()
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    for row, results in enumerate(results_list):
        method = results["resize_method"]

        # Column 1: Test image (good)
        ax = axes[row, 0]
        ax.imshow(norm_display(results["test_image"]))
        ax.set_title(f"{method}\nGood Image", fontsize=10)
        ax.axis("off")

        # Column 2: Cold reference
        ax = axes[row, 1]
        ax.imshow(norm_display(results["cold_ref"]))
        ax.set_title(f"Cold Ref", fontsize=10)
        ax.axis("off")

        # Column 3: Max sim to cold
        ax = axes[row, 2]
        im = ax.imshow(results["max_sim_cold"], cmap="Blues", vmin=0.5, vmax=1.0)
        ax.set_title(f"Sim to Cold\nmean={results['max_sim_cold'].mean():.3f}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Column 4: Max sim to warm
        ax = axes[row, 3]
        im = ax.imshow(results["max_sim_warm"], cmap="Reds", vmin=0.5, vmax=1.0)
        ax.set_title(f"Sim to Warm\nmean={results['max_sim_warm'].mean():.3f}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Column 5: Anomaly score
        ax = axes[row, 4]
        im = ax.imshow(results["anomaly_scores"], cmap="hot", vmin=0, vmax=0.25)
        ax.set_title(f"Anomaly Score\nmean={results['anomaly_scores'].mean():.3f}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("Patch Analysis: Resize Methods Comparison (Good Sample)", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved to: {output_path}")


def print_summary(fault_results, good_results):
    """Print summary comparison table."""

    print("\n" + "=" * 80)
    print("SUMMARY: Fault vs Good Discrimination")
    print("=" * 80)
    print(f"{'Method':<25} {'Fault Anomaly':<15} {'Good Anomaly':<15} {'Diff':<10} {'Discrimination'}")
    print("-" * 80)

    for fault_r, good_r in zip(fault_results, good_results):
        method = fault_r["resize_method"]
        fault_score = fault_r["anomaly_scores"].mean()
        good_score = good_r["anomaly_scores"].mean()
        diff = fault_score - good_score

        # Discrimination rating
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

    # Analyze fault sample
    print("\n" + "#" * 60)
    print("# FAULT SAMPLE ANALYSIS")
    print("#" * 60)

    fault_results = []
    for method in resize_methods:
        results = analyze_with_resize_method(
            method, model, device, args.domain, "fault", args.fault_index
        )
        fault_results.append(results)

    # Analyze good sample (same index for fair comparison)
    print("\n" + "#" * 60)
    print("# GOOD SAMPLE ANALYSIS")
    print("#" * 60)

    good_results = []
    for method in resize_methods:
        results = analyze_with_resize_method(
            method, model, device, args.domain, "good", args.good_index
        )
        good_results.append(results)

    # Print summary
    print_summary(fault_results, good_results)

    # Visualize
    output_fault = OUTPUT_DIR / args.domain / f"resize_methods_fault_{args.fault_index:06d}.png"
    output_good = OUTPUT_DIR / args.domain / f"resize_methods_good_{args.good_index:06d}.png"

    visualize_comparison(fault_results, output_fault)
    visualize_good_comparison(good_results, output_good)


if __name__ == "__main__":
    main()
