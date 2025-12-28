#!/usr/bin/env python3
"""WinCLIP Patch Matching Analysis.

Analyzes which reference patches each test patch matches with,
to understand why Cold Fault detection is difficult.

This script visualizes:
1. Test image with patch grid overlay
2. Per-patch similarity to Cold reference vs Warm reference
3. Which reference (cold/warm) each patch prefers
4. Anomaly score distribution

Usage:
    python analyze_patch_matching.py --test-image 000009 --domain domain_C
"""

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from anomalib.data.datasets.image.hdmap import HDMAPDataset
from anomalib.models.image.winclip.torch_model import WinClipModel
from anomalib.models.image.winclip.utils import cosine_similarity


# Default paths
ANOMALIB_ROOT = Path(__file__).parent.parent.parent.parent
DATASET_ROOT = ANOMALIB_ROOT / "datasets" / "HDMAP" / "1000_tiff_minmax"
OUTPUT_DIR = Path(__file__).parent / "patch_analysis"


def load_image(domain: str, split: str, label: str, index: int, image_size: int = 240) -> Tuple[torch.Tensor, str]:
    """Load a single image from HDMAP dataset using HDMAPDataset.

    Args:
        domain: Domain name (e.g., "domain_C")
        split: "train" or "test"
        label: "good" or "fault"
        index: File index (0-999)
        image_size: Target image size

    Returns:
        Tuple of (image_tensor, image_path)
    """
    # Use HDMAPDataset to load images consistently
    dataset = HDMAPDataset(
        root=str(DATASET_ROOT),
        domain=domain,
        split=split,
        target_size=(image_size, image_size),
        resize_method="resize",
    )

    # Find the correct index in dataset
    # Dataset structure: good (0-999), fault (0-999)
    # In test split: good samples first, then fault samples
    image_path = DATASET_ROOT / domain / split / label / f"{index:06d}.tiff"

    # Search for the matching item
    for item in dataset:
        if str(image_path) in item.image_path or item.image_path.endswith(f"{label}/{index:06d}.tiff"):
            return item.image, item.image_path

    # Fallback: manual loading if not found
    import tifffile
    from torchvision.transforms import functional as TF
    from PIL import Image

    img_array = tifffile.imread(str(image_path))
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[0] == 3:
        img_array = np.transpose(img_array, (1, 2, 0))

    # Resize using PIL (same as HDMAPDataset)
    img_pil = Image.fromarray((img_array * 255).clip(0, 255).astype(np.uint8))
    img_pil = img_pil.resize((image_size, image_size), Image.BILINEAR)

    # Convert back to float tensor (HDMAPDataset keeps original range)
    img_array_resized = np.array(img_pil).astype(np.float32) / 255.0
    # Scale back to original range
    orig_min, orig_max = img_array.min(), img_array.max()
    img_array_resized = img_array_resized * (orig_max - orig_min) + orig_min

    img_tensor = torch.from_numpy(img_array_resized).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

    return img_tensor, str(image_path)


def analyze_patch_matching(
    test_image: torch.Tensor,
    cold_ref: torch.Tensor,
    warm_ref: torch.Tensor,
    model: WinClipModel,
    device: torch.device,
) -> dict:
    """Analyze patch-level matching between test and reference images.

    Args:
        test_image: Test image tensor (3, H, W)
        cold_ref: Cold reference image tensor (3, H, W)
        warm_ref: Warm reference image tensor (3, H, W)
        model: WinClipModel instance
        device: Torch device

    Returns:
        Dictionary with analysis results
    """
    # Move to device and add batch dimension
    test_batch = test_image.unsqueeze(0).to(device)
    cold_batch = cold_ref.unsqueeze(0).to(device)
    warm_batch = warm_ref.unsqueeze(0).to(device)

    # Extract embeddings
    with torch.no_grad():
        # Test image
        test_img_emb, test_visual_emb, test_patch_emb = model.encode_image(test_batch)

        # Cold reference
        cold_img_emb, cold_visual_emb, cold_patch_emb = model.encode_image(cold_batch)

        # Warm reference
        warm_img_emb, warm_visual_emb, warm_patch_emb = model.encode_image(warm_batch)

    grid_size = model.grid_size  # (15, 15) for 240x240 with patch_size=16
    n_patches = grid_size[0] * grid_size[1]  # 225

    # Compute similarity matrices
    # test_patch_emb: (1, 225, 896)
    # cold_patch_emb: (1, 225, 896)

    # Similarity between each test patch and each cold reference patch
    test_emb = test_patch_emb.squeeze(0)  # (225, 896)
    cold_emb = cold_patch_emb.squeeze(0)  # (225, 896)
    warm_emb = warm_patch_emb.squeeze(0)  # (225, 896)

    # Compute pairwise cosine similarity
    sim_to_cold = cosine_similarity(test_emb, cold_emb).squeeze(0)  # (225, 225)
    sim_to_warm = cosine_similarity(test_emb, warm_emb).squeeze(0)  # (225, 225)

    # For each test patch, find max similarity to any cold/warm patch
    max_sim_cold, best_cold_patch = sim_to_cold.max(dim=1)  # (225,)
    max_sim_warm, best_warm_patch = sim_to_warm.max(dim=1)  # (225,)

    # Which reference does each patch prefer?
    prefers_cold = max_sim_cold > max_sim_warm  # (225,)

    # Anomaly scores (1 - max_sim) / 2
    # Combined: use max across both references
    max_sim_combined = torch.maximum(max_sim_cold, max_sim_warm)
    anomaly_scores = (1 - max_sim_combined) / 2

    # Anomaly scores using only cold reference
    anomaly_scores_cold_only = (1 - max_sim_cold) / 2

    # Anomaly scores using only warm reference
    anomaly_scores_warm_only = (1 - max_sim_warm) / 2

    return {
        "grid_size": grid_size,
        "n_patches": n_patches,
        # Similarity matrices
        "sim_to_cold": sim_to_cold.cpu().numpy(),  # (225, 225)
        "sim_to_warm": sim_to_warm.cpu().numpy(),  # (225, 225)
        # Per-patch max similarity
        "max_sim_cold": max_sim_cold.cpu().numpy(),  # (225,)
        "max_sim_warm": max_sim_warm.cpu().numpy(),  # (225,)
        # Best matching patch indices
        "best_cold_patch": best_cold_patch.cpu().numpy(),  # (225,)
        "best_warm_patch": best_warm_patch.cpu().numpy(),  # (225,)
        # Preference
        "prefers_cold": prefers_cold.cpu().numpy(),  # (225,)
        # Anomaly scores
        "anomaly_scores": anomaly_scores.cpu().numpy(),  # (225,)
        "anomaly_scores_cold_only": anomaly_scores_cold_only.cpu().numpy(),
        "anomaly_scores_warm_only": anomaly_scores_warm_only.cpu().numpy(),
        # Global image embeddings for comparison
        "test_img_emb": test_img_emb.cpu().numpy(),
        "cold_img_emb": cold_img_emb.cpu().numpy(),
        "warm_img_emb": warm_img_emb.cpu().numpy(),
    }


def visualize_analysis(
    test_image: torch.Tensor,
    cold_ref: torch.Tensor,
    warm_ref: torch.Tensor,
    analysis: dict,
    test_info: str,
    output_path: Path,
) -> None:
    """Visualize patch matching analysis.

    Creates a comprehensive visualization showing:
    1. Original images
    2. Per-patch similarity to cold/warm
    3. Reference preference map
    4. Anomaly score maps
    """
    grid_size = analysis["grid_size"]

    # Convert tensors to numpy for display (normalize to [0, 1] for visualization only)
    def normalize_for_display(img_tensor):
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_min, img_max = img_np.min(), img_np.max()
        if img_max > img_min:
            return (img_np - img_min) / (img_max - img_min)
        return np.zeros_like(img_np)

    test_np = normalize_for_display(test_image)
    cold_np = normalize_for_display(cold_ref)
    warm_np = normalize_for_display(warm_ref)

    # Store original ranges for display
    test_range = (test_image.min().item(), test_image.max().item())
    cold_range = (cold_ref.min().item(), cold_ref.max().item())
    warm_range = (warm_ref.min().item(), warm_ref.max().item())

    # Reshape to grid
    max_sim_cold_grid = analysis["max_sim_cold"].reshape(grid_size)
    max_sim_warm_grid = analysis["max_sim_warm"].reshape(grid_size)
    prefers_cold_grid = analysis["prefers_cold"].reshape(grid_size)
    anomaly_grid = analysis["anomaly_scores"].reshape(grid_size)
    anomaly_cold_grid = analysis["anomaly_scores_cold_only"].reshape(grid_size)
    anomaly_warm_grid = analysis["anomaly_scores_warm_only"].reshape(grid_size)

    # Create figure
    fig = plt.figure(figsize=(20, 16))

    # Row 1: Original images
    ax1 = fig.add_subplot(4, 4, 1)
    ax1.imshow(cold_np)
    ax1.set_title(f"Cold Reference (000000.tiff)\nrange: [{cold_range[0]:.2f}, {cold_range[1]:.2f}]", fontsize=10)
    ax1.axis("off")

    ax2 = fig.add_subplot(4, 4, 2)
    ax2.imshow(test_np)
    ax2.set_title(f"Test Image ({test_info})\nrange: [{test_range[0]:.2f}, {test_range[1]:.2f}]", fontsize=10)
    ax2.axis("off")

    ax3 = fig.add_subplot(4, 4, 3)
    ax3.imshow(warm_np)
    ax3.set_title(f"Warm Reference (000999.tiff)\nrange: [{warm_range[0]:.2f}, {warm_range[1]:.2f}]", fontsize=10)
    ax3.axis("off")

    # Intensity histograms (use original values, not normalized)
    ax4 = fig.add_subplot(4, 4, 4)
    cold_orig = cold_ref.permute(1, 2, 0).numpy().mean(axis=2).flatten()
    test_orig = test_image.permute(1, 2, 0).numpy().mean(axis=2).flatten()
    warm_orig = warm_ref.permute(1, 2, 0).numpy().mean(axis=2).flatten()
    ax4.hist(cold_orig, bins=50, alpha=0.5, label=f"Cold Ref", color="blue")
    ax4.hist(test_orig, bins=50, alpha=0.5, label=f"Test", color="green")
    ax4.hist(warm_orig, bins=50, alpha=0.5, label=f"Warm Ref", color="red")
    ax4.set_title("Intensity Distribution (Original)", fontsize=10)
    ax4.legend(fontsize=8)
    ax4.set_xlabel("Intensity (Original Values)")

    # Row 2: Similarity maps
    ax5 = fig.add_subplot(4, 4, 5)
    im5 = ax5.imshow(max_sim_cold_grid, cmap="Blues", vmin=0.5, vmax=1.0)
    ax5.set_title(f"Max Sim to Cold Ref\n(mean={max_sim_cold_grid.mean():.3f})", fontsize=10)
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    ax5.axis("off")

    ax6 = fig.add_subplot(4, 4, 6)
    im6 = ax6.imshow(max_sim_warm_grid, cmap="Reds", vmin=0.5, vmax=1.0)
    ax6.set_title(f"Max Sim to Warm Ref\n(mean={max_sim_warm_grid.mean():.3f})", fontsize=10)
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    ax6.axis("off")

    ax7 = fig.add_subplot(4, 4, 7)
    # Difference: cold - warm (positive = prefers cold)
    diff_grid = max_sim_cold_grid - max_sim_warm_grid
    im7 = ax7.imshow(diff_grid, cmap="RdBu", vmin=-0.2, vmax=0.2)
    ax7.set_title(f"Sim Diff (Cold - Warm)\n(mean={diff_grid.mean():.3f})", fontsize=10)
    plt.colorbar(im7, ax=ax7, fraction=0.046)
    ax7.axis("off")

    ax8 = fig.add_subplot(4, 4, 8)
    # Create custom colormap for preference
    cmap_pref = LinearSegmentedColormap.from_list("pref", ["red", "blue"])
    im8 = ax8.imshow(prefers_cold_grid.astype(float), cmap=cmap_pref, vmin=0, vmax=1)
    cold_pct = prefers_cold_grid.mean() * 100
    ax8.set_title(f"Prefers Cold? (Blue=Cold)\n({cold_pct:.1f}% patches)", fontsize=10)
    ax8.axis("off")

    # Row 3: Anomaly scores
    ax9 = fig.add_subplot(4, 4, 9)
    im9 = ax9.imshow(anomaly_cold_grid, cmap="hot", vmin=0, vmax=0.3)
    ax9.set_title(f"Anomaly (Cold Ref Only)\n(mean={anomaly_cold_grid.mean():.3f})", fontsize=10)
    plt.colorbar(im9, ax=ax9, fraction=0.046)
    ax9.axis("off")

    ax10 = fig.add_subplot(4, 4, 10)
    im10 = ax10.imshow(anomaly_warm_grid, cmap="hot", vmin=0, vmax=0.3)
    ax10.set_title(f"Anomaly (Warm Ref Only)\n(mean={anomaly_warm_grid.mean():.3f})", fontsize=10)
    plt.colorbar(im10, ax=ax10, fraction=0.046)
    ax10.axis("off")

    ax11 = fig.add_subplot(4, 4, 11)
    im11 = ax11.imshow(anomaly_grid, cmap="hot", vmin=0, vmax=0.3)
    ax11.set_title(f"Anomaly (Combined)\n(mean={anomaly_grid.mean():.3f})", fontsize=10)
    plt.colorbar(im11, ax=ax11, fraction=0.046)
    ax11.axis("off")

    ax12 = fig.add_subplot(4, 4, 12)
    # Show which reference contributes to final score
    uses_cold = analysis["max_sim_cold"] >= analysis["max_sim_warm"]
    uses_cold_grid = uses_cold.reshape(grid_size)
    anomaly_source = np.where(uses_cold_grid, anomaly_cold_grid, anomaly_warm_grid)
    im12 = ax12.imshow(anomaly_source, cmap="hot", vmin=0, vmax=0.3)
    ax12.set_title("Anomaly (Per-patch Best)\n", fontsize=10)
    plt.colorbar(im12, ax=ax12, fraction=0.046)
    ax12.axis("off")

    # Row 4: Statistics and summary
    ax13 = fig.add_subplot(4, 4, 13)
    ax13.hist(analysis["max_sim_cold"], bins=30, alpha=0.6, label="To Cold", color="blue")
    ax13.hist(analysis["max_sim_warm"], bins=30, alpha=0.6, label="To Warm", color="red")
    ax13.axvline(analysis["max_sim_cold"].mean(), color="blue", linestyle="--", label=f"Cold mean: {analysis['max_sim_cold'].mean():.3f}")
    ax13.axvline(analysis["max_sim_warm"].mean(), color="red", linestyle="--", label=f"Warm mean: {analysis['max_sim_warm'].mean():.3f}")
    ax13.set_title("Similarity Distribution", fontsize=10)
    ax13.set_xlabel("Max Cosine Similarity")
    ax13.legend(fontsize=7)

    ax14 = fig.add_subplot(4, 4, 14)
    ax14.hist(analysis["anomaly_scores_cold_only"], bins=30, alpha=0.6, label="Cold Only", color="blue")
    ax14.hist(analysis["anomaly_scores_warm_only"], bins=30, alpha=0.6, label="Warm Only", color="red")
    ax14.hist(analysis["anomaly_scores"], bins=30, alpha=0.6, label="Combined", color="green")
    ax14.set_title("Anomaly Score Distribution", fontsize=10)
    ax14.set_xlabel("Anomaly Score")
    ax14.legend(fontsize=7)

    # Global embedding similarities
    ax15 = fig.add_subplot(4, 4, 15)
    test_emb = analysis["test_img_emb"].flatten()
    cold_emb = analysis["cold_img_emb"].flatten()
    warm_emb = analysis["warm_img_emb"].flatten()

    global_sim_cold = np.dot(test_emb, cold_emb) / (np.linalg.norm(test_emb) * np.linalg.norm(cold_emb))
    global_sim_warm = np.dot(test_emb, warm_emb) / (np.linalg.norm(test_emb) * np.linalg.norm(warm_emb))

    bars = ax15.bar(["Cold", "Warm"], [global_sim_cold, global_sim_warm], color=["blue", "red"])
    ax15.set_ylim(0.5, 1.0)
    ax15.set_title("Global Image Similarity", fontsize=10)
    ax15.set_ylabel("Cosine Similarity")
    for bar, val in zip(bars, [global_sim_cold, global_sim_warm]):
        ax15.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                  f"{val:.3f}", ha="center", fontsize=9)

    # Summary text
    ax16 = fig.add_subplot(4, 4, 16)
    ax16.axis("off")

    summary_text = f"""
    Analysis Summary
    ================

    Test Image: {test_info}
    Grid Size: {grid_size[0]} x {grid_size[1]} = {analysis['n_patches']} patches

    Global Similarity:
      To Cold Ref: {global_sim_cold:.4f}
      To Warm Ref: {global_sim_warm:.4f}
      → Prefers: {'COLD' if global_sim_cold > global_sim_warm else 'WARM'}

    Patch-level Statistics:
      Mean Sim to Cold: {analysis['max_sim_cold'].mean():.4f}
      Mean Sim to Warm: {analysis['max_sim_warm'].mean():.4f}

      Patches preferring Cold: {analysis['prefers_cold'].sum()} ({cold_pct:.1f}%)
      Patches preferring Warm: {analysis['n_patches'] - analysis['prefers_cold'].sum()} ({100-cold_pct:.1f}%)

    Anomaly Scores:
      Cold Only: {analysis['anomaly_scores_cold_only'].mean():.4f} (max: {analysis['anomaly_scores_cold_only'].max():.4f})
      Warm Only: {analysis['anomaly_scores_warm_only'].mean():.4f} (max: {analysis['anomaly_scores_warm_only'].max():.4f})
      Combined:  {analysis['anomaly_scores'].mean():.4f} (max: {analysis['anomaly_scores'].max():.4f})
    """
    ax16.text(0.05, 0.95, summary_text, transform=ax16.transAxes, fontsize=9,
              verticalalignment="top", fontfamily="monospace",
              bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle(f"WinCLIP Patch Matching Analysis: {test_info}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze WinCLIP patch matching")
    parser.add_argument(
        "--domain",
        type=str,
        default="domain_C",
        help="Domain to analyze",
    )
    parser.add_argument(
        "--test-index",
        type=int,
        default=9,
        help="Test image index (fault)",
    )
    parser.add_argument(
        "--test-label",
        type=str,
        default="fault",
        choices=["good", "fault"],
        help="Test image label",
    )
    parser.add_argument(
        "--cold-ref-index",
        type=int,
        default=0,
        help="Cold reference index",
    )
    parser.add_argument(
        "--warm-ref-index",
        type=int,
        default=999,
        help="Warm reference index",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory",
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load images
    print(f"\nLoading images from {args.domain}...")
    cold_ref, cold_path = load_image(args.domain, "test", "good", args.cold_ref_index)
    warm_ref, warm_path = load_image(args.domain, "test", "good", args.warm_ref_index)
    test_image, test_path = load_image(args.domain, "test", args.test_label, args.test_index)

    print(f"  Cold reference: {cold_path}")
    print(f"  Warm reference: {warm_path}")
    print(f"  Test image: {test_path}")

    # Create model
    print("\nCreating WinCLIP model...")
    model = WinClipModel(class_name="industrial sensor data")
    model = model.to(device)
    model.eval()

    # Analyze
    print("\nAnalyzing patch matching...")
    analysis = analyze_patch_matching(test_image, cold_ref, warm_ref, model, device)

    # Print summary
    print("\n" + "=" * 60)
    print("Analysis Summary")
    print("=" * 60)
    print(f"Grid size: {analysis['grid_size']}")
    print(f"Total patches: {analysis['n_patches']}")
    print(f"\nMean similarity to Cold ref: {analysis['max_sim_cold'].mean():.4f}")
    print(f"Mean similarity to Warm ref: {analysis['max_sim_warm'].mean():.4f}")
    print(f"\nPatches preferring Cold: {analysis['prefers_cold'].sum()} ({analysis['prefers_cold'].mean()*100:.1f}%)")
    print(f"Patches preferring Warm: {(~analysis['prefers_cold']).sum()} ({(1-analysis['prefers_cold'].mean())*100:.1f}%)")
    print(f"\nAnomaly score (Cold only): {analysis['anomaly_scores_cold_only'].mean():.4f}")
    print(f"Anomaly score (Warm only): {analysis['anomaly_scores_warm_only'].mean():.4f}")
    print(f"Anomaly score (Combined): {analysis['anomaly_scores'].mean():.4f}")
    print("=" * 60)

    # Visualize
    test_info = f"{args.test_label}/{args.test_index:06d}.tiff"
    output_path = args.output_dir / args.domain / f"patch_matching_{args.test_label}_{args.test_index:06d}.png"

    visualize_analysis(test_image, cold_ref, warm_ref, analysis, test_info, output_path)


if __name__ == "__main__":
    main()
