#!/usr/bin/env python3
"""Detailed Patch Mapping Analysis.

Analyzes exactly which reference patch each test patch matches with,
to understand why defect patches still have high similarity scores.

The key insight: WinCLIP computes max similarity across ALL reference patches,
not just the spatially corresponding patch. So a defect patch might match
with some OTHER patch in the reference that happens to look similar.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch

from anomalib.data.datasets.image.hdmap import HDMAPDataset
from anomalib.models.image.winclip.torch_model import WinClipModel
from anomalib.models.image.winclip.utils import cosine_similarity


ANOMALIB_ROOT = Path(__file__).parent.parent.parent.parent
DATASET_ROOT = ANOMALIB_ROOT / "datasets" / "HDMAP" / "1000_tiff_minmax"
OUTPUT_DIR = Path(__file__).parent / "patch_analysis"


def load_image_from_dataset(domain: str, split: str, label: str, index: int):
    """Load image using HDMAPDataset."""
    dataset = HDMAPDataset(
        root=str(DATASET_ROOT),
        domain=domain,
        split=split,
        target_size=(240, 240),
        resize_method="resize",
    )

    target_path = f"{label}/{index:06d}.tiff"
    for item in dataset:
        if target_path in item.image_path:
            return item.image, item.image_path

    raise ValueError(f"Image not found: {target_path}")


def analyze_patch_mapping(
    test_image: torch.Tensor,
    cold_ref: torch.Tensor,
    warm_ref: torch.Tensor,
    model: WinClipModel,
    device: torch.device,
):
    """Analyze detailed patch-to-patch mapping."""

    test_batch = test_image.unsqueeze(0).to(device)
    cold_batch = cold_ref.unsqueeze(0).to(device)
    warm_batch = warm_ref.unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, test_patch_emb = model.encode_image(test_batch)
        _, _, cold_patch_emb = model.encode_image(cold_batch)
        _, _, warm_patch_emb = model.encode_image(warm_batch)

    grid_size = model.grid_size  # (15, 15)

    # Compute full similarity matrices
    test_emb = test_patch_emb.squeeze(0)  # (225, 896)
    cold_emb = cold_patch_emb.squeeze(0)  # (225, 896)
    warm_emb = warm_patch_emb.squeeze(0)  # (225, 896)

    # Full similarity matrix: (225 test patches) x (225 ref patches)
    sim_to_cold_full = cosine_similarity(test_emb, cold_emb).squeeze(0).cpu().numpy()
    sim_to_warm_full = cosine_similarity(test_emb, warm_emb).squeeze(0).cpu().numpy()

    # For each test patch, find best matching reference patch
    best_cold_patch_idx = sim_to_cold_full.argmax(axis=1)  # (225,)
    best_warm_patch_idx = sim_to_warm_full.argmax(axis=1)  # (225,)
    max_sim_cold = sim_to_cold_full.max(axis=1)
    max_sim_warm = sim_to_warm_full.max(axis=1)

    # Convert flat indices to (row, col)
    def idx_to_rc(idx, grid_size):
        return idx // grid_size[1], idx % grid_size[1]

    best_cold_rc = [idx_to_rc(i, grid_size) for i in best_cold_patch_idx]
    best_warm_rc = [idx_to_rc(i, grid_size) for i in best_warm_patch_idx]

    return {
        "grid_size": grid_size,
        "sim_to_cold_full": sim_to_cold_full,  # (225, 225)
        "sim_to_warm_full": sim_to_warm_full,  # (225, 225)
        "best_cold_patch_idx": best_cold_patch_idx,
        "best_warm_patch_idx": best_warm_patch_idx,
        "best_cold_rc": best_cold_rc,
        "best_warm_rc": best_warm_rc,
        "max_sim_cold": max_sim_cold,
        "max_sim_warm": max_sim_warm,
    }


def visualize_defect_mapping(
    test_image: torch.Tensor,
    cold_ref: torch.Tensor,
    warm_ref: torch.Tensor,
    analysis: dict,
    defect_patches: list,  # List of (row, col) for defect patches
    output_path: Path,
):
    """Visualize where defect patches map to in reference images."""

    grid_size = analysis["grid_size"]

    # Normalize for display
    def norm_display(t):
        arr = t.permute(1, 2, 0).numpy()
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    test_np = norm_display(test_image)
    cold_np = norm_display(cold_ref)
    warm_np = norm_display(warm_ref)

    patch_h = test_np.shape[0] // grid_size[0]
    patch_w = test_np.shape[1] // grid_size[1]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Test image with defect patches highlighted
    ax = axes[0, 0]
    ax.imshow(test_np)
    ax.set_title("Test Image (Defect patches highlighted)", fontsize=12)
    for (r, c) in defect_patches:
        rect = mpatches.Rectangle(
            (c * patch_w, r * patch_h), patch_w, patch_h,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
    ax.axis("off")

    # Row 1: Cold reference with matched patches highlighted
    ax = axes[0, 1]
    ax.imshow(cold_np)
    ax.set_title("Cold Ref: Where defect patches match", fontsize=12)

    # Draw lines from defect to matched location
    for i, (r, c) in enumerate(defect_patches):
        test_idx = r * grid_size[1] + c
        match_r, match_c = analysis["best_cold_rc"][test_idx]
        sim = analysis["max_sim_cold"][test_idx]

        # Highlight matched patch
        rect = mpatches.Rectangle(
            (match_c * patch_w, match_r * patch_h), patch_w, patch_h,
            linewidth=2, edgecolor='yellow', facecolor='yellow', alpha=0.3
        )
        ax.add_patch(rect)
        ax.text(match_c * patch_w + patch_w/2, match_r * patch_h + patch_h/2,
                f"{sim:.2f}", ha='center', va='center', fontsize=7, color='black',
                fontweight='bold')
    ax.axis("off")

    # Row 1: Warm reference with matched patches highlighted
    ax = axes[0, 2]
    ax.imshow(warm_np)
    ax.set_title("Warm Ref: Where defect patches match", fontsize=12)

    for i, (r, c) in enumerate(defect_patches):
        test_idx = r * grid_size[1] + c
        match_r, match_c = analysis["best_warm_rc"][test_idx]
        sim = analysis["max_sim_warm"][test_idx]

        rect = mpatches.Rectangle(
            (match_c * patch_w, match_r * patch_h), patch_w, patch_h,
            linewidth=2, edgecolor='yellow', facecolor='yellow', alpha=0.3
        )
        ax.add_patch(rect)
        ax.text(match_c * patch_w + patch_w/2, match_r * patch_h + patch_h/2,
                f"{sim:.2f}", ha='center', va='center', fontsize=7, color='black',
                fontweight='bold')
    ax.axis("off")

    # Row 2: Detailed mapping visualization
    # Show spatial offset: how far does each patch match from its "correct" position?

    # Offset for cold ref
    ax = axes[1, 0]
    offset_cold = np.zeros(grid_size)
    for test_r in range(grid_size[0]):
        for test_c in range(grid_size[1]):
            test_idx = test_r * grid_size[1] + test_c
            match_r, match_c = analysis["best_cold_rc"][test_idx]
            # Euclidean distance from "correct" position
            offset_cold[test_r, test_c] = np.sqrt((match_r - test_r)**2 + (match_c - test_c)**2)

    im = ax.imshow(offset_cold, cmap="hot", vmin=0, vmax=10)
    ax.set_title(f"Spatial Offset to Cold Ref\n(0=same pos, higher=different pos)", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Highlight defect patches
    for (r, c) in defect_patches:
        rect = mpatches.Rectangle(
            (c - 0.5, r - 0.5), 1, 1,
            linewidth=2, edgecolor='cyan', facecolor='none'
        )
        ax.add_patch(rect)

    # Offset for warm ref
    ax = axes[1, 1]
    offset_warm = np.zeros(grid_size)
    for test_r in range(grid_size[0]):
        for test_c in range(grid_size[1]):
            test_idx = test_r * grid_size[1] + test_c
            match_r, match_c = analysis["best_warm_rc"][test_idx]
            offset_warm[test_r, test_c] = np.sqrt((match_r - test_r)**2 + (match_c - test_c)**2)

    im = ax.imshow(offset_warm, cmap="hot", vmin=0, vmax=10)
    ax.set_title(f"Spatial Offset to Warm Ref\n(0=same pos, higher=different pos)", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)

    for (r, c) in defect_patches:
        rect = mpatches.Rectangle(
            (c - 0.5, r - 0.5), 1, 1,
            linewidth=2, edgecolor='cyan', facecolor='none'
        )
        ax.add_patch(rect)

    # Summary table for defect patches
    ax = axes[1, 2]
    ax.axis("off")

    summary_lines = ["Defect Patch Analysis\n" + "="*40 + "\n"]
    summary_lines.append(f"{'Patch':<10} {'Cold Sim':<10} {'Cold→':<12} {'Warm Sim':<10} {'Warm→':<12}")
    summary_lines.append("-" * 55)

    for (r, c) in defect_patches:
        test_idx = r * grid_size[1] + c
        cold_sim = analysis["max_sim_cold"][test_idx]
        warm_sim = analysis["max_sim_warm"][test_idx]
        cold_match = analysis["best_cold_rc"][test_idx]
        warm_match = analysis["best_warm_rc"][test_idx]

        cold_offset = np.sqrt((cold_match[0] - r)**2 + (cold_match[1] - c)**2)
        warm_offset = np.sqrt((warm_match[0] - r)**2 + (warm_match[1] - c)**2)

        summary_lines.append(
            f"({r},{c}){'':<5} {cold_sim:.3f}{'':<5} ({cold_match[0]},{cold_match[1]}) d={cold_offset:.1f}"
            f"   {warm_sim:.3f}{'':<5} ({warm_match[0]},{warm_match[1]}) d={warm_offset:.1f}"
        )

    summary_lines.append("\n" + "="*40)
    summary_lines.append("\nKey Insight:")
    summary_lines.append("Defect patches find high similarity by matching")
    summary_lines.append("to DIFFERENT spatial locations in the reference,")
    summary_lines.append("not the same (row, col) position.")
    summary_lines.append("\nThis is why localized defects don't show")
    summary_lines.append("low similarity scores!")

    ax.text(0.05, 0.95, "\n".join(summary_lines), transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle("Defect Patch → Reference Patch Mapping Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="domain_C")
    parser.add_argument("--test-index", type=int, default=9)
    parser.add_argument("--test-label", default="fault")
    parser.add_argument("--defect-row", type=int, default=3, help="Defect patch row (0-indexed)")
    parser.add_argument("--defect-cols", type=str, default="5,6,7,8,9,10", help="Defect patch columns")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Parse defect region
    defect_cols = [int(c) for c in args.defect_cols.split(",")]
    defect_patches = [(args.defect_row, c) for c in defect_cols]

    print(f"Analyzing defect patches: {defect_patches}")

    # Load images
    test_image, _ = load_image_from_dataset(args.domain, "test", args.test_label, args.test_index)
    cold_ref, _ = load_image_from_dataset(args.domain, "test", "good", 0)
    warm_ref, _ = load_image_from_dataset(args.domain, "test", "good", 999)

    # Create model
    model = WinClipModel(class_name="industrial sensor data")
    model = model.to(device)
    model.eval()

    # Analyze
    analysis = analyze_patch_mapping(test_image, cold_ref, warm_ref, model, device)

    # Print defect patch details
    print("\n" + "="*60)
    print("Defect Patch Matching Details")
    print("="*60)

    for (r, c) in defect_patches:
        test_idx = r * analysis["grid_size"][1] + c
        cold_sim = analysis["max_sim_cold"][test_idx]
        warm_sim = analysis["max_sim_warm"][test_idx]
        cold_match = analysis["best_cold_rc"][test_idx]
        warm_match = analysis["best_warm_rc"][test_idx]

        print(f"\nPatch ({r}, {c}):")
        print(f"  → Cold Ref: sim={cold_sim:.4f}, matches patch {cold_match}")
        print(f"  → Warm Ref: sim={warm_sim:.4f}, matches patch {warm_match}")
        print(f"  → Spatial offset: Cold={np.sqrt((cold_match[0]-r)**2 + (cold_match[1]-c)**2):.1f}, "
              f"Warm={np.sqrt((warm_match[0]-r)**2 + (warm_match[1]-c)**2):.1f}")

    # Visualize
    output_path = OUTPUT_DIR / args.domain / f"defect_mapping_{args.test_label}_{args.test_index:06d}.png"
    visualize_defect_mapping(test_image, cold_ref, warm_ref, analysis, defect_patches, output_path)


if __name__ == "__main__":
    main()
