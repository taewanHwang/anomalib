#!/usr/bin/env python3
"""Coarse Patch Mapping Analysis using ViT-B-32 (7x7 patches).

Similar to analyze_patch_mapping_detail.py but using coarser patches.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import open_clip
import torch
import torch.nn.functional as F

from anomalib.data.datasets.image.hdmap import HDMAPDataset


ANOMALIB_ROOT = Path(__file__).parent.parent.parent.parent
DATASET_ROOT = ANOMALIB_ROOT / "datasets" / "HDMAP" / "1000_tiff_minmax"
OUTPUT_DIR = Path(__file__).parent / "patch_analysis"


def load_image(domain: str, split: str, label: str, index: int, target_size: int):
    """Load image from HDMAPDataset."""
    dataset = HDMAPDataset(
        root=str(DATASET_ROOT),
        domain=domain,
        split=split,
        target_size=(target_size, target_size),
        resize_method="resize",
    )

    target_path = f"{label}/{index:06d}.tiff"
    for item in dataset:
        if target_path in item.image_path:
            return item.image, item.image_path

    raise ValueError(f"Image not found: {target_path}")


def extract_patch_embeddings(model, images: torch.Tensor, device):
    """Extract patch embeddings from CLIP ViT-B-32 model."""
    images = images.to(device)
    visual = model.visual

    x = visual.conv1(images)
    grid_h, grid_w = x.shape[2], x.shape[3]

    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)

    x = torch.cat([
        visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
        x
    ], dim=1)

    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)

    patch_tokens = x[:, 1:, :]
    patch_tokens = F.normalize(patch_tokens, dim=-1)

    return patch_tokens, (grid_h, grid_w)


def analyze_patch_mapping(test_emb, cold_emb, warm_emb, grid_size):
    """Analyze patch-to-patch mapping."""

    # Full similarity matrices
    sim_to_cold = torch.mm(test_emb, cold_emb.t()).cpu().numpy()
    sim_to_warm = torch.mm(test_emb, warm_emb.t()).cpu().numpy()

    best_cold_idx = sim_to_cold.argmax(axis=1)
    best_warm_idx = sim_to_warm.argmax(axis=1)
    max_sim_cold = sim_to_cold.max(axis=1)
    max_sim_warm = sim_to_warm.max(axis=1)

    def idx_to_rc(idx):
        return idx // grid_size[1], idx % grid_size[1]

    best_cold_rc = [idx_to_rc(i) for i in best_cold_idx]
    best_warm_rc = [idx_to_rc(i) for i in best_warm_idx]

    return {
        "grid_size": grid_size,
        "sim_to_cold": sim_to_cold,
        "sim_to_warm": sim_to_warm,
        "best_cold_idx": best_cold_idx,
        "best_warm_idx": best_warm_idx,
        "best_cold_rc": best_cold_rc,
        "best_warm_rc": best_warm_rc,
        "max_sim_cold": max_sim_cold,
        "max_sim_warm": max_sim_warm,
    }


def visualize_defect_mapping(
    test_image, cold_ref, warm_ref,
    analysis, defect_patches, output_path,
    image_size
):
    """Visualize defect patch mapping for coarse model."""

    grid_size = analysis["grid_size"]

    def norm_display(t):
        arr = t.permute(1, 2, 0).numpy()
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    test_np = norm_display(test_image)
    cold_np = norm_display(cold_ref)
    warm_np = norm_display(warm_ref)

    patch_h = image_size // grid_size[0]
    patch_w = image_size // grid_size[1]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Images with defect patches
    ax = axes[0, 0]
    ax.imshow(test_np)
    ax.set_title(f"Test Image (Defect patches)\nGrid: {grid_size[0]}x{grid_size[1]}", fontsize=12)
    for (r, c) in defect_patches:
        rect = mpatches.Rectangle(
            (c * patch_w, r * patch_h), patch_w, patch_h,
            linewidth=3, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
    ax.axis("off")

    # Cold ref with matched patches
    ax = axes[0, 1]
    ax.imshow(cold_np)
    ax.set_title("Cold Ref: Where defect patches match", fontsize=12)

    for (r, c) in defect_patches:
        test_idx = r * grid_size[1] + c
        match_r, match_c = analysis["best_cold_rc"][test_idx]
        sim = analysis["max_sim_cold"][test_idx]

        rect = mpatches.Rectangle(
            (match_c * patch_w, match_r * patch_h), patch_w, patch_h,
            linewidth=3, edgecolor='yellow', facecolor='yellow', alpha=0.4
        )
        ax.add_patch(rect)
        ax.text(match_c * patch_w + patch_w/2, match_r * patch_h + patch_h/2,
                f"{sim:.2f}", ha='center', va='center', fontsize=10, color='black',
                fontweight='bold')
    ax.axis("off")

    # Warm ref with matched patches
    ax = axes[0, 2]
    ax.imshow(warm_np)
    ax.set_title("Warm Ref: Where defect patches match", fontsize=12)

    for (r, c) in defect_patches:
        test_idx = r * grid_size[1] + c
        match_r, match_c = analysis["best_warm_rc"][test_idx]
        sim = analysis["max_sim_warm"][test_idx]

        rect = mpatches.Rectangle(
            (match_c * patch_w, match_r * patch_h), patch_w, patch_h,
            linewidth=3, edgecolor='yellow', facecolor='yellow', alpha=0.4
        )
        ax.add_patch(rect)
        ax.text(match_c * patch_w + patch_w/2, match_r * patch_h + patch_h/2,
                f"{sim:.2f}", ha='center', va='center', fontsize=10, color='black',
                fontweight='bold')
    ax.axis("off")

    # Row 2: Similarity and anomaly maps
    max_sim_cold_grid = analysis["max_sim_cold"].reshape(grid_size)
    max_sim_warm_grid = analysis["max_sim_warm"].reshape(grid_size)
    anomaly_grid = (1 - np.maximum(max_sim_cold_grid, max_sim_warm_grid)) / 2

    ax = axes[1, 0]
    im = ax.imshow(max_sim_cold_grid, cmap="Blues", vmin=0.5, vmax=1.0)
    ax.set_title(f"Max Sim to Cold\n(mean={max_sim_cold_grid.mean():.3f})", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046)
    for (r, c) in defect_patches:
        rect = mpatches.Rectangle((c-0.5, r-0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    ax = axes[1, 1]
    im = ax.imshow(max_sim_warm_grid, cmap="Reds", vmin=0.5, vmax=1.0)
    ax.set_title(f"Max Sim to Warm\n(mean={max_sim_warm_grid.mean():.3f})", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046)
    for (r, c) in defect_patches:
        rect = mpatches.Rectangle((c-0.5, r-0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    ax = axes[1, 2]
    ax.axis("off")

    # Summary table
    summary_lines = [
        "Coarse Patch (ViT-B-32, 7x7) Analysis",
        "=" * 45,
        f"Original image: 31 x 95 pixels",
        f"Each patch covers: {31/grid_size[0]:.1f} x {95/grid_size[1]:.1f} orig pixels",
        "",
        f"{'Patch':<8} {'Cold Sim':<10} {'Cold→':<10} {'Warm Sim':<10} {'Warm→':<10}",
        "-" * 50,
    ]

    for (r, c) in defect_patches:
        test_idx = r * grid_size[1] + c
        cold_sim = analysis["max_sim_cold"][test_idx]
        warm_sim = analysis["max_sim_warm"][test_idx]
        cold_match = analysis["best_cold_rc"][test_idx]
        warm_match = analysis["best_warm_rc"][test_idx]

        summary_lines.append(
            f"({r},{c}){'':<3} {cold_sim:.3f}{'':<5} ({cold_match[0]},{cold_match[1]}){'':<4} "
            f"{warm_sim:.3f}{'':<5} ({warm_match[0]},{warm_match[1]})"
        )

    summary_lines.extend([
        "",
        "=" * 45,
        "",
        "Observation:",
        "With coarser patches (7x7), defect regions",
        "get more diluted by surrounding normal areas,",
        "resulting in HIGHER similarity scores.",
    ])

    ax.text(0.05, 0.95, "\n".join(summary_lines), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle("Coarse Patch (ViT-B-32) Defect Mapping Analysis", fontsize=14, fontweight='bold')
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
    parser.add_argument("--defect-row", type=int, default=2, help="Defect row in 7x7 grid")
    parser.add_argument("--defect-cols", type=str, default="2,3,4", help="Defect columns in 7x7 grid")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    image_size = 224  # ViT-B-32 uses 224x224

    # Parse defect patches (for 7x7 grid)
    defect_cols = [int(c) for c in args.defect_cols.split(",")]
    defect_patches = [(args.defect_row, c) for c in defect_cols]

    print(f"Analyzing defect patches (7x7 grid): {defect_patches}")

    # Load model
    print("Loading ViT-B-32 model...")
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion400m_e31")
    model = model.to(device)
    model.eval()

    # Load images
    test_image, _ = load_image(args.domain, "test", args.test_label, args.test_index, image_size)
    cold_ref, _ = load_image(args.domain, "test", "good", 0, image_size)
    warm_ref, _ = load_image(args.domain, "test", "good", 999, image_size)

    images = torch.stack([test_image, cold_ref, warm_ref])

    # Extract embeddings
    with torch.no_grad():
        patch_emb, grid_size = extract_patch_embeddings(model, images, device)

    print(f"Grid size: {grid_size[0]}x{grid_size[1]} = {grid_size[0]*grid_size[1]} patches")

    test_emb = patch_emb[0]
    cold_emb = patch_emb[1]
    warm_emb = patch_emb[2]

    # Analyze
    analysis = analyze_patch_mapping(test_emb, cold_emb, warm_emb, grid_size)

    # Print details
    print("\n" + "="*60)
    print("Defect Patch Details (Coarse 7x7)")
    print("="*60)

    for (r, c) in defect_patches:
        test_idx = r * grid_size[1] + c
        cold_sim = analysis["max_sim_cold"][test_idx]
        warm_sim = analysis["max_sim_warm"][test_idx]
        cold_match = analysis["best_cold_rc"][test_idx]
        warm_match = analysis["best_warm_rc"][test_idx]

        print(f"\nPatch ({r}, {c}):")
        print(f"  → Cold Ref: sim={cold_sim:.4f}, matches ({cold_match[0]}, {cold_match[1]})")
        print(f"  → Warm Ref: sim={warm_sim:.4f}, matches ({warm_match[0]}, {warm_match[1]})")

    # Visualize
    output_path = OUTPUT_DIR / args.domain / f"defect_mapping_coarse_{args.test_label}_{args.test_index:06d}.png"
    visualize_defect_mapping(test_image, cold_ref, warm_ref, analysis, defect_patches, output_path, image_size)


if __name__ == "__main__":
    main()
