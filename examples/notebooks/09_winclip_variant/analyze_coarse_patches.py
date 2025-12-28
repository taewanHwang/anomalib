#!/usr/bin/env python3
"""Coarse Patch Analysis using ViT-B-32.

Compares patch-level analysis between:
- ViT-B-16-plus-240: 15x15 = 225 patches (fine-grained)
- ViT-B-32: 7x7 = 49 patches (coarse)

For HDMAP data (original 31x95), coarser patches might be more appropriate.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn.functional as F

from anomalib.data.datasets.image.hdmap import HDMAPDataset


ANOMALIB_ROOT = Path(__file__).parent.parent.parent.parent
DATASET_ROOT = ANOMALIB_ROOT / "datasets" / "HDMAP" / "1000_tiff_minmax"
OUTPUT_DIR = Path(__file__).parent / "patch_analysis"


def load_image(domain: str, split: str, label: str, index: int, target_size: int):
    """Load image from HDMAPDataset with specific size."""
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


def extract_patch_embeddings(model, preprocess, images: torch.Tensor, device):
    """Extract patch embeddings from CLIP model.

    Args:
        model: CLIP model
        preprocess: Preprocessing transform (not used, images already preprocessed)
        images: Tensor of shape (B, 3, H, W)
        device: torch device

    Returns:
        patch_embeddings: (B, num_patches, dim)
    """
    images = images.to(device)

    # Get the visual encoder
    visual = model.visual

    # Forward through patch embedding
    x = visual.conv1(images)  # (B, width, grid_h, grid_w)
    grid_h, grid_w = x.shape[2], x.shape[3]

    x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, width, grid_h*grid_w)
    x = x.permute(0, 2, 1)  # (B, grid_h*grid_w, width)

    # Add class token and positional embedding
    x = torch.cat([
        visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
        x
    ], dim=1)

    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    # Transformer
    x = x.permute(1, 0, 2)  # (seq_len, B, width)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)  # (B, seq_len, width)

    # Extract patch tokens (exclude class token)
    patch_tokens = x[:, 1:, :]  # (B, num_patches, width)

    # Normalize
    patch_tokens = F.normalize(patch_tokens, dim=-1)

    return patch_tokens, (grid_h, grid_w)


def cosine_similarity_matrix(emb1, emb2):
    """Compute cosine similarity between two sets of embeddings."""
    # emb1: (n1, d), emb2: (n2, d)
    emb1_norm = F.normalize(emb1, dim=-1)
    emb2_norm = F.normalize(emb2, dim=-1)
    return torch.mm(emb1_norm, emb2_norm.t())


def analyze_with_model(
    model_name: str,
    pretrained: str,
    image_size: int,
    test_image_path: str,
    cold_ref_path: str,
    warm_ref_path: str,
    domain: str,
    test_label: str,
    test_index: int,
    device: torch.device,
):
    """Analyze patch matching with a specific CLIP model."""

    print(f"\n{'='*60}")
    print(f"Model: {model_name} ({pretrained})")
    print(f"Image size: {image_size}x{image_size}")
    print(f"{'='*60}")

    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device)
    model.eval()

    # Load images
    test_img, _ = load_image(domain, "test", test_label, test_index, image_size)
    cold_ref, _ = load_image(domain, "test", "good", 0, image_size)
    warm_ref, _ = load_image(domain, "test", "good", 999, image_size)

    # Stack for batch processing
    images = torch.stack([test_img, cold_ref, warm_ref])

    # Extract patch embeddings
    with torch.no_grad():
        patch_emb, grid_size = extract_patch_embeddings(model, preprocess, images, device)

    print(f"Grid size: {grid_size[0]}x{grid_size[1]} = {grid_size[0]*grid_size[1]} patches")

    # Original image size mapping
    orig_h, orig_w = 31, 95
    patch_h_orig = orig_h / grid_size[0]
    patch_w_orig = orig_w / grid_size[1]
    print(f"Each patch covers (original): {patch_h_orig:.1f} x {patch_w_orig:.1f} pixels")

    # Compute similarities
    test_emb = patch_emb[0]  # (num_patches, dim)
    cold_emb = patch_emb[1]
    warm_emb = patch_emb[2]

    sim_to_cold = cosine_similarity_matrix(test_emb, cold_emb)  # (P, P)
    sim_to_warm = cosine_similarity_matrix(test_emb, warm_emb)

    max_sim_cold = sim_to_cold.max(dim=1)[0].cpu().numpy()
    max_sim_warm = sim_to_warm.max(dim=1)[0].cpu().numpy()

    # Combined (max of both)
    max_sim_combined = np.maximum(max_sim_cold, max_sim_warm)
    anomaly_scores = (1 - max_sim_combined) / 2

    print(f"\nSimilarity Statistics:")
    print(f"  Cold Ref - mean: {max_sim_cold.mean():.4f}, min: {max_sim_cold.min():.4f}, max: {max_sim_cold.max():.4f}")
    print(f"  Warm Ref - mean: {max_sim_warm.mean():.4f}, min: {max_sim_warm.min():.4f}, max: {max_sim_warm.max():.4f}")
    print(f"  Anomaly  - mean: {anomaly_scores.mean():.4f}, min: {anomaly_scores.min():.4f}, max: {anomaly_scores.max():.4f}")

    return {
        "model_name": model_name,
        "grid_size": grid_size,
        "patch_orig_size": (patch_h_orig, patch_w_orig),
        "max_sim_cold": max_sim_cold.reshape(grid_size),
        "max_sim_warm": max_sim_warm.reshape(grid_size),
        "anomaly_scores": anomaly_scores.reshape(grid_size),
        "test_img": test_img,
        "cold_ref": cold_ref,
        "warm_ref": warm_ref,
    }


def visualize_comparison(results_fine, results_coarse, output_path):
    """Compare fine vs coarse patch analysis."""

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    def norm_display(t):
        arr = t.permute(1, 2, 0).numpy()
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    # Row 1: Images
    ax = axes[0, 0]
    ax.imshow(norm_display(results_fine["test_img"]))
    ax.set_title("Test Image (fault/000009)")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(norm_display(results_fine["cold_ref"]))
    ax.set_title("Cold Ref (good/000000)")
    ax.axis("off")

    ax = axes[0, 2]
    ax.imshow(norm_display(results_fine["warm_ref"]))
    ax.set_title("Warm Ref (good/000999)")
    ax.axis("off")

    ax = axes[0, 3]
    ax.text(0.5, 0.7, "Original: 31 x 95 pixels", ha='center', fontsize=12)
    ax.text(0.5, 0.5, f"Fine: {results_fine['grid_size'][0]}x{results_fine['grid_size'][1]} patches", ha='center', fontsize=12)
    ax.text(0.5, 0.4, f"  → {results_fine['patch_orig_size'][0]:.1f}x{results_fine['patch_orig_size'][1]:.1f} orig pixels/patch", ha='center', fontsize=10)
    ax.text(0.5, 0.25, f"Coarse: {results_coarse['grid_size'][0]}x{results_coarse['grid_size'][1]} patches", ha='center', fontsize=12)
    ax.text(0.5, 0.15, f"  → {results_coarse['patch_orig_size'][0]:.1f}x{results_coarse['patch_orig_size'][1]:.1f} orig pixels/patch", ha='center', fontsize=10)
    ax.axis("off")
    ax.set_title("Patch Resolution")

    # Row 2: Fine-grained (ViT-B-16)
    ax = axes[1, 0]
    im = ax.imshow(results_fine["max_sim_cold"], cmap="Blues", vmin=0.5, vmax=1.0)
    ax.set_title(f"Fine: Sim to Cold\n(mean={results_fine['max_sim_cold'].mean():.3f})")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    im = ax.imshow(results_fine["max_sim_warm"], cmap="Reds", vmin=0.5, vmax=1.0)
    ax.set_title(f"Fine: Sim to Warm\n(mean={results_fine['max_sim_warm'].mean():.3f})")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 2]
    im = ax.imshow(results_fine["anomaly_scores"], cmap="hot", vmin=0, vmax=0.3)
    ax.set_title(f"Fine: Anomaly Score\n(mean={results_fine['anomaly_scores'].mean():.3f})")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 3]
    ax.hist(results_fine["anomaly_scores"].flatten(), bins=20, alpha=0.7, label="Fine", color="blue")
    ax.axvline(results_fine["anomaly_scores"].mean(), color="blue", linestyle="--")
    ax.set_title("Fine: Anomaly Distribution")
    ax.set_xlabel("Anomaly Score")
    ax.legend()

    # Row 3: Coarse (ViT-B-32)
    ax = axes[2, 0]
    im = ax.imshow(results_coarse["max_sim_cold"], cmap="Blues", vmin=0.5, vmax=1.0)
    ax.set_title(f"Coarse: Sim to Cold\n(mean={results_coarse['max_sim_cold'].mean():.3f})")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2, 1]
    im = ax.imshow(results_coarse["max_sim_warm"], cmap="Reds", vmin=0.5, vmax=1.0)
    ax.set_title(f"Coarse: Sim to Warm\n(mean={results_coarse['max_sim_warm'].mean():.3f})")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2, 2]
    im = ax.imshow(results_coarse["anomaly_scores"], cmap="hot", vmin=0, vmax=0.3)
    ax.set_title(f"Coarse: Anomaly Score\n(mean={results_coarse['anomaly_scores'].mean():.3f})")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2, 3]
    ax.hist(results_coarse["anomaly_scores"].flatten(), bins=20, alpha=0.7, label="Coarse", color="red")
    ax.axvline(results_coarse["anomaly_scores"].mean(), color="red", linestyle="--")
    ax.set_title("Coarse: Anomaly Distribution")
    ax.set_xlabel("Anomaly Score")
    ax.legend()

    plt.suptitle("Fine vs Coarse Patch Analysis (Cold Fault 000009)", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="domain_C")
    parser.add_argument("--test-index", type=int, default=9)
    parser.add_argument("--test-label", default="fault")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Fine-grained analysis (ViT-B-16, 240x240, 15x15 patches)
    results_fine = analyze_with_model(
        model_name="ViT-B-16-plus-240",
        pretrained="laion400m_e31",
        image_size=240,
        test_image_path=None,
        cold_ref_path=None,
        warm_ref_path=None,
        domain=args.domain,
        test_label=args.test_label,
        test_index=args.test_index,
        device=device,
    )

    # Coarse analysis (ViT-B-32, 224x224, 7x7 patches)
    results_coarse = analyze_with_model(
        model_name="ViT-B-32",
        pretrained="laion400m_e31",
        image_size=224,
        test_image_path=None,
        cold_ref_path=None,
        warm_ref_path=None,
        domain=args.domain,
        test_label=args.test_label,
        test_index=args.test_index,
        device=device,
    )

    # Visualize comparison
    output_path = OUTPUT_DIR / args.domain / f"fine_vs_coarse_{args.test_label}_{args.test_index:06d}.png"
    visualize_comparison(results_fine, results_coarse, output_path)


if __name__ == "__main__":
    main()
