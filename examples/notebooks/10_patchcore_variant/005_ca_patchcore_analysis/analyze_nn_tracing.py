#!/usr/bin/env python3
"""Analysis 1: Patch-level Nearest Neighbor Tracing.

For cold fault samples, analyze which memory bank (cold/warm) provides
closer matches for fault region patches.

Hypothesis: Cold fault patches might be closer to warm normal patches
due to intensity/scale differences, causing CA-PatchCore gating to fail.

Usage:
    CUDA_VISIBLE_DEVICES=0 python analyze_nn_tracing.py \
        --domain domain_C \
        --k-per-bank 16 \
        --n-samples 10

Outputs:
    - Per-sample JSON with patch-level NN distances
    - Visualization showing fault patches and their NN matches
    - Summary statistics: % of fault patches closer to warm bank
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results" / "nn_tracing"
ANOMALIB_ROOT = SCRIPT_DIR.parent.parent.parent.parent
DATASET_ROOT = ANOMALIB_ROOT / "datasets" / "HDMAP" / "1000_tiff_minmax"

# P90 thresholds
P90_THRESHOLDS = {
    'domain_A': 0.2985,
    'domain_B': 0.3128,
    'domain_C': 0.3089,
    'domain_D': 0.2919,
}


def get_config(domain: str, k_per_bank: int) -> dict:
    """Get experiment configuration."""
    return {
        "backbone": "vit_base_patch14_dinov2",
        "layers": ["blocks.8"],
        "target_size": (518, 518),
        "resize_method": "resize_bilinear",
        "num_neighbors": 9,
        "batch_size": 4,
        "seed": 42,
        "domain": domain,
        "k_per_bank": k_per_bank,
        "p90_threshold": P90_THRESHOLDS.get(domain, 0.30),
    }


def setup_model_and_data(domain: str, config: dict):
    """Setup PatchCore model and HDMAP datamodule."""
    from anomalib.data.datamodules.image.hdmap import HDMAPDataModule
    from anomalib.data.utils import ValSplitMode
    from anomalib.models import Patchcore
    from anomalib.pre_processing import PreProcessor
    from torchvision.transforms.v2 import Compose, Normalize

    datamodule = HDMAPDataModule(
        root=str(DATASET_ROOT),
        domain=domain,
        train_batch_size=config["batch_size"],
        eval_batch_size=config["batch_size"],
        num_workers=8,
        val_split_mode=ValSplitMode.FROM_TEST,
        val_split_ratio=0.01,
        target_size=config["target_size"],
        resize_method=config["resize_method"],
        seed=config["seed"],
    )

    pre_processor = PreProcessor(
        transform=Compose([
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )

    model = Patchcore(
        backbone=config["backbone"],
        layers=config["layers"],
        pre_trained=True,
        coreset_sampling_ratio=1.0,
        num_neighbors=config["num_neighbors"],
        pre_processor=pre_processor,
    )

    return model, datamodule


def extract_features(model, images: torch.Tensor, device) -> torch.Tensor:
    """Extract patch embeddings from images."""
    model.model.eval()
    pre_processor = model.pre_processor

    with torch.no_grad():
        images = images.to(device)
        normalized = pre_processor(images)

        feature_extractor = model.model.feature_extractor
        feature_pooler = model.model.feature_pooler

        features = feature_extractor(normalized)
        features = {layer: feature_pooler(feature) for layer, feature in features.items()}

        embedding = model.model.generate_embedding(features)
        embedding = model.model.reshape_embedding(embedding)

    return embedding


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distances."""
    x_norm = x.pow(2).sum(dim=-1, keepdim=True)
    y_norm = y.pow(2).sum(dim=-1, keepdim=True)
    res = x_norm - 2 * torch.matmul(x, y.transpose(-2, -1)) + y_norm.transpose(-2, -1)
    return res.clamp_min_(0).sqrt_()


def get_reference_samples(datamodule, k_per_bank: int) -> Tuple[List, List]:
    """Get cold and warm reference sample indices."""
    datamodule.setup(stage="test")
    test_dataset = datamodule.test_data

    good_samples = []
    for i, item in enumerate(test_dataset):
        if item.gt_label == 0:
            file_idx = int(Path(item.image_path).stem)
            good_samples.append((i, file_idx))

    good_samples.sort(key=lambda x: x[1])

    cold_indices = [s[0] for s in good_samples[:k_per_bank]]
    warm_indices = [s[0] for s in good_samples[-k_per_bank:]]

    return cold_indices, warm_indices


def build_memory_banks(
    model,
    datamodule,
    cold_indices: List[int],
    warm_indices: List[int],
    device
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    """Build separate memory banks for cold and warm references.

    Returns:
        cold_bank, warm_bank, cold_file_indices, warm_file_indices
    """
    test_dataset = datamodule.test_data
    cold_embeddings = []
    warm_embeddings = []
    cold_file_indices = []
    warm_file_indices = []

    model.model.to(device)
    model.model.eval()

    print(f"Building cold memory bank from {len(cold_indices)} samples...")
    for idx in cold_indices:
        item = test_dataset[idx]
        file_idx = int(Path(item.image_path).stem)
        cold_file_indices.append(file_idx)
        image = item.image.unsqueeze(0)
        emb = extract_features(model, image, device)
        cold_embeddings.append(emb)

    cold_bank = torch.cat(cold_embeddings, dim=0)

    print(f"Building warm memory bank from {len(warm_indices)} samples...")
    for idx in warm_indices:
        item = test_dataset[idx]
        file_idx = int(Path(item.image_path).stem)
        warm_file_indices.append(file_idx)
        image = item.image.unsqueeze(0)
        emb = extract_features(model, image, device)
        warm_embeddings.append(emb)

    warm_bank = torch.cat(warm_embeddings, dim=0)

    print(f"Cold bank: {cold_bank.shape}, Warm bank: {warm_bank.shape}")
    print(f"Cold file indices: {cold_file_indices}")
    print(f"Warm file indices: {warm_file_indices}")

    return cold_bank, warm_bank, cold_file_indices, warm_file_indices


def get_cold_fault_samples(datamodule, n_samples: int) -> List[Tuple[int, int]]:
    """Get cold fault sample indices."""
    datamodule.setup(stage="test")
    test_dataset = datamodule.test_data

    cold_fault_samples = []
    for i, item in enumerate(test_dataset):
        if item.gt_label == 1:  # fault
            file_idx = int(Path(item.image_path).stem)
            if file_idx < 500:  # cold
                cold_fault_samples.append((i, file_idx))

    cold_fault_samples.sort(key=lambda x: x[1])

    # Sample evenly
    if len(cold_fault_samples) > n_samples:
        step = len(cold_fault_samples) // n_samples
        cold_fault_samples = cold_fault_samples[::step][:n_samples]

    print(f"Selected {len(cold_fault_samples)} cold fault samples: {[s[1] for s in cold_fault_samples]}")
    return cold_fault_samples


def analyze_single_sample(
    model,
    sample_item,
    cold_bank: torch.Tensor,
    warm_bank: torch.Tensor,
    device,
    n_patches_per_dim: int = 37
) -> Dict:
    """Analyze NN distances for all patches of a single sample.

    Returns:
        Analysis results with per-patch NN info
    """
    image = sample_item.image.unsqueeze(0)
    file_idx = int(Path(sample_item.image_path).stem)

    # Extract query features
    query_emb = extract_features(model, image, device).squeeze(0)  # (N_patches, D)

    # Compute distances to both banks
    dist_cold = euclidean_dist(query_emb, cold_bank)  # (N_patches, M_cold)
    dist_warm = euclidean_dist(query_emb, warm_bank)  # (N_patches, M_warm)

    # Get nearest neighbor info for each patch
    nn_cold_dist, nn_cold_idx = dist_cold.min(dim=1)  # (N_patches,)
    nn_warm_dist, nn_warm_idx = dist_warm.min(dim=1)  # (N_patches,)

    # Determine which bank is closer
    closer_to_warm = nn_warm_dist < nn_cold_dist

    # Reshape to spatial grid
    n_patches = query_emb.shape[0]
    assert n_patches == n_patches_per_dim ** 2, f"Expected {n_patches_per_dim**2} patches, got {n_patches}"

    results = {
        "file_idx": file_idx,
        "n_patches": n_patches,
        "grid_size": n_patches_per_dim,
        "patches": [],
        "summary": {
            "n_closer_to_cold": int((~closer_to_warm).sum()),
            "n_closer_to_warm": int(closer_to_warm.sum()),
            "pct_closer_to_warm": float(closer_to_warm.float().mean() * 100),
            "mean_dist_cold": float(nn_cold_dist.mean()),
            "mean_dist_warm": float(nn_warm_dist.mean()),
            "mean_dist_ratio": float((nn_cold_dist / nn_warm_dist).mean()),
        }
    }

    # Store per-patch info
    for p_idx in range(n_patches):
        i = p_idx // n_patches_per_dim
        j = p_idx % n_patches_per_dim

        results["patches"].append({
            "idx": p_idx,
            "coord": [i, j],
            "dist_cold": float(nn_cold_dist[p_idx]),
            "dist_warm": float(nn_warm_dist[p_idx]),
            "closer_to": "warm" if closer_to_warm[p_idx] else "cold",
            "dist_ratio": float(nn_cold_dist[p_idx] / nn_warm_dist[p_idx]),
        })

    return results, query_emb, nn_cold_dist, nn_warm_dist, closer_to_warm


def identify_anomalous_patches(
    nn_cold_dist: torch.Tensor,
    nn_warm_dist: torch.Tensor,
    closer_to_warm: torch.Tensor,
    top_k: int = 20,
    grid_size: int = 37
) -> List[Dict]:
    """Identify the most anomalous patches (highest distance to their selected bank).

    These are likely fault region patches.
    """
    # For CA-PatchCore with gating, the score would be from the selected bank
    # Here we analyze both scenarios

    anomalous_patches = []

    # Top patches with highest cold bank distance (potential faults when using cold bank)
    top_cold_idx = nn_cold_dist.topk(top_k).indices

    for rank, p_idx in enumerate(top_cold_idx.tolist()):
        i = p_idx // grid_size
        j = p_idx % grid_size
        anomalous_patches.append({
            "rank": rank + 1,
            "patch_idx": p_idx,
            "coord": [i, j],
            "dist_cold": float(nn_cold_dist[p_idx]),
            "dist_warm": float(nn_warm_dist[p_idx]),
            "closer_to": "warm" if closer_to_warm[p_idx] else "cold",
            "dist_ratio": float(nn_cold_dist[p_idx] / nn_warm_dist[p_idx]),
        })

    return anomalous_patches


def visualize_nn_analysis(
    sample_item,
    nn_cold_dist: torch.Tensor,
    nn_warm_dist: torch.Tensor,
    closer_to_warm: torch.Tensor,
    anomalous_patches: List[Dict],
    output_path: Path,
    grid_size: int = 37
):
    """Create visualization of NN analysis for a sample."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Get raw image
    img = sample_item.image.cpu().numpy()
    if img.shape[0] == 3:
        img = img[0]  # Take first channel (grayscale replicated)

    file_idx = int(Path(sample_item.image_path).stem)

    # 1. Original image
    ax = axes[0, 0]
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Original Image (file {file_idx})")
    ax.axis('off')

    # 2. Distance to cold bank heatmap
    ax = axes[0, 1]
    dist_cold_map = nn_cold_dist.reshape(grid_size, grid_size).cpu().numpy()
    im = ax.imshow(dist_cold_map, cmap='hot')
    ax.set_title(f"Distance to Cold Bank\n(mean={dist_cold_map.mean():.3f})")
    plt.colorbar(im, ax=ax)

    # 3. Distance to warm bank heatmap
    ax = axes[0, 2]
    dist_warm_map = nn_warm_dist.reshape(grid_size, grid_size).cpu().numpy()
    im = ax.imshow(dist_warm_map, cmap='hot')
    ax.set_title(f"Distance to Warm Bank\n(mean={dist_warm_map.mean():.3f})")
    plt.colorbar(im, ax=ax)

    # 4. Closer-to-warm map (binary)
    ax = axes[1, 0]
    closer_map = closer_to_warm.reshape(grid_size, grid_size).cpu().numpy().astype(float)
    pct_warm = closer_to_warm.float().mean() * 100
    im = ax.imshow(closer_map, cmap='RdBu', vmin=0, vmax=1)
    ax.set_title(f"Closer to Warm Bank\n(Blue=Cold, Red=Warm, {pct_warm:.1f}% to Warm)")
    plt.colorbar(im, ax=ax)

    # 5. Distance ratio map (cold/warm)
    ax = axes[1, 1]
    ratio_map = (nn_cold_dist / nn_warm_dist).reshape(grid_size, grid_size).cpu().numpy()
    im = ax.imshow(ratio_map, cmap='RdBu_r', vmin=0.5, vmax=1.5)
    ax.set_title(f"Distance Ratio (Cold/Warm)\n(>1 = closer to Warm)")
    plt.colorbar(im, ax=ax)

    # 6. Top anomalous patches on original image
    ax = axes[1, 2]
    ax.imshow(img, cmap='gray')

    # Scale factor from patch grid to image pixels
    scale = img.shape[0] / grid_size

    for patch in anomalous_patches[:10]:  # Top 10
        i, j = patch["coord"]
        color = 'red' if patch["closer_to"] == "warm" else 'blue'
        rect = mpatches.Rectangle(
            (j * scale, i * scale), scale, scale,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(j * scale + scale/2, i * scale + scale/2,
                f"{patch['rank']}", color='white', fontsize=8,
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))

    legend_elements = [
        mpatches.Patch(facecolor='red', label='Closer to Warm'),
        mpatches.Patch(facecolor='blue', label='Closer to Cold'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_title(f"Top 10 Anomalous Patches\n(Ranked by Cold Bank Distance)")
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze NN tracing for CA-PatchCore")
    parser.add_argument("--domain", type=str, required=True,
                        choices=["domain_A", "domain_B", "domain_C", "domain_D"])
    parser.add_argument("--k-per-bank", type=int, default=16,
                        help="Number of reference samples per bank")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of cold fault samples to analyze")
    args = parser.parse_args()

    # Setup
    config = get_config(args.domain, args.k_per_bank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"{args.domain}_k{args.k_per_bank}" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Analysis 1: Patch-level NN Tracing")
    print(f"Domain: {args.domain}, k_per_bank: {args.k_per_bank}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Initialize
    model, datamodule = setup_model_and_data(args.domain, config)

    # Get reference samples
    cold_indices, warm_indices = get_reference_samples(datamodule, args.k_per_bank)

    # Build memory banks
    cold_bank, warm_bank, cold_file_indices, warm_file_indices = build_memory_banks(
        model, datamodule, cold_indices, warm_indices, device
    )

    # Get cold fault samples
    cold_fault_samples = get_cold_fault_samples(datamodule, args.n_samples)

    # Analyze each sample
    all_results = []
    all_summaries = []

    test_dataset = datamodule.test_data

    for dataset_idx, file_idx in tqdm(cold_fault_samples, desc="Analyzing samples"):
        sample_item = test_dataset[dataset_idx]

        # Run analysis
        results, query_emb, nn_cold_dist, nn_warm_dist, closer_to_warm = analyze_single_sample(
            model, sample_item, cold_bank, warm_bank, device
        )

        # Identify anomalous patches
        anomalous_patches = identify_anomalous_patches(
            nn_cold_dist, nn_warm_dist, closer_to_warm, top_k=20
        )
        results["anomalous_patches"] = anomalous_patches

        # Visualize
        sample_dir = output_dir / f"sample_{file_idx:06d}"
        sample_dir.mkdir(exist_ok=True)

        visualize_nn_analysis(
            sample_item, nn_cold_dist, nn_warm_dist, closer_to_warm,
            anomalous_patches, sample_dir / "nn_analysis.png"
        )

        # Save per-sample results
        with open(sample_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        all_results.append(results)
        all_summaries.append({
            "file_idx": file_idx,
            **results["summary"],
            "top_10_closer_to_warm": sum(1 for p in anomalous_patches[:10] if p["closer_to"] == "warm"),
        })

    # Aggregate summary
    summary = {
        "domain": args.domain,
        "k_per_bank": args.k_per_bank,
        "n_samples": len(all_summaries),
        "cold_ref_files": cold_file_indices,
        "warm_ref_files": warm_file_indices,
        "per_sample": all_summaries,
        "aggregate": {
            "mean_pct_closer_to_warm": np.mean([s["pct_closer_to_warm"] for s in all_summaries]),
            "mean_dist_ratio": np.mean([s["mean_dist_ratio"] for s in all_summaries]),
            "top_10_avg_closer_to_warm": np.mean([s["top_10_closer_to_warm"] for s in all_summaries]),
        }
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Domain: {args.domain}")
    print(f"Samples analyzed: {len(all_summaries)}")
    print(f"\nAggregate Results:")
    print(f"  Mean % patches closer to WARM bank: {summary['aggregate']['mean_pct_closer_to_warm']:.1f}%")
    print(f"  Mean distance ratio (cold/warm): {summary['aggregate']['mean_dist_ratio']:.3f}")
    print(f"  Top-10 anomalous patches avg closer to WARM: {summary['aggregate']['top_10_avg_closer_to_warm']:.1f}/10")
    print(f"\nPer-sample breakdown:")
    for s in all_summaries:
        print(f"  File {s['file_idx']:06d}: {s['pct_closer_to_warm']:.1f}% to warm, "
              f"ratio={s['mean_dist_ratio']:.3f}, top-10 to warm: {s['top_10_closer_to_warm']}/10")

    if summary['aggregate']['mean_pct_closer_to_warm'] > 50:
        print(f"\n*** HYPOTHESIS SUPPORTED: Cold fault patches are closer to WARM bank! ***")
    else:
        print(f"\n*** HYPOTHESIS NOT SUPPORTED: Cold fault patches are closer to COLD bank. ***")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
