#!/usr/bin/env python3
"""EDA: Patch-level DINO Feature Analysis for CA-PatchCore.

Analyzes patch-level features that PatchCore actually uses:
1. Cold Normal vs Warm Normal patch feature distribution
2. Within-condition Fault vs Good separation
3. Cross-condition feature overlap analysis
4. Memory Bank composition analysis

This helps validate the need for Condition-Aware Memory Bank.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        examples/notebooks/10_patchcore_variant/002_eda_dino_features/eda_patch_features.py \
        --domain domain_C
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from anomalib.data.datasets.image.hdmap import HDMAPDataset
from anomalib.models.components import TimmFeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
ANOMALIB_ROOT = Path(__file__).parent.parent.parent.parent.parent
DATASET_ROOT = ANOMALIB_ROOT / "datasets" / "HDMAP" / "1000_tiff_minmax"
OUTPUT_DIR = Path(__file__).parent / "results"

# DINO settings (matching exp-23)
DINO_BACKBONE = "vit_small_patch14_dinov2"
DINO_LAYER = "blocks.8"
TARGET_SIZE = (518, 518)
RESIZE_METHOD = "resize_bilinear"  # Best from 001 analysis


class PatchFeatureExtractor:
    """Extract patch-level DINO features like PatchCore does."""

    def __init__(self, backbone: str = DINO_BACKBONE, layer: str = DINO_LAYER, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = backbone
        self.layer = layer

        logger.info(f"Loading DINO backbone: {backbone}, layer: {layer}")
        self.feature_extractor = TimmFeatureExtractor(
            backbone=backbone,
            pre_trained=True,
            layers=[layer],
        ).eval().to(self.device)

        # PatchCore uses AvgPool2d(3, 1, 1) for smoothing
        self.pooler = torch.nn.AvgPool2d(3, 1, 1)

    @torch.no_grad()
    def extract_patch_features(self, images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Extract patch-level features.

        Args:
            images: (B, C, H, W) tensor

        Returns:
            features: (B, N, D) patch features where N = H' * W'
            spatial_size: (H', W') spatial dimensions
        """
        images = images.to(self.device)

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        images_norm = (images - mean) / std

        # Extract features
        features_dict = self.feature_extractor(images_norm)
        features = features_dict[self.layer]  # (B, D, H, W)
        features = self.pooler(features)  # Apply PatchCore's pooling

        B, D, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(B, H * W, D)  # (B, N, D)

        return features, (H, W)


def load_samples_by_condition(
    domain: str,
    condition: str,  # "cold" or "warm"
    label: str,  # "good" or "fault"
    n_samples: int = 20,
) -> Tuple[torch.Tensor, List[str]]:
    """Load samples for specific condition and label.

    Args:
        domain: Domain name
        condition: "cold" (idx 0-499) or "warm" (idx 500-999)
        label: "good" or "fault"
        n_samples: Number of samples to load

    Returns:
        images: (N, C, H, W) tensor
        paths: List of image paths
    """
    dataset = HDMAPDataset(
        root=str(DATASET_ROOT),
        domain=domain,
        split="test",
        target_size=TARGET_SIZE,
        resize_method=RESIZE_METHOD,
    )

    # Determine index range
    # Dataset order: fault (0-999) → good (1000-1999)
    # File indices: 0-499 cold, 500-999 warm
    if label == "fault":
        base_idx = 0
    else:  # good
        base_idx = 1000

    if condition == "cold":
        start_file_idx = 0
        end_file_idx = 500
    else:  # warm
        start_file_idx = 500
        end_file_idx = 1000

    # Sample evenly across the condition range
    file_indices = np.linspace(start_file_idx, end_file_idx - 1, n_samples, dtype=int)
    dataset_indices = [base_idx + idx for idx in file_indices]

    images = []
    paths = []

    for ds_idx in dataset_indices:
        item = dataset[ds_idx]
        images.append(item.image)
        paths.append(item.image_path)

    return torch.stack(images), paths


def compute_patch_statistics(
    extractor: PatchFeatureExtractor,
    domain: str,
    n_samples: int = 20,
) -> Dict:
    """Compute patch feature statistics for all conditions.

    Returns:
        Dict with patch features and statistics for each condition-label combination
    """
    results = {}

    for condition in ["cold", "warm"]:
        for label in ["good", "fault"]:
            key = f"{condition}_{label}"
            logger.info(f"Loading {key}...")

            images, paths = load_samples_by_condition(domain, condition, label, n_samples)
            features, spatial_size = extractor.extract_patch_features(images)

            # features: (N_samples, N_patches, D)
            results[key] = {
                "features": features.cpu().numpy(),
                "paths": paths,
                "n_samples": len(images),
                "n_patches_per_image": features.shape[1],
                "feature_dim": features.shape[2],
                "spatial_size": spatial_size,
            }

            # Compute statistics
            all_patches = features.cpu().numpy().reshape(-1, features.shape[-1])
            results[key]["mean"] = all_patches.mean(axis=0)
            results[key]["std"] = all_patches.std(axis=0)
            results[key]["norm_mean"] = np.linalg.norm(all_patches, axis=1).mean()

    return results


def analyze_cross_condition_overlap(results: Dict) -> Dict:
    """Analyze feature overlap between Cold and Warm conditions.

    This is key for understanding why CA-PatchCore might help:
    If Cold Normal and Warm Normal patches are very different,
    then a unified Memory Bank might cause cross-condition confusion.
    """
    analysis = {}

    # Compare Cold Good vs Warm Good (normal patches)
    cold_good = results["cold_good"]["features"].reshape(-1, results["cold_good"]["feature_dim"])
    warm_good = results["warm_good"]["features"].reshape(-1, results["warm_good"]["feature_dim"])

    # Compute centroid distance
    cold_centroid = cold_good.mean(axis=0)
    warm_centroid = warm_good.mean(axis=0)
    centroid_distance = np.linalg.norm(cold_centroid - warm_centroid)

    # Compute average within-condition distance
    cold_within_dist = cdist([cold_centroid], cold_good, 'euclidean').mean()
    warm_within_dist = cdist([warm_centroid], warm_good, 'euclidean').mean()

    # Cross-condition distance ratio
    # If centroid_distance >> within_dist, conditions are well separated
    avg_within_dist = (cold_within_dist + warm_within_dist) / 2
    separation_ratio = centroid_distance / avg_within_dist if avg_within_dist > 0 else 0

    analysis["normal_patches"] = {
        "centroid_distance": centroid_distance,
        "cold_within_dist": cold_within_dist,
        "warm_within_dist": warm_within_dist,
        "separation_ratio": separation_ratio,
        "interpretation": "High ratio (>2) = Strong separation, CA-PatchCore likely beneficial"
    }

    # Check if Cold Fault patches are closer to Warm Good than Cold Good
    cold_fault = results["cold_fault"]["features"].reshape(-1, results["cold_fault"]["feature_dim"])
    warm_fault = results["warm_fault"]["features"].reshape(-1, results["warm_fault"]["feature_dim"])

    # Sample some fault patches and compute distances
    n_sample_patches = min(500, len(cold_fault))
    sample_indices = np.random.choice(len(cold_fault), n_sample_patches, replace=False)
    cold_fault_sample = cold_fault[sample_indices]

    # Distance from Cold Fault patches to Cold Good centroid vs Warm Good centroid
    dist_to_cold_good = cdist(cold_fault_sample, [cold_centroid], 'euclidean').mean()
    dist_to_warm_good = cdist(cold_fault_sample, [warm_centroid], 'euclidean').mean()

    analysis["cold_fault_confusion"] = {
        "dist_to_cold_good_centroid": dist_to_cold_good,
        "dist_to_warm_good_centroid": dist_to_warm_good,
        "closer_to": "cold" if dist_to_cold_good < dist_to_warm_good else "warm",
        "interpretation": "If closer to warm, Cold Fault may be confused with Warm Good"
    }

    return analysis


def compute_within_condition_auroc(results: Dict) -> Dict:
    """Compute AUROC for within-condition anomaly detection.

    Simulates PatchCore's approach: use Good patches as reference,
    compute distance-based anomaly score for Fault patches.
    """
    auroc_results = {}

    for condition in ["cold", "warm"]:
        good_key = f"{condition}_good"
        fault_key = f"{condition}_fault"

        good_features = results[good_key]["features"]  # (N_good, N_patches, D)
        fault_features = results[fault_key]["features"]  # (N_fault, N_patches, D)

        # Flatten to get all patches
        good_patches = good_features.reshape(-1, good_features.shape[-1])
        fault_patches = fault_features.reshape(-1, fault_features.shape[-1])

        # Subsample for efficiency
        max_patches = 2000
        if len(good_patches) > max_patches:
            good_patches = good_patches[np.random.choice(len(good_patches), max_patches, replace=False)]

        # Compute centroid of good patches
        good_centroid = good_patches.mean(axis=0)

        # Compute distances to centroid
        good_distances = np.linalg.norm(good_patches - good_centroid, axis=1)
        fault_distances = np.linalg.norm(fault_patches - good_centroid, axis=1)

        # AUROC
        labels = np.concatenate([np.zeros(len(good_distances)), np.ones(len(fault_distances))])
        scores = np.concatenate([good_distances, fault_distances])

        auroc = roc_auc_score(labels, scores)

        auroc_results[condition] = {
            "auroc": auroc,
            "good_dist_mean": good_distances.mean(),
            "good_dist_std": good_distances.std(),
            "fault_dist_mean": fault_distances.mean(),
            "fault_dist_std": fault_distances.std(),
            "separation": fault_distances.mean() - good_distances.mean(),
        }

    return auroc_results


def visualize_patch_distribution(results: Dict, domain: str, output_path: Path):
    """Visualize patch feature distribution using t-SNE."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Collect patches from each group
    all_patches = []
    all_labels = []
    all_conditions = []

    for condition in ["cold", "warm"]:
        for label in ["good", "fault"]:
            key = f"{condition}_{label}"
            features = results[key]["features"]

            # Sample patches (to keep visualization manageable)
            n_patches = min(200, features.shape[0] * features.shape[1])
            flat_features = features.reshape(-1, features.shape[-1])

            if len(flat_features) > n_patches:
                indices = np.random.choice(len(flat_features), n_patches, replace=False)
                sampled = flat_features[indices]
            else:
                sampled = flat_features

            all_patches.append(sampled)
            all_labels.extend([1 if label == "fault" else 0] * len(sampled))
            all_conditions.extend([0 if condition == "cold" else 1] * len(sampled))

    all_patches = np.vstack(all_patches)
    all_labels = np.array(all_labels)
    all_conditions = np.array(all_conditions)

    # t-SNE
    logger.info("Computing t-SNE for patch features...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_patches) - 1))
    patches_2d = tsne.fit_transform(all_patches)

    # Plot 1: By Condition
    ax = axes[0]
    for cond, cond_name, color in [(0, "Cold", "tab:blue"), (1, "Warm", "tab:red")]:
        mask = all_conditions == cond
        ax.scatter(patches_2d[mask, 0], patches_2d[mask, 1], c=color, alpha=0.5, s=10, label=cond_name)
    ax.set_title("Patches by Condition\n(Cold vs Warm)")
    ax.legend()
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # Plot 2: By Label
    ax = axes[1]
    for label, label_name, color in [(0, "Good", "tab:green"), (1, "Fault", "tab:red")]:
        mask = all_labels == label
        ax.scatter(patches_2d[mask, 0], patches_2d[mask, 1], c=color, alpha=0.5, s=10, label=label_name)
    ax.set_title("Patches by Label\n(Good vs Fault)")
    ax.legend()
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # Plot 3: All 4 groups
    ax = axes[2]
    for cond, cond_name, marker in [(0, "Cold", "o"), (1, "Warm", "^")]:
        for label, label_name, color in [(0, "Good", "tab:blue"), (1, "Fault", "tab:red")]:
            mask = (all_conditions == cond) & (all_labels == label)
            ax.scatter(patches_2d[mask, 0], patches_2d[mask, 1],
                      c=color, marker=marker, alpha=0.5, s=15,
                      label=f"{cond_name} {label_name}")
    ax.set_title("Patches by Condition & Label")
    ax.legend(fontsize=8)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    plt.suptitle(f"Patch-level Feature Distribution - {domain}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved t-SNE to: {output_path}")


def visualize_cross_condition_analysis(analysis: Dict, auroc_results: Dict, domain: str, output_path: Path):
    """Visualize cross-condition analysis results."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Separation metrics
    ax = axes[0]
    normal = analysis["normal_patches"]

    metrics = {
        "Centroid\nDistance": normal["centroid_distance"],
        "Cold\nWithin Dist": normal["cold_within_dist"],
        "Warm\nWithin Dist": normal["warm_within_dist"],
    }

    bars = ax.bar(metrics.keys(), metrics.values(), color=['tab:purple', 'tab:blue', 'tab:red'])
    ax.set_ylabel("Distance")
    ax.set_title(f"Cold vs Warm Normal Patch Separation\nRatio: {normal['separation_ratio']:.2f}")

    # Add ratio annotation
    ax.axhline(y=normal["centroid_distance"], color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Within-condition AUROC
    ax = axes[1]

    conditions = ["cold", "warm"]
    aurocs = [auroc_results[c]["auroc"] for c in conditions]
    separations = [auroc_results[c]["separation"] for c in conditions]

    x = np.arange(len(conditions))
    width = 0.35

    bars1 = ax.bar(x - width/2, aurocs, width, label='AUROC', color='tab:blue')
    ax.set_ylabel('AUROC', color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_ylim(0.5, 1.05)

    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, separations, width, label='Separation', color='tab:orange')
    ax2.set_ylabel('Mean Distance Separation', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    ax.set_xticks(x)
    ax.set_xticklabels(['Cold\n(Fault vs Good)', 'Warm\n(Fault vs Good)'])
    ax.set_title("Within-Condition Anomaly Detection")

    # Add AUROC values as text
    for i, v in enumerate(aurocs):
        ax.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

    plt.suptitle(f"Cross-Condition Analysis - {domain}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved analysis to: {output_path}")


def print_summary(results: Dict, analysis: Dict, auroc_results: Dict, domain: str):
    """Print summary of analysis."""

    print("\n" + "=" * 80)
    print(f"PATCH-LEVEL FEATURE ANALYSIS SUMMARY - {domain}")
    print("=" * 80)

    # Basic stats
    print("\n[1] Patch Feature Statistics")
    print("-" * 40)
    for key in ["cold_good", "warm_good", "cold_fault", "warm_fault"]:
        r = results[key]
        print(f"  {key}: {r['n_samples']} images × {r['n_patches_per_image']} patches = {r['n_samples'] * r['n_patches_per_image']} total")
    print(f"  Feature dimension: {results['cold_good']['feature_dim']}")
    print(f"  Spatial size: {results['cold_good']['spatial_size']}")

    # Cross-condition analysis
    print("\n[2] Cold vs Warm Normal Patch Separation")
    print("-" * 40)
    normal = analysis["normal_patches"]
    print(f"  Centroid distance: {normal['centroid_distance']:.4f}")
    print(f"  Cold within-dist:  {normal['cold_within_dist']:.4f}")
    print(f"  Warm within-dist:  {normal['warm_within_dist']:.4f}")
    print(f"  Separation ratio:  {normal['separation_ratio']:.2f}")
    print(f"  → {normal['interpretation']}")

    # Cold fault confusion
    print("\n[3] Cold Fault Confusion Analysis")
    print("-" * 40)
    confusion = analysis["cold_fault_confusion"]
    print(f"  Cold Fault → Cold Good centroid: {confusion['dist_to_cold_good_centroid']:.4f}")
    print(f"  Cold Fault → Warm Good centroid: {confusion['dist_to_warm_good_centroid']:.4f}")
    print(f"  Closer to: {confusion['closer_to'].upper()}")
    print(f"  → {confusion['interpretation']}")

    # Within-condition AUROC
    print("\n[4] Within-Condition AUROC (Patch-level)")
    print("-" * 40)
    for cond in ["cold", "warm"]:
        r = auroc_results[cond]
        print(f"  {cond.upper()}: AUROC={r['auroc']:.4f}, Separation={r['separation']:.4f}")
        print(f"         Good dist: {r['good_dist_mean']:.4f} ± {r['good_dist_std']:.4f}")
        print(f"         Fault dist: {r['fault_dist_mean']:.4f} ± {r['fault_dist_std']:.4f}")

    # Recommendation
    print("\n[5] Recommendation for CA-PatchCore")
    print("-" * 40)
    if normal['separation_ratio'] > 1.5:
        print("  ✓ Strong Cold/Warm separation in patch features")
        print("  ✓ CA-PatchCore (separate Memory Banks) is RECOMMENDED")
    else:
        print("  △ Moderate Cold/Warm separation")
        print("  △ CA-PatchCore may help but effect might be limited")

    if confusion['closer_to'] == 'warm':
        print("  ⚠ Cold Fault patches are closer to Warm Good!")
        print("  ⚠ This explains cross-condition confusion - CA-PatchCore should help")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="EDA: Patch-level DINO Feature Analysis")
    parser.add_argument("--domain", default="domain_C", help="Domain to analyze")
    parser.add_argument("--n-samples", type=int, default=20, help="Samples per condition-label group")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Domain: {args.domain}")
    logger.info(f"Resize method: {RESIZE_METHOD}")

    output_dir = OUTPUT_DIR / args.domain
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract patch features
    extractor = PatchFeatureExtractor(device=device)
    results = compute_patch_statistics(extractor, args.domain, args.n_samples)

    # Analyze cross-condition overlap
    logger.info("Analyzing cross-condition overlap...")
    analysis = analyze_cross_condition_overlap(results)

    # Compute within-condition AUROC
    logger.info("Computing within-condition AUROC...")
    auroc_results = compute_within_condition_auroc(results)

    # Visualize
    logger.info("Creating visualizations...")
    visualize_patch_distribution(results, args.domain, output_dir / "patch_tsne.png")
    visualize_cross_condition_analysis(analysis, auroc_results, args.domain, output_dir / "cross_condition_analysis.png")

    # Print summary
    print_summary(results, analysis, auroc_results, args.domain)

    logger.info(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
