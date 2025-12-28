#!/usr/bin/env python3
"""EDA: Resize Method Comparison for PatchCore on HDMAP.

Compares different resize methods for HDMAP dataset (31x95 → 518x518):
- resize (nearest neighbor, aspect ratio ignored) - baseline
- resize_bilinear (smooth interpolation, aspect ratio ignored)
- resize_aspect_padding (aspect ratio preserved + padding)

Evaluates:
1. Visual quality: How well defect patterns are preserved
2. DINO feature quality: Fault/Good separation in feature space
3. Simple kNN classification: Within-condition discrimination

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python examples/notebooks/10_patchcore_variant/001_eda_resize_methods/eda_resize_methods.py --domain domain_C
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

# Anomalib imports
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
TARGET_SIZE = (518, 518)  # DINOv2 ViT-S/14 optimal size

# Resize methods to compare
RESIZE_METHODS = ["resize", "resize_bilinear", "resize_aspect_padding"]


class DINOFeatureExtractor:
    """DINO feature extractor wrapper."""

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

        self.pooler = torch.nn.AvgPool2d(3, 1, 1)

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract DINO features from images.

        Args:
            images: (B, C, H, W) tensor

        Returns:
            features: (B, D) global features (average pooled)
        """
        images = images.to(self.device)

        # Normalize with ImageNet stats (required for DINO)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        images_norm = (images - mean) / std

        # Extract features
        features_dict = self.feature_extractor(images_norm)
        features = features_dict[self.layer]  # (B, D, H, W)

        # Apply pooling and global average
        features = self.pooler(features)  # (B, D, H, W)
        features = features.mean(dim=[2, 3])  # (B, D) - global average pooling

        return features

    @torch.no_grad()
    def extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract patch-level DINO features.

        Args:
            images: (B, C, H, W) tensor

        Returns:
            features: (B, N, D) patch features where N = H' * W'
        """
        images = images.to(self.device)

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        images_norm = (images - mean) / std

        # Extract features
        features_dict = self.feature_extractor(images_norm)
        features = features_dict[self.layer]  # (B, D, H, W)
        features = self.pooler(features)

        B, D, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(B, H * W, D)  # (B, N, D)

        return features


def load_dataset_samples(
    domain: str,
    resize_method: str,
    split: str = "test",
    max_samples: int = None,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, List[str]]:
    """Load samples from HDMAP dataset with balanced sampling.

    The HDMAP dataset is ordered as: fault (1000) → good (1000).
    This function samples evenly from both labels and conditions.

    Returns:
        images: (N, C, H, W) tensor
        labels: (N,) array - 0: good, 1: fault
        conditions: (N,) array - 0: cold, 1: warm
        paths: list of image paths
    """
    dataset = HDMAPDataset(
        root=str(DATASET_ROOT),
        domain=domain,
        split=split,
        target_size=TARGET_SIZE,
        resize_method=resize_method,
    )

    # Dataset structure: fault (0-999) → good (1000-1999)
    # File indices: 000000-000999 for each label
    # Cold: file index 0-499, Warm: file index 500-999

    total_samples = len(dataset)  # 2000
    fault_start, fault_end = 0, 1000
    good_start, good_end = 1000, 2000

    # Determine samples per group
    if max_samples is None:
        samples_per_group = 250  # cold_fault, cold_good, warm_fault, warm_good
    else:
        samples_per_group = max_samples // 4

    # Sample indices for each group
    # fault: dataset idx 0-999, file idx 0-999 (cold: 0-499, warm: 500-999)
    # good: dataset idx 1000-1999, file idx 0-999 (cold: 0-499, warm: 500-999)

    indices_to_load = []

    # Cold fault: dataset idx 0-499
    cold_fault_indices = list(range(0, min(500, samples_per_group)))[:samples_per_group]
    indices_to_load.extend(cold_fault_indices)

    # Warm fault: dataset idx 500-999
    warm_fault_indices = list(range(500, min(1000, 500 + samples_per_group)))[:samples_per_group]
    indices_to_load.extend(warm_fault_indices)

    # Cold good: dataset idx 1000-1499
    cold_good_indices = list(range(1000, min(1500, 1000 + samples_per_group)))[:samples_per_group]
    indices_to_load.extend(cold_good_indices)

    # Warm good: dataset idx 1500-1999
    warm_good_indices = list(range(1500, min(2000, 1500 + samples_per_group)))[:samples_per_group]
    indices_to_load.extend(warm_good_indices)

    images = []
    labels = []
    conditions = []
    paths = []

    for i in tqdm(indices_to_load, desc=f"Loading {resize_method}"):
        item = dataset[i]
        images.append(item.image)
        labels.append(item.gt_label.item() if hasattr(item.gt_label, 'item') else int(item.gt_label))
        paths.append(item.image_path)

        # Determine condition from file index (0-499: cold, 500-999: warm)
        file_idx = int(Path(item.image_path).stem)
        conditions.append(0 if file_idx < 500 else 1)

    images = torch.stack(images, dim=0)
    labels = np.array(labels)
    conditions = np.array(conditions)

    return images, labels, conditions, paths


def load_single_image(domain: str, split: str, label: str, index: int, resize_method: str) -> torch.Tensor:
    """Load a single image with specified resize method."""
    dataset = HDMAPDataset(
        root=str(DATASET_ROOT),
        domain=domain,
        split=split,
        target_size=TARGET_SIZE,
        resize_method=resize_method,
    )

    target_path = f"{label}/{index:06d}.tiff"
    for item in dataset:
        if target_path in item.image_path:
            return item.image

    raise ValueError(f"Image not found: {target_path}")


def visualize_resize_comparison(domain: str, sample_indices: List[Tuple[str, int]], output_path: Path):
    """Visualize comparison of resize methods on sample images.

    Args:
        domain: Domain name
        sample_indices: List of (label, index) tuples
        output_path: Output file path
    """
    n_samples = len(sample_indices)
    n_methods = len(RESIZE_METHODS)

    fig, axes = plt.subplots(n_samples, n_methods + 1, figsize=(4 * (n_methods + 1), 4 * n_samples))

    def norm_display(t):
        """Normalize tensor for display."""
        if t.dim() == 3:
            arr = t.permute(1, 2, 0).numpy()
        else:
            arr = t.numpy()
        arr = arr[:, :, 0] if arr.ndim == 3 else arr  # Take first channel for grayscale
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    for row, (label, idx) in enumerate(sample_indices):
        # Column 0: Original image info
        ax = axes[row, 0] if n_samples > 1 else axes[0]

        # Load original (no resize) for reference - use black_padding to preserve original
        try:
            orig_dataset = HDMAPDataset(
                root=str(DATASET_ROOT),
                domain=domain,
                split="test",
                target_size=None,  # No resize
                resize_method="resize",
            )
            target_path = f"{label}/{idx:06d}.tiff"
            orig_img = None
            for item in orig_dataset:
                if target_path in item.image_path:
                    orig_img = item.image
                    break

            if orig_img is not None:
                # Show original size
                ax.imshow(norm_display(orig_img), cmap='viridis')
                ax.set_title(f"Original\n{label}/{idx:06d}\nShape: {orig_img.shape[1]}x{orig_img.shape[2]}", fontsize=10)
            else:
                ax.text(0.5, 0.5, "Original\nNot Found", ha='center', va='center')
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
        ax.axis("off")

        # Columns 1+: Each resize method
        for col, method in enumerate(RESIZE_METHODS):
            ax = axes[row, col + 1] if n_samples > 1 else axes[col + 1]

            img = load_single_image(domain, "test", label, idx, method)
            ax.imshow(norm_display(img), cmap='viridis')
            ax.set_title(f"{method}\n{TARGET_SIZE[0]}x{TARGET_SIZE[1]}", fontsize=10)
            ax.axis("off")

    plt.suptitle(f"Resize Methods Comparison - {domain}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved visual comparison to: {output_path}")


def analyze_feature_distribution(
    extractor: DINOFeatureExtractor,
    domain: str,
    resize_method: str,
    max_samples: int = 200,
) -> Dict:
    """Analyze DINO feature distribution for a resize method.

    Returns dict with features, labels, conditions, and statistics.
    """
    logger.info(f"Analyzing {resize_method} for {domain}...")

    images, labels, conditions, paths = load_dataset_samples(
        domain, resize_method, split="test", max_samples=max_samples
    )

    # Extract features in batches
    batch_size = 32
    all_features = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        features = extractor.extract_features(batch)
        all_features.append(features.cpu())

    features = torch.cat(all_features, dim=0).numpy()

    # Calculate statistics
    cold_fault_mask = (conditions == 0) & (labels == 1)
    cold_good_mask = (conditions == 0) & (labels == 0)
    warm_fault_mask = (conditions == 1) & (labels == 1)
    warm_good_mask = (conditions == 1) & (labels == 0)

    stats = {
        "resize_method": resize_method,
        "n_samples": len(features),
        "feature_dim": features.shape[1],
        "cold_fault_count": cold_fault_mask.sum(),
        "cold_good_count": cold_good_mask.sum(),
        "warm_fault_count": warm_fault_mask.sum(),
        "warm_good_count": warm_good_mask.sum(),
    }

    # Feature statistics per group
    for name, mask in [
        ("cold_fault", cold_fault_mask),
        ("cold_good", cold_good_mask),
        ("warm_fault", warm_fault_mask),
        ("warm_good", warm_good_mask),
    ]:
        if mask.sum() > 0:
            group_features = features[mask]
            stats[f"{name}_mean_norm"] = np.linalg.norm(group_features.mean(axis=0))
            stats[f"{name}_std"] = group_features.std()

    return {
        "features": features,
        "labels": labels,
        "conditions": conditions,
        "paths": paths,
        "stats": stats,
    }


def compute_within_condition_auroc(features: np.ndarray, labels: np.ndarray, conditions: np.ndarray) -> Dict:
    """Compute AUROC for within-condition classification using simple kNN."""

    results = {}

    for cond_name, cond_val in [("cold", 0), ("warm", 1)]:
        mask = conditions == cond_val
        cond_features = features[mask]
        cond_labels = labels[mask]

        if len(np.unique(cond_labels)) < 2:
            results[f"{cond_name}_auroc"] = None
            continue

        # Use centroid distance as anomaly score
        good_mask = cond_labels == 0
        if good_mask.sum() == 0:
            results[f"{cond_name}_auroc"] = None
            continue

        good_centroid = cond_features[good_mask].mean(axis=0)
        distances = np.linalg.norm(cond_features - good_centroid, axis=1)

        auroc = roc_auc_score(cond_labels, distances)
        results[f"{cond_name}_auroc"] = auroc
        results[f"{cond_name}_n_samples"] = len(cond_labels)

    # Overall AUROC
    good_mask = labels == 0
    if good_mask.sum() > 0 and len(np.unique(labels)) >= 2:
        good_centroid = features[good_mask].mean(axis=0)
        distances = np.linalg.norm(features - good_centroid, axis=1)
        results["overall_auroc"] = roc_auc_score(labels, distances)

    return results


def visualize_tsne(results_list: List[Dict], domain: str, output_path: Path):
    """Visualize t-SNE for each resize method."""

    n_methods = len(results_list)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))

    for idx, results in enumerate(results_list):
        ax = axes[idx] if n_methods > 1 else axes

        features = results["features"]
        labels = results["labels"]
        conditions = results["conditions"]
        method = results["stats"]["resize_method"]
        auroc_results = results.get("auroc_results", {})

        # t-SNE
        logger.info(f"Computing t-SNE for {method}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
        features_2d = tsne.fit_transform(features)

        # Plot with different markers for condition and color for label
        # Cold Good: blue circle, Cold Fault: red circle
        # Warm Good: blue triangle, Warm Fault: red triangle

        for cond, cond_name, marker in [(0, "Cold", "o"), (1, "Warm", "^")]:
            for label, label_name, color in [(0, "Good", "tab:blue"), (1, "Fault", "tab:red")]:
                mask = (conditions == cond) & (labels == label)
                if mask.sum() > 0:
                    ax.scatter(
                        features_2d[mask, 0],
                        features_2d[mask, 1],
                        c=color,
                        marker=marker,
                        alpha=0.6,
                        s=30,
                        label=f"{cond_name} {label_name}",
                    )

        # Title with AUROC
        cold_auroc = auroc_results.get("cold_auroc", None)
        warm_auroc = auroc_results.get("warm_auroc", None)
        overall_auroc = auroc_results.get("overall_auroc", None)

        title = f"{method}\n"
        if cold_auroc is not None:
            title += f"Cold AUROC: {cold_auroc:.3f} | "
        if warm_auroc is not None:
            title += f"Warm AUROC: {warm_auroc:.3f}\n"
        if overall_auroc is not None:
            title += f"Overall AUROC: {overall_auroc:.3f}"

        ax.set_title(title, fontsize=11)
        ax.legend(loc="best", fontsize=8)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    plt.suptitle(f"DINO Feature Distribution - {domain}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved t-SNE visualization to: {output_path}")


def print_summary(results_list: List[Dict], domain: str):
    """Print summary comparison table."""

    print("\n" + "=" * 100)
    print(f"SUMMARY: Resize Method Comparison for {domain}")
    print("=" * 100)
    print(f"{'Method':<25} {'Cold AUROC':<15} {'Warm AUROC':<15} {'Overall AUROC':<15} {'Feature Dim'}")
    print("-" * 100)

    for results in results_list:
        method = results["stats"]["resize_method"]
        auroc = results.get("auroc_results", {})

        cold_auroc = auroc.get("cold_auroc", None)
        warm_auroc = auroc.get("warm_auroc", None)
        overall_auroc = auroc.get("overall_auroc", None)
        feature_dim = results["stats"]["feature_dim"]

        cold_str = f"{cold_auroc:.4f}" if cold_auroc else "N/A"
        warm_str = f"{warm_auroc:.4f}" if warm_auroc else "N/A"
        overall_str = f"{overall_auroc:.4f}" if overall_auroc else "N/A"

        print(f"{method:<25} {cold_str:<15} {warm_str:<15} {overall_str:<15} {feature_dim}")

    print("=" * 100)

    # Recommendation
    print("\n[RECOMMENDATION]")
    best_method = None
    best_cold_auroc = 0

    for results in results_list:
        auroc = results.get("auroc_results", {})
        cold_auroc = auroc.get("cold_auroc", 0) or 0
        if cold_auroc > best_cold_auroc:
            best_cold_auroc = cold_auroc
            best_method = results["stats"]["resize_method"]

    if best_method:
        print(f"Best resize method for Cold condition: {best_method} (AUROC: {best_cold_auroc:.4f})")
    print("Note: Cold condition is more important as Domain C's cold fault is the hardest case.")


def main():
    parser = argparse.ArgumentParser(description="EDA: Resize Method Comparison for PatchCore")
    parser.add_argument("--domain", default="domain_C", help="Domain to analyze")
    parser.add_argument("--max-samples", type=int, default=200, help="Max samples per method")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--skip-visual", action="store_true", help="Skip visual comparison")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Domain: {args.domain}")
    logger.info(f"DINO backbone: {DINO_BACKBONE}, layer: {DINO_LAYER}")
    logger.info(f"Target size: {TARGET_SIZE}")

    output_dir = OUTPUT_DIR / args.domain
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Visual comparison
    if not args.skip_visual:
        logger.info("\n" + "=" * 60)
        logger.info("Step 1: Visual Comparison of Resize Methods")
        logger.info("=" * 60)

        sample_indices = [
            ("fault", 9),   # Cold fault sample
            ("good", 9),    # Cold good sample
            ("fault", 509), # Warm fault sample
            ("good", 509),  # Warm good sample
        ]

        visualize_resize_comparison(
            args.domain,
            sample_indices,
            output_dir / "visual_comparison.png"
        )

    # Step 2: DINO feature analysis
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: DINO Feature Distribution Analysis")
    logger.info("=" * 60)

    extractor = DINOFeatureExtractor(device=device)

    results_list = []
    for method in RESIZE_METHODS:
        results = analyze_feature_distribution(
            extractor, args.domain, method, max_samples=args.max_samples
        )

        # Compute AUROC
        auroc_results = compute_within_condition_auroc(
            results["features"], results["labels"], results["conditions"]
        )
        results["auroc_results"] = auroc_results

        results_list.append(results)

        # Log stats
        stats = results["stats"]
        logger.info(f"\n{method}:")
        logger.info(f"  Samples: {stats['n_samples']}, Feature dim: {stats['feature_dim']}")
        logger.info(f"  Cold: {stats['cold_fault_count']} fault, {stats['cold_good_count']} good")
        logger.info(f"  Warm: {stats['warm_fault_count']} fault, {stats['warm_good_count']} good")
        logger.info(f"  Cold AUROC: {auroc_results.get('cold_auroc', 'N/A')}")
        logger.info(f"  Warm AUROC: {auroc_results.get('warm_auroc', 'N/A')}")
        logger.info(f"  Overall AUROC: {auroc_results.get('overall_auroc', 'N/A')}")

    # Step 3: t-SNE visualization
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: t-SNE Visualization")
    logger.info("=" * 60)

    visualize_tsne(results_list, args.domain, output_dir / "tsne_comparison.png")

    # Step 4: Summary
    print_summary(results_list, args.domain)

    logger.info(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
