"""
Dinomaly Baseline for HDMAP Dataset (v2).

This script implements the original Dinomaly model for HDMAP dataset experiments.
Supports both multi-class unified training and single-class per-domain training.

Key features:
- --seed: Specify random seed for reproducibility
- --result-dir: Specify output directory for results
- Folder naming: {timestamp}_seed{seed}/

Usage:
    python dinomaly_baseline.py --mode multiclass --seed 42 --gpu 0 --result-dir results/dinomaly_baseline

HDMAP Dataset Structure:
    datasets/HDMAP/1000_tiff_minmax/
    ├── domain_A/
    │   ├── train/good/
    │   └── test/{good,fault}/
    ├── domain_B/
    ├── domain_C/
    └── domain_D/
"""

from __future__ import annotations

import argparse
import atexit
import errno
import json
import logging
import os
import random
import shutil
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from PIL import Image
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms

from anomalib.data import Folder
from anomalib.data.datamodules.image.all_domains_hdmap import AllDomainsHDMAPDataModule
from anomalib.data.datamodules.image.hdmap import HDMAPDataModule
from anomalib.data.datasets.image.hdmap import HDMAPDataset
from anomalib.engine import Engine
from anomalib.metrics import AUROC
from anomalib.metrics.evaluator import Evaluator
from anomalib.models.image.dinomaly import Dinomaly
from anomalib.visualization.image.item_visualizer import visualize_image_item
from anomalib.data import NumpyImageItem

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


# =============================================================================
# Constants
# =============================================================================

# Dataset paths (use environment variables for portability)
DEFAULT_HDMAP_PNG_ROOT = "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png"
DEFAULT_HDMAP_TIFF_ROOT = "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_tiff_minmax"

HDMAP_PNG_ROOT = os.getenv("HDMAP_PNG_ROOT", DEFAULT_HDMAP_PNG_ROOT)
HDMAP_TIFF_ROOT = os.getenv("HDMAP_TIFF_ROOT", DEFAULT_HDMAP_TIFF_ROOT)

# Allowed domains (whitelist for security)
ALLOWED_DOMAINS = frozenset({"domain_A", "domain_B", "domain_C", "domain_D"})
DOMAINS = list(ALLOWED_DOMAINS)

# Training constants (from original Dinomaly paper)
DEFAULT_BOTTLENECK_DROPOUT = 0.2  # Fixed per original paper
DEFAULT_MAX_STEPS = 10000
DEFAULT_BATCH_SIZE = 16
DEFAULT_IMAGE_SIZE = 448
DEFAULT_CROP_SIZE = 392
DEFAULT_GRADIENT_CLIP_VAL = 0.1

# Validation constants
VALIDATION_CHECK_EVERY_N_STEPS = 1000
MAX_VAL_CHECK_INTERVAL = 500
DEFAULT_VAL_CHECK_INTERVAL = 100

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# Logging Setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_domain(domain: str) -> None:
    """Validate domain name is safe and allowed.

    Args:
        domain: Domain name to validate.

    Raises:
        ValueError: If domain is invalid or contains unsafe characters.
    """
    if domain not in ALLOWED_DOMAINS:
        raise ValueError(f"Invalid domain: {domain}. Allowed: {ALLOWED_DOMAINS}")
    if any(char in domain for char in ['/', '\\', '..', '\0']):
        raise ValueError(f"Domain contains invalid characters: {domain}")


def format_auroc(auroc: float) -> str:
    """Format AUROC as percentage string.

    Args:
        auroc: AUROC value (0.0 to 1.0).

    Returns:
        Formatted string like "95.23%".
    """
    return f"{auroc * 100:.2f}%"


def safe_mean(values: list[float]) -> float:
    """Calculate mean with division by zero protection.

    Args:
        values: List of float values.

    Returns:
        Mean value, or 0.0 if list is empty.
    """
    return sum(values) / len(values) if values else 0.0


def create_safe_symlink(src: Path, dst: Path) -> bool:
    """Safely create symlink with proper error handling.

    Args:
        src: Source file path.
        dst: Destination symlink path.

    Returns:
        True if symlink was created successfully, False otherwise.
    """
    # Validate source exists and is a file
    if not src.exists() or not src.is_file():
        logger.warning(f"Invalid source file: {src}")
        return False

    # Sanitize filename to prevent path traversal
    safe_name = src.name.replace('/', '_').replace('\\', '_').replace('..', '_')
    dst = dst.parent / safe_name

    try:
        os.symlink(src, dst)
        return True
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Symlink already exists - verify it points to correct target
            try:
                if dst.resolve() != src.resolve():
                    logger.warning(f"Conflicting symlink exists: {dst}")
                    dst.unlink()
                    os.symlink(src, dst)
            except OSError:
                pass  # Could not resolve, skip
        else:
            logger.error(f"Failed to create symlink {dst}: {e}")
            return False
    return True


@contextmanager
def temporary_combined_dataset(data_root: Path, domains: list[str], seed: int = 42):
    """Context manager for temporary combined dataset with cleanup.

    Args:
        data_root: Root path of dataset.
        domains: List of domain names.
        seed: Random seed (used to create unique folder per run).

    Yields:
        Path to combined dataset root.
    """
    combined_root = data_root / f"_combined_multiclass_seed{seed}"
    # Clean up any existing folder before starting (ensures fresh symlinks)
    if combined_root.exists():
        logger.info(f"Removing existing combined dataset: {combined_root}")
        shutil.rmtree(combined_root)
    try:
        yield combined_root
    finally:
        # Cleanup on exit
        if combined_root.exists():
            logger.info(f"Cleaning up temporary dataset: {combined_root}")
            try:
                shutil.rmtree(combined_root)
            except OSError as e:
                logger.warning(f"Failed to cleanup {combined_root}: {e}")


def evaluate_domain(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    """Evaluate model on a single domain using GPU.

    Args:
        model: PyTorch model to evaluate.
        dataloader: DataLoader for the domain.
        device: Device to run inference on.

    Returns:
        Tuple of (labels, scores) lists.
    """
    labels: list[int] = []
    scores: list[float] = []

    # Ensure model is on GPU and in eval mode
    model = model.to(device)
    model.eval()

    logger.info(f"  Evaluating on device: {device} (samples: {len(dataloader.dataset)})")

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
        for batch in dataloader:
            # HDMAPDataset returns ImageBatch (dataclass), not dict
            # Support both dict and dataclass access
            if hasattr(batch, 'image'):
                images = batch.image.to(device, non_blocking=True)
                batch_labels = batch.gt_label
            else:
                images = batch["image"].to(device, non_blocking=True)
                batch_labels = batch["label"]

            # Forward pass - handle AnomalibModule output
            output = model(images)

            # Robust score extraction for AnomalibModule
            if hasattr(output, "pred_score") and output.pred_score is not None:
                batch_scores = output.pred_score.cpu()
            elif hasattr(output, "anomaly_map") and output.anomaly_map is not None:
                # Fallback: use max of anomaly map as score
                batch_scores = output.anomaly_map.flatten(1).max(dim=1)[0].cpu()
            elif isinstance(output, dict):
                if "pred_score" in output and output["pred_score"] is not None:
                    batch_scores = output["pred_score"].cpu()
                elif "anomaly_map" in output and output["anomaly_map"] is not None:
                    batch_scores = output["anomaly_map"].flatten(1).max(dim=1)[0].cpu()
                else:
                    raise ValueError(f"No valid score in output dict: {output.keys()}")
            elif torch.is_tensor(output):
                batch_scores = output.cpu()
            else:
                raise ValueError(f"Unexpected output format: {type(output)}")

            # Convert to list
            if batch_scores.dim() == 0:
                scores.append(float(batch_scores))
            else:
                scores.extend(batch_scores.numpy().tolist())
            labels.extend(batch_labels.numpy().tolist())

    # Cleanup GPU memory once after all batches (NOT per-batch - that's very slow)
    torch.cuda.empty_cache()

    return labels, scores


def visualize_domain_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    domain: str,
    max_samples: int = 50,
) -> None:
    """Visualize model predictions for a domain and save images.

    Args:
        model: PyTorch model to evaluate.
        dataloader: DataLoader for the domain.
        device: Device to run inference on.
        output_dir: Directory to save visualizations.
        domain: Domain name for folder organization.
        max_samples: Maximum number of samples to visualize per class (randomly selected).
    """
    # Create output directories
    vis_dir = output_dir / "visualizations" / domain
    (vis_dir / "good").mkdir(parents=True, exist_ok=True)
    (vis_dir / "fault").mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.eval()

    # Collect all indices by class for random sampling
    dataset = dataloader.dataset
    good_indices = []
    fault_indices = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        if hasattr(sample, 'gt_label'):
            label = int(sample.gt_label)
        elif isinstance(sample, dict):
            label = int(sample.get('label', sample.get('gt_label', 0)))
        else:
            label = 0
        if label == 1:
            fault_indices.append(idx)
        else:
            good_indices.append(idx)

    # Randomly sample indices
    random.shuffle(good_indices)
    random.shuffle(fault_indices)
    selected_good = good_indices[:max_samples]
    selected_fault = fault_indices[:max_samples]
    selected_indices = selected_good + selected_fault

    logger.info(f"  Generating visualizations for {domain} (random sampling: {len(selected_good)} good, {len(selected_fault)} fault)...")

    good_count = 0
    fault_count = 0

    # Process selected indices directly (random order)
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
        for idx in selected_indices:
            sample = dataset[idx]

            # Get sample data
            if hasattr(sample, 'image'):
                image = sample.image.unsqueeze(0).to(device, non_blocking=True)
                label = int(sample.gt_label)
                image_path = sample.image_path if hasattr(sample, 'image_path') else f"sample_{idx}"
            elif isinstance(sample, dict):
                image = sample["image"].unsqueeze(0).to(device, non_blocking=True)
                label = int(sample.get("label", sample.get("gt_label", 0)))
                image_path = sample.get("image_path", f"sample_{idx}")
            else:
                continue

            # Forward pass (single image)
            output = model(image)

            # Get anomaly map and score
            if hasattr(output, "anomaly_map") and output.anomaly_map is not None:
                anomaly_map = output.anomaly_map[0].cpu().numpy()  # Remove batch dim
            else:
                anomaly_map = None

            if hasattr(output, "pred_score") and output.pred_score is not None:
                pred_score = float(output.pred_score[0].cpu())
            elif anomaly_map is not None:
                pred_score = float(anomaly_map.max())
            else:
                pred_score = None

            # Denormalize image for visualization (ImageNet)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            image_denorm = image * std + mean
            image_denorm = image_denorm.clamp(0, 1)[0].cpu().numpy()  # Remove batch dim

            label_str = "fault" if label == 1 else "good"

            # Get image path stem for filename
            if isinstance(image_path, Path):
                img_name = image_path.stem
            elif isinstance(image_path, str):
                img_name = Path(image_path).stem
            else:
                img_name = f"sample_{idx}"

            # Prepare data for visualization
            image_np = image_denorm.transpose(1, 2, 0)  # CHW -> HWC
            image_np = (image_np * 255).astype(np.uint8)

            # Create NumpyImageItem for visualization
            item_data = {
                "image": image_np,
                "image_path": str(image_path) if image_path else None,
            }

            if anomaly_map is not None:
                # Normalize anomaly map for visualization (0-1 range)
                if anomaly_map.ndim == 3:
                    anomaly_map = anomaly_map.squeeze(0)  # Remove channel dim if present
                amap_min, amap_max = anomaly_map.min(), anomaly_map.max()
                if amap_max > amap_min:
                    amap_norm = (anomaly_map - amap_min) / (amap_max - amap_min)
                else:
                    amap_norm = np.zeros_like(anomaly_map)
                item_data["anomaly_map"] = amap_norm

            if pred_score is not None:
                item_data["pred_score"] = pred_score

            try:
                item = NumpyImageItem(**item_data)

                # Generate visualization
                vis_image = visualize_image_item(
                    item,
                    fields=["image"],
                    overlay_fields=[("image", ["anomaly_map"])] if anomaly_map is not None else [],
                    field_size=(256, 256),
                )

                if vis_image is not None:
                    # Add score text to filename
                    score_str = f"_score{pred_score:.3f}" if pred_score is not None else ""
                    save_path = vis_dir / label_str / f"{img_name}{score_str}.png"
                    vis_image.save(save_path)

                    if label_str == "good":
                        good_count += 1
                    else:
                        fault_count += 1

            except Exception as e:
                logger.warning(f"  Failed to visualize {img_name}: {e}")
                continue

    logger.info(f"  Saved {good_count} good, {fault_count} fault visualizations to {vis_dir}")


def visualize_score_distribution(
    labels: list[int],
    scores: list[float],
    domain: str,
    output_dir: Path,
) -> None:
    """Visualize anomaly score distribution for a domain.

    Creates histogram plot showing good vs fault score distributions.

    Args:
        labels: Ground truth labels (0=normal, 1=anomaly).
        scores: Anomaly scores.
        domain: Domain name.
        output_dir: Directory to save visualizations.
    """
    # Separate scores by class
    good_scores = [s for l, s in zip(labels, scores) if l == 0]
    fault_scores = [s for l, s in zip(labels, scores) if l == 1]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histograms with transparency
    bins = 50
    if good_scores:
        ax.hist(good_scores, bins=bins, alpha=0.6, label=f'Good (n={len(good_scores)})',
                color='green', density=True)
    if fault_scores:
        ax.hist(fault_scores, bins=bins, alpha=0.6, label=f'Fault (n={len(fault_scores)})',
                color='red', density=True)

    # Add statistics text
    stats_lines = []
    if good_scores:
        stats_lines.append(f"Good:  mean={np.mean(good_scores):.4f}, std={np.std(good_scores):.4f}")
    if fault_scores:
        stats_lines.append(f"Fault: mean={np.mean(fault_scores):.4f}, std={np.std(fault_scores):.4f}")

    if stats_lines:
        stats_text = "\n".join(stats_lines)
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Density')
    ax.set_title(f'{domain} - Anomaly Score Distribution')
    ax.legend()

    # Save
    vis_dir = output_dir / "visualizations" / domain
    vis_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(vis_dir / "score_distribution.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"  Saved score distribution plot to {vis_dir / 'score_distribution.png'}")


def visualize_all_domains_score_comparison(
    domain_results: dict[str, tuple[list[int], list[float]]],
    output_dir: Path,
) -> None:
    """Create 2x2 subplot comparing score distributions across all domains.

    Args:
        domain_results: Dictionary mapping domain names to (labels, scores) tuples.
        output_dir: Directory to save visualizations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    domains = sorted(domain_results.keys())

    for idx, domain in enumerate(domains):
        if idx >= 4:
            break  # Only show 4 domains max
        labels, scores = domain_results[domain]
        ax = axes[idx]

        good_scores = [s for l, s in zip(labels, scores) if l == 0]
        fault_scores = [s for l, s in zip(labels, scores) if l == 1]

        if good_scores:
            ax.hist(good_scores, bins=30, alpha=0.6, label='Good', color='green', density=True)
        if fault_scores:
            ax.hist(fault_scores, bins=30, alpha=0.6, label='Fault', color='red', density=True)
        ax.set_title(f'{domain}')
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Density')
        ax.legend()

    plt.suptitle('Per-Domain Anomaly Score Distributions', fontsize=14)
    plt.tight_layout()

    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(vis_dir / "all_domains_score_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved all domains score comparison to {vis_dir / 'all_domains_score_comparison.png'}")


def export_domain_scores_to_csv(
    dataloader: DataLoader,
    labels: list[int],
    scores: list[float],
    domain: str,
    output_dir: Path,
) -> Path:
    """Export per-domain prediction scores to CSV file.

    Args:
        dataloader: DataLoader for the domain (to get image paths).
        labels: Ground truth labels (0=normal, 1=anomaly).
        scores: Predicted anomaly scores.
        domain: Domain name.
        output_dir: Directory to save CSV file.

    Returns:
        Path to the saved CSV file.
    """
    import csv

    # Get image paths from dataset
    dataset = dataloader.dataset
    image_paths = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        if hasattr(sample, 'image_path'):
            img_path = sample.image_path
        elif isinstance(sample, dict):
            img_path = sample.get('image_path', f'sample_{idx}')
        else:
            img_path = f'sample_{idx}'

        if isinstance(img_path, Path):
            image_paths.append(str(img_path))
        else:
            image_paths.append(str(img_path))

    # Ensure we have matching lengths
    if len(image_paths) != len(labels) or len(image_paths) != len(scores):
        logger.warning(f"Length mismatch: paths={len(image_paths)}, labels={len(labels)}, scores={len(scores)}")
        # Use minimum length
        min_len = min(len(image_paths), len(labels), len(scores))
        image_paths = image_paths[:min_len]
        labels = labels[:min_len]
        scores = scores[:min_len]

    # Create output directory
    csv_dir = output_dir / "scores"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{domain}_scores.csv"

    # Write CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'image_path', 'gt_label', 'gt_label_str', 'pred_score', 'domain'])

        for img_path, label, score in zip(image_paths, labels, scores):
            img_name = Path(img_path).stem if img_path else 'unknown'
            label_str = 'fault' if label == 1 else 'good'
            writer.writerow([img_name, img_path, label, label_str, f'{score:.6f}', domain])

    logger.info(f"  Saved {len(labels)} scores to {csv_path}")
    return csv_path


def compute_auroc_safe(labels: list[int], scores: list[float]) -> float:
    """Compute AUROC with edge case handling.

    Args:
        labels: Ground truth labels.
        scores: Predicted scores.

    Returns:
        AUROC value, or 0.0 if cannot be computed.
    """
    if len(set(labels)) < 2:
        logger.warning("Cannot compute AUROC: only one class present")
        return 0.0
    try:
        return roc_auc_score(labels, scores)
    except Exception as e:
        logger.error(f"Failed to compute AUROC: {e}")
        return 0.0


def compute_comprehensive_metrics(
    labels: list[int],
    scores: list[float],
) -> dict[str, float]:
    """Compute comprehensive evaluation metrics.

    Metrics computed:
    - AUROC: Area Under ROC Curve
    - TPR@FPR=1%: True Positive Rate at 1% False Positive Rate
    - TPR@FPR=5%: True Positive Rate at 5% False Positive Rate
    - Precision: at optimal threshold (Youden's J)
    - Recall: at optimal threshold
    - F1 Score: at optimal threshold
    - Accuracy: at optimal threshold

    Args:
        labels: Ground truth labels (0=normal, 1=anomaly).
        scores: Anomaly scores (higher = more anomalous).

    Returns:
        Dictionary of metric names to values.
    """
    labels_arr = np.array(labels)
    scores_arr = np.array(scores)

    metrics = {
        "auroc": 0.0,
        "tpr_at_fpr_1pct": 0.0,
        "tpr_at_fpr_5pct": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "accuracy": 0.0,
        "optimal_threshold": 0.0,
    }

    # Check if we have both classes
    if len(set(labels)) < 2:
        logger.warning("Cannot compute metrics: only one class present")
        return metrics

    try:
        # AUROC
        metrics["auroc"] = roc_auc_score(labels_arr, scores_arr)

        # ROC curve for TPR@FPR calculations
        fpr, tpr, thresholds = roc_curve(labels_arr, scores_arr)

        # TPR@FPR=1% (find TPR where FPR is closest to 0.01)
        idx_1pct = np.argmin(np.abs(fpr - 0.01))
        metrics["tpr_at_fpr_1pct"] = float(tpr[idx_1pct])

        # TPR@FPR=5% (find TPR where FPR is closest to 0.05)
        idx_5pct = np.argmin(np.abs(fpr - 0.05))
        metrics["tpr_at_fpr_5pct"] = float(tpr[idx_5pct])

        # Optimal threshold using Youden's J statistic (maximize TPR - FPR)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        metrics["optimal_threshold"] = float(optimal_threshold)

        # Binary predictions at optimal threshold
        predictions = (scores_arr >= optimal_threshold).astype(int)

        # Precision, Recall, F1, Accuracy at optimal threshold
        metrics["precision"] = float(precision_score(labels_arr, predictions, zero_division=0))
        metrics["recall"] = float(recall_score(labels_arr, predictions, zero_division=0))
        metrics["f1_score"] = float(f1_score(labels_arr, predictions, zero_division=0))
        metrics["accuracy"] = float(accuracy_score(labels_arr, predictions))

    except Exception as e:
        logger.error(f"Failed to compute comprehensive metrics: {e}")

    return metrics


def format_metrics_table(domain_metrics: dict[str, dict[str, float]]) -> str:
    """Format metrics as a readable table string.

    Args:
        domain_metrics: Dict mapping domain names to their metrics dicts.

    Returns:
        Formatted table string.
    """
    header = (
        f"{'Domain':<10} {'AUROC':>8} {'TPR@1%':>8} {'TPR@5%':>8} "
        f"{'Prec':>8} {'Recall':>8} {'F1':>8} {'Acc':>8}"
    )
    separator = "-" * len(header)

    rows = [header, separator]
    for domain, m in domain_metrics.items():
        row = (
            f"{domain:<10} {m['auroc']*100:>7.2f}% {m['tpr_at_fpr_1pct']*100:>7.2f}% "
            f"{m['tpr_at_fpr_5pct']*100:>7.2f}% {m['precision']*100:>7.2f}% "
            f"{m['recall']*100:>7.2f}% {m['f1_score']*100:>7.2f}% {m['accuracy']*100:>7.2f}%"
        )
        rows.append(row)

    return "\n".join(rows)


# =============================================================================
# Dataset
# =============================================================================

class HDMAPMultiClassDataset(torch.utils.data.Dataset):
    """Multi-class dataset combining all HDMAP domains for unified training.

    Args:
        root: Path to HDMAP dataset root.
        domains: List of domain names to include.
        split: "train" or "test".
        transform: Image transformations.
    """

    def __init__(
        self,
        root: str | Path,
        domains: list[str],
        split: str = "train",
        transform: transforms.Compose | None = None,
    ) -> None:
        """Initialize multi-class dataset."""
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # Validate domains
        for domain in domains:
            validate_domain(domain)
        self.domains = domains

        self.samples: list[dict[str, Any]] = []
        self.domain_ids: dict[str, int] = {}

        for domain_idx, domain in enumerate(domains):
            self.domain_ids[domain] = domain_idx
            self._load_domain_samples(domain, domain_idx)

        logger.info(f"Loaded {len(self.samples)} samples from {len(domains)} domains ({split})")

        # Efficient counting using Counter
        domain_counts = Counter(s["domain"] for s in self.samples)
        for domain in domains:
            logger.info(f"  {domain}: {domain_counts.get(domain, 0)} samples")

    def _load_domain_samples(self, domain: str, domain_idx: int) -> None:
        """Load samples from a single domain.

        Args:
            domain: Domain name.
            domain_idx: Domain index.
        """
        if self.split == "train":
            domain_path = self.root / domain / "train" / "good"
            if not domain_path.exists():
                raise FileNotFoundError(f"Training path not found: {domain_path}")

            tiff_files = list(domain_path.glob("*.tiff"))
            if not tiff_files:
                logger.warning(f"No TIFF files found in {domain_path}")

            for img_path in tiff_files:
                self.samples.append({
                    "image_path": str(img_path),
                    "domain": domain,
                    "domain_idx": domain_idx,
                    "label": 0,  # Good
                })
        else:
            # Test: both good and fault
            for label_name, label_value in [("good", 0), ("fault", 1)]:
                domain_path = self.root / domain / "test" / label_name
                if domain_path.exists():
                    for img_path in domain_path.glob("*.tiff"):
                        self.samples.append({
                            "image_path": str(img_path),
                            "domain": domain,
                            "domain_idx": domain_idx,
                            "label": label_value,
                        })

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with image, label, domain info.
        """
        sample = self.samples[idx]
        image_path = sample["image_path"]

        # Load TIFF as float32 without clipping
        if image_path.endswith(('.tiff', '.tif')):
            # Use tifffile for proper float32 loading
            img_array = tifffile.imread(image_path)  # float32, shape: (H, W) or (H, W, C)

            # Handle grayscale -> RGB
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[-1] == 1:
                img_array = np.repeat(img_array, 3, axis=-1)

            # Convert to tensor: (H, W, C) -> (C, H, W), float32
            image = torch.from_numpy(img_array).permute(2, 0, 1).float()
        else:
            # Fallback for PNG/JPG
            with Image.open(image_path) as img:
                img_array = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
                image = torch.from_numpy(img_array).permute(2, 0, 1)

        # Apply transforms (Resize, CenterCrop, Normalize)
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": sample["label"],
            "domain_idx": sample["domain_idx"],
            "image_path": image_path,
            "domain": sample["domain"],
        }


# =============================================================================
# Callbacks
# =============================================================================

class ValidationAUROCPerDomain(Callback):
    """Callback to track validation AUROC per domain during training.

    Args:
        test_dataloaders: Dictionary mapping domain names to DataLoaders.
    """

    def __init__(self, test_dataloaders: dict[str, DataLoader]) -> None:
        """Initialize callback."""
        super().__init__()
        self.test_dataloaders = test_dataloaders
        self.history: list[dict[str, Any]] = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Evaluate per-domain AUROC at end of each epoch.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: Lightning module being trained.
        """
        if trainer.global_step % VALIDATION_CHECK_EVERY_N_STEPS != 0:
            return

        pl_module.eval()
        device = pl_module.device

        domain_results: dict[str, float] = {}
        all_labels: list[int] = []
        all_scores: list[float] = []

        with torch.no_grad():
            for domain, dataloader in self.test_dataloaders.items():
                # Use shared evaluation function
                domain_labels, domain_scores = evaluate_domain(
                    pl_module, dataloader, device
                )

                auroc = compute_auroc_safe(domain_labels, domain_scores)
                domain_results[domain] = auroc
                all_labels.extend(domain_labels)
                all_scores.extend(domain_scores)

                logger.info(f"Step {trainer.global_step} - {domain}: AUROC = {format_auroc(auroc)}")

        # Overall AUROC
        overall_auroc = compute_auroc_safe(all_labels, all_scores)
        logger.info(f"Step {trainer.global_step} - Overall: AUROC = {format_auroc(overall_auroc)}")

        # Log to TensorBoard
        if trainer.logger:
            for domain, auroc in domain_results.items():
                trainer.logger.experiment.add_scalar(
                    f"val_auroc/{domain}", auroc, trainer.global_step
                )
            trainer.logger.experiment.add_scalar(
                "val_auroc/overall", overall_auroc, trainer.global_step
            )

        self.history.append({
            "step": trainer.global_step,
            "domain_aurocs": domain_results,
            "overall_auroc": overall_auroc,
        })

        pl_module.train()


# =============================================================================
# Transforms
# =============================================================================

def get_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    crop_size: int = DEFAULT_CROP_SIZE,
) -> transforms.Compose:
    """Get data transforms matching original Dinomaly paper.

    Args:
        image_size: Size to resize images to.
        crop_size: Size to center crop to.

    Returns:
        Composed transforms.
    """
    # transforms.v2 for tensor input (TIFF loaded as float32 tensor)
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.CenterCrop(crop_size),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# =============================================================================
# Experiment Functions
# =============================================================================

def create_combined_dataset(
    data_root: Path,
    domains: list[str],
    combined_root: Path,
) -> tuple[int, int, int]:
    """Create combined dataset with symlinks.

    Args:
        data_root: Root path of original dataset.
        domains: List of domain names.
        combined_root: Path for combined dataset.

    Returns:
        Tuple of (train_count, test_good_count, test_fault_count).
    """
    # Create directories
    train_dir = combined_root / "train" / "good"
    train_dir.mkdir(parents=True, exist_ok=True)

    test_good_dir = combined_root / "test" / "good"
    test_good_dir.mkdir(parents=True, exist_ok=True)

    test_fault_dir = combined_root / "test" / "fault"
    test_fault_dir.mkdir(parents=True, exist_ok=True)

    train_count = 0
    test_good_count = 0
    test_fault_count = 0

    for domain in domains:
        validate_domain(domain)
        domain_root = data_root / domain

        # Training samples
        train_path = domain_root / "train" / "good"
        if not train_path.exists():
            raise FileNotFoundError(f"Training path not found: {train_path}")

        for img_path in train_path.glob("*.tiff"):
            link_path = train_dir / f"{domain}_{img_path.name}"
            if create_safe_symlink(img_path, link_path):
                train_count += 1

        # Test good samples
        test_good_path = domain_root / "test" / "good"
        if test_good_path.exists():
            for img_path in test_good_path.glob("*.tiff"):
                link_path = test_good_dir / f"{domain}_{img_path.name}"
                if create_safe_symlink(img_path, link_path):
                    test_good_count += 1

        # Test fault samples
        test_fault_path = domain_root / "test" / "fault"
        if test_fault_path.exists():
            for img_path in test_fault_path.glob("*.tiff"):
                link_path = test_fault_dir / f"{domain}_{img_path.name}"
                if create_safe_symlink(img_path, link_path):
                    test_fault_count += 1

    return train_count, test_good_count, test_fault_count


def run_multiclass_experiment(
    data_root: str,
    domains: list[str],
    max_steps: int,
    gpu_id: int,
    output_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    crop_size: int = DEFAULT_CROP_SIZE,
    encoder_name: str = "dinov2reg_vit_base_14",
    seed: int = 42,
) -> dict[str, Any]:
    """Run multi-class unified training experiment.

    This follows the original Dinomaly paper's unified training approach:
    - Combine all domains into single training dataset (using AllDomainsHDMAPDataModule)
    - Train one model on all domains
    - Evaluate per-domain and overall AUROC

    **통합 데이터 로딩**: Training과 Testing 모두 동일한 HDMAPDataset을 사용합니다.
    - TIFF 파일: tifffile로 로딩 (float32 정밀도 유지, NO clipping)
    - 동일한 전처리 파이프라인 보장

    Args:
        data_root: Path to dataset root.
        domains: List of domain names.
        max_steps: Maximum training steps.
        gpu_id: GPU device ID.
        output_dir: Output directory for results.
        batch_size: Training batch size.
        image_size: Image resize size.
        crop_size: Center crop size.
        encoder_name: DINOv2 encoder variant.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing experiment results.
    """
    set_seed(seed)

    logger.info("=" * 60)
    logger.info("MULTI-CLASS UNIFIED TRAINING")
    logger.info("=" * 60)
    logger.info(f"Seed: {seed}")
    logger.info(f"Domains: {domains}")
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Encoder: {encoder_name}")
    logger.info("Using AllDomainsHDMAPDataModule (tifffile-based TIFF loading)")

    experiment_dir = output_dir / "multiclass_unified"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Create per-domain test dataloaders using HDMAPDataset (same loading as training)
    # 이렇게 하면 Training과 Testing에서 동일한 이미지 로딩 방식 사용
    test_dataloaders: dict[str, DataLoader] = {}
    for domain in domains:
        test_dataset = HDMAPDataset(
            root=data_root,
            domain=domain,
            split="test",
            target_size=(image_size, image_size),  # Resize to match training
        )
        test_dataloaders[domain] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=test_dataset.collate_fn,  # HDMAPDataset returns ImageItem
        )

    # Setup evaluator
    val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
    test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
    evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])

    # Create model (standard Dinomaly with fixed dropout)
    model = Dinomaly(
        encoder_name=encoder_name,
        bottleneck_dropout=DEFAULT_BOTTLENECK_DROPOUT,
        evaluator=evaluator,
        pre_processor=True,
        visualizer=False,  # Disable default ImageVisualizer (per-domain viz in visualizations/ folder)
    )

    # Setup callbacks
    per_domain_callback = ValidationAUROCPerDomain(test_dataloaders)

    callbacks = [
        ModelCheckpoint(
            dirpath=experiment_dir / "checkpoints",
            filename="best-{step}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
        ),
        per_domain_callback,
    ]

    # Setup TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=str(experiment_dir),
        name="tensorboard",
        version="",
    )

    # Create unified datamodule using AllDomainsHDMAPDataModule
    # 이 datamodule은 HDMAPDataset을 사용하여 TIFF를 tifffile로 로딩
    logger.info("Creating AllDomainsHDMAPDataModule...")
    datamodule = AllDomainsHDMAPDataModule(
        root=data_root,
        domains=domains,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=8,
        val_split_mode="from_test",
        val_split_ratio=0.1,
        seed=seed,
    )

    # Set val_check_interval to safe fixed value
    val_check_interval = 200
    logger.info(f"Using val_check_interval={val_check_interval}")

    engine = Engine(
        max_steps=max_steps,
        accelerator="gpu",
        devices=[gpu_id],
        val_check_interval=val_check_interval,
        callbacks=callbacks,
        default_root_dir=str(experiment_dir),
        logger=tb_logger,
        gradient_clip_val=DEFAULT_GRADIENT_CLIP_VAL,
    )

    # Train
    logger.info("Starting training...")
    engine.fit(model, datamodule=datamodule)

    # Test
    logger.info("Evaluating model...")
    test_results = engine.test(model, datamodule=datamodule)

    # Extract overall AUROC
    if not test_results:
        logger.warning("Test failed to produce results")
        overall_auroc = 0.0
    else:
        overall_auroc = test_results[0].get("test_image_AUROC", 0.0)

    logger.info(f"Overall Test AUROC: {format_auroc(overall_auroc)}")

    # Evaluate per-domain (final evaluation) using same HDMAPDataset-based dataloaders
    logger.info("\nPer-Domain Evaluation (Final):")

    # Explicitly use GPU (engine.test() may move model to CPU)
    device = torch.device(f"cuda:{gpu_id}")
    model = model.to(device)
    model.eval()

    domain_metrics: dict[str, dict[str, float]] = {}
    domain_results_for_viz: dict[str, tuple[list[int], list[float]]] = {}  # For score distribution plots
    all_labels: list[int] = []
    all_scores: list[float] = []

    for domain in domains:
        labels, scores = evaluate_domain(model, test_dataloaders[domain], device)
        metrics = compute_comprehensive_metrics(labels, scores)
        domain_metrics[domain] = metrics
        domain_results_for_viz[domain] = (labels, scores)  # Store for score distribution visualization
        all_labels.extend(labels)
        all_scores.extend(scores)
        logger.info(f"  {domain}: AUROC={metrics['auroc']*100:.2f}%, TPR@1%={metrics['tpr_at_fpr_1pct']*100:.2f}%, TPR@5%={metrics['tpr_at_fpr_5pct']*100:.2f}%")

        # Generate per-domain score distribution visualization
        visualize_score_distribution(labels, scores, domain, experiment_dir)

        # Export per-domain scores to CSV
        export_domain_scores_to_csv(test_dataloaders[domain], labels, scores, domain, experiment_dir)

        # Generate per-domain sample visualizations (anomaly maps overlaid on images)
        visualize_domain_predictions(
            model=model,
            dataloader=test_dataloaders[domain],
            device=device,
            output_dir=experiment_dir,
            domain=domain,
            max_samples=50,  # 50 good + 50 fault per domain
        )

    # Generate all-domains score comparison visualization
    visualize_all_domains_score_comparison(domain_results_for_viz, experiment_dir)

    # Compute overall metrics from pooled data
    overall_metrics = compute_comprehensive_metrics(all_labels, all_scores)

    # Compute mean of per-domain metrics
    metric_names = ["auroc", "tpr_at_fpr_1pct", "tpr_at_fpr_5pct", "precision", "recall", "f1_score", "accuracy"]
    mean_metrics = {
        name: safe_mean([domain_metrics[d][name] for d in domains])
        for name in metric_names
    }

    logger.info(f"\n  Per-Domain Mean AUROC: {mean_metrics['auroc']*100:.2f}%")
    logger.info(f"  Overall (Pooled) AUROC: {overall_metrics['auroc']*100:.2f}%")

    # Print comprehensive table
    logger.info("\n" + format_metrics_table(domain_metrics))
    logger.info(f"\n{'Mean':<10} {mean_metrics['auroc']*100:>7.2f}% {mean_metrics['tpr_at_fpr_1pct']*100:>7.2f}% "
                f"{mean_metrics['tpr_at_fpr_5pct']*100:>7.2f}% {mean_metrics['precision']*100:>7.2f}% "
                f"{mean_metrics['recall']*100:>7.2f}% {mean_metrics['f1_score']*100:>7.2f}% {mean_metrics['accuracy']*100:>7.2f}%")

    # Save results
    results: dict[str, Any] = {
        "experiment_type": "multiclass_unified",
        "domains": domains,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "encoder_name": encoder_name,
        "overall_auroc": float(overall_auroc),
        "per_domain_metrics": {k: v for k, v in domain_metrics.items()},
        "mean_metrics": mean_metrics,
        "overall_pooled_metrics": overall_metrics,
        "training_history": per_domain_callback.history,
        "timestamp": datetime.now().isoformat(),
        "data_loading": "tifffile (unified)",  # 통합 로딩 방식 기록
    }

    with open(experiment_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {experiment_dir}")

    return results


def run_singleclass_experiments(
    data_root: str,
    domains: list[str],
    max_steps: int,
    gpu_id: int,
    output_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    encoder_name: str = "dinov2reg_vit_base_14",
    seed: int = 42,
) -> dict[str, Any]:
    """Run single-class (per-domain) training for comparison.

    This is the traditional approach where each domain has its own model.

    **통합 데이터 로딩**: HDMAPDataModule을 사용하여 Training과 Testing 동일한 방식 사용.
    - TIFF 파일: tifffile로 로딩 (float32 정밀도 유지, NO clipping)

    Args:
        data_root: Path to dataset root.
        domains: List of domain names.
        max_steps: Maximum training steps.
        gpu_id: GPU device ID.
        output_dir: Output directory for results.
        batch_size: Training batch size.
        encoder_name: DINOv2 encoder variant.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing per-domain results.
    """
    logger.info("=" * 60)
    logger.info("SINGLE-CLASS (PER-DOMAIN) TRAINING")
    logger.info("=" * 60)
    logger.info(f"Seed: {seed}")
    logger.info("Using HDMAPDataModule (tifffile-based TIFF loading)")

    all_results: dict[str, dict[str, Any]] = {}

    for domain in domains:
        set_seed(seed)
        validate_domain(domain)

        logger.info(f"\n{'='*40}")
        logger.info(f"Training on: {domain}")
        logger.info(f"{'='*40}")

        experiment_dir = output_dir / f"singleclass_{domain}"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Setup evaluator
        val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
        test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
        evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])

        # Create datamodule using HDMAPDataModule (tifffile-based TIFF loading)
        datamodule = HDMAPDataModule(
            root=data_root,
            domain=domain,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=8,
            val_split_mode="from_test",
            val_split_ratio=0.1,
            seed=seed,
        )

        # Create model
        model = Dinomaly(
            encoder_name=encoder_name,
            bottleneck_dropout=DEFAULT_BOTTLENECK_DROPOUT,
            evaluator=evaluator,
            pre_processor=True,
            visualizer=False,  # Disable default ImageVisualizer
        )

        # Setup TensorBoard logger
        tb_logger = TensorBoardLogger(
            save_dir=str(experiment_dir),
            name="tensorboard",
            version="",
        )

        # Val check interval - safe fixed value
        val_check_interval = 50

        engine = Engine(
            max_steps=max_steps,
            accelerator="gpu",
            devices=[gpu_id],
            val_check_interval=val_check_interval,
            default_root_dir=str(experiment_dir),
            logger=tb_logger,
            gradient_clip_val=DEFAULT_GRADIENT_CLIP_VAL,
        )

        # Train
        engine.fit(model, datamodule=datamodule)

        # Test
        test_results = engine.test(model, datamodule=datamodule)

        if not test_results:
            logger.warning(f"Test failed for {domain}")
            auroc = 0.0
        else:
            auroc = test_results[0].get("test_image_AUROC", 0.0)

        logger.info(f"{domain} Test AUROC: {format_auroc(auroc)}")

        all_results[domain] = {
            "auroc": float(auroc),
            "max_steps": max_steps,
            "data_loading": "tifffile (unified)",
        }

        # Save domain results
        with open(experiment_dir / "results.json", "w") as f:
            json.dump(all_results[domain], f, indent=2)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SINGLE-CLASS SUMMARY")
    logger.info("=" * 60)
    for domain, result in all_results.items():
        logger.info(f"  {domain}: {format_auroc(result['auroc'])}")

    aurocs = [r["auroc"] for r in all_results.values()]
    mean_auroc = safe_mean(aurocs)
    logger.info(f"  Mean: {format_auroc(mean_auroc)}")

    return all_results


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Dinomaly Baseline for HDMAP (v2)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["multiclass", "singleclass", "compare"],
        default="multiclass",
        help="Training mode: multiclass (unified), singleclass (per-domain), or compare (both)",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=DOMAINS,
        help="Domains to include (default: all 4)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum training steps (default: {DEFAULT_MAX_STEPS}, same as original paper)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE}, same as original paper)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=HDMAP_TIFF_ROOT,
        help="Path to HDMAP dataset",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="results/dinomaly_baseline",
        help="Result directory (default: results/dinomaly_baseline)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="dinov2reg_vit_base_14",
        choices=["dinov2reg_vit_small_14", "dinov2reg_vit_base_14", "dinov2reg_vit_large_14"],
        help="DINOv2 encoder variant",
    )

    args = parser.parse_args()

    # Set global seed
    set_seed(args.seed)

    # Validate domains
    for domain in args.domains:
        validate_domain(domain)

    # Setup output directory with seed in folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.result_dir) / f"{timestamp}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment settings
    settings = {
        "mode": args.mode,
        "domains": args.domains,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "encoder": args.encoder,
        "data_root": args.data_root,
        "seed": args.seed,
        "timestamp": timestamp,
    }
    with open(output_dir / "experiment_settings.json", "w") as f:
        json.dump(settings, f, indent=2)

    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output directory: {output_dir}")

    results: dict[str, Any] = {}

    if args.mode in ["multiclass", "compare"]:
        results["multiclass"] = run_multiclass_experiment(
            data_root=args.data_root,
            domains=args.domains,
            max_steps=args.max_steps,
            gpu_id=args.gpu,
            output_dir=output_dir,
            batch_size=args.batch_size,
            encoder_name=args.encoder,
            seed=args.seed,
        )

    if args.mode in ["singleclass", "compare"]:
        results["singleclass"] = run_singleclass_experiments(
            data_root=args.data_root,
            domains=args.domains,
            max_steps=args.max_steps,
            gpu_id=args.gpu,
            output_dir=output_dir,
            batch_size=args.batch_size,
            seed=args.seed,
            encoder_name=args.encoder,
        )

    # Comparison summary
    if args.mode == "compare":
        logger.info("\n" + "=" * 60)
        logger.info("MULTICLASS vs SINGLECLASS COMPARISON")
        logger.info("=" * 60)

        multiclass_mean = results["multiclass"]["mean_domain_auroc"]
        singleclass_aurocs = [r["auroc"] for r in results["singleclass"].values()]
        singleclass_mean = safe_mean(singleclass_aurocs)

        logger.info(f"Multi-class (Unified) Mean AUROC: {format_auroc(multiclass_mean)}")
        logger.info(f"Single-class (Per-domain) Mean AUROC: {format_auroc(singleclass_mean)}")
        logger.info(f"Difference: {(multiclass_mean - singleclass_mean)*100:+.2f}%")

        # Per-domain comparison
        logger.info("\nPer-Domain Comparison:")
        for domain in args.domains:
            multi_auroc = results["multiclass"]["per_domain_auroc"].get(domain, 0)
            single_auroc = results["singleclass"].get(domain, {}).get("auroc", 0)
            diff = multi_auroc - single_auroc
            logger.info(
                f"  {domain}: Multi={format_auroc(multi_auroc)}, "
                f"Single={format_auroc(single_auroc)}, Diff={diff*100:+.2f}%"
            )

    # Save final summary
    with open(output_dir / "final_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    logger.info(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
