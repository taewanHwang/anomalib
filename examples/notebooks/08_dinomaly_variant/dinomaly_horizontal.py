"""
Dinomaly Horizontal Segment Dropout for HDMAP Dataset.

This script implements Horizontal Segment Dropout variant of Dinomaly.
It applies spatially-aware dropout that drops consecutive tokens within
the same row, suppressing the decoder's ability to use horizontal neighbor
information for reconstruction.

Key insight: Domain C defects appear as horizontal patterns. By dropping
consecutive horizontal tokens, we force the decoder to be less proficient
at horizontal reconstruction, making horizontal defects more detectable.

Ablation configurations:
    A. Baseline: elem_p=0.2, no segment dropout (original Dinomaly)
    B. Element↓: elem_p=0.1, no segment dropout (reduced regularization)
    C. Segment Only: elem_p=0.0, segment dropout enabled
    D. Hybrid: elem_p=0.1, segment dropout enabled (recommended)

Key features:
- --elem-p: Element-wise dropout probability (default: 0.1)
- --row-p: Row selection probability for segment dropout (default: 0.2)
- --seg-len: Length of consecutive tokens to drop (default: 2)
- --seg-drop-p: Drop probability within segments (default: 0.6)
- --disable-segment: Disable segment dropout (for Config A, B)
- --seed: Specify random seed for reproducibility
- --result-dir: Specify output directory for results

Usage:
    # Config D: Hybrid (recommended)
    python dinomaly_horizontal.py --seed 42 --gpu 0 \\
        --elem-p 0.1 --row-p 0.2 --seg-len 2 --seg-drop-p 0.6 \\
        --result-dir results/dinomaly_horizontal

    # Config B: Element only (no segment dropout)
    python dinomaly_horizontal.py --seed 42 --gpu 0 \\
        --elem-p 0.1 --disable-segment \\
        --result-dir results/dinomaly_horizontal/config_B

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
from anomalib.models.image.dinomaly_variants import DinomalyHorizontal, DinomalyHorizontalTopK
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
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_domain(domain: str) -> None:
    """Validate domain name is safe and allowed."""
    if domain not in ALLOWED_DOMAINS:
        raise ValueError(f"Invalid domain: {domain}. Allowed: {ALLOWED_DOMAINS}")
    if any(char in domain for char in ['/', '\\', '..', '\0']):
        raise ValueError(f"Domain contains invalid characters: {domain}")


def format_auroc(auroc: float) -> str:
    """Format AUROC as percentage string."""
    return f"{auroc * 100:.2f}%"


def safe_mean(values: list[float]) -> float:
    """Calculate mean with division by zero protection."""
    return sum(values) / len(values) if values else 0.0


def evaluate_domain(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    """Evaluate model on a single domain using GPU."""
    labels: list[int] = []
    scores: list[float] = []

    model = model.to(device)
    model.eval()

    logger.info(f"  Evaluating on device: {device} (samples: {len(dataloader.dataset)})")

    with torch.no_grad():
        for batch in dataloader:
            if hasattr(batch, 'image'):
                images = batch.image.to(device, non_blocking=True)
                batch_labels = batch.gt_label
            else:
                images = batch["image"].to(device, non_blocking=True)
                batch_labels = batch["label"]

            output = model(images)

            if hasattr(output, "pred_score") and output.pred_score is not None:
                batch_scores = output.pred_score.cpu()
            elif hasattr(output, "anomaly_map") and output.anomaly_map is not None:
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

            if batch_scores.dim() == 0:
                scores.append(float(batch_scores))
            else:
                scores.extend(batch_scores.numpy().tolist())
            labels.extend(batch_labels.numpy().tolist())

    torch.cuda.empty_cache()
    return labels, scores


def visualize_anomaly_maps(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    domain: str,
    output_dir: Path,
    num_samples: int = 10,
    save_good: bool = True,
    save_fault: bool = True,
) -> None:
    """Generate and save per-sample anomaly map visualizations.

    Args:
        model: Trained model
        dataloader: DataLoader for the domain
        device: Device to run inference on
        domain: Domain name for folder organization
        output_dir: Output directory
        num_samples: Number of samples to visualize per class
        save_good: Whether to save good samples
        save_fault: Whether to save fault samples
    """
    model = model.to(device)
    model.eval()

    vis_dir = output_dir / "visualizations" / domain / "anomaly_maps"
    vis_dir.mkdir(parents=True, exist_ok=True)

    good_count = 0
    fault_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if good_count >= num_samples and fault_count >= num_samples:
                break

            # Get images and labels
            if hasattr(batch, 'image'):
                images = batch.image.to(device, non_blocking=True)
                batch_labels = batch.gt_label.numpy()
                if hasattr(batch, 'image_path'):
                    image_paths = batch.image_path
                else:
                    image_paths = [f"sample_{batch_idx}_{i}" for i in range(len(images))]
            else:
                images = batch["image"].to(device, non_blocking=True)
                batch_labels = batch["label"].numpy()
                image_paths = batch.get("image_path", [f"sample_{batch_idx}_{i}" for i in range(len(images))])

            # Get model output with anomaly maps
            output = model(images)

            # Extract anomaly maps
            if hasattr(output, "anomaly_map") and output.anomaly_map is not None:
                anomaly_maps = output.anomaly_map.cpu().numpy()
            elif isinstance(output, dict) and "anomaly_map" in output:
                anomaly_maps = output["anomaly_map"].cpu().numpy()
            else:
                logger.warning("No anomaly map available in model output")
                return

            # Get prediction scores
            if hasattr(output, "pred_score") and output.pred_score is not None:
                pred_scores = output.pred_score.cpu().numpy()
            elif hasattr(output, "anomaly_map") and output.anomaly_map is not None:
                pred_scores = output.anomaly_map.flatten(1).max(dim=1)[0].cpu().numpy()
            else:
                pred_scores = anomaly_maps.reshape(len(anomaly_maps), -1).max(axis=1)

            # Convert images back to numpy for visualization
            images_np = images.cpu().numpy()

            # Denormalize images (ImageNet normalization)
            mean = np.array(IMAGENET_MEAN).reshape(1, 3, 1, 1)
            std = np.array(IMAGENET_STD).reshape(1, 3, 1, 1)
            images_np = images_np * std + mean
            images_np = np.clip(images_np, 0, 1)

            # Process each sample in batch
            for i in range(len(images)):
                label = batch_labels[i]
                is_fault = label == 1

                # Skip if we have enough samples of this type
                if is_fault and fault_count >= num_samples:
                    continue
                if not is_fault and good_count >= num_samples:
                    continue
                if is_fault and not save_fault:
                    continue
                if not is_fault and not save_good:
                    continue

                # Get sample data
                img = images_np[i].transpose(1, 2, 0)  # CHW -> HWC
                amap = anomaly_maps[i]
                if amap.ndim == 3:
                    amap = amap.squeeze(0)  # Remove channel dim if present
                score = pred_scores[i] if isinstance(pred_scores, np.ndarray) else float(pred_scores)

                # Get image name
                if isinstance(image_paths[i], Path):
                    img_name = image_paths[i].stem
                else:
                    img_name = Path(str(image_paths[i])).stem

                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Original image
                axes[0].imshow(img)
                axes[0].set_title(f"Original\n{img_name}")
                axes[0].axis('off')

                # Anomaly map
                im = axes[1].imshow(amap, cmap='hot', vmin=0, vmax=1)
                axes[1].set_title(f"Anomaly Map\nScore: {score:.4f}")
                axes[1].axis('off')
                plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

                # Overlay
                axes[2].imshow(img)
                axes[2].imshow(amap, cmap='hot', alpha=0.5, vmin=0, vmax=1)
                axes[2].set_title(f"Overlay\nLabel: {'Fault' if is_fault else 'Good'}")
                axes[2].axis('off')

                # Save
                label_str = "fault" if is_fault else "good"
                save_path = vis_dir / f"{label_str}_{img_name}_score{score:.4f}.png"
                fig.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close(fig)

                if is_fault:
                    fault_count += 1
                else:
                    good_count += 1

    logger.info(f"  Saved {good_count} good + {fault_count} fault anomaly maps to {vis_dir}")


def visualize_score_distribution(
    labels: list[int],
    scores: list[float],
    domain: str,
    output_dir: Path,
) -> None:
    """Visualize anomaly score distribution for a domain."""
    good_scores = [s for l, s in zip(labels, scores) if l == 0]
    fault_scores = [s for l, s in zip(labels, scores) if l == 1]

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = 50
    if good_scores:
        ax.hist(good_scores, bins=bins, alpha=0.6, label=f'Good (n={len(good_scores)})',
                color='green', density=True)
    if fault_scores:
        ax.hist(fault_scores, bins=bins, alpha=0.6, label=f'Fault (n={len(fault_scores)})',
                color='red', density=True)

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

    vis_dir = output_dir / "visualizations" / domain
    vis_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(vis_dir / "score_distribution.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"  Saved score distribution plot to {vis_dir / 'score_distribution.png'}")


def visualize_all_domains_score_comparison(
    domain_results: dict[str, tuple[list[int], list[float]]],
    output_dir: Path,
) -> None:
    """Create 2x2 subplot comparing score distributions across all domains."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    domains = sorted(domain_results.keys())

    for idx, domain in enumerate(domains):
        if idx >= 4:
            break
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
    """Export per-domain prediction scores to CSV file."""
    import csv

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

    if len(image_paths) != len(labels) or len(image_paths) != len(scores):
        logger.warning(f"Length mismatch: paths={len(image_paths)}, labels={len(labels)}, scores={len(scores)}")
        min_len = min(len(image_paths), len(labels), len(scores))
        image_paths = image_paths[:min_len]
        labels = labels[:min_len]
        scores = scores[:min_len]

    csv_dir = output_dir / "scores"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{domain}_scores.csv"

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
    """Compute AUROC with edge case handling."""
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
    """Compute comprehensive evaluation metrics."""
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

    if len(set(labels)) < 2:
        logger.warning("Cannot compute metrics: only one class present")
        return metrics

    try:
        metrics["auroc"] = roc_auc_score(labels_arr, scores_arr)
        fpr, tpr, thresholds = roc_curve(labels_arr, scores_arr)

        idx_1pct = np.argmin(np.abs(fpr - 0.01))
        metrics["tpr_at_fpr_1pct"] = float(tpr[idx_1pct])

        idx_5pct = np.argmin(np.abs(fpr - 0.05))
        metrics["tpr_at_fpr_5pct"] = float(tpr[idx_5pct])

        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        metrics["optimal_threshold"] = float(optimal_threshold)

        predictions = (scores_arr >= optimal_threshold).astype(int)

        metrics["precision"] = float(precision_score(labels_arr, predictions, zero_division=0))
        metrics["recall"] = float(recall_score(labels_arr, predictions, zero_division=0))
        metrics["f1_score"] = float(f1_score(labels_arr, predictions, zero_division=0))
        metrics["accuracy"] = float(accuracy_score(labels_arr, predictions))

    except Exception as e:
        logger.error(f"Failed to compute comprehensive metrics: {e}")

    return metrics


def format_metrics_table(domain_metrics: dict[str, dict[str, float]]) -> str:
    """Format metrics as a readable table string."""
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
# Callbacks
# =============================================================================

class ValidationAUROCPerDomain(Callback):
    """Callback to track validation AUROC per domain during training."""

    def __init__(self, test_dataloaders: dict[str, DataLoader]) -> None:
        super().__init__()
        self.test_dataloaders = test_dataloaders
        self.history: list[dict[str, Any]] = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.global_step % VALIDATION_CHECK_EVERY_N_STEPS != 0:
            return

        pl_module.eval()
        device = pl_module.device

        domain_results: dict[str, float] = {}
        all_labels: list[int] = []
        all_scores: list[float] = []

        with torch.no_grad():
            for domain, dataloader in self.test_dataloaders.items():
                domain_labels, domain_scores = evaluate_domain(
                    pl_module, dataloader, device
                )

                auroc = compute_auroc_safe(domain_labels, domain_scores)
                domain_results[domain] = auroc
                all_labels.extend(domain_labels)
                all_scores.extend(domain_scores)

                logger.info(f"Step {trainer.global_step} - {domain}: AUROC = {format_auroc(auroc)}")

        overall_auroc = compute_auroc_safe(all_labels, all_scores)
        logger.info(f"Step {trainer.global_step} - Overall: AUROC = {format_auroc(overall_auroc)}")

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
# Experiment Functions
# =============================================================================

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
    elem_drop: float = 0.1,
    row_p: float = 0.2,
    seg_len: int = 2,
    seg_drop_p: float = 0.6,
    enable_segment_dropout: bool = True,
    use_topk: bool = False,
    q_percent: float = 2.0,
) -> dict[str, Any]:
    """Run multi-class unified training experiment with DinomalyHorizontal.

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
        elem_drop: Element-wise dropout probability.
        row_p: Row selection probability for segment dropout.
        seg_len: Length of consecutive tokens to drop.
        seg_drop_p: Drop probability within segments.
        enable_segment_dropout: Whether to enable segment dropout.
        use_topk: Whether to use TopK Loss (v3.1).
        q_percent: TopK q percent (default: 2.0).

    Returns:
        Dictionary containing experiment results.
    """
    set_seed(seed)

    # Get config name for logging (use DinomalyHorizontalTopK.get_config_name if using TopK)
    if use_topk:
        config_name = DinomalyHorizontalTopK.get_config_name(
            elem_drop, enable_segment_dropout, row_p, seg_len, use_topk=True
        )
    else:
        config_name = DinomalyHorizontal.get_config_name(elem_drop, enable_segment_dropout)

    logger.info("=" * 60)
    if use_topk:
        logger.info("MULTI-CLASS UNIFIED TRAINING (Horizontal + TopK Loss v3.1)")
    else:
        logger.info("MULTI-CLASS UNIFIED TRAINING (Horizontal Segment Dropout)")
    logger.info("=" * 60)
    logger.info(f"Config: {config_name}")
    logger.info(f"Seed: {seed}")
    logger.info(f"elem_drop: {elem_drop}")
    logger.info(f"Segment dropout: {'enabled' if enable_segment_dropout else 'disabled'}")
    if enable_segment_dropout:
        logger.info(f"  row_p: {row_p}, seg_len: {seg_len}, seg_drop_p: {seg_drop_p}")
    if use_topk:
        logger.info(f"TopK Loss: enabled (q={q_percent}%)")
    logger.info(f"Domains: {domains}")
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Encoder: {encoder_name}")
    logger.info("Using AllDomainsHDMAPDataModule (tifffile-based TIFF loading)")

    experiment_dir = output_dir / "multiclass_unified"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Create per-domain test dataloaders using HDMAPDataset
    test_dataloaders: dict[str, DataLoader] = {}
    for domain in domains:
        test_dataset = HDMAPDataset(
            root=data_root,
            domain=domain,
            split="test",
            target_size=None,  # Let PreProcessor handle resize
        )
        test_dataloaders[domain] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=test_dataset.collate_fn,
        )

    # Setup evaluator
    val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
    test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
    evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])

    # Create model (DinomalyHorizontal or DinomalyHorizontalTopK)
    if use_topk:
        # v3.1: TopK Loss + Horizontal Dropout
        model = DinomalyHorizontalTopK(
            encoder_name=encoder_name,
            elem_drop=elem_drop,
            row_p=row_p,
            seg_len=seg_len,
            seg_drop_p=seg_drop_p,
            enable_segment_dropout=enable_segment_dropout,
            q_percent=q_percent,
            q_schedule=True,
            warmup_steps=200,
            evaluator=evaluator,
            pre_processor=True,
            visualizer=False,
        )
    else:
        # v3.0: Horizontal Dropout only
        model = DinomalyHorizontal(
            encoder_name=encoder_name,
            elem_drop=elem_drop,
            row_p=row_p,
            seg_len=seg_len,
            seg_drop_p=seg_drop_p,
            enable_segment_dropout=enable_segment_dropout,
            evaluator=evaluator,
            pre_processor=True,
            visualizer=False,
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

    # Create unified datamodule
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

    if not test_results:
        logger.warning("Test failed to produce results")
        overall_auroc = 0.0
    else:
        overall_auroc = test_results[0].get("test_image_AUROC", 0.0)

    logger.info(f"Overall Test AUROC: {format_auroc(overall_auroc)}")

    # Evaluate per-domain (final evaluation)
    logger.info("\nPer-Domain Evaluation (Final):")

    device = torch.device(f"cuda:{gpu_id}")
    model = model.to(device)
    model.eval()

    domain_metrics: dict[str, dict[str, float]] = {}
    domain_results_for_viz: dict[str, tuple[list[int], list[float]]] = {}
    all_labels: list[int] = []
    all_scores: list[float] = []

    for domain in domains:
        labels, scores = evaluate_domain(model, test_dataloaders[domain], device)
        metrics = compute_comprehensive_metrics(labels, scores)
        domain_metrics[domain] = metrics
        domain_results_for_viz[domain] = (labels, scores)
        all_labels.extend(labels)
        all_scores.extend(scores)
        logger.info(f"  {domain}: AUROC={metrics['auroc']*100:.2f}%, TPR@1%={metrics['tpr_at_fpr_1pct']*100:.2f}%, TPR@5%={metrics['tpr_at_fpr_5pct']*100:.2f}%")

        visualize_score_distribution(labels, scores, domain, experiment_dir)
        visualize_anomaly_maps(model, test_dataloaders[domain], device, domain, experiment_dir, num_samples=10)
        export_domain_scores_to_csv(test_dataloaders[domain], labels, scores, domain, experiment_dir)

    visualize_all_domains_score_comparison(domain_results_for_viz, experiment_dir)

    overall_metrics = compute_comprehensive_metrics(all_labels, all_scores)

    metric_names = ["auroc", "tpr_at_fpr_1pct", "tpr_at_fpr_5pct", "precision", "recall", "f1_score", "accuracy"]
    mean_metrics = {
        name: safe_mean([domain_metrics[d][name] for d in domains])
        for name in metric_names
    }

    logger.info(f"\n  Per-Domain Mean AUROC: {mean_metrics['auroc']*100:.2f}%")
    logger.info(f"  Overall (Pooled) AUROC: {overall_metrics['auroc']*100:.2f}%")

    logger.info("\n" + format_metrics_table(domain_metrics))
    logger.info(f"\n{'Mean':<10} {mean_metrics['auroc']*100:>7.2f}% {mean_metrics['tpr_at_fpr_1pct']*100:>7.2f}% "
                f"{mean_metrics['tpr_at_fpr_5pct']*100:>7.2f}% {mean_metrics['precision']*100:>7.2f}% "
                f"{mean_metrics['recall']*100:>7.2f}% {mean_metrics['f1_score']*100:>7.2f}% {mean_metrics['accuracy']*100:>7.2f}%")

    # Save results
    method_name = "Horizontal_TopK" if use_topk else "Horizontal_Segment_Dropout"
    results: dict[str, Any] = {
        "experiment_type": "multiclass_unified",
        "method": method_name,
        "config_name": config_name,
        "elem_drop": elem_drop,
        "enable_segment_dropout": enable_segment_dropout,
        "row_p": row_p if enable_segment_dropout else None,
        "seg_len": seg_len if enable_segment_dropout else None,
        "seg_drop_p": seg_drop_p if enable_segment_dropout else None,
        "use_topk": use_topk,
        "q_percent": q_percent if use_topk else None,
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
        "data_loading": "tifffile (unified)",
    }

    with open(experiment_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {experiment_dir}")

    return results


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Dinomaly Horizontal Segment Dropout for HDMAP")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["multiclass", "singleclass"],
        default="multiclass",
        help="Training mode: multiclass (unified) or singleclass (per-domain)",
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
        help=f"Maximum training steps (default: {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
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
    # Horizontal Segment Dropout parameters
    parser.add_argument(
        "--elem-p",
        type=float,
        default=0.1,
        help="Element-wise dropout probability (default: 0.1)",
    )
    parser.add_argument(
        "--row-p",
        type=float,
        default=0.2,
        help="Row selection probability for segment dropout (default: 0.2)",
    )
    parser.add_argument(
        "--seg-len",
        type=int,
        default=2,
        help="Length of consecutive tokens to drop (default: 2)",
    )
    parser.add_argument(
        "--seg-drop-p",
        type=float,
        default=0.6,
        help="Drop probability within segments (default: 0.6)",
    )
    parser.add_argument(
        "--disable-segment",
        action="store_true",
        help="Disable segment dropout (for Config A or B)",
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
        default="results/dinomaly_horizontal",
        help="Result directory (default: results/dinomaly_horizontal)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="dinov2reg_vit_base_14",
        choices=["dinov2reg_vit_small_14", "dinov2reg_vit_base_14", "dinov2reg_vit_large_14"],
        help="DINOv2 encoder variant",
    )
    # v3.1: TopK Loss integration
    parser.add_argument(
        "--use-topk",
        action="store_true",
        help="Use TopK Loss with Horizontal Dropout (v3.1)",
    )
    parser.add_argument(
        "--q-percent",
        type=float,
        default=2.0,
        help="TopK q percent (default: 2.0, best from v2 experiments)",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    for domain in args.domains:
        validate_domain(domain)

    # Determine config name for folder naming
    enable_segment = not args.disable_segment
    if args.use_topk:
        config_name = DinomalyHorizontalTopK.get_config_name(
            args.elem_p, enable_segment, args.row_p, args.seg_len, use_topk=True
        )
    else:
        config_name = DinomalyHorizontal.get_config_name(args.elem_p, enable_segment)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.result_dir) / f"{timestamp}_{config_name}_seed{args.seed}"
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
        "elem_drop": args.elem_p,
        "row_p": args.row_p,
        "seg_len": args.seg_len,
        "seg_drop_p": args.seg_drop_p,
        "enable_segment_dropout": enable_segment,
        "use_topk": args.use_topk,
        "q_percent": args.q_percent,
        "config_name": config_name,
        "timestamp": timestamp,
    }
    with open(output_dir / "experiment_settings.json", "w") as f:
        json.dump(settings, f, indent=2)

    logger.info(f"Config: {config_name}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output directory: {output_dir}")

    results: dict[str, Any] = {}

    if args.mode == "multiclass":
        results["multiclass"] = run_multiclass_experiment(
            data_root=args.data_root,
            domains=args.domains,
            max_steps=args.max_steps,
            gpu_id=args.gpu,
            output_dir=output_dir,
            batch_size=args.batch_size,
            encoder_name=args.encoder,
            seed=args.seed,
            elem_drop=args.elem_p,
            row_p=args.row_p,
            seg_len=args.seg_len,
            seg_drop_p=args.seg_drop_p,
            enable_segment_dropout=enable_segment,
            use_topk=args.use_topk,
            q_percent=args.q_percent,
        )

    # Save final summary
    with open(output_dir / "final_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    logger.info(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
