"""
Dinomaly K/V Row-wise Dropout for HDMAP Dataset.

This script implements K/V Row-wise Dropout variant of Dinomaly (v4).
It applies V (Value) dropout directly in decoder attention layers,
addressing the key limitation of bottleneck dropout from v3.1.

Key insight: Bottleneck dropout is recovered by global attention.
K/V dropout removes reconstruction material itself, which cannot be recovered.

Safety mechanisms (all ablatable):
1. V-only masking: Preserve K for attention stability (--v-only/--no-v-only)
2. Head-wise dropout: Apply to subset of heads (--head-ratio)
3. Layer-wise scheduling: Apply to specific layers (--apply-layers)
4. Warmup schedule: Gradual dropout increase (--warmup-steps)
5. Row-internal segment: Drop segments within rows (--band-width, --row-p)

Ablation configurations:
    H. Primary: v_drop_p=0.05, head_ratio=0.5, layers=4-7, warmup=200
    I. Strong: v_drop_p=0.1
    J. All Heads: head_ratio=1.0
    K. All Layers: apply_layers=0-7
    L. No Warmup: warmup_steps=0
    M. K+V Dropout: --no-v-only

Usage:
    # Config H: Primary (recommended)
    python dinomaly_kv_dropout.py --seed 42 --gpu 0 \\
        --use-topk --q-percent 2.0 \\
        --v-drop-p 0.05 --row-p 0.1 --band-width 2 \\
        --head-ratio 0.5 --apply-layers 4 5 6 7 \\
        --warmup-steps 200 --v-only \\
        --result-dir results/dinomaly_kv_dropout

HDMAP Dataset Structure:
    datasets/HDMAP/1000_tiff_minmax/
    |-- domain_A/
    |   |-- train/good/
    |   `-- test/{good,fault}/
    |-- domain_B/
    |-- domain_C/
    `-- domain_D/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from anomalib.data.datamodules.image.all_domains_hdmap import AllDomainsHDMAPDataModule
from anomalib.data.datasets.image.hdmap import HDMAPDataset
from anomalib.engine import Engine
from anomalib.metrics import AUROC
from anomalib.metrics.evaluator import Evaluator
from anomalib.models.image.dinomaly_variants import DinomalyKVDropout

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


# =============================================================================
# Constants
# =============================================================================

DEFAULT_HDMAP_TIFF_ROOT = "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_tiff_minmax"
HDMAP_TIFF_ROOT = os.getenv("HDMAP_TIFF_ROOT", DEFAULT_HDMAP_TIFF_ROOT)

ALLOWED_DOMAINS = frozenset({"domain_A", "domain_B", "domain_C", "domain_D"})
DOMAINS = list(ALLOWED_DOMAINS)

DEFAULT_MAX_STEPS = 10000
DEFAULT_BATCH_SIZE = 16
DEFAULT_GRADIENT_CLIP_VAL = 0.1
VALIDATION_CHECK_EVERY_N_STEPS = 1000

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
    """Set random seed for reproducibility."""
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
    """Evaluate model on a single domain."""
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
    """Generate and save per-sample anomaly map visualizations."""
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

            output = model(images)

            if hasattr(output, "anomaly_map") and output.anomaly_map is not None:
                anomaly_maps = output.anomaly_map.cpu().numpy()
            elif isinstance(output, dict) and "anomaly_map" in output:
                anomaly_maps = output["anomaly_map"].cpu().numpy()
            else:
                logger.warning("No anomaly map available in model output")
                return

            if hasattr(output, "pred_score") and output.pred_score is not None:
                pred_scores = output.pred_score.cpu().numpy()
            elif hasattr(output, "anomaly_map") and output.anomaly_map is not None:
                pred_scores = output.anomaly_map.flatten(1).max(dim=1)[0].cpu().numpy()
            else:
                pred_scores = anomaly_maps.reshape(len(anomaly_maps), -1).max(axis=1)

            images_np = images.cpu().numpy()
            mean = np.array(IMAGENET_MEAN).reshape(1, 3, 1, 1)
            std = np.array(IMAGENET_STD).reshape(1, 3, 1, 1)
            images_np = images_np * std + mean
            images_np = np.clip(images_np, 0, 1)

            for i in range(len(images)):
                label = batch_labels[i]
                is_fault = label == 1

                if is_fault and fault_count >= num_samples:
                    continue
                if not is_fault and good_count >= num_samples:
                    continue
                if is_fault and not save_fault:
                    continue
                if not is_fault and not save_good:
                    continue

                img = images_np[i].transpose(1, 2, 0)
                amap = anomaly_maps[i]
                if amap.ndim == 3:
                    amap = amap.squeeze(0)
                score = pred_scores[i] if isinstance(pred_scores, np.ndarray) else float(pred_scores)

                if isinstance(image_paths[i], Path):
                    img_name = image_paths[i].stem
                else:
                    img_name = Path(str(image_paths[i])).stem

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                axes[0].imshow(img)
                axes[0].set_title(f"Original\n{img_name}")
                axes[0].axis('off')

                im = axes[1].imshow(amap, cmap='hot', vmin=0, vmax=1)
                axes[1].set_title(f"Anomaly Map\nScore: {score:.4f}")
                axes[1].axis('off')
                plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

                axes[2].imshow(img)
                axes[2].imshow(amap, cmap='hot', alpha=0.5, vmin=0, vmax=1)
                axes[2].set_title(f"Overlay\nLabel: {'Fault' if is_fault else 'Good'}")
                axes[2].axis('off')

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
    encoder_name: str = "dinov2reg_vit_base_14",
    seed: int = 42,
    # K/V Dropout parameters
    v_drop_p: float = 0.05,
    row_p: float = 0.1,
    band_width: int = 2,
    head_ratio: float = 0.5,
    v_only: bool = True,
    apply_layers: list[int] | None = None,
    warmup_steps: int = 200,
    # TopK Loss parameters
    use_topk: bool = True,
    q_percent: float = 2.0,
) -> dict[str, Any]:
    """Run multi-class unified training experiment with DinomalyKVDropout.

    Args:
        data_root: Path to dataset root.
        domains: List of domain names.
        max_steps: Maximum training steps.
        gpu_id: GPU device ID.
        output_dir: Output directory for results.
        batch_size: Training batch size.
        encoder_name: DINOv2 encoder variant.
        seed: Random seed.
        v_drop_p: V dropout probability.
        row_p: Row selection probability.
        band_width: Segment width.
        head_ratio: Ratio of heads to apply dropout.
        v_only: Only dropout V (not K).
        apply_layers: Decoder layers to apply V dropout.
        warmup_steps: Warmup steps.
        use_topk: Whether to use TopK Loss.
        q_percent: TopK q percent.

    Returns:
        Dictionary containing experiment results.
    """
    set_seed(seed)

    if apply_layers is None:
        apply_layers = [4, 5, 6, 7]

    config_name = DinomalyKVDropout.get_config_name(
        v_drop_p, head_ratio, apply_layers, v_only
    )

    logger.info("=" * 60)
    logger.info("MULTI-CLASS UNIFIED TRAINING (K/V Row-wise Dropout v4)")
    logger.info("=" * 60)
    logger.info(f"Config: {config_name}")
    logger.info(f"Seed: {seed}")
    logger.info(f"V Dropout: v_drop_p={v_drop_p}, row_p={row_p}, band_width={band_width}")
    logger.info(f"Head ratio: {head_ratio}, V-only: {v_only}")
    logger.info(f"Apply layers: {apply_layers}")
    logger.info(f"Warmup steps: {warmup_steps}")
    if use_topk:
        logger.info(f"TopK Loss: enabled (q={q_percent}%)")
    logger.info(f"Domains: {domains}")
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Encoder: {encoder_name}")

    experiment_dir = output_dir / "multiclass_unified"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Create per-domain test dataloaders
    test_dataloaders: dict[str, DataLoader] = {}
    for domain in domains:
        test_dataset = HDMAPDataset(
            root=data_root,
            domain=domain,
            split="test",
            target_size=None,
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

    # Create model
    model = DinomalyKVDropout(
        encoder_name=encoder_name,
        v_drop_p=v_drop_p,
        row_p=row_p,
        band_width=band_width,
        head_ratio=head_ratio,
        v_only=v_only,
        apply_layers=apply_layers,
        warmup_steps=warmup_steps,
        q_percent=q_percent if use_topk else 100.0,  # 100% = no TopK
        q_schedule=use_topk,
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

    # Create datamodule
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

    # Evaluate per-domain
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
    results: dict[str, Any] = {
        "experiment_type": "multiclass_unified",
        "method": "KV_Row_Dropout",
        "config_name": config_name,
        "v_drop_p": v_drop_p,
        "row_p": row_p,
        "band_width": band_width,
        "head_ratio": head_ratio,
        "v_only": v_only,
        "apply_layers": apply_layers,
        "warmup_steps": warmup_steps,
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
    parser = argparse.ArgumentParser(description="Dinomaly K/V Row-wise Dropout for HDMAP")
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
    # K/V Dropout parameters
    parser.add_argument(
        "--v-drop-p",
        type=float,
        default=0.05,
        help="V dropout probability (default: 0.05)",
    )
    parser.add_argument(
        "--row-p",
        type=float,
        default=0.1,
        help="Row selection probability (default: 0.1)",
    )
    parser.add_argument(
        "--band-width",
        type=int,
        default=2,
        help="Segment width within row (default: 2)",
    )
    parser.add_argument(
        "--head-ratio",
        type=float,
        default=0.5,
        help="Ratio of heads to apply dropout (default: 0.5)",
    )
    parser.add_argument(
        "--v-only",
        action="store_true",
        default=True,
        help="Only dropout V, preserve K (default: True)",
    )
    parser.add_argument(
        "--no-v-only",
        action="store_false",
        dest="v_only",
        help="Dropout both K and V",
    )
    parser.add_argument(
        "--apply-layers",
        type=int,
        nargs="+",
        default=[4, 5, 6, 7],
        help="Decoder layer indices to apply V dropout (default: 4 5 6 7)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=200,
        help="Warmup steps for dropout scheduling (default: 200)",
    )
    # TopK Loss parameters
    parser.add_argument(
        "--use-topk",
        action="store_true",
        default=True,
        help="Use TopK Loss (default: True)",
    )
    parser.add_argument(
        "--no-topk",
        action="store_false",
        dest="use_topk",
        help="Disable TopK Loss",
    )
    parser.add_argument(
        "--q-percent",
        type=float,
        default=2.0,
        help="TopK q percent (default: 2.0)",
    )
    # Paths
    parser.add_argument(
        "--data-root",
        type=str,
        default=HDMAP_TIFF_ROOT,
        help="Path to HDMAP dataset",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="results/dinomaly_kv_dropout",
        help="Result directory (default: results/dinomaly_kv_dropout)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="dinov2reg_vit_base_14",
        choices=["dinov2reg_vit_small_14", "dinov2reg_vit_base_14", "dinov2reg_vit_large_14"],
        help="DINOv2 encoder variant",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    for domain in args.domains:
        validate_domain(domain)

    config_name = DinomalyKVDropout.get_config_name(
        args.v_drop_p, args.head_ratio, args.apply_layers, args.v_only
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.result_dir) / f"{timestamp}_{config_name}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment settings
    settings = {
        "domains": args.domains,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "encoder": args.encoder,
        "data_root": args.data_root,
        "seed": args.seed,
        "v_drop_p": args.v_drop_p,
        "row_p": args.row_p,
        "band_width": args.band_width,
        "head_ratio": args.head_ratio,
        "v_only": args.v_only,
        "apply_layers": args.apply_layers,
        "warmup_steps": args.warmup_steps,
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

    results = run_multiclass_experiment(
        data_root=args.data_root,
        domains=args.domains,
        max_steps=args.max_steps,
        gpu_id=args.gpu,
        output_dir=output_dir,
        batch_size=args.batch_size,
        encoder_name=args.encoder,
        seed=args.seed,
        v_drop_p=args.v_drop_p,
        row_p=args.row_p,
        band_width=args.band_width,
        head_ratio=args.head_ratio,
        v_only=args.v_only,
        apply_layers=args.apply_layers,
        warmup_steps=args.warmup_steps,
        use_topk=args.use_topk,
        q_percent=args.q_percent,
    )

    # Save final summary
    with open(output_dir / "final_summary.json", "w") as f:
        json.dump({"multiclass": results}, f, indent=2, default=float)

    logger.info(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
