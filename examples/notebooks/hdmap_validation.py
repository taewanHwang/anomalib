"""HDMAP Validation Script for UniNet and Dinomaly.

Purpose: Compare UniNet and Dinomaly (DINOv2) performance on HDMAP dataset.
Dataset: HDMAP PNG (256x256, RGB, 4 domains)
         HDMAP FFT (256x256, 2D FFT magnitude spectrum)

Usage:
    # UniNet on domain_A (original PNG)
    python hdmap_validation.py --model uninet --domain domain_A --gpu 0

    # Dinomaly on domain_A (original PNG)
    python hdmap_validation.py --model dinomaly --domain domain_A --gpu 0

    # Dinomaly on domain_A (2D FFT)
    python hdmap_validation.py --model dinomaly --domain domain_A --dataset fft --gpu 0

    # All domains with Dinomaly Large on FFT
    python hdmap_validation.py --model dinomaly --domain all --dataset fft --encoder dinov2reg_vit_large_14 --gpu 0

    # Quick test (fewer epochs/steps)
    python hdmap_validation.py --model uninet --domain domain_A --epochs 10 --gpu 0
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.metrics import AUROC, Evaluator
from anomalib.models import Dinomaly, UniNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# All HDMAP domains
ALL_DOMAINS = ["domain_A", "domain_B", "domain_C", "domain_D"]

# Dataset paths
HDMAP_DATASETS = {
    "png": "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png",
    "fft": "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png_2dfft",  # FFT Magnitude
    "fft_phase": "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png_2dfft_phase",  # FFT Phase
}
HDMAP_PNG_ROOT = HDMAP_DATASETS["png"]  # Default

# Model settings
UNINET_SETTINGS = {
    "student_backbone": "wide_resnet50_2",
    "teacher_backbone": "wide_resnet50_2",
    "temperature": 2.0,
    "learning_rate": 0.005,
    "weight_decay": 1e-5,
    "warmup_epochs": 0,
    "batch_size": 8,
    "max_epochs": 100,
    "check_val_every_n_epoch": 10,
}

DINOMALY_SETTINGS = {
    "encoder_name": "dinov2reg_vit_base_14",
    "bottleneck_dropout": 0.2,
    "decoder_depth": 8,
    "batch_size": 8,
    "max_steps": 5000,
    "check_val_every_n_epoch": 1,
}


def plot_anomaly_score_histogram(
    model,
    datamodule,
    experiment_dir: Path,
    model_type: str,
    domain: str,
):
    """Plot anomaly score distribution histogram for test set.

    Args:
        model: Trained model
        datamodule: Data module with test data
        experiment_dir: Directory to save the plot
        model_type: Model type name
        domain: Domain name
    """
    logger.info(f"Generating anomaly score histogram for {model_type} on {domain}...")

    model.eval()
    device = next(model.parameters()).device

    normal_scores = []
    abnormal_scores = []

    # Get test dataloader
    test_dataloader = datamodule.test_dataloader()

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Collecting scores"):
            # Move image to device
            images = batch.image.to(device)
            labels = batch.gt_label.cpu().numpy()

            # Get predictions from the torch model directly (not the lightning module)
            output = model.model(images)

            # Extract anomaly scores based on output type
            if hasattr(output, 'anomaly_map') and output.anomaly_map is not None:
                # Compute image-level score from anomaly map (max pooling)
                anomaly_map = output.anomaly_map
                scores = anomaly_map.amax(dim=(1, 2, 3)).cpu().numpy()
            elif hasattr(output, 'pred_score') and output.pred_score is not None:
                scores = output.pred_score.cpu().numpy()
            else:
                # Fallback for tensor output (e.g., raw anomaly maps)
                if isinstance(output, torch.Tensor):
                    scores = output.amax(dim=(1, 2, 3)).cpu().numpy()
                else:
                    raise ValueError(f"Cannot extract anomaly scores from output: {type(output)}")

            # Separate normal and abnormal scores
            for score, label in zip(scores, labels):
                if label == 0:  # Normal
                    normal_scores.append(score)
                else:  # Abnormal
                    abnormal_scores.append(score)

    normal_scores = np.array(normal_scores)
    abnormal_scores = np.array(abnormal_scores)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Overlapping histogram
    ax1 = axes[0]

    # Determine bin range
    all_scores = np.concatenate([normal_scores, abnormal_scores])
    bin_min, bin_max = all_scores.min(), all_scores.max()
    bins = np.linspace(bin_min, bin_max, 50)

    ax1.hist(normal_scores, bins=bins, alpha=0.7, label=f'Normal (n={len(normal_scores)})',
             color='green', edgecolor='darkgreen')
    ax1.hist(abnormal_scores, bins=bins, alpha=0.7, label=f'Abnormal (n={len(abnormal_scores)})',
             color='red', edgecolor='darkred')

    ax1.set_xlabel('Anomaly Score', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'{model_type.upper()} - {domain}\nAnomaly Score Distribution', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = (
        f"Normal: mean={normal_scores.mean():.4f}, std={normal_scores.std():.4f}\n"
        f"Abnormal: mean={abnormal_scores.mean():.4f}, std={abnormal_scores.std():.4f}\n"
        f"Separation: {abnormal_scores.mean() - normal_scores.mean():.4f}"
    )
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Box plot comparison
    ax2 = axes[1]

    box_data = [normal_scores, abnormal_scores]
    bp = ax2.boxplot(box_data, labels=['Normal', 'Abnormal'], patch_artist=True)

    # Color the boxes
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')

    ax2.set_ylabel('Anomaly Score', fontsize=12)
    ax2.set_title(f'{model_type.upper()} - {domain}\nScore Distribution (Box Plot)', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Add median values
    for i, (data, label) in enumerate(zip(box_data, ['Normal', 'Abnormal'])):
        median = np.median(data)
        ax2.annotate(f'median: {median:.4f}',
                     xy=(i + 1, median),
                     xytext=(i + 1.3, median),
                     fontsize=9)

    plt.tight_layout()

    # Save figure
    histogram_path = experiment_dir / f"anomaly_score_histogram_{domain}.png"
    plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Histogram saved to: {histogram_path}")

    # Calculate AUROC directly from collected scores
    all_scores = np.concatenate([normal_scores, abnormal_scores])
    all_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(abnormal_scores))])
    calculated_auroc = float(roc_auc_score(all_labels, all_scores))
    logger.info(f"Calculated AUROC from scores: {calculated_auroc:.4f}")

    # Also save score statistics to JSON
    stats = {
        "domain": domain,
        "model": model_type,
        "calculated_auroc": calculated_auroc,  # Directly calculated AUROC
        "normal": {
            "count": len(normal_scores),
            "mean": float(normal_scores.mean()),
            "std": float(normal_scores.std()),
            "min": float(normal_scores.min()),
            "max": float(normal_scores.max()),
            "median": float(np.median(normal_scores)),
        },
        "abnormal": {
            "count": len(abnormal_scores),
            "mean": float(abnormal_scores.mean()),
            "std": float(abnormal_scores.std()),
            "min": float(abnormal_scores.min()),
            "max": float(abnormal_scores.max()),
            "median": float(np.median(abnormal_scores)),
        },
        "separation": float(abnormal_scores.mean() - normal_scores.mean()),
    }

    stats_path = experiment_dir / f"score_statistics_{domain}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Score statistics saved to: {stats_path}")

    return stats


def create_model(model_type: str, settings: dict, evaluator: Evaluator):
    """Create model based on type."""
    if model_type == "uninet":
        return UniNet(
            student_backbone=settings["student_backbone"],
            teacher_backbone=settings["teacher_backbone"],
            temperature=settings["temperature"],
            learning_rate=settings["learning_rate"],
            weight_decay=settings["weight_decay"],
            warmup_epochs=settings["warmup_epochs"],
            evaluator=evaluator,
            pre_processor=True,
        )
    elif model_type == "dinomaly":
        return Dinomaly(
            encoder_name=settings["encoder_name"],
            bottleneck_dropout=settings["bottleneck_dropout"],
            decoder_depth=settings["decoder_depth"],
            evaluator=evaluator,
            pre_processor=True,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_and_evaluate(
    model_type: str,
    domain: str,
    output_dir: Path,
    settings: dict,
    gpu_id: int = 0,
    dataset_root: str = None,
    dataset_type: str = "png",
) -> dict:
    """Train model on a single HDMAP domain and evaluate.

    Args:
        model_type: 'uninet' or 'dinomaly'
        domain: HDMAP domain name (domain_A, domain_B, etc.)
        output_dir: Directory to save results
        settings: Training settings dict
        gpu_id: GPU device ID to use
        dataset_root: Root path for the dataset (defaults to HDMAP_PNG_ROOT)
        dataset_type: Dataset type for naming ('png' or 'fft')

    Returns:
        Dictionary with evaluation results
    """
    if dataset_root is None:
        dataset_root = HDMAP_PNG_ROOT

    logger.info(f"Starting experiment: {model_type} on {domain} ({dataset_type} dataset)")

    # Create output directory
    experiment_dir = output_dir / f"{model_type}_{domain}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Setup data module using Folder
    domain_path = Path(dataset_root) / domain

    datamodule = Folder(
        name=f"HDMAP_{domain}",
        root=domain_path,
        normal_dir="train/good",
        abnormal_dir="test/fault",
        normal_test_dir="test/good",
        train_batch_size=settings["batch_size"],
        eval_batch_size=settings["batch_size"],
        num_workers=8,
        seed=42,
        val_split_mode="from_test",
        val_split_ratio=0.1,  # 10%만 validation, 90%는 test (기본값 0.5 → 50%)
    )

    # Setup evaluator with AUROC metrics
    val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
    test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
    evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])

    # Create model
    model = create_model(model_type, settings, evaluator)

    # Setup callbacks based on model type
    if model_type == "uninet":
        callbacks = [
            EarlyStopping(
                monitor="val_image_AUROC",
                mode="max",
                patience=20,
            ),
            ModelCheckpoint(
                dirpath=experiment_dir / "checkpoints",
                filename="best-{epoch}-{val_image_AUROC:.4f}",
                monitor="val_image_AUROC",
                mode="max",
                save_top_k=1,
                save_weights_only=True,
            ),
        ]
        engine = Engine(
            max_epochs=settings["max_epochs"],
            accelerator="gpu",
            devices=[gpu_id],
            check_val_every_n_epoch=settings["check_val_every_n_epoch"],
            callbacks=callbacks,
            default_root_dir=str(experiment_dir),
        )
    else:  # dinomaly - uses val_loss, no early stopping (max_steps based)
        callbacks = [
            ModelCheckpoint(
                dirpath=experiment_dir / "checkpoints",
                filename="best-{step}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_weights_only=True,
            ),
        ]
        engine = Engine(
            max_steps=settings["max_steps"],
            accelerator="gpu",
            devices=[gpu_id],
            check_val_every_n_epoch=settings["check_val_every_n_epoch"],
            callbacks=callbacks,
            default_root_dir=str(experiment_dir),
            gradient_clip_val=0.1,
            num_sanity_val_steps=0,
        )

    # Train
    logger.info(f"Training {model_type} on {domain}...")
    engine.fit(model=model, datamodule=datamodule)

    # Test
    logger.info(f"Testing {model_type} on {domain}...")
    test_results = engine.test(model=model, datamodule=datamodule)

    # Plot anomaly score histogram
    score_stats = plot_anomaly_score_histogram(
        model=model,
        datamodule=datamodule,
        experiment_dir=experiment_dir,
        model_type=model_type,
        domain=domain,
    )

    # Extract results - use calculated AUROC from score_stats (more reliable)
    engine_results = test_results[0] if test_results else {}

    # Override buggy AUROC with directly calculated one
    if "calculated_auroc" in score_stats:
        engine_results["test_image_AUROC"] = score_stats["calculated_auroc"]
        logger.info(f"Using calculated AUROC: {score_stats['calculated_auroc']:.4f}")

    results = {
        "model": model_type,
        "domain": domain,
        "dataset_type": dataset_type,
        "test_results": engine_results,
        "score_statistics": score_stats,
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in settings.items()},
    }

    # Save results
    results_file = experiment_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results for {model_type} on {domain}: {test_results}")
    return results


def main():
    parser = argparse.ArgumentParser(description="HDMAP Validation (UniNet/Dinomaly)")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["uninet", "dinomaly"],
        help="Model to use",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="domain_A",
        choices=ALL_DOMAINS + ["all"],
        help="HDMAP domain (use 'all' for all domains)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/hdmap_validation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID",
    )
    # UniNet specific
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Max epochs (UniNet)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Temperature (UniNet)",
    )
    # Dinomaly specific
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Max steps (Dinomaly)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="dinov2reg_vit_base_14",
        choices=["dinov2reg_vit_small_14", "dinov2reg_vit_base_14", "dinov2reg_vit_large_14"],
        help="DINOv2 encoder (Dinomaly)",
    )
    # Common
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="png",
        choices=["png", "fft", "fft_phase"],
        help="Dataset type: 'png' (original), 'fft' (FFT magnitude), or 'fft_phase' (FFT phase)",
    )
    args = parser.parse_args()

    # Expand 'all' to all domains
    domains = ALL_DOMAINS if args.domain == "all" else [args.domain]

    # Get dataset root based on dataset type
    dataset_root = HDMAP_DATASETS[args.dataset]
    logger.info(f"Using dataset: {args.dataset} ({dataset_root})")

    # Setup output directory with timestamp and dataset type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{timestamp}_{args.dataset}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup settings based on model type
    if args.model == "uninet":
        settings = UNINET_SETTINGS.copy()
        settings["max_epochs"] = args.epochs
        settings["temperature"] = args.temperature
        settings["batch_size"] = args.batch_size
    else:  # dinomaly
        settings = DINOMALY_SETTINGS.copy()
        settings["max_steps"] = args.max_steps
        settings["encoder_name"] = args.encoder
        settings["batch_size"] = args.batch_size

    # Save experiment settings
    with open(output_dir / "experiment_settings.json", "w") as f:
        json.dump(
            {
                "model": args.model,
                "domains": domains,
                "dataset_type": args.dataset,
                "dataset_root": dataset_root,
                "settings": settings,
                "timestamp": timestamp,
            },
            f,
            indent=2,
        )

    # Run experiments
    all_results = []
    for domain in domains:
        try:
            results = train_and_evaluate(
                model_type=args.model,
                domain=domain,
                output_dir=output_dir,
                settings=settings,
                gpu_id=args.gpu,
                dataset_root=dataset_root,
                dataset_type=args.dataset,
            )
            all_results.append(results)
        except Exception as e:
            logger.error(f"Failed to train {args.model} on {domain}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"model": args.model, "domain": domain, "error": str(e)})

    # Print summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: HDMAP Validation Results ({args.model.upper()} - {args.dataset.upper()} dataset)")
    print("=" * 70)

    for result in all_results:
        domain = result["domain"]
        model = result["model"]
        if "error" in result:
            print(f"{domain}: ERROR - {result['error']}")
        else:
            test_results = result.get("test_results", {})
            auroc = test_results.get("test_image_AUROC", "N/A")
            if isinstance(auroc, float):
                auroc_str = f"{auroc * 100:.2f}%"
            else:
                auroc_str = str(auroc)

            # Score statistics
            score_stats = result.get("score_statistics", {})
            if score_stats:
                separation = score_stats.get("separation", 0)
                normal_mean = score_stats.get("normal", {}).get("mean", 0)
                abnormal_mean = score_stats.get("abnormal", {}).get("mean", 0)
                print(f"{domain} ({model}): AUROC = {auroc_str} | "
                      f"Score separation = {separation:.4f} "
                      f"(normal: {normal_mean:.4f}, abnormal: {abnormal_mean:.4f})")
            else:
                print(f"{domain} ({model}): Image AUROC = {auroc_str}")

    print("=" * 70)

    # Save final summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"Summary file: {summary_file}")

    # Reference comparison
    print("\n" + "-" * 70)
    print("REFERENCE (MVTec-AD bottle):")
    print("-" * 70)
    print("UniNet: 99.92% AUROC")
    print("Dinomaly: 100.00% AUROC")
    print("-" * 70)


if __name__ == "__main__":
    main()
