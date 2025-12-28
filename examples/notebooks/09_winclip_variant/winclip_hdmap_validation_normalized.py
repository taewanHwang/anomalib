"""WinCLIP HDMAP Validation Script with Per-Image Normalization.

Purpose: Evaluate WinCLIP model performance on HDMAP dataset with per-image
normalization to address Cold Start vs Warmed Up amplitude scale differences.

Background:
    - Cold Start images have ~1.4x lower mean intensity
    - Cold Start images have ~1.6x lower dynamic range
    - This causes Fault Cold and Normal Warm score distributions to overlap
    - Per-image normalization removes this scale dependency

Normalization Methods:
    - minmax: (x - min) / (max - min) -> [0, 1] range
    - robust: (x - p5) / (p95 - p5) -> robust to outliers

Usage:
    # Min-max normalization
    python winclip_hdmap_validation_normalized.py \
        --domains domain_C \
        --k-shots 0 \
        --normalize-method minmax \
        --gpu 0

    # Robust scaling (p5, p95)
    python winclip_hdmap_validation_normalized.py \
        --domains domain_C \
        --k-shots 0 \
        --normalize-method robust \
        --gpu 0
"""

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from anomalib.data.datamodules.image.hdmap import HDMAPDataModule
from anomalib.data.datasets.image.hdmap import HDMAPDataset
from anomalib.engine import Engine
from anomalib.models.image.winclip import WinClip

# Import custom transforms and visualizer
import sys
sys.path.insert(0, str(Path(__file__).parent))
from transforms.per_image_normalize import PerImageNormalize, normalize_batch
from custom_visualizer import FourColumnVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# All 4 HDMAP domains
ALL_DOMAINS = [
    "domain_A",
    "domain_B",
    "domain_C",
    "domain_D",
]

# Default k_shot modes to test
DEFAULT_K_SHOTS = [0, 1, 2, 4]


# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_score_distribution(
    labels: list[int],
    scores: list[float],
    domain: str,
    mode_name: str,
    output_dir: Path,
    normalize_method: str = None,
) -> None:
    """Visualize anomaly score distribution for a domain."""
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
    norm_str = f" [norm={normalize_method}]" if normalize_method else ""
    ax.set_title(f'{domain} ({mode_name}){norm_str} - Anomaly Score Distribution')
    ax.legend()

    # Save
    vis_dir = output_dir / "visualizations" / domain
    vis_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(vis_dir / "score_distribution.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"  Saved score distribution plot to {vis_dir / 'score_distribution.png'}")


def visualize_all_domains_score_comparison(
    domain_results: dict[str, tuple[list[int], list[float]]],
    mode_name: str,
    output_dir: Path,
    normalize_method: str = None,
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

    norm_str = f" [norm={normalize_method}]" if normalize_method else ""
    plt.suptitle(f'Per-Domain Anomaly Score Distributions ({mode_name}){norm_str}', fontsize=14)
    plt.tight_layout()

    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(vis_dir / f"all_domains_score_comparison_{mode_name.replace('-', '_')}.png",
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved all domains score comparison to {vis_dir}")


def export_domain_scores_to_csv(
    labels: list[int],
    scores: list[float],
    image_paths: list[str],
    domain: str,
    mode_name: str,
    output_dir: Path,
    normalize_method: str = None,
) -> Path:
    """Export per-domain prediction scores to CSV file."""
    csv_dir = output_dir / "scores"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{domain}_{mode_name.replace('-', '_')}_scores.csv"

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'image_path', 'gt_label', 'gt_label_str', 'pred_score',
                        'domain', 'mode', 'normalize_method'])

        for img_path, label, score in zip(image_paths, labels, scores):
            img_name = Path(img_path).stem if img_path else 'unknown'
            label_str = 'fault' if label == 1 else 'good'
            writer.writerow([img_name, img_path, label, label_str, f'{score:.6f}',
                            domain, mode_name, normalize_method or 'none'])

    logger.info(f"  Saved {len(labels)} scores to {csv_path}")
    return csv_path


def collect_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    normalize_method: str = None,
) -> tuple[list[int], list[float], list[str]]:
    """Collect predictions from model with optional per-image normalization.

    Args:
        model: WinCLIP model.
        dataloader: DataLoader for evaluation.
        device: Device to run inference on.
        normalize_method: Per-image normalization method ("minmax" or "robust").
                         None means no normalization.

    Returns:
        Tuple of (labels, scores, image_paths).
    """
    labels = []
    scores = []
    image_paths = []

    model = model.to(device)
    model.eval()

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
        for batch in dataloader:
            # Get batch data
            if hasattr(batch, 'image'):
                images = batch.image.to(device, non_blocking=True)
                batch_labels = batch.gt_label
                batch_paths = batch.image_path if hasattr(batch, 'image_path') else [None] * len(images)
            else:
                images = batch["image"].to(device, non_blocking=True)
                batch_labels = batch.get("label", batch.get("gt_label"))
                batch_paths = batch.get("image_path", [None] * len(images))

            # Apply per-image normalization if specified
            if normalize_method:
                images = normalize_batch(images, method=normalize_method)

            # Forward pass
            output = model(images)

            # Extract scores
            if hasattr(output, "pred_score") and output.pred_score is not None:
                batch_scores = output.pred_score.cpu()
            elif hasattr(output, "anomaly_map") and output.anomaly_map is not None:
                batch_scores = output.anomaly_map.flatten(1).max(dim=1)[0].cpu()
            elif isinstance(output, dict):
                if "pred_score" in output and output["pred_score"] is not None:
                    batch_scores = output["pred_score"].cpu()
                else:
                    batch_scores = torch.zeros(len(images))
            else:
                batch_scores = torch.zeros(len(images))

            # Collect results
            if batch_scores.dim() == 0:
                scores.append(float(batch_scores))
            else:
                scores.extend(batch_scores.numpy().tolist())

            if hasattr(batch_labels, 'numpy'):
                labels.extend(batch_labels.numpy().tolist())
            else:
                labels.extend(list(batch_labels))

            if isinstance(batch_paths, (list, tuple)):
                image_paths.extend([str(p) if p else '' for p in batch_paths])
            else:
                image_paths.extend([''] * len(images))

    return labels, scores, image_paths


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_domain(
    domain: str,
    k_shot: int,
    result_dir: Path,
    dataset_root: Path,
    gpu_id: int = 0,
    class_name: str = "industrial sensor data",
    image_size: int = 240,
    generate_visualizations: bool = True,
    normalize_method: str = None,
) -> dict:
    """Evaluate WinCLIP on a single HDMAP domain with per-image normalization.

    Args:
        domain: HDMAP domain name
        k_shot: Number of reference images (0 for zero-shot)
        result_dir: Directory to save results
        dataset_root: Root path to HDMAP dataset
        gpu_id: GPU device ID to use
        class_name: Class name for WinCLIP prompts
        image_size: Target image size for CLIP
        generate_visualizations: Whether to generate visualizations
        normalize_method: Per-image normalization method ("minmax" or "robust")

    Returns:
        Dictionary with evaluation results
    """
    mode_name = "zero-shot" if k_shot == 0 else f"{k_shot}-shot"
    norm_str = f" [norm={normalize_method}]" if normalize_method else ""
    logger.info(f"Evaluating {domain} with {mode_name} mode{norm_str}...")

    # Create output directory
    domain_dir = result_dir / domain / mode_name
    domain_dir.mkdir(parents=True, exist_ok=True)

    # Setup data module
    datamodule = HDMAPDataModule(
        root=str(dataset_root),
        domain=domain,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=8,
        target_size=(image_size, image_size),
        resize_method="resize",
    )

    # Create WinCLIP model
    model = WinClip(
        class_name=class_name,
        k_shot=k_shot,
        scales=(2, 3),
        visualizer=False,
    )

    # Create custom visualizer
    custom_visualizer = FourColumnVisualizer(
        field_size=(256, 256),
        alpha=0.5,
        output_dir=domain_dir / "images_4col",
    )

    # Create engine
    engine = Engine(
        accelerator="gpu",
        devices=[gpu_id],
        default_root_dir=str(domain_dir),
        callbacks=[custom_visualizer],
    )

    # Note: engine.test() uses default preprocessing without per-image normalization
    # We'll collect predictions separately with normalization
    logger.info(f"Testing {domain} ({mode_name}){norm_str}...")

    # Collect predictions with normalization
    labels, scores, image_paths = [], [], []
    if generate_visualizations or normalize_method:
        logger.info(f"Collecting predictions with normalization...")
        device = torch.device(f"cuda:{gpu_id}")

        # Create test dataloader
        test_dataset = HDMAPDataset(
            root=str(dataset_root),
            domain=domain,
            split="test",
            target_size=(image_size, image_size),
            resize_method="resize",
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            collate_fn=test_dataset.collate_fn,
        )

        # Setup model for inference
        if k_shot > 0:
            datamodule.setup()
            train_dataloader = datamodule.train_dataloader()
            ref_images = []
            for batch in train_dataloader:
                if hasattr(batch, 'image'):
                    ref_images.append(batch.image)
                else:
                    ref_images.append(batch["image"])
                if sum(img.shape[0] for img in ref_images) >= k_shot:
                    break
            ref_images = torch.cat(ref_images, dim=0)[:k_shot]

            # Apply normalization to reference images too
            if normalize_method:
                ref_images = normalize_batch(ref_images, method=normalize_method)

            model.model.to(device)
            model.model.setup(class_name, ref_images.to(device))
        else:
            model.model.setup(class_name, None)

        labels, scores, image_paths = collect_predictions(
            model, test_dataloader, device, normalize_method
        )

        # Generate visualizations
        visualize_score_distribution(
            labels, scores, domain, mode_name, result_dir, normalize_method
        )
        export_domain_scores_to_csv(
            labels, scores, image_paths, domain, mode_name, result_dir, normalize_method
        )

    # Compute AUROC
    from sklearn.metrics import roc_auc_score
    metrics = {}
    if labels and scores:
        try:
            auroc = roc_auc_score(labels, scores)
            metrics["image_AUROC"] = auroc
            logger.info(f"{domain} ({mode_name}){norm_str}: Image AUROC = {auroc:.4f} ({auroc*100:.2f}%)")
        except Exception as e:
            logger.warning(f"Could not compute AUROC: {e}")

    # Build results dict
    results = {
        "domain": domain,
        "k_shot": k_shot,
        "mode": mode_name,
        "class_name": class_name,
        "normalize_method": normalize_method,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
        "labels": labels,
        "scores": scores,
        "image_paths": image_paths,
    }

    # Save results
    results_to_save = {k: v for k, v in results.items() if k not in ["labels", "scores", "image_paths"]}
    results_file = domain_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results_to_save, f, indent=2)

    return results


def main():
    """Run WinCLIP validation on HDMAP with per-image normalization."""
    parser = argparse.ArgumentParser(
        description="WinCLIP HDMAP Validation with Per-Image Normalization"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["domain_C"],
        choices=ALL_DOMAINS + ["all"],
        help="Domains to test (use 'all' for all domains)",
    )
    parser.add_argument(
        "--k-shots",
        nargs="+",
        type=int,
        default=[0],
        help="k_shot values to test (default: 0)",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="./datasets/HDMAP/1000_tiff_minmax",
        help="Root path to HDMAP dataset",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="./results/winclip_hdmap_normalized",
        help="Directory to save results",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="industrial sensor data",
        help="Class name for WinCLIP prompts",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=240,
        help="Target image size for CLIP",
    )
    parser.add_argument(
        "--normalize-method",
        type=str,
        choices=["minmax", "robust", "robust_soft", "none"],
        default="minmax",
        help="Per-image normalization method (default: minmax). robust_soft uses p1/p99 for less noise amplification.",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip generating visualizations",
    )
    args = parser.parse_args()

    # Handle 'none' as no normalization
    normalize_method = None if args.normalize_method == "none" else args.normalize_method

    # Expand 'all' to all domains
    domains = ALL_DOMAINS if "all" in args.domains else args.domains

    # Resolve dataset root path
    anomalib_root = Path(__file__).parent.parent.parent.parent
    if args.dataset_root.startswith("./"):
        dataset_root = anomalib_root / args.dataset_root[2:]
    else:
        dataset_root = Path(args.dataset_root)

    # Setup result directory with timestamp and normalization method
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    norm_suffix = f"_{normalize_method}" if normalize_method else "_baseline"
    result_dir = Path(args.result_dir) / f"{timestamp}{norm_suffix}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment settings
    settings = {
        "domains": domains,
        "k_shots": args.k_shots,
        "dataset_root": str(dataset_root),
        "gpu": args.gpu,
        "class_name": args.class_name,
        "image_size": args.image_size,
        "normalize_method": normalize_method,
        "timestamp": timestamp,
    }
    with open(result_dir / "experiment_settings.json", "w") as f:
        json.dump(settings, f, indent=2)

    norm_str = f" with {normalize_method} normalization" if normalize_method else ""
    logger.info(f"Starting WinCLIP validation on HDMAP{norm_str}")
    logger.info(f"Domains: {domains}")
    logger.info(f"k_shot modes: {args.k_shots}")
    logger.info(f"Dataset root: {dataset_root}")
    logger.info(f"Class name: {args.class_name}")
    logger.info(f"Normalization: {normalize_method or 'none'}")
    logger.info(f"Results will be saved to: {result_dir}")

    # Run experiments
    all_results = []

    for k_shot in args.k_shots:
        mode_name = "zero-shot" if k_shot == 0 else f"{k_shot}-shot"
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {mode_name} experiments{norm_str}...")
        logger.info(f"{'='*60}")

        domain_results_for_viz = {}

        for domain in domains:
            try:
                results = evaluate_domain(
                    domain=domain,
                    k_shot=k_shot,
                    result_dir=result_dir,
                    dataset_root=dataset_root,
                    gpu_id=args.gpu,
                    class_name=args.class_name,
                    image_size=args.image_size,
                    generate_visualizations=not args.no_visualizations,
                    normalize_method=normalize_method,
                )
                all_results.append(results)

                if results.get("labels") and results.get("scores"):
                    domain_results_for_viz[domain] = (results["labels"], results["scores"])

            except Exception as e:
                logger.error(f"Failed to evaluate {domain} with k_shot={k_shot}: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "domain": domain,
                    "k_shot": k_shot,
                    "error": str(e),
                })

        # Generate all-domains comparison
        if domain_results_for_viz and not args.no_visualizations:
            visualize_all_domains_score_comparison(
                domain_results_for_viz, mode_name, result_dir, normalize_method
            )

    # Print summary
    print("\n" + "=" * 80)
    print(f"SUMMARY: WinCLIP HDMAP Validation Results{norm_str}")
    print("=" * 80)

    for k_shot in args.k_shots:
        mode_name = "Zero-shot" if k_shot == 0 else f"{k_shot}-shot"
        print(f"\n{mode_name} Results:")
        print("-" * 60)

        for result in all_results:
            if result.get("k_shot") == k_shot:
                domain = result["domain"]
                if "error" in result:
                    print(f"  {domain}: ERROR - {result['error']}")
                else:
                    metrics = result.get("metrics", {})
                    image_auroc = metrics.get("image_AUROC", "N/A")

                    if isinstance(image_auroc, (int, float)):
                        auroc_str = f"{image_auroc * 100:.2f}%"
                    else:
                        auroc_str = str(image_auroc)

                    print(f"  {domain}: Image AUROC = {auroc_str}")

    print("\n" + "=" * 80)

    # Save final summary
    summary_results = []
    for r in all_results:
        summary_r = {k: v for k, v in r.items() if k not in ["labels", "scores", "image_paths"]}
        summary_results.append(summary_r)

    summary_file = result_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_results, f, indent=2)

    print(f"\nResults saved to: {result_dir}")
    print(f"Summary file: {summary_file}")


if __name__ == "__main__":
    main()
