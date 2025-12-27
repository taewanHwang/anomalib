"""WinCLIP HDMAP Validation Script.

Purpose: Evaluate WinCLIP model performance on HDMAP dataset.
Dataset: HDMAP (4 domains: domain_A, domain_B, domain_C, domain_D)
Model: WinCLIP (Zero-shot / Few-shot)

WinCLIP is a zero-shot/few-shot model that uses CLIP embeddings and
sliding window approach to detect anomalies without training.

Modes:
    - Zero-shot (k_shot=0): No reference images, text prompts only
    - Few-shot (k_shot=1,2,4): Uses k normal reference images

Note:
    HDMAP은 classification task (no pixel-level masks)
    WinCLIP의 Image-level AUROC가 주요 메트릭

Visualizations:
    - Per-domain score distribution (histogram)
    - All domains score comparison (2x2 subplot)
    - Per-domain image predictions (sorted by score)
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

# Import custom 4-column visualizer
import sys
sys.path.insert(0, str(Path(__file__).parent))
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
) -> None:
    """Visualize anomaly score distribution for a domain.

    Creates histogram plot showing good vs fault score distributions.

    Args:
        labels: Ground truth labels (0=normal, 1=anomaly).
        scores: Anomaly scores.
        domain: Domain name.
        mode_name: Mode name (zero-shot, 1-shot, etc.)
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
    ax.set_title(f'{domain} ({mode_name}) - Anomaly Score Distribution')
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
) -> None:
    """Create 2x2 subplot comparing score distributions across all domains.

    Args:
        domain_results: Dictionary mapping domain names to (labels, scores) tuples.
        mode_name: Mode name (zero-shot, 1-shot, etc.)
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

    plt.suptitle(f'Per-Domain Anomaly Score Distributions ({mode_name})', fontsize=14)
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
) -> Path:
    """Export per-domain prediction scores to CSV file.

    Args:
        labels: Ground truth labels (0=normal, 1=anomaly).
        scores: Predicted anomaly scores.
        image_paths: List of image paths.
        domain: Domain name.
        mode_name: Mode name.
        output_dir: Directory to save CSV file.

    Returns:
        Path to the saved CSV file.
    """
    # Create output directory
    csv_dir = output_dir / "scores"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{domain}_{mode_name.replace('-', '_')}_scores.csv"

    # Write CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'image_path', 'gt_label', 'gt_label_str', 'pred_score', 'domain', 'mode'])

        for img_path, label, score in zip(image_paths, labels, scores):
            img_name = Path(img_path).stem if img_path else 'unknown'
            label_str = 'fault' if label == 1 else 'good'
            writer.writerow([img_name, img_path, label, label_str, f'{score:.6f}', domain, mode_name])

    logger.info(f"  Saved {len(labels)} scores to {csv_path}")
    return csv_path


def collect_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[float], list[str]]:
    """Collect predictions from model.

    Args:
        model: WinCLIP model.
        dataloader: DataLoader for evaluation.
        device: Device to run inference on.

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
) -> dict:
    """Evaluate WinCLIP on a single HDMAP domain.

    Args:
        domain: HDMAP domain name (domain_A, domain_B, domain_C, domain_D)
        k_shot: Number of reference images (0 for zero-shot)
        result_dir: Directory to save results
        dataset_root: Root path to HDMAP dataset
        gpu_id: GPU device ID to use
        class_name: Class name for WinCLIP prompts
        image_size: Target image size for CLIP (default: 240 for ViT-B-16-plus-240)
        generate_visualizations: Whether to generate visualizations

    Returns:
        Dictionary with evaluation results including labels, scores, paths
    """
    mode_name = "zero-shot" if k_shot == 0 else f"{k_shot}-shot"
    logger.info(f"Evaluating {domain} with {mode_name} mode...")

    # Create output directory for this domain/k_shot
    domain_dir = result_dir / domain / mode_name
    domain_dir.mkdir(parents=True, exist_ok=True)

    # Setup data module
    # WinCLIP uses ViT-B-16-plus-240 which expects 240x240 images
    # HDMAP images are very small (31x95), so we need to resize them
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
    # class_name is used for generating text prompts
    # Disable default visualizer (will use custom FourColumnVisualizer)
    model = WinClip(
        class_name=class_name,
        k_shot=k_shot,
        scales=(2, 3),  # Default scales from paper
        visualizer=False,  # Disable default visualizer
    )

    # Create custom 4-column visualizer
    custom_visualizer = FourColumnVisualizer(
        field_size=(256, 256),
        alpha=0.5,
        output_dir=domain_dir / "images_4col",
    )

    # Create engine (no training needed for WinCLIP)
    engine = Engine(
        accelerator="gpu",
        devices=[gpu_id],
        default_root_dir=str(domain_dir),
        callbacks=[custom_visualizer],
    )

    # Test (WinCLIP doesn't need training)
    logger.info(f"Testing {domain} ({mode_name})...")
    test_results = engine.test(model=model, datamodule=datamodule)

    # Extract metrics
    metrics = {}
    if test_results and len(test_results) > 0:
        metrics = test_results[0]

    # Collect predictions for visualization
    labels, scores, image_paths = [], [], []
    if generate_visualizations:
        logger.info(f"Collecting predictions for visualization...")
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

        # Ensure model is set up for inference
        # For few-shot, we need to setup with reference images
        if k_shot > 0:
            # Setup datamodule to get train data for reference images
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
            # Move model to GPU before setup with GPU tensors
            model.model.to(device)
            model.model.setup(class_name, ref_images.to(device))
        else:
            model.model.setup(class_name, None)

        labels, scores, image_paths = collect_predictions(model, test_dataloader, device)

        # Generate visualizations
        visualize_score_distribution(labels, scores, domain, mode_name, result_dir)
        export_domain_scores_to_csv(labels, scores, image_paths, domain, mode_name, result_dir)

    # Build results dict
    results = {
        "domain": domain,
        "k_shot": k_shot,
        "mode": mode_name,
        "class_name": class_name,
        "metrics": {k: float(v) if isinstance(v, (int, float, torch.Tensor)) else str(v)
                   for k, v in metrics.items()},
        "timestamp": datetime.now().isoformat(),
        "labels": labels,
        "scores": scores,
        "image_paths": image_paths,
    }

    # Save results (without large arrays)
    results_to_save = {k: v for k, v in results.items() if k not in ["labels", "scores", "image_paths"]}
    results_file = domain_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results_to_save, f, indent=2)

    # Log summary - HDMAP is classification only (no pixel AUROC)
    image_auroc = metrics.get("image_AUROC", metrics.get("test_image_AUROC", "N/A"))
    f1_score = metrics.get("image_F1Score", metrics.get("test_F1Score", "N/A"))

    if isinstance(image_auroc, (int, float, torch.Tensor)):
        image_auroc = float(image_auroc)
        logger.info(f"{domain} ({mode_name}): Image AUROC = {image_auroc:.4f} ({image_auroc*100:.2f}%)")
    if isinstance(f1_score, (int, float, torch.Tensor)):
        f1_score = float(f1_score)
        logger.info(f"{domain} ({mode_name}): F1 Score = {f1_score:.4f}")

    return results


def main():
    """Run WinCLIP validation on HDMAP."""
    parser = argparse.ArgumentParser(description="WinCLIP HDMAP Validation")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["domain_A"],
        choices=ALL_DOMAINS + ["all"],
        help="Domains to test (use 'all' for all domains)",
    )
    parser.add_argument(
        "--k-shots",
        nargs="+",
        type=int,
        default=DEFAULT_K_SHOTS,
        help="k_shot values to test (default: 0 1 2 4)",
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
        default="./results/winclip_hdmap_validation",
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
        help="Class name for WinCLIP prompts (affects text embeddings)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=240,
        help="Target image size for CLIP (default: 240 for ViT-B-16-plus-240)",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip generating visualizations (faster)",
    )
    args = parser.parse_args()

    # Expand 'all' to all domains
    domains = ALL_DOMAINS if "all" in args.domains else args.domains

    # Resolve dataset root path (absolute)
    anomalib_root = Path(__file__).parent.parent.parent.parent
    if args.dataset_root.startswith("./"):
        dataset_root = anomalib_root / args.dataset_root[2:]
    else:
        dataset_root = Path(args.dataset_root)

    # Setup result directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(args.result_dir) / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment settings
    settings = {
        "domains": domains,
        "k_shots": args.k_shots,
        "dataset_root": str(dataset_root),
        "gpu": args.gpu,
        "class_name": args.class_name,
        "image_size": args.image_size,
        "timestamp": timestamp,
    }
    with open(result_dir / "experiment_settings.json", "w") as f:
        json.dump(settings, f, indent=2)

    logger.info(f"Starting WinCLIP validation on HDMAP")
    logger.info(f"Domains: {domains}")
    logger.info(f"k_shot modes: {args.k_shots}")
    logger.info(f"Dataset root: {dataset_root}")
    logger.info(f"Class name: {args.class_name}")
    logger.info(f"Image size: {args.image_size}x{args.image_size}")
    logger.info(f"Results will be saved to: {result_dir}")

    # Run experiments grouped by k_shot (for all-domains comparison)
    all_results = []

    for k_shot in args.k_shots:
        mode_name = "zero-shot" if k_shot == 0 else f"{k_shot}-shot"
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {mode_name} experiments...")
        logger.info(f"{'='*60}")

        domain_results_for_viz = {}  # For all-domains comparison plot

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
                )
                all_results.append(results)

                # Store for all-domains comparison
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

        # Generate all-domains comparison for this k_shot
        if domain_results_for_viz and not args.no_visualizations:
            visualize_all_domains_score_comparison(domain_results_for_viz, mode_name, result_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: WinCLIP HDMAP Validation Results")
    print("=" * 80)

    # Group by k_shot
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
                    image_auroc = metrics.get("image_AUROC", metrics.get("test_image_AUROC", "N/A"))
                    f1_score = metrics.get("image_F1Score", metrics.get("test_F1Score", "N/A"))

                    if isinstance(image_auroc, (int, float)):
                        auroc_str = f"{image_auroc * 100:.2f}%"
                    else:
                        auroc_str = str(image_auroc)

                    if isinstance(f1_score, (int, float)):
                        f1_str = f"{f1_score * 100:.2f}%"
                    else:
                        f1_str = str(f1_score)

                    print(f"  {domain}: Image AUROC = {auroc_str}, F1 Score = {f1_str}")

    print("\n" + "=" * 80)

    # Save final summary (without large arrays)
    summary_results = []
    for r in all_results:
        summary_r = {k: v for k, v in r.items() if k not in ["labels", "scores", "image_paths"]}
        summary_results.append(summary_r)

    summary_file = result_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_results, f, indent=2)

    print(f"\nResults saved to: {result_dir}")
    print(f"Summary file: {summary_file}")

    if not args.no_visualizations:
        print(f"Visualizations: {result_dir / 'visualizations'}")
        print(f"Score CSVs: {result_dir / 'scores'}")


if __name__ == "__main__":
    main()
