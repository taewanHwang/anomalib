#!/usr/bin/env python3
"""003. Baseline PatchCore Evaluation on HDMAP Dataset.

This script evaluates PatchCore with DINO backbone on HDMAP dataset,
measuring overall, cold-only, and warm-only accuracy for each domain.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        examples/notebooks/10_patchcore_variant/003_baseline_patchcore/baseline_patchcore.py \
        --domain domain_C

    # All domains
    for domain in domain_A domain_B domain_C domain_D; do
        CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
            examples/notebooks/10_patchcore_variant/003_baseline_patchcore/baseline_patchcore.py \
            --domain $domain
    done
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "results"
ANOMALIB_ROOT = SCRIPT_DIR.parent.parent.parent.parent  # examples/notebooks/10_patchcore_variant/003... -> anomalib root
DATASET_ROOT = ANOMALIB_ROOT / "datasets" / "HDMAP" / "1000_tiff_minmax"

# Model settings (from exp-23)
CONFIG = {
    "backbone": "vit_small_patch14_dinov2",
    "layers": ["blocks.8"],
    "target_size": (518, 518),
    "resize_method": "resize_bilinear",  # Changed from "resize" based on 001 EDA
    "coreset_sampling_ratio": 0.01,
    "num_neighbors": 9,
    "batch_size": 8,
    "seed": 52,
}


def setup_model_and_data(domain: str):
    """Setup PatchCore model and HDMAP datamodule."""
    from anomalib.data.datamodules.image.hdmap import HDMAPDataModule
    from anomalib.data.utils import ValSplitMode
    from anomalib.models import Patchcore
    from anomalib.pre_processing import PreProcessor
    from torchvision.transforms.v2 import Compose, Normalize

    # Create datamodule
    # HDMAP handles its own resizing via target_size and resize_method
    datamodule = HDMAPDataModule(
        root=str(DATASET_ROOT),
        domain=domain,
        train_batch_size=CONFIG["batch_size"],
        eval_batch_size=CONFIG["batch_size"],
        num_workers=8,
        val_split_mode=ValSplitMode.FROM_TEST,
        val_split_ratio=0.01,  # Minimal validation
        target_size=CONFIG["target_size"],
        resize_method=CONFIG["resize_method"],
        seed=CONFIG["seed"],
    )

    # Create custom pre-processor that only normalizes (no resize)
    # HDMAP already handles resizing to 518x518
    pre_processor = PreProcessor(
        transform=Compose([
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )

    # Create model with custom pre-processor
    model = Patchcore(
        backbone=CONFIG["backbone"],
        layers=CONFIG["layers"],
        pre_trained=True,
        coreset_sampling_ratio=CONFIG["coreset_sampling_ratio"],
        num_neighbors=CONFIG["num_neighbors"],
        pre_processor=pre_processor,
    )

    return model, datamodule


def train_and_evaluate(model, datamodule, domain: str):
    """Train PatchCore and get predictions on test set."""
    from anomalib.engine import Engine

    # Setup engine
    engine = Engine(
        max_epochs=1,
        devices=1,
        accelerator="auto",
        default_root_dir=str(OUTPUT_DIR / domain / "lightning_logs"),
        enable_progress_bar=True,
    )

    # Train (builds memory bank)
    print(f"\n[{domain}] Training PatchCore (building memory bank)...")
    engine.fit(model=model, datamodule=datamodule)

    # Test and collect predictions
    print(f"[{domain}] Running inference on test set...")
    predictions = engine.predict(model=model, datamodule=datamodule)

    return predictions


def analyze_predictions(predictions, domain: str):
    """Analyze predictions with cold/warm breakdown.

    HDMAP test set structure:
    - Fault images: indices 0-999 (files 000000.tiff ~ 000999.tiff)
        - Cold fault: indices 0-499
        - Warm fault: indices 500-999
    - Good images: indices 1000-1999 (files 000000.tiff ~ 000999.tiff in good folder)
        - Cold good: indices 1000-1499 (corresponding to file indices 0-499)
        - Warm good: indices 1500-1999 (corresponding to file indices 500-999)
    """
    # Collect all predictions
    all_scores = []
    all_labels = []
    all_indices = []  # Track original indices for cold/warm split

    for batch in predictions:
        scores = batch.pred_score.cpu().numpy()
        labels = batch.gt_label.cpu().numpy()

        # Get file indices from image paths
        for i, path in enumerate(batch.image_path):
            file_idx = int(Path(path).stem)  # e.g., "000123.tiff" -> 123
            is_fault = "fault" in str(path).lower()

            # Determine original dataset index
            if is_fault:
                dataset_idx = file_idx  # 0-999 for fault
            else:
                dataset_idx = 1000 + file_idx  # 1000-1999 for good

            all_indices.append(dataset_idx)

        all_scores.extend(scores)
        all_labels.extend(labels)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_indices = np.array(all_indices)

    # Define cold/warm masks based on file index (not dataset index)
    # Cold: file index 0-499, Warm: file index 500-999
    file_indices = all_indices % 1000  # Convert back to file index
    cold_mask = file_indices < 500
    warm_mask = file_indices >= 500

    # Calculate metrics
    results = {}

    # Find optimal threshold using overall data
    thresholds = np.percentile(all_scores, np.arange(0, 101, 1))
    best_acc = 0
    best_threshold = 0
    for thresh in thresholds:
        preds = (all_scores >= thresh).astype(int)
        acc = accuracy_score(all_labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_threshold = thresh

    # Overall metrics
    overall_preds = (all_scores >= best_threshold).astype(int)
    results["overall"] = {
        "accuracy": accuracy_score(all_labels, overall_preds),
        "auroc": roc_auc_score(all_labels, all_scores),
        "threshold": best_threshold,
        "n_samples": len(all_labels),
    }

    # Cold metrics
    cold_scores = all_scores[cold_mask]
    cold_labels = all_labels[cold_mask]
    cold_preds = (cold_scores >= best_threshold).astype(int)
    results["cold"] = {
        "accuracy": accuracy_score(cold_labels, cold_preds),
        "auroc": roc_auc_score(cold_labels, cold_scores) if len(np.unique(cold_labels)) > 1 else 0.0,
        "n_samples": len(cold_labels),
        "n_fault": int(cold_labels.sum()),
        "n_good": int(len(cold_labels) - cold_labels.sum()),
    }

    # Warm metrics
    warm_scores = all_scores[warm_mask]
    warm_labels = all_labels[warm_mask]
    warm_preds = (warm_scores >= best_threshold).astype(int)
    results["warm"] = {
        "accuracy": accuracy_score(warm_labels, warm_preds),
        "auroc": roc_auc_score(warm_labels, warm_scores) if len(np.unique(warm_labels)) > 1 else 0.0,
        "n_samples": len(warm_labels),
        "n_fault": int(warm_labels.sum()),
        "n_good": int(len(warm_labels) - warm_labels.sum()),
    }

    # Confusion analysis: Cold fault predictions
    cold_fault_mask = cold_mask & (all_labels == 1)
    warm_fault_mask = warm_mask & (all_labels == 1)
    cold_good_mask = cold_mask & (all_labels == 0)
    warm_good_mask = warm_mask & (all_labels == 0)

    results["confusion"] = {
        "cold_fault_detected": float((all_scores[cold_fault_mask] >= best_threshold).mean()) if cold_fault_mask.sum() > 0 else 0.0,
        "warm_fault_detected": float((all_scores[warm_fault_mask] >= best_threshold).mean()) if warm_fault_mask.sum() > 0 else 0.0,
        "cold_good_fp_rate": float((all_scores[cold_good_mask] >= best_threshold).mean()) if cold_good_mask.sum() > 0 else 0.0,
        "warm_good_fp_rate": float((all_scores[warm_good_mask] >= best_threshold).mean()) if warm_good_mask.sum() > 0 else 0.0,
    }

    return results


def print_results(results: dict, domain: str):
    """Print results in a formatted table."""
    print(f"\n{'='*60}")
    print(f"Results for {domain}")
    print(f"{'='*60}")

    print(f"\n{'Metric':<20} {'Overall':>12} {'Cold':>12} {'Warm':>12}")
    print("-" * 60)
    print(f"{'Accuracy':<20} {results['overall']['accuracy']:>12.4f} {results['cold']['accuracy']:>12.4f} {results['warm']['accuracy']:>12.4f}")
    print(f"{'AUROC':<20} {results['overall']['auroc']:>12.4f} {results['cold']['auroc']:>12.4f} {results['warm']['auroc']:>12.4f}")
    print(f"{'N Samples':<20} {results['overall']['n_samples']:>12d} {results['cold']['n_samples']:>12d} {results['warm']['n_samples']:>12d}")

    print(f"\nConfusion Analysis:")
    print(f"  Cold Fault Detection Rate: {results['confusion']['cold_fault_detected']:.4f}")
    print(f"  Warm Fault Detection Rate: {results['confusion']['warm_fault_detected']:.4f}")
    print(f"  Cold Good FP Rate: {results['confusion']['cold_good_fp_rate']:.4f}")
    print(f"  Warm Good FP Rate: {results['confusion']['warm_good_fp_rate']:.4f}")

    print(f"\nThreshold: {results['overall']['threshold']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Baseline PatchCore evaluation on HDMAP")
    parser.add_argument("--domain", type=str, default="domain_C",
                        choices=["domain_A", "domain_B", "domain_C", "domain_D"],
                        help="Domain to evaluate")
    args = parser.parse_args()

    domain = args.domain

    # Create output directory
    output_dir = OUTPUT_DIR / domain
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Baseline PatchCore Evaluation")
    print(f"{'='*60}")
    print(f"Domain: {domain}")
    print(f"Backbone: {CONFIG['backbone']}")
    print(f"Layers: {CONFIG['layers']}")
    print(f"Resize Method: {CONFIG['resize_method']}")
    print(f"Target Size: {CONFIG['target_size']}")
    print(f"Coreset Ratio: {CONFIG['coreset_sampling_ratio']}")

    # Setup
    model, datamodule = setup_model_and_data(domain)

    # Train and evaluate
    predictions = train_and_evaluate(model, datamodule, domain)

    # Analyze results
    results = analyze_predictions(predictions, domain)

    # Print and save results
    print_results(results, domain)

    # Save results
    results["config"] = CONFIG
    results["domain"] = domain

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    return results


if __name__ == "__main__":
    main()
