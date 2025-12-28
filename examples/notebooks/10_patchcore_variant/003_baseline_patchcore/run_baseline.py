#!/usr/bin/env python3
"""003. Baseline PatchCore Evaluation on HDMAP Dataset.

Improved experiment script with structured result saving:
- results/{exp_name}/{timestamp}/config.json
- results/{exp_name}/{timestamp}/results.json

Usage:
    # Full training baseline (1000 samples)
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        examples/notebooks/10_patchcore_variant/003_baseline_patchcore/run_baseline.py \
        --domain domain_C --mode full

    # Few-shot baseline
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        examples/notebooks/10_patchcore_variant/003_baseline_patchcore/run_baseline.py \
        --domain domain_C --mode fewshot --n-cold 1 --n-warm 1
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
ANOMALIB_ROOT = SCRIPT_DIR.parent.parent.parent.parent
DATASET_ROOT = ANOMALIB_ROOT / "datasets" / "HDMAP" / "1000_tiff_minmax"


def get_config(mode: str, n_cold: int = 0, n_warm: int = 0, backbone: str = "vit_small_patch14_dinov2") -> dict:
    """Get experiment configuration."""
    # Backbone-specific settings
    backbone_configs = {
        "vit_small_patch14_dinov2": {"layers": ["blocks.8"], "batch_size": 8},
        "vit_base_patch14_dinov2": {"layers": ["blocks.8"], "batch_size": 4},  # Larger model, smaller batch
        "vit_large_patch14_dinov2": {"layers": ["blocks.8"], "batch_size": 2},
    }
    bb_config = backbone_configs.get(backbone, {"layers": ["blocks.8"], "batch_size": 4})

    config = {
        "backbone": backbone,
        "layers": bb_config["layers"],
        "target_size": (518, 518),
        "resize_method": "resize_bilinear",
        "coreset_sampling_ratio": 0.01,
        "num_neighbors": 9,
        "batch_size": bb_config["batch_size"],
        "seed": 52,
        "mode": mode,
    }
    if mode == "fewshot":
        config["n_cold"] = n_cold
        config["n_warm"] = n_warm
    return config


def get_exp_name(mode: str, domain: str, n_cold: int = 0, n_warm: int = 0, backbone: str = "vit_small_patch14_dinov2") -> str:
    """Generate experiment name."""
    # Short backbone name
    bb_short = backbone.replace("_patch14_dinov2", "").replace("vit_", "")  # e.g., "small", "base", "large"
    if mode == "full":
        return f"patchcore003_full_{bb_short}_{domain}"
    else:
        return f"patchcore003_fewshot_{n_cold}_{n_warm}_{bb_short}_{domain}"


def create_output_dir(exp_name: str) -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / exp_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def setup_model_and_data(domain: str, config: dict):
    """Setup PatchCore model and HDMAP datamodule."""
    from anomalib.data.datamodules.image.hdmap import HDMAPDataModule
    from anomalib.data.utils import ValSplitMode
    from anomalib.models import Patchcore
    from anomalib.post_processing import PostProcessor
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

    # Disable normalization to get raw distance scores (same as few-shot and CA-PatchCore)
    post_processor = PostProcessor(enable_normalization=False)

    model = Patchcore(
        backbone=config["backbone"],
        layers=config["layers"],
        pre_trained=True,
        coreset_sampling_ratio=config["coreset_sampling_ratio"],
        num_neighbors=config["num_neighbors"],
        pre_processor=pre_processor,
        post_processor=post_processor,
    )

    return model, datamodule


def get_reference_loader(datamodule, n_cold: int, n_warm: int, config: dict):
    """Get reference samples from test/good for few-shot mode.

    test/good indices: 0=coldest, 999=warmest
    Select n_cold from start, n_warm from end.
    """
    datamodule.setup(stage="test")
    test_dataset = datamodule.test_data

    # Find good samples
    good_indices = []
    for i, item in enumerate(test_dataset):
        if item.gt_label == 0:  # good
            file_idx = int(Path(item.image_path).stem)
            good_indices.append((i, file_idx))

    # Sort by file index
    good_indices.sort(key=lambda x: x[1])

    # Select coldest and warmest
    cold_indices = [idx for idx, _ in good_indices[:n_cold]]
    warm_indices = [idx for idx, _ in good_indices[-n_warm:]]
    selected_indices = cold_indices + warm_indices

    reference_subset = Subset(test_dataset, selected_indices)

    return DataLoader(
        reference_subset,
        batch_size=config["batch_size"],
        collate_fn=test_dataset.collate_fn,
        num_workers=4,
    ), selected_indices


def build_memory_bank_from_reference(model, reference_loader, device):
    """Build memory bank from reference samples (few-shot mode)."""
    model.model.train()
    pre_processor = model.pre_processor

    with torch.no_grad():
        for batch in tqdm(reference_loader, desc="Building memory bank"):
            images = batch.image.to(device)
            normalized_images = pre_processor(images)
            _ = model.model(normalized_images)

    model.model.eval()


def train_full(model, datamodule, output_dir: Path):
    """Train PatchCore with full training set."""
    from anomalib.engine import Engine

    engine = Engine(
        max_epochs=1,
        devices=1,
        accelerator="auto",
        default_root_dir=str(output_dir / "lightning_logs"),
        enable_progress_bar=True,
    )

    print("Training PatchCore (full training set)...")
    engine.fit(model=model, datamodule=datamodule)

    print("Running inference...")
    predictions = engine.predict(model=model, datamodule=datamodule)

    return predictions


def train_fewshot(model, datamodule, n_cold: int, n_warm: int, config: dict):
    """Train PatchCore with few-shot reference samples."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Get reference samples
    reference_loader, ref_indices = get_reference_loader(datamodule, n_cold, n_warm, config)
    print(f"Using {len(ref_indices)} reference samples (cold: {n_cold}, warm: {n_warm})")

    # Build memory bank
    build_memory_bank_from_reference(model, reference_loader, device)

    # For few-shot, memory bank is small enough - skip coreset sampling
    if hasattr(model.model, 'memory_bank') and model.model.memory_bank is not None:
        print(f"Memory bank size: {model.model.memory_bank.shape[0]}")

    # Inference
    print("Running inference...")
    datamodule.setup(stage="test")
    test_loader = DataLoader(
        datamodule.test_data,
        batch_size=config["batch_size"],
        collate_fn=datamodule.test_data.collate_fn,
        num_workers=8,
    )

    predictions = []
    pre_processor = model.pre_processor

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            images = batch.image.to(device)
            normalized_images = pre_processor(images)

            output = model.model(normalized_images)

            # Create prediction batch (output is InferenceBatch NamedTuple)
            batch.pred_score = output.pred_score.cpu()
            predictions.append(batch)

    return predictions


def analyze_predictions(predictions, domain: str) -> tuple[dict, dict]:
    """Analyze predictions with cold/warm breakdown.

    Returns:
        results: dict with accuracy/auroc metrics
        score_data: dict with raw scores for visualization
    """
    all_scores = []
    all_labels = []
    all_indices = []

    for batch in predictions:
        scores = batch.pred_score.cpu().numpy()
        labels = batch.gt_label.cpu().numpy()

        for i, path in enumerate(batch.image_path):
            file_idx = int(Path(path).stem)
            is_fault = "fault" in str(path).lower()
            dataset_idx = file_idx if is_fault else 1000 + file_idx
            all_indices.append(dataset_idx)

        all_scores.extend(scores)
        all_labels.extend(labels)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_indices = np.array(all_indices)

    file_indices = all_indices % 1000
    cold_mask = file_indices < 500
    warm_mask = file_indices >= 500

    # Find optimal threshold
    thresholds = np.percentile(all_scores, np.arange(0, 101, 1))
    best_acc, best_threshold = 0, 0
    for thresh in thresholds:
        preds = (all_scores >= thresh).astype(int)
        acc = accuracy_score(all_labels, preds)
        if acc > best_acc:
            best_acc, best_threshold = acc, thresh

    # Metrics
    overall_preds = (all_scores >= best_threshold).astype(int)

    results = {
        "overall": {
            "accuracy": float(accuracy_score(all_labels, overall_preds)),
            "auroc": float(roc_auc_score(all_labels, all_scores)),
            "threshold": float(best_threshold),
            "n_samples": int(len(all_labels)),
        },
        "cold": {
            "accuracy": float(accuracy_score(all_labels[cold_mask], (all_scores[cold_mask] >= best_threshold).astype(int))),
            "auroc": float(roc_auc_score(all_labels[cold_mask], all_scores[cold_mask])) if len(np.unique(all_labels[cold_mask])) > 1 else 0.0,
            "n_samples": int(cold_mask.sum()),
        },
        "warm": {
            "accuracy": float(accuracy_score(all_labels[warm_mask], (all_scores[warm_mask] >= best_threshold).astype(int))),
            "auroc": float(roc_auc_score(all_labels[warm_mask], all_scores[warm_mask])) if len(np.unique(all_labels[warm_mask])) > 1 else 0.0,
            "n_samples": int(warm_mask.sum()),
        },
    }

    # Score data for visualization
    score_data = {
        "all_scores": all_scores,
        "all_labels": all_labels,
        "cold_mask": cold_mask,
        "warm_mask": warm_mask,
        "threshold": best_threshold,
    }

    return results, score_data


def plot_score_distribution(score_data: dict, output_dir: Path, domain: str):
    """Plot and save score distribution histograms."""
    scores = score_data["all_scores"]
    labels = score_data["all_labels"]
    cold_mask = score_data["cold_mask"]
    warm_mask = score_data["warm_mask"]
    threshold = score_data["threshold"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # All samples
    ax = axes[0]
    ax.hist(scores[labels == 0], bins=50, alpha=0.7, label="Good", color="green", density=True)
    ax.hist(scores[labels == 1], bins=50, alpha=0.7, label="Fault", color="red", density=True)
    ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold:.3f}")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title(f"{domain} - All Samples")
    ax.legend()

    # Cold samples
    ax = axes[1]
    cold_scores = scores[cold_mask]
    cold_labels = labels[cold_mask]
    ax.hist(cold_scores[cold_labels == 0], bins=50, alpha=0.7, label="Good", color="green", density=True)
    ax.hist(cold_scores[cold_labels == 1], bins=50, alpha=0.7, label="Fault", color="red", density=True)
    ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold:.3f}")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title(f"{domain} - Cold Samples")
    ax.legend()

    # Warm samples
    ax = axes[2]
    warm_scores = scores[warm_mask]
    warm_labels = labels[warm_mask]
    ax.hist(warm_scores[warm_labels == 0], bins=50, alpha=0.7, label="Good", color="green", density=True)
    ax.hist(warm_scores[warm_labels == 1], bins=50, alpha=0.7, label="Fault", color="red", density=True)
    ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold:.3f}")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title(f"{domain} - Warm Samples")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution.png", dpi=150)
    plt.close()
    print(f"Saved score distribution plot to: {output_dir / 'score_distribution.png'}")


def print_results(results: dict, domain: str, mode: str):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"Results: {domain} ({mode})")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Overall':>12} {'Cold':>12} {'Warm':>12}")
    print("-" * 55)
    print(f"{'Accuracy':<15} {results['overall']['accuracy']:>12.2%} {results['cold']['accuracy']:>12.2%} {results['warm']['accuracy']:>12.2%}")
    print(f"{'AUROC':<15} {results['overall']['auroc']:>12.2%} {results['cold']['auroc']:>12.2%} {results['warm']['auroc']:>12.2%}")


def main():
    parser = argparse.ArgumentParser(description="Baseline PatchCore on HDMAP")
    parser.add_argument("--domain", type=str, required=True,
                        choices=["domain_A", "domain_B", "domain_C", "domain_D"])
    parser.add_argument("--mode", type=str, default="full", choices=["full", "fewshot"])
    parser.add_argument("--n-cold", type=int, default=1, help="Number of cold reference samples")
    parser.add_argument("--n-warm", type=int, default=1, help="Number of warm reference samples")
    parser.add_argument("--backbone", type=str, default="vit_small_patch14_dinov2",
                        choices=["vit_small_patch14_dinov2", "vit_base_patch14_dinov2", "vit_large_patch14_dinov2"],
                        help="Backbone model")
    args = parser.parse_args()

    # Config and output
    config = get_config(args.mode, args.n_cold, args.n_warm, args.backbone)
    exp_name = get_exp_name(args.mode, args.domain, args.n_cold, args.n_warm, args.backbone)
    output_dir = create_output_dir(exp_name)

    print(f"\nExperiment: {exp_name}")
    print(f"Output: {output_dir}")

    # Save config
    config["domain"] = args.domain
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Setup
    model, datamodule = setup_model_and_data(args.domain, config)

    # Train and evaluate
    if args.mode == "full":
        predictions = train_full(model, datamodule, output_dir)
    else:
        predictions = train_fewshot(model, datamodule, args.n_cold, args.n_warm, config)

    # Analyze
    results, score_data = analyze_predictions(predictions, args.domain)
    results["domain"] = args.domain
    results["mode"] = args.mode
    if args.mode == "fewshot":
        results["n_cold"] = args.n_cold
        results["n_warm"] = args.n_warm

    # Print and save
    print_results(results, args.domain, args.mode)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save score distribution plot
    plot_score_distribution(score_data, output_dir, args.domain)

    print(f"\nSaved to: {output_dir}")

    return results


if __name__ == "__main__":
    main()
