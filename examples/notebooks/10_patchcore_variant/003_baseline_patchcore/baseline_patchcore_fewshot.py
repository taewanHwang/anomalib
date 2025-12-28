#!/usr/bin/env python3
"""003. Few-shot Baseline PatchCore Evaluation on HDMAP Dataset.

Uses test/good samples as reference instead of full train set.
Reference selection:
- Cold: index 0-499 (0 is coldest) -> pick from near 0
- Warm: index 500-999 (999 is warmest) -> pick from near 999

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        examples/notebooks/10_patchcore_variant/003_baseline_patchcore/baseline_patchcore_fewshot.py \
        --domain domain_C --n-cold 1 --n-warm 1

    # Multiple settings
    for n in 1 2 4; do
        CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
            examples/notebooks/10_patchcore_variant/003_baseline_patchcore/baseline_patchcore_fewshot.py \
            --domain domain_C --n-cold $n --n-warm $n
    done
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "results_fewshot"
ANOMALIB_ROOT = SCRIPT_DIR.parent.parent.parent.parent
DATASET_ROOT = ANOMALIB_ROOT / "datasets" / "HDMAP" / "1000_tiff_minmax"

# Model settings
CONFIG = {
    "backbone": "vit_small_patch14_dinov2",
    "layers": ["blocks.8"],
    "target_size": (518, 518),
    "resize_method": "resize_bilinear",
    "coreset_sampling_ratio": 1.0,  # Use all reference samples (no subsampling for few-shot)
    "num_neighbors": 9,
    "batch_size": 8,
    "seed": 52,
}


def get_reference_indices(n_cold: int, n_warm: int):
    """Get reference sample indices from test/good.

    Cold samples: pick from indices near 0 (coldest)
    Warm samples: pick from indices near 999 (warmest)

    In test/good folder:
    - Files 000000.tiff ~ 000499.tiff are cold (index 0-499)
    - Files 000500.tiff ~ 000999.tiff are warm (index 500-999)
    """
    # Pick coldest samples (near index 0)
    cold_indices = list(range(n_cold))  # [0, 1, 2, ...] for n_cold samples

    # Pick warmest samples (near index 999)
    warm_indices = list(range(999, 999 - n_warm, -1))  # [999, 998, 997, ...] for n_warm samples

    return cold_indices, warm_indices


def create_reference_dataloader(domain: str, cold_indices: list, warm_indices: list):
    """Create dataloader with only reference samples from test/good."""
    from anomalib.data.datasets.image.hdmap import HDMAPDataset
    from anomalib.data.utils import Split

    # Load test dataset (which includes both good and fault)
    dataset = HDMAPDataset(
        root=str(DATASET_ROOT),
        domain=domain,
        split=Split.TEST,
        target_size=CONFIG["target_size"],
        resize_method=CONFIG["resize_method"],
    )

    # Find indices of good samples in the dataset
    # Dataset structure: fault samples first (0-999), then good samples (1000-1999)
    # So test/good index 0 = dataset index 1000
    good_offset = 1000  # Good samples start at index 1000 in the full test dataset

    # Map file indices to dataset indices
    reference_dataset_indices = []
    for idx in cold_indices:
        reference_dataset_indices.append(good_offset + idx)
    for idx in warm_indices:
        reference_dataset_indices.append(good_offset + idx)

    # Create subset with only reference samples
    reference_subset = Subset(dataset, reference_dataset_indices)

    return DataLoader(
        reference_subset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn,  # Use anomalib's collate function
    )


def create_test_dataloader(domain: str):
    """Create dataloader for full test set."""
    from anomalib.data.datasets.image.hdmap import HDMAPDataset
    from anomalib.data.utils import Split

    dataset = HDMAPDataset(
        root=str(DATASET_ROOT),
        domain=domain,
        split=Split.TEST,
        target_size=CONFIG["target_size"],
        resize_method=CONFIG["resize_method"],
    )

    return DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=8,
        collate_fn=dataset.collate_fn,  # Use anomalib's collate function
    )


def build_memory_bank_from_reference(model, reference_loader, device):
    """Build memory bank from reference samples only."""
    print(f"Building memory bank from {len(reference_loader.dataset)} reference samples...")

    # Set model to training mode to populate memory bank
    model.model.train()

    # Apply pre-processor (normalization) and forward pass
    pre_processor = model.pre_processor

    with torch.no_grad():
        for batch in tqdm(reference_loader, desc="Building memory bank"):
            # batch is anomalib Batch object
            images = batch.image.to(device)
            # Apply normalization
            normalized_images = pre_processor(images)
            # Forward pass in training mode adds embeddings to memory bank
            _ = model.model(normalized_images)

    print(f"Memory bank size: {model.model.memory_bank.shape}")

    # Set model back to eval mode
    model.model.eval()

    return model


def run_inference(model, test_loader, device):
    """Run inference and collect predictions."""
    model.model.eval()
    pre_processor = model.pre_processor

    all_scores = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            # batch is anomalib Batch object
            images = batch.image.to(device)
            labels = batch.gt_label
            paths = batch.image_path

            # Apply normalization
            normalized_images = pre_processor(images)

            # Get predictions
            predictions = model.model(normalized_images)
            scores = predictions.pred_score.cpu().numpy()

            all_scores.extend(scores)
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)

    return np.array(all_scores), np.array(all_labels), all_paths


def analyze_predictions(scores, labels, paths, domain: str):
    """Analyze predictions with cold/warm breakdown."""
    all_indices = []

    for path in paths:
        file_idx = int(Path(path).stem)
        is_fault = "fault" in str(path).lower()

        if is_fault:
            dataset_idx = file_idx
        else:
            dataset_idx = 1000 + file_idx
        all_indices.append(dataset_idx)

    all_indices = np.array(all_indices)
    file_indices = all_indices % 1000
    cold_mask = file_indices < 500
    warm_mask = file_indices >= 500

    # Find optimal threshold
    thresholds = np.percentile(scores, np.arange(0, 101, 1))
    best_acc = 0
    best_threshold = 0
    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_threshold = thresh

    results = {}

    # Overall
    overall_preds = (scores >= best_threshold).astype(int)
    results["overall"] = {
        "accuracy": accuracy_score(labels, overall_preds),
        "auroc": roc_auc_score(labels, scores),
        "threshold": best_threshold,
        "n_samples": len(labels),
    }

    # Cold
    cold_scores = scores[cold_mask]
    cold_labels = labels[cold_mask]
    cold_preds = (cold_scores >= best_threshold).astype(int)
    results["cold"] = {
        "accuracy": accuracy_score(cold_labels, cold_preds),
        "auroc": roc_auc_score(cold_labels, cold_scores) if len(np.unique(cold_labels)) > 1 else 0.0,
        "n_samples": len(cold_labels),
    }

    # Warm
    warm_scores = scores[warm_mask]
    warm_labels = labels[warm_mask]
    warm_preds = (warm_scores >= best_threshold).astype(int)
    results["warm"] = {
        "accuracy": accuracy_score(warm_labels, warm_preds),
        "auroc": roc_auc_score(warm_labels, warm_scores) if len(np.unique(warm_labels)) > 1 else 0.0,
        "n_samples": len(warm_labels),
    }

    # Confusion analysis
    cold_fault_mask = cold_mask & (labels == 1)
    warm_fault_mask = warm_mask & (labels == 1)

    results["confusion"] = {
        "cold_fault_detected": float((scores[cold_fault_mask] >= best_threshold).mean()) if cold_fault_mask.sum() > 0 else 0.0,
        "warm_fault_detected": float((scores[warm_fault_mask] >= best_threshold).mean()) if warm_fault_mask.sum() > 0 else 0.0,
    }

    return results


def print_results(results: dict, domain: str, n_cold: int, n_warm: int):
    """Print results."""
    print(f"\n{'='*60}")
    print(f"Results for {domain} (Few-shot: {n_cold} cold + {n_warm} warm)")
    print(f"{'='*60}")

    print(f"\n{'Metric':<20} {'Overall':>12} {'Cold':>12} {'Warm':>12}")
    print("-" * 60)
    print(f"{'Accuracy':<20} {results['overall']['accuracy']:>12.4f} {results['cold']['accuracy']:>12.4f} {results['warm']['accuracy']:>12.4f}")
    print(f"{'AUROC':<20} {results['overall']['auroc']:>12.4f} {results['cold']['auroc']:>12.4f} {results['warm']['auroc']:>12.4f}")

    print(f"\nCold Fault Detection Rate: {results['confusion']['cold_fault_detected']:.4f}")
    print(f"Warm Fault Detection Rate: {results['confusion']['warm_fault_detected']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Few-shot PatchCore evaluation")
    parser.add_argument("--domain", type=str, default="domain_C",
                        choices=["domain_A", "domain_B", "domain_C", "domain_D"])
    parser.add_argument("--n-cold", type=int, default=1, help="Number of cold reference samples")
    parser.add_argument("--n-warm", type=int, default=1, help="Number of warm reference samples")
    args = parser.parse_args()

    domain = args.domain
    n_cold = args.n_cold
    n_warm = args.n_warm

    # Output directory
    output_dir = OUTPUT_DIR / domain / f"cold{n_cold}_warm{n_warm}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Few-shot PatchCore Baseline")
    print(f"{'='*60}")
    print(f"Domain: {domain}")
    print(f"Reference: {n_cold} cold + {n_warm} warm samples")
    print(f"Backbone: {CONFIG['backbone']}")
    print(f"Resize: {CONFIG['resize_method']}")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get reference indices
    cold_indices, warm_indices = get_reference_indices(n_cold, n_warm)
    print(f"Cold reference indices: {cold_indices}")
    print(f"Warm reference indices: {warm_indices}")

    # Create model
    from anomalib.models import Patchcore
    from anomalib.pre_processing import PreProcessor
    from torchvision.transforms.v2 import Compose, Normalize

    pre_processor = PreProcessor(
        transform=Compose([
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )

    model = Patchcore(
        backbone=CONFIG["backbone"],
        layers=CONFIG["layers"],
        pre_trained=True,
        coreset_sampling_ratio=CONFIG["coreset_sampling_ratio"],
        num_neighbors=CONFIG["num_neighbors"],
        pre_processor=pre_processor,
    )
    model = model.to(device)

    # Create dataloaders
    reference_loader = create_reference_dataloader(domain, cold_indices, warm_indices)
    test_loader = create_test_dataloader(domain)

    # Build memory bank from reference
    model = build_memory_bank_from_reference(model, reference_loader, device)

    # Run inference
    scores, labels, paths = run_inference(model, test_loader, device)

    # Analyze results
    results = analyze_predictions(scores, labels, paths, domain)

    # Print and save
    print_results(results, domain, n_cold, n_warm)

    results["config"] = CONFIG
    results["domain"] = domain
    results["n_cold"] = n_cold
    results["n_warm"] = n_warm
    results["cold_indices"] = cold_indices
    results["warm_indices"] = warm_indices

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    return results


if __name__ == "__main__":
    main()
