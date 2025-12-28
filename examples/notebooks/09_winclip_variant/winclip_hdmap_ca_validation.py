"""Condition-Aware WinCLIP (CA-WinCLIP) HDMAP Validation Script.

Purpose: Evaluate CA-WinCLIP model performance on HDMAP dataset with
         condition-aware reference bank selection.

Key Concept:
    CA-WinCLIP maintains separate reference banks for different conditions
    (cold/warm) and automatically selects the appropriate bank for each
    test image based on global embedding similarity.

Dataset: HDMAP (4 domains: domain_A, domain_B, domain_C, domain_D)
    - Test data structure: 0-499 (cold), 500-999 (warm) for both good/fault
    - Reference images come from TEST data (train has no cold/warm labels)

Gating Modes:
    - oracle: Use ground truth condition (index-based) - upper bound
    - topk: Use Top-K global similarity gating (CLIP-based, 88.8%)
    - p90: Use 90th percentile intensity gating (96.7%, recommended)
    - random: Random bank selection (~50%) - baseline
    - inverse: Always wrong bank (0%) - worst case baseline
    - mixed: Use all references without gating (standard few-shot) - key baseline

Reference Budget:
    - k_per_bank: Number of references per bank (1-2 recommended)
    - Total references = k_per_bank * 2 (cold + warm banks)

Metrics:
    - Overall AUROC
    - Cold-only AUROC
    - Warm-only AUROC
    - Cross-condition AUROC (Cold Fault vs Warm Normal) - KEY METRIC
    - Gating Accuracy (vs oracle)
"""

import argparse
import csv
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from anomalib.data.datasets.image.hdmap import HDMAPDataset
from anomalib.models.image.winclip.torch_model import WinClipModel

# Import CA-WinCLIP components
import sys
sys.path.insert(0, str(Path(__file__).parent))
from ca_winclip import (
    ConditionAwareWinCLIP,
    InverseGating,
    P90IntensityGating,
    RandomGating,
)

# Configure logging with explicit flush for nohup compatibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,  # Use stdout for immediate visibility
    force=True,
)
# Add flush handler for immediate output
for handler in logging.root.handlers:
    handler.flush = lambda: sys.stdout.flush()

logger = logging.getLogger(__name__)

# All 4 HDMAP domains
ALL_DOMAINS = ["domain_A", "domain_B", "domain_C", "domain_D"]


def create_experiment_meta(
    result_dir: Path,
    args: argparse.Namespace,
    dataset_root: Path,
) -> Dict:
    """Create experiment.json with initial metadata (status=running).

    Args:
        result_dir: Timestamped result directory.
        args: Parsed command-line arguments.
        dataset_root: Resolved dataset root path.

    Returns:
        Dictionary containing experiment metadata.
    """
    meta = {
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "experiment_config": {
            "domain": args.domain,
            "k_per_bank": args.k_per_bank,
            "gating_mode": args.gating,
            "compare_all": args.compare_all,
            "class_name": args.class_name,
            "gpu": args.gpu,
            "verbose": args.verbose,
        },
        "paths": {
            "dataset_root": str(dataset_root),
            "result_dir": str(result_dir),
        },
        "results": None,
        "error": None,
    }

    # Save initial metadata
    meta_file = result_dir / "experiment.json"
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Created experiment metadata: {meta_file}")
    return meta


def update_experiment_meta(
    result_dir: Path,
    status: str,
    results: Dict = None,
    error: str = None,
) -> None:
    """Update experiment.json with final status and results.

    Args:
        result_dir: Timestamped result directory.
        status: Final status ("completed" or "failed").
        results: Optional results dictionary.
        error: Optional error message if failed.
    """
    meta_file = result_dir / "experiment.json"

    # Load existing metadata
    with open(meta_file, 'r') as f:
        meta = json.load(f)

    # Update metadata
    meta["status"] = status
    meta["end_time"] = datetime.now().isoformat()

    # Calculate duration
    start_time = datetime.fromisoformat(meta["start_time"])
    end_time = datetime.fromisoformat(meta["end_time"])
    meta["duration_seconds"] = (end_time - start_time).total_seconds()

    if results is not None:
        meta["results"] = results
    if error is not None:
        meta["error"] = error

    # Save updated metadata
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Updated experiment metadata: status={status}, duration={meta['duration_seconds']:.1f}s")

# Index boundaries for cold/warm conditions
COLD_RANGE = range(0, 500)  # indices 0-499
WARM_RANGE = range(500, 1000)  # indices 500-999


def get_condition_from_file_index(file_idx: int) -> str:
    """Get condition (cold/warm) from file index.

    File naming: 000000.tiff to 000999.tiff
    - 0-499: cold (index 0 is coldest)
    - 500-999: warm (index 999 is warmest)
    """
    return "cold" if file_idx in COLD_RANGE else "warm"


def extract_file_index_from_path(path: str) -> int:
    """Extract file index from image path like '000123.tiff'."""
    try:
        return int(Path(path).stem)
    except (ValueError, AttributeError):
        return -1


def load_references_from_test(
    dataset_root: Path,
    domain: str,
    condition: str,
    k: int,
    image_size: int = 240,
    device: torch.device = None,
) -> Tuple[torch.Tensor, List[int]]:
    """Load reference images from TEST data.

    Since train data has no cold/warm labels, we use test/good data
    for reference images.

    Dataset structure (2000 total samples):
        - Dataset indices 0-999: fault samples (file indices 000000-000999)
        - Dataset indices 1000-1999: good samples (file indices 000000-000999)
        - File index 0-499 = cold condition, 500-999 = warm condition

    Args:
        dataset_root: Root path to HDMAP dataset.
        domain: Domain name.
        condition: "cold" or "warm".
        k: Number of reference images to load.
        image_size: Target image size.
        device: Target device for tensors.

    Returns:
        Tuple of (reference_images tensor, list of file indices used).

    Index strategy:
        - Cold: Use file index 0, 1, ... (most cold) from good samples
        - Warm: Use file index 999, 998, ... (most warm) from good samples
    """
    # Create dataset for test
    dataset = HDMAPDataset(
        root=str(dataset_root),
        domain=domain,
        split="test",
        target_size=(image_size, image_size),
        resize_method="resize",
    )

    # Dataset has fault samples first (0-999), then good samples (1000-1999)
    # We need good samples only for reference
    # Good samples are at dataset indices 1000-1999, with file indices 0-999

    # Determine which file indices to use for references
    if condition == "cold":
        # Cold: file indices 0, 1, 2, ... (most cold first)
        # These correspond to dataset indices 1000, 1001, 1002, ...
        file_indices_to_use = list(range(k))  # [0, 1, 2, ...]
    else:  # warm
        # Warm: file indices 999, 998, 997, ... (most warm first)
        # These correspond to dataset indices 1999, 1998, 1997, ...
        file_indices_to_use = list(range(999, 999 - k, -1))  # [999, 998, ...]

    # Load reference images
    ref_images = []
    used_file_indices = []

    for file_idx in file_indices_to_use:
        # Good samples start at dataset index 1000
        # File index 0 -> dataset index 1000, file index 999 -> dataset index 1999
        dataset_idx = 1000 + file_idx

        if dataset_idx < len(dataset):
            sample = dataset[dataset_idx]
            # Verify it's a good sample
            if sample.gt_label == 0 or sample.gt_label == False:
                ref_images.append(sample.image)
                used_file_indices.append(file_idx)
            else:
                logger.warning(f"  Dataset index {dataset_idx} is not a good sample, skipping")

    if len(ref_images) == 0:
        raise RuntimeError(f"No {condition} reference images found!")

    ref_tensor = torch.stack(ref_images)
    if device is not None:
        ref_tensor = ref_tensor.to(device)

    logger.info(f"  Loaded {len(used_file_indices)} {condition} references from file indices: {used_file_indices}")
    return ref_tensor, used_file_indices


def compute_condition_aurocs(
    labels: List[int],
    scores: List[float],
    indices: List[int],
) -> Dict[str, float]:
    """Compute AUROC metrics by condition.

    Args:
        labels: Ground truth labels (0=good, 1=fault).
        scores: Predicted anomaly scores.
        indices: Sample indices for condition lookup.

    Returns:
        Dictionary with various AUROC metrics.
    """
    results = {}

    # Convert to numpy
    # indices here are file indices (0-999), not dataset indices
    labels_np = np.array(labels)
    scores_np = np.array(scores)
    conditions = np.array([get_condition_from_file_index(i) for i in indices])

    # Overall AUROC
    if len(np.unique(labels_np)) > 1:
        results["overall_auroc"] = roc_auc_score(labels_np, scores_np)
    else:
        results["overall_auroc"] = 0.0

    # Cold-only AUROC
    cold_mask = conditions == "cold"
    cold_labels = labels_np[cold_mask]
    cold_scores = scores_np[cold_mask]
    if len(np.unique(cold_labels)) > 1:
        results["cold_only_auroc"] = roc_auc_score(cold_labels, cold_scores)
    else:
        results["cold_only_auroc"] = 0.0

    # Warm-only AUROC
    warm_mask = conditions == "warm"
    warm_labels = labels_np[warm_mask]
    warm_scores = scores_np[warm_mask]
    if len(np.unique(warm_labels)) > 1:
        results["warm_only_auroc"] = roc_auc_score(warm_labels, warm_scores)
    else:
        results["warm_only_auroc"] = 0.0

    # Cross-condition: Cold Fault vs Warm Normal (KEY METRIC)
    # This tests if the model can distinguish cold faults from warm normals
    cold_fault_mask = cold_mask & (labels_np == 1)
    warm_good_mask = warm_mask & (labels_np == 0)
    cross_labels = np.concatenate([
        np.ones(cold_fault_mask.sum()),  # cold faults = 1
        np.zeros(warm_good_mask.sum()),  # warm goods = 0
    ])
    cross_scores = np.concatenate([
        scores_np[cold_fault_mask],
        scores_np[warm_good_mask],
    ])
    if len(cross_labels) > 0 and len(np.unique(cross_labels)) > 1:
        results["cold_fault_vs_warm_good_auroc"] = roc_auc_score(cross_labels, cross_scores)
    else:
        results["cold_fault_vs_warm_good_auroc"] = 0.0

    # Add counts
    results["n_cold_samples"] = int(cold_mask.sum())
    results["n_warm_samples"] = int(warm_mask.sum())
    results["n_cold_fault"] = int(cold_fault_mask.sum())
    results["n_warm_good"] = int(warm_good_mask.sum())

    return results


def collect_predictions_ca(
    ca_model: ConditionAwareWinCLIP,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[float], List[str], List[int], List[str], List[Dict]]:
    """Collect predictions from CA-WinCLIP model.

    Returns:
        Tuple of (labels, scores, image_paths, indices, selected_banks, gating_details).
    """
    all_labels = []
    all_scores = []
    all_paths = []
    all_indices = []
    all_selected_banks = []
    all_gating_details = []

    ca_model.eval()

    total_batches = len(dataloader)
    total_samples = len(dataloader.dataset)
    processed = 0

    # Gating statistics
    gating_stats = {"cold": 0, "warm": 0, "correct": 0}

    logger.info(f"  Starting inference on {total_samples} samples ({total_batches} batches)...")
    sys.stdout.flush()

    import time
    start_time = time.time()

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
        for batch_idx, batch in enumerate(dataloader):
            # Get batch data
            images = batch.image.to(device, non_blocking=True)
            batch_labels = batch.gt_label.numpy().tolist()
            batch_paths = list(batch.image_path) if hasattr(batch, 'image_path') else [''] * len(images)

            # Get file indices from paths (e.g., "000123.tiff" -> 123)
            batch_indices = [extract_file_index_from_path(p) for p in batch_paths]

            # Forward pass
            pixel_scores, selected_banks, gating_details = ca_model.forward(
                images, indices=batch_indices
            )

            # Get image-level scores (max over spatial)
            batch_scores = pixel_scores.amax(dim=(-2, -1)).cpu().numpy().tolist()

            # Update gating statistics
            for idx, bank in zip(batch_indices, selected_banks):
                gating_stats[bank] += 1
                gt_bank = get_condition_from_file_index(idx)
                if bank == gt_bank:
                    gating_stats["correct"] += 1

            # Collect results
            all_labels.extend(batch_labels)
            all_scores.extend(batch_scores)
            all_paths.extend(batch_paths)
            all_indices.extend(batch_indices)
            all_selected_banks.extend(selected_banks)
            all_gating_details.extend(gating_details)

            processed += len(images)

            # Log progress for every batch
            elapsed = time.time() - start_time
            gating_acc = gating_stats["correct"] / processed * 100 if processed > 0 else 0
            samples_per_sec = processed / elapsed if elapsed > 0 else 0
            logger.info(
                f"  Batch {batch_idx + 1}/{total_batches} ({processed}/{total_samples}) | "
                f"{elapsed:.1f}s, {samples_per_sec:.1f} samples/sec | "
                f"Gating: cold={gating_stats['cold']}, warm={gating_stats['warm']}, acc={gating_acc:.1f}%"
            )
            sys.stdout.flush()  # Force flush for nohup

    # Final gating summary
    elapsed = time.time() - start_time
    gating_acc = gating_stats["correct"] / processed * 100 if processed > 0 else 0
    logger.info(
        f"  Inference complete ({elapsed:.1f}s). Gating summary: "
        f"cold={gating_stats['cold']}, warm={gating_stats['warm']}, accuracy={gating_acc:.2f}%"
    )
    sys.stdout.flush()

    return all_labels, all_scores, all_paths, all_indices, all_selected_banks, all_gating_details


def collect_predictions_mixed(
    model: WinClipModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[float], List[str], List[int]]:
    """Collect predictions from standard WinCLIP model (mixed refs, no gating).

    Returns:
        Tuple of (labels, scores, image_paths, indices).
    """
    all_labels = []
    all_scores = []
    all_paths = []
    all_indices = []

    model.eval()

    total_batches = len(dataloader)
    total_samples = len(dataloader.dataset)
    processed = 0

    logger.info(f"  Starting mixed-mode inference on {total_samples} samples ({total_batches} batches)...")
    sys.stdout.flush()

    import time
    start_time = time.time()

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
        for batch_idx, batch in enumerate(dataloader):
            # Get batch data
            images = batch.image.to(device, non_blocking=True)
            batch_labels = batch.gt_label.numpy().tolist()
            batch_paths = list(batch.image_path) if hasattr(batch, 'image_path') else [''] * len(images)

            # Get file indices from paths
            batch_indices = [extract_file_index_from_path(p) for p in batch_paths]

            # Forward pass using WinCLIP few-shot scoring
            output = model(images)

            # Get image-level scores from InferenceBatch
            # output.anomaly_map: (B, H, W), output.pred_score: (B,)
            batch_scores = output.pred_score.cpu().numpy().tolist()

            # Collect results
            all_labels.extend(batch_labels)
            all_scores.extend(batch_scores)
            all_paths.extend(batch_paths)
            all_indices.extend(batch_indices)

            processed += len(images)

            # Log progress
            elapsed = time.time() - start_time
            samples_per_sec = processed / elapsed if elapsed > 0 else 0
            logger.info(
                f"  Batch {batch_idx + 1}/{total_batches} ({processed}/{total_samples}) | "
                f"{elapsed:.1f}s, {samples_per_sec:.1f} samples/sec"
            )
            sys.stdout.flush()

    elapsed = time.time() - start_time
    logger.info(f"  Mixed-mode inference complete ({elapsed:.1f}s).")
    sys.stdout.flush()

    return all_labels, all_scores, all_paths, all_indices


def visualize_gating_analysis(
    indices: List[int],
    selected_banks: List[str],
    gating_details: List[Dict],
    domain: str,
    output_dir: Path,
) -> None:
    """Visualize gating analysis results.

    Args:
        indices: File indices (0-999), not dataset indices.
    """
    # Compute gating accuracy
    gt_banks = [get_condition_from_file_index(i) for i in indices]
    correct = sum(s == g for s, g in zip(selected_banks, gt_banks))
    accuracy = correct / len(indices) if indices else 0.0

    # Separate by condition (based on file index)
    cold_correct = sum(
        s == "cold" for s, i in zip(selected_banks, indices) if i in COLD_RANGE
    )
    cold_total = sum(1 for i in indices if i in COLD_RANGE)
    warm_correct = sum(
        s == "warm" for s, i in zip(selected_banks, indices) if i in WARM_RANGE
    )
    warm_total = sum(1 for i in indices if i in WARM_RANGE)

    # Plot gating margin histogram
    margins = []
    for detail in gating_details:
        if "cold" in detail and "warm" in detail:
            margins.append(detail["cold"] - detail["warm"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Margin histogram
    ax = axes[0]
    if margins:
        cold_margins = [m for m, i in zip(margins, indices) if i in COLD_RANGE]
        warm_margins = [m for m, i in zip(margins, indices) if i in WARM_RANGE]
        ax.hist(cold_margins, bins=50, alpha=0.6, label=f'Cold (n={len(cold_margins)})', color='blue')
        ax.hist(warm_margins, bins=50, alpha=0.6, label=f'Warm (n={len(warm_margins)})', color='red')
        ax.axvline(x=0, color='black', linestyle='--', label='Decision boundary')
        ax.set_xlabel('Gating Margin (cold_sim - warm_sim)')
        ax.set_ylabel('Count')
        ax.set_title(f'{domain} - Gating Margin Distribution')
        ax.legend()

    # Accuracy bar chart
    ax = axes[1]
    categories = ['Overall', 'Cold', 'Warm']
    accuracies = [
        accuracy * 100,
        cold_correct / cold_total * 100 if cold_total > 0 else 0,
        warm_correct / warm_total * 100 if warm_total > 0 else 0,
    ]
    bars = ax.bar(categories, accuracies, color=['gray', 'blue', 'red'])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{domain} - Gating Accuracy')
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom')

    plt.tight_layout()

    vis_dir = output_dir / "visualizations" / domain
    vis_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(vis_dir / "gating_analysis.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"  Saved gating analysis to {vis_dir / 'gating_analysis.png'}")


def visualize_score_distribution_by_condition(
    labels: List[int],
    scores: List[float],
    indices: List[int],
    domain: str,
    output_dir: Path,
) -> None:
    """Visualize score distribution split by condition.

    Args:
        indices: File indices (0-999), not dataset indices.
    """
    labels_np = np.array(labels)
    scores_np = np.array(scores)
    conditions = np.array([get_condition_from_file_index(i) for i in indices])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Cold Good vs Cold Fault
    ax = axes[0, 0]
    cold_mask = conditions == "cold"
    cold_good = scores_np[cold_mask & (labels_np == 0)]
    cold_fault = scores_np[cold_mask & (labels_np == 1)]
    if len(cold_good) > 0:
        ax.hist(cold_good, bins=30, alpha=0.6, label=f'Cold Good (n={len(cold_good)})', color='green')
    if len(cold_fault) > 0:
        ax.hist(cold_fault, bins=30, alpha=0.6, label=f'Cold Fault (n={len(cold_fault)})', color='red')
    ax.set_title('Cold Condition')
    ax.legend()

    # Warm Good vs Warm Fault
    ax = axes[0, 1]
    warm_mask = conditions == "warm"
    warm_good = scores_np[warm_mask & (labels_np == 0)]
    warm_fault = scores_np[warm_mask & (labels_np == 1)]
    if len(warm_good) > 0:
        ax.hist(warm_good, bins=30, alpha=0.6, label=f'Warm Good (n={len(warm_good)})', color='green')
    if len(warm_fault) > 0:
        ax.hist(warm_fault, bins=30, alpha=0.6, label=f'Warm Fault (n={len(warm_fault)})', color='red')
    ax.set_title('Warm Condition')
    ax.legend()

    # All Good vs All Fault
    ax = axes[1, 0]
    all_good = scores_np[labels_np == 0]
    all_fault = scores_np[labels_np == 1]
    if len(all_good) > 0:
        ax.hist(all_good, bins=30, alpha=0.6, label=f'Good (n={len(all_good)})', color='green')
    if len(all_fault) > 0:
        ax.hist(all_fault, bins=30, alpha=0.6, label=f'Fault (n={len(all_fault)})', color='red')
    ax.set_title('All Conditions Combined')
    ax.legend()

    # Cross-condition: Cold Fault vs Warm Good (KEY)
    ax = axes[1, 1]
    if len(cold_fault) > 0:
        ax.hist(cold_fault, bins=30, alpha=0.6, label=f'Cold Fault (n={len(cold_fault)})', color='blue')
    if len(warm_good) > 0:
        ax.hist(warm_good, bins=30, alpha=0.6, label=f'Warm Good (n={len(warm_good)})', color='orange')
    ax.set_title('Cross-Condition: Cold Fault vs Warm Good (KEY METRIC)')
    ax.legend()

    plt.suptitle(f'{domain} - Score Distribution by Condition', fontsize=14)
    plt.tight_layout()

    vis_dir = output_dir / "visualizations" / domain
    vis_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(vis_dir / "score_distribution_by_condition.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"  Saved score distribution to {vis_dir / 'score_distribution_by_condition.png'}")


def evaluate_ca_winclip(
    domain: str,
    k_per_bank: int,
    gating_mode: str,
    result_dir: Path,
    dataset_root: Path,
    gpu_id: int = 0,
    class_name: str = "industrial sensor data",
    image_size: int = 240,
    verbose: bool = False,
) -> Dict:
    """Evaluate CA-WinCLIP on a single HDMAP domain.

    Args:
        domain: HDMAP domain name.
        k_per_bank: Number of references per bank.
        gating_mode: "oracle" or "topk".
        result_dir: Directory to save results.
        dataset_root: Root path to HDMAP dataset.
        gpu_id: GPU device ID.
        class_name: Class name for WinCLIP prompts.
        image_size: Target image size.
        verbose: Enable verbose gating logging.

    Returns:
        Dictionary with evaluation results.
    """
    mode_name = f"ca-{gating_mode}-k{k_per_bank}"
    logger.info(f"Evaluating {domain} with {mode_name}...")

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Create output directory
    domain_dir = result_dir / domain / mode_name
    domain_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create base WinCLIP model (text embeddings only)
    logger.info(f"  Creating base WinCLIP model...")
    base_model = WinClipModel(class_name=class_name)
    base_model = base_model.to(device)
    base_model.eval()

    # 2. Load reference images from TEST data
    logger.info(f"  Loading reference images from test data...")
    cold_refs, cold_indices = load_references_from_test(
        dataset_root, domain, "cold", k_per_bank, image_size, device
    )
    warm_refs, warm_indices = load_references_from_test(
        dataset_root, domain, "warm", k_per_bank, image_size, device
    )

    reference_banks = {"cold": cold_refs, "warm": warm_refs}

    # 4. Create test dataloader (needed for both mixed and CA modes)
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

    # 3. Handle mixed mode separately (standard WinCLIP few-shot without gating)
    if gating_mode == "mixed":
        logger.info(f"  Using MIXED mode: all {k_per_bank * 2} refs without gating...")
        # Combine cold and warm refs into single reference set
        mixed_refs = torch.cat([cold_refs, warm_refs], dim=0)
        logger.info(f"    Mixed refs shape: {mixed_refs.shape}")

        # Setup WinCLIP with mixed references
        base_model._collect_visual_embeddings(mixed_refs)
        base_model.k_shot = mixed_refs.shape[0]

        # Collect predictions using standard WinCLIP
        labels, scores, paths, indices = collect_predictions_mixed(
            base_model, test_dataloader, device
        )
        # No gating in mixed mode
        selected_banks = ["mixed"] * len(indices)
        gating_details = [{"mixed": True}] * len(indices)
    else:
        # 3b. Create CA-WinCLIP model with appropriate gating
        logger.info(f"  Creating CA-WinCLIP model with {gating_mode} gating...")
        use_oracle = gating_mode == "oracle"

        # Create gating object based on mode
        gating = None
        if gating_mode == "p90":
            gating = P90IntensityGating(domain=domain)
        elif gating_mode == "random":
            gating = RandomGating(seed=42)  # Fixed seed for reproducibility
        elif gating_mode == "inverse":
            gating = InverseGating()
        # For "oracle" and "topk", gating is None (handled internally)

        ca_model = ConditionAwareWinCLIP(
            base_model=base_model,
            reference_banks=reference_banks,
            gating=gating,
            gating_k=1,  # For 1-2 refs per bank, k=1 is appropriate
            use_oracle=use_oracle,
            verbose_gating=verbose,
        )
        ca_model = ca_model.to(device)

        # 5. Collect predictions
        logger.info(f"  Running inference...")
        labels, scores, paths, indices, selected_banks, gating_details = collect_predictions_ca(
            ca_model, test_dataloader, device
        )

    # 6. Compute metrics
    logger.info(f"  Computing metrics...")
    auroc_metrics = compute_condition_aurocs(labels, scores, indices)

    # Compute gating accuracy (not applicable for mixed mode)
    if gating_mode == "mixed":
        gating_accuracy, gating_correct, gating_total = None, None, len(indices)
    else:
        gating_accuracy, gating_correct, gating_total = ca_model.get_gating_accuracy(
            selected_banks, indices
        )

    # 7. Visualizations
    logger.info(f"  Generating visualizations...")
    visualize_score_distribution_by_condition(labels, scores, indices, domain, result_dir)
    if gating_mode in ["topk", "p90"]:
        visualize_gating_analysis(indices, selected_banks, gating_details, domain, result_dir)

    # 8. Save scores to CSV
    csv_dir = result_dir / "scores"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{domain}_{mode_name}_scores.csv"

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'image_name', 'image_path', 'index', 'condition', 'gt_label',
            'pred_score', 'selected_bank', 'cold_sim', 'warm_sim'
        ])
        for i, (path, idx, label, score, bank, detail) in enumerate(
            zip(paths, indices, labels, scores, selected_banks, gating_details)
        ):
            writer.writerow([
                Path(path).stem if path else '',
                path,
                idx,
                get_condition_from_file_index(idx),
                label,
                f'{score:.6f}',
                bank,
                f'{detail.get("cold", 0):.4f}',
                f'{detail.get("warm", 0):.4f}',
            ])

    logger.info(f"  Saved {len(labels)} scores to {csv_path}")

    # Build results
    results = {
        "domain": domain,
        "k_per_bank": k_per_bank,
        "gating_mode": gating_mode,
        "mode_name": mode_name,
        "class_name": class_name,
        "reference_indices": {
            "cold": cold_indices,
            "warm": warm_indices,
        },
        "auroc_metrics": auroc_metrics,
        "gating_accuracy": gating_accuracy,
        "gating_correct": gating_correct,
        "gating_total": gating_total,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    results_file = domain_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Results for {domain} ({mode_name}):")
    logger.info(f"  Overall AUROC:                    {auroc_metrics['overall_auroc']*100:.2f}%")
    logger.info(f"  Cold-only AUROC:                  {auroc_metrics['cold_only_auroc']*100:.2f}%")
    logger.info(f"  Warm-only AUROC:                  {auroc_metrics['warm_only_auroc']*100:.2f}%")
    logger.info(f"  Cold Fault vs Warm Good AUROC:    {auroc_metrics['cold_fault_vs_warm_good_auroc']*100:.2f}%")
    if gating_mode == "topk":
        logger.info(f"  Gating Accuracy:                  {gating_accuracy*100:.2f}%")
    logger.info(f"{'='*60}\n")

    return results


def main():
    """Run CA-WinCLIP validation on HDMAP."""
    parser = argparse.ArgumentParser(description="CA-WinCLIP HDMAP Validation")
    parser.add_argument(
        "--domain",
        type=str,
        default="domain_C",
        choices=ALL_DOMAINS,
        help="Domain to test",
    )
    parser.add_argument(
        "--k-per-bank",
        type=int,
        default=1,
        help="Number of references per bank (default: 1)",
    )
    parser.add_argument(
        "--gating",
        type=str,
        default="p90",
        choices=["oracle", "topk", "p90", "random", "inverse", "mixed"],
        help="Gating mode: oracle (GT), topk (CLIP), p90 (intensity), random/inverse/mixed (baselines)",
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
        default="./results/winclip_hdmap_ca",
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
        "--compare-all",
        action="store_true",
        help="Run comparison: zero-shot, oracle, topk",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose gating logging (detailed similarity scores)",
    )
    args = parser.parse_args()

    # Resolve dataset root path
    anomalib_root = Path(__file__).parent.parent.parent.parent
    if args.dataset_root.startswith("./"):
        dataset_root = anomalib_root / args.dataset_root[2:]
    else:
        dataset_root = Path(args.dataset_root)

    # Setup result directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(args.result_dir) / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting CA-WinCLIP validation on HDMAP")
    logger.info(f"Domain: {args.domain}")
    logger.info(f"k_per_bank: {args.k_per_bank}")
    logger.info(f"Gating mode: {args.gating}")
    logger.info(f"Dataset root: {dataset_root}")
    logger.info(f"Results will be saved to: {result_dir}")

    # Create experiment.json with initial metadata (status=running)
    create_experiment_meta(result_dir, args, dataset_root)

    try:
        if args.compare_all:
            # Run comparison experiments
            logger.info("\n" + "="*60)
            logger.info("Running comparison experiments...")
            logger.info("="*60 + "\n")

            all_results = []

            # 1. Oracle CA-WinCLIP
            results_oracle = evaluate_ca_winclip(
                domain=args.domain,
                k_per_bank=args.k_per_bank,
                gating_mode="oracle",
                result_dir=result_dir,
                dataset_root=dataset_root,
                gpu_id=args.gpu,
                class_name=args.class_name,
                verbose=args.verbose,
            )
            all_results.append(results_oracle)

            # 2. TopK CA-WinCLIP
            results_topk = evaluate_ca_winclip(
                domain=args.domain,
                k_per_bank=args.k_per_bank,
                gating_mode="topk",
                result_dir=result_dir,
                dataset_root=dataset_root,
                gpu_id=args.gpu,
                class_name=args.class_name,
                verbose=args.verbose,
            )
            all_results.append(results_topk)

            # Print comparison summary
            print("\n" + "="*80)
            print(f"COMPARISON SUMMARY: {args.domain} (k_per_bank={args.k_per_bank})")
            print("="*80)
            print(f"{'Method':<25} {'Overall':<12} {'Cold':<12} {'Warm':<12} {'Cross':<12} {'Gate Acc':<12}")
            print("-"*80)

            for r in all_results:
                m = r["auroc_metrics"]
                gate_acc = r.get("gating_accuracy", 1.0) * 100
                print(f"{r['mode_name']:<25} "
                      f"{m['overall_auroc']*100:>10.2f}% "
                      f"{m['cold_only_auroc']*100:>10.2f}% "
                      f"{m['warm_only_auroc']*100:>10.2f}% "
                      f"{m['cold_fault_vs_warm_good_auroc']*100:>10.2f}% "
                      f"{gate_acc:>10.2f}%")

            print("="*80 + "\n")

            # Save comparison summary
            summary_file = result_dir / "comparison_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"Comparison summary saved to: {summary_file}")

            # Update experiment.json with completed status and results
            update_experiment_meta(result_dir, "completed", results=all_results)

        else:
            # Run single experiment
            single_result = evaluate_ca_winclip(
                domain=args.domain,
                k_per_bank=args.k_per_bank,
                gating_mode=args.gating,
                result_dir=result_dir,
                dataset_root=dataset_root,
                gpu_id=args.gpu,
                class_name=args.class_name,
                verbose=args.verbose,
            )

            # Update experiment.json with completed status and results
            update_experiment_meta(result_dir, "completed", results=single_result)

        print(f"\nResults saved to: {result_dir}")

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"Experiment failed: {error_msg}")

        # Update experiment.json with failed status and error message
        update_experiment_meta(result_dir, "failed", error=error_msg)

        # Re-raise the exception to ensure proper exit code
        raise


if __name__ == "__main__":
    main()
