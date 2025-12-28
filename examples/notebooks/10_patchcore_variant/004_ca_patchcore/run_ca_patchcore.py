#!/usr/bin/env python3
"""004. Condition-Aware PatchCore (CA-PatchCore) Evaluation on HDMAP Dataset.

Implements condition-aware bank selection for PatchCore:
- Separate memory banks for cold and warm reference samples
- Gating modes:
  - oracle: Use ground truth condition (upper bound, 100% accuracy)
  - p90: Use p90 intensity threshold (~94-100% accuracy depending on domain)
  - mixed: No gating, use all references (baseline)
  - random: Random bank selection (~50% accuracy)
  - inverse: Always select wrong bank (0% accuracy, worst case)

Usage:
    # Oracle gating (upper bound)
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py \
        --domain domain_C --k-per-bank 1 --gating oracle

    # P90 intensity gating
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py \
        --domain domain_C --k-per-bank 1 --gating p90

    # Mixed (no gating, use all references - baseline)
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py \
        --domain domain_C --k-per-bank 1 --gating mixed

    # Random gating (~50% accuracy)
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py \
        --domain domain_C --k-per-bank 1 --gating random

    # Inverse gating (worst case)
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py \
        --domain domain_C --k-per-bank 1 --gating inverse
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
ANOMALIB_ROOT = SCRIPT_DIR.parent.parent.parent.parent
DATASET_ROOT = ANOMALIB_ROOT / "datasets" / "HDMAP" / "1000_tiff_minmax"

# P90 thresholds for each domain (from CA-WinCLIP analysis)
P90_THRESHOLDS = {
    'domain_A': 0.2985,
    'domain_B': 0.3128,
    'domain_C': 0.3089,
    'domain_D': 0.2919,
}


def get_config(
    domain: str,
    k_per_bank: int,
    gating: str,
    backbone: str = "vit_base_patch14_dinov2"
) -> dict:
    """Get experiment configuration."""
    backbone_configs = {
        "vit_small_patch14_dinov2": {"layers": ["blocks.8"], "batch_size": 8},
        "vit_base_patch14_dinov2": {"layers": ["blocks.8"], "batch_size": 4},
    }
    bb_config = backbone_configs.get(backbone, {"layers": ["blocks.8"], "batch_size": 4})

    return {
        "backbone": backbone,
        "layers": bb_config["layers"],
        "target_size": (518, 518),
        "resize_method": "resize_bilinear",
        "num_neighbors": 9,
        "batch_size": bb_config["batch_size"],
        "seed": 42,
        "domain": domain,
        "k_per_bank": k_per_bank,
        "gating": gating,
        "p90_threshold": P90_THRESHOLDS.get(domain, 0.30),
    }


def get_exp_name(domain: str, k_per_bank: int, gating: str) -> str:
    """Generate experiment name."""
    return f"ca_patchcore_{gating}_k{k_per_bank}_{domain}"


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

    model = Patchcore(
        backbone=config["backbone"],
        layers=config["layers"],
        pre_trained=True,
        coreset_sampling_ratio=1.0,  # No coreset for few-shot
        num_neighbors=config["num_neighbors"],
        pre_processor=pre_processor,
    )

    return model, datamodule


def get_reference_samples(
    datamodule,
    k_per_bank: int,
    config: dict
) -> Tuple[List, List, List, List]:
    """Get cold and warm reference samples from test/good.

    Returns:
        cold_indices, warm_indices, cold_images, warm_images
    """
    datamodule.setup(stage="test")
    test_dataset = datamodule.test_data

    # Find good samples and sort by file index
    good_samples = []
    for i, item in enumerate(test_dataset):
        if item.gt_label == 0:  # good
            file_idx = int(Path(item.image_path).stem)
            good_samples.append((i, file_idx, item))

    good_samples.sort(key=lambda x: x[1])

    # Select coldest k and warmest k
    cold_samples = good_samples[:k_per_bank]
    warm_samples = good_samples[-k_per_bank:]

    cold_indices = [s[0] for s in cold_samples]
    warm_indices = [s[0] for s in warm_samples]
    cold_file_indices = [s[1] for s in cold_samples]
    warm_file_indices = [s[1] for s in warm_samples]

    print(f"Cold reference samples: file indices {cold_file_indices}")
    print(f"Warm reference samples: file indices {warm_file_indices}")

    return cold_indices, warm_indices, cold_samples, warm_samples


def extract_features(model, images: torch.Tensor, device) -> torch.Tensor:
    """Extract patch embeddings from images.

    Returns:
        embeddings: (N_patches, D) flattened patch embeddings
    """
    model.model.eval()
    pre_processor = model.pre_processor

    with torch.no_grad():
        images = images.to(device)
        normalized = pre_processor(images)

        # Get feature extractor and pooler
        feature_extractor = model.model.feature_extractor
        feature_pooler = model.model.feature_pooler

        # Extract features
        features = feature_extractor(normalized)
        features = {layer: feature_pooler(feature) for layer, feature in features.items()}

        # Generate embedding
        embedding = model.model.generate_embedding(features)

        # Reshape to (N_patches, D)
        embedding = model.model.reshape_embedding(embedding)

    return embedding


def build_memory_banks(
    model,
    datamodule,
    cold_indices: List[int],
    warm_indices: List[int],
    config: dict,
    device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build separate memory banks for cold and warm references."""
    test_dataset = datamodule.test_data

    cold_embeddings = []
    warm_embeddings = []

    model.model.to(device)
    model.model.eval()

    # Extract cold embeddings
    print(f"Building cold memory bank from {len(cold_indices)} samples...")
    for idx in cold_indices:
        item = test_dataset[idx]
        file_name = Path(item.image_path).name
        print(f"  - {file_name}")
        image = item.image.unsqueeze(0)
        emb = extract_features(model, image, device)
        cold_embeddings.append(emb)

    cold_bank = torch.cat(cold_embeddings, dim=0)
    print(f"Cold memory bank shape: {cold_bank.shape}")

    # Extract warm embeddings
    print(f"Building warm memory bank from {len(warm_indices)} samples...")
    for idx in warm_indices:
        item = test_dataset[idx]
        file_name = Path(item.image_path).name
        print(f"  - {file_name}")
        image = item.image.unsqueeze(0)
        emb = extract_features(model, image, device)
        warm_embeddings.append(emb)

    warm_bank = torch.cat(warm_embeddings, dim=0)
    print(f"Warm memory bank shape: {warm_bank.shape}")

    return cold_bank, warm_bank


def compute_p90(image: torch.Tensor) -> float:
    """Compute 90th percentile of image pixels."""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    return float(np.percentile(image, 90))


def select_bank_oracle(file_idx: int) -> str:
    """Oracle gating: use ground truth condition."""
    return "cold" if file_idx < 500 else "warm"


def select_bank_p90(image: torch.Tensor, threshold: float) -> str:
    """P90 gating: use intensity percentile."""
    p90 = compute_p90(image)
    return "cold" if p90 <= threshold else "warm"


def select_bank_random(rng: np.random.RandomState) -> str:
    """Random gating: 50% probability for each bank."""
    return "cold" if rng.random() < 0.5 else "warm"


def select_bank_inverse(file_idx: int) -> str:
    """Inverse gating: always select the wrong bank (worst case)."""
    gt_condition = "cold" if file_idx < 500 else "warm"
    return "warm" if gt_condition == "cold" else "cold"


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distances (same as PatchCore implementation).

    Uses efficient matrix computation: sqrt(|x|^2 - 2*x@y.T + |y|^2.T)
    """
    x_norm = x.pow(2).sum(dim=-1, keepdim=True)
    y_norm = y.pow(2).sum(dim=-1, keepdim=True)
    res = x_norm - 2 * torch.matmul(x, y.transpose(-2, -1)) + y_norm.transpose(-2, -1)
    return res.clamp_min_(0).sqrt_()


def compute_anomaly_score(
    query_embedding: torch.Tensor,
    memory_bank: torch.Tensor,
    num_neighbors: int = 9
) -> torch.Tensor:
    """Compute anomaly score using nearest neighbor distance.

    Matches the standard PatchCore scoring:
    1. Find nearest neighbor distance for each patch (n_neighbors=1 for patch scores)
    2. Image score = max of patch scores (with optional weighting for num_neighbors > 1)
    """
    # query_embedding: (N_patches, D)
    # memory_bank: (M_patches, D)

    # Compute pairwise distances (same as PatchCore)
    distances = euclidean_dist(query_embedding, memory_bank)  # (N, M)

    # Get nearest neighbor distance for each patch
    patch_scores, locations = distances.min(dim=1)  # (N,)

    # For num_neighbors=1, just return max patch score (same as PatchCore)
    if num_neighbors == 1:
        return patch_scores.max()

    # For num_neighbors > 1, apply weighted scoring (same as PatchCore paper)
    # Find the patch with largest score
    max_patch_idx = patch_scores.argmax()
    max_patch_feature = query_embedding[max_patch_idx].unsqueeze(0)  # (1, D)
    score = patch_scores[max_patch_idx]  # s^* in paper

    # Find support samples of the nearest neighbor in memory bank
    nn_index = locations[max_patch_idx]
    nn_sample = memory_bank[nn_index].unsqueeze(0)  # (1, D)

    # Find k-nearest neighbors of the nn_sample in memory bank
    k = min(num_neighbors, memory_bank.shape[0])
    nn_distances = euclidean_dist(nn_sample, memory_bank)  # (1, M)
    _, support_indices = nn_distances.topk(k, dim=1, largest=False)  # (1, k)

    # Compute distances from max patch to support samples
    support_features = memory_bank[support_indices.squeeze(0)]  # (k, D)
    support_distances = euclidean_dist(max_patch_feature, support_features)  # (1, k)

    # Apply softmax weighting
    weights = (1 - F.softmax(support_distances.squeeze(0), dim=0))[0]

    return weights * score


def run_inference(
    model,
    datamodule,
    cold_bank: torch.Tensor,
    warm_bank: torch.Tensor,
    config: dict,
    device
) -> List[Dict]:
    """Run inference with condition-aware bank selection."""
    datamodule.setup(stage="test")
    test_dataset = datamodule.test_data
    gating_mode = config["gating"]
    p90_threshold = config["p90_threshold"]
    n_neighbors = config["num_neighbors"]

    # For mixed mode, concatenate banks
    if gating_mode == "mixed":
        combined_bank = torch.cat([cold_bank, warm_bank], dim=0)
        print(f"Mixed mode: combined bank shape {combined_bank.shape}")

    # For random mode, initialize RNG
    if gating_mode == "random":
        rng = np.random.RandomState(config["seed"])

    results = []
    gating_stats = {"correct": 0, "total": 0, "cold_selected": 0, "warm_selected": 0}

    model.model.to(device)
    model.model.eval()

    for i in tqdm(range(len(test_dataset)), desc="Inference"):
        item = test_dataset[i]
        image = item.image.unsqueeze(0)
        file_idx = int(Path(item.image_path).stem)
        gt_label = item.gt_label.item()
        is_fault = "fault" in str(item.image_path).lower()

        # Ground truth condition
        gt_condition = "cold" if file_idx < 500 else "warm"

        # Select bank based on gating mode
        if gating_mode == "oracle":
            selected_bank = select_bank_oracle(file_idx)
        elif gating_mode == "p90":
            selected_bank = select_bank_p90(item.image, p90_threshold)
        elif gating_mode == "random":
            selected_bank = select_bank_random(rng)
        elif gating_mode == "inverse":
            selected_bank = select_bank_inverse(file_idx)
        elif gating_mode == "mixed":
            selected_bank = "mixed"
        else:
            raise ValueError(f"Unknown gating mode: {gating_mode}")

        # Track gating accuracy
        if gating_mode != "mixed":
            gating_stats["total"] += 1
            if selected_bank == gt_condition:
                gating_stats["correct"] += 1
            if selected_bank == "cold":
                gating_stats["cold_selected"] += 1
            else:
                gating_stats["warm_selected"] += 1

        # Extract features
        query_emb = extract_features(model, image, device).squeeze(0)

        # Compute anomaly score
        if gating_mode == "mixed":
            score = compute_anomaly_score(query_emb, combined_bank, n_neighbors)
        else:
            bank = cold_bank if selected_bank == "cold" else warm_bank
            score = compute_anomaly_score(query_emb, bank, n_neighbors)

        results.append({
            "file_idx": file_idx,
            "gt_label": gt_label,
            "gt_condition": gt_condition,
            "selected_bank": selected_bank,
            "score": score.item(),
            "is_fault": is_fault,
        })

    # Print gating stats
    if gating_mode != "mixed":
        acc = gating_stats["correct"] / gating_stats["total"] * 100
        print(f"\nGating Accuracy: {acc:.2f}% ({gating_stats['correct']}/{gating_stats['total']})")
        print(f"  Cold selected: {gating_stats['cold_selected']}, Warm selected: {gating_stats['warm_selected']}")

    return results, gating_stats


def analyze_results(results: List[Dict], domain: str) -> Dict:
    """Analyze results with cold/warm breakdown."""
    all_scores = np.array([r["score"] for r in results])
    all_labels = np.array([r["gt_label"] for r in results])
    all_conditions = np.array([r["gt_condition"] for r in results])

    cold_mask = all_conditions == "cold"
    warm_mask = all_conditions == "warm"

    # Find optimal threshold
    thresholds = np.percentile(all_scores, np.arange(0, 101, 1))
    best_acc, best_threshold = 0, 0
    for thresh in thresholds:
        preds = (all_scores >= thresh).astype(int)
        acc = accuracy_score(all_labels, preds)
        if acc > best_acc:
            best_acc, best_threshold = acc, thresh

    overall_preds = (all_scores >= best_threshold).astype(int)

    analysis = {
        "overall": {
            "accuracy": float(accuracy_score(all_labels, overall_preds)),
            "auroc": float(roc_auc_score(all_labels, all_scores)),
            "threshold": float(best_threshold),
            "n_samples": int(len(all_labels)),
        },
        "cold": {
            "accuracy": float(accuracy_score(
                all_labels[cold_mask],
                (all_scores[cold_mask] >= best_threshold).astype(int)
            )),
            "auroc": float(roc_auc_score(all_labels[cold_mask], all_scores[cold_mask]))
                if len(np.unique(all_labels[cold_mask])) > 1 else 0.0,
            "n_samples": int(cold_mask.sum()),
        },
        "warm": {
            "accuracy": float(accuracy_score(
                all_labels[warm_mask],
                (all_scores[warm_mask] >= best_threshold).astype(int)
            )),
            "auroc": float(roc_auc_score(all_labels[warm_mask], all_scores[warm_mask]))
                if len(np.unique(all_labels[warm_mask])) > 1 else 0.0,
            "n_samples": int(warm_mask.sum()),
        },
    }

    return analysis, all_scores, all_labels, cold_mask, warm_mask, best_threshold


def plot_score_distribution(
    all_scores: np.ndarray,
    all_labels: np.ndarray,
    cold_mask: np.ndarray,
    warm_mask: np.ndarray,
    threshold: float,
    output_dir: Path,
    domain: str
):
    """Plot and save score distribution histograms."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # All samples
    ax = axes[0]
    ax.hist(all_scores[all_labels == 0], bins=50, alpha=0.7, label="Good", color="green", density=True)
    ax.hist(all_scores[all_labels == 1], bins=50, alpha=0.7, label="Fault", color="red", density=True)
    ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold:.3f}")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title(f"{domain} - All Samples")
    ax.legend()

    # Cold samples
    ax = axes[1]
    cold_scores = all_scores[cold_mask]
    cold_labels = all_labels[cold_mask]
    ax.hist(cold_scores[cold_labels == 0], bins=50, alpha=0.7, label="Good", color="green", density=True)
    ax.hist(cold_scores[cold_labels == 1], bins=50, alpha=0.7, label="Fault", color="red", density=True)
    ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold:.3f}")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title(f"{domain} - Cold Samples")
    ax.legend()

    # Warm samples
    ax = axes[2]
    warm_scores = all_scores[warm_mask]
    warm_labels = all_labels[warm_mask]
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


def print_results(analysis: Dict, domain: str, gating: str, k_per_bank: int):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"Results: {domain} | Gating={gating} | k={k_per_bank}")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Overall':>12} {'Cold':>12} {'Warm':>12}")
    print("-" * 55)
    print(f"{'Accuracy':<15} {analysis['overall']['accuracy']:>12.2%} "
          f"{analysis['cold']['accuracy']:>12.2%} {analysis['warm']['accuracy']:>12.2%}")
    print(f"{'AUROC':<15} {analysis['overall']['auroc']:>12.2%} "
          f"{analysis['cold']['auroc']:>12.2%} {analysis['warm']['auroc']:>12.2%}")


def main():
    parser = argparse.ArgumentParser(description="CA-PatchCore on HDMAP")
    parser.add_argument("--domain", type=str, required=True,
                        choices=["domain_A", "domain_B", "domain_C", "domain_D"])
    parser.add_argument("--k-per-bank", type=int, default=1,
                        help="Number of reference samples per bank (cold/warm)")
    parser.add_argument("--gating", type=str, default="oracle",
                        choices=["oracle", "p90", "mixed", "random", "inverse"],
                        help="Gating method: oracle (GT), p90 (intensity), mixed (no gating), random (~50%%), inverse (0%%)")
    parser.add_argument("--backbone", type=str, default="vit_base_patch14_dinov2",
                        choices=["vit_small_patch14_dinov2", "vit_base_patch14_dinov2"])
    args = parser.parse_args()

    # Config and output
    config = get_config(args.domain, args.k_per_bank, args.gating, args.backbone)
    exp_name = get_exp_name(args.domain, args.k_per_bank, args.gating)
    output_dir = create_output_dir(exp_name)

    print(f"\nExperiment: {exp_name}")
    print(f"Output: {output_dir}")
    print(f"Config: k_per_bank={args.k_per_bank}, gating={args.gating}")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, datamodule = setup_model_and_data(args.domain, config)

    # Get reference samples
    cold_indices, warm_indices, _, _ = get_reference_samples(
        datamodule, args.k_per_bank, config
    )

    # Build memory banks
    cold_bank, warm_bank = build_memory_banks(
        model, datamodule, cold_indices, warm_indices, config, device
    )

    # Run inference
    results, gating_stats = run_inference(
        model, datamodule, cold_bank, warm_bank, config, device
    )

    # Analyze results
    analysis, all_scores, all_labels, cold_mask, warm_mask, threshold = analyze_results(
        results, args.domain
    )

    # Add metadata
    analysis["domain"] = args.domain
    analysis["gating"] = args.gating
    analysis["k_per_bank"] = args.k_per_bank
    if args.gating != "mixed":
        analysis["gating_accuracy"] = gating_stats["correct"] / gating_stats["total"]

    # Print and save
    print_results(analysis, args.domain, args.gating, args.k_per_bank)

    with open(output_dir / "results.json", "w") as f:
        json.dump(analysis, f, indent=2)

    # Save detailed results
    with open(output_dir / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot score distribution
    plot_score_distribution(
        all_scores, all_labels, cold_mask, warm_mask, threshold, output_dir, args.domain
    )

    print(f"\nSaved to: {output_dir}")

    return analysis


if __name__ == "__main__":
    main()
