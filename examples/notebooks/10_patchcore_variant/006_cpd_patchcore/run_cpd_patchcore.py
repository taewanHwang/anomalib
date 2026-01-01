#!/usr/bin/env python3
"""006. CPD-PatchCore (Contextual Patch Descriptor PatchCore).

Adds horizontal context to patch features before kNN matching.
This helps detect thin horizontal faults by aggregating weak per-patch signals.

Key idea:
    Instead of: d(f[i,j], bank)
    Use:        d([f[i,j] | ctx[i,j]], bank)

    where ctx[i,j] = aggregate(f[i, j-k:j+k+1]) along horizontal axis

Usage:
    # Basic CPD with mean aggregation, k=2
    CUDA_VISIBLE_DEVICES=0 python run_cpd_patchcore.py \
        --domain domain_C --k-ref 16 --context-k 2 --aggregation mean

    # CPD with max aggregation
    CUDA_VISIBLE_DEVICES=0 python run_cpd_patchcore.py \
        --domain domain_C --k-ref 16 --context-k 2 --aggregation max

    # CPD with std feature
    CUDA_VISIBLE_DEVICES=0 python run_cpd_patchcore.py \
        --domain domain_C --k-ref 16 --context-k 2 --feature-mode f_ctx_std
"""

import argparse
import json
import math
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
ANOMALIB_ROOT = SCRIPT_DIR.parent.parent.parent.parent
DATASET_ROOT = ANOMALIB_ROOT / "datasets" / "HDMAP" / "1000_tiff_minmax"


def get_config(
    domain: str,
    k_ref: int,
    context_k: int,
    aggregation: str,
    feature_mode: str,
    trim_ratio: float = 0.2,
    attention_alpha: float = 5.0,
    sampling_mode: str = "adjacent",
    distance_decay: float = 1.0,
    multik_mode: str = None,
    multik_list: list = None,
    multik_weights: list = None,
    ablation_mode: str = None,
    ablation_n_concat: int = 1,
    backbone: str = "vit_base_patch14_dinov2",
    layers: list = None,
    resize_method: str = "resize_bilinear",
    coreset_ratio: float = 1.0,
    ref_source: str = "test"
) -> dict:
    """Get experiment configuration."""
    if layers is None:
        layers = ["blocks.8"]
    return {
        "backbone": backbone,
        "layers": layers,
        "target_size": (518, 518),
        "resize_method": resize_method,
        "coreset_sampling_ratio": coreset_ratio,
        "num_neighbors": 9,
        "batch_size": 4,
        "seed": 42,
        "domain": domain,
        "k_ref": k_ref,  # Number of reference samples (total, split cold/warm)
        "context_k": context_k,  # Horizontal context radius
        "aggregation": aggregation,  # mean, max, gaussian, median, trimmed, attention
        "feature_mode": feature_mode,  # f_only, f_ctx, f_ctx_std, ctx_only
        "trim_ratio": trim_ratio,  # For trimmed mean
        "attention_alpha": attention_alpha,  # For attention-weighted
        "sampling_mode": sampling_mode,  # adjacent, random, distance_weighted
        "distance_decay": distance_decay,  # For distance_weighted sampling
        "multik_mode": multik_mode,  # weighted, concat (None = single k)
        "multik_list": multik_list,  # List of k values for multi-k
        "multik_weights": multik_weights,  # Weights for weighted mode
        "ablation_mode": ablation_mode,  # random, global, zeros (for dimension ablation)
        "ablation_n_concat": ablation_n_concat,  # 1 for 2D, 2 for 3D
        "ref_source": ref_source,  # test (from test/good) or train (from train/good)
    }


def get_layer_suffix(layers: list) -> str:
    """Generate layer suffix for experiment name.

    Examples:
        ["blocks.8"] -> "" (default, no suffix)
        ["blocks.8", "blocks.11"] -> "_layers_b8_b11"
        ["blocks.4", "blocks.8", "blocks.11"] -> "_layers_b4_b8_b11"
    """
    if layers is None or layers == ["blocks.8"]:
        return ""  # Default single layer, no suffix needed

    # Convert "blocks.X" to "bX"
    short_names = []
    for layer in layers:
        if layer.startswith("blocks."):
            short_names.append(f"b{layer.split('.')[1]}")
        else:
            short_names.append(layer.replace(".", "_"))

    return f"_layers_{'_'.join(short_names)}"


def get_exp_name(config: dict) -> str:
    """Generate experiment name."""
    layer_suffix = get_layer_suffix(config.get('layers'))

    # Check for ablation mode first
    ablation_mode = config.get('ablation_mode')
    if ablation_mode is not None:
        n_concat = config.get('ablation_n_concat', 1)
        dim_str = f"{n_concat+1}D"  # 1 concat -> 2D, 2 concat -> 3D
        return (f"ablation_{ablation_mode}_{dim_str}_"
                f"{config['domain']}_ref{config['k_ref']}{layer_suffix}")

    # ref_condition suffix (only if not balanced)
    ref_cond = config.get('ref_condition', 'balanced')
    ref_suffix = f"_{ref_cond}" if ref_cond != "balanced" else ""

    # ref_source suffix (only if train mode - test is default/legacy)
    ref_source = config.get('ref_source', 'test')
    source_suffix = "_trainref" if ref_source == "train" else ""

    # Check for multi-k mode
    multik_mode = config.get('multik_mode')
    if multik_mode is not None:
        k_list = config.get('multik_list', [2, 3])
        k_str = "_".join(map(str, k_list))
        return (f"cpd_multik_{multik_mode}_k{k_str}_"
                f"{config['domain']}_ref{config['k_ref']}{ref_suffix}{source_suffix}{layer_suffix}")

    # f_only 모드는 CPD 비활성화 = vanilla PatchCore
    if config['feature_mode'] == 'f_only':
        return (f"vanilla_patchcore_{config['domain']}_ref{config['k_ref']}{ref_suffix}{source_suffix}{layer_suffix}")

    sampling = config.get('sampling_mode', 'adjacent')
    if sampling == 'adjacent':
        return (f"cpd_{config['aggregation']}_k{config['context_k']}_"
                f"{config['feature_mode']}_{config['domain']}_ref{config['k_ref']}{ref_suffix}{source_suffix}{layer_suffix}")
    elif sampling == 'distance_weighted':
        decay = config.get('distance_decay', 1.0)
        return (f"cpd_{config['aggregation']}_k{config['context_k']}_"
                f"{sampling}_d{decay}_{config['feature_mode']}_"
                f"{config['domain']}_ref{config['k_ref']}{ref_suffix}{source_suffix}{layer_suffix}")
    else:  # random
        return (f"cpd_{config['aggregation']}_k{config['context_k']}_"
                f"{sampling}_{config['feature_mode']}_"
                f"{config['domain']}_ref{config['k_ref']}{ref_suffix}{source_suffix}{layer_suffix}")


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
        coreset_sampling_ratio=config["coreset_sampling_ratio"],
        num_neighbors=config["num_neighbors"],
        pre_processor=pre_processor,
    )

    return model, datamodule


def extract_features(model, images: torch.Tensor, device) -> torch.Tensor:
    """Extract patch embeddings from images.

    Returns:
        embeddings: (B, D, H, W) spatial feature map
    """
    model.model.eval()
    pre_processor = model.pre_processor

    with torch.no_grad():
        images = images.to(device)
        normalized = pre_processor(images)

        feature_extractor = model.model.feature_extractor
        feature_pooler = model.model.feature_pooler

        features = feature_extractor(normalized)
        features = {layer: feature_pooler(feature) for layer, feature in features.items()}

        # Get spatial feature map (B, D, H, W)
        embedding = model.model.generate_embedding(features)

    return embedding  # (B, D, H, W)


def get_gaussian_kernel(k: int, sigma: float = None) -> torch.Tensor:
    """Create 1D Gaussian kernel for weighted averaging."""
    if sigma is None:
        sigma = k / 2.0
    x = torch.arange(-k, k + 1, dtype=torch.float32)
    kernel = torch.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def sample_row_indices_random(
    W: int,
    n_samples: int,
    seed: int = 42
) -> torch.Tensor:
    """Generate random sampling indices for each position in a row.

    For each position j in [0, W), sample n_samples indices from the entire row
    (excluding j itself).

    Args:
        W: row width
        n_samples: number of patches to sample per position
        seed: random seed for reproducibility

    Returns:
        indices: (W, n_samples) tensor of sampled indices
    """
    torch.manual_seed(seed)
    indices = torch.zeros(W, n_samples, dtype=torch.long)

    for j in range(W):
        # All positions except j
        candidates = torch.cat([torch.arange(0, j), torch.arange(j+1, W)])
        # Randomly sample n_samples (with replacement if needed)
        n_actual = min(n_samples, len(candidates))
        perm = torch.randperm(len(candidates))[:n_actual]
        sampled = candidates[perm]
        # Pad if needed
        if n_actual < n_samples:
            sampled = torch.cat([sampled, sampled[:n_samples - n_actual]])
        indices[j] = sampled

    return indices


def sample_row_indices_distance_weighted(
    W: int,
    n_samples: int,
    distance_decay: float = 1.0,
    seed: int = 42
) -> torch.Tensor:
    """Generate distance-weighted random sampling indices for each position.

    For each position j, sample n_samples indices with probability proportional
    to 1 / |i - j|^decay (closer patches have higher probability).

    Args:
        W: row width
        n_samples: number of patches to sample per position
        distance_decay: power for distance weighting (higher = more local)
        seed: random seed for reproducibility

    Returns:
        indices: (W, n_samples) tensor of sampled indices
    """
    torch.manual_seed(seed)
    indices = torch.zeros(W, n_samples, dtype=torch.long)

    for j in range(W):
        # All positions except j
        candidates = torch.cat([torch.arange(0, j), torch.arange(j+1, W)])

        # Distance-based weights: p ~ 1 / d^decay
        distances = torch.abs(candidates.float() - j)
        weights = 1.0 / (distances ** distance_decay + 1e-6)
        weights = weights / weights.sum()  # Normalize to probabilities

        # Sample with replacement according to weights
        sampled_idx = torch.multinomial(weights, n_samples, replacement=True)
        indices[j] = candidates[sampled_idx]

    return indices


def aggregate_with_random_sampling(
    features: torch.Tensor,
    sample_indices: torch.Tensor,
    aggregation: str = "mean"
) -> torch.Tensor:
    """Aggregate features using random sampling indices.

    Args:
        features: (B, D, H, W) spatial feature map
        sample_indices: (W, n_samples) indices for each position
        aggregation: mean, max, median

    Returns:
        ctx: (B, D, H, W) aggregated context features
    """
    B, D, H, W = features.shape
    n_samples = sample_indices.shape[1]
    device = features.device

    # Move indices to device
    sample_indices = sample_indices.to(device)  # (W, n_samples)

    # Gather features for all sampled positions
    # features: (B, D, H, W) -> we need to gather along W dimension for each position
    # Expand indices for gathering: (W, n_samples) -> (B, D, H, W, n_samples)
    indices_expanded = sample_indices.view(1, 1, 1, W, n_samples).expand(B, D, H, -1, -1)

    # Gather: features[:,:,:,sample_indices[j]] for each j
    # features_expanded: (B, D, H, W, 1)
    # We need to gather from the W dimension
    gathered = torch.zeros(B, D, H, W, n_samples, device=device)
    for i in range(n_samples):
        idx = sample_indices[:, i]  # (W,)
        gathered[:, :, :, :, i] = features[:, :, :, idx]  # gather from W dim

    # Aggregate along the last dimension
    if aggregation == "mean":
        ctx = gathered.mean(dim=-1)
    elif aggregation == "max":
        ctx = gathered.max(dim=-1).values
    elif aggregation == "median":
        ctx = gathered.median(dim=-1).values
    else:
        ctx = gathered.mean(dim=-1)  # default to mean

    return ctx


def apply_ablation_concat(
    features: torch.Tensor,
    ablation_mode: str = "random",
    n_concat: int = 1,
    seed: int = 42
) -> torch.Tensor:
    """Apply ablation: concat with random or global average embeddings.

    This is to test whether the Multi-k improvement is due to increased
    dimensionality alone (2D→3D) or due to meaningful context information.

    Args:
        features: (B, D, H, W) spatial feature map
        ablation_mode: 'random' (random vectors) or 'global' (global avg pool)
        n_concat: number of D-dim vectors to concat (1 for 2D, 2 for 3D)
        seed: random seed for reproducibility

    Returns:
        features_out: (B, (1+n_concat)*D, H, W)
    """
    B, D, H, W = features.shape
    device = features.device

    concat_list = [features]

    for i in range(n_concat):
        if ablation_mode == "random":
            # Random embedding (different per spatial position, same across batch)
            torch.manual_seed(seed + i)
            random_emb = torch.randn(1, D, H, W, device=device)
            random_emb = random_emb.expand(B, -1, -1, -1)
            concat_list.append(random_emb)

        elif ablation_mode == "global":
            # Global average pooling (same for all spatial positions)
            global_avg = features.mean(dim=(2, 3), keepdim=True)  # (B, D, 1, 1)
            global_avg = global_avg.expand(-1, -1, H, W)  # (B, D, H, W)
            concat_list.append(global_avg)

        elif ablation_mode == "zeros":
            # Zero padding
            zeros = torch.zeros(B, D, H, W, device=device)
            concat_list.append(zeros)

        else:
            raise ValueError(f"Unknown ablation_mode: {ablation_mode}")

    return torch.cat(concat_list, dim=1)


def apply_multik_context(
    features: torch.Tensor,
    k_list: list = [2, 3],
    multik_mode: str = "weighted",
    weights: list = None
) -> torch.Tensor:
    """Apply Multi-k CPD: combine contexts from multiple scales.

    Args:
        features: (B, D, H, W) spatial feature map
        k_list: list of context radii to use (e.g., [2, 3])
        multik_mode: 'weighted' (average) or 'concat'
        weights: weights for weighted mode (default: equal weights)

    Returns:
        features_cpd: (B, D', H, W) where D' = 2D (weighted) or (1+len(k_list))*D (concat)
    """
    B, D, H, W = features.shape

    # Compute context for each k
    ctx_list = []
    for k in k_list:
        ctx_k = F.avg_pool2d(features, kernel_size=(1, 2*k+1), stride=1, padding=(0, k))
        ctx_list.append(ctx_k)

    if multik_mode == "weighted":
        # Weighted average of contexts
        if weights is None:
            weights = [1.0 / len(k_list)] * len(k_list)  # Equal weights

        ctx_combined = torch.zeros_like(ctx_list[0])
        for w, ctx_k in zip(weights, ctx_list):
            ctx_combined = ctx_combined + w * ctx_k

        return torch.cat([features, ctx_combined], dim=1)  # (B, 2D, H, W)

    elif multik_mode == "concat":
        # Concatenate all contexts
        return torch.cat([features] + ctx_list, dim=1)  # (B, (1+len(k_list))*D, H, W)

    else:
        raise ValueError(f"Unknown multik_mode: {multik_mode}")


def apply_horizontal_context(
    features: torch.Tensor,
    k: int,
    aggregation: str = "mean",
    feature_mode: str = "f_ctx",
    trim_ratio: float = 0.2,
    attention_alpha: float = 5.0,
    sampling_mode: str = "adjacent",
    distance_decay: float = 1.0,
    seed: int = 42,
    multik_mode: str = None,
    multik_list: list = None,
    multik_weights: list = None,
    ablation_mode: str = None,
    ablation_n_concat: int = 1
) -> torch.Tensor:
    """Apply horizontal context aggregation to features.

    Args:
        features: (B, D, H, W) spatial feature map
        k: context radius (total 2k+1 patches horizontally for adjacent mode,
           or 2k patches for random modes)
        aggregation: mean, max, gaussian, median, trimmed, attention
        feature_mode: f_only, f_ctx, f_ctx_std, ctx_only
        trim_ratio: ratio to trim for trimmed mean (default 0.2 = 20% each side)
        attention_alpha: temperature for attention weights (higher = sharper)
        sampling_mode: adjacent (default), random, distance_weighted
        distance_decay: power for distance weighting in distance_weighted mode
        seed: random seed for reproducibility in random modes
        ablation_mode: random, global, zeros (for dimension ablation study)
        ablation_n_concat: number of D-dim vectors to concat for ablation

    Returns:
        features_cpd: (B, D', H, W) where D' depends on feature_mode
    """
    B, D, H, W = features.shape

    if feature_mode == "f_only":
        return features

    # Handle ablation mode (for dimension ablation study)
    if ablation_mode is not None:
        return apply_ablation_concat(
            features,
            ablation_mode=ablation_mode,
            n_concat=ablation_n_concat,
            seed=seed
        )

    # Handle Multi-k mode
    if multik_mode is not None:
        return apply_multik_context(
            features,
            k_list=multik_list if multik_list else [2, 3],
            multik_mode=multik_mode,
            weights=multik_weights
        )

    # Handle random sampling modes
    if sampling_mode in ["random", "distance_weighted"]:
        n_samples = 2 * k  # Sample 2k patches (excluding center)

        if sampling_mode == "random":
            sample_indices = sample_row_indices_random(W, n_samples, seed)
        else:  # distance_weighted
            sample_indices = sample_row_indices_distance_weighted(
                W, n_samples, distance_decay, seed
            )

        ctx = aggregate_with_random_sampling(features, sample_indices, aggregation)

        # Handle feature modes
        if feature_mode == "f_ctx":
            return torch.cat([features, ctx], dim=1)
        elif feature_mode == "f_ctx_std":
            # Compute std using the same sampling
            gathered = torch.zeros(B, D, H, W, n_samples, device=features.device)
            sample_indices = sample_indices.to(features.device)
            for i in range(n_samples):
                idx = sample_indices[:, i]
                gathered[:, :, :, :, i] = features[:, :, :, idx]
            std = gathered.std(dim=-1)
            return torch.cat([features, ctx, std], dim=1)
        elif feature_mode == "ctx_only":
            return ctx
        else:
            return torch.cat([features, ctx], dim=1)

    # Compute horizontal context
    if aggregation == "mean":
        # Average pooling along width (horizontal)
        ctx = F.avg_pool2d(features, kernel_size=(1, 2*k+1), stride=1, padding=(0, k))

    elif aggregation == "max":
        # Max pooling along width
        ctx = F.max_pool2d(features, kernel_size=(1, 2*k+1), stride=1, padding=(0, k))

    elif aggregation == "gaussian":
        # Gaussian weighted average
        kernel = get_gaussian_kernel(k).to(features.device)
        kernel = kernel.view(1, 1, 1, -1).expand(D, 1, 1, -1)
        ctx = F.conv2d(features, kernel, padding=(0, k), groups=D)

    elif aggregation == "median":
        # Median (requires unfold + median)
        unfolded = F.unfold(features, kernel_size=(1, 2*k+1), padding=(0, k))
        unfolded = unfolded.view(B, D, 2*k+1, H, W).permute(0, 1, 3, 4, 2)
        ctx = unfolded.median(dim=-1).values

    elif aggregation == "trimmed":
        # Trimmed mean: remove top/bottom trim_ratio% and average the rest
        # Unfold to get neighbors
        unfolded = F.unfold(features, kernel_size=(1, 2*k+1), padding=(0, k))
        unfolded = unfolded.view(B, D, 2*k+1, H, W).permute(0, 1, 3, 4, 2)  # (B, D, H, W, 2k+1)

        # Sort along last dimension
        sorted_vals, _ = unfolded.sort(dim=-1)

        # Calculate trim indices
        n_neighbors = 2 * k + 1
        n_trim = max(1, int(n_neighbors * trim_ratio))
        # Keep middle values (trim from both ends)
        trimmed = sorted_vals[..., n_trim:-n_trim] if n_trim > 0 else sorted_vals

        # Mean of remaining values
        ctx = trimmed.mean(dim=-1)

    elif aggregation == "attention":
        # Attention-weighted average using cosine similarity
        # Unfold to get neighbors: (B, D, H, W) -> (B, D, H, W, 2k+1)
        unfolded = F.unfold(features, kernel_size=(1, 2*k+1), padding=(0, k))
        unfolded = unfolded.view(B, D, 2*k+1, H, W).permute(0, 1, 3, 4, 2)  # (B, D, H, W, 2k+1)

        # Center feature (the patch itself)
        center = features.unsqueeze(-1)  # (B, D, H, W, 1)

        # Compute cosine similarity between center and each neighbor
        # Normalize features
        center_norm = F.normalize(center, dim=1)  # (B, D, H, W, 1)
        neighbors_norm = F.normalize(unfolded, dim=1)  # (B, D, H, W, 2k+1)

        # Cosine similarity: sum over D dimension
        cos_sim = (center_norm * neighbors_norm).sum(dim=1)  # (B, H, W, 2k+1)

        # Attention weights with temperature
        attention_weights = F.softmax(attention_alpha * cos_sim, dim=-1)  # (B, H, W, 2k+1)

        # Weighted sum: (B, D, H, W, 2k+1) * (B, 1, H, W, 2k+1) -> sum over last dim
        attention_weights = attention_weights.unsqueeze(1)  # (B, 1, H, W, 2k+1)
        ctx = (unfolded * attention_weights).sum(dim=-1)  # (B, D, H, W)

    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    # Compute std if needed
    if "std" in feature_mode:
        # Compute variance using E[X^2] - E[X]^2
        features_sq = features ** 2
        ctx_sq = F.avg_pool2d(features_sq, kernel_size=(1, 2*k+1), stride=1, padding=(0, k))
        ctx_mean = F.avg_pool2d(features, kernel_size=(1, 2*k+1), stride=1, padding=(0, k))
        std = (ctx_sq - ctx_mean ** 2).clamp(min=1e-6).sqrt()

    # Combine features based on mode
    if feature_mode == "f_ctx":
        return torch.cat([features, ctx], dim=1)  # (B, 2D, H, W)
    elif feature_mode == "f_ctx_std":
        return torch.cat([features, ctx, std], dim=1)  # (B, 3D, H, W)
    elif feature_mode == "ctx_only":
        return ctx  # (B, D, H, W)
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")


def reshape_to_patches(features: torch.Tensor) -> torch.Tensor:
    """Reshape (B, D, H, W) to (B*H*W, D) patch embeddings."""
    B, D, H, W = features.shape
    return features.permute(0, 2, 3, 1).reshape(-1, D)


def get_reference_samples(
    datamodule,
    k_ref: int,
    ref_condition: str = "balanced",
    ref_source: str = "test"
) -> Tuple[List[int], str]:
    """Get reference sample indices.

    Args:
        datamodule: Data module
        k_ref: Number of reference samples
        ref_condition: "balanced" (split cold/warm), "cold" (cold only), "warm" (warm only)
        ref_source: "test" (from test/good) or "train" (from train/good)

    Returns:
        Tuple of (indices list, source type "test" or "train")
    """
    if ref_source == "train":
        # Use train data
        datamodule.setup(stage="fit")
        dataset = datamodule.train_data
        source = "train"
    else:
        # Use test data (original behavior)
        datamodule.setup(stage="test")
        dataset = datamodule.test_data
        source = "test"

    good_samples = []
    for i, item in enumerate(dataset):
        if item.gt_label == 0:
            file_idx = int(Path(item.image_path).stem)
            good_samples.append((i, file_idx))

    good_samples.sort(key=lambda x: x[1])

    # Clamp k_ref to available samples
    max_samples = len(good_samples)
    if k_ref > max_samples:
        print(f"  Warning: k_ref={k_ref} > available samples={max_samples}, using {max_samples}")
        k_ref = max_samples

    # Train data has no cold/warm distinction - just take first k_ref samples
    if ref_source == "train":
        indices = [s[0] for s in good_samples[:k_ref]]
        print(f"  Train data: selecting first {k_ref} samples (no cold/warm distinction)")
    elif ref_condition == "cold":
        # Select k_ref coldest samples (lowest file indices)
        indices = [s[0] for s in good_samples[:k_ref]]
    elif ref_condition == "warm":
        # Select k_ref warmest samples (highest file indices)
        indices = [s[0] for s in good_samples[-k_ref:]]
    else:  # balanced
        # Split evenly between cold and warm
        k_per_condition = k_ref // 2
        # Handle odd k_ref: give extra sample to cold
        k_cold = k_per_condition + (k_ref % 2)
        k_warm = k_per_condition

        # Safety check: ensure we don't get empty lists
        if k_cold == 0:
            k_cold = 1
            k_warm = 0

        cold_indices = [s[0] for s in good_samples[:k_cold]]
        warm_indices = [s[0] for s in good_samples[-k_warm:]] if k_warm > 0 else []
        indices = cold_indices + warm_indices

    return indices, source


def build_memory_bank(
    model,
    datamodule,
    ref_indices: List[int],
    config: dict,
    device,
    ref_source: str = "test"
) -> torch.Tensor:
    """Build memory bank with CPD features.

    Args:
        model: PatchCore model
        datamodule: Data module
        ref_indices: List of sample indices
        config: Configuration dict
        device: Torch device
        ref_source: "test" or "train" - which dataset to use
    """
    if ref_source == "train":
        dataset = datamodule.train_data
    else:
        dataset = datamodule.test_data

    all_embeddings = []

    model.model.to(device)
    model.model.eval()

    print(f"Building memory bank from {len(ref_indices)} samples ({ref_source} data)...")

    # CPD 상태 및 feature mode 설명
    feature_mode = config['feature_mode']
    if feature_mode == 'f_only':
        print(f"  CPD: OFF (Vanilla PatchCore)")
        print(f"  Feature: original patch features only")
    else:
        print(f"  CPD: ON (context_k={config['context_k']}, aggregation={config['aggregation']})")
        if feature_mode == 'f_ctx':
            print(f"  Feature: [original | context] concatenated")
        elif feature_mode == 'f_ctx_std':
            print(f"  Feature: [original | context | std] concatenated")
        elif feature_mode == 'ctx_only':
            print(f"  Feature: context only (no original)")

    for idx in ref_indices:
        item = dataset[idx]
        file_idx = int(Path(item.image_path).stem)
        image = item.image.unsqueeze(0)

        # Extract spatial features
        features = extract_features(model, image, device)  # (1, D, H, W)

        # Apply horizontal context
        features_cpd = apply_horizontal_context(
            features,
            k=config["context_k"],
            aggregation=config["aggregation"],
            feature_mode=config["feature_mode"],
            trim_ratio=config["trim_ratio"],
            attention_alpha=config["attention_alpha"],
            sampling_mode=config.get("sampling_mode", "adjacent"),
            distance_decay=config.get("distance_decay", 1.0),
            seed=config["seed"],
            multik_mode=config.get("multik_mode"),
            multik_list=config.get("multik_list"),
            multik_weights=config.get("multik_weights"),
            ablation_mode=config.get("ablation_mode"),
            ablation_n_concat=config.get("ablation_n_concat", 1)
        )

        # Reshape to patches
        patches = reshape_to_patches(features_cpd)  # (H*W, D')
        all_embeddings.append(patches)

    memory_bank = torch.cat(all_embeddings, dim=0)
    print(f"Memory bank shape: {memory_bank.shape}")

    return memory_bank


def euclidean_dist(x: torch.Tensor, y: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    """Compute pairwise Euclidean distances with chunking to save memory.

    Args:
        x: (N, D) query embeddings
        y: (M, D) memory bank embeddings
        chunk_size: process x in chunks to reduce memory usage

    Returns:
        (N, M) distance matrix
    """
    N = x.shape[0]
    M = y.shape[0]

    # Precompute y norms (only once)
    y_norm = y.pow(2).sum(dim=-1)  # (M,)

    # If small enough, compute directly
    if N * M * 4 < 1e9:  # Less than ~1GB
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)
        res = x_norm - 2 * torch.matmul(x, y.transpose(-2, -1)) + y_norm.unsqueeze(0)
        return res.clamp_min_(0).sqrt_()

    # Chunked computation for large matrices
    distances = torch.empty(N, M, device=x.device, dtype=x.dtype)

    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        x_chunk = x[i:end_i]  # (chunk, D)
        x_norm_chunk = x_chunk.pow(2).sum(dim=-1, keepdim=True)  # (chunk, 1)

        # Compute distances for this chunk
        dist_chunk = x_norm_chunk - 2 * torch.matmul(x_chunk, y.transpose(-2, -1)) + y_norm.unsqueeze(0)
        distances[i:end_i] = dist_chunk.clamp_min_(0).sqrt_()

        # Free intermediate tensors
        del x_chunk, x_norm_chunk, dist_chunk

    return distances


def compute_anomaly_score(
    query_embedding: torch.Tensor,
    memory_bank: torch.Tensor,
    num_neighbors: int = 9,
    return_map: bool = False,
    spatial_shape: Tuple[int, int] = None
) -> torch.Tensor:
    """Compute anomaly score using nearest neighbor distance.

    Args:
        query_embedding: (H*W, D') patch embeddings
        memory_bank: (N, D') memory bank embeddings
        num_neighbors: number of neighbors for re-weighting
        return_map: if True, also return spatial anomaly map
        spatial_shape: (H, W) for reshaping to spatial map

    Returns:
        score: scalar anomaly score
        anomaly_map: (H, W) spatial map (only if return_map=True)
    """
    distances = euclidean_dist(query_embedding, memory_bank)
    patch_scores, locations = distances.min(dim=1)

    if num_neighbors == 1:
        score = patch_scores.max()
        if return_map and spatial_shape:
            anomaly_map = patch_scores.view(*spatial_shape)
            return score, anomaly_map
        return score

    max_patch_idx = patch_scores.argmax()
    max_patch_feature = query_embedding[max_patch_idx].unsqueeze(0)
    score = patch_scores[max_patch_idx]

    nn_index = locations[max_patch_idx]
    nn_sample = memory_bank[nn_index].unsqueeze(0)

    k = min(num_neighbors, memory_bank.shape[0])
    nn_distances = euclidean_dist(nn_sample, memory_bank)
    _, support_indices = nn_distances.topk(k, dim=1, largest=False)

    support_features = memory_bank[support_indices.squeeze(0)]
    support_distances = euclidean_dist(max_patch_feature, support_features)

    weights = (1 - F.softmax(support_distances.squeeze(0), dim=0))[0]

    final_score = weights * score

    if return_map and spatial_shape:
        anomaly_map = patch_scores.view(*spatial_shape)
        return final_score, anomaly_map

    return final_score


def run_inference(
    model,
    datamodule,
    memory_bank: torch.Tensor,
    config: dict,
    device,
    output_dir: Path = None,
    save_vis: bool = False,
    vis_max_samples: int = 100
) -> List[Dict]:
    """Run inference on test set.

    Args:
        model: PatchCore model
        datamodule: HDMAP datamodule
        memory_bank: memory bank tensor
        config: experiment configuration
        device: torch device
        output_dir: output directory for visualizations
        save_vis: whether to save visualizations
        vis_max_samples: max samples to visualize per class
    """
    datamodule.setup(stage="test")
    test_dataset = datamodule.test_data
    n_neighbors = config["num_neighbors"]

    results = []
    vis_count = 0

    model.model.to(device)
    model.model.eval()

    # Get spatial shape from first sample
    first_item = test_dataset[0]
    first_features = extract_features(model, first_item.image.unsqueeze(0), device)
    spatial_shape = (first_features.shape[2], first_features.shape[3])  # (H, W)

    # Determine which samples to visualize (random subset or all)
    n_samples = len(test_dataset)
    if save_vis:
        if vis_max_samples <= 0 or vis_max_samples >= n_samples:
            vis_indices = set(range(n_samples))  # Visualize all
        else:
            np.random.seed(config["seed"])
            vis_indices = set(np.random.choice(n_samples, vis_max_samples, replace=False))
        print(f"Will visualize {len(vis_indices)} samples")

    for i in tqdm(range(n_samples), desc="Inference"):
        item = test_dataset[i]
        image = item.image.unsqueeze(0)
        file_idx = int(Path(item.image_path).stem)
        gt_label = item.gt_label.item()
        gt_condition = "cold" if file_idx < 500 else "warm"

        # Extract and transform features
        features = extract_features(model, image, device)
        features_cpd = apply_horizontal_context(
            features,
            k=config["context_k"],
            aggregation=config["aggregation"],
            feature_mode=config["feature_mode"],
            trim_ratio=config["trim_ratio"],
            attention_alpha=config["attention_alpha"],
            sampling_mode=config.get("sampling_mode", "adjacent"),
            distance_decay=config.get("distance_decay", 1.0),
            seed=config["seed"],
            multik_mode=config.get("multik_mode"),
            multik_list=config.get("multik_list"),
            multik_weights=config.get("multik_weights"),
            ablation_mode=config.get("ablation_mode"),
            ablation_n_concat=config.get("ablation_n_concat", 1)
        )
        query_emb = reshape_to_patches(features_cpd)

        # Compute anomaly score with map
        if save_vis and output_dir and i in vis_indices:
            score, anomaly_map = compute_anomaly_score(
                query_emb, memory_bank, n_neighbors,
                return_map=True, spatial_shape=spatial_shape
            )
            save_visualization(
                item.image, anomaly_map, gt_label, file_idx, output_dir
            )
            vis_count += 1
        else:
            score = compute_anomaly_score(query_emb, memory_bank, n_neighbors)

        results.append({
            "file_idx": file_idx,
            "gt_label": gt_label,
            "gt_condition": gt_condition,
            "score": score.item() if hasattr(score, 'item') else float(score),
        })

    if save_vis:
        print(f"Saved {vis_count} visualizations")

    return results


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


def save_visualization(
    image: torch.Tensor,
    anomaly_map: torch.Tensor,
    gt_label: int,
    file_idx: int,
    output_dir: Path,
    threshold: float = None
):
    """Save 3-column visualization: original | anomaly map | overlay.

    Args:
        image: (3, H, W) or (1, 3, H, W) input image tensor
        anomaly_map: (h, w) patch-level anomaly scores
        gt_label: 0=good, 1=fault
        file_idx: file index for naming
        output_dir: output directory
        threshold: optional threshold for binarization
    """
    import cv2
    from PIL import Image

    # Prepare image
    if image.dim() == 4:
        image = image.squeeze(0)
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    H, W = img_np.shape[:2]

    # Prepare anomaly map - upsample to image size
    amap = anomaly_map.cpu().numpy()
    amap_resized = cv2.resize(amap, (W, H), interpolation=cv2.INTER_CUBIC)

    # Normalize to [0, 1]
    amap_norm = (amap_resized - amap_resized.min()) / (amap_resized.max() - amap_resized.min() + 1e-8)

    # Create heatmap (jet colormap)
    heatmap = plt.cm.jet(amap_norm)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    # Create overlay
    alpha = 0.5
    overlay = (alpha * img_np + (1 - alpha) * heatmap).astype(np.uint8)

    # Create 3-column figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(amap_resized, cmap="jet")
    axes[1].set_title("Anomaly Map")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    label_str = "Fault" if gt_label == 1 else "Good"
    axes[2].set_title(f"Overlay ({label_str})")
    axes[2].axis("off")

    plt.tight_layout()

    # Save
    label_dir = "fault" if gt_label == 1 else "good"
    save_dir = output_dir / "visualizations" / label_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"{file_idx:06d}.png", dpi=100, bbox_inches="tight")
    plt.close()


def plot_score_distribution(
    all_scores: np.ndarray,
    all_labels: np.ndarray,
    cold_mask: np.ndarray,
    warm_mask: np.ndarray,
    threshold: float,
    output_dir: Path,
    config: dict
):
    """Plot score distribution histograms."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    title_prefix = f"CPD-{config['aggregation']}-k{config['context_k']}"

    # All samples
    ax = axes[0]
    ax.hist(all_scores[all_labels == 0], bins=50, alpha=0.7, label="Good", color="green", density=True)
    ax.hist(all_scores[all_labels == 1], bins=50, alpha=0.7, label="Fault", color="red", density=True)
    ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold:.3f}")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title(f"{title_prefix} - All Samples")
    ax.legend()

    # Cold samples
    ax = axes[1]
    cold_scores = all_scores[cold_mask]
    cold_labels = all_labels[cold_mask]
    ax.hist(cold_scores[cold_labels == 0], bins=50, alpha=0.7, label="Good", color="green", density=True)
    ax.hist(cold_scores[cold_labels == 1], bins=50, alpha=0.7, label="Fault", color="red", density=True)
    ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold:.3f}")
    ax.set_xlabel("Anomaly Score")
    ax.set_title(f"{title_prefix} - Cold Samples")
    ax.legend()

    # Warm samples
    ax = axes[2]
    warm_scores = all_scores[warm_mask]
    warm_labels = all_labels[warm_mask]
    ax.hist(warm_scores[warm_labels == 0], bins=50, alpha=0.7, label="Good", color="green", density=True)
    ax.hist(warm_scores[warm_labels == 1], bins=50, alpha=0.7, label="Fault", color="red", density=True)
    ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold:.3f}")
    ax.set_xlabel("Anomaly Score")
    ax.set_title(f"{title_prefix} - Warm Samples")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution.png", dpi=150)
    plt.close()


def print_results(analysis: Dict, config: dict):
    """Print formatted results."""
    print(f"\n{'='*70}")
    print(f"Results: {config['domain']} | CPD-{config['aggregation']}-k{config['context_k']}")
    print(f"Feature mode: {config['feature_mode']} | k_ref: {config['k_ref']}")
    print(f"{'='*70}")
    print(f"{'Metric':<15} {'Overall':>12} {'Cold':>12} {'Warm':>12}")
    print("-" * 55)
    print(f"{'Accuracy':<15} {analysis['overall']['accuracy']:>12.2%} "
          f"{analysis['cold']['accuracy']:>12.2%} {analysis['warm']['accuracy']:>12.2%}")
    print(f"{'AUROC':<15} {analysis['overall']['auroc']:>12.2%} "
          f"{analysis['cold']['auroc']:>12.2%} {analysis['warm']['auroc']:>12.2%}")


def main():
    parser = argparse.ArgumentParser(description="CPD-PatchCore on HDMAP")
    parser.add_argument("--domain", type=str, required=True,
                        choices=["domain_A", "domain_B", "domain_C", "domain_D"])
    parser.add_argument("--k-ref", type=int, default=32,
                        help="Total number of reference samples (split between cold/warm)")
    parser.add_argument("--ref-condition", type=str, default="balanced",
                        choices=["balanced", "cold", "warm"],
                        help="Reference sample condition: balanced (split), cold only, warm only")
    parser.add_argument("--ref-source", type=str, default="test",
                        choices=["test", "train"],
                        help="Reference sample source: test (from test/good) or train (from train/good)")
    parser.add_argument("--context-k", type=int, default=2,
                        help="Horizontal context radius (2k+1 patches)")
    parser.add_argument("--aggregation", type=str, default="mean",
                        choices=["mean", "max", "gaussian", "median", "trimmed", "attention"],
                        help="Context aggregation method")
    parser.add_argument("--feature-mode", type=str, default="f_ctx",
                        choices=["f_only", "f_ctx", "f_ctx_std", "ctx_only"],
                        help="Feature combination mode")
    parser.add_argument("--trim-ratio", type=float, default=0.2,
                        help="Trim ratio for trimmed mean (default: 0.2 = 20%% each side)")
    parser.add_argument("--attention-alpha", type=float, default=5.0,
                        help="Temperature for attention weights (higher = sharper)")
    parser.add_argument("--sampling-mode", type=str, default="adjacent",
                        choices=["adjacent", "random", "distance_weighted"],
                        help="Sampling mode: adjacent (2k+1 neighbors), random (2k random), "
                             "distance_weighted (2k with p~1/d^decay)")
    parser.add_argument("--distance-decay", type=float, default=1.0,
                        help="Distance decay for distance_weighted sampling (higher = more local)")
    parser.add_argument("--multik-mode", type=str, default=None,
                        choices=["weighted", "concat"],
                        help="Multi-k mode: weighted (average) or concat")
    parser.add_argument("--multik-list", type=int, nargs="+", default=None,
                        help="List of k values for multi-k (e.g., --multik-list 2 3)")
    parser.add_argument("--multik-weights", type=float, nargs="+", default=None,
                        help="Weights for weighted multi-k (e.g., --multik-weights 0.5 0.5)")
    parser.add_argument("--ablation-mode", type=str, default=None,
                        choices=["random", "global", "zeros"],
                        help="Ablation mode: concat random/global/zeros instead of context")
    parser.add_argument("--ablation-n-concat", type=int, default=1,
                        help="Number of D-dim vectors to concat (1=2D, 2=3D)")
    parser.add_argument("--no-save-vis", action="store_false", dest="save_vis",
                        help="Disable visualization saving (default: save enabled)")
    parser.set_defaults(save_vis=True)
    parser.add_argument("--vis-max-samples", type=int, default=50,
                        help="Max samples to visualize (default: 50, 0 or negative = all)")
    parser.add_argument("--backbone", type=str, default="vit_base_patch14_dinov2")
    parser.add_argument("--layers", type=str, nargs="+", default=["blocks.8"],
                        help="Layer names to extract features from "
                             "(e.g., --layers blocks.8 blocks.11)")
    parser.add_argument("--resize-method", type=str, default="resize_bilinear",
                        choices=["resize", "resize_bilinear"],
                        help="Resize method: resize (nearest) or resize_bilinear")
    parser.add_argument("--coreset-ratio", type=float, default=1.0,
                        help="Coreset sampling ratio (0.01 = 1%%, 1.0 = 100%%)")
    args = parser.parse_args()

    # Config and output
    config = get_config(
        args.domain, args.k_ref, args.context_k,
        args.aggregation, args.feature_mode,
        args.trim_ratio, args.attention_alpha,
        args.sampling_mode, args.distance_decay,
        args.multik_mode, args.multik_list, args.multik_weights,
        args.ablation_mode, args.ablation_n_concat,
        args.backbone, args.layers, args.resize_method, args.coreset_ratio,
        args.ref_source
    )
    exp_name = get_exp_name(config)
    output_dir = create_output_dir(exp_name)

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Print all experiment parameters
    print(f"\n[Model Configuration]")
    print(f"  Backbone: {config['backbone']}")
    print(f"  Layers: {config['layers']}")
    print(f"  Target size: {config['target_size']}")
    print(f"  Resize method: {config['resize_method']}")
    print(f"  Coreset ratio: {config['coreset_sampling_ratio']}")
    print(f"  Num neighbors (k-NN): {config['num_neighbors']}")

    print(f"\n[Data Configuration]")
    print(f"  Domain: {config['domain']}")
    print(f"  Reference samples (k_ref): {config['k_ref']}")
    print(f"  Reference source: {config['ref_source']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Seed: {config['seed']}")

    print(f"\n[CPD Configuration]")
    if config['feature_mode'] == 'f_only':
        print(f"  CPD: OFF (Vanilla PatchCore)")
    else:
        print(f"  CPD: ON")
        print(f"  Context k: {config['context_k']} (window = {2*config['context_k']+1} patches)")
        print(f"  Aggregation: {config['aggregation']}")
        print(f"  Sampling mode: {config['sampling_mode']}")
        if config['sampling_mode'] == 'distance_weighted':
            print(f"  Distance decay: {config['distance_decay']}")
        if config['aggregation'] == 'trimmed':
            print(f"  Trim ratio: {config['trim_ratio']}")
        if config['aggregation'] == 'attention':
            print(f"  Attention alpha: {config['attention_alpha']}")
    print(f"  Feature mode: {config['feature_mode']}")

    if config['multik_mode']:
        print(f"\n[Multi-K Configuration]")
        print(f"  Mode: {config['multik_mode']}")
        print(f"  K list: {config['multik_list']}")
        print(f"  Weights: {config['multik_weights']}")

    if config['ablation_mode']:
        print(f"\n[Ablation Configuration]")
        print(f"  Mode: {config['ablation_mode']}")
        print(f"  N concat: {config['ablation_n_concat']}")

    print(f"{'='*60}\n")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, datamodule = setup_model_and_data(args.domain, config)

    # Get reference samples
    ref_indices, ref_source_used = get_reference_samples(
        datamodule, args.k_ref, args.ref_condition, args.ref_source
    )
    condition_str = args.ref_condition if args.ref_condition != "balanced" else "cold+warm"
    print(f"Reference samples: {len(ref_indices)} ({condition_str} from {ref_source_used})")

    # Build memory bank
    memory_bank = build_memory_bank(
        model, datamodule, ref_indices, config, device, ref_source=ref_source_used
    )

    # Run inference
    results = run_inference(
        model, datamodule, memory_bank, config, device,
        output_dir=output_dir,
        save_vis=args.save_vis,
        vis_max_samples=args.vis_max_samples
    )

    # Analyze
    analysis, all_scores, all_labels, cold_mask, warm_mask, threshold = analyze_results(
        results, args.domain
    )

    # Add metadata
    analysis["domain"] = args.domain
    analysis["context_k"] = args.context_k
    analysis["aggregation"] = args.aggregation
    analysis["feature_mode"] = args.feature_mode
    analysis["k_ref"] = args.k_ref
    analysis["sampling_mode"] = args.sampling_mode
    analysis["distance_decay"] = args.distance_decay
    analysis["multik_mode"] = args.multik_mode
    analysis["multik_list"] = args.multik_list
    analysis["multik_weights"] = args.multik_weights
    analysis["ablation_mode"] = args.ablation_mode
    analysis["ablation_n_concat"] = args.ablation_n_concat
    analysis["ref_source"] = ref_source_used

    # Print and save
    print_results(analysis, config)

    with open(output_dir / "results.json", "w") as f:
        json.dump(analysis, f, indent=2)

    with open(output_dir / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    plot_score_distribution(
        all_scores, all_labels, cold_mask, warm_mask, threshold, output_dir, config
    )

    print(f"\nSaved to: {output_dir}")

    return analysis


if __name__ == "__main__":
    main()
