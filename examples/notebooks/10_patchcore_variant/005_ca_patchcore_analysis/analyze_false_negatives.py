#!/usr/bin/env python3
"""Analyze False Negative samples from CA-PatchCore oracle mode.

Focus on cold fault samples that were misclassified as good.
Hypothesis: These samples' fault patches are closer to warm bank than cold bank.

Usage:
    CUDA_VISIBLE_DEVICES=0 python analyze_false_negatives.py \
        --domain domain_C \
        --k-per-bank 16 \
        --n-samples 10
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results" / "false_negatives"
ANOMALIB_ROOT = SCRIPT_DIR.parent.parent.parent.parent
DATASET_ROOT = ANOMALIB_ROOT / "datasets" / "HDMAP" / "1000_tiff_minmax"
CA_RESULTS_DIR = SCRIPT_DIR.parent / "004_ca_patchcore" / "results"

P90_THRESHOLDS = {
    'domain_A': 0.2985,
    'domain_B': 0.3128,
    'domain_C': 0.3089,
    'domain_D': 0.2919,
}


def get_config(domain: str, k_per_bank: int) -> dict:
    return {
        "backbone": "vit_base_patch14_dinov2",
        "layers": ["blocks.8"],
        "target_size": (518, 518),
        "resize_method": "resize_bilinear",
        "num_neighbors": 9,
        "batch_size": 4,
        "seed": 42,
        "domain": domain,
        "k_per_bank": k_per_bank,
    }


def setup_model_and_data(domain: str, config: dict):
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
        coreset_sampling_ratio=1.0,
        num_neighbors=config["num_neighbors"],
        pre_processor=pre_processor,
    )

    return model, datamodule


def extract_features(model, images: torch.Tensor, device) -> torch.Tensor:
    model.model.eval()
    pre_processor = model.pre_processor

    with torch.no_grad():
        images = images.to(device)
        normalized = pre_processor(images)
        feature_extractor = model.model.feature_extractor
        feature_pooler = model.model.feature_pooler
        features = feature_extractor(normalized)
        features = {layer: feature_pooler(feature) for layer, feature in features.items()}
        embedding = model.model.generate_embedding(features)
        embedding = model.model.reshape_embedding(embedding)

    return embedding


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = x.pow(2).sum(dim=-1, keepdim=True)
    y_norm = y.pow(2).sum(dim=-1, keepdim=True)
    res = x_norm - 2 * torch.matmul(x, y.transpose(-2, -1)) + y_norm.transpose(-2, -1)
    return res.clamp_min_(0).sqrt_()


def get_reference_samples(datamodule, k_per_bank: int) -> Tuple[List, List]:
    datamodule.setup(stage="test")
    test_dataset = datamodule.test_data

    good_samples = []
    for i, item in enumerate(test_dataset):
        if item.gt_label == 0:
            file_idx = int(Path(item.image_path).stem)
            good_samples.append((i, file_idx))

    good_samples.sort(key=lambda x: x[1])

    cold_indices = [s[0] for s in good_samples[:k_per_bank]]
    warm_indices = [s[0] for s in good_samples[-k_per_bank:]]

    return cold_indices, warm_indices


def build_memory_banks(model, datamodule, cold_indices, warm_indices, device):
    test_dataset = datamodule.test_data
    cold_embeddings = []
    warm_embeddings = []
    cold_file_indices = []
    warm_file_indices = []

    model.model.to(device)
    model.model.eval()

    for idx in cold_indices:
        item = test_dataset[idx]
        file_idx = int(Path(item.image_path).stem)
        cold_file_indices.append(file_idx)
        image = item.image.unsqueeze(0)
        emb = extract_features(model, image, device)
        cold_embeddings.append(emb)

    cold_bank = torch.cat(cold_embeddings, dim=0)

    for idx in warm_indices:
        item = test_dataset[idx]
        file_idx = int(Path(item.image_path).stem)
        warm_file_indices.append(file_idx)
        image = item.image.unsqueeze(0)
        emb = extract_features(model, image, device)
        warm_embeddings.append(emb)

    warm_bank = torch.cat(warm_embeddings, dim=0)

    return cold_bank, warm_bank, cold_file_indices, warm_file_indices


def load_false_negatives(domain: str, k_per_bank: int, condition: str = "cold") -> List[Dict]:
    """Load false negative samples from CA-PatchCore oracle results."""
    # Find the most recent oracle results
    oracle_dir = CA_RESULTS_DIR / f"ca_patchcore_oracle_k{k_per_bank}_{domain}"
    if not oracle_dir.exists():
        raise FileNotFoundError(f"Oracle results not found: {oracle_dir}")

    # Get most recent timestamp
    timestamps = sorted([d.name for d in oracle_dir.iterdir() if d.is_dir()])
    if not timestamps:
        raise FileNotFoundError(f"No results in {oracle_dir}")

    results_dir = oracle_dir / timestamps[-1]
    results_file = results_dir / "results.json"
    detailed_file = results_dir / "detailed_results.json"

    with open(results_file) as f:
        summary = json.load(f)
    with open(detailed_file) as f:
        detailed = json.load(f)

    threshold = summary["overall"]["threshold"]

    # Find false negatives
    fn_samples = [
        r for r in detailed
        if r["gt_condition"] == condition
        and r["is_fault"]
        and r["score"] < threshold
    ]

    fn_samples.sort(key=lambda x: x["score"])

    return fn_samples, threshold


def find_sample_by_file_idx(datamodule, file_idx: int):
    """Find sample in dataset by file index."""
    test_dataset = datamodule.test_data
    for i, item in enumerate(test_dataset):
        if int(Path(item.image_path).stem) == file_idx:
            return i, item
    return None, None


def analyze_single_fn(
    model,
    sample_item,
    cold_bank: torch.Tensor,
    warm_bank: torch.Tensor,
    device,
    grid_size: int = 37
) -> Dict:
    """Analyze a false negative sample."""
    image = sample_item.image.unsqueeze(0)
    file_idx = int(Path(sample_item.image_path).stem)

    query_emb = extract_features(model, image, device).squeeze(0)

    dist_cold = euclidean_dist(query_emb, cold_bank)
    dist_warm = euclidean_dist(query_emb, warm_bank)

    nn_cold_dist, nn_cold_idx = dist_cold.min(dim=1)
    nn_warm_dist, nn_warm_idx = dist_warm.min(dim=1)

    closer_to_warm = nn_warm_dist < nn_cold_dist

    # Find score if using cold bank vs warm bank vs combined
    score_cold = nn_cold_dist.max().item()
    score_warm = nn_warm_dist.max().item()

    combined_bank = torch.cat([cold_bank, warm_bank], dim=0)
    dist_combined = euclidean_dist(query_emb, combined_bank)
    nn_combined_dist, _ = dist_combined.min(dim=1)
    score_combined = nn_combined_dist.max().item()

    # Get most anomalous patch info
    max_cold_patch_idx = nn_cold_dist.argmax().item()
    max_cold_i, max_cold_j = max_cold_patch_idx // grid_size, max_cold_patch_idx % grid_size

    results = {
        "file_idx": file_idx,
        "grid_size": grid_size,
        "score_cold_bank": score_cold,
        "score_warm_bank": score_warm,
        "score_combined_bank": score_combined,
        "score_improvement": score_cold - score_combined,
        "pct_closer_to_warm": float(closer_to_warm.float().mean() * 100),
        "mean_dist_cold": float(nn_cold_dist.mean()),
        "mean_dist_warm": float(nn_warm_dist.mean()),
        "max_anomalous_patch": {
            "idx": max_cold_patch_idx,
            "coord": [max_cold_i, max_cold_j],
            "dist_cold": float(nn_cold_dist[max_cold_patch_idx]),
            "dist_warm": float(nn_warm_dist[max_cold_patch_idx]),
            "closer_to": "warm" if nn_warm_dist[max_cold_patch_idx] < nn_cold_dist[max_cold_patch_idx] else "cold",
        },
    }

    # Analyze top-10 anomalous patches
    top_10_idx = nn_cold_dist.topk(10).indices.tolist()
    top_10_closer_to_warm = sum(
        1 for idx in top_10_idx
        if nn_warm_dist[idx] < nn_cold_dist[idx]
    )
    results["top_10_closer_to_warm"] = top_10_closer_to_warm

    return results, nn_cold_dist, nn_warm_dist, closer_to_warm


def visualize_fn_analysis(
    sample_item,
    fn_info: Dict,
    analysis: Dict,
    nn_cold_dist: torch.Tensor,
    nn_warm_dist: torch.Tensor,
    closer_to_warm: torch.Tensor,
    threshold: float,
    output_path: Path,
    grid_size: int = 37
):
    """Visualize false negative analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    img = sample_item.image.cpu().numpy()
    if img.shape[0] == 3:
        img = img[0]

    file_idx = analysis["file_idx"]

    # 1. Original image with info
    ax = axes[0, 0]
    ax.imshow(img, cmap='gray')
    ax.set_title(f"File {file_idx} (Cold Fault - FALSE NEGATIVE)\n"
                 f"Oracle Score: {fn_info['score']:.3f} < Threshold: {threshold:.3f}")
    ax.axis('off')

    # 2. Distance to cold bank
    ax = axes[0, 1]
    dist_cold_map = nn_cold_dist.reshape(grid_size, grid_size).cpu().numpy()
    im = ax.imshow(dist_cold_map, cmap='hot')
    ax.set_title(f"Distance to COLD Bank (selected)\n"
                 f"Max={analysis['score_cold_bank']:.3f}")
    plt.colorbar(im, ax=ax)

    # 3. Distance to warm bank
    ax = axes[0, 2]
    dist_warm_map = nn_warm_dist.reshape(grid_size, grid_size).cpu().numpy()
    im = ax.imshow(dist_warm_map, cmap='hot')
    ax.set_title(f"Distance to WARM Bank (not used)\n"
                 f"Max={analysis['score_warm_bank']:.3f}")
    plt.colorbar(im, ax=ax)

    # 4. Which bank is closer
    ax = axes[1, 0]
    closer_map = closer_to_warm.reshape(grid_size, grid_size).cpu().numpy().astype(float)
    pct_warm = analysis["pct_closer_to_warm"]
    im = ax.imshow(closer_map, cmap='RdBu', vmin=0, vmax=1)
    ax.set_title(f"Closer to Warm Bank (Red)\n{pct_warm:.1f}% patches closer to Warm")
    plt.colorbar(im, ax=ax)

    # 5. Score comparison bar chart
    ax = axes[1, 1]
    scores = [analysis['score_cold_bank'], analysis['score_warm_bank'], analysis['score_combined_bank']]
    labels = ['Cold Bank\n(Oracle)', 'Warm Bank', 'Combined\n(Mixed)']
    colors = ['blue', 'red', 'purple']

    bars = ax.bar(labels, scores, color=colors, alpha=0.7)
    ax.axhline(y=threshold, color='black', linestyle='--', label=f'Threshold={threshold:.3f}')

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Anomaly Score')
    ax.set_title(f"Score by Bank Selection\n"
                 f"Improvement if Combined: +{analysis['score_improvement']:.3f}")
    ax.legend()

    # Add pass/fail annotations
    for i, (bar, score) in enumerate(zip(bars, scores)):
        status = "PASS" if score >= threshold else "FAIL"
        color = 'green' if status == "PASS" else 'red'
        ax.text(bar.get_x() + bar.get_width()/2, threshold - 0.3,
                status, ha='center', va='top', fontsize=9, fontweight='bold', color=color)

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis('off')

    summary_text = f"""
    FALSE NEGATIVE ANALYSIS
    =======================
    File: {file_idx} (Cold Fault)

    Oracle Gating Result:
      Score: {fn_info['score']:.3f}
      Threshold: {threshold:.3f}
      Result: MISS (False Negative)

    Bank Score Comparison:
      Cold Bank (used):   {analysis['score_cold_bank']:.3f}
      Warm Bank:          {analysis['score_warm_bank']:.3f}
      Combined (Mixed):   {analysis['score_combined_bank']:.3f}

    Would PASS with:
      - Warm Bank: {'YES' if analysis['score_warm_bank'] >= threshold else 'NO'}
      - Combined:  {'YES' if analysis['score_combined_bank'] >= threshold else 'NO'}

    Patch Analysis:
      % closer to Warm: {analysis['pct_closer_to_warm']:.1f}%
      Top-10 anomalous closer to Warm: {analysis['top_10_closer_to_warm']}/10

    Most Anomalous Patch:
      Location: ({analysis['max_anomalous_patch']['coord'][0]}, {analysis['max_anomalous_patch']['coord'][1]})
      Dist to Cold: {analysis['max_anomalous_patch']['dist_cold']:.3f}
      Dist to Warm: {analysis['max_anomalous_patch']['dist_warm']:.3f}
      Closer to: {analysis['max_anomalous_patch']['closer_to'].upper()}
    """

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze False Negatives from CA-PatchCore")
    parser.add_argument("--domain", type=str, required=True,
                        choices=["domain_A", "domain_B", "domain_C", "domain_D"])
    parser.add_argument("--k-per-bank", type=int, default=16)
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of worst false negatives to analyze")
    args = parser.parse_args()

    config = get_config(args.domain, args.k_per_bank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"{args.domain}_k{args.k_per_bank}" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"False Negative Analysis for CA-PatchCore Oracle Mode")
    print(f"Domain: {args.domain}, k_per_bank: {args.k_per_bank}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Load false negatives
    print("Loading false negatives from oracle results...")
    fn_samples, threshold = load_false_negatives(args.domain, args.k_per_bank, "cold")
    print(f"Found {len(fn_samples)} cold false negatives (threshold={threshold:.3f})")
    print(f"Analyzing top {args.n_samples} worst cases...")

    # Setup
    model, datamodule = setup_model_and_data(args.domain, config)
    cold_indices, warm_indices = get_reference_samples(datamodule, args.k_per_bank)
    cold_bank, warm_bank, _, _ = build_memory_banks(
        model, datamodule, cold_indices, warm_indices, device
    )

    # Analyze each FN
    all_analyses = []
    datamodule.setup(stage="test")

    for fn_info in tqdm(fn_samples[:args.n_samples], desc="Analyzing FN samples"):
        file_idx = fn_info["file_idx"]
        dataset_idx, sample_item = find_sample_by_file_idx(datamodule, file_idx)

        if sample_item is None:
            print(f"Warning: Could not find file {file_idx}")
            continue

        analysis, nn_cold_dist, nn_warm_dist, closer_to_warm = analyze_single_fn(
            model, sample_item, cold_bank, warm_bank, device
        )

        # Add original score
        analysis["oracle_score"] = fn_info["score"]

        # Visualize
        sample_dir = output_dir / f"fn_{file_idx:04d}"
        sample_dir.mkdir(exist_ok=True)

        visualize_fn_analysis(
            sample_item, fn_info, analysis,
            nn_cold_dist, nn_warm_dist, closer_to_warm,
            threshold, sample_dir / "analysis.png"
        )

        with open(sample_dir / "results.json", "w") as f:
            json.dump(analysis, f, indent=2)

        all_analyses.append(analysis)

    # Summary
    summary = {
        "domain": args.domain,
        "k_per_bank": args.k_per_bank,
        "threshold": threshold,
        "n_analyzed": len(all_analyses),
        "total_cold_fn": len(fn_samples),
        "analyses": all_analyses,
        "aggregate": {
            "mean_pct_closer_to_warm": np.mean([a["pct_closer_to_warm"] for a in all_analyses]),
            "mean_top10_to_warm": np.mean([a["top_10_closer_to_warm"] for a in all_analyses]),
            "mean_score_improvement": np.mean([a["score_improvement"] for a in all_analyses]),
            "would_pass_with_combined": sum(1 for a in all_analyses if a["score_combined_bank"] >= threshold),
            "would_pass_with_warm": sum(1 for a in all_analyses if a["score_warm_bank"] >= threshold),
        }
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("FALSE NEGATIVE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Domain: {args.domain}")
    print(f"Total Cold False Negatives: {len(fn_samples)}")
    print(f"Analyzed: {len(all_analyses)}")
    print(f"\nAggregate Results:")
    print(f"  Mean % patches closer to WARM: {summary['aggregate']['mean_pct_closer_to_warm']:.1f}%")
    print(f"  Mean Top-10 anomalous closer to WARM: {summary['aggregate']['mean_top10_to_warm']:.1f}/10")
    print(f"  Mean score improvement with Combined: +{summary['aggregate']['mean_score_improvement']:.3f}")
    print(f"\nRecovery potential:")
    print(f"  Would PASS with Combined bank: {summary['aggregate']['would_pass_with_combined']}/{len(all_analyses)}")
    print(f"  Would PASS with Warm bank: {summary['aggregate']['would_pass_with_warm']}/{len(all_analyses)}")

    print(f"\nPer-sample breakdown:")
    for a in all_analyses:
        status_combined = "PASS" if a["score_combined_bank"] >= threshold else "FAIL"
        status_warm = "PASS" if a["score_warm_bank"] >= threshold else "FAIL"
        print(f"  File {a['file_idx']:04d}: oracle={a['oracle_score']:.3f}, "
              f"cold={a['score_cold_bank']:.3f}, warm={a['score_warm_bank']:.3f}, "
              f"combined={a['score_combined_bank']:.3f} [{status_combined}], "
              f"top10→warm: {a['top_10_closer_to_warm']}/10")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
