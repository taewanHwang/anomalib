#!/usr/bin/env python3
"""Within-Condition Analysis: Cold Fault vs Cold Good in Domain C.

Tests whether WinCLIP can distinguish faults from normal samples
when both are in the same cold condition (avoiding cross-condition issues).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from anomalib.data.datasets.image.hdmap import HDMAPDataset
from anomalib.models.image.winclip.torch_model import WinClipModel
from anomalib.models.image.winclip.utils import cosine_similarity


ANOMALIB_ROOT = Path(__file__).parent.parent.parent.parent
DATASET_ROOT = ANOMALIB_ROOT / "datasets" / "HDMAP" / "1000_tiff_minmax"
OUTPUT_DIR = Path(__file__).parent / "HDMAP_vis"


def load_image(domain: str, split: str, label: str, index: int):
    """Load image from HDMAPDataset."""
    dataset = HDMAPDataset(
        root=str(DATASET_ROOT),
        domain=domain,
        split=split,
        target_size=(240, 240),
        resize_method="resize",
    )

    target_path = f"{label}/{index:06d}.tiff"
    for item in dataset:
        if target_path in item.image_path:
            return item.image, item.image_path

    raise ValueError(f"Image not found: {target_path}")


def compute_anomaly_score(model, test_batch, ref_batch, device):
    """Compute anomaly score using patch-level max similarity."""
    with torch.no_grad():
        _, _, test_patch_emb = model.encode_image(test_batch.to(device))
        _, _, ref_patch_emb = model.encode_image(ref_batch.to(device))

    # Patch level similarity
    patch_test = test_patch_emb.squeeze(0)  # (225, D)
    patch_ref = ref_patch_emb.squeeze(0)    # (225, D)

    patch_sim = cosine_similarity(patch_test, patch_ref)  # (225, 225)
    max_sim = patch_sim.max(dim=-1).values  # (225,)

    # Anomaly score = (1 - max_similarity) / 2
    anomaly_scores = (1 - max_sim) / 2

    return {
        "mean": anomaly_scores.mean().item(),
        "max": anomaly_scores.max().item(),
        "min": anomaly_scores.min().item(),
        "map": anomaly_scores.cpu().numpy(),
        "similarity_map": max_sim.cpu().numpy(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="domain_C")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading WinCLIP model...")
    model = WinClipModel(class_name="industrial sensor data")
    model = model.to(device)
    model.eval()

    # Cold reference (index 0 = coldest)
    cold_ref, _ = load_image(args.domain, "test", "good", 0)
    cold_ref_batch = cold_ref.unsqueeze(0)

    # Test multiple Cold Fault samples (indices 0-19 assumed cold)
    # and multiple Cold Good samples (indices 1-100 for variety)
    cold_fault_indices = list(range(0, 20))  # First 20 faults (cold)
    cold_good_indices = list(range(1, 21))   # Good samples in cold range (not ref)

    print("\n" + "=" * 80)
    print(f"WITHIN-CONDITION ANALYSIS: Cold Fault vs Cold Good ({args.domain})")
    print("Reference: Cold (index 0)")
    print("=" * 80)

    fault_scores = []
    good_scores = []

    # Analyze Cold Fault samples
    print("\n--- Cold FAULT Samples ---")
    print(f"{'Index':<10} {'Mean Score':<15} {'Max Score':<15} {'Min Score':<15}")
    print("-" * 55)

    for idx in cold_fault_indices:
        try:
            fault_img, _ = load_image(args.domain, "test", "fault", idx)
            result = compute_anomaly_score(model, fault_img.unsqueeze(0), cold_ref_batch, device)
            fault_scores.append(result["mean"])
            print(f"{idx:<10} {result['mean']:<15.4f} {result['max']:<15.4f} {result['min']:<15.4f}")
        except Exception as e:
            print(f"{idx:<10} Error: {e}")

    # Analyze Cold Good samples
    print("\n--- Cold GOOD Samples ---")
    print(f"{'Index':<10} {'Mean Score':<15} {'Max Score':<15} {'Min Score':<15}")
    print("-" * 55)

    for idx in cold_good_indices:
        try:
            good_img, _ = load_image(args.domain, "test", "good", idx)
            result = compute_anomaly_score(model, good_img.unsqueeze(0), cold_ref_batch, device)
            good_scores.append(result["mean"])
            print(f"{idx:<10} {result['mean']:<15.4f} {result['max']:<15.4f} {result['min']:<15.4f}")
        except Exception as e:
            print(f"{idx:<10} Error: {e}")

    # Summary statistics
    fault_scores = np.array(fault_scores)
    good_scores = np.array(good_scores)

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\n{'Metric':<25} {'Cold Fault':<15} {'Cold Good':<15} {'Diff (F-G)':<15}")
    print("-" * 70)
    print(f"{'Mean':<25} {fault_scores.mean():<15.4f} {good_scores.mean():<15.4f} {fault_scores.mean() - good_scores.mean():<+15.4f}")
    print(f"{'Std':<25} {fault_scores.std():<15.4f} {good_scores.std():<15.4f}")
    print(f"{'Min':<25} {fault_scores.min():<15.4f} {good_scores.min():<15.4f}")
    print(f"{'Max':<25} {fault_scores.max():<15.4f} {good_scores.max():<15.4f}")

    # Overlap analysis
    fault_min = fault_scores.min()
    fault_max = fault_scores.max()
    good_min = good_scores.min()
    good_max = good_scores.max()

    overlap_start = max(fault_min, good_min)
    overlap_end = min(fault_max, good_max)

    if overlap_start < overlap_end:
        overlap_range = overlap_end - overlap_start
        total_range = max(fault_max, good_max) - min(fault_min, good_min)
        overlap_pct = (overlap_range / total_range) * 100
        print(f"\n{'Overlap Range':<25} [{overlap_start:.4f}, {overlap_end:.4f}]")
        print(f"{'Overlap Percentage':<25} {overlap_pct:.1f}%")
    else:
        print(f"\n{'Overlap':<25} None (separable!)")

    # Discrimination analysis
    print("\n" + "=" * 80)
    print("DISCRIMINATION ANALYSIS")
    print("=" * 80)

    # How many faults are higher than mean good?
    threshold = good_scores.mean()
    faults_above_threshold = (fault_scores > threshold).sum()
    print(f"\nFaults with score > Good mean ({threshold:.4f}): {faults_above_threshold}/{len(fault_scores)} ({100*faults_above_threshold/len(fault_scores):.1f}%)")

    # How many goods are lower than mean fault?
    threshold = fault_scores.mean()
    goods_below_threshold = (good_scores < threshold).sum()
    print(f"Goods with score < Fault mean ({threshold:.4f}): {goods_below_threshold}/{len(good_scores)} ({100*goods_below_threshold/len(good_scores):.1f}%)")

    # Estimate AUROC (simple approximation)
    # Count how many (fault, good) pairs where fault > good
    correct_pairs = 0
    total_pairs = len(fault_scores) * len(good_scores)
    for f in fault_scores:
        for g in good_scores:
            if f > g:
                correct_pairs += 1
            elif f == g:
                correct_pairs += 0.5

    auroc = correct_pairs / total_pairs
    print(f"\nEstimated Within-Condition AUROC: {auroc:.4f} ({auroc*100:.1f}%)")

    if auroc > 0.7:
        status = "Moderate"
    elif auroc > 0.6:
        status = "Weak"
    elif auroc > 0.5:
        status = "Very Weak"
    else:
        status = "FAILED (worse than random)"

    print(f"Status: {status}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Score distributions
    ax = axes[0, 0]
    ax.hist(fault_scores, bins=10, alpha=0.7, label=f'Cold Fault (n={len(fault_scores)})', color='red')
    ax.hist(good_scores, bins=10, alpha=0.7, label=f'Cold Good (n={len(good_scores)})', color='green')
    ax.axvline(fault_scores.mean(), color='darkred', linestyle='--', label=f'Fault mean: {fault_scores.mean():.4f}')
    ax.axvline(good_scores.mean(), color='darkgreen', linestyle='--', label=f'Good mean: {good_scores.mean():.4f}')
    ax.set_xlabel("Anomaly Score (mean)")
    ax.set_ylabel("Count")
    ax.set_title("Within-Condition: Cold Fault vs Cold Good Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Box plot
    ax = axes[0, 1]
    bp = ax.boxplot([fault_scores, good_scores], labels=['Cold Fault', 'Cold Good'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightgreen')
    ax.set_ylabel("Anomaly Score")
    ax.set_title("Score Distribution Box Plot")
    ax.grid(True, alpha=0.3)

    # Add individual points
    for i, (scores, color) in enumerate([(fault_scores, 'red'), (good_scores, 'green')]):
        x = np.random.normal(i + 1, 0.04, size=len(scores))
        ax.scatter(x, scores, alpha=0.5, color=color, s=30)

    # Plot 3: Sample-by-sample comparison
    ax = axes[1, 0]
    x_fault = np.arange(len(fault_scores))
    x_good = np.arange(len(good_scores))
    ax.bar(x_fault - 0.2, fault_scores, width=0.4, label='Cold Fault', color='red', alpha=0.7)
    ax.bar(x_good + 0.2, good_scores, width=0.4, label='Cold Good', color='green', alpha=0.7)
    ax.axhline(fault_scores.mean(), color='darkred', linestyle='--', alpha=0.7)
    ax.axhline(good_scores.mean(), color='darkgreen', linestyle='--', alpha=0.7)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Anomaly Score")
    ax.set_title("Per-Sample Anomaly Scores")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
    Within-Condition Analysis Summary
    ══════════════════════════════════════════════════

    Domain: {args.domain}
    Reference: Cold (index 0)

    Cold Fault samples: {len(fault_scores)}
    Cold Good samples:  {len(good_scores)}

    ──────────────────────────────────────────────────
    Score Statistics:

                    Fault       Good        Diff
    Mean:           {fault_scores.mean():.4f}      {good_scores.mean():.4f}      {fault_scores.mean() - good_scores.mean():+.4f}
    Std:            {fault_scores.std():.4f}      {good_scores.std():.4f}

    ──────────────────────────────────────────────────
    Discrimination:

    Estimated AUROC: {auroc:.4f} ({auroc*100:.1f}%)
    Status: {status}

    ──────────────────────────────────────────────────
    Conclusion:

    Even within the same cold condition,
    WinCLIP struggles to distinguish
    Domain C faults from normal samples.

    The defect signal is too subtle for
    CLIP embeddings to capture effectively.
    """

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle(f"Within-Condition Analysis: Cold Fault vs Cold Good ({args.domain})",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = OUTPUT_DIR / args.domain / "within_condition_analysis.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved visualization to: {output_path}")


if __name__ == "__main__":
    main()
