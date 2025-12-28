#!/usr/bin/env python3
"""Multi-scale Analysis for WinCLIP on HDMAP.

Analyzes discrimination capability at each scale:
- Full image (CLS token)
- Small scale (2x2 windows, 14x14 = 196 windows)
- Mid scale (3x3 windows, 13x13 = 169 windows)
- Patch level (15x15 = 225 patches)

Compares:
1. Cold vs Warm reference discrimination
2. Fault vs Good discrimination within same condition
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

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


def compute_scale_similarities(model, test_batch, ref_batch, device):
    """Compute similarities at each scale.

    Returns dict with similarities for each scale:
    - full_image: similarity between CLS tokens
    - small_scale: max similarity for 2x2 windows
    - mid_scale: max similarity for 3x3 windows
    - patch_level: max similarity for patches
    """
    with torch.no_grad():
        test_img_emb, test_win_embs, test_patch_emb = model.encode_image(test_batch.to(device))
        ref_img_emb, ref_win_embs, ref_patch_emb = model.encode_image(ref_batch.to(device))

    results = {}

    # 1. Full image (CLS token) similarity
    # Shape: test_img_emb (N, D), ref_img_emb (K, D)
    img_sim = cosine_similarity(test_img_emb, ref_img_emb)  # (N, K)
    results["full_image"] = {
        "similarity": img_sim.max(dim=-1).values.cpu().numpy(),  # (N,)
        "shape": test_img_emb.shape,
        "name": "Full Image (CLS)",
    }

    # 2. Small scale (2x2 windows) - index 0
    # Shape: test_win_embs[0] (N, 196, D), ref_win_embs[0] (K, 196, D)
    small_win_test = test_win_embs[0].squeeze(0)  # (196, D)
    small_win_ref = ref_win_embs[0].squeeze(0)    # (196, D)
    small_sim = cosine_similarity(small_win_test, small_win_ref)  # (196, 196)
    small_max_sim = small_sim.max(dim=-1).values  # (196,)
    results["small_scale"] = {
        "similarity": small_max_sim.cpu().numpy(),
        "grid_size": (14, 14),
        "shape": test_win_embs[0].shape,
        "name": "Small Scale (2x2 win, 14x14)",
    }

    # 3. Mid scale (3x3 windows) - index 1
    # Shape: test_win_embs[1] (N, 169, D), ref_win_embs[1] (K, 169, D)
    mid_win_test = test_win_embs[1].squeeze(0)  # (169, D)
    mid_win_ref = ref_win_embs[1].squeeze(0)    # (169, D)
    mid_sim = cosine_similarity(mid_win_test, mid_win_ref)  # (169, 169)
    mid_max_sim = mid_sim.max(dim=-1).values  # (169,)
    results["mid_scale"] = {
        "similarity": mid_max_sim.cpu().numpy(),
        "grid_size": (13, 13),
        "shape": test_win_embs[1].shape,
        "name": "Mid Scale (3x3 win, 13x13)",
    }

    # 4. Patch level
    # Shape: test_patch_emb (N, 225, D), ref_patch_emb (K, 225, D)
    patch_test = test_patch_emb.squeeze(0)  # (225, D)
    patch_ref = ref_patch_emb.squeeze(0)    # (225, D)
    patch_sim = cosine_similarity(patch_test, patch_ref)  # (225, 225)
    patch_max_sim = patch_sim.max(dim=-1).values  # (225,)
    results["patch_level"] = {
        "similarity": patch_max_sim.cpu().numpy(),
        "grid_size": (15, 15),
        "shape": test_patch_emb.shape,
        "name": "Patch Level (15x15)",
    }

    return results


def analyze_discrimination(model, device, domain, cold_indices, warm_indices, fault_indices, good_indices):
    """Analyze discrimination at each scale."""

    # Load reference images
    cold_ref, _ = load_image(domain, "test", "good", cold_indices[0])
    warm_ref, _ = load_image(domain, "test", "good", warm_indices[0])

    # Load test images
    cold_fault, _ = load_image(domain, "test", "fault", fault_indices[0])  # Cold fault
    cold_good, _ = load_image(domain, "test", "good", good_indices[0])    # Cold good (different from ref)
    warm_fault, _ = load_image(domain, "test", "fault", fault_indices[1])  # Warm fault
    warm_good, _ = load_image(domain, "test", "good", good_indices[1])    # Warm good (different from ref)

    # Prepare batches
    cold_ref_batch = cold_ref.unsqueeze(0)
    warm_ref_batch = warm_ref.unsqueeze(0)

    test_samples = {
        "Cold Fault": cold_fault.unsqueeze(0),
        "Cold Good": cold_good.unsqueeze(0),
        "Warm Fault": warm_fault.unsqueeze(0),
        "Warm Good": warm_good.unsqueeze(0),
    }

    all_results = {}

    print("\n" + "=" * 80)
    print("MULTI-SCALE ANALYSIS")
    print("=" * 80)

    for test_name, test_batch in test_samples.items():
        print(f"\n--- {test_name} ---")

        # Compare to Cold reference
        cold_results = compute_scale_similarities(model, test_batch, cold_ref_batch, device)

        # Compare to Warm reference
        warm_results = compute_scale_similarities(model, test_batch, warm_ref_batch, device)

        all_results[test_name] = {
            "to_cold": cold_results,
            "to_warm": warm_results,
        }

        print(f"\n{'Scale':<30} {'To Cold':<15} {'To Warm':<15} {'Max (Combined)':<15}")
        print("-" * 75)

        for scale in ["full_image", "small_scale", "mid_scale", "patch_level"]:
            cold_sim = cold_results[scale]["similarity"]
            warm_sim = warm_results[scale]["similarity"]

            if scale == "full_image":
                cold_val = cold_sim[0]
                warm_val = warm_sim[0]
                combined = max(cold_val, warm_val)
            else:
                cold_val = cold_sim.mean()
                warm_val = warm_sim.mean()
                combined = np.maximum(cold_sim, warm_sim).mean()

            scale_name = cold_results[scale]["name"]
            print(f"{scale_name:<30} {cold_val:<15.4f} {warm_val:<15.4f} {combined:<15.4f}")

    return all_results, {
        "cold_ref": cold_ref,
        "warm_ref": warm_ref,
        "cold_fault": cold_fault,
        "cold_good": cold_good,
        "warm_fault": warm_fault,
        "warm_good": warm_good,
    }


def compute_anomaly_scores(results):
    """Compute anomaly scores from similarities."""
    scores = {}

    for test_name, test_results in results.items():
        scores[test_name] = {}

        for scale in ["full_image", "small_scale", "mid_scale", "patch_level"]:
            cold_sim = test_results["to_cold"][scale]["similarity"]
            warm_sim = test_results["to_warm"][scale]["similarity"]

            # Combined max similarity
            combined = np.maximum(cold_sim, warm_sim)

            # Anomaly score = (1 - max_similarity) / 2
            anomaly = (1 - combined) / 2

            if scale == "full_image":
                scores[test_name][scale] = anomaly[0]
            else:
                scores[test_name][scale] = {
                    "mean": anomaly.mean(),
                    "max": anomaly.max(),
                    "min": anomaly.min(),
                    "map": anomaly,
                }

    return scores


def visualize_multiscale(results, images, scores, output_path, domain):
    """Visualize multi-scale analysis results."""

    fig = plt.figure(figsize=(24, 20))

    # Create grid spec
    gs = fig.add_gridspec(5, 6, hspace=0.4, wspace=0.3)

    def norm_display(t):
        arr = t.permute(1, 2, 0).numpy() if len(t.shape) == 3 else t.numpy()
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    # Row 0: Reference images
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(norm_display(images["cold_ref"]))
    ax.set_title("Cold Reference\n(index 0)", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(norm_display(images["warm_ref"]))
    ax.set_title("Warm Reference\n(index 999)", fontsize=10)
    ax.axis("off")

    # Test images
    test_imgs = [
        ("Cold Fault", images["cold_fault"]),
        ("Cold Good", images["cold_good"]),
        ("Warm Fault", images["warm_fault"]),
        ("Warm Good", images["warm_good"]),
    ]

    for i, (name, img) in enumerate(test_imgs):
        ax = fig.add_subplot(gs[0, 2 + i])
        ax.imshow(norm_display(img))
        ax.set_title(f"{name}", fontsize=10)
        ax.axis("off")

    # Row 1-4: Anomaly maps per scale
    scales = ["full_image", "small_scale", "mid_scale", "patch_level"]
    scale_names = ["Full Image", "Small (2x2)", "Mid (3x3)", "Patch (1x1)"]
    grid_sizes = [None, (14, 14), (13, 13), (15, 15)]

    test_names = ["Cold Fault", "Cold Good", "Warm Fault", "Warm Good"]

    for row, (scale, scale_name, grid_size) in enumerate(zip(scales, scale_names, grid_sizes)):
        # Scale label
        ax = fig.add_subplot(gs[row + 1, 0])
        ax.axis("off")
        ax.text(0.5, 0.5, scale_name, fontsize=14, fontweight='bold',
                ha='center', va='center', transform=ax.transAxes)

        # Stats summary for this scale
        ax = fig.add_subplot(gs[row + 1, 1])
        ax.axis("off")

        if scale == "full_image":
            text = "Single score\nper image"
            for name in test_names:
                score = scores[name][scale]
                text += f"\n{name}: {score:.4f}"
        else:
            text = f"Grid: {grid_size[0]}x{grid_size[1]}"
            for name in test_names:
                score = scores[name][scale]["mean"]
                text += f"\n{name}: {score:.4f}"

        ax.text(0.1, 0.5, text, fontsize=9, va='center', transform=ax.transAxes,
                fontfamily='monospace')

        # Anomaly maps for each test sample
        for col, test_name in enumerate(test_names):
            ax = fig.add_subplot(gs[row + 1, col + 2])

            if scale == "full_image":
                # Just show the score as text
                score = scores[test_name][scale]
                ax.text(0.5, 0.5, f"Score:\n{score:.4f}", fontsize=14,
                        ha='center', va='center', transform=ax.transAxes,
                        fontweight='bold')
                ax.set_facecolor('lightgray')
            else:
                # Show anomaly map
                amap = scores[test_name][scale]["map"].reshape(grid_size)
                im = ax.imshow(amap, cmap="hot", vmin=0, vmax=0.15)
                mean_score = scores[test_name][scale]["mean"]
                ax.set_title(f"mean={mean_score:.4f}", fontsize=9)

            ax.axis("off")

    plt.suptitle(f"WinCLIP Multi-Scale Analysis ({domain})\nAnomaly Score Maps at Different Scales",
                 fontsize=14, fontweight='bold')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved visualization to: {output_path}")


def print_discrimination_summary(scores):
    """Print discrimination summary."""

    print("\n" + "=" * 90)
    print("DISCRIMINATION SUMMARY: Fault vs Good Anomaly Scores")
    print("=" * 90)

    scales = ["full_image", "small_scale", "mid_scale", "patch_level"]
    scale_names = ["Full Image", "Small (2x2)", "Mid (3x3)", "Patch (1x1)"]

    print(f"\n{'Scale':<15} {'Cold Fault':<12} {'Cold Good':<12} {'Diff (F-G)':<12} {'Warm Fault':<12} {'Warm Good':<12} {'Diff (F-G)':<12}")
    print("-" * 90)

    best_cold_scale = None
    best_cold_diff = -float('inf')
    best_warm_scale = None
    best_warm_diff = -float('inf')

    for scale, scale_name in zip(scales, scale_names):
        if scale == "full_image":
            cf = scores["Cold Fault"][scale]
            cg = scores["Cold Good"][scale]
            wf = scores["Warm Fault"][scale]
            wg = scores["Warm Good"][scale]
        else:
            cf = scores["Cold Fault"][scale]["mean"]
            cg = scores["Cold Good"][scale]["mean"]
            wf = scores["Warm Fault"][scale]["mean"]
            wg = scores["Warm Good"][scale]["mean"]

        cold_diff = cf - cg
        warm_diff = wf - wg

        if cold_diff > best_cold_diff:
            best_cold_diff = cold_diff
            best_cold_scale = scale_name
        if warm_diff > best_warm_diff:
            best_warm_diff = warm_diff
            best_warm_scale = scale_name

        cold_status = "+" if cold_diff > 0 else ""
        warm_status = "+" if warm_diff > 0 else ""

        print(f"{scale_name:<15} {cf:<12.4f} {cg:<12.4f} {cold_status}{cold_diff:<12.4f} {wf:<12.4f} {wg:<12.4f} {warm_status}{warm_diff:<12.4f}")

    print("-" * 90)
    print(f"\nBest scale for Cold discrimination: {best_cold_scale} (diff = {best_cold_diff:+.4f})")
    print(f"Best scale for Warm discrimination: {best_warm_scale} (diff = {best_warm_diff:+.4f})")

    # Cross-condition analysis
    print("\n" + "=" * 90)
    print("CROSS-CONDITION: Cold Fault vs Warm Good (Key Challenge)")
    print("=" * 90)

    print(f"\n{'Scale':<15} {'Cold Fault':<12} {'Warm Good':<12} {'Diff':<12} {'Status'}")
    print("-" * 60)

    for scale, scale_name in zip(scales, scale_names):
        if scale == "full_image":
            cf = scores["Cold Fault"][scale]
            wg = scores["Warm Good"][scale]
        else:
            cf = scores["Cold Fault"][scale]["mean"]
            wg = scores["Warm Good"][scale]["mean"]

        diff = cf - wg

        if diff > 0.01:
            status = "GOOD"
        elif diff > 0:
            status = "Weak"
        else:
            status = "FAILED"

        print(f"{scale_name:<15} {cf:<12.4f} {wg:<12.4f} {diff:<+12.4f} {status}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="domain_C")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model
    print("Loading WinCLIP model...")
    model = WinClipModel(class_name="industrial sensor data")
    model = model.to(device)
    model.eval()

    print(f"Grid size: {model.grid_size}")
    print(f"Scales: {model.scales}")

    # Define sample indices
    # Cold samples: index 0-499 (0 is coldest)
    # Warm samples: index 500-999 (999 is warmest)
    cold_indices = [0]  # Cold reference
    warm_indices = [999]  # Warm reference

    # For fault samples, we need to check which are cold/warm
    # Let's use fault indices 0-9 (assumed cold) and 10-19 (check warm)
    # Actually, let's use specific indices we know
    fault_indices = [9, 15]  # Cold fault, Warm fault (need to verify)
    good_indices = [100, 800]  # Cold good (different from ref), Warm good

    # Run analysis
    results, images = analyze_discrimination(
        model, device, args.domain,
        cold_indices, warm_indices,
        fault_indices, good_indices
    )

    # Compute anomaly scores
    scores = compute_anomaly_scores(results)

    # Print summary
    print_discrimination_summary(scores)

    # Visualize
    output_path = OUTPUT_DIR / args.domain / "multiscale_analysis.png"
    visualize_multiscale(results, images, scores, output_path, args.domain)


if __name__ == "__main__":
    main()
