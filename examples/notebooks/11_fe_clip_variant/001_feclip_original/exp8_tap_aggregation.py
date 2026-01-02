"""Experiment 8: Tap aggregation alternatives for PRO optimization.

Test different tap aggregation methods without retraining:
1. avg (current) - simple average across taps
2. max - maximum across taps
3. weighted - decreasing weights for later taps (e.g., [0.4, 0.3, 0.2, 0.1])
4. best_single - use only the best performing tap

Hypothesis: Average dilutes good taps with noisy ones, hurting PRO.
"""

import argparse
import logging
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.v2 import Resize, CenterCrop, InterpolationMode

from anomalib.data import MVTecAD, Visa
from anomalib.models.image import FEClip
from anomalib.metrics.aupro import _AUPRO
from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
MVTEC_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/MVTecAD")
VISA_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/VisA")

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"
]
VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
]

IMAGE_SIZE = 336


def aggregate_maps(tap_maps, method="avg", weights=None):
    """Aggregate multiple tap maps using different methods.

    Args:
        tap_maps: List of (H, W) numpy arrays
        method: One of "avg", "max", "weighted", "tap0", "tap1", "tap2", "tap3"
        weights: Optional weights for "weighted" method

    Returns:
        (H, W) aggregated map
    """
    stacked = np.stack(tap_maps, axis=0)  # (n_taps, H, W)

    if method == "avg":
        return np.mean(stacked, axis=0)
    elif method == "max":
        return np.max(stacked, axis=0)
    elif method == "weighted":
        if weights is None:
            weights = np.array([0.4, 0.3, 0.2, 0.1])
        weights = weights / weights.sum()
        return np.sum(stacked * weights[:, None, None], axis=0)
    elif method == "weighted_reverse":
        # Increasing weights for later taps
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        weights = weights / weights.sum()
        return np.sum(stacked * weights[:, None, None], axis=0)
    elif method.startswith("tap"):
        idx = int(method[3:])
        return stacked[idx]
    else:
        raise ValueError(f"Unknown method: {method}")


def aggregate_scores(tap_scores, method="avg"):
    """Aggregate multiple tap scores."""
    stacked = np.array(tap_scores)

    if method == "avg":
        return np.mean(stacked)
    elif method == "max":
        return np.max(stacked)
    elif method == "weighted":
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        weights = weights / weights.sum()
        return np.sum(stacked * weights)
    elif method == "weighted_reverse":
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        weights = weights / weights.sum()
        return np.sum(stacked * weights)
    elif method.startswith("tap"):
        idx = int(method[3:])
        return stacked[idx]
    else:
        return np.mean(stacked)


def evaluate_with_aggregation(model, dataset_class, root, categories, device, transform, agg_method):
    """Evaluate model with specific aggregation method."""
    model.eval()

    mask_resize = Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST, antialias=False)
    mask_crop = CenterCrop((IMAGE_SIZE, IMAGE_SIZE))

    all_scores = []
    all_labels = []
    all_maps = []
    all_masks = []

    for cat in categories:
        dm = dataset_class(root=root, category=cat, eval_batch_size=1, num_workers=4)
        dm.setup()

        for batch in dm.test_dataloader():
            images = transform(batch.image).to(device)
            gt_label = batch.gt_label.item()
            gt_mask = batch.gt_mask[0] if batch.gt_mask is not None else None

            with torch.no_grad():
                # Get per-tap outputs
                scores_list, maps_list = model.model.forward_tokens(images)

            # Convert to numpy and upsample maps to IMAGE_SIZE
            tap_scores = [s[0].cpu().item() for s in scores_list]
            tap_maps = []
            for m in maps_list:
                # Upsample from feature size (24x24) to IMAGE_SIZE (336x336)
                m_up = F.interpolate(
                    m.unsqueeze(1), size=(IMAGE_SIZE, IMAGE_SIZE),
                    mode="bilinear", align_corners=False
                ).squeeze(1)
                tap_maps.append(m_up[0].cpu().numpy())

            # Aggregate
            agg_score = aggregate_scores(tap_scores, agg_method)
            agg_map = aggregate_maps(tap_maps, agg_method)

            all_scores.append(agg_score)
            all_labels.append(gt_label)
            all_maps.append(agg_map)

            if gt_mask is not None:
                if gt_mask.ndim == 2:
                    gt_mask = gt_mask.unsqueeze(0)
                gt_mask_processed = mask_crop(mask_resize(gt_mask)).squeeze(0).numpy()
                all_masks.append((gt_mask_processed > 0.5).astype(int))
            else:
                all_masks.append(np.zeros_like(agg_map))

    # Image-level metrics
    auroc = roc_auc_score(all_labels, all_scores) * 100
    ap = average_precision_score(all_labels, all_scores) * 100

    # Pixel-level metrics
    all_maps_flat = np.concatenate([m.flatten() for m in all_maps])
    all_masks_flat = np.concatenate([m.flatten() for m in all_masks])
    pauroc = roc_auc_score(all_masks_flat, all_maps_flat) * 100

    # Skip PRO for now due to connected_components bug - use pAUROC for comparison
    return {"AUROC": auroc, "AP": ap, "pAUROC": pauroc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="visa", choices=["mvtec", "visa"])
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.dataset == "visa":
        CHECKPOINT = "examples/notebooks/11_fe_clip_variant/results/feclip_visa_seed42_20260101_061222/checkpoints/best_model.pt"
        dataset_class = Visa
        root = VISA_ROOT
        categories = VISA_CATEGORIES
    else:
        # MVTec checkpoint (trained on VisA)
        CHECKPOINT = "examples/notebooks/11_fe_clip_variant/results/feclip_mvtec_seed42_20260101_061222/checkpoints/best_model.pt"
        dataset_class = MVTecAD
        root = MVTEC_ROOT
        categories = MVTEC_CATEGORIES

    logger.info("=" * 80)
    logger.info(f"Experiment 8: Tap Aggregation Methods ({args.dataset})")
    logger.info("=" * 80)

    # Load model
    logger.info(f"Loading model from {CHECKPOINT}...")
    model = FEClip(tap_indices=[20, 21, 22, 23])
    model.to(args.device)
    checkpoint = torch.load(CHECKPOINT, map_location=args.device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    transform = model.pre_processor.transform

    # Methods to test
    methods = ["avg", "max", "weighted", "weighted_reverse", "tap0", "tap1", "tap2", "tap3"]

    results = {}
    for method in methods:
        logger.info(f"\nTesting method: {method}")
        result = evaluate_with_aggregation(model, dataset_class, root, categories, args.device, transform, method)
        results[method] = result
        logger.info(f"  AUROC: {result['AUROC']:.1f}%, pAUROC: {result['pAUROC']:.1f}%")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info(f"SUMMARY: Tap Aggregation Methods ({args.dataset})")
    logger.info("=" * 80)
    logger.info(f"{'Method':<20} {'AUROC':>10} {'pAUROC':>10}")
    logger.info("-" * 45)

    baseline_pauroc = results["avg"]["pAUROC"]
    for method, r in results.items():
        gap = r["pAUROC"] - baseline_pauroc
        gap_str = f"({gap:+.1f}%)" if method != "avg" else "(baseline)"
        logger.info(f"{method:<20} {r['AUROC']:>9.1f}% {r['pAUROC']:>9.1f}% {gap_str}")

    # Find best for pAUROC
    best_method = max(results.keys(), key=lambda m: results[m]["pAUROC"])
    logger.info(f"\nBest method for pAUROC: {best_method} ({results[best_method]['pAUROC']:.1f}%)")

    # Save results
    output_path = SCRIPT_DIR / f"exp8_tap_aggregation_{args.dataset}_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
