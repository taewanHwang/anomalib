"""Experiment 9: PRO Metric Implementation Comparison.

Priority 1: Compare anomalib's AUPRO with the original PRO definition.

The hypothesis is that "pAUROC matches but PRO doesn't" pattern suggests
metric calculation differences, not model issues.

Key differences to check:
1. Threshold sweep method: ROC-based vs linspace
2. Number of thresholds
3. FPR limit handling
4. Connected component labeling
5. Per-image vs global normalization

References:
- PRO was introduced in MVTec AD paper (Bergmann et al., 2019)
- Standard implementation: threshold sweep + per-region recall + integration
"""

import argparse
import logging
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.v2 import Resize, CenterCrop, InterpolationMode
from scipy import ndimage
from scipy.integrate import trapezoid
from skimage.measure import label as sk_label
from sklearn.metrics import roc_curve, auc

from anomalib.data import MVTecAD, Visa
from anomalib.models.image import FEClip

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


# ============================================================================
# Method 1: Original PRO (MVTec paper style - linspace thresholds)
# ============================================================================

def compute_pro_original(predictions: list, masks: list, num_thresholds: int = 200, fpr_limit: float = 0.3):
    """Compute PRO using original MVTec paper style.

    Key characteristics:
    - Linspace thresholds from min to max of predictions
    - Per-region overlap (recall) for each connected component
    - Integration over FPR (not threshold)
    """
    # Flatten for FPR calculation
    all_preds = np.concatenate([p.flatten() for p in predictions])
    all_masks = np.concatenate([m.flatten() for m in masks])

    # Get thresholds using linspace (original style)
    pred_min, pred_max = all_preds.min(), all_preds.max()
    thresholds = np.linspace(pred_max, pred_min, num_thresholds)

    pros = []
    fprs = []

    for thresh in thresholds:
        # Compute FPR at this threshold (global)
        pred_binary_all = (all_preds >= thresh).astype(int)
        fp = np.sum((pred_binary_all == 1) & (all_masks == 0))
        tn = np.sum((pred_binary_all == 0) & (all_masks == 0))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        # Compute per-region overlap
        region_overlaps = []
        for pred, mask in zip(predictions, masks):
            if mask.sum() == 0:
                continue

            pred_binary = (pred >= thresh).astype(int)

            # Label connected components in GT mask
            labeled_mask, num_components = ndimage.label(mask)

            for comp_idx in range(1, num_components + 1):
                component_mask = (labeled_mask == comp_idx)
                component_size = component_mask.sum()

                # Overlap = how much of this component is covered by prediction
                overlap = np.sum(pred_binary[component_mask]) / component_size
                region_overlaps.append(overlap)

        if region_overlaps:
            pro = np.mean(region_overlaps)
        else:
            pro = 1.0

        pros.append(pro)
        fprs.append(fpr)

    # Convert to numpy and sort by FPR
    fprs = np.array(fprs)
    pros = np.array(pros)

    # Sort by FPR (ascending)
    sorted_idx = np.argsort(fprs)
    fprs = fprs[sorted_idx]
    pros = pros[sorted_idx]

    # Apply FPR limit
    valid_idx = fprs <= fpr_limit
    if valid_idx.sum() < 2:
        return 0.0

    fprs_limited = fprs[valid_idx]
    pros_limited = pros[valid_idx]

    # Normalize FPR to [0, 1] within limit
    fprs_normalized = fprs_limited / fpr_limit

    # Compute AUC
    aupro = trapezoid(pros_limited, fprs_normalized)

    return aupro * 100


# ============================================================================
# Method 2: Quantile-based thresholds (common alternative)
# ============================================================================

def compute_pro_quantile(predictions: list, masks: list, num_thresholds: int = 200, fpr_limit: float = 0.3):
    """Compute PRO using quantile-based thresholds."""
    all_preds = np.concatenate([p.flatten() for p in predictions])
    all_masks = np.concatenate([m.flatten() for m in masks])

    # Use quantiles instead of linspace
    quantiles = np.linspace(1, 0, num_thresholds)
    thresholds = np.quantile(all_preds, quantiles)

    pros = []
    fprs = []

    for thresh in thresholds:
        pred_binary_all = (all_preds >= thresh).astype(int)
        fp = np.sum((pred_binary_all == 1) & (all_masks == 0))
        tn = np.sum((pred_binary_all == 0) & (all_masks == 0))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        region_overlaps = []
        for pred, mask in zip(predictions, masks):
            if mask.sum() == 0:
                continue

            pred_binary = (pred >= thresh).astype(int)
            labeled_mask, num_components = ndimage.label(mask)

            for comp_idx in range(1, num_components + 1):
                component_mask = (labeled_mask == comp_idx)
                component_size = component_mask.sum()
                overlap = np.sum(pred_binary[component_mask]) / component_size
                region_overlaps.append(overlap)

        pro = np.mean(region_overlaps) if region_overlaps else 1.0
        pros.append(pro)
        fprs.append(fpr)

    fprs = np.array(fprs)
    pros = np.array(pros)
    sorted_idx = np.argsort(fprs)
    fprs = fprs[sorted_idx]
    pros = pros[sorted_idx]

    valid_idx = fprs <= fpr_limit
    if valid_idx.sum() < 2:
        return 0.0

    fprs_limited = fprs[valid_idx]
    pros_limited = pros[valid_idx]
    fprs_normalized = fprs_limited / fpr_limit

    aupro = trapezoid(pros_limited, fprs_normalized)
    return aupro * 100


# ============================================================================
# Method 3: ROC-based thresholds (like sklearn)
# ============================================================================

def compute_pro_roc_based(predictions: list, masks: list, fpr_limit: float = 0.3):
    """Compute PRO using ROC curve thresholds (sklearn style)."""
    all_preds = np.concatenate([p.flatten() for p in predictions])
    all_masks = np.concatenate([m.flatten() for m in masks])

    # Get thresholds from ROC curve
    fprs_roc, tprs_roc, thresholds = roc_curve(all_masks, all_preds)

    # Use unique thresholds (may be many)
    pros = []
    fprs = []

    for thresh in thresholds[::max(1, len(thresholds)//200)]:  # Sample ~200 thresholds
        pred_binary_all = (all_preds >= thresh).astype(int)
        fp = np.sum((pred_binary_all == 1) & (all_masks == 0))
        tn = np.sum((pred_binary_all == 0) & (all_masks == 0))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        region_overlaps = []
        for pred, mask in zip(predictions, masks):
            if mask.sum() == 0:
                continue

            pred_binary = (pred >= thresh).astype(int)
            labeled_mask, num_components = ndimage.label(mask)

            for comp_idx in range(1, num_components + 1):
                component_mask = (labeled_mask == comp_idx)
                component_size = component_mask.sum()
                overlap = np.sum(pred_binary[component_mask]) / component_size
                region_overlaps.append(overlap)

        pro = np.mean(region_overlaps) if region_overlaps else 1.0
        pros.append(pro)
        fprs.append(fpr)

    fprs = np.array(fprs)
    pros = np.array(pros)
    sorted_idx = np.argsort(fprs)
    fprs = fprs[sorted_idx]
    pros = pros[sorted_idx]

    valid_idx = fprs <= fpr_limit
    if valid_idx.sum() < 2:
        return 0.0

    fprs_limited = fprs[valid_idx]
    pros_limited = pros[valid_idx]
    fprs_normalized = fprs_limited / fpr_limit

    aupro = trapezoid(pros_limited, fprs_normalized)
    return aupro * 100


# ============================================================================
# Method 4: Per-image PRO then average (alternative normalization)
# ============================================================================

def compute_pro_per_image(predictions: list, masks: list, num_thresholds: int = 200, fpr_limit: float = 0.3):
    """Compute PRO per-image then average (alternative to global)."""
    per_image_pros = []

    for pred, mask in zip(predictions, masks):
        if mask.sum() == 0:
            continue

        pred_flat = pred.flatten()
        mask_flat = mask.flatten()

        pred_min, pred_max = pred_flat.min(), pred_flat.max()
        if pred_max == pred_min:
            continue

        thresholds = np.linspace(pred_max, pred_min, num_thresholds)

        pros = []
        fprs = []

        for thresh in thresholds:
            pred_binary = (pred >= thresh).astype(int)
            pred_binary_flat = pred_binary.flatten()

            fp = np.sum((pred_binary_flat == 1) & (mask_flat == 0))
            tn = np.sum((pred_binary_flat == 0) & (mask_flat == 0))
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            labeled_mask, num_components = ndimage.label(mask)
            region_overlaps = []

            for comp_idx in range(1, num_components + 1):
                component_mask = (labeled_mask == comp_idx)
                component_size = component_mask.sum()
                overlap = np.sum(pred_binary[component_mask]) / component_size
                region_overlaps.append(overlap)

            pro = np.mean(region_overlaps) if region_overlaps else 1.0
            pros.append(pro)
            fprs.append(fpr)

        fprs = np.array(fprs)
        pros = np.array(pros)
        sorted_idx = np.argsort(fprs)
        fprs = fprs[sorted_idx]
        pros = pros[sorted_idx]

        valid_idx = fprs <= fpr_limit
        if valid_idx.sum() >= 2:
            fprs_limited = fprs[valid_idx]
            pros_limited = pros[valid_idx]
            fprs_normalized = fprs_limited / fpr_limit
            aupro = trapezoid(pros_limited, fprs_normalized)
            per_image_pros.append(aupro)

    return np.mean(per_image_pros) * 100 if per_image_pros else 0.0


# ============================================================================
# Method 5: Simplified PRO (no integration, fixed threshold)
# ============================================================================

def compute_pro_simplified(predictions: list, masks: list, threshold_percentile: float = 50):
    """Compute PRO at a single threshold (simplified version)."""
    all_preds = np.concatenate([p.flatten() for p in predictions])
    threshold = np.percentile(all_preds, threshold_percentile)

    region_overlaps = []
    for pred, mask in zip(predictions, masks):
        if mask.sum() == 0:
            continue

        pred_binary = (pred >= threshold).astype(int)
        labeled_mask, num_components = ndimage.label(mask)

        for comp_idx in range(1, num_components + 1):
            component_mask = (labeled_mask == comp_idx)
            component_size = component_mask.sum()
            overlap = np.sum(pred_binary[component_mask]) / component_size
            region_overlaps.append(overlap)

    return np.mean(region_overlaps) * 100 if region_overlaps else 0.0


# ============================================================================
# Collect predictions
# ============================================================================

def collect_predictions(model, dataset_class, root, categories, device, transform):
    """Collect all predictions and masks."""
    model.eval()

    mask_resize = Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST, antialias=False)
    mask_crop = CenterCrop((IMAGE_SIZE, IMAGE_SIZE))

    all_preds = []
    all_masks = []

    for cat in categories:
        dm = dataset_class(root=root, category=cat, eval_batch_size=1, num_workers=4)
        dm.setup()

        for batch in dm.test_dataloader():
            gt_mask = batch.gt_mask[0] if batch.gt_mask is not None else None
            if gt_mask is None or gt_mask.sum() == 0:
                continue

            images = transform(batch.image).to(device)

            with torch.no_grad():
                output = model.model(images)

            pred_map = output.anomaly_map[0].cpu().numpy()

            if gt_mask.ndim == 2:
                gt_mask = gt_mask.unsqueeze(0)
            gt_mask_processed = mask_crop(mask_resize(gt_mask)).squeeze(0).numpy()
            gt_binary = (gt_mask_processed > 0.5).astype(int)

            all_preds.append(pred_map)
            all_masks.append(gt_binary)

    logger.info(f"Collected {len(all_preds)} defect samples")
    return all_preds, all_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="visa", choices=["mvtec", "visa"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fpr_limit", type=float, default=0.3)
    args = parser.parse_args()

    if args.dataset == "visa":
        CHECKPOINT = "examples/notebooks/11_fe_clip_variant/results/feclip_visa_seed42_20260101_061222/checkpoints/best_model.pt"
        dataset_class = Visa
        root = VISA_ROOT
        categories = VISA_CATEGORIES
    else:
        CHECKPOINT = "examples/notebooks/11_fe_clip_variant/results/feclip_mvtec_seed42_20260101_061222/checkpoints/best_model.pt"
        dataset_class = MVTecAD
        root = MVTEC_ROOT
        categories = MVTEC_CATEGORIES

    logger.info("=" * 80)
    logger.info(f"Experiment 9: PRO Metric Implementation Comparison ({args.dataset})")
    logger.info("=" * 80)
    logger.info(f"FPR limit: {args.fpr_limit}")

    # Load model
    logger.info(f"\nLoading model from {CHECKPOINT}...")
    model = FEClip(tap_indices=[20, 21, 22, 23])
    model.to(args.device)
    checkpoint = torch.load(CHECKPOINT, map_location=args.device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    transform = model.pre_processor.transform

    # Collect predictions
    logger.info(f"\nCollecting predictions from {args.dataset}...")
    predictions, masks = collect_predictions(model, dataset_class, root, categories, args.device, transform)

    # Compare different PRO implementations
    logger.info("\n" + "=" * 80)
    logger.info("Computing PRO with different implementations...")
    logger.info("=" * 80)

    results = {}

    # Method 1: Original (linspace)
    logger.info("\n[1] Original PRO (linspace thresholds, 200 steps)...")
    pro_original = compute_pro_original(predictions, masks, num_thresholds=200, fpr_limit=args.fpr_limit)
    results["original_linspace_200"] = pro_original
    logger.info(f"    PRO = {pro_original:.2f}%")

    # Method 1b: More thresholds
    logger.info("\n[1b] Original PRO (linspace, 500 steps)...")
    pro_original_500 = compute_pro_original(predictions, masks, num_thresholds=500, fpr_limit=args.fpr_limit)
    results["original_linspace_500"] = pro_original_500
    logger.info(f"    PRO = {pro_original_500:.2f}%")

    # Method 2: Quantile-based
    logger.info("\n[2] Quantile-based PRO (200 steps)...")
    pro_quantile = compute_pro_quantile(predictions, masks, num_thresholds=200, fpr_limit=args.fpr_limit)
    results["quantile_200"] = pro_quantile
    logger.info(f"    PRO = {pro_quantile:.2f}%")

    # Method 3: ROC-based
    logger.info("\n[3] ROC-based PRO...")
    pro_roc = compute_pro_roc_based(predictions, masks, fpr_limit=args.fpr_limit)
    results["roc_based"] = pro_roc
    logger.info(f"    PRO = {pro_roc:.2f}%")

    # Method 4: Per-image
    logger.info("\n[4] Per-image PRO (then average)...")
    pro_per_image = compute_pro_per_image(predictions, masks, num_thresholds=200, fpr_limit=args.fpr_limit)
    results["per_image"] = pro_per_image
    logger.info(f"    PRO = {pro_per_image:.2f}%")

    # Method 5: Simplified (single threshold)
    logger.info("\n[5] Simplified PRO (single threshold at p50)...")
    pro_simplified = compute_pro_simplified(predictions, masks, threshold_percentile=50)
    results["simplified_p50"] = pro_simplified
    logger.info(f"    PRO = {pro_simplified:.2f}%")

    # Different FPR limits
    logger.info("\n[6] FPR limit sensitivity (original method)...")
    for fpr in [0.1, 0.2, 0.3, 0.5]:
        pro_fpr = compute_pro_original(predictions, masks, num_thresholds=200, fpr_limit=fpr)
        results[f"fpr_{fpr}"] = pro_fpr
        logger.info(f"    fpr={fpr}: PRO = {pro_fpr:.2f}%")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info(f"SUMMARY: PRO Implementation Comparison ({args.dataset})")
    logger.info("=" * 80)
    logger.info(f"{'Method':<30} {'PRO':>10}")
    logger.info("-" * 45)

    for method, pro in results.items():
        if not method.startswith("fpr_"):
            logger.info(f"{method:<30} {pro:>9.2f}%")

    logger.info("\nFPR sensitivity:")
    for method, pro in results.items():
        if method.startswith("fpr_"):
            fpr = method.split("_")[1]
            logger.info(f"  fpr={fpr}: {pro:.2f}%")

    # Identify the range
    pro_values = [v for k, v in results.items() if not k.startswith("fpr_")]
    logger.info(f"\nPRO range: {min(pro_values):.2f}% ~ {max(pro_values):.2f}%")
    logger.info(f"Max difference: {max(pro_values) - min(pro_values):.2f}%")

    # Save results
    output_path = SCRIPT_DIR / f"exp9_pro_comparison_{args.dataset}_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
