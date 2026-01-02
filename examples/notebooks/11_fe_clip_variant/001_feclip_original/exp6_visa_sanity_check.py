"""Experiment 6: VisA GT mask / evaluation pipeline sanity check.

Priority 0: Before assuming "model problem", verify GT mask/pipeline issues.

Checks:
(A) GT mask alignment via bbox IoU - do predicted anomaly regions overlap with GT?
(B) Mask inversion/label sanity - correct values, no object mask confusion
(C) Connected component analysis - is prediction "salt-and-pepper" fragmented?

The suspicious pattern:
- Image-level: VisA +3.1% AUROC (BETTER than paper)
- Pixel-level PRO: VisA -14% (MUCH WORSE than paper)
This suggests pipeline/definition issue, not model capability issue.
"""

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from torch.nn import functional as F
from torchvision.transforms.v2 import Resize, CenterCrop, InterpolationMode
from scipy import ndimage
from skimage.measure import label, regionprops
import cv2

from anomalib.data.datamodules.image.visa import Visa
from anomalib.models.image import FEClip

import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_bbox_iou(mask1, mask2):
    """Compute IoU between bounding boxes of two binary masks."""
    def get_bbox(mask):
        if mask.sum() == 0:
            return None
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    bbox1 = get_bbox(mask1)
    bbox2 = get_bbox(mask2)

    if bbox1 is None or bbox2 is None:
        return 0.0

    r1_min, r1_max, c1_min, c1_max = bbox1
    r2_min, r2_max, c2_min, c2_max = bbox2

    # Intersection
    ri_min = max(r1_min, r2_min)
    ri_max = min(r1_max, r2_max)
    ci_min = max(c1_min, c2_min)
    ci_max = min(c1_max, c2_max)

    if ri_max < ri_min or ci_max < ci_min:
        return 0.0

    inter_area = (ri_max - ri_min + 1) * (ci_max - ci_min + 1)
    area1 = (r1_max - r1_min + 1) * (c1_max - c1_min + 1)
    area2 = (r2_max - r2_min + 1) * (c2_max - c2_min + 1)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def compute_pixel_iou(mask1, mask2):
    """Compute pixel-level IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def analyze_connected_components(mask):
    """Analyze connected components in a binary mask."""
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return {"count": 0, "sizes": [], "mean_size": 0, "max_size": 0}

    sizes = []
    for i in range(1, num_features + 1):
        sizes.append((labeled == i).sum())

    return {
        "count": num_features,
        "sizes": sizes,
        "mean_size": np.mean(sizes),
        "max_size": max(sizes)
    }


def check_A_bbox_alignment(model, dm, transform, mask_resize, mask_crop, device, cat_name):
    """Check (A): GT mask alignment via bbox IoU."""
    logger.info(f"\n=== Check A: Bbox IoU Alignment for {cat_name} ===")

    bbox_ious = []
    pixel_ious = []

    for batch in dm.test_dataloader():
        if batch.gt_mask is None:
            continue

        gt_mask = batch.gt_mask[0]
        if gt_mask.sum() == 0:
            continue

        # Preprocess GT mask same as model
        if gt_mask.ndim == 2:
            gt_mask = gt_mask.unsqueeze(0)
        gt_mask_processed = mask_crop(mask_resize(gt_mask)).squeeze(0).numpy()
        gt_binary = (gt_mask_processed > 0.5).astype(np.uint8)

        # Get prediction
        images = transform(batch.image).to(device)
        with torch.no_grad():
            output = model.model(images)
        pred_map = output.anomaly_map[0].cpu().numpy()

        # Binarize prediction at top 1% threshold (Otsu alternative)
        threshold = np.percentile(pred_map, 99)
        pred_binary = (pred_map > threshold).astype(np.uint8)

        bbox_iou = compute_bbox_iou(gt_binary, pred_binary)
        pixel_iou = compute_pixel_iou(gt_binary, pred_binary)

        bbox_ious.append(bbox_iou)
        pixel_ious.append(pixel_iou)

    if len(bbox_ious) == 0:
        logger.info("  No defect samples found!")
        return {}

    results = {
        "bbox_iou_mean": np.mean(bbox_ious),
        "bbox_iou_std": np.std(bbox_ious),
        "bbox_iou_min": np.min(bbox_ious),
        "bbox_iou_max": np.max(bbox_ious),
        "bbox_iou_zero_pct": (np.array(bbox_ious) == 0).sum() / len(bbox_ious) * 100,
        "pixel_iou_mean": np.mean(pixel_ious),
        "n_samples": len(bbox_ious)
    }

    logger.info(f"  Bbox IoU: mean={results['bbox_iou_mean']:.3f}, std={results['bbox_iou_std']:.3f}")
    logger.info(f"  Bbox IoU: min={results['bbox_iou_min']:.3f}, max={results['bbox_iou_max']:.3f}")
    logger.info(f"  Bbox IoU=0 (no overlap): {results['bbox_iou_zero_pct']:.1f}% of samples")
    logger.info(f"  Pixel IoU: mean={results['pixel_iou_mean']:.3f}")

    if results['bbox_iou_zero_pct'] > 30:
        logger.warning(f"  ⚠️ WARNING: >30% samples have ZERO bbox overlap - alignment issue likely!")
    if results['bbox_iou_mean'] < 0.3:
        logger.warning(f"  ⚠️ WARNING: Mean bbox IoU < 0.3 - significant misalignment!")

    return results


def check_B_mask_sanity(dm, mask_resize, mask_crop, cat_name):
    """Check (B): Mask inversion/label sanity."""
    logger.info(f"\n=== Check B: Mask Value/Label Sanity for {cat_name} ===")

    normal_mask_sums = []
    abnormal_mask_sums = []
    mask_unique_values = set()
    mask_dtypes = set()
    mask_shapes = []

    for batch in dm.test_dataloader():
        label = batch.gt_label[0].item() if hasattr(batch, 'gt_label') else None

        if batch.gt_mask is not None:
            gt_mask = batch.gt_mask[0]
            if gt_mask.ndim == 2:
                gt_mask = gt_mask.unsqueeze(0)
            gt_mask_processed = mask_crop(mask_resize(gt_mask)).squeeze(0)

            mask_unique_values.update(gt_mask_processed.unique().tolist())
            mask_dtypes.add(str(gt_mask_processed.dtype))
            mask_shapes.append(tuple(gt_mask_processed.shape))
            mask_sum = gt_mask_processed.sum().item()

            if label == 0:  # Normal
                normal_mask_sums.append(mask_sum)
            else:  # Abnormal
                abnormal_mask_sums.append(mask_sum)
        else:
            if label == 0:
                normal_mask_sums.append(0)

    results = {
        "mask_unique_values": sorted(list(mask_unique_values)),
        "mask_dtypes": list(mask_dtypes),
        "mask_shapes": list(set(mask_shapes)),
        "normal_mask_nonzero": sum(1 for s in normal_mask_sums if s > 0),
        "normal_mask_total": len(normal_mask_sums),
        "abnormal_mask_mean_sum": np.mean(abnormal_mask_sums) if abnormal_mask_sums else 0,
        "abnormal_mask_zero": sum(1 for s in abnormal_mask_sums if s == 0),
        "abnormal_mask_total": len(abnormal_mask_sums)
    }

    logger.info(f"  Mask unique values: {results['mask_unique_values']}")
    logger.info(f"  Mask dtypes: {results['mask_dtypes']}")
    logger.info(f"  Mask shapes: {results['mask_shapes']}")
    logger.info(f"  Normal samples with nonzero mask: {results['normal_mask_nonzero']}/{results['normal_mask_total']}")
    logger.info(f"  Abnormal samples with zero mask: {results['abnormal_mask_zero']}/{results['abnormal_mask_total']}")
    logger.info(f"  Abnormal mask mean sum: {results['abnormal_mask_mean_sum']:.1f} pixels")

    # Sanity checks
    if results['normal_mask_nonzero'] > 0:
        logger.warning(f"  ⚠️ WARNING: {results['normal_mask_nonzero']} normal samples have NON-ZERO masks!")
        logger.warning(f"      This is CRITICAL - might be object mask vs defect mask confusion!")

    if results['abnormal_mask_zero'] > 0:
        pct = results['abnormal_mask_zero'] / results['abnormal_mask_total'] * 100
        logger.warning(f"  ⚠️ WARNING: {pct:.1f}% of abnormal samples have ZERO masks!")

    # Check for inverted masks (values close to 255 instead of 1)
    if any(v > 1.5 for v in results['mask_unique_values']):
        logger.info(f"  Note: Mask values > 1 detected (likely 0-255 range, not 0-1)")

    return results


def check_C_component_analysis(model, dm, transform, mask_resize, mask_crop, device, cat_name):
    """Check (C): Connected component analysis - salt-and-pepper fragmentation."""
    logger.info(f"\n=== Check C: Connected Component Analysis for {cat_name} ===")

    gt_component_counts = []
    pred_component_counts = []
    pred_to_gt_count_ratios = []
    pred_mean_sizes = []
    gt_mean_sizes = []

    thresholds_to_test = [90, 95, 99]  # percentiles
    results_by_threshold = {}

    for pct in thresholds_to_test:
        pred_counts_at_pct = []
        pred_sizes_at_pct = []

        for batch in dm.test_dataloader():
            if batch.gt_mask is None:
                continue

            gt_mask = batch.gt_mask[0]
            if gt_mask.sum() == 0:
                continue

            # Preprocess GT
            if gt_mask.ndim == 2:
                gt_mask = gt_mask.unsqueeze(0)
            gt_mask_processed = mask_crop(mask_resize(gt_mask)).squeeze(0).numpy()
            gt_binary = (gt_mask_processed > 0.5).astype(np.uint8)
            gt_stats = analyze_connected_components(gt_binary)

            # Get prediction
            images = transform(batch.image).to(device)
            with torch.no_grad():
                output = model.model(images)
            pred_map = output.anomaly_map[0].cpu().numpy()

            # Binarize at percentile
            threshold = np.percentile(pred_map, pct)
            pred_binary = (pred_map > threshold).astype(np.uint8)
            pred_stats = analyze_connected_components(pred_binary)

            pred_counts_at_pct.append(pred_stats["count"])
            if pred_stats["count"] > 0:
                pred_sizes_at_pct.append(pred_stats["mean_size"])

            if pct == 95:  # Primary threshold
                gt_component_counts.append(gt_stats["count"])
                pred_component_counts.append(pred_stats["count"])
                if gt_stats["count"] > 0:
                    gt_mean_sizes.append(gt_stats["mean_size"])
                if pred_stats["count"] > 0:
                    pred_mean_sizes.append(pred_stats["mean_size"])
                if gt_stats["count"] > 0:
                    pred_to_gt_count_ratios.append(pred_stats["count"] / gt_stats["count"])

        results_by_threshold[pct] = {
            "mean_component_count": np.mean(pred_counts_at_pct) if pred_counts_at_pct else 0,
            "mean_component_size": np.mean(pred_sizes_at_pct) if pred_sizes_at_pct else 0
        }

    results = {
        "gt_mean_component_count": np.mean(gt_component_counts) if gt_component_counts else 0,
        "pred_mean_component_count_p95": np.mean(pred_component_counts) if pred_component_counts else 0,
        "pred_to_gt_count_ratio": np.mean(pred_to_gt_count_ratios) if pred_to_gt_count_ratios else 0,
        "gt_mean_component_size": np.mean(gt_mean_sizes) if gt_mean_sizes else 0,
        "pred_mean_component_size": np.mean(pred_mean_sizes) if pred_mean_sizes else 0,
        "by_threshold": results_by_threshold
    }

    logger.info(f"  GT mean component count: {results['gt_mean_component_count']:.1f}")
    logger.info(f"  Pred mean component count (@p95): {results['pred_mean_component_count_p95']:.1f}")
    logger.info(f"  Pred/GT count ratio: {results['pred_to_gt_count_ratio']:.2f}x")
    logger.info(f"  GT mean component size: {results['gt_mean_component_size']:.1f} pixels")
    logger.info(f"  Pred mean component size: {results['pred_mean_component_size']:.1f} pixels")

    logger.info(f"\n  By threshold:")
    for pct, stats in results_by_threshold.items():
        logger.info(f"    @p{pct}: count={stats['mean_component_count']:.1f}, size={stats['mean_component_size']:.1f}")

    if results['pred_to_gt_count_ratio'] > 5:
        logger.warning(f"  ⚠️ WARNING: Pred has {results['pred_to_gt_count_ratio']:.1f}x more components than GT!")
        logger.warning(f"      This indicates 'salt-and-pepper' fragmentation - hurts PRO!")

    if results['pred_mean_component_size'] < results['gt_mean_component_size'] * 0.2:
        logger.warning(f"  ⚠️ WARNING: Pred components are very small vs GT - fragmentation issue!")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--categories", type=str, default="all",
                        help="Comma-separated categories or 'all'")
    args = parser.parse_args()

    DEVICE = args.device
    CHECKPOINT = "examples/notebooks/11_fe_clip_variant/results/feclip_visa_seed42_20260101_061222/checkpoints/best_model.pt"

    ALL_CATEGORIES = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
                      "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]

    if args.categories == "all":
        CATEGORIES = ALL_CATEGORIES
    else:
        CATEGORIES = [c.strip() for c in args.categories.split(",")]

    logger.info("=" * 80)
    logger.info("Experiment 6: VisA GT Mask / Pipeline Sanity Check")
    logger.info("=" * 80)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Categories: {CATEGORIES}")

    # Load model
    logger.info("\nLoading FEClip model...")
    model = FEClip(tap_indices=[20, 21, 22, 23])
    model.to(DEVICE)
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    transform = model.pre_processor.transform
    logger.info(f"Loaded: {CHECKPOINT}")

    # GT preprocessing (same as evaluation)
    mask_resize = Resize(336, interpolation=InterpolationMode.NEAREST, antialias=False)
    mask_crop = CenterCrop((336, 336))

    all_results = {"A": {}, "B": {}, "C": {}}

    for cat in CATEGORIES:
        logger.info(f"\n{'='*60}")
        logger.info(f"Category: {cat}")
        logger.info(f"{'='*60}")

        dm = Visa(
            root="/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/VisA",
            category=cat,
            eval_batch_size=1
        )
        dm.setup()

        # Run all checks
        all_results["A"][cat] = check_A_bbox_alignment(model, dm, transform, mask_resize, mask_crop, DEVICE, cat)
        all_results["B"][cat] = check_B_mask_sanity(dm, mask_resize, mask_crop, cat)
        all_results["C"][cat] = check_C_component_analysis(model, dm, transform, mask_resize, mask_crop, DEVICE, cat)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: VisA Pipeline Sanity Check")
    logger.info("=" * 80)

    # Check A summary
    logger.info("\n[A] Bbox IoU Alignment:")
    logger.info(f"{'Category':<15} {'Mean IoU':>10} {'Zero%':>10} {'Status':>15}")
    logger.info("-" * 50)
    for cat in CATEGORIES:
        if cat in all_results["A"] and all_results["A"][cat]:
            r = all_results["A"][cat]
            status = "⚠️ ISSUE" if r['bbox_iou_zero_pct'] > 30 or r['bbox_iou_mean'] < 0.3 else "✓ OK"
            logger.info(f"{cat:<15} {r['bbox_iou_mean']:>10.3f} {r['bbox_iou_zero_pct']:>9.1f}% {status:>15}")

    # Check B summary
    logger.info("\n[B] Mask Value Sanity:")
    logger.info(f"{'Category':<15} {'Normal w/ mask':>15} {'Abnormal w/o mask':>18}")
    logger.info("-" * 50)
    for cat in CATEGORIES:
        if cat in all_results["B"]:
            r = all_results["B"][cat]
            normal_issue = f"{r['normal_mask_nonzero']}/{r['normal_mask_total']}"
            abnormal_issue = f"{r['abnormal_mask_zero']}/{r['abnormal_mask_total']}"
            logger.info(f"{cat:<15} {normal_issue:>15} {abnormal_issue:>18}")

    # Check C summary
    logger.info("\n[C] Component Fragmentation:")
    logger.info(f"{'Category':<15} {'GT count':>10} {'Pred count':>12} {'Ratio':>10} {'Status':>12}")
    logger.info("-" * 60)
    for cat in CATEGORIES:
        if cat in all_results["C"] and all_results["C"][cat]:
            r = all_results["C"][cat]
            status = "⚠️ FRAG" if r['pred_to_gt_count_ratio'] > 5 else "✓ OK"
            logger.info(f"{cat:<15} {r['gt_mean_component_count']:>10.1f} {r['pred_mean_component_count_p95']:>12.1f} {r['pred_to_gt_count_ratio']:>9.2f}x {status:>12}")

    # Overall diagnosis
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSIS:")
    logger.info("=" * 80)

    # Count issues
    bbox_issues = sum(1 for cat in CATEGORIES if cat in all_results["A"] and all_results["A"][cat]
                      and (all_results["A"][cat]['bbox_iou_zero_pct'] > 30 or all_results["A"][cat]['bbox_iou_mean'] < 0.3))
    normal_mask_issues = sum(1 for cat in CATEGORIES if cat in all_results["B"]
                             and all_results["B"][cat]['normal_mask_nonzero'] > 0)
    frag_issues = sum(1 for cat in CATEGORIES if cat in all_results["C"] and all_results["C"][cat]
                      and all_results["C"][cat]['pred_to_gt_count_ratio'] > 5)

    if normal_mask_issues > 0:
        logger.warning(f"❌ CRITICAL: {normal_mask_issues} categories have normal samples with non-zero masks!")
        logger.warning("   This suggests 'object mask vs defect mask' confusion in VisA loader!")

    if bbox_issues > 0:
        logger.warning(f"⚠️ {bbox_issues} categories have bbox alignment issues")

    if frag_issues > 0:
        logger.warning(f"⚠️ {frag_issues} categories have component fragmentation issues")

    if normal_mask_issues == 0 and bbox_issues < 3 and frag_issues < 3:
        logger.info("✓ Pipeline appears mostly correct. PRO gap likely from model/training.")


if __name__ == "__main__":
    main()
