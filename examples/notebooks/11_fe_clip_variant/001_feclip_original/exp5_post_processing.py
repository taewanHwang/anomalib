"""Experiment 5: Structural post-processing for PRO.

Hypothesis: The paper might use structural post-processing that helps PRO:
1. Percentile clipping (keep only top p%, zero out rest)
2. Morphological operations (opening/closing)
3. Small component removal

PRO is very sensitive to small FPs - post-processing can help significantly.
"""

import torch
import numpy as np
from pathlib import Path
from torch.nn import functional as F
from torchvision.transforms.v2 import Resize, CenterCrop, InterpolationMode
from scipy import ndimage
from skimage.morphology import opening, closing, disk

from anomalib.data import MVTecAD
from anomalib.data.datamodules.image.visa import Visa
from anomalib.models.image import FEClip
from anomalib.metrics.aupro import _AUPRO
from sklearn.metrics import roc_auc_score

import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_pixel_auroc(all_masks_pred: list, all_masks_gt: list) -> float:
    preds_flat = []
    gts_flat = []
    for pred, gt in zip(all_masks_pred, all_masks_gt):
        if gt is None:
            continue
        preds_flat.extend(pred.flatten())
        gts_flat.extend(gt.flatten())
    if len(preds_flat) == 0 or len(set(gts_flat)) < 2:
        return 0.0
    return roc_auc_score(gts_flat, preds_flat)


def compute_pro_score(all_masks_pred: list, all_masks_gt: list, device: str = "cuda") -> float:
    pro_metric = _AUPRO(fpr_limit=0.3)
    pro_metric = pro_metric.to(device)
    for pred, gt in zip(all_masks_pred, all_masks_gt):
        if gt is None:
            continue
        pred_tensor = torch.tensor(pred).float().to(device)
        gt_tensor = torch.tensor(gt).long().to(device)
        if gt_tensor.sum() == 0:
            continue
        # Normalize to [0, 1]
        p_min, p_max = pred_tensor.min(), pred_tensor.max()
        if p_max > p_min:
            pred_tensor = (pred_tensor - p_min) / (p_max - p_min)
        pro_metric.update(pred_tensor.unsqueeze(0), gt_tensor.unsqueeze(0))
    try:
        return pro_metric.compute().item()
    except:
        return 0.0


def post_process_none(amap: np.ndarray) -> np.ndarray:
    """No post-processing (baseline)."""
    return amap


def post_process_percentile_clip(amap: np.ndarray, percentile: float = 95) -> np.ndarray:
    """Keep only top percentile, zero out rest."""
    threshold = np.percentile(amap, percentile)
    result = amap.copy()
    result[result < threshold] = 0
    return result


def post_process_morphology_open(amap: np.ndarray, radius: int = 3) -> np.ndarray:
    """Apply morphological opening to remove small FPs."""
    # Threshold to binary
    threshold = np.percentile(amap, 90)
    binary = (amap > threshold).astype(np.float32)
    # Opening
    selem = disk(radius)
    opened = opening(binary, selem)
    # Apply mask to original
    return amap * opened


def post_process_morphology_close(amap: np.ndarray, radius: int = 3) -> np.ndarray:
    """Apply morphological closing to fill holes."""
    threshold = np.percentile(amap, 90)
    binary = (amap > threshold).astype(np.float32)
    selem = disk(radius)
    closed = closing(binary, selem)
    return amap * closed


def post_process_remove_small(amap: np.ndarray, min_size: int = 100) -> np.ndarray:
    """Remove small connected components."""
    threshold = np.percentile(amap, 90)
    binary = (amap > threshold).astype(np.int32)
    labeled, num_features = ndimage.label(binary)

    # Remove small components
    result = amap.copy()
    for i in range(1, num_features + 1):
        component_mask = labeled == i
        if component_mask.sum() < min_size:
            result[component_mask] = 0

    return result


def post_process_gaussian(amap: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Apply Gaussian smoothing."""
    return ndimage.gaussian_filter(amap, sigma=sigma)


def post_process_combined(amap: np.ndarray) -> np.ndarray:
    """Combined: percentile + remove small + gaussian."""
    # Step 1: Percentile clip
    threshold = np.percentile(amap, 90)
    result = amap.copy()
    result[result < threshold] = 0

    # Step 2: Remove small components
    binary = (result > 0).astype(np.int32)
    labeled, num_features = ndimage.label(binary)
    for i in range(1, num_features + 1):
        component_mask = labeled == i
        if component_mask.sum() < 50:
            result[component_mask] = 0

    # Step 3: Light gaussian
    result = ndimage.gaussian_filter(result, sigma=1.0)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="mvtec", choices=["mvtec", "visa"])
    args = parser.parse_args()

    DEVICE = args.device
    DATASET = args.dataset

    if DATASET == "mvtec":
        CATEGORIES = ["bottle", "cable", "capsule", "carpet", "hazelnut", "metal_nut",
                      "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper", "grid", "leather"]
        CHECKPOINT = "examples/notebooks/11_fe_clip_variant/results/feclip_mvtec_seed42_20260101_061222/checkpoints/best_model.pt"
    else:
        CATEGORIES = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
                      "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
        CHECKPOINT = "examples/notebooks/11_fe_clip_variant/results/feclip_visa_seed42_20260101_061222/checkpoints/best_model.pt"

    logger.info(f"Experiment 5: Post-processing for PRO")
    logger.info(f"Dataset: {DATASET}, Device: {DEVICE}")
    logger.info("=" * 80)

    # Load model
    model = FEClip(tap_indices=[20, 21, 22, 23])
    model.to(DEVICE)
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    transform = model.pre_processor.transform
    logger.info(f"Loaded: {CHECKPOINT}")

    # GT preprocessing
    mask_resize = Resize(336, interpolation=InterpolationMode.NEAREST, antialias=False)
    mask_crop = CenterCrop((336, 336))

    def preprocess_gt_mask(mask_tensor):
        if mask_tensor is None:
            return None
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        return mask_crop(mask_resize(mask_tensor)).squeeze(0)

    # Post-processing methods to test
    post_procs = {
        "none": post_process_none,
        "pct95": lambda x: post_process_percentile_clip(x, 95),
        "pct90": lambda x: post_process_percentile_clip(x, 90),
        "morph_open": post_process_morphology_open,
        "rm_small": post_process_remove_small,
        "gaussian": post_process_gaussian,
        "combined": post_process_combined,
    }

    all_results = {name: {"pauroc": [], "pro": []} for name in post_procs}

    for cat in CATEGORIES:
        logger.info(f"\nProcessing {cat}...")

        if DATASET == "mvtec":
            dm = MVTecAD(
                root="/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/MVTecAD",
                category=cat,
                eval_batch_size=1
            )
        else:
            dm = Visa(
                root="/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/VisA",
                category=cat,
                eval_batch_size=1
            )
        dm.setup()

        # Collect raw anomaly maps
        raw_maps = []
        all_gts = []

        for batch in dm.test_dataloader():
            images = transform(batch.image).to(DEVICE)

            with torch.no_grad():
                output = model.model(images)

            raw_maps.append(output.anomaly_map[0].cpu().numpy())

            if batch.gt_mask is not None:
                gt = preprocess_gt_mask(batch.gt_mask[0])
                if gt is not None and gt.sum() > 0:
                    all_gts.append(gt.numpy())
                else:
                    all_gts.append(None)
            else:
                all_gts.append(None)

        n_with_mask = sum(1 for g in all_gts if g is not None)

        # Apply each post-processing and compute metrics
        for name, func in post_procs.items():
            processed_maps = [func(m) for m in raw_maps]
            pauroc = compute_pixel_auroc(processed_maps, all_gts)
            pro = compute_pro_score(processed_maps, all_gts, DEVICE) if n_with_mask > 0 else 0.0

            all_results[name]["pauroc"].append(pauroc)
            all_results[name]["pro"].append(pro)

        # Log for this category
        logger.info(f"  none: PRO={all_results['none']['pro'][-1]*100:.1f}%")
        for name in list(post_procs.keys())[1:]:
            diff = all_results[name]["pro"][-1] - all_results["none"]["pro"][-1]
            logger.info(f"  {name}: PRO={all_results[name]['pro'][-1]*100:.1f}% ({diff*100:+.1f}%)")

    # Summary
    logger.info("\n" + "=" * 100)
    logger.info("SUMMARY: Post-processing comparison")
    logger.info("=" * 100)

    header = f"{'Category':<15}" + "".join(f"{name:>12}" for name in post_procs)
    logger.info("PRO results:")
    logger.info(header)
    logger.info("-" * 100)

    for i, cat in enumerate(CATEGORIES):
        row = f"{cat:<15}"
        for name in post_procs:
            row += f"{all_results[name]['pro'][i]*100:>11.1f}%"
        logger.info(row)

    logger.info("-" * 100)
    mean_row = f"{'MEAN':<15}"
    for name in post_procs:
        mean_pro = np.mean(all_results[name]["pro"])
        mean_row += f"{mean_pro*100:>11.1f}%"
    logger.info(mean_row)

    # pAUROC comparison
    logger.info("\npAUROC results:")
    pauroc_row = f"{'MEAN':<15}"
    for name in post_procs:
        mean_pauroc = np.mean(all_results[name]["pauroc"])
        pauroc_row += f"{mean_pauroc*100:>11.1f}%"
    logger.info(pauroc_row)
    logger.info("=" * 100)

    # Best method
    mean_pros = {name: np.mean(all_results[name]["pro"]) for name in post_procs}
    best_method = max(mean_pros, key=mean_pros.get)
    baseline_pro = mean_pros["none"]
    best_pro = mean_pros[best_method]

    logger.info(f"\nBest method for PRO: {best_method}")
    logger.info(f"  none (current): PRO = {baseline_pro*100:.1f}%")
    logger.info(f"  {best_method}: PRO = {best_pro*100:.1f}%")
    logger.info(f"  Improvement: {(best_pro - baseline_pro)*100:+.1f}%")


if __name__ == "__main__":
    main()
