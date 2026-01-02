"""Experiment 3: PRO-optimal tap configurations.

Hypothesis: The tap configuration [20,21,22,23] was optimized for image-level AUROC.
For PRO, different configurations might be better:
1. [20,21,22] - exclude block 23 which hurts PRO
2. [21] alone - best individual tap for PRO
3. Weighted average with block 23 downweighted

This experiment uses the TRAINED model but evaluates with different tap combinations at inference.
"""

import torch
import numpy as np
from pathlib import Path
from torch.nn import functional as F
from torchvision.transforms.v2 import Resize, CenterCrop, InterpolationMode

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
        if pred_tensor.max() > 1.0:
            pred_tensor = pred_tensor / pred_tensor.max()
        pro_metric.update(pred_tensor.unsqueeze(0), gt_tensor.unsqueeze(0))
    try:
        return pro_metric.compute().item()
    except:
        return 0.0


def combine_maps(map_list: list, method: str) -> torch.Tensor:
    """
    Combine anomaly maps from different taps.

    Args:
        map_list: List of 4 tensors, each (B, H, W) from taps [20,21,22,23]
        method: Combination method

    Returns:
        Combined map (B, H, W)
    """
    if method == "avg_all":
        # Average all 4 taps (current implementation)
        return torch.stack(map_list, dim=0).mean(dim=0)

    elif method == "avg_first3":
        # Exclude tap 3 (block 23) which hurts PRO
        return torch.stack(map_list[:3], dim=0).mean(dim=0)

    elif method == "tap1_only":
        # Use only tap 1 (block 21) - best for PRO
        return map_list[1]

    elif method == "avg_tap01":
        # Average tap 0 and 1 (blocks 20, 21)
        return torch.stack(map_list[:2], dim=0).mean(dim=0)

    elif method == "avg_tap12":
        # Average tap 1 and 2 (blocks 21, 22)
        return torch.stack(map_list[1:3], dim=0).mean(dim=0)

    elif method == "weighted_down23":
        # Weighted average with block 23 downweighted (0.5)
        weights = torch.tensor([1.0, 1.0, 1.0, 0.5])
        weights = weights / weights.sum()
        stacked = torch.stack(map_list, dim=0)  # (4, B, H, W)
        weighted = stacked * weights.view(-1, 1, 1, 1).to(stacked.device)
        return weighted.sum(dim=0)

    elif method == "max_all":
        # Max across all taps
        return torch.stack(map_list, dim=0).max(dim=0)[0]

    else:
        raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:2")
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

    logger.info(f"Experiment 3: PRO-optimal Tap Configurations")
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

    # Combination methods to test
    methods = ["avg_all", "avg_first3", "tap1_only", "avg_tap01", "avg_tap12", "weighted_down23", "max_all"]

    all_results = {m: {"pauroc": [], "pro": []} for m in methods}

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

        # Collect per-tap maps
        all_map_lists = []  # List of map_list per sample
        all_gts = []

        for batch in dm.test_dataloader():
            images = transform(batch.image).to(DEVICE)

            with torch.no_grad():
                scores, map_list = model.model.forward_tokens(images)

            # Upsample each tap map to 336x336
            upsampled_maps = []
            for tap_map in map_list:
                upsampled = F.interpolate(
                    tap_map.unsqueeze(1),
                    size=(336, 336),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                upsampled_maps.append(upsampled.cpu())

            all_map_lists.append(upsampled_maps)

            if batch.gt_mask is not None:
                gt = preprocess_gt_mask(batch.gt_mask[0])
                if gt is not None and gt.sum() > 0:
                    all_gts.append(gt.numpy())
                else:
                    all_gts.append(None)
            else:
                all_gts.append(None)

        n_with_mask = sum(1 for g in all_gts if g is not None)

        # Test each combination method
        for method in methods:
            all_preds = []
            for map_list in all_map_lists:
                combined = combine_maps(map_list, method)
                all_preds.append(combined[0].numpy())

            pauroc = compute_pixel_auroc(all_preds, all_gts)
            pro = compute_pro_score(all_preds, all_gts, DEVICE) if n_with_mask > 0 else 0.0

            all_results[method]["pauroc"].append(pauroc)
            all_results[method]["pro"].append(pro)

    # Summary
    logger.info("\n" + "=" * 100)
    logger.info("SUMMARY: PRO by tap combination method")
    logger.info("=" * 100)

    header = f"{'Category':<15}" + "".join(f"{m:>12}" for m in methods)
    logger.info(header)
    logger.info("-" * 100)

    for i, cat in enumerate(CATEGORIES):
        row = f"{cat:<15}"
        for method in methods:
            row += f"{all_results[method]['pro'][i]*100:>11.1f}%"
        logger.info(row)

    logger.info("-" * 100)
    mean_row = f"{'MEAN':<15}"
    for method in methods:
        mean_pro = np.mean(all_results[method]["pro"])
        mean_row += f"{mean_pro*100:>11.1f}%"
    logger.info(mean_row)

    # Also show pAUROC
    logger.info("\npAUROC comparison:")
    pauroc_row = f"{'Mean pAUROC':<15}"
    for method in methods:
        mean_pauroc = np.mean(all_results[method]["pauroc"])
        pauroc_row += f"{mean_pauroc*100:>11.1f}%"
    logger.info(pauroc_row)
    logger.info("=" * 100)

    # Best method
    mean_pros = {m: np.mean(all_results[m]["pro"]) for m in methods}
    best_method = max(mean_pros, key=mean_pros.get)
    baseline_pro = mean_pros["avg_all"]
    best_pro = mean_pros[best_method]

    logger.info(f"\nBest method for PRO: {best_method}")
    logger.info(f"  avg_all (current): PRO = {baseline_pro*100:.1f}%")
    logger.info(f"  {best_method}: PRO = {best_pro*100:.1f}%")
    logger.info(f"  Improvement: {(best_pro - baseline_pro)*100:+.1f}%")


if __name__ == "__main__":
    main()
