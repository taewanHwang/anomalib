"""Experiment 1: Original resolution evaluation vs 336 crop evaluation.

Hypothesis: The paper might evaluate segmentation metrics on original image coordinates,
not the 336x336 CLIP-cropped coordinates. This could explain the large PRO gap on VisA/BTAD.

Test:
1. Evaluate on 336 crop (current implementation)
2. Evaluate by resizing anomaly map back to original size and using original GT mask
"""

import torch
import numpy as np
from pathlib import Path
from torch.nn import functional as F
from torchvision.transforms.v2 import Resize, CenterCrop, InterpolationMode
from PIL import Image

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
    valid_count = 0
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
        valid_count += 1
    try:
        return pro_metric.compute().item(), valid_count
    except:
        return 0.0, valid_count


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

    logger.info(f"Experiment 1: Original Resolution Evaluation")
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

    # CLIP-style GT preprocessing (for 336 crop evaluation)
    mask_resize_336 = Resize(336, interpolation=InterpolationMode.NEAREST, antialias=False)
    mask_crop_336 = CenterCrop((336, 336))

    def preprocess_gt_mask_336(mask_tensor):
        if mask_tensor is None:
            return None
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        return mask_crop_336(mask_resize_336(mask_tensor)).squeeze(0)

    results_336 = {"pauroc": [], "pro": []}
    results_orig = {"pauroc": [], "pro": []}

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

        preds_336 = []
        gts_336 = []
        preds_orig = []
        gts_orig = []

        for batch in dm.test_dataloader():
            images = transform(batch.image).to(DEVICE)

            with torch.no_grad():
                output = model.model(images)

            # Anomaly map (336x336)
            amap_336 = output.anomaly_map[0].cpu()

            # GT mask processing
            if batch.gt_mask is not None and batch.gt_mask[0] is not None:
                gt_orig = batch.gt_mask[0]  # Original size
                if gt_orig.sum() > 0:
                    # Method 1: 336 crop evaluation (current)
                    gt_336 = preprocess_gt_mask_336(gt_orig)
                    preds_336.append(amap_336.numpy())
                    gts_336.append(gt_336.numpy())

                    # Method 2: Original resolution evaluation
                    # Resize anomaly map back to original GT size
                    orig_h, orig_w = gt_orig.shape[-2:]
                    amap_orig = F.interpolate(
                        amap_336.unsqueeze(0).unsqueeze(0),
                        size=(orig_h, orig_w),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0).squeeze(0)

                    gt_orig_np = gt_orig.squeeze(0).numpy() if gt_orig.ndim == 3 else gt_orig.numpy()
                    preds_orig.append(amap_orig.numpy())
                    gts_orig.append(gt_orig_np)
                else:
                    preds_336.append(amap_336.numpy())
                    gts_336.append(None)
                    preds_orig.append(None)
                    gts_orig.append(None)
            else:
                preds_336.append(amap_336.numpy())
                gts_336.append(None)
                preds_orig.append(None)
                gts_orig.append(None)

        # Compute metrics
        pauroc_336 = compute_pixel_auroc(preds_336, gts_336)
        pro_336, n_336 = compute_pro_score(preds_336, gts_336, DEVICE)

        # Filter None for original resolution
        preds_orig_valid = [p for p, g in zip(preds_orig, gts_orig) if p is not None and g is not None]
        gts_orig_valid = [g for p, g in zip(preds_orig, gts_orig) if p is not None and g is not None]
        pauroc_orig = compute_pixel_auroc(preds_orig_valid, gts_orig_valid)
        pro_orig, n_orig = compute_pro_score(preds_orig_valid, gts_orig_valid, DEVICE)

        results_336["pauroc"].append(pauroc_336)
        results_336["pro"].append(pro_336)
        results_orig["pauroc"].append(pauroc_orig)
        results_orig["pro"].append(pro_orig)

        logger.info(f"  336 crop:  pAUROC={pauroc_336*100:.1f}%, PRO={pro_336*100:.1f}% (n={n_336})")
        logger.info(f"  Original:  pAUROC={pauroc_orig*100:.1f}%, PRO={pro_orig*100:.1f}% (n={n_orig})")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Method':<15} {'Mean pAUROC':>15} {'Mean PRO':>15}")
    logger.info("-" * 80)

    mean_pauroc_336 = np.mean(results_336["pauroc"])
    mean_pro_336 = np.mean(results_336["pro"])
    mean_pauroc_orig = np.mean(results_orig["pauroc"])
    mean_pro_orig = np.mean(results_orig["pro"])

    logger.info(f"{'336 crop':<15} {mean_pauroc_336*100:>14.1f}% {mean_pro_336*100:>14.1f}%")
    logger.info(f"{'Original':<15} {mean_pauroc_orig*100:>14.1f}% {mean_pro_orig*100:>14.1f}%")
    logger.info("-" * 80)
    logger.info(f"{'Difference':<15} {(mean_pauroc_orig-mean_pauroc_336)*100:>+14.1f}% {(mean_pro_orig-mean_pro_336)*100:>+14.1f}%")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
