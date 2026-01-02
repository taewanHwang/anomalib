"""Experiment 2: PRO calculation settings comparison.

Hypothesis: The paper might use different PRO calculation settings:
1. fpr_limit (0.3 vs 0.1 vs 0.05)
2. Normal image inclusion (include vs exclude from PRO calculation)
3. Per-image normalization vs dataset-level normalization
4. Different threshold sweep methods
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


def compute_pro_with_settings(
    all_masks_pred: list,
    all_masks_gt: list,
    all_labels: list,
    device: str = "cuda",
    fpr_limit: float = 0.3,
    include_normal: bool = False,
    per_image_norm: bool = False,
    num_thresholds: int = None
) -> tuple:
    """
    Compute PRO with different settings.

    Args:
        all_masks_pred: List of predicted anomaly maps
        all_masks_gt: List of GT masks (None for normal images)
        all_labels: List of labels (0=normal, 1=anomaly)
        fpr_limit: FPR limit for PRO calculation
        include_normal: Whether to include normal images in PRO calculation
        per_image_norm: Whether to normalize each image's map to [0,1] independently
        num_thresholds: Number of thresholds for sweep (None = auto)
    """
    pro_metric = _AUPRO(fpr_limit=fpr_limit, num_thresholds=num_thresholds)
    pro_metric = pro_metric.to(device)

    valid_count = 0
    skipped_normal = 0
    skipped_empty = 0

    for pred, gt, label in zip(all_masks_pred, all_masks_gt, all_labels):
        # Skip normal images if configured
        if label == 0 and not include_normal:
            skipped_normal += 1
            continue

        if gt is None:
            skipped_empty += 1
            continue

        pred_tensor = torch.tensor(pred).float().to(device)
        gt_tensor = torch.tensor(gt).long().to(device)

        if gt_tensor.sum() == 0:
            skipped_empty += 1
            continue

        # Per-image normalization
        if per_image_norm:
            p_min, p_max = pred_tensor.min(), pred_tensor.max()
            if p_max > p_min:
                pred_tensor = (pred_tensor - p_min) / (p_max - p_min)
        elif pred_tensor.max() > 1.0:
            pred_tensor = pred_tensor / pred_tensor.max()

        pro_metric.update(pred_tensor.unsqueeze(0), gt_tensor.unsqueeze(0))
        valid_count += 1

    try:
        pro_value = pro_metric.compute().item()
    except Exception as e:
        logger.warning(f"PRO computation failed: {e}")
        pro_value = 0.0

    return pro_value, valid_count, skipped_normal, skipped_empty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:1")
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

    logger.info(f"Experiment 2: PRO Calculation Settings")
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

    # PRO settings to test
    settings = [
        {"name": "default", "fpr_limit": 0.3, "include_normal": False, "per_image_norm": False},
        {"name": "fpr=0.1", "fpr_limit": 0.1, "include_normal": False, "per_image_norm": False},
        {"name": "fpr=0.05", "fpr_limit": 0.05, "include_normal": False, "per_image_norm": False},
        {"name": "incl_normal", "fpr_limit": 0.3, "include_normal": True, "per_image_norm": False},
        {"name": "per_img_norm", "fpr_limit": 0.3, "include_normal": False, "per_image_norm": True},
        {"name": "per_img+incl", "fpr_limit": 0.3, "include_normal": True, "per_image_norm": True},
    ]

    all_results = {s["name"]: [] for s in settings}

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

        all_preds = []
        all_gts = []
        all_labels = []

        for batch in dm.test_dataloader():
            images = transform(batch.image).to(DEVICE)

            with torch.no_grad():
                output = model.model(images)

            amap = output.anomaly_map[0].cpu().numpy()
            all_preds.append(amap)

            label = batch.gt_label[0].item() if batch.gt_label is not None else 0
            all_labels.append(label)

            if batch.gt_mask is not None and batch.gt_mask[0] is not None:
                gt = preprocess_gt_mask(batch.gt_mask[0])
                if gt is not None and gt.sum() > 0:
                    all_gts.append(gt.numpy())
                else:
                    all_gts.append(None)
            else:
                all_gts.append(None)

        # Test each setting
        for setting in settings:
            pro, n_valid, n_skip_normal, n_skip_empty = compute_pro_with_settings(
                all_preds, all_gts, all_labels, DEVICE,
                fpr_limit=setting["fpr_limit"],
                include_normal=setting["include_normal"],
                per_image_norm=setting["per_image_norm"]
            )
            all_results[setting["name"]].append(pro)

            if setting["name"] == "default":
                logger.info(f"  {setting['name']}: PRO={pro*100:.1f}% (valid={n_valid}, skip_norm={n_skip_normal}, skip_empty={n_skip_empty})")

    # Summary
    logger.info("\n" + "=" * 100)
    logger.info("SUMMARY: PRO by different calculation settings")
    logger.info("=" * 100)

    header = f"{'Category':<15}" + "".join(f"{s['name']:>15}" for s in settings)
    logger.info(header)
    logger.info("-" * 100)

    for i, cat in enumerate(CATEGORIES):
        row = f"{cat:<15}"
        for setting in settings:
            row += f"{all_results[setting['name']][i]*100:>14.1f}%"
        logger.info(row)

    logger.info("-" * 100)
    mean_row = f"{'MEAN':<15}"
    for setting in settings:
        mean_pro = np.mean(all_results[setting["name"]])
        mean_row += f"{mean_pro*100:>14.1f}%"
    logger.info(mean_row)
    logger.info("=" * 100)

    # Comparison
    logger.info("\nDifference from default (fpr=0.3, exclude normal, no per-image norm):")
    default_mean = np.mean(all_results["default"])
    for setting in settings[1:]:
        diff = np.mean(all_results[setting["name"]]) - default_mean
        logger.info(f"  {setting['name']}: {diff*100:+.1f}%")


if __name__ == "__main__":
    main()
