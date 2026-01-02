"""Test different tap block selections for PRO.

The last4 [20,21,22,23] was optimized for image-level AUROC.
Earlier taps might preserve more spatial detail for PRO.
"""

import torch
import numpy as np
from pathlib import Path
from torch.nn import functional as F
from torchvision.transforms.v2 import Resize, CenterCrop, InterpolationMode

from anomalib.data import MVTecAD
from anomalib.models.image import FEClip
from anomalib.metrics.aupro import _AUPRO
from sklearn.metrics import roc_auc_score

DEVICE = "cuda:0"
CATEGORIES = ["bottle", "cable", "capsule", "carpet", "hazelnut", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper", "grid", "leather"]

# Different tap configurations
TAP_CONFIGS = {
    "last4": [20, 21, 22, 23],
    "linspace4": [0, 8, 15, 23],
    "mid4": [10, 14, 18, 22],
    "early4": [4, 8, 12, 16],
    "late8": [16, 17, 18, 19, 20, 21, 22, 23],
}


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


def main():
    print("Testing different tap configurations for PRO")
    print("=" * 80)

    # GT mask preprocessing
    mask_resize = Resize(336, interpolation=InterpolationMode.NEAREST, antialias=False)
    mask_crop = CenterCrop((336, 336))

    def preprocess_gt_mask(mask_tensor):
        if mask_tensor is None:
            return None
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        return mask_crop(mask_resize(mask_tensor)).squeeze(0)

    # Store results per tap config
    all_results = {tap_name: {"pauroc": [], "pro": []} for tap_name in TAP_CONFIGS}

    for tap_name, tap_indices in TAP_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Testing: {tap_name} = {tap_indices}")
        print("=" * 60)

        # Load model with this tap config
        model = FEClip(tap_indices=tap_indices)
        model.to(DEVICE)
        model.model.setup_text()  # Random init, just testing architecture effect
        model.eval()
        transform = model.pre_processor.transform

        for cat in CATEGORIES:
            dm = MVTecAD(
                root="/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/MVTecAD",
                category=cat,
                eval_batch_size=1
            )
            dm.setup()

            all_preds = []
            all_gts = []

            for batch in dm.test_dataloader():
                images = transform(batch.image).to(DEVICE)

                with torch.no_grad():
                    output = model.model(images)

                pred_map = output.anomaly_map[0].cpu().numpy()
                all_preds.append(pred_map)

                if batch.gt_mask is not None:
                    gt = preprocess_gt_mask(batch.gt_mask[0])
                    if gt is not None and gt.sum() > 0:
                        all_gts.append(gt.numpy())
                    else:
                        all_gts.append(None)
                else:
                    all_gts.append(None)

            n_with_mask = sum(1 for g in all_gts if g is not None)
            pauroc = compute_pixel_auroc(all_preds, all_gts)
            pro = compute_pro_score(all_preds, all_gts, DEVICE) if n_with_mask > 0 else 0.0

            all_results[tap_name]["pauroc"].append(pauroc)
            all_results[tap_name]["pro"].append(pro)
            print(f"  {cat}: pAUROC={pauroc*100:.1f}%, PRO={pro*100:.1f}%")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: Mean metrics by tap configuration (zero-shot, no training)")
    print("=" * 80)
    print(f"{'Tap Config':<20} {'Taps':<25} {'pAUROC':>10} {'PRO':>10}")
    print("-" * 80)

    for tap_name, tap_indices in TAP_CONFIGS.items():
        mean_pauroc = np.mean(all_results[tap_name]["pauroc"])
        mean_pro = np.mean(all_results[tap_name]["pro"])
        taps_str = str(tap_indices)
        print(f"{tap_name:<20} {taps_str:<25} {mean_pauroc*100:>9.1f}% {mean_pro*100:>9.1f}%")

    print("=" * 80)


if __name__ == "__main__":
    main()
