"""Analyze per-tap PRO contribution from trained model.

The trained model uses last4 [20,21,22,23]. We want to understand
which taps contribute most to PRO vs AUROC.
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
CHECKPOINT = "examples/notebooks/11_fe_clip_variant/results/feclip_mvtec_seed42_20260101_061222/checkpoints/best_model.pt"
CATEGORIES = ["bottle", "cable", "capsule", "carpet", "hazelnut", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper", "grid", "leather"]


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
    print("Analyzing per-tap PRO contribution")
    print("=" * 80)

    # Load trained model
    model = FEClip(tap_indices=[20, 21, 22, 23])
    model.to(DEVICE)

    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    transform = model.pre_processor.transform
    print(f"Loaded: {CHECKPOINT}")

    # GT mask preprocessing
    mask_resize = Resize(336, interpolation=InterpolationMode.NEAREST, antialias=False)
    mask_crop = CenterCrop((336, 336))

    def preprocess_gt_mask(mask_tensor):
        if mask_tensor is None:
            return None
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        return mask_crop(mask_resize(mask_tensor)).squeeze(0)

    # Store per-tap results
    tap_names = ["Tap 0 (block 20)", "Tap 1 (block 21)", "Tap 2 (block 22)", "Tap 3 (block 23)", "Average"]
    results = {name: {"pauroc": [], "pro": []} for name in tap_names}

    for cat in CATEGORIES:
        print(f"\nProcessing {cat}...")

        dm = MVTecAD(
            root="/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/MVTecAD",
            category=cat,
            eval_batch_size=1
        )
        dm.setup()

        # Store maps per tap
        per_tap_maps = {i: [] for i in range(4)}
        avg_maps = []
        all_gts = []

        for batch in dm.test_dataloader():
            images = transform(batch.image).to(DEVICE)

            with torch.no_grad():
                scores, map_list = model.model.forward_tokens(images)

            # Per-tap maps (24x24 → 336x336)
            for i, tap_map in enumerate(map_list):
                upsampled = F.interpolate(
                    tap_map.unsqueeze(1),
                    size=(336, 336),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                per_tap_maps[i].append(upsampled[0].cpu().numpy())

            # Average map
            avg_map = torch.stack(map_list, dim=0).mean(dim=0)
            avg_upsampled = F.interpolate(
                avg_map.unsqueeze(1),
                size=(336, 336),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            avg_maps.append(avg_upsampled[0].cpu().numpy())

            # GT mask
            if batch.gt_mask is not None:
                gt = preprocess_gt_mask(batch.gt_mask[0])
                if gt is not None and gt.sum() > 0:
                    all_gts.append(gt.numpy())
                else:
                    all_gts.append(None)
            else:
                all_gts.append(None)

        n_with_mask = sum(1 for g in all_gts if g is not None)

        # Compute metrics for each tap and average
        for i in range(4):
            pauroc = compute_pixel_auroc(per_tap_maps[i], all_gts)
            pro = compute_pro_score(per_tap_maps[i], all_gts, DEVICE) if n_with_mask > 0 else 0.0
            results[tap_names[i]]["pauroc"].append(pauroc)
            results[tap_names[i]]["pro"].append(pro)

        pauroc = compute_pixel_auroc(avg_maps, all_gts)
        pro = compute_pro_score(avg_maps, all_gts, DEVICE) if n_with_mask > 0 else 0.0
        results["Average"]["pauroc"].append(pauroc)
        results["Average"]["pro"].append(pro)

    # Print per-category results
    print("\n" + "=" * 100)
    print("Per-Category PRO by Tap")
    print("=" * 100)
    header = f"{'Category':<15}" + "".join(f"{name:>18}" for name in tap_names)
    print(header)
    print("-" * 100)

    for i, cat in enumerate(CATEGORIES):
        row = f"{cat:<15}"
        for name in tap_names:
            row += f"{results[name]['pro'][i]*100:>17.1f}%"
        print(row)

    # Print means
    print("-" * 100)
    mean_row = f"{'MEAN':<15}"
    for name in tap_names:
        mean_pro = np.mean(results[name]["pro"])
        mean_row += f"{mean_pro*100:>17.1f}%"
    print(mean_row)
    print("=" * 100)

    # Print summary comparison
    print("\nSUMMARY:")
    print("-" * 60)
    for name in tap_names:
        mean_pauroc = np.mean(results[name]["pauroc"])
        mean_pro = np.mean(results[name]["pro"])
        print(f"{name:<25}: pAUROC={mean_pauroc*100:.1f}%, PRO={mean_pro*100:.1f}%")

    # Check if any single tap is better than average for PRO
    avg_pro = np.mean(results["Average"]["pro"])
    for i in range(4):
        tap_pro = np.mean(results[tap_names[i]]["pro"])
        diff = tap_pro - avg_pro
        print(f"  {tap_names[i]} vs Average PRO: {diff*100:+.1f}%")


if __name__ == "__main__":
    main()
