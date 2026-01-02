"""Test different upsampling methods and Gaussian smoothing impact on PRO.

Priority experiment to identify PRO gap cause:
1. bilinear vs nearest upsampling
2. With/without Gaussian smoothing
3. 2x2 comparison
"""

import torch
import numpy as np
from pathlib import Path
from torch.nn import functional as F
from torchvision.transforms.v2 import Resize, CenterCrop, InterpolationMode, GaussianBlur

from anomalib.data import MVTecAD
from anomalib.data.datamodules.image.visa import Visa
from anomalib.models.image import FEClip
from anomalib.metrics.aupro import _AUPRO

# Settings
DEVICE = "cuda:0"
DATASET = "mvtec"  # "mvtec" or "visa"
# Test multiple categories
CATEGORIES = ["bottle", "cable", "capsule", "carpet", "hazelnut", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper", "grid", "leather"]
CHECKPOINT = "examples/notebooks/11_fe_clip_variant/results/feclip_mvtec_seed42_20260101_061222/checkpoints/best_model.pt"


def compute_pixel_auroc(all_masks_pred: list, all_masks_gt: list) -> float:
    """Compute pixel-level AUROC."""
    from sklearn.metrics import roc_auc_score
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
    """Compute PRO (Per-Region Overlap) score."""
    pro_metric = _AUPRO(fpr_limit=0.3)
    pro_metric = pro_metric.to(device)  # Move metric to device
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
    except RuntimeError as e:
        print(f"PRO compute error: {e}")
        return 0.0


def upsample_anomaly_map(amap: torch.Tensor, target_size: tuple, mode: str, gaussian_sigma: float = 0.0) -> torch.Tensor:
    """Upsample anomaly map with different methods.

    Args:
        amap: Anomaly map of shape (B, Ht, Wt)
        target_size: Target (H, W)
        mode: "bilinear", "nearest", or "patch_repeat"
        gaussian_sigma: Sigma for Gaussian blur (0 = no blur)

    Returns:
        Upsampled map of shape (B, H, W)
    """
    B, Ht, Wt = amap.shape

    if mode == "patch_repeat":
        # Repeat each patch to fill a 14x14 tile (assuming 24x24 -> 336x336)
        scale_h = target_size[0] // Ht
        scale_w = target_size[1] // Wt
        # Repeat using repeat_interleave
        out = amap.unsqueeze(1)  # (B, 1, Ht, Wt)
        out = out.repeat_interleave(scale_h, dim=2).repeat_interleave(scale_w, dim=3)
        out = out.squeeze(1)  # (B, H, W)
    else:
        out = F.interpolate(
            amap.unsqueeze(1),
            size=target_size,
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
        ).squeeze(1)

    # Apply Gaussian blur if specified
    if gaussian_sigma > 0:
        kernel_size = int(gaussian_sigma * 6) | 1  # Make odd
        kernel_size = max(kernel_size, 3)
        blur = GaussianBlur(kernel_size=kernel_size, sigma=gaussian_sigma)
        out = blur(out.unsqueeze(1)).squeeze(1)

    return out


def main():
    print(f"Testing upsampling methods on {DATASET}")
    print("=" * 60)

    # Load model
    model = FEClip(tap_indices=[20, 21, 22, 23])
    model.to(DEVICE)

    # Load checkpoint
    if Path(CHECKPOINT).exists():
        checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
        # Checkpoint has 'model_state_dict' key
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded: {CHECKPOINT}")
    else:
        print("No checkpoint found, using random weights")
        model.model.setup_text()

    model.eval()
    transform = model.pre_processor.transform

    # GT mask preprocessing (CLIP-style)
    mask_resize = Resize(336, interpolation=InterpolationMode.NEAREST, antialias=False)
    mask_crop = CenterCrop((336, 336))

    def preprocess_gt_mask(mask_tensor):
        if mask_tensor is None:
            return None
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        return mask_crop(mask_resize(mask_tensor)).squeeze(0)

    # Upsampling configs to test
    configs = [
        ("bilinear", 0.0),
        ("bilinear", 4.0),
        ("nearest", 0.0),
        ("nearest", 4.0),
    ]

    # Store results per category
    all_results = {cat: {} for cat in CATEGORIES}

    for cat in CATEGORIES:
        print(f"\nProcessing {cat}...")

        # Load data
        if DATASET == "mvtec":
            dm = MVTecAD(
                root="/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/MVTecAD",
                category=cat,
                eval_batch_size=1
            )
        else:
            dm = Visa(
                root="/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/VisA_highshot",
                category=cat,
                eval_batch_size=1
            )
        dm.setup()

        # Collect raw anomaly maps
        raw_maps = []
        gt_masks = []

        for batch in dm.test_dataloader():
            images = transform(batch.image).to(DEVICE)

            with torch.no_grad():
                scores, map_list = model.model.forward_tokens(images)

            avg_map = torch.stack(map_list, dim=0).mean(dim=0)
            raw_maps.append(avg_map.cpu())

            if batch.gt_mask is not None:
                gt = preprocess_gt_mask(batch.gt_mask[0])
                if gt is not None and gt.sum() > 0:
                    gt_masks.append(gt.numpy())
                else:
                    gt_masks.append(None)
            else:
                gt_masks.append(None)

        n_with_mask = sum(1 for g in gt_masks if g is not None)

        # Test each config
        for mode, sigma in configs:
            all_preds = []
            all_gts = []

            for raw_map, gt in zip(raw_maps, gt_masks):
                upsampled = upsample_anomaly_map(raw_map, (336, 336), mode, sigma)
                all_preds.append(upsampled[0].numpy())
                all_gts.append(gt)

            pauroc = compute_pixel_auroc(all_preds, all_gts)
            pro = compute_pro_score(all_preds, all_gts, DEVICE) if n_with_mask > 0 else 0.0

            config_name = f"{mode}_s{int(sigma)}"
            all_results[cat][config_name] = {"pauroc": pauroc, "pro": pro}

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY: PRO scores by category and upsampling method")
    print("=" * 100)
    configs_names = [f"{m}_s{int(s)}" for m, s in configs]
    header = f"{'Category':<15}" + "".join(f"{c:>15}" for c in configs_names)
    print(header)
    print("-" * 100)

    for cat in CATEGORIES:
        row = f"{cat:<15}"
        for cfg in configs_names:
            pro = all_results[cat][cfg]["pro"]
            row += f"{pro*100:>14.1f}%"
        print(row)

    # Calculate means
    print("-" * 100)
    mean_row = f"{'MEAN':<15}"
    for cfg in configs_names:
        mean_pro = np.mean([all_results[cat][cfg]["pro"] for cat in CATEGORIES])
        mean_row += f"{mean_pro*100:>14.1f}%"
    print(mean_row)
    print("=" * 100)

    # Find best overall config
    means = {}
    for cfg in configs_names:
        means[cfg] = np.mean([all_results[cat][cfg]["pro"] for cat in CATEGORIES])

    best_cfg = max(means, key=means.get)
    current_cfg = "bilinear_s0"
    print(f"\nBest config: {best_cfg} with mean PRO={means[best_cfg]*100:.1f}%")
    print(f"Current (bilinear_s0): mean PRO={means[current_cfg]*100:.1f}%")
    print(f"PRO improvement: {(means[best_cfg] - means[current_cfg])*100:+.1f}%")


if __name__ == "__main__":
    main()
