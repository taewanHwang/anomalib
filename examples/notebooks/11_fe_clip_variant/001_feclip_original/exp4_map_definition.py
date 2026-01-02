"""Experiment 4: Map definition comparison for PRO.

Hypothesis: The paper might use logit_margin instead of probability for anomaly map.
- Current: amap = softmax(z·t/τ)[...,1] → narrow dynamic range (0.39~0.63)
- Alternative 1: logit_margin = (z·t_abn - z·t_nor)/τ → wider range
- Alternative 2: cos_gap = cos(z, t_abn) - cos(z, t_nor)

This experiment tests without retraining - only changes the inference map definition.
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
        # Normalize to [0, 1]
        p_min, p_max = pred_tensor.min(), pred_tensor.max()
        if p_max > p_min:
            pred_tensor = (pred_tensor - p_min) / (p_max - p_min)
        pro_metric.update(pred_tensor.unsqueeze(0), gt_tensor.unsqueeze(0))
    try:
        return pro_metric.compute().item()
    except:
        return 0.0


def compute_map_prob_abnormal(patch_txt: torch.Tensor, text_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    """Original: P(abnormal) via softmax."""
    z = F.normalize(patch_txt, dim=-1)
    logits = (z @ text_emb.t()) / temperature  # (B, H, W, 2)
    return logits.softmax(dim=-1)[..., 1]  # P(abnormal)


def compute_map_logit_margin(patch_txt: torch.Tensor, text_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    """Alternative 1: logit_margin = logit_abn - logit_nor."""
    z = F.normalize(patch_txt, dim=-1)
    logits = (z @ text_emb.t()) / temperature  # (B, H, W, 2)
    margin = logits[..., 1] - logits[..., 0]  # abnormal - normal
    return margin


def compute_map_cos_gap(patch_txt: torch.Tensor, text_emb: torch.Tensor, temperature: float = None) -> torch.Tensor:
    """Alternative 2: cos(z, t_abn) - cos(z, t_nor)."""
    z = F.normalize(patch_txt, dim=-1)
    t_emb = F.normalize(text_emb, dim=-1)
    cos_abn = (z * t_emb[1:2]).sum(dim=-1)  # cos with abnormal
    cos_nor = (z * t_emb[0:1]).sum(dim=-1)  # cos with normal
    return cos_abn - cos_nor


def compute_map_sigmoid_margin(patch_txt: torch.Tensor, text_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    """Alternative 3: sigmoid(margin) for bounded output."""
    z = F.normalize(patch_txt, dim=-1)
    logits = (z @ text_emb.t()) / temperature
    margin = logits[..., 1] - logits[..., 0]
    return torch.sigmoid(margin)


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

    logger.info(f"Experiment 4: Map Definition Comparison")
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

    # Get text embeddings and temperature
    text_emb = model.model.text_emb  # (2, text_dim)
    temperature = model.model.temperature
    logger.info(f"Temperature: {temperature}")

    # GT preprocessing
    mask_resize = Resize(336, interpolation=InterpolationMode.NEAREST, antialias=False)
    mask_crop = CenterCrop((336, 336))

    def preprocess_gt_mask(mask_tensor):
        if mask_tensor is None:
            return None
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        return mask_crop(mask_resize(mask_tensor)).squeeze(0)

    # Map definitions to test
    map_funcs = {
        "prob_abnormal": compute_map_prob_abnormal,
        "logit_margin": compute_map_logit_margin,
        "cos_gap": compute_map_cos_gap,
        "sigmoid_margin": compute_map_sigmoid_margin,
    }

    all_results = {name: {"pauroc": [], "pro": [], "range_min": [], "range_max": []} for name in map_funcs}

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

        # Collect patch features
        all_patch_txts = []
        all_gts = []

        for batch in dm.test_dataloader():
            images = transform(batch.image).to(DEVICE)

            with torch.no_grad():
                # Get patch features before prob_abnormal
                scores, map_list = model.model.forward_tokens(images)

                # We need to access internal patch_txt before prob_abnormal
                # For this, we'll recompute from the raw map values
                # The map_list contains prob_abnormal outputs, but we need logits

                # Actually, we need to modify the forward to get raw patch_txt
                # For now, let's compute from scratch using the forward_tokens logic

            # Store GT
            if batch.gt_mask is not None:
                gt = preprocess_gt_mask(batch.gt_mask[0])
                if gt is not None and gt.sum() > 0:
                    all_gts.append(gt.numpy())
                else:
                    all_gts.append(None)
            else:
                all_gts.append(None)

        # Since we can't easily access internal patch_txt, let's modify the approach:
        # We'll compute the different maps by modifying prob_abnormal output

        # Actually, for a proper test, we need access to the raw logits
        # Let me create a custom forward that returns logits

        # For this experiment, let's compute maps differently
        # by using the model's internal method with modifications

        for batch in dm.test_dataloader():
            break  # Just get one batch to test

        # Test: compute different map definitions
        # We need to hook into the model to get patch_txt

        logger.info(f"  Computing maps with different definitions...")

        # Collect all maps per definition
        maps_by_def = {name: [] for name in map_funcs}

        dm.setup()  # Reset dataloader
        all_gts = []

        for batch in dm.test_dataloader():
            images = transform(batch.image).to(DEVICE)

            with torch.no_grad():
                # Get features through transformer
                x = model.model.visual.conv1(images)
                x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
                cls = model.model.visual.class_embedding.to(x.dtype)
                cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype)
                x = torch.cat([cls, x], dim=1)
                x = x + model.model.visual.positional_embedding.to(x.dtype)
                x = model.model.visual.ln_pre(x)

                Ht, Wt = model.model.grid_size
                B = x.shape[0]
                x = x.permute(1, 0, 2)

                tap_i = 0
                patch_hat_list = []

                for bi, blk in enumerate(model.model.visual.transformer.resblocks):
                    x = blk(x)
                    if bi in model.model.tap_blocks:
                        x_bld = x.permute(1, 0, 2)
                        patch = x_bld[:, 1:, :].reshape(B, Ht, Wt, -1)
                        ffe_out = model.model.ffe[tap_i](patch)
                        lfs_out = model.model.lfs[tap_i](patch, valid_mask=None)
                        patch_hat = model.model.lambda_fuse * (ffe_out + lfs_out) + (1 - model.model.lambda_fuse) * patch
                        x_bld = torch.cat([x_bld[:, :1, :], patch_hat.reshape(B, -1, x_bld.shape[-1])], dim=1)
                        x = x_bld.permute(1, 0, 2)

                        # Get patch_txt for this tap
                        patch_flat = patch_hat.reshape(B, -1, patch_hat.shape[-1])
                        patch_norm = model.model.visual.ln_post(patch_flat)
                        patch_norm = patch_norm.reshape(B, Ht, Wt, -1)
                        patch_txt = model.model.fc_patch(patch_norm)  # (B, Ht, Wt, text_dim)
                        patch_hat_list.append(patch_txt)
                        tap_i += 1

                # Average patch_txt across taps
                avg_patch_txt = torch.stack(patch_hat_list, dim=0).mean(dim=0)  # (B, Ht, Wt, text_dim)

                # Compute different map definitions
                for name, func in map_funcs.items():
                    amap = func(avg_patch_txt, text_emb, temperature)  # (B, Ht, Wt)
                    # Upsample to 336x336
                    amap_up = F.interpolate(
                        amap.unsqueeze(1),
                        size=(336, 336),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                    maps_by_def[name].append(amap_up[0].cpu().numpy())

            # GT
            if batch.gt_mask is not None:
                gt = preprocess_gt_mask(batch.gt_mask[0])
                if gt is not None and gt.sum() > 0:
                    all_gts.append(gt.numpy())
                else:
                    all_gts.append(None)
            else:
                all_gts.append(None)

        # Compute metrics for each map definition
        n_with_mask = sum(1 for g in all_gts if g is not None)

        for name in map_funcs:
            maps = maps_by_def[name]
            pauroc = compute_pixel_auroc(maps, all_gts)
            pro = compute_pro_score(maps, all_gts, DEVICE) if n_with_mask > 0 else 0.0

            # Compute range statistics
            valid_maps = [m for m, g in zip(maps, all_gts) if g is not None]
            if valid_maps:
                all_vals = np.concatenate([m.flatten() for m in valid_maps])
                range_min = float(np.min(all_vals))
                range_max = float(np.max(all_vals))
            else:
                range_min = range_max = 0.0

            all_results[name]["pauroc"].append(pauroc)
            all_results[name]["pro"].append(pro)
            all_results[name]["range_min"].append(range_min)
            all_results[name]["range_max"].append(range_max)

        # Log for this category
        for name in map_funcs:
            idx = len(all_results[name]["pauroc"]) - 1
            logger.info(f"  {name}: pAUROC={all_results[name]['pauroc'][idx]*100:.1f}%, "
                       f"PRO={all_results[name]['pro'][idx]*100:.1f}%, "
                       f"range=[{all_results[name]['range_min'][idx]:.3f}, {all_results[name]['range_max'][idx]:.3f}]")

    # Summary
    logger.info("\n" + "=" * 100)
    logger.info("SUMMARY: Map definition comparison")
    logger.info("=" * 100)

    header = f"{'Category':<15}" + "".join(f"{name:>18}" for name in map_funcs)
    logger.info("PRO results:")
    logger.info(header)
    logger.info("-" * 100)

    for i, cat in enumerate(CATEGORIES):
        row = f"{cat:<15}"
        for name in map_funcs:
            row += f"{all_results[name]['pro'][i]*100:>17.1f}%"
        logger.info(row)

    logger.info("-" * 100)
    mean_row = f"{'MEAN':<15}"
    for name in map_funcs:
        mean_pro = np.mean(all_results[name]["pro"])
        mean_row += f"{mean_pro*100:>17.1f}%"
    logger.info(mean_row)

    # pAUROC comparison
    logger.info("\npAUROC results:")
    pauroc_row = f"{'MEAN':<15}"
    for name in map_funcs:
        mean_pauroc = np.mean(all_results[name]["pauroc"])
        pauroc_row += f"{mean_pauroc*100:>17.1f}%"
    logger.info(pauroc_row)

    # Range comparison
    logger.info("\nValue range (min, max):")
    for name in map_funcs:
        avg_min = np.mean(all_results[name]["range_min"])
        avg_max = np.mean(all_results[name]["range_max"])
        logger.info(f"  {name}: [{avg_min:.3f}, {avg_max:.3f}]")

    logger.info("=" * 100)

    # Best method
    mean_pros = {name: np.mean(all_results[name]["pro"]) for name in map_funcs}
    best_method = max(mean_pros, key=mean_pros.get)
    baseline_pro = mean_pros["prob_abnormal"]
    best_pro = mean_pros[best_method]

    logger.info(f"\nBest method for PRO: {best_method}")
    logger.info(f"  prob_abnormal (current): PRO = {baseline_pro*100:.1f}%")
    logger.info(f"  {best_method}: PRO = {best_pro*100:.1f}%")
    logger.info(f"  Improvement: {(best_pro - baseline_pro)*100:+.1f}%")


if __name__ == "__main__":
    main()
