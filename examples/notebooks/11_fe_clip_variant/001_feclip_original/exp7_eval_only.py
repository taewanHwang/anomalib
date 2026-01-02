"""Evaluate trained exp7 models (abnormal-only vs baseline)."""

import argparse
import logging
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.v2 import Resize, CenterCrop, InterpolationMode

from anomalib.data import MVTecAD, Visa
from anomalib.models.image import FEClip
from sklearn.metrics import roc_auc_score, average_precision_score

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


def evaluate_model(model, dataset_class, root, categories, device, transform):
    """Evaluate model on target dataset."""
    model.eval()

    mask_resize = Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST, antialias=False)
    mask_crop = CenterCrop((IMAGE_SIZE, IMAGE_SIZE))

    all_scores = []
    all_labels = []
    all_maps = []
    all_masks = []

    for cat in categories:
        dm = dataset_class(root=root, category=cat, eval_batch_size=1, num_workers=4)
        dm.setup()

        for batch in dm.test_dataloader():
            images = transform(batch.image).to(device)
            gt_label = batch.gt_label.item()
            gt_mask = batch.gt_mask[0] if batch.gt_mask is not None else None

            with torch.no_grad():
                output = model.model(images)

            score = output.pred_score[0].cpu().item()
            amap = output.anomaly_map[0].cpu().numpy()

            all_scores.append(score)
            all_labels.append(gt_label)
            all_maps.append(amap)

            if gt_mask is not None:
                if gt_mask.ndim == 2:
                    gt_mask = gt_mask.unsqueeze(0)
                gt_mask_processed = mask_crop(mask_resize(gt_mask)).squeeze(0).numpy()
                all_masks.append((gt_mask_processed > 0.5).astype(int))
            else:
                all_masks.append(np.zeros_like(amap))

    # Image-level metrics
    auroc = roc_auc_score(all_labels, all_scores) * 100
    ap = average_precision_score(all_labels, all_scores) * 100

    # Pixel-level metrics
    all_maps_flat = np.concatenate([m.flatten() for m in all_maps])
    all_masks_flat = np.concatenate([m.flatten() for m in all_masks])
    pauroc = roc_auc_score(all_masks_flat, all_maps_flat) * 100

    # Skip PRO for now due to connected_components bug
    return {"AUROC": auroc, "AP": ap, "pAUROC": pauroc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--target", type=str, required=True, choices=["mvtec", "visa"])
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.target == "visa":
        dataset_class = Visa
        root = VISA_ROOT
        categories = VISA_CATEGORIES
    else:
        dataset_class = MVTecAD
        root = MVTEC_ROOT
        categories = MVTEC_CATEGORIES

    logger.info("=" * 80)
    logger.info(f"Evaluating checkpoint: {args.checkpoint}")
    logger.info(f"Target: {args.target}")
    logger.info("=" * 80)

    # Load model
    model = FEClip(tap_indices=[20, 21, 22, 23])
    model.to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    transform = model.pre_processor.transform

    results = evaluate_model(model, dataset_class, root, categories, args.device, transform)

    logger.info("\n" + "=" * 80)
    logger.info(f"RESULTS ({args.target}):")
    logger.info(f"  Image AUROC: {results['AUROC']:.1f}%")
    logger.info(f"  Image AP:    {results['AP']:.1f}%")
    logger.info(f"  Pixel AUROC: {results['pAUROC']:.1f}%")
    logger.info("=" * 80)

    # Save results
    ckpt_dir = Path(args.checkpoint).parent.parent
    with open(ckpt_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
