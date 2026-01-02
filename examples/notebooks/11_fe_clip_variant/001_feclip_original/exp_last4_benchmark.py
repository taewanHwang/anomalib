"""Benchmark last4 [20,21,22,23] tap configuration (optimal baseline).

Full evaluation on MVTec, VisA, BTAD with 5 seeds each.
Uses optimal settings: fc_patch lr=1/100, macro-average evaluation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python exp_last4_benchmark.py --mode visa --seed 42
"""

import argparse
import logging
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms.v2 import Normalize, Resize, CenterCrop, InterpolationMode

from anomalib.data import MVTecAD, Visa, BTech
from anomalib.data.dataclasses.torch.image import ImageBatch, ImageItem
from anomalib.models.image import FEClip
from anomalib.models.image.feclip.losses import bce_loss, focal_loss, dice_loss

from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results_last4_benchmark"
MVTEC_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/MVTecAD")
VISA_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/VisA")
BTAD_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/BTech")

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"
]
VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
]
BTAD_CATEGORIES = ["01", "02", "03"]

# Optimal baseline configuration
TAP_INDICES = [20, 21, 22, 23]  # last4

# Fixed optimal settings
IMAGE_SIZE = 336
TEMPERATURE = 0.07
EPOCHS = 9
BATCH_SIZE = 16
ADAPTER_LR = 5e-4
FC_PATCH_LR = 5e-6  # adapter_lr / 100


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CLIPStyleDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, image_size=IMAGE_SIZE):
        self.dataset = dataset
        self.resize = Resize(image_size, antialias=True, interpolation=InterpolationMode.BICUBIC)
        self.crop = CenterCrop((image_size, image_size))
        self.mask_resize = Resize(image_size, antialias=False, interpolation=InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self.crop(self.resize(item.image))
        gt_mask = None
        if item.gt_mask is not None:
            gt_mask = self.crop(self.mask_resize(item.gt_mask))
        return ImageItem(
            image=image,
            gt_label=item.gt_label,
            gt_mask=gt_mask,
            image_path=item.image_path,
            mask_path=item.mask_path,
        )


def collect_test_data(dataset_class, root, categories):
    all_datasets = []
    for cat in categories:
        dm = dataset_class(root=root, category=cat, eval_batch_size=16, num_workers=4)
        dm.setup()
        all_datasets.append(dm.test_data)
    combined = ConcatDataset(all_datasets)
    return CLIPStyleDatasetWrapper(combined)


def train(train_dataset, exp_dir: Path, device: str = "cuda:0"):
    """Train FE-CLIP with last4 configuration."""

    model = FEClip(lr=ADAPTER_LR, w_cls=1.0, w_mask=1.0, tap_indices=TAP_INDICES)
    model.to(device)
    model.model.setup_text()
    model.train()

    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=ImageBatch.collate,
    )

    normalize = Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )

    # Separate parameters: adapters vs fc_patch (optimal setting)
    adapter_params = []
    fc_patch_params = []
    for name, param in model.model.named_parameters():
        if param.requires_grad:
            if 'fc_patch' in name:
                fc_patch_params.append(param)
            else:
                adapter_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': adapter_params, 'lr': ADAPTER_LR},
        {'params': fc_patch_params, 'lr': FC_PATCH_LR}
    ])

    logger.info(f"Training with tap_indices: {TAP_INDICES}")
    logger.info(f"  adapter_lr: {ADAPTER_LR}, fc_patch_lr: {FC_PATCH_LR}")

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        epoch_loss = []

        for batch in train_loader:
            images = normalize(batch.image).to(device)
            gt_label = batch.gt_label.float().to(device) if batch.gt_label is not None else None
            gt_mask = batch.gt_mask.float().to(device) if batch.gt_mask is not None else None

            if gt_mask is not None and gt_mask.ndim == 4:
                gt_mask = gt_mask.squeeze(1)

            scores, maps = model.model.forward_tokens(images)
            n_taps = len(scores)

            loss_cls = torch.tensor(0.0, device=device)
            if gt_label is not None:
                for s in scores:
                    loss_cls = loss_cls + bce_loss(s, gt_label)
                loss_cls = loss_cls / n_taps

            loss_mask = torch.tensor(0.0, device=device)
            if gt_mask is not None:
                for m in maps:
                    m_up = F.interpolate(
                        m.unsqueeze(1),
                        size=(IMAGE_SIZE, IMAGE_SIZE),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(1)
                    loss_mask = loss_mask + focal_loss(m_up, gt_mask) + dice_loss(m_up, gt_mask)
                loss_mask = loss_mask / n_taps

            total_loss = loss_cls + loss_mask

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss.append(total_loss.item())

        avg_loss = np.mean(epoch_loss)
        logger.info(f"Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'tap_indices': TAP_INDICES,
                'loss': best_loss,
            }, ckpt_dir / "best_model.pt")

    return model


def evaluate_macro(model, dataset_class, root, categories, device, transform):
    """Macro-average evaluation (paper method)."""
    model.eval()

    mask_resize = Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST, antialias=False)
    mask_crop = CenterCrop((IMAGE_SIZE, IMAGE_SIZE))

    category_results = {}

    for cat in categories:
        dm = dataset_class(root=root, category=cat, eval_batch_size=1, num_workers=4)
        dm.setup()

        cat_scores, cat_labels, cat_maps, cat_masks = [], [], [], []

        for batch in dm.test_dataloader():
            images = transform(batch.image).to(device)
            gt_label = batch.gt_label.item()
            gt_mask = batch.gt_mask[0] if batch.gt_mask is not None else None

            with torch.no_grad():
                output = model.model(images)

            score = output.pred_score[0].cpu().item()
            amap = output.anomaly_map[0].cpu().numpy()

            cat_scores.append(score)
            cat_labels.append(gt_label)
            cat_maps.append(amap)

            if gt_mask is not None:
                if gt_mask.ndim == 2:
                    gt_mask = gt_mask.unsqueeze(0)
                gt_mask_processed = mask_crop(mask_resize(gt_mask)).squeeze(0).numpy()
                cat_masks.append((gt_mask_processed > 0.5).astype(int))
            else:
                cat_masks.append(np.zeros_like(amap))

        # Category-level metrics
        if len(set(cat_labels)) > 1:
            cat_auroc = roc_auc_score(cat_labels, cat_scores) * 100
            cat_ap = average_precision_score(cat_labels, cat_scores) * 100
        else:
            cat_auroc, cat_ap = 50.0, 50.0

        cat_maps_flat = np.concatenate([m.flatten() for m in cat_maps])
        cat_masks_flat = np.concatenate([m.flatten() for m in cat_masks])

        if len(np.unique(cat_masks_flat)) > 1:
            cat_pauroc = roc_auc_score(cat_masks_flat, cat_maps_flat) * 100
        else:
            cat_pauroc = 50.0

        category_results[cat] = {
            "AUROC": cat_auroc, "AP": cat_ap, "pAUROC": cat_pauroc,
        }
        logger.info(f"  {cat}: AUROC={cat_auroc:.1f}%, pAUROC={cat_pauroc:.1f}%")

    # Macro-average
    auroc = np.mean([r["AUROC"] for r in category_results.values()])
    ap = np.mean([r["AP"] for r in category_results.values()])
    pauroc = np.mean([r["pAUROC"] for r in category_results.values()])

    return {"AUROC": auroc, "AP": ap, "pAUROC": pauroc, "per_category": category_results}


def main():
    parser = argparse.ArgumentParser(description="Last4 Benchmark")
    parser.add_argument("--mode", type=str, required=True, choices=["mvtec", "visa", "btad"],
                        help="mvtec: train on VisA, eval on MVTec | visa/btad: train on MVTec")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    set_seed(args.seed)

    exp_name = f"last4_{args.mode}_seed{args.seed}"
    exp_dir = RESULTS_DIR / f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"Last4 Benchmark: {TAP_INDICES}")
    logger.info(f"Mode: {args.mode}, Seed: {args.seed}")
    logger.info("=" * 80)

    # Setup datasets based on mode
    if args.mode == "mvtec":
        # Train on VisA, eval on MVTec
        logger.info("Collecting VisA test data for training...")
        train_dataset = collect_test_data(Visa, VISA_ROOT, VISA_CATEGORIES)
        eval_class, eval_root, eval_cats = MVTecAD, MVTEC_ROOT, MVTEC_CATEGORIES
    elif args.mode == "visa":
        # Train on MVTec, eval on VisA
        logger.info("Collecting MVTec test data for training...")
        train_dataset = collect_test_data(MVTecAD, MVTEC_ROOT, MVTEC_CATEGORIES)
        eval_class, eval_root, eval_cats = Visa, VISA_ROOT, VISA_CATEGORIES
    else:  # btad
        # Train on MVTec, eval on BTAD
        logger.info("Collecting MVTec test data for training...")
        train_dataset = collect_test_data(MVTecAD, MVTEC_ROOT, MVTEC_CATEGORIES)
        eval_class, eval_root, eval_cats = BTech, BTAD_ROOT, BTAD_CATEGORIES

    # Train
    model = train(train_dataset, exp_dir, device=args.device)

    # Evaluate with macro-average
    logger.info("\nEvaluating with macro-average...")
    transform = model.pre_processor.transform
    results = evaluate_macro(model, eval_class, eval_root, eval_cats, args.device, transform)

    logger.info("\n" + "=" * 80)
    logger.info(f"RESULTS: last4 {TAP_INDICES} on {args.mode.upper()}")
    logger.info(f"  Image AUROC: {results['AUROC']:.1f}%")
    logger.info(f"  Image AP:    {results['AP']:.1f}%")
    logger.info(f"  Pixel AUROC: {results['pAUROC']:.1f}%")
    logger.info("=" * 80)

    # Save results
    save_results = {
        "tap_indices": TAP_INDICES,
        "mode": args.mode,
        "seed": args.seed,
        "AUROC": results["AUROC"],
        "AP": results["AP"],
        "pAUROC": results["pAUROC"],
        "per_category": results["per_category"],
    }

    with open(exp_dir / "results.json", "w") as f:
        json.dump(save_results, f, indent=2)

    # Append to summary
    summary_file = RESULTS_DIR / "summary.txt"
    with open(summary_file, "a") as f:
        f.write(f"{args.mode}\t{args.seed}\t{results['AUROC']:.1f}\t{results['AP']:.1f}\t{results['pAUROC']:.1f}\n")

    logger.info(f"Results saved to {exp_dir}")


if __name__ == "__main__":
    main()
