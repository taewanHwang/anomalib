"""Experiment 13: Macro-average vs Micro-average Evaluation.

The paper states "dataset-level results = sub-datasets average".
This means macro-average (category-wise metric → average), not micro-average.

Current implementation uses micro-average (all samples pooled together).
This experiment compares both methods to see the impact on PRO/pAUROC.

Uses best settings from Exp12: fc_patch low_lr_100x
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
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.v2 import Normalize, Resize, CenterCrop, InterpolationMode

from anomalib.data import MVTecAD, Visa
from anomalib.data.dataclasses.torch.image import ImageBatch, ImageItem
from anomalib.models.image import FEClip
from anomalib.models.image.feclip.losses import bce_loss, focal_loss, dice_loss

from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
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
        self.image_size = image_size
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
        logger.info(f"  {cat}: {len(dm.test_data)} samples")
    combined = ConcatDataset(all_datasets)
    logger.info(f"  Total: {len(combined)} samples")
    return CLIPStyleDatasetWrapper(combined)


def train_model(
    train_dataset,
    exp_dir: Path,
    epochs: int = 9,
    batch_size: int = 16,
    lr: float = 5e-4,
    device: str = "cuda:0",
):
    """Train FE-CLIP with best settings (fc_patch low_lr_100x)."""

    model = FEClip(lr=lr, w_cls=1.0, w_mask=1.0, tap_indices=[20, 21, 22, 23])
    model.to(device)
    model.model.setup_text()
    model.train()

    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=ImageBatch.collate,
    )

    normalize = Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )

    # Separate parameters: adapters vs fc_patch
    adapter_params = []
    fc_patch_params = []

    for name, param in model.model.named_parameters():
        if param.requires_grad:
            if 'fc_patch' in name:
                fc_patch_params.append(param)
            else:
                adapter_params.append(param)

    # Best setting: fc_patch with 100x lower lr
    optimizer = torch.optim.Adam([
        {'params': adapter_params, 'lr': lr},
        {'params': fc_patch_params, 'lr': lr / 100}
    ])

    logger.info(f"Training with fc_patch low_lr_100x policy")
    logger.info(f"  Adapter lr: {lr}, fc_patch lr: {lr/100}")

    best_loss = float('inf')

    for epoch in range(epochs):
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
        logger.info(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
            }, ckpt_dir / "best_model.pt")

    return model


def evaluate_micro(model, dataset_class, root, categories, device, transform):
    """Micro-average: pool all samples together."""
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

    # Micro-average: all samples pooled
    auroc = roc_auc_score(all_labels, all_scores) * 100
    ap = average_precision_score(all_labels, all_scores) * 100

    all_maps_flat = np.concatenate([m.flatten() for m in all_maps])
    all_masks_flat = np.concatenate([m.flatten() for m in all_masks])
    pauroc = roc_auc_score(all_masks_flat, all_maps_flat) * 100

    return {"AUROC": auroc, "AP": ap, "pAUROC": pauroc}


def evaluate_macro(model, dataset_class, root, categories, device, transform):
    """Macro-average: category-wise metric → average (paper method)."""
    model.eval()

    mask_resize = Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST, antialias=False)
    mask_crop = CenterCrop((IMAGE_SIZE, IMAGE_SIZE))

    category_results = {}

    for cat in categories:
        dm = dataset_class(root=root, category=cat, eval_batch_size=1, num_workers=4)
        dm.setup()

        cat_scores = []
        cat_labels = []
        cat_maps = []
        cat_masks = []

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
        # AUROC: need both classes
        if len(set(cat_labels)) > 1:
            cat_auroc = roc_auc_score(cat_labels, cat_scores) * 100
            cat_ap = average_precision_score(cat_labels, cat_scores) * 100
        else:
            cat_auroc = 50.0  # undefined
            cat_ap = 50.0

        # pAUROC: need both classes in masks
        cat_maps_flat = np.concatenate([m.flatten() for m in cat_maps])
        cat_masks_flat = np.concatenate([m.flatten() for m in cat_masks])

        if len(np.unique(cat_masks_flat)) > 1:
            cat_pauroc = roc_auc_score(cat_masks_flat, cat_maps_flat) * 100
        else:
            cat_pauroc = 50.0  # undefined (all normal or all defect)

        category_results[cat] = {
            "AUROC": cat_auroc,
            "AP": cat_ap,
            "pAUROC": cat_pauroc,
            "n_samples": len(cat_labels),
            "n_defect": sum(cat_labels),
        }

        logger.info(f"  {cat}: AUROC={cat_auroc:.1f}%, pAUROC={cat_pauroc:.1f}% (n={len(cat_labels)}, defect={sum(cat_labels)})")

    # Macro-average: mean of category metrics
    aurocs = [r["AUROC"] for r in category_results.values()]
    aps = [r["AP"] for r in category_results.values()]
    paurocs = [r["pAUROC"] for r in category_results.values()]

    return {
        "AUROC": np.mean(aurocs),
        "AP": np.mean(aps),
        "pAUROC": np.mean(paurocs),
        "per_category": category_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="visa", choices=["mvtec", "visa"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    set_seed(args.seed)

    exp_name = f"exp13_macro_{args.mode}_seed{args.seed}"
    exp_dir = RESULTS_DIR / f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"Experiment 13: Macro vs Micro Average Evaluation")
    logger.info(f"Mode: {args.mode}, Seed: {args.seed}")
    logger.info("=" * 80)

    # Collect training data
    if args.mode == "visa":
        logger.info("Collecting MVTec test data for training...")
        train_dataset = collect_test_data(MVTecAD, MVTEC_ROOT, MVTEC_CATEGORIES)
        eval_class = Visa
        eval_root = VISA_ROOT
        eval_cats = VISA_CATEGORIES
    else:
        logger.info("Collecting VisA test data for training...")
        train_dataset = collect_test_data(Visa, VISA_ROOT, VISA_CATEGORIES)
        eval_class = MVTecAD
        eval_root = MVTEC_ROOT
        eval_cats = MVTEC_CATEGORIES

    # Train with best settings
    model = train_model(train_dataset, exp_dir, epochs=args.epochs, device=args.device)

    # Evaluate with both methods
    logger.info("\n" + "=" * 80)
    logger.info("Evaluating with MICRO-average (current implementation)...")
    logger.info("=" * 80)
    transform = model.pre_processor.transform
    micro_results = evaluate_micro(model, eval_class, eval_root, eval_cats, args.device, transform)

    logger.info(f"\nMICRO Results:")
    logger.info(f"  Image AUROC: {micro_results['AUROC']:.1f}%")
    logger.info(f"  Image AP:    {micro_results['AP']:.1f}%")
    logger.info(f"  Pixel AUROC: {micro_results['pAUROC']:.1f}%")

    logger.info("\n" + "=" * 80)
    logger.info("Evaluating with MACRO-average (paper method)...")
    logger.info("=" * 80)
    macro_results = evaluate_macro(model, eval_class, eval_root, eval_cats, args.device, transform)

    logger.info(f"\nMACRO Results:")
    logger.info(f"  Image AUROC: {macro_results['AUROC']:.1f}%")
    logger.info(f"  Image AP:    {macro_results['AP']:.1f}%")
    logger.info(f"  Pixel AUROC: {macro_results['pAUROC']:.1f}%")

    # Comparison
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON: Micro vs Macro")
    logger.info("=" * 80)
    logger.info(f"  AUROC:  Micro={micro_results['AUROC']:.1f}% vs Macro={macro_results['AUROC']:.1f}% (diff={macro_results['AUROC']-micro_results['AUROC']:+.1f}%)")
    logger.info(f"  AP:     Micro={micro_results['AP']:.1f}% vs Macro={macro_results['AP']:.1f}% (diff={macro_results['AP']-micro_results['AP']:+.1f}%)")
    logger.info(f"  pAUROC: Micro={micro_results['pAUROC']:.1f}% vs Macro={macro_results['pAUROC']:.1f}% (diff={macro_results['pAUROC']-micro_results['pAUROC']:+.1f}%)")

    # Save results
    results = {
        "micro": micro_results,
        "macro": {k: v for k, v in macro_results.items() if k != "per_category"},
        "per_category": macro_results["per_category"],
        "seed": args.seed,
        "mode": args.mode,
    }

    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {exp_dir / 'results.json'}")


if __name__ == "__main__":
    main()
