"""Experiment 10: GT Downsample for Mask Loss.

Instead of upsampling the 24x24 map to 336x336 GT,
downsample the GT to 24x24 and compute loss at token-grid resolution.

Hypothesis: This helps the model learn at the resolution it can actually represent,
leading to better region-wise consistency and higher PRO.

Variants:
- nearest: Nearest-neighbor interpolation (may lose thin defects)
- maxpool: Max pooling (preserves thin defects better)
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
TOKEN_SIZE = 24  # CLIP ViT-L-14-336 patch grid size


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


def downsample_gt(gt_mask, target_size, method="nearest"):
    """Downsample GT mask to token grid size.

    Args:
        gt_mask: (B, H, W) tensor
        target_size: target size (e.g., 24)
        method: "nearest" or "maxpool"

    Returns:
        (B, target_size, target_size) tensor
    """
    if method == "nearest":
        # Simple nearest neighbor downsample
        gt_down = F.interpolate(
            gt_mask.unsqueeze(1),
            size=(target_size, target_size),
            mode="nearest"
        ).squeeze(1)
    elif method == "maxpool":
        # Max pooling preserves thin defects better
        kernel_size = gt_mask.shape[-1] // target_size
        gt_down = F.max_pool2d(
            gt_mask.unsqueeze(1),
            kernel_size=kernel_size,
            stride=kernel_size
        ).squeeze(1)
    elif method == "avgpool":
        # Average pooling with threshold
        kernel_size = gt_mask.shape[-1] // target_size
        gt_down = F.avg_pool2d(
            gt_mask.unsqueeze(1),
            kernel_size=kernel_size,
            stride=kernel_size
        ).squeeze(1)
        gt_down = (gt_down > 0.5).float()  # Re-binarize
    else:
        raise ValueError(f"Unknown method: {method}")

    return gt_down


def train_with_gt_downsample(
    train_dataset,
    exp_dir: Path,
    epochs: int = 9,
    batch_size: int = 16,
    lr: float = 5e-4,
    device: str = "cuda:0",
    downsample_method: str = "nearest",  # "nearest", "maxpool", "avgpool"
):
    """Train FE-CLIP with GT downsampled to token grid."""

    model = FEClip(lr=lr, w_cls=1.0, w_mask=1.0, tap_indices=[20, 21, 22, 23])
    model.to(device)
    model.model.setup_text()
    model.train()

    tb_dir = exp_dir / "tensorboard_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

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

    params = [p for p in model.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    logger.info(f"Training: {len(train_dataset)} samples, {len(train_loader)} batches/epoch")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in params):,}")
    logger.info(f"Downsample method: {downsample_method}")
    logger.info(f"Token grid size: {TOKEN_SIZE}x{TOKEN_SIZE}")

    global_step = 0
    best_loss = float('inf')

    # Track GT stats
    total_gt_positive_original = 0
    total_gt_positive_downsampled = 0

    for epoch in range(epochs):
        epoch_losses = {"loss_total": [], "loss_cls": [], "loss_mask": []}

        for batch_idx, batch in enumerate(train_loader):
            images = normalize(batch.image).to(device)
            gt_label = batch.gt_label.float().to(device) if batch.gt_label is not None else None
            gt_mask = batch.gt_mask.float().to(device) if batch.gt_mask is not None else None

            if gt_mask is not None and gt_mask.ndim == 4:
                gt_mask = gt_mask.squeeze(1)

            # Forward - get 24x24 maps
            scores, maps = model.model.forward_tokens(images)
            n_taps = len(scores)

            # Classification loss (unchanged)
            loss_cls = torch.tensor(0.0, device=device)
            if gt_label is not None:
                for s in scores:
                    loss_cls = loss_cls + bce_loss(s, gt_label)
                loss_cls = loss_cls / n_taps

            # Mask loss with GT downsampled to 24x24
            loss_mask = torch.tensor(0.0, device=device)
            if gt_mask is not None:
                # KEY CHANGE: Downsample GT to 24x24 instead of upsampling map to 336
                gt_down = downsample_gt(gt_mask, TOKEN_SIZE, method=downsample_method)

                # Track stats (first epoch only)
                if epoch == 0:
                    total_gt_positive_original += gt_mask.sum().item()
                    total_gt_positive_downsampled += gt_down.sum().item()

                for m in maps:
                    # m is already 24x24, no upsampling needed
                    loss_mask = loss_mask + focal_loss(m, gt_down) + dice_loss(m, gt_down)
                loss_mask = loss_mask / n_taps

            total_loss = loss_cls + loss_mask

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_losses["loss_total"].append(total_loss.item())
            epoch_losses["loss_cls"].append(loss_cls.item())
            epoch_losses["loss_mask"].append(loss_mask.item())

            writer.add_scalar("train/loss_total", total_loss.item(), global_step)
            writer.add_scalar("train/loss_cls", loss_cls.item(), global_step)
            writer.add_scalar("train/loss_mask", loss_mask.item(), global_step)
            global_step += 1

        avg_loss = np.mean(epoch_losses["loss_total"])
        avg_cls = np.mean(epoch_losses["loss_cls"])
        avg_mask = np.mean(epoch_losses["loss_mask"])

        logger.info(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} (cls={avg_cls:.4f}, mask={avg_mask:.4f})")

        if epoch == 0:
            ratio = total_gt_positive_downsampled / total_gt_positive_original if total_gt_positive_original > 0 else 0
            logger.info(f"  GT positive ratio (down/orig): {ratio:.4f}")

        writer.add_scalar("epoch/loss_total", avg_loss, epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'downsample_method': downsample_method,
            }, ckpt_dir / "best_model.pt")
            logger.info(f"  Saved best model")

    writer.close()
    return model


def evaluate_model(model, dataset_class, root, categories, device, transform):
    """Evaluate model - same as before."""
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

    # Metrics
    auroc = roc_auc_score(all_labels, all_scores) * 100
    ap = average_precision_score(all_labels, all_scores) * 100

    all_maps_flat = np.concatenate([m.flatten() for m in all_maps])
    all_masks_flat = np.concatenate([m.flatten() for m in all_masks])
    pauroc = roc_auc_score(all_masks_flat, all_maps_flat) * 100

    return {"AUROC": auroc, "AP": ap, "pAUROC": pauroc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="visa", choices=["mvtec", "visa"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--downsample", type=str, default="nearest",
                        choices=["nearest", "maxpool", "avgpool", "baseline"])
    args = parser.parse_args()

    set_seed(args.seed)

    # baseline = original upsample method (for comparison)
    if args.downsample == "baseline":
        exp_name = f"exp10_baseline_{args.mode}_seed{args.seed}"
    else:
        exp_name = f"exp10_{args.downsample}_{args.mode}_seed{args.seed}"

    exp_dir = RESULTS_DIR / f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"Experiment 10: GT Downsample Training")
    logger.info(f"Mode: {args.mode}, Method: {args.downsample}, Seed: {args.seed}")
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

    # Train
    if args.downsample == "baseline":
        # Use original training (upsample map)
        from run_feclip_engine import train_feclip
        model = train_feclip(
            train_dataset, exp_dir, epochs=args.epochs, device=args.device,
            tap_indices=[20, 21, 22, 23]
        )
    else:
        model = train_with_gt_downsample(
            train_dataset, exp_dir, epochs=args.epochs, device=args.device,
            downsample_method=args.downsample
        )

    # Evaluate
    logger.info("\nEvaluating on target dataset...")
    transform = model.pre_processor.transform
    results = evaluate_model(model, eval_class, eval_root, eval_cats, args.device, transform)

    logger.info("\n" + "=" * 80)
    logger.info(f"RESULTS ({args.mode}, downsample={args.downsample}):")
    logger.info(f"  Image AUROC: {results['AUROC']:.1f}%")
    logger.info(f"  Image AP:    {results['AP']:.1f}%")
    logger.info(f"  Pixel AUROC: {results['pAUROC']:.1f}%")
    logger.info("=" * 80)

    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
