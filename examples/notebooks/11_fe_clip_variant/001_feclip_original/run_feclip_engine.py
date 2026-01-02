"""FE-CLIP Benchmark using Anomalib Engine.

This script implements the FE-CLIP paper's training protocol using anomalib Engine:
- Fine-tune on auxiliary dataset's TEST data (all categories)
- Evaluate on target dataset
- TensorBoard logging for training analysis
- Automatic checkpointing and visualization

Paper Protocol:
- MVTec AD evaluation: Fine-tune on VisA test data -> Test on MVTec AD
- BTAD evaluation: Fine-tune on MVTec AD test data -> Test on BTAD
- VisA evaluation: Fine-tune on MVTec AD test data -> Test on VisA

Usage:
    python run_feclip_engine.py --mode mvtec --seed 42
    python run_feclip_engine.py --mode btad --seed 42 --visualize
    python run_feclip_engine.py --mode visa --seed 42
"""

import argparse
import logging
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from anomalib.data import MVTecAD, BTech, Visa
from anomalib.data.dataclasses.torch.image import ImageBatch
from anomalib.models.image import FEClip
from anomalib.engine import Engine

from sklearn.metrics import roc_auc_score, average_precision_score
from torchvision.transforms.v2 import Resize, CenterCrop, InterpolationMode

from anomalib.metrics.aupro import _AUPRO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
MVTEC_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/MVTecAD")
BTAD_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/BTech")
VISA_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/VisA")

# Categories
MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]
VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
]
BTAD_CATEGORIES = ["01", "02", "03"]

# Image size
IMAGE_SIZE = 336


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CLIPStyleDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper to apply CLIP-style preprocessing (Resize shorter edge + CenterCrop)."""

    def __init__(self, dataset, image_size=IMAGE_SIZE):
        self.dataset = dataset
        self.image_size = image_size
        self.resize = Resize(image_size, antialias=True, interpolation=InterpolationMode.BICUBIC)
        self.crop = CenterCrop((image_size, image_size))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Apply CLIP-style transform
        image = self.crop(self.resize(item.image))
        gt_mask = None
        if item.gt_mask is not None:
            mask_resize = Resize(self.image_size, antialias=False, interpolation=InterpolationMode.NEAREST)
            gt_mask = self.crop(mask_resize(item.gt_mask))

        from anomalib.data.dataclasses.torch.image import ImageItem
        return ImageItem(
            image=image,
            gt_label=item.gt_label,
            gt_mask=gt_mask,
            image_path=item.image_path,
            mask_path=item.mask_path,
        )


def collect_test_data(dataset_class, root, categories):
    """Collect test data from all categories."""
    all_datasets = []
    for cat in categories:
        dm = dataset_class(root=root, category=cat, eval_batch_size=16, num_workers=4)
        dm.setup()
        all_datasets.append(dm.test_data)
        logger.info(f"  {cat}: {len(dm.test_data)} samples")

    combined = ConcatDataset(all_datasets)
    logger.info(f"  Total: {len(combined)} samples")
    return CLIPStyleDatasetWrapper(combined)


def train_feclip(
    train_dataset,
    exp_dir: Path,
    epochs: int = 9,
    batch_size: int = 16,
    lr: float = 5e-4,
    device: str = "cuda:0",
    tap_indices: list[int] | None = None,
    freeze_fc_patch: bool = False,
    use_clip_logit_scale: bool = False,
    lfs_agg_mode: str = "mean",
):
    """Train FE-CLIP model using manual training loop with TensorBoard logging."""
    from torch.utils.tensorboard import SummaryWriter
    from torchvision.transforms.v2 import Normalize
    from anomalib.models.image.feclip.losses import bce_loss, focal_loss, dice_loss

    # Create model
    model = FEClip(
        lr=lr, w_cls=1.0, w_mask=1.0, tap_indices=tap_indices,
        freeze_fc_patch=freeze_fc_patch, use_clip_logit_scale=use_clip_logit_scale,
        lfs_agg_mode=lfs_agg_mode
    )
    model.to(device)
    model.model.setup_text()
    model.train()

    # Setup TensorBoard
    tb_dir = exp_dir / "tensorboard_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    # Checkpoint directory
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=ImageBatch.collate,
    )

    # CLIP normalization
    normalize = Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )

    # Optimizer (Adam per paper)
    params = [p for p in model.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    logger.info(f"Training: {len(train_dataset)} samples, {len(train_loader)} batches/epoch")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in params):,}")
    logger.info(f"Tap blocks: {model.model.tap_blocks}")
    logger.info(f"freeze_fc_patch: {freeze_fc_patch}, use_clip_logit_scale: {use_clip_logit_scale}, lfs_agg_mode: {lfs_agg_mode}")

    # Save hyperparameters
    hparams = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "w_cls": 1.0,
        "w_mask": 1.0,
        "optimizer": "Adam",
        "n_samples": len(train_dataset),
        "tap_indices": model.model.tap_blocks,
        "freeze_fc_patch": freeze_fc_patch,
        "use_clip_logit_scale": use_clip_logit_scale,
        "lfs_agg_mode": lfs_agg_mode,
    }
    with open(tb_dir / "hparams.yaml", "w") as f:
        import yaml
        yaml.dump(hparams, f)

    global_step = 0
    best_loss = float('inf')

    for epoch in range(epochs):
        epoch_losses = {"loss_total": [], "loss_cls": [], "loss_mask": []}

        for batch_idx, batch in enumerate(train_loader):
            images = normalize(batch.image).to(device)
            gt_label = batch.gt_label.float().to(device) if batch.gt_label is not None else None
            gt_mask = batch.gt_mask.float().to(device) if batch.gt_mask is not None else None

            if gt_mask is not None and gt_mask.ndim == 4:
                gt_mask = gt_mask.squeeze(1)

            # Forward
            scores, maps = model.model.forward_tokens(images)
            n_taps = len(scores)

            # Per-tap classification loss
            loss_cls = torch.tensor(0.0, device=device)
            if gt_label is not None:
                for s in scores:
                    loss_cls = loss_cls + bce_loss(s, gt_label)
                loss_cls = loss_cls / n_taps

            # Per-tap mask loss
            loss_mask = torch.tensor(0.0, device=device)
            if gt_mask is not None:
                for m in maps:
                    m_up = torch.nn.functional.interpolate(
                        m.unsqueeze(1), size=images.shape[-2:],
                        mode="bilinear", align_corners=False
                    ).squeeze(1)
                    loss_mask = loss_mask + focal_loss(m_up, gt_mask) + dice_loss(m_up, gt_mask)
                loss_mask = loss_mask / n_taps

            # Total loss
            total_loss = loss_cls + loss_mask

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Log to TensorBoard
            writer.add_scalar("train/loss_total", total_loss.item(), global_step)
            writer.add_scalar("train/loss_cls", loss_cls.item(), global_step)
            writer.add_scalar("train/loss_mask", loss_mask.item(), global_step)

            epoch_losses["loss_total"].append(total_loss.item())
            epoch_losses["loss_cls"].append(loss_cls.item())
            epoch_losses["loss_mask"].append(loss_mask.item())

            global_step += 1

        # Epoch summary
        avg_loss = np.mean(epoch_losses["loss_total"])
        avg_cls = np.mean(epoch_losses["loss_cls"])
        avg_mask = np.mean(epoch_losses["loss_mask"])

        writer.add_scalar("epoch/loss_total", avg_loss, epoch)
        writer.add_scalar("epoch/loss_cls", avg_cls, epoch)
        writer.add_scalar("epoch/loss_mask", avg_mask, epoch)

        logger.info(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} (cls={avg_cls:.4f}, mask={avg_mask:.4f})")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_dir / "best_model.pt")

        # Save last checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, ckpt_dir / "last_model.pt")

    writer.close()
    logger.info(f"Training complete. Best loss: {best_loss:.4f}")
    return model


def compute_pixel_auroc(all_masks_pred: list, all_masks_gt: list) -> float:
    """Compute pixel-level AUROC.

    Args:
        all_masks_pred: List of predicted anomaly maps (H, W)
        all_masks_gt: List of ground truth masks (H, W)

    Returns:
        Pixel-level AUROC score
    """
    preds_flat = []
    targets_flat = []

    for pred, gt in zip(all_masks_pred, all_masks_gt):
        if gt is None or gt.sum() == 0:
            continue
        preds_flat.append(pred.flatten())
        targets_flat.append(gt.flatten())

    if len(preds_flat) == 0:
        return 0.5

    preds = np.concatenate(preds_flat)
    targets = np.concatenate(targets_flat)

    if len(np.unique(targets)) < 2:
        return 0.5

    return roc_auc_score(targets, preds)


def compute_pro_score(all_masks_pred: list, all_masks_gt: list, device: str = "cuda") -> float:
    """Compute PRO (Per-Region Overlap) score using AUPRO metric on GPU.

    Args:
        all_masks_pred: List of predicted anomaly maps (H, W)
        all_masks_gt: List of ground truth masks (H, W)
        device: Device for computation (default: cuda for GPU acceleration)

    Returns:
        PRO score (AUPRO with fpr_limit=0.3)
    """
    pro_metric = _AUPRO(fpr_limit=0.3)
    n_valid = 0

    for pred, gt in zip(all_masks_pred, all_masks_gt):
        if gt is None:
            continue
        # Ensure proper tensor format - pred should be float, target should be int/long
        # Move to GPU for faster connected components analysis
        pred_tensor = torch.tensor(pred).float().to(device) if not isinstance(pred, torch.Tensor) else pred.float().to(device)
        gt_tensor = torch.tensor(gt).long().to(device) if not isinstance(gt, torch.Tensor) else gt.long().to(device)

        # Skip if no anomaly in ground truth
        if gt_tensor.sum() == 0:
            continue

        # Normalize pred to [0, 1] if needed
        if pred_tensor.max() > 1.0:
            pred_tensor = pred_tensor / pred_tensor.max()

        pro_metric.update(pred_tensor.unsqueeze(0), gt_tensor.unsqueeze(0))
        n_valid += 1

    if n_valid == 0:
        return 0.0

    try:
        return pro_metric.compute().item()
    except Exception as e:
        logger.warning(f"PRO computation failed: {e}")
        return 0.0


def evaluate_and_visualize(
    model,
    dataset_class,
    root,
    categories,
    exp_dir: Path,
    dataset_name: str,
    device: str,
    visualize: bool = True,
    n_vis: int = 10,
):
    """Evaluate model and optionally save visualizations.

    Args:
        model: FE-CLIP model
        dataset_class: Dataset class (MVTecAD, BTech, Visa)
        root: Dataset root path
        categories: List of categories to evaluate
        exp_dir: Experiment directory
        dataset_name: Name of the dataset for logging
        device: Device to use
        visualize: Whether to save visualizations
        n_vis: Number of samples to visualize per class (normal/anomaly), default 10
    """
    import matplotlib.pyplot as plt

    model.eval()
    transform = model.pre_processor.transform

    # CLIP-style mask transform: Resize shorter edge + CenterCrop (same as image)
    # This ensures GT masks are properly aligned with preprocessed images
    mask_resize = Resize(336, interpolation=InterpolationMode.NEAREST, antialias=False)
    mask_crop = CenterCrop((336, 336))

    def preprocess_gt_mask(mask_tensor):
        """Apply CLIP-style preprocessing to GT mask."""
        if mask_tensor is None:
            return None
        # Ensure 3D tensor (C, H, W)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        # Apply same Resize + CenterCrop as image
        mask_processed = mask_crop(mask_resize(mask_tensor))
        return mask_processed.squeeze(0)  # Return (H, W)

    vis_dir = exp_dir / "visualizations" / dataset_name
    results = {"categories": {}}
    aurocs = []
    aps = []
    pixel_aurocs = []
    pros = []

    logger.info(f"\nEvaluating on {dataset_name}...")
    logger.info(f"{'Category':<15} {'AUROC':>8} {'AP':>8} {'pAUROC':>8} {'PRO':>8} {'N':>6}")
    logger.info("-" * 60)

    for category in categories:
        dm = dataset_class(root=root, category=category, eval_batch_size=16, num_workers=8)
        dm.setup()

        all_scores = []
        all_labels = []
        all_masks_pred = []
        all_masks_gt = []
        n_normal_vis, n_anomaly_vis = 0, 0

        with torch.no_grad():
            for batch in dm.test_dataloader():
                images = transform(batch.image).to(device)
                out = model.model(images)
                all_scores.append(out.pred_score.cpu())
                all_labels.append(batch.gt_label.cpu())

                # Collect pixel-level predictions and ground truth
                for i in range(len(images)):
                    amap = out.anomaly_map[i].cpu().numpy()
                    all_masks_pred.append(amap)

                    gt_mask = None
                    if batch.gt_mask is not None:
                        # Apply CLIP-style preprocessing to GT mask (Resize + CenterCrop)
                        gt_mask_tensor = batch.gt_mask[i]
                        gt_mask_processed = preprocess_gt_mask(gt_mask_tensor)
                        gt_mask = gt_mask_processed.cpu().numpy()

                        # Resize to match anomaly map size if still different
                        if gt_mask.shape != amap.shape:
                            gt_mask = torch.nn.functional.interpolate(
                                torch.tensor(gt_mask).unsqueeze(0).unsqueeze(0).float(),
                                size=amap.shape,
                                mode='nearest'
                            ).squeeze().numpy()
                    all_masks_gt.append(gt_mask)

                # Save visualizations
                if visualize:
                    for i in range(len(images)):
                        gt_label = batch.gt_label[i].item()
                        if gt_label == 0 and n_normal_vis >= n_vis:
                            continue
                        if gt_label == 1 and n_anomaly_vis >= n_vis:
                            continue

                        # Apply CLIP-style preprocessing to GT mask for visualization
                        gt_mask_vis = None
                        if batch.gt_mask is not None:
                            gt_mask_vis = preprocess_gt_mask(batch.gt_mask[i])

                        save_visualization(
                            images[i], out.anomaly_map[i],
                            gt_mask_vis,
                            out.pred_score[i].item(), gt_label,
                            vis_dir / category,
                            f"{'anomaly' if gt_label else 'normal'}_{n_anomaly_vis if gt_label else n_normal_vis}.png"
                        )

                        if gt_label == 0:
                            n_normal_vis += 1
                        else:
                            n_anomaly_vis += 1

        scores = torch.cat(all_scores).numpy()
        labels = torch.cat(all_labels).numpy()

        # Image-level metrics
        auroc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.5
        ap = average_precision_score(labels, scores) if len(np.unique(labels)) > 1 else 0.5

        # Pixel-level metrics
        pixel_auroc = compute_pixel_auroc(all_masks_pred, all_masks_gt)
        pro = compute_pro_score(all_masks_pred, all_masks_gt, device)

        aurocs.append(auroc)
        aps.append(ap)
        pixel_aurocs.append(pixel_auroc)
        pros.append(pro)

        results["categories"][category] = {
            "auroc": auroc, "ap": ap,
            "pixel_auroc": pixel_auroc, "pro": pro,
            "n_samples": len(labels)
        }
        logger.info(f"{category:<15} {auroc*100:>7.1f}% {ap*100:>7.1f}% {pixel_auroc*100:>7.1f}% {pro*100:>7.1f}% {len(labels):>6}")

    mean_auroc = np.mean(aurocs)
    mean_ap = np.mean(aps)
    mean_pixel_auroc = np.mean(pixel_aurocs)
    mean_pro = np.mean(pros)

    results["mean_auroc"] = mean_auroc
    results["mean_ap"] = mean_ap
    results["mean_pixel_auroc"] = mean_pixel_auroc
    results["mean_pro"] = mean_pro

    logger.info("-" * 60)
    logger.info(f"{'Mean':<15} {mean_auroc*100:>7.1f}% {mean_ap*100:>7.1f}% {mean_pixel_auroc*100:>7.1f}% {mean_pro*100:>7.1f}%")

    return results


def save_visualization(image, anomaly_map, gt_mask, pred_score, gt_label, save_dir, filename):
    """Save heatmap visualization."""
    import matplotlib.pyplot as plt

    # Denormalize
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    img = image.cpu() * std + mean
    img = img.permute(1, 2, 0).numpy().clip(0, 1)

    amap = anomaly_map.cpu().numpy()

    # Resize GT mask if needed
    if gt_mask is not None:
        gt_mask = gt_mask.cpu() if isinstance(gt_mask, torch.Tensor) else torch.tensor(gt_mask)
        if gt_mask.ndim == 3:
            gt_mask = gt_mask.squeeze(0)
        if gt_mask.shape[-2:] != image.shape[-2:]:
            gt_mask = torch.nn.functional.interpolate(
                gt_mask.unsqueeze(0).unsqueeze(0).float(),
                size=image.shape[-2:],
                mode='nearest'
            ).squeeze(0).squeeze(0)

    n_cols = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    axes[0].imshow(img)
    axes[0].set_title(f"Input (GT: {'Anomaly' if gt_label else 'Normal'})")
    axes[0].axis('off')

    axes[1].imshow(img)
    heatmap = axes[1].imshow(amap, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title(f"Prediction (Score: {pred_score:.3f})")
    axes[1].axis('off')
    plt.colorbar(heatmap, ax=axes[1], fraction=0.046)

    if gt_mask is not None and n_cols == 3:
        gt = gt_mask.numpy() if isinstance(gt_mask, torch.Tensor) else gt_mask
        axes[2].imshow(img)
        axes[2].imshow(gt, cmap='Reds', alpha=0.5)
        axes[2].set_title("Ground Truth Mask")
        axes[2].axis('off')

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="FE-CLIP Benchmark with Engine")
    parser.add_argument("--mode", type=str, choices=["mvtec", "btad", "visa"], required=True)
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize", action="store_true", help="Save visualizations")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--tap_indices", type=str, default=None,
                        help="Tap block indices (comma-separated, e.g., '20,21,22,23')")
    parser.add_argument("--freeze_fc_patch", action="store_true",
                        help="Freeze fc_patch (only train adapters)")
    parser.add_argument("--use_clip_logit_scale", action="store_true",
                        help="Use CLIP's learned logit_scale instead of fixed temperature")
    parser.add_argument("--lfs_agg_mode", type=str, default="mean", choices=["mean", "abs", "power"],
                        help="LFS aggregation mode: mean (signed), abs, power")
    parser.add_argument("--n_vis", type=int, default=10,
                        help="Number of samples to visualize per class (normal/anomaly)")
    args = parser.parse_args()

    # Parse tap_indices
    tap_indices = None
    if args.tap_indices:
        tap_indices = [int(x.strip()) for x in args.tap_indices.split(",")]

    # Set seed
    set_seed(args.seed)

    # Device
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name or f"feclip_{args.mode}"
    exp_dir = RESULTS_DIR / f"{exp_name}_seed{args.seed}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging to experiment directory
    file_handler = logging.FileHandler(exp_dir / "train.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Mode: {args.mode}, Seed: {args.seed}, Device: {device}")
    if tap_indices:
        logger.info(f"Custom tap indices: {tap_indices}")

    # Define training and testing datasets based on mode
    if args.mode == "mvtec":
        # VisA (all) -> MVTec AD
        logger.info("\n" + "=" * 60)
        logger.info("Training on VisA (all categories)")
        logger.info("=" * 60)
        train_dataset = collect_test_data(Visa, VISA_ROOT, VISA_CATEGORIES)
        test_class, test_root, test_categories = MVTecAD, MVTEC_ROOT, MVTEC_CATEGORIES
        paper_auroc, paper_ap = 91.9, 96.5

    elif args.mode == "btad":
        # MVTec AD (all) -> BTAD
        logger.info("\n" + "=" * 60)
        logger.info("Training on MVTec AD (all categories)")
        logger.info("=" * 60)
        train_dataset = collect_test_data(MVTecAD, MVTEC_ROOT, MVTEC_CATEGORIES)
        test_class, test_root, test_categories = BTech, BTAD_ROOT, BTAD_CATEGORIES
        paper_auroc, paper_ap = 90.3, 90.0

    elif args.mode == "visa":
        # MVTec AD (all) -> VisA
        logger.info("\n" + "=" * 60)
        logger.info("Training on MVTec AD (all categories)")
        logger.info("=" * 60)
        train_dataset = collect_test_data(MVTecAD, MVTEC_ROOT, MVTEC_CATEGORIES)
        test_class, test_root, test_categories = Visa, VISA_ROOT, VISA_CATEGORIES
        paper_auroc, paper_ap = 84.6, 86.6

    # Train
    model = train_feclip(
        train_dataset=train_dataset,
        exp_dir=exp_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        tap_indices=tap_indices,
        freeze_fc_patch=args.freeze_fc_patch,
        use_clip_logit_scale=args.use_clip_logit_scale,
        lfs_agg_mode=args.lfs_agg_mode,
    )

    # Evaluate
    results = evaluate_and_visualize(
        model=model,
        dataset_class=test_class,
        root=test_root,
        categories=test_categories,
        exp_dir=exp_dir,
        dataset_name=args.mode,
        device=device,
        visualize=args.visualize,
        n_vis=args.n_vis,
    )

    # Paper reference values (segmentation: pAUROC/PRO)
    paper_seg = {
        "mvtec": {"pixel_auroc": 92.6, "pro": 88.3},
        "visa": {"pixel_auroc": 95.9, "pro": 92.8},
        "btad": {"pixel_auroc": 95.6, "pro": 80.4},
    }

    # Add paper comparison
    results["paper_auroc"] = paper_auroc
    results["paper_ap"] = paper_ap
    results["gap_auroc"] = results["mean_auroc"] * 100 - paper_auroc
    results["paper_pixel_auroc"] = paper_seg[args.mode]["pixel_auroc"]
    results["paper_pro"] = paper_seg[args.mode]["pro"]
    results["gap_pixel_auroc"] = results["mean_pixel_auroc"] * 100 - paper_seg[args.mode]["pixel_auroc"]
    results["gap_pro"] = results["mean_pro"] * 100 - paper_seg[args.mode]["pro"]

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info("Image-level Detection (ZSAD):")
    logger.info(f"  Ours:  AUROC={results['mean_auroc']*100:.1f}%, AP={results['mean_ap']*100:.1f}%")
    logger.info(f"  Paper: AUROC={paper_auroc}%, AP={paper_ap}%")
    logger.info(f"  Gap:   {results['gap_auroc']:+.1f}%")
    logger.info("")
    logger.info("Pixel-level Segmentation (ZSAS):")
    logger.info(f"  Ours:  pAUROC={results['mean_pixel_auroc']*100:.1f}%, PRO={results['mean_pro']*100:.1f}%")
    logger.info(f"  Paper: pAUROC={paper_seg[args.mode]['pixel_auroc']}%, PRO={paper_seg[args.mode]['pro']}%")
    logger.info(f"  Gap:   pAUROC {results['gap_pixel_auroc']:+.1f}%, PRO {results['gap_pro']:+.1f}%")

    # Save results
    results["args"] = vars(args)
    results["timestamp"] = timestamp
    with open(exp_dir / f"results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {exp_dir}")
    logger.info(f"TensorBoard: tensorboard --logdir {exp_dir / 'tensorboard_logs'}")


if __name__ == "__main__":
    main()
