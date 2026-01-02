"""FE-CLIP Benchmark with ALL categories (following exact paper methodology).

Paper Section 4.2:
- "we fine-tune FE-CLIP using the test data of MVTec AD" (ALL 15 categories)
- "As for MVTec AD, we fine-tune FE-CLIP on the test data of VisA" (ALL 12 categories)
- 9 epochs, lr=5e-4, batch_size=16
"""

import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from torchvision.transforms.v2 import Resize, CenterCrop, InterpolationMode, Compose

from anomalib.data import MVTecAD, BTech, Visa
from anomalib.data.dataclasses.torch.image import ImageBatch, ImageItem
from anomalib.models.image import FEClip
from anomalib.models.image.feclip.losses import focal_loss, dice_loss, bce_loss
from anomalib.metrics.aupro import _AUPRO

# Image size for FE-CLIP
IMAGE_SIZE = 336


class ResizedDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper to resize images using CLIP-style preprocessing (Resize shorter edge + CenterCrop).

    This matches CLIP's default transform which preserves aspect ratio:
    1. Resize shorter edge to image_size (maintains aspect ratio)
    2. CenterCrop to (image_size, image_size)
    """

    def __init__(self, dataset, image_size=IMAGE_SIZE):
        self.dataset = dataset
        self.image_size = image_size
        # CLIP-style: Resize shorter edge, then center crop (preserves aspect ratio)
        self.resize = Resize(image_size, antialias=True, interpolation=InterpolationMode.BICUBIC)
        self.crop = CenterCrop((image_size, image_size))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Apply CLIP-style transform: resize shorter edge + center crop
        image = self.crop(self.resize(item.image))
        gt_mask = None
        if item.gt_mask is not None:
            # Same transform for mask (but with nearest interpolation)
            mask_resize = Resize(self.image_size, antialias=False, interpolation=InterpolationMode.NEAREST)
            gt_mask = self.crop(mask_resize(item.gt_mask))
        return ImageItem(
            image=image,
            gt_label=item.gt_label,
            gt_mask=gt_mask,
            image_path=item.image_path,
            mask_path=item.mask_path,
        )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
MVTEC_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/MVTecAD")
BTAD_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/BTech")
VISA_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/VisA")

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

# FE-CLIP settings (from paper - exact reproduction)
# Paper: Adam optimizer, lr=5e-4, epochs=9, batch_size=16
FECLIP_CONFIG = {
    "backbone": "ViT-L-14-336",
    "pretrained": "openai",
    "n_taps": 4,
    "lambda_fuse": 0.1,
    "P": 3,
    "Q": 3,
    "lr": 5e-4,
    "w_cls": 1.0,  # Paper: L_total = L_cls + L_mask
    "w_mask": 1.0,
}


def collect_all_test_data(dataset_class, root, categories, batch_size=16):
    """Collect test data from all categories into a single dataset."""
    all_test_datasets = []

    for cat in categories:
        dm = dataset_class(root=root, category=cat, eval_batch_size=batch_size, num_workers=4)
        dm.setup()
        all_test_datasets.append(dm.test_data)
        logger.info(f"  {cat}: {len(dm.test_data)} test samples")

    combined = ConcatDataset(all_test_datasets)
    logger.info(f"  Total: {len(combined)} samples")

    # Wrap with resize to ensure consistent image sizes for batching
    wrapped = ResizedDatasetWrapper(combined)
    return wrapped


def manual_train_loop(model, train_dataset, epochs=9, batch_size=16, lr=5e-4, device="cuda"):
    """Manual training loop (more control than Engine)."""
    from torchvision.transforms.v2 import Normalize

    model.to(device)
    model.model.setup_text()
    model.train()

    # DataLoader with proper collate function for anomalib
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=ImageBatch.collate,
    )

    # CLIP normalization (images are already resized in ResizedDatasetWrapper)
    normalize = Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )

    # Optimizer (only trainable params: adapters + fc_patch)
    # Paper uses Adam (not AdamW), no weight_decay mentioned
    params = [p for p in model.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    logger.info(f"Training: {len(train_dataset)} samples, {len(train_loader)} batches/epoch")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in params):,}")

    for epoch in range(epochs):
        epoch_losses = []

        for batch_idx, batch in enumerate(train_loader):
            # Apply CLIP normalization (resize already done in ResizedDatasetWrapper)
            images = normalize(batch.image).to(device)
            gt_label = batch.gt_label.float().to(device) if batch.gt_label is not None else None
            gt_mask = batch.gt_mask.float().to(device) if batch.gt_mask is not None else None

            if gt_mask is not None and gt_mask.ndim == 4:
                gt_mask = gt_mask.squeeze(1)

            # Forward
            scores, maps = model.model.forward_tokens(images)
            n_taps = len(scores)

            # Per-tap losses (Eq.4 & Eq.5 in paper)
            loss_cls = torch.tensor(0.0, device=device)
            if gt_label is not None:
                for s in scores:
                    loss_cls = loss_cls + bce_loss(s, gt_label)
                loss_cls = loss_cls / n_taps

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
            total_loss = FECLIP_CONFIG["w_cls"] * loss_cls + FECLIP_CONFIG["w_mask"] * loss_mask

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_losses.append(total_loss.item())

        avg_loss = np.mean(epoch_losses)
        logger.info(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    return model


def save_heatmap_visualization(image, anomaly_map, gt_mask, pred_score, gt_label, save_path, image_path=None):
    """Save heatmap visualization overlay.

    Args:
        image: Original image tensor (C, H, W) normalized
        anomaly_map: Anomaly map tensor (H, W)
        gt_mask: Ground truth mask tensor (H, W) or None
        pred_score: Predicted anomaly score
        gt_label: Ground truth label (0=normal, 1=anomaly)
        save_path: Path to save the visualization
        image_path: Original image path for title
    """
    # Denormalize image (CLIP normalization)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    img = image.cpu() * std + mean
    img = img.permute(1, 2, 0).numpy().clip(0, 1)

    # Anomaly map to numpy
    amap = anomaly_map.cpu().numpy()

    # Resize GT mask to match image size if needed
    if gt_mask is not None:
        gt_mask = gt_mask.cpu() if isinstance(gt_mask, torch.Tensor) else torch.tensor(gt_mask)
        if gt_mask.shape[-2:] != image.shape[-2:]:
            gt_mask = torch.nn.functional.interpolate(
                gt_mask.unsqueeze(0).unsqueeze(0).float(),
                size=image.shape[-2:],
                mode='nearest'
            ).squeeze(0).squeeze(0)

    # Create figure
    n_cols = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    # Original image
    axes[0].imshow(img)
    axes[0].set_title(f"Input (GT: {'Anomaly' if gt_label else 'Normal'})")
    axes[0].axis('off')

    # Heatmap overlay
    axes[1].imshow(img)
    heatmap = axes[1].imshow(amap, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title(f"Prediction (Score: {pred_score:.3f})")
    axes[1].axis('off')
    plt.colorbar(heatmap, ax=axes[1], fraction=0.046)

    # Ground truth mask (if available)
    if gt_mask is not None and n_cols == 3:
        gt = gt_mask.numpy() if isinstance(gt_mask, torch.Tensor) else gt_mask
        axes[2].imshow(img)
        axes[2].imshow(gt, cmap='Reds', alpha=0.5)
        axes[2].set_title("Ground Truth Mask")
        axes[2].axis('off')

    # Add image path as suptitle if available
    if image_path:
        fig.suptitle(str(image_path).split('/')[-1], fontsize=10)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_pixel_auroc(all_masks_pred: list, all_masks_gt: list) -> float:
    """Compute pixel-level AUROC."""
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
    """Compute PRO (Per-Region Overlap) score using AUPRO metric on GPU."""
    pro_metric = _AUPRO(fpr_limit=0.3)
    n_valid = 0

    for pred, gt in zip(all_masks_pred, all_masks_gt):
        if gt is None:
            continue
        # pred should be float, target should be int/long
        # Move to GPU for faster connected components analysis
        pred_tensor = torch.tensor(pred).float().to(device) if not isinstance(pred, torch.Tensor) else pred.float().to(device)
        gt_tensor = torch.tensor(gt).long().to(device) if not isinstance(gt, torch.Tensor) else gt.long().to(device)

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
    except Exception:
        return 0.0


def evaluate_dataset(model, datamodule, device, transform, visualize=False, vis_dir=None, category=None, n_vis=10):
    """Evaluate model on a datamodule.

    Args:
        model: FE-CLIP model
        datamodule: Anomalib datamodule
        device: Device to use
        transform: Image transform
        visualize: Whether to save visualizations
        vis_dir: Directory to save visualizations
        category: Category name for visualization folder
        n_vis: Number of samples to visualize per class (normal/anomaly), default 10
    """
    datamodule.setup()

    # CLIP-style mask transform: Resize shorter edge + CenterCrop (same as image)
    mask_resize = Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST, antialias=False)
    mask_crop = CenterCrop((IMAGE_SIZE, IMAGE_SIZE))

    def preprocess_gt_mask(mask_tensor):
        """Apply CLIP-style preprocessing to GT mask."""
        if mask_tensor is None:
            return None
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        mask_processed = mask_crop(mask_resize(mask_tensor))
        return mask_processed.squeeze(0)

    all_scores = []
    all_labels = []
    all_masks_pred = []
    all_masks_gt = []

    # For visualization: track how many we've saved
    n_normal_saved = 0
    n_anomaly_saved = 0

    model.eval()
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
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
                    gt_mask_processed = preprocess_gt_mask(batch.gt_mask[i])
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
            if visualize and vis_dir and category:
                for i in range(len(images)):
                    gt_label = batch.gt_label[i].item()

                    # Save n_vis normal and n_vis anomaly samples
                    if gt_label == 0 and n_normal_saved >= n_vis:
                        continue
                    if gt_label == 1 and n_anomaly_saved >= n_vis:
                        continue

                    label_str = "anomaly" if gt_label == 1 else "normal"
                    idx = n_anomaly_saved if gt_label == 1 else n_normal_saved
                    save_path = vis_dir / category / f"{label_str}_{idx:03d}.png"

                    # Apply CLIP-style preprocessing to GT mask for visualization
                    gt_mask_vis = None
                    if batch.gt_mask is not None:
                        gt_mask_vis = preprocess_gt_mask(batch.gt_mask[i])

                    image_path = batch.image_path[i] if hasattr(batch, 'image_path') and batch.image_path else None

                    save_heatmap_visualization(
                        image=images[i],
                        anomaly_map=out.anomaly_map[i],
                        gt_mask=gt_mask_vis,
                        pred_score=out.pred_score[i].item(),
                        gt_label=gt_label,
                        save_path=save_path,
                        image_path=image_path,
                    )

                    if gt_label == 0:
                        n_normal_saved += 1
                    else:
                        n_anomaly_saved += 1

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()

    auroc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    pixel_auroc = compute_pixel_auroc(all_masks_pred, all_masks_gt)
    pro = compute_pro_score(all_masks_pred, all_masks_gt, device=device)

    return auroc, ap, pixel_auroc, pro, len(labels)


def finetune_on_visa_all(epochs: int = 9, batch_size: int = 16, device: str = "cuda"):
    """Finetune FE-CLIP on ALL VisA categories' test data."""
    logger.info("=" * 70)
    logger.info("Finetuning FE-CLIP on VisA (ALL 12 categories)")
    logger.info("=" * 70)

    # Collect all VisA test data
    logger.info("Collecting VisA test data...")
    train_dataset = collect_all_test_data(Visa, VISA_ROOT, VISA_CATEGORIES, batch_size)

    # Create model
    model = FEClip(**FECLIP_CONFIG)

    # Train
    model = manual_train_loop(model, train_dataset, epochs=epochs, batch_size=batch_size, device=device)

    logger.info("VisA (all categories) finetuning complete!")
    return model


def finetune_on_mvtec_all(epochs: int = 9, batch_size: int = 16, device: str = "cuda"):
    """Finetune FE-CLIP on ALL MVTec AD categories' test data."""
    logger.info("=" * 70)
    logger.info("Finetuning FE-CLIP on MVTec AD (ALL 15 categories)")
    logger.info("=" * 70)

    # Collect all MVTec test data
    logger.info("Collecting MVTec AD test data...")
    train_dataset = collect_all_test_data(MVTecAD, MVTEC_ROOT, MVTEC_CATEGORIES, batch_size)

    # Create model
    model = FEClip(**FECLIP_CONFIG)

    # Train
    model = manual_train_loop(model, train_dataset, epochs=epochs, batch_size=batch_size, device=device)

    logger.info("MVTec AD (all categories) finetuning complete!")
    return model


def test_on_mvtec(model, device, transform, visualize=False, vis_dir=None, n_vis=10):
    """Test model on all MVTec AD categories."""
    logger.info("\n" + "-" * 80)
    logger.info("Testing on MVTec AD (all categories)")
    logger.info("-" * 80)
    logger.info(f"{'Category':<15} {'AUROC':>8} {'AP':>8} {'pAUROC':>8} {'PRO':>8} {'N':>6}")
    logger.info("-" * 80)

    aurocs = []
    aps = []
    pixel_aurocs = []
    pros = []

    for category in MVTEC_CATEGORIES:
        datamodule = MVTecAD(
            root=MVTEC_ROOT,
            category=category,
            eval_batch_size=16,
            num_workers=8,
        )
        auroc, ap, pixel_auroc, pro, n = evaluate_dataset(
            model, datamodule, device, transform,
            visualize=visualize, vis_dir=vis_dir, category=category, n_vis=n_vis
        )
        aurocs.append(auroc)
        aps.append(ap)
        pixel_aurocs.append(pixel_auroc)
        pros.append(pro)
        logger.info(f"{category:<15} {auroc*100:>7.1f}% {ap*100:>7.1f}% {pixel_auroc*100:>7.1f}% {pro*100:>7.1f}% {n:>6}")

    mean_auroc = np.mean(aurocs)
    mean_ap = np.mean(aps)
    mean_pixel_auroc = np.mean(pixel_aurocs)
    mean_pro = np.mean(pros)

    logger.info("-" * 80)
    logger.info(f"{'Mean':<15} {mean_auroc*100:>7.1f}% {mean_ap*100:>7.1f}% {mean_pixel_auroc*100:>7.1f}% {mean_pro*100:>7.1f}%")
    logger.info(f"{'Paper':<15} {'91.9%':>8} {'96.5%':>8} {'92.6%':>8} {'88.3%':>8}")
    logger.info(f"{'Gap':<15} {(mean_auroc*100-91.9):>+7.1f}% {' ':>8} {(mean_pixel_auroc*100-92.6):>+7.1f}% {(mean_pro*100-88.3):>+7.1f}%")

    return mean_auroc, mean_ap, mean_pixel_auroc, mean_pro


def test_on_btad(model, device, transform, visualize=False, vis_dir=None, n_vis=10):
    """Test model on all BTAD categories."""
    logger.info("\n" + "-" * 80)
    logger.info("Testing on BTAD (all categories)")
    logger.info("-" * 80)
    logger.info(f"{'Category':<15} {'AUROC':>8} {'AP':>8} {'pAUROC':>8} {'PRO':>8} {'N':>6}")
    logger.info("-" * 80)

    aurocs = []
    aps = []
    pixel_aurocs = []
    pros = []

    for category in BTAD_CATEGORIES:
        datamodule = BTech(
            root=BTAD_ROOT,
            category=category,
            eval_batch_size=16,
            num_workers=8,
        )
        auroc, ap, pixel_auroc, pro, n = evaluate_dataset(
            model, datamodule, device, transform,
            visualize=visualize, vis_dir=vis_dir, category=category, n_vis=n_vis
        )
        aurocs.append(auroc)
        aps.append(ap)
        pixel_aurocs.append(pixel_auroc)
        pros.append(pro)
        logger.info(f"{category:<15} {auroc*100:>7.1f}% {ap*100:>7.1f}% {pixel_auroc*100:>7.1f}% {pro*100:>7.1f}% {n:>6}")

    mean_auroc = np.mean(aurocs)
    mean_ap = np.mean(aps)
    mean_pixel_auroc = np.mean(pixel_aurocs)
    mean_pro = np.mean(pros)

    logger.info("-" * 80)
    logger.info(f"{'Mean':<15} {mean_auroc*100:>7.1f}% {mean_ap*100:>7.1f}% {mean_pixel_auroc*100:>7.1f}% {mean_pro*100:>7.1f}%")
    logger.info(f"{'Paper':<15} {'90.3%':>8} {'90.0%':>8} {'95.6%':>8} {'80.4%':>8}")
    logger.info(f"{'Gap':<15} {(mean_auroc*100-90.3):>+7.1f}% {' ':>8} {(mean_pixel_auroc*100-95.6):>+7.1f}% {(mean_pro*100-80.4):>+7.1f}%")

    return mean_auroc, mean_ap, mean_pixel_auroc, mean_pro


def test_on_visa(model, device, transform, visualize=False, vis_dir=None, n_vis=10):
    """Test model on all VisA categories."""
    logger.info("\n" + "-" * 80)
    logger.info("Testing on VisA (all categories)")
    logger.info("-" * 80)
    logger.info(f"{'Category':<15} {'AUROC':>8} {'AP':>8} {'pAUROC':>8} {'PRO':>8} {'N':>6}")
    logger.info("-" * 80)

    aurocs = []
    aps = []
    pixel_aurocs = []
    pros = []

    for category in VISA_CATEGORIES:
        datamodule = Visa(
            root=VISA_ROOT,
            category=category,
            eval_batch_size=16,
            num_workers=8,
        )
        auroc, ap, pixel_auroc, pro, n = evaluate_dataset(
            model, datamodule, device, transform,
            visualize=visualize, vis_dir=vis_dir, category=category, n_vis=n_vis
        )
        aurocs.append(auroc)
        aps.append(ap)
        pixel_aurocs.append(pixel_auroc)
        pros.append(pro)
        logger.info(f"{category:<15} {auroc*100:>7.1f}% {ap*100:>7.1f}% {pixel_auroc*100:>7.1f}% {pro*100:>7.1f}% {n:>6}")

    mean_auroc = np.mean(aurocs)
    mean_ap = np.mean(aps)
    mean_pixel_auroc = np.mean(pixel_aurocs)
    mean_pro = np.mean(pros)

    logger.info("-" * 80)
    logger.info(f"{'Mean':<15} {mean_auroc*100:>7.1f}% {mean_ap*100:>7.1f}% {mean_pixel_auroc*100:>7.1f}% {mean_pro*100:>7.1f}%")
    logger.info(f"{'Paper':<15} {'84.6%':>8} {'86.6%':>8} {'95.9%':>8} {'92.8%':>8}")
    logger.info(f"{'Gap':<15} {(mean_auroc*100-84.6):>+7.1f}% {' ':>8} {(mean_pixel_auroc*100-95.9):>+7.1f}% {(mean_pro*100-92.8):>+7.1f}%")

    return mean_auroc, mean_ap, mean_pixel_auroc, mean_pro


def set_seed(seed):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="FE-CLIP Benchmark (All Categories)")
    parser.add_argument("--mode", type=str, choices=["mvtec", "btad", "visa", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize", action="store_true", help="Save heatmap visualizations")
    parser.add_argument("--vis_dir", type=str, default="examples/notebooks/11_fe_clip_variant/results/feclip_vis", help="Directory for visualizations")
    parser.add_argument("--n_vis", type=int, default=10, help="Number of samples to visualize per class (normal/anomaly)")
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)
    logger.info(f"Random seed: {args.seed}")

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Setup visualization directory
    vis_dir = Path(args.vis_dir) / f"seed_{args.seed}" if args.visualize else None
    if args.visualize:
        logger.info(f"Saving visualizations to: {vis_dir}")

    results = {}

    if args.mode in ["mvtec", "all"]:
        logger.info("\n" + "=" * 70)
        logger.info("Benchmark 1: VisA (all) -> MVTec AD")
        logger.info("=" * 70)

        # Finetune on ALL VisA categories
        model_visa = finetune_on_visa_all(epochs=args.epochs, batch_size=args.batch_size, device=device)
        model_visa.to(device)
        model_visa.eval()
        transform = model_visa.pre_processor.transform

        # Test on MVTec AD
        mvtec_vis_dir = vis_dir / "mvtec" if vis_dir else None
        mvtec_auroc, mvtec_ap, mvtec_pauroc, mvtec_pro = test_on_mvtec(
            model_visa, device, transform, visualize=args.visualize, vis_dir=mvtec_vis_dir, n_vis=args.n_vis
        )
        results["MVTec AD"] = {"AUROC": mvtec_auroc, "AP": mvtec_ap, "pAUROC": mvtec_pauroc, "PRO": mvtec_pro}

    if args.mode in ["btad", "all"]:
        logger.info("\n" + "=" * 70)
        logger.info("Benchmark 2: MVTec AD (all) -> BTAD")
        logger.info("=" * 70)

        # Finetune on ALL MVTec AD categories
        model_mvtec = finetune_on_mvtec_all(epochs=args.epochs, batch_size=args.batch_size, device=device)
        model_mvtec.to(device)
        model_mvtec.eval()
        transform = model_mvtec.pre_processor.transform

        # Test on BTAD
        btad_vis_dir = vis_dir / "btad" if vis_dir else None
        btad_auroc, btad_ap, btad_pauroc, btad_pro = test_on_btad(
            model_mvtec, device, transform, visualize=args.visualize, vis_dir=btad_vis_dir, n_vis=args.n_vis
        )
        results["BTAD"] = {"AUROC": btad_auroc, "AP": btad_ap, "pAUROC": btad_pauroc, "PRO": btad_pro}

    if args.mode in ["visa"]:
        logger.info("\n" + "=" * 70)
        logger.info("Benchmark 3: MVTec AD (all) -> VisA")
        logger.info("=" * 70)

        # Finetune on ALL MVTec AD categories
        model_mvtec = finetune_on_mvtec_all(epochs=args.epochs, batch_size=args.batch_size, device=device)
        model_mvtec.to(device)
        model_mvtec.eval()
        transform = model_mvtec.pre_processor.transform

        # Test on VisA
        visa_vis_dir = vis_dir / "visa" if vis_dir else None
        visa_auroc, visa_ap, visa_pauroc, visa_pro = test_on_visa(
            model_mvtec, device, transform, visualize=args.visualize, vis_dir=visa_vis_dir, n_vis=args.n_vis
        )
        results["VisA"] = {"AUROC": visa_auroc, "AP": visa_ap, "pAUROC": visa_pauroc, "PRO": visa_pro}

    # Summary
    logger.info("\n" + "=" * 100)
    logger.info("BENCHMARK SUMMARY (All Categories Training)")
    logger.info("=" * 100)
    logger.info("Image-level Detection (ZSAD):")
    logger.info(f"{'Dataset':<15} {'Ours AUROC/AP':>20} {'Paper AUROC/AP':>20} {'Gap':>10}")
    logger.info("-" * 70)

    if "MVTec AD" in results:
        r = results["MVTec AD"]
        logger.info(f"{'MVTec AD':<15} {r['AUROC']*100:>8.1f}/{r['AP']*100:<8.1f} {'91.9/96.5':>20} {(r['AUROC']*100-91.9):>+.1f}%")

    if "BTAD" in results:
        r = results["BTAD"]
        logger.info(f"{'BTAD':<15} {r['AUROC']*100:>8.1f}/{r['AP']*100:<8.1f} {'90.3/90.0':>20} {(r['AUROC']*100-90.3):>+.1f}%")

    if "VisA" in results:
        r = results["VisA"]
        logger.info(f"{'VisA':<15} {r['AUROC']*100:>8.1f}/{r['AP']*100:<8.1f} {'84.6/86.6':>20} {(r['AUROC']*100-84.6):>+.1f}%")

    logger.info("")
    logger.info("Pixel-level Segmentation (ZSAS):")
    logger.info(f"{'Dataset':<15} {'Ours pAUROC/PRO':>20} {'Paper pAUROC/PRO':>20} {'Gap pAUROC/PRO':>20}")
    logger.info("-" * 80)

    if "MVTec AD" in results:
        r = results["MVTec AD"]
        logger.info(f"{'MVTec AD':<15} {r['pAUROC']*100:>8.1f}/{r['PRO']*100:<8.1f} {'92.6/88.3':>20} {(r['pAUROC']*100-92.6):>+.1f}/{(r['PRO']*100-88.3):>+.1f}%")

    if "BTAD" in results:
        r = results["BTAD"]
        logger.info(f"{'BTAD':<15} {r['pAUROC']*100:>8.1f}/{r['PRO']*100:<8.1f} {'95.6/80.4':>20} {(r['pAUROC']*100-95.6):>+.1f}/{(r['PRO']*100-80.4):>+.1f}%")

    if "VisA" in results:
        r = results["VisA"]
        logger.info(f"{'VisA':<15} {r['pAUROC']*100:>8.1f}/{r['PRO']*100:<8.1f} {'95.9/92.8':>20} {(r['pAUROC']*100-95.9):>+.1f}/{(r['PRO']*100-92.8):>+.1f}%")

    logger.info("=" * 100)


if __name__ == "__main__":
    main()
