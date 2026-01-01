"""Loss functions for FE-CLIP model.

This module implements the loss functions for training FE-CLIP adapters:
- BCE loss for image-level classification
- Focal loss for pixel-level segmentation
- Dice loss for pixel-level segmentation

Paper: FE-CLIP: Frequency Enhanced CLIP Model for Zero-Shot Anomaly Detection
"""

import torch
import torch.nn.functional as F


def bce_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binary cross-entropy loss for image-level classification.

    L_cls = -[S_gt * log(S_a) + (1 - S_gt) * log(1 - S_a)]

    Args:
        pred: Predicted anomaly scores of shape (B,).
        target: Ground truth labels of shape (B,) where 1=abnormal, 0=normal.
        eps: Small epsilon for numerical stability.

    Returns:
        Scalar BCE loss.
    """
    pred = pred.clamp(eps, 1 - eps)
    return F.binary_cross_entropy(pred, target)


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Focal loss for pixel-level segmentation.

    Focal loss helps address class imbalance by down-weighting easy examples.

    Args:
        pred: Predicted anomaly map of shape (B, H, W).
        target: Ground truth mask of shape (B, H, W) where 1=abnormal, 0=normal.
        alpha: Weighting factor for positive class. Default: 0.25.
        gamma: Focusing parameter. Default: 2.0.
        eps: Small epsilon for numerical stability.

    Returns:
        Scalar focal loss.
    """
    pred = pred.clamp(eps, 1 - eps)
    pt = torch.where(target > 0.5, pred, 1 - pred)
    w = torch.where(target > 0.5, alpha, 1 - alpha)
    return (-w * (1 - pt) ** gamma * torch.log(pt)).mean()


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dice loss for pixel-level segmentation.

    Dice loss is useful for handling class imbalance in segmentation tasks.

    Args:
        pred: Predicted anomaly map of shape (B, H, W).
        target: Ground truth mask of shape (B, H, W) where 1=abnormal, 0=normal.
        eps: Small epsilon for numerical stability.

    Returns:
        Scalar dice loss.
    """
    pred = pred.view(pred.shape[0], -1)
    target = target.view(target.shape[0], -1)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return (1 - dice).mean()


def feclip_loss(
    pred_score: torch.Tensor,
    pred_map: torch.Tensor,
    gt_label: torch.Tensor,
    gt_mask: torch.Tensor | None = None,
    w_cls: float = 1.0,
    w_mask: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combined FE-CLIP loss.

    L_total = w_cls * L_cls + w_mask * L_mask
    L_mask = Focal + Dice

    Args:
        pred_score: Predicted anomaly scores of shape (B,).
        pred_map: Predicted anomaly map of shape (B, H, W).
        gt_label: Ground truth labels of shape (B,).
        gt_mask: Optional ground truth mask of shape (B, H, W).
        w_cls: Weight for classification loss. Default: 1.0.
        w_mask: Weight for segmentation loss. Default: 1.0.

    Returns:
        Tuple of (total_loss, loss_dict) where loss_dict contains individual losses.
    """
    loss_dict = {}

    # Image-level classification loss
    loss_cls = bce_loss(pred_score, gt_label.float())
    loss_dict["loss_cls"] = loss_cls
    total_loss = w_cls * loss_cls

    # Pixel-level segmentation loss (if mask is available)
    if gt_mask is not None:
        loss_focal = focal_loss(pred_map, gt_mask)
        loss_dice = dice_loss(pred_map, gt_mask)
        loss_mask = loss_focal + loss_dice
        loss_dict["loss_focal"] = loss_focal
        loss_dict["loss_dice"] = loss_dice
        loss_dict["loss_mask"] = loss_mask
        total_loss = total_loss + w_mask * loss_mask

    loss_dict["loss_total"] = total_loss
    return total_loss, loss_dict
