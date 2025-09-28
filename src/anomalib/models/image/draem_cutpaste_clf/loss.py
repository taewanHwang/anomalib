"""Loss function for DRAEM CutPaste Classification model.

This module extends the existing DRAEM loss with additional classification loss
for the DraemCutPasteClf model.

The loss combines:
1. L2 reconstruction loss (from DRAEM)
2. SSIM reconstruction loss (from DRAEM)
3. Focal loss for anomaly mask prediction (with configurable alpha)
4. Cross-entropy loss for binary classification (new)
"""

import torch
from torch import nn
from kornia.losses import FocalLoss, SSIMLoss

class DraemCutPasteLoss(nn.Module):
    """Extended DRAEM loss with classification component.

    Extends the existing DraemLoss with additional classification loss
    for binary anomaly classification.

    Args:
        clf_weight (float, optional): Weight for classification loss. Defaults to ``1.0``.

    Example:
        >>> loss_fn = DraemCutPasteLoss(
        ...     clf_weight=1.0,
        ... )
        >>> # During training forward pass
        >>> total_loss, loss_dict = loss_fn(
        ...     reconstruction, original_batch,
        ...     prediction, anomaly_mask,
        ...     classification, anomaly_labels
        ... )
    """

    def __init__(
        self,
        clf_weight: float = 1.0,
    ):
        super().__init__()

        self.clf_weight = clf_weight

        # All loss components managed directly (no inheritance confusion)
        self.l2_loss = nn.MSELoss()
        self.focal_loss = FocalLoss(alpha=1.0, reduction="mean")
        self.ssim_loss = SSIMLoss(window_size=11)
        self.clf_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        reconstruction: torch.Tensor,
        original: torch.Tensor,
        prediction: torch.Tensor,
        anomaly_mask: torch.Tensor,
        classification: torch.Tensor,
        anomaly_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute extended DRAEM loss with classification.

        Args:
            reconstruction (torch.Tensor): Reconstructed images
            original (torch.Tensor): Original input images
            prediction (torch.Tensor): Predicted anomaly masks (logits)
            anomaly_mask (torch.Tensor): Ground truth anomaly masks
            classification (torch.Tensor): Classification logits
            anomaly_labels (torch.Tensor): Ground truth anomaly labels (0/1)

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                - total_loss: Combined weighted loss
                - loss_dict: Dictionary with individual loss components
        """
        # 1. Compute individual DRAEM loss components directly
        loss_l2 = self.l2_loss(reconstruction, original)
        loss_ssim = self.ssim_loss(reconstruction, original) * 2  # DRAEM multiplies by 2
        loss_focal = self.focal_loss(prediction, anomaly_mask.squeeze(1).long())

        # 2. Base DRAEM loss (L2 + SSIM + Focal with custom alpha)
        base_loss = loss_l2 + loss_ssim + loss_focal

        # 3. Classification loss (CrossEntropy)
        if anomaly_labels.dtype != torch.long:
            anomaly_labels = anomaly_labels.long()

        loss_clf = self.clf_loss(classification, anomaly_labels)

        # 4. Total loss with classification component
        total_loss = base_loss + self.clf_weight * loss_clf

        loss_dict = {
            "loss_l2": loss_l2,
            "loss_ssim": loss_ssim,
            "loss_focal": loss_focal,
            "loss_clf": loss_clf,
            "loss_base": base_loss,
            "total_loss": total_loss,
        }

        return total_loss, loss_dict

    def get_config(self) -> dict:
        """Get loss configuration.

        Returns:
            dict: Loss configuration parameters
        """
        return {
            "loss_type": "DraemCutPasteLoss",
            "clf_weight": self.clf_weight,
        }


