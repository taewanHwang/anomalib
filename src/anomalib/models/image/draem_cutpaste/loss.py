"""Loss function for the DRAEM CutPaste model implementation.

Same as DRAEM loss but works with CutPaste augmentation.
"""

import torch
from kornia.losses import FocalLoss, SSIMLoss
from torch import nn


class DraemCutPasteLoss(nn.Module):
    """Overall loss function of the DRAEM CutPaste model.

    The total loss consists of three components:
    1. L2 loss between the reconstructed and input images
    2. Focal loss between predicted and ground truth anomaly masks
    3. Structural Similarity (SSIM) loss between reconstructed and input images

    The final loss is computed as: ``loss = l2_loss + ssim_loss + focal_loss``

    Example:
        >>> criterion = DraemCutPasteLoss()
        >>> loss = criterion(input_image, reconstruction, anomaly_mask, prediction)
    """

    def __init__(self) -> None:
        """Initialize loss components with default parameters."""
        super().__init__()

        self.l2_loss = nn.modules.loss.MSELoss()
        self.focal_loss = FocalLoss(alpha=1, reduction="mean")
        self.ssim_loss = SSIMLoss(window_size=11)

    def forward(
        self,
        input_image: torch.Tensor,
        reconstruction: torch.Tensor,
        anomaly_mask: torch.Tensor,
        prediction: torch.Tensor,
        use_focal_loss: bool = True,
    ) -> torch.Tensor:
        """Compute the combined loss over a batch for the DRAEM CutPaste model.

        Args:
            input_image: Original input images of shape
                ``(batch_size, num_channels, height, width)``
            reconstruction: Reconstructed images from the model of shape
                ``(batch_size, num_channels, height, width)``
            anomaly_mask: Ground truth anomaly masks of shape
                ``(batch_size, 1, height, width)``
            prediction: Model predictions of shape
                ``(batch_size, num_classes, height, width)``
            use_focal_loss: Whether to include focal loss in the total loss.
                Set to False during validation/test when pixel-level GT is unavailable.
                Defaults to ``True``.

        Returns:
            torch.Tensor: Combined loss value
        """
        l2_loss_val = self.l2_loss(reconstruction, input_image)
        ssim_loss_val = self.ssim_loss(reconstruction, input_image) * 2

        if use_focal_loss:
            focal_loss_val = self.focal_loss(prediction, anomaly_mask.squeeze(1).long())
            return l2_loss_val + ssim_loss_val + focal_loss_val
        else:
            # Validation/Test: only reconstruction losses (no pixel-level GT available)
            return l2_loss_val + ssim_loss_val
