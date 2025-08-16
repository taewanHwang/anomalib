
"""Loss function for Custom DRAEM model.

Multi-task loss combining reconstruction, segmentation, and severity prediction losses.
Includes DRAEM-SevNet loss function for unified severity-aware architecture.

Author: Taewan Hwang
"""

import torch
from kornia.losses import FocalLoss, SSIMLoss
from torch import nn
from anomalib.models.image.draem.loss import DraemLoss


class CustomDraemLoss(nn.Module):
    """Multi-task loss function for Custom DRAEM model.

    The total loss consists of four components:
    1. L2 loss between reconstructed and input images
    2. Structural Similarity (SSIM) loss between reconstructed and input images  
    3. Focal loss between predicted and ground truth anomaly masks
    4. Severity prediction loss (MSE or SmoothL1)

    The final loss is computed as:
    ```
    total_loss = reconstruction_weight * (l2_loss + ssim_loss) + 
                 segmentation_weight * focal_loss + 
                 severity_weight * severity_loss
    ```

    Args:
        reconstruction_weight (float, optional): Weight for reconstruction losses.
            Defaults to ``1.0``.
        segmentation_weight (float, optional): Weight for segmentation loss.
            Defaults to ``1.0``.
        severity_weight (float, optional): Weight for severity prediction loss.
            Defaults to ``0.5``.
        severity_loss_type (str, optional): Type of severity loss.
            Options: "mse", "smooth_l1". Defaults to ``"mse"``.

    Example:
        >>> criterion = CustomDraemLoss(
        ...     reconstruction_weight=1.0,
        ...     segmentation_weight=1.0,
        ...     severity_weight=0.5,
        ...     severity_loss_type="smooth_l1"
        ... )
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        segmentation_weight: float = 1.0,
        severity_weight: float = 0.5,
        severity_loss_type: str = "mse"
    ) -> None:
        """Initialize loss components with configurable weights."""
        super().__init__()

        # Store weights
        self.reconstruction_weight = reconstruction_weight
        self.segmentation_weight = segmentation_weight  
        self.severity_weight = severity_weight

        # Original DRAEM loss components
        self.l2_loss = nn.MSELoss()
        self.focal_loss = FocalLoss(alpha=1, reduction="mean")
        self.ssim_loss = SSIMLoss(window_size=11)

        # NEW: Severity prediction loss
        if severity_loss_type == "mse":
            self.severity_loss = nn.MSELoss()
        elif severity_loss_type == "smooth_l1":
            self.severity_loss = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported severity loss type: {severity_loss_type}")

    def forward(
        self,
        input_image: torch.Tensor,
        reconstruction: torch.Tensor,
        anomaly_mask: torch.Tensor,
        prediction: torch.Tensor,
        severity_gt: torch.Tensor,
        severity_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the combined multi-task loss for Custom DRAEM model.

        Args:
            input_image: Original input images of shape
                ``(batch_size, 1, height, width)`` (1-channel grayscale)
            reconstruction: Reconstructed images from the model of shape
                ``(batch_size, 1, height, width)``
            anomaly_mask: Ground truth anomaly masks of shape
                ``(batch_size, 1, height, width)``
            prediction: Model predictions of shape
                ``(batch_size, 2, height, width)`` (background/anomaly)
            severity_gt: Ground truth severity values of shape
                ``(batch_size, 1)``
            severity_pred: Predicted severity values of shape
                ``(batch_size, 1)``

        Returns:
            torch.Tensor: Combined loss value

        Note:
            The loss weights allow for balancing the importance of different tasks:
            - reconstruction_weight: Controls image reconstruction quality
            - segmentation_weight: Controls anomaly localization accuracy  
            - severity_weight: Controls severity prediction accuracy (typically lower)
        """
        # Original DRAEM losses
        l2_loss_val = self.l2_loss(reconstruction, input_image)
        
        # Handle multi-channel anomaly mask - convert to single channel
        if anomaly_mask.dim() == 4 and anomaly_mask.shape[1] > 1:
            # Take first channel or convert RGB to grayscale
            anomaly_mask_single = anomaly_mask[:, 0, :, :]  # Take first channel
        else:
            anomaly_mask_single = anomaly_mask.squeeze(1)
        
        focal_loss_val = self.focal_loss(prediction, anomaly_mask_single.long())
        ssim_loss_val = self.ssim_loss(reconstruction, input_image) * 2

        # NEW: Severity prediction loss
        severity_loss_val = self.severity_loss(severity_pred, severity_gt)

        # Weighted combination
        reconstruction_loss = l2_loss_val + ssim_loss_val
        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            self.segmentation_weight * focal_loss_val +
            self.severity_weight * severity_loss_val
        )

        return total_loss

    def get_individual_losses(
        self,
        input_image: torch.Tensor,
        reconstruction: torch.Tensor,
        anomaly_mask: torch.Tensor,
        prediction: torch.Tensor,
        severity_gt: torch.Tensor,
        severity_pred: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Get individual loss components for logging and analysis.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing individual loss values:
                - "l2_loss": L2 reconstruction loss
                - "ssim_loss": SSIM reconstruction loss
                - "focal_loss": Focal segmentation loss
                - "severity_loss": Severity prediction loss
                - "total_loss": Combined weighted loss
        """
        l2_loss_val = self.l2_loss(reconstruction, input_image)
        
        # Handle multi-channel anomaly mask - convert to single channel
        if anomaly_mask.dim() == 4 and anomaly_mask.shape[1] > 1:
            anomaly_mask_single = anomaly_mask[:, 0, :, :]  # Take first channel
        else:
            anomaly_mask_single = anomaly_mask.squeeze(1)
        
        focal_loss_val = self.focal_loss(prediction, anomaly_mask_single.long())
        ssim_loss_val = self.ssim_loss(reconstruction, input_image) * 2
        severity_loss_val = self.severity_loss(severity_pred, severity_gt)

        reconstruction_loss = l2_loss_val + ssim_loss_val
        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            self.segmentation_weight * focal_loss_val +
            self.severity_weight * severity_loss_val
        )

        return {
            "l2_loss": l2_loss_val,
            "ssim_loss": ssim_loss_val,
            "focal_loss": focal_loss_val,
            "severity_loss": severity_loss_val,
            "total_loss": total_loss,
        }


class DraemSevNetLoss(nn.Module):
    """Loss function for DRAEM-SevNet (Unified Severity-Aware Architecture).
    
    Combines the original DRAEM loss (reconstruction + segmentation) with 
    severity prediction loss in a multi-task learning framework.
    
    The total loss is computed as:
    ```
    L = L_draem + λ * L_severity
    ```
    
    Where:
    - L_draem: Original DRAEM loss (l2_loss + ssim_loss + focal_loss)
    - L_severity: Severity prediction loss (MSE or SmoothL1)
    - λ: Fixed weighting factor for severity loss
    
    Args:
        severity_weight (float, optional): Weight λ for severity loss. Defaults to ``0.5``.
        severity_loss_type (str, optional): Type of severity loss.
            Options: "mse", "smooth_l1". Defaults to ``"mse"``.
            
    Example:
        >>> criterion = DraemSevNetLoss(severity_weight=0.3, severity_loss_type="smooth_l1")
        >>> loss = criterion(input_image, reconstruction, anomaly_mask, prediction, severity_gt, severity_pred)
    """
    
    def __init__(
        self,
        severity_weight: float = 0.5,
        severity_loss_type: str = "mse"
    ) -> None:
        super().__init__()
        
        self.severity_weight = severity_weight
        
        # Use original DRAEM loss for L_draem
        self.draem_loss = DraemLoss()
        
        # Severity prediction loss L_severity
        if severity_loss_type == "mse":
            self.severity_loss = nn.MSELoss()
        elif severity_loss_type == "smooth_l1":
            self.severity_loss = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported severity loss type: {severity_loss_type}. "
                           f"Choose from ['mse', 'smooth_l1']")
        
        self.severity_loss_type = severity_loss_type
    
    def forward(
        self,
        input_image: torch.Tensor,
        reconstruction: torch.Tensor,
        anomaly_mask: torch.Tensor,
        prediction: torch.Tensor,
        severity_gt: torch.Tensor,
        severity_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DRAEM-SevNet loss: L = L_draem + λ * L_severity.
        
        Args:
            input_image: Original input images of shape
                ``(batch_size, channels, height, width)``
            reconstruction: Reconstructed images from the model of shape
                ``(batch_size, channels, height, width)``
            anomaly_mask: Ground truth anomaly masks of shape
                ``(batch_size, 1, height, width)``
            prediction: Model predictions of shape
                ``(batch_size, 2, height, width)`` (background/anomaly)
            severity_gt: Ground truth severity values of shape
                ``(batch_size,)`` or ``(batch_size, 1)``
            severity_pred: Predicted severity values of shape
                ``(batch_size,)`` or ``(batch_size, 1)``
                
        Returns:
            torch.Tensor: Combined DRAEM-SevNet loss value
        """
        # L_draem: Original DRAEM loss
        draem_loss_val = self.draem_loss(input_image, reconstruction, anomaly_mask, prediction)
        
        # L_severity: Severity prediction loss
        # Ensure both tensors have the same shape
        if severity_gt.dim() != severity_pred.dim():
            if severity_gt.dim() == 1 and severity_pred.dim() == 2:
                severity_pred = severity_pred.squeeze(-1)
            elif severity_gt.dim() == 2 and severity_pred.dim() == 1:
                severity_gt = severity_gt.squeeze(-1)
        
        severity_loss_val = self.severity_loss(severity_pred, severity_gt)
        
        # Combined loss: L = L_draem + λ * L_severity
        total_loss = draem_loss_val + self.severity_weight * severity_loss_val
        
        return total_loss
    
    def get_individual_losses(
        self,
        input_image: torch.Tensor,
        reconstruction: torch.Tensor,
        anomaly_mask: torch.Tensor,
        prediction: torch.Tensor,
        severity_gt: torch.Tensor,
        severity_pred: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Get individual loss components for logging and analysis.
        
        Returns:
            dict[str, torch.Tensor]: Dictionary containing individual loss values:
                - "l2_loss": L2 reconstruction loss
                - "ssim_loss": SSIM reconstruction loss  
                - "focal_loss": Focal segmentation loss
                - "draem_loss": Combined DRAEM loss (l2 + ssim + focal)
                - "severity_loss": Severity prediction loss
                - "total_loss": Combined DRAEM-SevNet loss
        """
        # Get individual DRAEM loss components
        l2_loss_val = self.draem_loss.l2_loss(reconstruction, input_image)
        focal_loss_val = self.draem_loss.focal_loss(prediction, anomaly_mask.squeeze(1).long())
        ssim_loss_val = self.draem_loss.ssim_loss(reconstruction, input_image) * 2
        
        # Combined DRAEM loss
        draem_loss_val = l2_loss_val + ssim_loss_val + focal_loss_val
        
        # Severity loss
        if severity_gt.dim() != severity_pred.dim():
            if severity_gt.dim() == 1 and severity_pred.dim() == 2:
                severity_pred = severity_pred.squeeze(-1)
            elif severity_gt.dim() == 2 and severity_pred.dim() == 1:
                severity_gt = severity_gt.squeeze(-1)
        
        severity_loss_val = self.severity_loss(severity_pred, severity_gt)
        
        # Total DRAEM-SevNet loss
        total_loss = draem_loss_val + self.severity_weight * severity_loss_val
        
        return {
            "l2_loss": l2_loss_val,
            "ssim_loss": ssim_loss_val, 
            "focal_loss": focal_loss_val,
            "draem_loss": draem_loss_val,
            "severity_loss": severity_loss_val,
            "total_loss": total_loss,
        }
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"severity_weight={self.severity_weight}, "
                f"severity_loss_type='{self.severity_loss_type}')")


class DraemSevNetLossFactory:
    """Factory class for creating DraemSevNetLoss instances with common configurations."""
    
    @staticmethod
    def create_balanced(severity_weight: float = 0.5) -> DraemSevNetLoss:
        """Create balanced DRAEM-SevNet loss with MSE severity loss.
        
        Args:
            severity_weight (float, optional): Weight for severity loss. Defaults to ``0.5``.
            
        Returns:
            DraemSevNetLoss: Configured loss function
        """
        return DraemSevNetLoss(severity_weight=severity_weight, severity_loss_type="mse")
    
    @staticmethod
    def create_robust(severity_weight: float = 0.3) -> DraemSevNetLoss:
        """Create robust DRAEM-SevNet loss with SmoothL1 severity loss.
        
        Args:
            severity_weight (float, optional): Weight for severity loss. Defaults to ``0.3``.
            
        Returns:
            DraemSevNetLoss: Configured loss function with SmoothL1 loss
        """
        return DraemSevNetLoss(severity_weight=severity_weight, severity_loss_type="smooth_l1")
    
    @staticmethod
    def create_mask_focused(severity_weight: float = 0.1) -> DraemSevNetLoss:
        """Create mask-focused DRAEM-SevNet loss with low severity weight.
        
        Args:
            severity_weight (float, optional): Weight for severity loss. Defaults to ``0.1``.
            
        Returns:
            DraemSevNetLoss: Configured loss function prioritizing mask prediction
        """
        return DraemSevNetLoss(severity_weight=severity_weight, severity_loss_type="mse")
    
    @staticmethod
    def create_severity_focused(severity_weight: float = 1.0) -> DraemSevNetLoss:
        """Create severity-focused DRAEM-SevNet loss with high severity weight.
        
        Args:
            severity_weight (float, optional): Weight for severity loss. Defaults to ``1.0``.
            
        Returns:
            DraemSevNetLoss: Configured loss function prioritizing severity prediction
        """
        return DraemSevNetLoss(severity_weight=severity_weight, severity_loss_type="mse")
