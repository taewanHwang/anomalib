"""Adaptive Multi-task Loss for Custom DRAEM.

Advanced multi-task loss with uncertainty-based weighting and progressive training.

Author: Taewan Hwang
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Optional

from kornia.losses import FocalLoss, SSIMLoss


class AdaptiveCustomDraemLoss(nn.Module):
    """Adaptive multi-task loss for Custom DRAEM with uncertainty weighting and scheduling.
    
    Implements the strategy recommended by multi-task learning experts:
    1. Warmup phase: Focus on reconstruction (epochs 0 ~ warmup_epochs)
    2. Ramp-up phase: Gradually increase segmentation/severity weights
    3. Adaptive phase: Use uncertainty weighting or DWA for automatic balancing
    
    Args:
        warmup_epochs (int): Number of warmup epochs focusing on reconstruction.
        ramp_epochs (int): Number of ramp-up epochs for gradual weight increase.
        use_uncertainty_weighting (bool): Enable learnable uncertainty weights.
        use_dwa (bool): Enable Dynamic Weight Average after ramp-up.
        initial_weights (Dict[str, float]): Initial task weights.
        severity_loss_type (str): Type of severity loss ("mse" or "smooth_l1").
    """
    
    def __init__(
        self,
        warmup_epochs: int = 5,
        ramp_epochs: int = 10, 
        use_uncertainty_weighting: bool = True,
        use_dwa: bool = False,
        initial_weights: Optional[Dict[str, float]] = None,
        severity_loss_type: str = "mse",
    ) -> None:
        super().__init__()
        
        # Training phase configuration
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.use_dwa = use_dwa
        
        # Default initial weights
        if initial_weights is None:
            initial_weights = {
                "reconstruction": 1.0,
                "segmentation": 1.0, 
                "severity": 0.5
            }
        self.initial_weights = initial_weights
        
        # Loss components
        self.l2_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss(window_size=11)
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        
        if severity_loss_type == "mse":
            self.severity_loss = nn.MSELoss()
        elif severity_loss_type == "smooth_l1":
            self.severity_loss = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported severity loss type: {severity_loss_type}")
        
        # Learnable uncertainty parameters (log variance)
        if self.use_uncertainty_weighting:
            self.log_var_reconstruction = nn.Parameter(torch.tensor(0.0))
            self.log_var_segmentation = nn.Parameter(torch.tensor(0.0))
            self.log_var_severity = nn.Parameter(torch.tensor(0.0))
        
        # DWA history tracking
        if self.use_dwa:
            self.register_buffer('loss_history', torch.zeros(3, 2))  # [3 tasks, 2 epochs]
            self.dwa_temp = 2.0  # Temperature parameter
        
        # Current epoch tracking
        self.current_epoch = 0
        
    def _get_phase_weights(self, epoch: int) -> Dict[str, float]:
        """Get task weights based on current training phase."""
        if epoch < self.warmup_epochs:
            # Warmup phase: Focus on reconstruction
            return {
                "reconstruction": self.initial_weights["reconstruction"],
                "segmentation": 0.1 * self.initial_weights["segmentation"],  # Very low
                "severity": 0.05 * self.initial_weights["severity"]  # Almost zero
            }
        elif epoch < self.warmup_epochs + self.ramp_epochs:
            # Ramp-up phase: Linear increase
            progress = (epoch - self.warmup_epochs) / self.ramp_epochs
            return {
                "reconstruction": self.initial_weights["reconstruction"],
                "segmentation": (0.1 + 0.9 * progress) * self.initial_weights["segmentation"],
                "severity": (0.05 + 0.95 * progress) * self.initial_weights["severity"]
            }
        else:
            # Adaptive phase: Use initial weights as base
            return self.initial_weights
    
    def _get_uncertainty_weights(self) -> Dict[str, float]:
        """Get weights from learnable uncertainty parameters."""
        if not self.use_uncertainty_weighting:
            return {"reconstruction": 1.0, "segmentation": 1.0, "severity": 1.0}
        
        # Uncertainty weighting: w_i = 1 / (2 * sigma_i^2)
        # Additional regularization: + log(sigma_i) to prevent sigma -> 0
        var_rec = torch.exp(self.log_var_reconstruction)
        var_seg = torch.exp(self.log_var_segmentation) 
        var_sev = torch.exp(self.log_var_severity)
        
        return {
            "reconstruction": 1.0 / (2 * var_rec),
            "segmentation": 1.0 / (2 * var_seg),
            "severity": 1.0 / (2 * var_sev),
            # Regularization terms
            "reconstruction_reg": 0.5 * self.log_var_reconstruction,
            "segmentation_reg": 0.5 * self.log_var_segmentation,
            "severity_reg": 0.5 * self.log_var_severity,
        }
    
    def _get_dwa_weights(self, current_losses: torch.Tensor) -> Dict[str, float]:
        """Get Dynamic Weight Average weights based on loss decrease rates."""
        if not self.use_dwa or self.current_epoch < 2:
            return {"reconstruction": 1.0, "segmentation": 1.0, "severity": 1.0}
        
        # Calculate relative decrease rates: L(t-1) / L(t-2)
        rates = self.loss_history[:, 0] / (self.loss_history[:, 1] + 1e-8)
        
        # Softmax with temperature to get weights
        weights = F.softmax(rates / self.dwa_temp, dim=0)
        
        # Update history
        self.loss_history[:, 1] = self.loss_history[:, 0].clone()
        self.loss_history[:, 0] = current_losses.detach()
        
        return {
            "reconstruction": weights[0].item() * 3,  # Normalize to sum ~3
            "segmentation": weights[1].item() * 3,
            "severity": weights[2].item() * 3,
        }
    
    def forward(
        self,
        input_image: torch.Tensor,
        reconstruction: torch.Tensor,
        anomaly_mask: torch.Tensor,
        prediction: torch.Tensor,
        severity_gt: torch.Tensor,
        severity_pred: torch.Tensor,
        epoch: int = 0
    ) -> torch.Tensor:
        """Compute adaptive multi-task loss."""
        self.current_epoch = epoch
        
        # Compute individual losses
        l2_loss_val = self.l2_loss(reconstruction, input_image)
        ssim_loss_val = self.ssim_loss(reconstruction, input_image) * 2
        # Handle multi-channel anomaly mask - convert to single channel
        if anomaly_mask.dim() == 4 and anomaly_mask.shape[1] > 1:
            anomaly_mask_single = anomaly_mask[:, 0, :, :]  # Take first channel
        else:
            anomaly_mask_single = anomaly_mask.squeeze(1)
        
        focal_loss_val = self.focal_loss(prediction, anomaly_mask_single.long()).mean()
        severity_loss_val = self.severity_loss(severity_pred, severity_gt)
        
        reconstruction_loss = l2_loss_val + ssim_loss_val
        
        # Current individual losses for DWA
        current_losses = torch.tensor([
            reconstruction_loss.item(),
            focal_loss_val.mean().item(), 
            severity_loss_val.item()
        ], device=input_image.device)
        
        # Get phase-based weights
        phase_weights = self._get_phase_weights(epoch)
        
        # Get adaptive weights
        if epoch >= self.warmup_epochs + self.ramp_epochs:
            if self.use_uncertainty_weighting:
                uncertainty_weights = self._get_uncertainty_weights()
                # Combine uncertainty weights with phase weights
                final_weights = {
                    "reconstruction": phase_weights["reconstruction"] * uncertainty_weights["reconstruction"],
                    "segmentation": phase_weights["segmentation"] * uncertainty_weights["segmentation"],
                    "severity": phase_weights["severity"] * uncertainty_weights["severity"]
                }
                # Add uncertainty regularization
                uncertainty_reg = (
                    uncertainty_weights["reconstruction_reg"] + 
                    uncertainty_weights["segmentation_reg"] + 
                    uncertainty_weights["severity_reg"]
                )
            elif self.use_dwa:
                dwa_weights = self._get_dwa_weights(current_losses)
                final_weights = {
                    "reconstruction": phase_weights["reconstruction"] * dwa_weights["reconstruction"],
                    "segmentation": phase_weights["segmentation"] * dwa_weights["segmentation"], 
                    "severity": phase_weights["severity"] * dwa_weights["severity"]
                }
                uncertainty_reg = 0.0
            else:
                final_weights = phase_weights
                uncertainty_reg = 0.0
        else:
            final_weights = phase_weights
            uncertainty_reg = 0.0
        
        # Compute total loss
        total_loss = (
            final_weights["reconstruction"] * reconstruction_loss +
            final_weights["segmentation"] * focal_loss_val +
            final_weights["severity"] * severity_loss_val +
            uncertainty_reg
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
        epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Get individual loss components for logging."""
        l2_loss_val = self.l2_loss(reconstruction, input_image)
        ssim_loss_val = self.ssim_loss(reconstruction, input_image) * 2
        # Handle multi-channel anomaly mask - convert to single channel
        if anomaly_mask.dim() == 4 and anomaly_mask.shape[1] > 1:
            anomaly_mask_single = anomaly_mask[:, 0, :, :]  # Take first channel
        else:
            anomaly_mask_single = anomaly_mask.squeeze(1)
        
        focal_loss_val = self.focal_loss(prediction, anomaly_mask_single.long()).mean()
        severity_loss_val = self.severity_loss(severity_pred, severity_gt)
        
        total_loss = self.forward(
            input_image, reconstruction, anomaly_mask, 
            prediction, severity_gt, severity_pred, epoch
        )
        
        result = {
            "l2_loss": l2_loss_val,
            "ssim_loss": ssim_loss_val,
            "focal_loss": focal_loss_val,
            "severity_loss": severity_loss_val,
            "total_loss": total_loss,
        }
        
        # Add phase and adaptive weight information for debugging
        phase_weights = self._get_phase_weights(epoch)
        result.update({f"weight_{k}": torch.tensor(v) for k, v in phase_weights.items()})
        
        if self.use_uncertainty_weighting and epoch >= self.warmup_epochs + self.ramp_epochs:
            uncertainty_weights = self._get_uncertainty_weights()
            result.update({f"uncertainty_{k}": torch.tensor(v) for k, v in uncertainty_weights.items() if not k.endswith("_reg")})
        
        return result

