"""PyTorch model implementation for Custom DRAEM.

Custom DRAEM extends the original DRAEM with additional Fault Severity Prediction Sub-Network.
Uses pretrained DRAEM backbone for fair comparison with custom severity prediction capability.

Author: Taewan Hwang
"""

import torch
from torch import nn

from anomalib.data import InferenceBatch
from anomalib.models.components.layers import SSPCAB
from anomalib.models.image.draem.torch_model import DraemModel


class CustomDraemModel(nn.Module):
    """Custom DRAEM model with severity prediction capability.
    
    Args:
        sspcab (bool, optional): Enable SSPCAB training. Defaults to ``False``.
        severity_max (float, optional): Maximum severity value. Defaults to ``10.0``.
        severity_input_mode (str, optional): Input mode for severity network.
            Options: "discriminative_only", "with_original", "with_reconstruction", 
            "with_error_map", "multi_modal". Defaults to ``"discriminative_only"``.
    """
    
    def __init__(
        self, 
        sspcab: bool = False, 
        severity_max: float = 10.0,
        severity_input_mode: str = "discriminative_only"
    ) -> None:
        super().__init__()
        self.severity_max = severity_max
        self.severity_input_mode = severity_input_mode
        
        # Use pretrained DRAEM backbone components with SSPCAB option
        draem_backbone = DraemModel(sspcab=sspcab)
        self.reconstructive_subnetwork = draem_backbone.reconstructive_subnetwork
        self.discriminative_subnetwork = draem_backbone.discriminative_subnetwork
        
        # Determine input channels for severity network based on mode
        severity_input_channels = self._get_severity_input_channels()
        self.fault_severity_subnetwork = FaultSeveritySubNetwork(
            in_channels=severity_input_channels,
            severity_max=severity_max
        )
    
    def _get_severity_input_channels(self) -> int:
        """Get number of input channels for severity network based on input mode."""
        mode_channels = {
            "discriminative_only": 2,    # discriminative result only (2)
            "with_original": 5,          # discriminative + original (2 + 3 = 5)
            "with_reconstruction": 5,    # discriminative + reconstruction (2 + 3 = 5)  
            "with_error_map": 5,         # discriminative + error map (2 + 3 = 5)
            "multi_modal": 11           # discriminative + original + reconstruction + error (2 + 3 + 3 + 3 = 11)
        }
        return mode_channels.get(self.severity_input_mode, 2)
    
    def _prepare_severity_input(
        self, 
        original: torch.Tensor, 
        reconstruction: torch.Tensor, 
        discriminative_result: torch.Tensor
    ) -> torch.Tensor:
        """Prepare input tensor for severity network based on input mode."""
        if self.severity_input_mode == "discriminative_only":
            return discriminative_result
        elif self.severity_input_mode == "with_original":
            return torch.cat([discriminative_result, original], dim=1)
        elif self.severity_input_mode == "with_reconstruction":
            return torch.cat([discriminative_result, reconstruction], dim=1)
        elif self.severity_input_mode == "with_error_map":
            error_map = torch.abs(original - reconstruction)
            return torch.cat([discriminative_result, error_map], dim=1)
        elif self.severity_input_mode == "multi_modal":
            error_map = torch.abs(original - reconstruction)
            return torch.cat([discriminative_result, original, reconstruction, error_map], dim=1)
        else:
            raise ValueError(f"Unknown severity input mode: {self.severity_input_mode}")
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | InferenceBatch:
        """Forward pass through all sub-networks.
        
        Args:
            batch (torch.Tensor): Input batch of 3-channel RGB images of shape
                ``(batch_size, 3, height, width)``
        
        Returns:
            During training:
                tuple: Tuple containing:
                    - Reconstructed images (batch_size, 3, H, W)
                    - Predicted anomaly masks (batch_size, 2, H, W) 
                    - Predicted severity values (batch_size, 1)
            During inference:
                InferenceBatch: Contains anomaly map, prediction score, and severity
        """
        # Reconstruction step
        reconstruction = self.reconstructive_subnetwork(batch)
        
        # Discrimination step (concatenate original + reconstruction)
        concatenated_inputs = torch.cat([batch, reconstruction], dim=1)  # (B, 6, H, W) = (B, 3, H, W) + (B, 3, H, W)
        prediction = self.discriminative_subnetwork(concatenated_inputs)  # (B, 2, H, W)
        
        # Severity prediction step
        severity_input = self._prepare_severity_input(batch, reconstruction, prediction)
        severity_prediction = self.fault_severity_subnetwork(severity_input)  # (B, 1)
        
        if self.training:
            return reconstruction, prediction, severity_prediction
        
        # Inference mode
        anomaly_map = torch.softmax(prediction, dim=1)[:, 1, ...]  # (B, H, W) - channel 1 is anomaly probability
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))  # (B,)
        severity_score = severity_prediction.squeeze(-1)  # (B,)
        
        return InferenceBatch(
            pred_score=pred_score, 
            anomaly_map=anomaly_map,
            pred_label=severity_score  # Use pred_label field for severity scores
        )


class FaultSeveritySubNetwork(nn.Module):
    """Fault severity prediction network.
    
    Predicts continuous severity values from discriminative features.
    Uses CNN for feature extraction + Global Pooling + MLP for regression.
    
    Args:
        in_channels (int, optional): Number of input channels. Defaults to ``2``.
        severity_max (float, optional): Maximum severity value. Defaults to ``10.0``.
        hidden_dim (int, optional): Hidden dimension for MLP. Defaults to ``128``.
    """
    
    def __init__(
        self, 
        in_channels: int = 2, 
        severity_max: float = 10.0,
        hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.severity_max = severity_max
        
        # Feature extraction (CNN backbone) - Input: 224x224 or 256x256
        self.feature_extractor = nn.Sequential(
            # First conv block: 224x224 -> 112x112 (or 256x256 -> 128x128)
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second conv block: 112x112 -> 56x56 (or 128x128 -> 64x64)
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Global pooling
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        
        # MLP regressor
        self.regressor = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1] range, then scale to [0, severity_max]
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Kaiming initialization for Conv2d layers (good for ReLU)
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                # Standard initialization for BatchNorm
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Xavier initialization for Linear layers
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass through severity prediction network.
        
        Args:
            batch (torch.Tensor): Input batch of varying channels depending on input mode
                                  Shape: (batch_size, in_channels, 224, 224) or (batch_size, in_channels, 256, 256)
            
        Returns:
            torch.Tensor: Predicted severity values of shape (batch_size, 1)
                         Values are in range [0, severity_max]
        """
        # Feature extraction
        features = self.feature_extractor(batch)  # (B, 128, H', W')
        
        # Global pooling
        pooled_features = self.global_pooling(features)  # (B, 128, 1, 1)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  # (B, 128)
        
        # Regression
        severity_raw = self.regressor(pooled_features)  # (B, 1)
        
        # Scale to [0, severity_max] range
        severity_prediction = severity_raw * self.severity_max
        
        return severity_prediction