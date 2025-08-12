"""PyTorch model implementation for Custom DRAEM.

Custom DRAEM extends the original DRAEM with:
1. 1-channel grayscale support for HDMAP data
2. Additional Fault Severity Prediction Sub-Network
3. Multi-task loss optimization

Architecture:
- ReconstructiveSubNetwork: Autoencoder for image reconstruction (1→1 channel)
- DiscriminativeSubNetwork: Anomaly detection (original + reconstruction = 2→2 channels)  
- FaultSeveritySubNetwork: Severity prediction (discriminative result → continuous severity)
"""

import torch
from torch import nn

from anomalib.data import InferenceBatch
from anomalib.models.components.layers import SSPCAB


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
        
        # Initialize sub-networks
        self.reconstructive_subnetwork = ReconstructiveSubNetwork(sspcab=sspcab)
        self.discriminative_subnetwork = DiscriminativeSubNetwork(in_channels=2, out_channels=2)
        
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
            "with_original": 3,          # discriminative + original (2 + 1 = 3)
            "with_reconstruction": 3,    # discriminative + reconstruction (2 + 1 = 3)  
            "with_error_map": 3,         # discriminative + error map (2 + 1 = 3)
            "multi_modal": 5            # discriminative + original + reconstruction + error (2 + 1 + 1 + 1 = 5)
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
            batch (torch.Tensor): Input batch of 1-channel grayscale images of shape
                ``(batch_size, 1, height, width)``
        
        Returns:
            During training:
                tuple: Tuple containing:
                    - Reconstructed images (batch_size, 1, H, W)
                    - Predicted anomaly masks (batch_size, 2, H, W) 
                    - Predicted severity values (batch_size, 1)
            During inference:
                InferenceBatch: Contains anomaly map, prediction score, and severity
        """
        # Reconstruction step
        reconstruction = self.reconstructive_subnetwork(batch)
        
        # Discrimination step (concatenate original + reconstruction)
        concatenated_inputs = torch.cat([batch, reconstruction], dim=1)  # (B, 2, H, W)
        prediction = self.discriminative_subnetwork(concatenated_inputs)  # (B, 2, H, W)
        
        # Severity prediction step
        severity_input = self._prepare_severity_input(batch, reconstruction, prediction)
        severity_prediction = self.fault_severity_subnetwork(severity_input)  # (B, 1)
        
        if self.training:
            return reconstruction, prediction, severity_prediction
        
        # Inference mode
        anomaly_map = torch.softmax(prediction, dim=1)[:, 1, ...]  # (B, H, W)
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))  # (B,)
        severity_score = severity_prediction.squeeze(-1)  # (B,)
        
        return InferenceBatch(
            pred_score=pred_score, 
            anomaly_map=anomaly_map,
            pred_label=severity_score  # Use pred_label field for severity scores
        )


class ReconstructiveSubNetwork(nn.Module):
    """1-channel grayscale autoencoder for HDMAP reconstruction.
    
    Modified from original DRAEM to support 1-channel input/output.
    
    Args:
        sspcab (bool, optional): Enable SSPCAB training. Defaults to ``False``.
    """
    
    def __init__(self, sspcab: bool = False) -> None:
        super().__init__()
        
        # Encoder layers (1 channel input)
        self.encoder_layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.encoder_layer2 = nn.Sequential(
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.encoder_layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.encoder_layer4 = nn.Sequential(
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.encoder_layer5 = nn.Sequential(
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.encoder_layer6 = nn.Sequential(
            nn.Conv2d(64, 64, 8),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Decoder layers (1 channel output)
        self.decoder_layer1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 8),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder_layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder_layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder_layer4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.decoder_layer5 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.decoder_layer6 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, 4, 2, 1),  # Output 1 channel
            nn.Sigmoid()  # Ensure output is in [0, 1] range
        )
        
        # Optional SSPCAB layer
        self.sspcab = SSPCAB(1) if sspcab else None
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                # Kaiming initialization for Conv layers (good for ReLU)
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                # Standard initialization for BatchNorm
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder.
        
        Args:
            batch (torch.Tensor): Input batch of shape (batch_size, 1, H, W)
            
        Returns:
            torch.Tensor: Reconstructed images of shape (batch_size, 1, H, W)
        """
        # Encoder
        x1 = self.encoder_layer1(batch)
        x2 = self.encoder_layer2(x1)
        x3 = self.encoder_layer3(x2)
        x4 = self.encoder_layer4(x3)
        x5 = self.encoder_layer5(x4)
        x6 = self.encoder_layer6(x5)
        
        # Decoder
        x7 = self.decoder_layer1(x6)
        x8 = self.decoder_layer2(x7)
        x9 = self.decoder_layer3(x8)
        x10 = self.decoder_layer4(x9)
        x11 = self.decoder_layer5(x10)
        reconstruction = self.decoder_layer6(x11)
        
        # Optional SSPCAB processing
        if self.sspcab is not None:
            reconstruction = self.sspcab(reconstruction)
            
        return reconstruction


class DiscriminativeSubNetwork(nn.Module):
    """Discriminative network for anomaly detection.
    
    Modified from original DRAEM to support 2-channel input (original + reconstruction).
    
    Args:
        in_channels (int, optional): Number of input channels. Defaults to ``2``.
        out_channels (int, optional): Number of output channels. Defaults to ``2``.
    """
    
    def __init__(self, in_channels: int = 2, out_channels: int = 2) -> None:
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, 8),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Decoder for pixel-wise prediction
        self.layer7 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 8),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer8 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer9 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer10 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer11 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer12 = nn.ConvTranspose2d(32, out_channels, 4, 2, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                # Kaiming initialization for Conv layers (good for ReLU)
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                # Standard initialization for BatchNorm
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminative network.
        
        Args:
            batch (torch.Tensor): Input batch of shape (batch_size, 2, H, W)
                                  Concatenated original + reconstruction
            
        Returns:
            torch.Tensor: Anomaly predictions of shape (batch_size, 2, H, W)
                         Channel 0: normal probability, Channel 1: anomaly probability
        """
        # Encoder
        x1 = self.layer1(batch)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        
        # Decoder
        x7 = self.layer7(x6)
        x8 = self.layer8(x7)
        x9 = self.layer9(x8)
        x10 = self.layer10(x9)
        x11 = self.layer11(x10)
        prediction = self.layer12(x11)
        
        return prediction


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
        
        # Feature extraction (CNN backbone)
        self.feature_extractor = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 256 -> 128
            
            # Second conv block  
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128 -> 64
            
            # Third conv block
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 -> 32
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
                                  Shape: (batch_size, in_channels, H, W)
            
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