"""PyTorch model implementation for Custom DRAEM.

Custom DRAEM extends the original DRAEM with additional Fault Severity Prediction Sub-Network.
Uses pretrained DRAEM backbone for fair comparison with custom severity prediction capability.

Author: Taewan Hwang
"""

from dataclasses import dataclass
import torch
from torch import nn

from anomalib.data import InferenceBatch
from anomalib.models.components.layers import SSPCAB
from anomalib.models.image.draem.torch_model import DraemModel, ReconstructiveSubNetwork, DiscriminativeSubNetwork as OriginalDiscriminativeSubNetwork
from .severity_head import SeverityHead


@dataclass
class DraemSevNetOutput:
    """Output dataclass for DRAEM-SevNet inference.
    
    Contains all outputs from DRAEM-SevNet model including individual scores
    and combined final score for comprehensive analysis.
    
    Attributes:
        reconstruction (torch.Tensor): Reconstructed images
        mask_logits (torch.Tensor): Raw mask prediction logits
        severity_score (torch.Tensor): Severity prediction scores [0,1]
        mask_score (torch.Tensor): Mask-based anomaly scores [0,1]
        final_score (torch.Tensor): Combined final anomaly scores [0,1]
        anomaly_map (torch.Tensor): Processed anomaly probability map
    """
    reconstruction: torch.Tensor
    mask_logits: torch.Tensor
    severity_score: torch.Tensor
    mask_score: torch.Tensor  
    final_score: torch.Tensor
    anomaly_map: torch.Tensor


class DiscriminativeSubNetwork(nn.Module):
    """Enhanced Discriminative Sub-Network with encoder features exposure.
    
    Extends the original DRAEM DiscriminativeSubNetwork to expose encoder features
    (act1~act6) for DRAEM-SevNet's SeverityHead usage.
    
    Args:
        in_channels (int, optional): Number of input channels. Defaults to ``6``.
        out_channels (int, optional): Number of output channels. Defaults to ``2``.
        base_width (int, optional): Base dimensionality of layers. Defaults to ``64``.
        expose_features (bool, optional): Whether to return encoder features. Defaults to ``True``.
    """
    
    def __init__(
        self, 
        in_channels: int = 6, 
        out_channels: int = 2, 
        base_width: int = 64,
        expose_features: bool = True
    ) -> None:
        super().__init__()
        self.expose_features = expose_features
        
        # Use original DRAEM discriminative components
        self.original_subnet = OriginalDiscriminativeSubNetwork(
            in_channels=in_channels, 
            out_channels=out_channels, 
            base_width=base_width
        )
        
        # Store references for direct access
        self.encoder_segment = self.original_subnet.encoder_segment
        self.decoder_segment = self.original_subnet.decoder_segment
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass through discriminative network with optional feature exposure.
        
        Args:
            batch (torch.Tensor): Concatenated original and reconstructed images of
                shape ``(batch_size, channels*2, height, width)``
                
        Returns:
            If expose_features=False:
                torch.Tensor: Pixel-level class scores for normal and anomalous regions
            If expose_features=True:
                tuple: (mask_logits, encoder_features)
                    - mask_logits: Same as above
                    - encoder_features: Dict containing 'act1'~'act6' feature maps
        """
        # Extract encoder features
        act1, act2, act3, act4, act5, act6 = self.encoder_segment(batch)
        
        # Generate mask logits using decoder
        mask_logits = self.decoder_segment(act1, act2, act3, act4, act5, act6)
        
        if not self.expose_features:
            # Backward compatibility: return only mask logits
            return mask_logits
        
        # Return both mask logits and encoder features for SeverityHead
        encoder_features = {
            'act1': act1,
            'act2': act2, 
            'act3': act3,
            'act4': act4,
            'act5': act5,
            'act6': act6
        }
        
        return mask_logits, encoder_features


class DraemSevNetModel(nn.Module):
    """DRAEM-SevNet: DRAEM with Severity Network.
    
    Unified severity-aware architecture that combines mask prediction and severity
    prediction in a multi-task learning framework. Uses discriminative encoder 
    features directly for severity prediction.
    
    Args:
        sspcab (bool, optional): Enable SSPCAB training. Defaults to ``False``.
        severity_head_mode (str, optional): SeverityHead mode.
            Options: "single_scale" (act6 only), "multi_scale" (act2~act6).
            Defaults to ``"single_scale"``.
        severity_head_hidden_dim (int, optional): Hidden dimension for SeverityHead.
            Defaults to ``128``.
        score_combination (str, optional): Method to combine mask and severity scores.
            Options: "simple_average", "weighted_average", "maximum".
            Defaults to ``"simple_average"``.
        severity_weight_for_combination (float, optional): Weight for severity score
            in weighted_average combination. Defaults to ``0.5``.
            
    Example:
        >>> model = DraemSevNetModel(severity_head_mode="multi_scale")
        >>> input_tensor = torch.randn(8, 3, 224, 224)
        >>> # Training mode
        >>> model.train()
        >>> reconstruction, mask_logits, severity_score = model(input_tensor)
        >>> # Inference mode
        >>> model.eval()
        >>> output = model(input_tensor)
        >>> assert isinstance(output, DraemSevNetOutput)
    """
    
    def __init__(
        self, 
        sspcab: bool = False,
        severity_head_mode: str = "single_scale",
        severity_head_hidden_dim: int = 128,
        score_combination: str = "simple_average",
        severity_weight_for_combination: float = 0.5
    ) -> None:
        super().__init__()
        
        self.severity_head_mode = severity_head_mode
        self.score_combination = score_combination
        self.severity_weight_for_combination = severity_weight_for_combination
        
        # DRAEM backbone components
        draem_backbone = DraemModel(sspcab=sspcab)
        self.reconstructive_subnetwork = draem_backbone.reconstructive_subnetwork
        self.discriminative_subnetwork = DiscriminativeSubNetwork(
            in_channels=6, 
            out_channels=2, 
            expose_features=True
        )
        
        # Severity prediction head
        if severity_head_mode == "single_scale":
            # Use act6 features (512 channels for base_width=64)
            self.severity_head = SeverityHead(
                in_dim=512, 
                hidden_dim=severity_head_hidden_dim,
                mode="single_scale"
            )
        elif severity_head_mode == "multi_scale":
            # Use act2~act6 features (base_width=64)
            self.severity_head = SeverityHead(
                mode="multi_scale",
                base_width=64,
                hidden_dim=severity_head_hidden_dim
            )
        else:
            raise ValueError(f"Unsupported severity_head_mode: {severity_head_mode}. "
                           f"Choose from ['single_scale', 'multi_scale']")
    
    def _get_mask_score(self, mask_logits: torch.Tensor) -> torch.Tensor:
        """Calculate reliable mask-based anomaly score.
        
        Uses manual softmax + amax calculation for consistent and reliable results,
        avoiding the inconsistency found in original DRAEM's pred_score calculation.
        
        Args:
            mask_logits: Raw mask prediction logits of shape (B, 2, H, W)
            
        Returns:
            torch.Tensor: Mask-based anomaly scores of shape (B,) in range [0, 1]
        """
        # Apply softmax to get probabilities
        anomaly_probabilities = torch.softmax(mask_logits, dim=1)[:, 1, ...]  # (B, H, W)
        
        # Take maximum probability as image-level score
        mask_score = torch.amax(anomaly_probabilities, dim=(-2, -1))  # (B,)
        
        return mask_score
    
    def _combine_scores(
        self, 
        mask_score: torch.Tensor, 
        severity_score: torch.Tensor
    ) -> torch.Tensor:
        """Combine mask and severity scores using specified combination method.
        
        Args:
            mask_score: Mask-based anomaly scores of shape (B,)
            severity_score: Severity prediction scores of shape (B,)
            
        Returns:
            torch.Tensor: Combined final scores of shape (B,) in range [0, 1]
        """
        if self.score_combination == "simple_average":
            return (mask_score + severity_score) / 2.0
        elif self.score_combination == "weighted_average":
            weight = self.severity_weight_for_combination
            return (1 - weight) * mask_score + weight * severity_score
        elif self.score_combination == "maximum":
            return torch.maximum(mask_score, severity_score)
        else:
            raise ValueError(f"Unsupported score_combination: {self.score_combination}. "
                           f"Choose from ['simple_average', 'weighted_average', 'maximum']")
    
    def forward(
        self, 
        batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | DraemSevNetOutput:
        """Forward pass through DRAEM-SevNet architecture.
        
        Args:
            batch (torch.Tensor): Input batch of 3-channel RGB images of shape
                ``(batch_size, 3, height, width)``
        
        Returns:
            During training:
                tuple: Tuple containing:
                    - reconstruction: Reconstructed images (batch_size, 3, H, W)
                    - mask_logits: Raw mask prediction logits (batch_size, 2, H, W)
                    - severity_score: Severity prediction scores (batch_size,) in [0, 1]
            During inference:
                DraemSevNetOutput: Complete output containing all scores and intermediate results
        """
        # 1. Reconstruction step
        reconstruction = self.reconstructive_subnetwork(batch)
        
        # 2. Discriminative prediction + feature extraction
        concatenated_inputs = torch.cat([batch, reconstruction], dim=1)  # (B, 6, H, W)
        mask_logits, encoder_features = self.discriminative_subnetwork(concatenated_inputs)
        
        # 3. Severity prediction from encoder features
        if self.severity_head_mode == "single_scale":
            # Use act6 features only
            severity_score = self.severity_head(encoder_features['act6'])
        elif self.severity_head_mode == "multi_scale":
            # Use act2~act6 features
            multi_scale_features = {
                key: encoder_features[key] 
                for key in ['act2', 'act3', 'act4', 'act5', 'act6']
            }
            severity_score = self.severity_head(multi_scale_features)
        else:
            raise ValueError(f"Unsupported severity_head_mode: {self.severity_head_mode}")
        
        if self.training:
            # Training mode: return components for loss calculation
            return reconstruction, mask_logits, severity_score
        
        # 4. Inference mode: calculate final scores
        
        # Calculate reliable mask-based score
        mask_score = self._get_mask_score(mask_logits)
        
        # Combine mask and severity scores
        final_score = self._combine_scores(mask_score, severity_score)
        
        # Generate anomaly probability map for visualization
        anomaly_map = torch.softmax(mask_logits, dim=1)[:, 1, ...]  # (B, H, W)
        
        return DraemSevNetOutput(
            reconstruction=reconstruction,
            mask_logits=mask_logits,
            severity_score=severity_score,
            mask_score=mask_score,
            final_score=final_score,
            anomaly_map=anomaly_map
        )


# Export the main classes
__all__ = ["DraemSevNetModel", "DiscriminativeSubNetwork", "DraemSevNetOutput"]