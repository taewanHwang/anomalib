"""Severity Head for DRAEM-SevNet.

Global Average Pooling + MLP 구조로 discriminative encoder features를 
severity score [0,1]로 변환하는 네트워크.

Author: Taewan Hwang
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Union


class SeverityHead(nn.Module):
    """Severity prediction head for DRAEM-SevNet.
    
    Discriminative encoder의 features (act6 또는 multi-scale)를 입력으로 받아
    Global Average Pooling과 MLP를 통해 severity score [0,1]를 출력합니다.
    
    Args:
        in_dim (int): 입력 feature dimension
        hidden_dim (int, optional): Hidden layer dimension. Defaults to ``128``.
        mode (str, optional): Feature scale mode. 
            Options: "single_scale" (act6 only), "multi_scale" (act2~act6).
            Defaults to ``"single_scale"``.
        base_width (int, optional): Base width for multi-scale mode. Defaults to ``64``.
        dropout_rate (float, optional): Dropout rate. Defaults to ``0.1``.
        
    Example:
        Single scale mode (act6 only):
        >>> severity_head = SeverityHead(in_dim=512, hidden_dim=128)
        >>> act6 = torch.randn(4, 512, 14, 14)  # [B, C, H, W]
        >>> severity = severity_head(act6)
        >>> severity.shape
        torch.Size([4])
        
        Multi-scale mode (act2~act6):
        >>> severity_head = SeverityHead(mode="multi_scale", base_width=64)
        >>> features = {
        ...     'act2': torch.randn(4, 128, 56, 56),
        ...     'act3': torch.randn(4, 256, 28, 28), 
        ...     'act4': torch.randn(4, 512, 14, 14),
        ...     'act5': torch.randn(4, 512, 7, 7),
        ...     'act6': torch.randn(4, 512, 7, 7)
        ... }
        >>> severity = severity_head(features)
        >>> severity.shape
        torch.Size([4])
    """
    
    def __init__(
        self,
        in_dim: int = None,
        hidden_dim: int = 128,
        mode: str = "single_scale",
        base_width: int = 64,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.mode = mode
        self.base_width = base_width
        
        # Feature dimension 계산
        if mode == "single_scale":
            if in_dim is None:
                raise ValueError("in_dim must be provided for single_scale mode")
            self.feature_dim = in_dim
        elif mode == "multi_scale":
            # act2~act6 channel dimensions: 2*base_width, 4*base_width, 8*base_width, 8*base_width, 8*base_width
            self.feature_dim = base_width * (2 + 4 + 8 + 8 + 8)  # 30 * base_width
        else:
            raise ValueError(f"Unsupported mode: {mode}. Choose 'single_scale' or 'multi_scale'")
        
        # MLP for severity regression
        self.severity_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # [0, 1] 범위로 제한
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _global_average_pooling(self, features: torch.Tensor) -> torch.Tensor:
        """Apply Global Average Pooling to features.
        
        Args:
            features (torch.Tensor): Input features of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: GAP result of shape [B, C]
        """
        return F.adaptive_avg_pool2d(features, 1).flatten(1)
    
    def _extract_single_scale_features(self, act6: torch.Tensor) -> torch.Tensor:
        """Extract features from single scale (act6 only).
        
        Args:
            act6 (torch.Tensor): Act6 features of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Extracted features of shape [B, feature_dim]
        """
        return self._global_average_pooling(act6)
    
    def _extract_multi_scale_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from multi-scale (act2~act6).
        
        Args:
            features (Dict[str, torch.Tensor]): Dictionary containing act2~act6 features
            
        Returns:
            torch.Tensor: Concatenated features of shape [B, feature_dim]
        """
        required_keys = ['act2', 'act3', 'act4', 'act5', 'act6']
        
        # Validate input
        for key in required_keys:
            if key not in features:
                raise ValueError(f"Missing required feature: {key}")
        
        # Apply GAP to each feature map and concatenate
        pooled_features = []
        for key in required_keys:
            pooled = self._global_average_pooling(features[key])
            pooled_features.append(pooled)
        
        return torch.cat(pooled_features, dim=1)
    
    def forward(self, input_features: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Forward pass for severity prediction.
        
        Args:
            input_features: 
                - For single_scale mode: torch.Tensor of shape [B, C, H, W] (act6)
                - For multi_scale mode: Dict containing act2~act6 features
                
        Returns:
            torch.Tensor: Severity scores of shape [B] with values in [0, 1]
        """
        if self.mode == "single_scale":
            if not isinstance(input_features, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor for single_scale mode, got {type(input_features)}")
            features = self._extract_single_scale_features(input_features)
        elif self.mode == "multi_scale":
            if not isinstance(input_features, dict):
                raise TypeError(f"Expected dict for multi_scale mode, got {type(input_features)}")
            features = self._extract_multi_scale_features(input_features)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        
        # Predict severity score [0, 1]
        severity_score = self.severity_mlp(features).squeeze(1)  # [B, 1] -> [B]
        
        return severity_score


class SeverityHeadFactory:
    """Factory class for creating SeverityHead instances."""
    
    @staticmethod
    def create_single_scale(in_dim: int, hidden_dim: int = 128, dropout_rate: float = 0.1) -> SeverityHead:
        """Create SeverityHead for single scale mode (act6 only).
        
        Args:
            in_dim (int): Input feature dimension (usually 512 for act6)
            hidden_dim (int, optional): Hidden layer dimension. Defaults to ``128``.
            dropout_rate (float, optional): Dropout rate. Defaults to ``0.1``.
            
        Returns:
            SeverityHead: Configured severity head
        """
        return SeverityHead(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            mode="single_scale",
            dropout_rate=dropout_rate
        )
    
    @staticmethod
    def create_multi_scale(base_width: int = 64, hidden_dim: int = 256, dropout_rate: float = 0.1) -> SeverityHead:
        """Create SeverityHead for multi-scale mode (act2~act6).
        
        Args:
            base_width (int, optional): Base width for calculating multi-scale dimensions. Defaults to ``64``.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to ``256``.
            dropout_rate (float, optional): Dropout rate. Defaults to ``0.1``.
            
        Returns:
            SeverityHead: Configured severity head
        """
        return SeverityHead(
            mode="multi_scale",
            base_width=base_width,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )


# Export classes
__all__ = ["SeverityHead", "SeverityHeadFactory"]
