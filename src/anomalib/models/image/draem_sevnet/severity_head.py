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
        # 🆕 새로운 Spatial-Aware 옵션들
        pooling_type: str = "gap",           # "gap", "spatial_aware"
        spatial_size: int = 4,               # spatial_aware 모드에서 보존할 공간 크기
        use_spatial_attention: bool = True,  # 공간 어텐션 사용 여부
    ) -> None:
        super().__init__()
        
        self.mode = mode
        self.base_width = base_width
        self.pooling_type = pooling_type
        self.spatial_size = spatial_size
        self.use_spatial_attention = use_spatial_attention
        
        # Feature dimension 계산
        if pooling_type == "gap":
            # 기존 GAP 방식
            if mode == "single_scale":
                if in_dim is None:
                    raise ValueError("in_dim must be provided for single_scale mode")
                self.feature_dim = in_dim
            elif mode == "multi_scale":
                # act2~act6 channel dimensions: 2*base_width, 4*base_width, 8*base_width, 8*base_width, 8*base_width
                self.feature_dim = base_width * (2 + 4 + 8 + 8 + 8)  # 30 * base_width
            else:
                raise ValueError(f"Unsupported mode: {mode}. Choose 'single_scale' or 'multi_scale'")
            
            # 기존 GAP용 MLP
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
        elif pooling_type == "spatial_aware":
            # 새로운 Spatial-Aware 방식
            self._init_spatial_aware_components(in_dim, hidden_dim, dropout_rate)
        else:
            raise ValueError(f"Unsupported pooling_type: {pooling_type}. Choose 'gap' or 'spatial_aware'")
        
        # Initialize weights
        self._initialize_weights()
    
    def _init_spatial_aware_components(self, in_dim: int, hidden_dim: int, dropout_rate: float) -> None:
        """Spatial-Aware 컴포넌트 초기화"""
        if self.mode == "single_scale":
            if in_dim is None:
                raise ValueError("in_dim must be provided for single_scale mode")
            
            # Single scale용 spatial attention
            if self.use_spatial_attention:
                self.spatial_attention = nn.Sequential(
                    nn.Conv2d(in_dim, 1, kernel_size=1),
                    nn.Sigmoid()
                )
            
            # Spatial reducer (GAP 대신)
            self.spatial_reducer = nn.Sequential(
                nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(self.spatial_size)  # 완전 제거 대신 축소
            )
            
            # 새로운 MLP (공간 정보 고려)
            spatial_features = hidden_dim * self.spatial_size * self.spatial_size
            self.spatial_severity_mlp = nn.Sequential(
                nn.Linear(spatial_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        
        elif self.mode == "multi_scale":
            # Multi-scale용 spatial processors
            self.scale_spatial_processors = nn.ModuleDict()
            scale_configs = {
                'act2': (self.base_width * 2, 32),
                'act3': (self.base_width * 4, 64), 
                'act4': (self.base_width * 8, 128),
                'act5': (self.base_width * 8, 128),
                'act6': (self.base_width * 8, 128),
            }
            
            for scale_name, (in_channels, out_channels) in scale_configs.items():
                self.scale_spatial_processors[scale_name] = self._make_spatial_processor(
                    in_channels, out_channels
                )
            
            # Multi-scale spatial MLP
            total_features = sum([32, 64, 128, 128, 128]) * self.spatial_size * self.spatial_size  # 공간 정보 포함
            self.multi_scale_spatial_mlp = nn.Sequential(
                nn.Linear(total_features, hidden_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
    
    def _make_spatial_processor(self, in_channels: int, out_channels: int) -> nn.Module:
        """각 스케일별 공간 정보 처리기"""
        layers = []
        
        # Spatial attention (옵션)
        if self.use_spatial_attention:
            layers.extend([
                nn.Conv2d(in_channels, 1, 1),
                nn.Sigmoid(),
            ])
        
        # Feature processing
        layers.extend([
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(self.spatial_size),  # 공간 정보 부분 보존
        ])
        
        return nn.Sequential(*layers)
    
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
        if self.pooling_type == "gap":
            # 기존 GAP 방식
            return self._forward_gap(input_features)
        elif self.pooling_type == "spatial_aware":
            # 새로운 Spatial-Aware 방식
            return self._forward_spatial_aware(input_features)
        else:
            raise ValueError(f"Unsupported pooling_type: {self.pooling_type}")

    def _forward_gap(self, input_features: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """기존 GAP 방식 (하위 호환성)"""
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
        
        severity_score = self.severity_mlp(features).squeeze(1)
        return severity_score

    def _forward_spatial_aware(self, input_features: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """새로운 Spatial-Aware 방식"""
        if self.mode == "single_scale":
            return self._forward_single_scale_spatial(input_features)
        elif self.mode == "multi_scale":
            return self._forward_multi_scale_spatial(input_features)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _forward_single_scale_spatial(self, act6: torch.Tensor) -> torch.Tensor:
        """Single scale spatial-aware forward"""
        if not isinstance(act6, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for single_scale mode, got {type(act6)}")
        
        # 1. Spatial attention 적용 (옵션)
        if self.use_spatial_attention:
            attention_map = self.spatial_attention(act6)  # [B, 1, H, W]
            attended_features = act6 * attention_map      # [B, C, H, W]
        else:
            attended_features = act6
        
        # 2. 공간 정보 부분 보존
        spatial_features = self.spatial_reducer(attended_features)  # [B, hidden_dim, spatial_size, spatial_size]
        
        # 3. Flatten하여 MLP 입력
        flattened = spatial_features.flatten(1)  # [B, hidden_dim * spatial_size^2]
        
        # 4. Severity 예측
        severity_score = self.spatial_severity_mlp(flattened).squeeze(1)  # [B]
        
        return severity_score

    def _forward_multi_scale_spatial(self, multi_scale_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Multi-scale spatial-aware forward"""
        if not isinstance(multi_scale_features, dict):
            raise TypeError(f"Expected dict for multi_scale mode, got {type(multi_scale_features)}")
        
        processed_features = []
        
        for scale_name, features in multi_scale_features.items():
            if scale_name in self.scale_spatial_processors:
                # 각 스케일별 spatial processing
                processor = self.scale_spatial_processors[scale_name]
                
                if self.use_spatial_attention:
                    # Attention 적용
                    attention = processor[0:2](features)  # Conv2d + Sigmoid
                    attended = features * attention
                    # Feature processing
                    processed = processor[2:](attended)  # 나머지 레이어들
                else:
                    # Attention 없이 바로 feature processing
                    processed = processor(features)
                
                processed = processed.flatten(1)
                processed_features.append(processed)
        
        # 모든 스케일 특징 결합
        combined_features = torch.cat(processed_features, dim=1)
        
        # Severity 예측
        severity_score = self.multi_scale_spatial_mlp(combined_features).squeeze(1)
        
        return severity_score


class SeverityHeadFactory:
    """Factory class for creating SeverityHead instances."""
    
    @staticmethod
    def create_single_scale(
        in_dim: int, 
        hidden_dim: int = 128, 
        dropout_rate: float = 0.1,
        pooling_type: str = "gap",
        spatial_size: int = 4,
        use_spatial_attention: bool = True
    ) -> SeverityHead:
        """Create SeverityHead for single scale mode (act6 only).
        
        Args:
            in_dim (int): Input feature dimension (usually 512 for act6)
            hidden_dim (int, optional): Hidden layer dimension. Defaults to ``128``.
            dropout_rate (float, optional): Dropout rate. Defaults to ``0.1``.
            pooling_type (str, optional): Pooling type. Defaults to ``"gap"``.
            spatial_size (int, optional): Spatial size for spatial_aware mode. Defaults to ``4``.
            use_spatial_attention (bool, optional): Use spatial attention. Defaults to ``True``.
            
        Returns:
            SeverityHead: Configured severity head
        """
        return SeverityHead(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            mode="single_scale",
            dropout_rate=dropout_rate,
            pooling_type=pooling_type,
            spatial_size=spatial_size,
            use_spatial_attention=use_spatial_attention
        )
    
    @staticmethod
    def create_multi_scale(
        base_width: int = 64, 
        hidden_dim: int = 256, 
        dropout_rate: float = 0.1,
        pooling_type: str = "gap",
        spatial_size: int = 4,
        use_spatial_attention: bool = True
    ) -> SeverityHead:
        """Create SeverityHead for multi-scale mode (act2~act6).
        
        Args:
            base_width (int, optional): Base width for calculating multi-scale dimensions. Defaults to ``64``.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to ``256``.
            dropout_rate (float, optional): Dropout rate. Defaults to ``0.1``.
            pooling_type (str, optional): Pooling type. Defaults to ``"gap"``.
            spatial_size (int, optional): Spatial size for spatial_aware mode. Defaults to ``4``.
            use_spatial_attention (bool, optional): Use spatial attention. Defaults to ``True``.
            
        Returns:
            SeverityHead: Configured severity head
        """
        return SeverityHead(
            mode="multi_scale",
            base_width=base_width,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            pooling_type=pooling_type,
            spatial_size=spatial_size,
            use_spatial_attention=use_spatial_attention
        )
    
    @staticmethod
    def create_spatial_aware_single_scale(
        in_dim: int, 
        hidden_dim: int = 128, 
        spatial_size: int = 4,
        use_spatial_attention: bool = True,
        dropout_rate: float = 0.1
    ) -> SeverityHead:
        """Create Spatial-Aware SeverityHead for single scale mode.
        
        Args:
            in_dim (int): Input feature dimension (usually 512 for act6)
            hidden_dim (int, optional): Hidden layer dimension. Defaults to ``128``.
            spatial_size (int, optional): Spatial size to preserve. Defaults to ``4``.
            use_spatial_attention (bool, optional): Use spatial attention. Defaults to ``True``.
            dropout_rate (float, optional): Dropout rate. Defaults to ``0.1``.
            
        Returns:
            SeverityHead: Configured spatial-aware severity head
        """
        return SeverityHead(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            mode="single_scale",
            dropout_rate=dropout_rate,
            pooling_type="spatial_aware",
            spatial_size=spatial_size,
            use_spatial_attention=use_spatial_attention
        )
    
    @staticmethod
    def create_spatial_aware_multi_scale(
        base_width: int = 64, 
        hidden_dim: int = 256, 
        spatial_size: int = 4,
        use_spatial_attention: bool = True,
        dropout_rate: float = 0.1
    ) -> SeverityHead:
        """Create Spatial-Aware SeverityHead for multi-scale mode.
        
        Args:
            base_width (int, optional): Base width for calculating multi-scale dimensions. Defaults to ``64``.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to ``256``.
            spatial_size (int, optional): Spatial size to preserve. Defaults to ``4``.
            use_spatial_attention (bool, optional): Use spatial attention. Defaults to ``True``.
            dropout_rate (float, optional): Dropout rate. Defaults to ``0.1``.
            
        Returns:
            SeverityHead: Configured spatial-aware severity head
        """
        return SeverityHead(
            mode="multi_scale",
            base_width=base_width,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            pooling_type="spatial_aware",
            spatial_size=spatial_size,
            use_spatial_attention=use_spatial_attention
        )


# Export classes
__all__ = ["SeverityHead", "SeverityHeadFactory"]
