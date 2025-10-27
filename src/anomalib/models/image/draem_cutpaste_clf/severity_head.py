"""Severity Head for DRAEM CutPaste Classification.

This module implements a shallow CNN classification head that takes original image
and predicted mask as input to perform binary classification (normal/anomaly).

Shallow architecture to allow input channel differences (original vs mask vs original+mask)
to be more clearly distinguished in performance.
"""

import torch
from torch import nn


class SeverityHead(nn.Module):
    """Shallow CNN Classification head for DRAEM CutPaste model.

    Takes multi-channel input (original image + predicted mask) and outputs binary classification.
    Uses a deliberately shallow architecture (2 conv layers) to make the model less powerful,
    allowing the benefits of additional input channels (mask) to be more apparent.

    Architecture (for 128x128 input):
        Conv1(5x5, 32, stride=1) -> ReLU -> MaxPool(3x3, stride=2)  [128->128->64]
        Conv2(5x5, 48, stride=1) -> ReLU -> MaxPool(3x3, stride=2)  [64->64->32]
        Flatten -> FC(100) -> ReLU -> FC(100) -> ReLU -> FC(2)

    Args:
        in_channels (int, optional): Number of input channels. Defaults to ``2`` (image + mask).
        dropout_rate (float, optional): Dropout rate (kept for compatibility, not used).
            Defaults to ``0.3``.
        input_size (tuple[int, int], optional): Expected input size (H, W).
            Defaults to ``(128, 128)``.

    Example:
        >>> severity_head = SeverityHead(in_channels=2, input_size=(128, 128))
        >>> # Input: [batch_size, 2, 128, 128] (original + mask)
        >>> input_tensor = torch.randn(4, 2, 128, 128)
        >>> output = severity_head(input_tensor)  # Shape: [4, 2] (logits)
        >>> probabilities = torch.softmax(output, dim=1)  # [4, 2] (probabilities)
    """

    def __init__(
        self,
        in_channels: int = 2,
        dropout_rate: float = 0.3,
        input_size: tuple[int, int] = (128, 128)
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.dropout_rate = dropout_rate
        self.input_size = input_size

        # Shallow CNN architecture
        # Input: 128x128
        # Conv1: 128x128 -> 128x128 (stride=1, pad=2 maintains size)
        # MaxPool1: 128x128 -> 64x64 (stride=2)
        # Conv2: 64x64 -> 64x64 (stride=1, pad=2 maintains size)
        # MaxPool2: 64x64 -> 32x32 (stride=2)

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Calculate flattened size: 48 channels * 32 * 32 = 49,152
        self.flatten_size = 48 * 32 * 32

        # Simple FC layers (shallow) with gradual dimension reduction
        self.fc1 = nn.Linear(self.flatten_size, 2000)  # 49152 -> 2000
        self.fc2 = nn.Linear(2000, 100)                 # 2000 -> 100
        self.fc3 = nn.Linear(100, 2)                    # 100 -> 2

        # Activation
        self.relu = nn.ReLU()

        print(f"SeverityHead: Shallow architecture with {self.flatten_size} flattened features")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the severity head.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W)
                Channels can be: original image, predicted mask, or both

        Returns:
            torch.Tensor: Classification logits of shape (batch_size, 2)
        """
        # Shallow conv layers (NO BatchNorm, NO Dropout)
        x = self.conv1(x)       # (B, in_channels, 128, 128) -> (B, 32, 128, 128)
        x = self.relu(x)
        x = self.maxpool1(x)    # (B, 32, 128, 128) -> (B, 32, 64, 64)

        x = self.conv2(x)       # (B, 32, 64, 64) -> (B, 48, 64, 64)
        x = self.relu(x)
        x = self.maxpool2(x)    # (B, 48, 64, 64) -> (B, 48, 32, 32)

        # Flatten
        x = x.view(x.size(0), -1)  # (B, 48*32*32) = (B, 49152)

        # Simple FC layers (NO Dropout)
        x = self.fc1(x)         # (B, 49152) -> (B, 2000)
        x = self.relu(x)

        x = self.fc2(x)         # (B, 2000) -> (B, 100)
        x = self.relu(x)

        x = self.fc3(x)         # (B, 100) -> (B, 2)

        return x

    def get_config(self) -> dict:
        """Get configuration dictionary for the severity head.

        Returns:
            dict: Configuration parameters
        """
        return {
            "in_channels": self.in_channels,
            "dropout_rate": self.dropout_rate,
            "input_size": self.input_size,
            "architecture": "ShallowCNN_2conv",
            "num_classes": 2,
            "conv_channels": [32, 48],
            "fc_layers": [self.flatten_size, 2000, 100, 2],
            "kernel_sizes": [5, 5],
            "pooling": ["MaxPool3x3", "MaxPool3x3"],
            "final_feature_map": "48x32x32",
            "flatten_size": self.flatten_size
        }
