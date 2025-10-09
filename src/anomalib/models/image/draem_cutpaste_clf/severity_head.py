"""Severity Head for DRAEM CutPaste Classification.

This module implements the CNN classification head that takes original image and predicted mask
as input to perform binary classification (normal/anomaly).

Based on CNN_simple from DRAME_CutPaste/utils/utils_model_for_HDmap_v4.py
"""

import torch
from torch import nn


class SeverityHead(nn.Module):
    """CNN Classification head for DRAEM CutPaste model.

    Takes multi-channel input (original image + predicted mask) and outputs binary classification.
    Based on the CNN_simple architecture from the original DRAME_CutPaste implementation v4.
    Optimized for 128x128 input resolution.

    Key success factors from original model:
    1. Natural feature map size preservation (no adaptive pooling)
    2. Minimal information loss
    3. Proper regularization (BatchNorm + Dropout)

    Args:
        in_channels (int, optional): Number of input channels. Defaults to ``2`` (image + mask).
        dropout_rate (float, optional): Dropout rate for regularization. Defaults to ``0.3``.
        input_size (tuple[int, int], optional): Expected input size (H, W).
            Used for calculating FC layer dimensions. Defaults to ``(128, 128)``.

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

        # Based on CNN_simple from utils_model_for_HDmap_v4.py
        # Architecture optimized for 128x128 input

        # Convolutional layers: 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3)   # 128x128 -> 64x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3)            # 64x64 -> 32x32
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)           # 32x32 -> 16x16
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)          # 16x16 -> 8x8
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)          # 8x8 -> 4x4

        # 🚫 No adaptive_pool: Following original model's success factor
        # After conv5: 4x4, so 512 * 4 * 4 = 8,192
        # Original model approach: Use natural size directly

        # 512 * 4 * 4 = 8,192 (preserving rich spatial information)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)  # First FC
        self.fc2 = nn.Linear(1024, 256)          # Second FC
        self.fc3 = nn.Linear(256, 2)             # Output layer

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)

        # Activation function
        self.relu = nn.ReLU()

        # Original model style regularization
        self.dropout_conv = nn.Dropout2d(dropout_rate * 0.5)  # For conv layers
        self.dropout_fc = nn.Dropout(dropout_rate)            # For FC layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the severity head.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W)
                Channels can be: original image, predicted mask, reconstruction

        Returns:
            torch.Tensor: Classification logits of shape (batch_size, 2)
        """
        # Conv layers with BatchNorm and strategic Dropout (matching v4)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout_conv(x)  # First dropout after 2nd conv

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.dropout_conv(x)  # Second dropout after 4th conv

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.dropout_conv(x)  # Third dropout after 5th conv

        # 🎯 Original model approach: Direct flatten without adaptive_pool
        x = x.view(x.size(0), -1)  # 4x4x512 = 8,192 features directly flattened

        # FC layers with Dropout
        x = self.dropout_fc(self.relu(self.fc1(x)))
        x = self.dropout_fc(self.relu(self.fc2(x)))
        x = self.fc3(x)  # No dropout on final layer

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
            "architecture": "CNN_simple_v4_128x128",
            "num_classes": 2,
            "conv_channels": [32, 64, 128, 256, 512],
            "fc_layers": [8192, 1024, 256, 2],
            "kernel_sizes": [7, 7, 5, 5, 3],
            "final_feature_map": "4x4x512"
        }


