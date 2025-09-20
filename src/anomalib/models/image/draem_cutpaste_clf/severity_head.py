"""Severity Head for DRAEM CutPaste Classification.

This module implements the CNN classification head that takes original image and predicted mask
as input to perform binary classification (normal/anomaly).

Based on CNN_simple from DRAME_CutPaste/utils/utils_model_for_HDmap_v2.py
"""

import torch
from torch import nn


class SeverityHead(nn.Module):
    """CNN Classification head for DRAEM CutPaste model.

    Takes 2-channel input (original image + predicted mask) and outputs binary classification.
    Based on the CNN_simple architecture from the original DRAME_CutPaste implementation.

    Args:
        in_channels (int, optional): Number of input channels. Defaults to ``2`` (image + mask).
        dropout_rate (float, optional): Dropout rate for regularization. Defaults to ``0.3``.
        input_size (tuple[int, int], optional): Expected input size (H, W).
            Used for calculating FC layer dimensions. Defaults to ``(256, 256)``.

    Example:
        >>> severity_head = SeverityHead(in_channels=2, input_size=(256, 256))
        >>> # Input: [batch_size, 2, 256, 256] (original + mask)
        >>> input_tensor = torch.randn(4, 2, 256, 256)
        >>> output = severity_head(input_tensor)  # Shape: [4, 2] (logits)
        >>> probabilities = torch.softmax(output, dim=1)  # [4, 2] (probabilities)
    """

    def __init__(
        self,
        in_channels: int = 2,
        dropout_rate: float = 0.3,
        input_size: tuple[int, int] = (256, 256)
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.dropout_rate = dropout_rate
        self.input_size = input_size

        # Based on CNN_simple from utils_model_for_HDmap_v2.py
        # Architecture optimized for 256x256 input with 2-channel input

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=11, stride=2, padding=5)  # 256->128
        self.conv2 = nn.Conv2d(32, 64, kernel_size=11, stride=2, padding=5)           # 128->64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=11, stride=2, padding=5)          # 64->32
        self.conv4 = nn.Conv2d(128, 256, kernel_size=7, stride=2, padding=3)          # 32->16

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        # Activation function
        self.relu = nn.ReLU(inplace=True)

        # Dropout layers
        self.dropout_conv = nn.Dropout2d(dropout_rate * 0.5)  # Lower rate for conv layers
        self.dropout_fc = nn.Dropout(dropout_rate)            # Full rate for FC layers

        # Calculate FC input size dynamically based on actual output size
        self._fc_input_size = None
        self.fc1 = None
        self.fc2 = None
        self.fc3 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the severity head.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2, H, W)
                First channel: original image, Second channel: predicted mask

        Returns:
            torch.Tensor: Classification logits of shape (batch_size, 2)
        """
        # Convolutional layers with BatchNorm, ReLU, and Dropout
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout_conv(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout_conv(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout_conv(x)

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.dropout_conv(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Initialize FC layers dynamically on first forward pass
        if self.fc1 is None:
            fc_input_size = x.size(1)
            self.fc1 = nn.Linear(fc_input_size, 512).to(x.device)
            self.fc2 = nn.Linear(512, 128).to(x.device)
            self.fc3 = nn.Linear(128, 2).to(x.device)

        # Fully connected layers with Dropout
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
            "architecture": "CNN_simple_2ch",
            "num_classes": 2,
            "conv_channels": [32, 64, 128, 256],
            "fc_layers": [512, 128, 2]
        }


