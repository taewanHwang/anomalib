"""CutPaste Classifier - Simple CNN Model (Table B1 Architecture).

This module implements a baseline comparison model using only CutPaste augmentation
and a simple CNN classifier without reconstruction or localization.

Architecture based on Table B1:
- Conv1: 5x5, 32 channels, stride=1, padding=2
- Pooling: 3x3, stride=2, padding=1
- Conv2: 5x5, 48 channels, stride=1, padding=2
- Pooling: 3x3, stride=2, padding=1
- FC1: 100 nodes
- FC2: 100 nodes
- Softmax: 2 outputs (Normal/Anomaly)
"""

import torch
import torch.nn as nn
from typing import Tuple

from anomalib.data import InferenceBatch


class CutPasteClf(nn.Module):
    """Simple CNN Classifier for CutPaste augmented images.

    Pure classification approach without reconstruction or localization.
    Architecture follows Table B1 baseline structure.

    Args:
        image_size: Input image size (H, W)
    """

    def __init__(self, image_size: Tuple[int, int] = (128, 128)):
        super().__init__()

        # Debug: Print module location
        import inspect
        print(f"\n🔍 CutPasteClf loaded from: {inspect.getfile(self.__class__)}")

        self.image_size = image_size

        # Table B1 Architecture
        # Convolution 1: Size=5, Ch=32, Stride=1, Padding=2
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()

        # Pooling: Size=3, Stride=2, Padding=1
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Convolution 2: Size=5, Ch=48, Stride=1, Padding=2
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()

        # Pooling: Size=3, Stride=2, Padding=1
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Calculate flattened size after convolutions and pooling
        # MaxPool2d(kernel_size=3, stride=2, padding=1):
        # output = (input + 2*padding - kernel_size) // stride + 1
        h, w = image_size

        # After pool1
        h_pool1 = (h + 2*1 - 3) // 2 + 1
        w_pool1 = (w + 2*1 - 3) // 2 + 1

        # After pool2
        h_out = (h_pool1 + 2*1 - 3) // 2 + 1
        w_out = (w_pool1 + 2*1 - 3) // 2 + 1

        flatten_size = 48 * h_out * w_out

        print(f"CutPasteClf initialized:")
        print(f"  Input size: {image_size}")
        print(f"  After pool1: {h_pool1}x{w_pool1}")
        print(f"  After pool2: {h_out}x{w_out}")
        print(f"  Flatten size: {flatten_size}")

        # Fully connected layers: Node=100
        self.fc1 = nn.Linear(flatten_size, 100)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(100, 100)
        self.relu4 = nn.ReLU()

        # Softmax: Output=2 (Normal/Anomaly)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the simple CNN.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            logits: Classification logits [B, 2]
        """
        # Convolution + ReLU + Pooling
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Convolution + ReLU + Pooling
        x = self.conv2(x)
        x = self.relu2(x)

        # Debug: print shape before pool2
        if not hasattr(self, '_before_pool2_printed'):
            print(f"  Before pool2: {x.shape}")
            self._before_pool2_printed = True

        x = self.pool2(x)

        # Debug: print shape after pool2
        if not hasattr(self, '_after_pool2_printed'):
            print(f"  After pool2: {x.shape}")
            print(f"  pool2 config: {self.pool2}")
            self._after_pool2_printed = True

        # Debug: print shape before flatten (only once)
        if not hasattr(self, '_shape_printed'):
            print(f"  Before flatten: {x.shape}")
            self._shape_printed = True

        # Flatten
        x = x.flatten(1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        x = self.relu4(x)

        # Output layer (logits)
        logits = self.fc3(x)

        return logits
