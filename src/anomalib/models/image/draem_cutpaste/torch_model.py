"""PyTorch model for DRAEM CutPaste implementation.

This module implements DRAEM with CutPaste augmentation.
No severity head - uses DRAEM's original anomaly score calculation (max of anomaly map).
"""

import torch
from torch import nn

from anomalib.data import InferenceBatch
from anomalib.models.image.draem.torch_model import (
    DiscriminativeSubNetwork,
    ReconstructiveSubNetwork,
)

from anomalib.models.image.draem_cutpaste_clf.synthetic_generator_v2 import CutPasteSyntheticGenerator


class DraemCutPasteModel(nn.Module):
    """DRAEM CutPaste PyTorch model with reconstructive and discriminative networks.

    This model extends the original DRAEM architecture with:
    1. CutPaste-based synthetic fault generation (instead of Perlin noise)
    2. 1-channel optimization for grayscale data
    3. Uses DRAEM's original anomaly scoring (max of anomaly map)

    Args:
        sspcab (bool, optional): Enable SSPCAB training in reconstructive network.
            Defaults to ``False``.

        # CutPaste generator parameters
        cut_w_range (tuple[int, int], optional): Range of patch widths. Defaults to ``(10, 80)``.
        cut_h_range (tuple[int, int], optional): Range of patch heights. Defaults to ``(1, 2)``.
        a_fault_start (float, optional): Minimum fault amplitude. Defaults to ``1.0``.
        a_fault_range_end (float, optional): Maximum fault amplitude. Defaults to ``10.0``.
        augment_probability (float, optional): Probability of applying augmentation. Defaults to ``0.5``.

    Example:
        >>> model = DraemCutPasteModel(sspcab=False)
        >>> input_tensor = torch.randn(4, 1, 256, 256)
        >>> reconstruction, prediction, anomaly_mask, anomaly_labels = model(input_tensor)
    """

    def __init__(
        self,
        sspcab: bool = False,
        # CutPaste generator parameters
        cut_w_range: tuple[int, int] = (10, 80),
        cut_h_range: tuple[int, int] = (1, 2),
        a_fault_start: float = 1.0,
        a_fault_range_end: float = 10.0,
        augment_probability: float = 0.5,
    ) -> None:
        super().__init__()

        # Core DRAEM networks (1-channel configuration)
        self.reconstructive_subnetwork = ReconstructiveSubNetwork(
            in_channels=1, out_channels=1, sspcab=sspcab
        )
        self.discriminative_subnetwork = DiscriminativeSubNetwork(
            in_channels=2, out_channels=2, base_width=128
        )

        # CutPaste synthetic generator for training
        self.synthetic_generator = CutPasteSyntheticGenerator(
            cut_w_range=cut_w_range,
            cut_h_range=cut_h_range,
            a_fault_start=a_fault_start,
            a_fault_range_end=a_fault_range_end,
            probability=augment_probability,
            validation_enabled=True
        )

    def forward(
        self,
        batch: torch.Tensor,
        training_phase: bool = True
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | InferenceBatch:
        """Forward pass through all sub-networks.

        Args:
            batch (torch.Tensor): Input batch of images of shape
                ``(batch_size, channels, height, width)``
            training_phase (bool, optional): Whether in training phase.
                If True, applies CutPaste augmentation. Defaults to ``True``.

        Returns:
            During training:
                tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                    - reconstruction: Reconstructed images from original input
                    - prediction: Anomaly masks (discriminative network output)
                    - anomaly_mask: Ground truth anomaly masks
                    - anomaly_labels: Ground truth fault severity labels

            During inference:
                InferenceBatch: Batch containing anomaly scores and predictions
        """
        if self.training and training_phase:
            return self._forward_training(batch)
        else:
            return self._forward_inference(batch)

    def _forward_training(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass during training with CutPaste augmentation.

        Args:
            batch (torch.Tensor): Original input images

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                (reconstruction, prediction, anomaly_mask, anomaly_labels)
        """
        # Apply CutPaste augmentation
        augmented_batch, anomaly_mask, anomaly_labels = self.synthetic_generator(batch)

        # Extract single channel for 1-channel model
        augmented_single_ch = augmented_batch[:, :1, :, :]

        # Pass through reconstructive network (reconstruct augmented images)
        reconstruction = self.reconstructive_subnetwork(augmented_single_ch)

        # Concatenate augmented and reconstruction for discriminative network
        joined_input = torch.cat([augmented_single_ch, reconstruction], dim=1)

        # Pass through discriminative network to get anomaly prediction
        prediction = self.discriminative_subnetwork(joined_input)

        return reconstruction, prediction, anomaly_mask, anomaly_labels

    def _forward_inference(self, batch: torch.Tensor) -> InferenceBatch:
        """Forward pass during inference without augmentation.

        Args:
            batch (torch.Tensor): Input images

        Returns:
            InferenceBatch: Batch containing anomaly scores and predictions
        """
        # Extract single channel for 1-channel model
        batch_single_ch = batch[:, :1, :, :]

        # Pass through reconstructive network (no augmentation during inference)
        reconstruction = self.reconstructive_subnetwork(batch_single_ch)

        # Concatenate original and reconstruction for discriminative network
        joined_input = torch.cat([batch_single_ch, reconstruction], dim=1)

        # Pass through discriminative network
        prediction = self.discriminative_subnetwork(joined_input)

        # Create anomaly maps (apply softmax like DRAEM)
        anomaly_map = torch.softmax(prediction, dim=1)[:, 1, ...]

        # Compute anomaly score as max of anomaly map (DRAEM style)
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
