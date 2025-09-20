"""PyTorch model for DRAEM CutPaste Classification implementation.

This module implements DRAEM with CutPaste augmentation and CNN classification head.
Based on the original DRAEM architecture with additional severity head for binary classification.
"""

import torch
from torch import nn

from anomalib.data import InferenceBatch
from anomalib.models.image.draem.torch_model import (
    DiscriminativeSubNetwork,
    ReconstructiveSubNetwork,
)

from .severity_head import SeverityHead
from .synthetic_generator_v2 import CutPasteSyntheticGenerator


class DraemCutPasteModel(nn.Module):
    """DRAEM CutPaste PyTorch model with reconstructive, discriminative and severity networks.

    This model extends the original DRAEM architecture with:
    1. CutPaste-based synthetic fault generation (instead of Perlin noise)
    2. CNN-based classification head for binary anomaly detection
    3. Configurable normalization in augmentation

    Args:
        sspcab (bool, optional): Enable SSPCAB training in reconstructive network.
            Defaults to ``False``.
        image_size (tuple[int, int], optional): Expected input size for severity head.
            Used for calculating FC layer dimensions. Defaults to ``(256, 256)``.
        severity_dropout (float, optional): Dropout rate for severity head. Defaults to ``0.3``.

        # CutPaste generator parameters
        cut_w_range (tuple[int, int], optional): Range of patch widths. Defaults to ``(10, 80)``.
        cut_h_range (tuple[int, int], optional): Range of patch heights. Defaults to ``(1, 2)``.
        a_fault_start (float, optional): Minimum fault amplitude. Defaults to ``1.0``.
        a_fault_range_end (float, optional): Maximum fault amplitude. Defaults to ``10.0``.
        augment_probability (float, optional): Probability of applying augmentation. Defaults to ``0.5``.
        norm (bool, optional): Enable normalization in CutPaste. Defaults to ``True``.

    Example:
        >>> model = DraemCutPasteModel(
        ...     sspcab=False,
        ...     image_size=(256, 256),
        ...     norm=True
        ... )
        >>> input_tensor = torch.randn(4, 3, 256, 256)
        >>> reconstruction, prediction, classification = model(input_tensor)
    """

    def __init__(
        self,
        sspcab: bool = False,
        image_size: tuple[int, int] = (256, 256),
        severity_dropout: float = 0.3,
        # CutPaste generator parameters
        cut_w_range: tuple[int, int] = (10, 80),
        cut_h_range: tuple[int, int] = (1, 2),
        a_fault_start: float = 1.0,
        a_fault_range_end: float = 10.0,
        augment_probability: float = 0.5,
        norm: bool = True,
    ) -> None:
        super().__init__()

        # Core DRAEM networks (reuse from original implementation)
        self.reconstructive_subnetwork = ReconstructiveSubNetwork(sspcab=sspcab)
        self.discriminative_subnetwork = DiscriminativeSubNetwork(in_channels=6, out_channels=2)

        # New: Severity head for classification (2-channel input: original + mask)
        self.severity_head = SeverityHead(
            in_channels=2,
            dropout_rate=severity_dropout,
            input_size=image_size
        )

        # CutPaste synthetic generator for training
        self.synthetic_generator = CutPasteSyntheticGenerator(
            cut_w_range=cut_w_range,
            cut_h_range=cut_h_range,
            a_fault_start=a_fault_start,
            a_fault_range_end=a_fault_range_end,
            probability=augment_probability,
            norm=norm,
            validation_enabled=True
        )

    def forward(
        self,
        batch: torch.Tensor,
        training_phase: bool = True
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | InferenceBatch:
        """Forward pass through all sub-networks.

        Args:
            batch (torch.Tensor): Input batch of images of shape
                ``(batch_size, channels, height, width)``
            training_phase (bool, optional): Whether in training phase.
                If True, applies CutPaste augmentation. Defaults to ``True``.

        Returns:
            During training:
                tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                    - reconstruction: Reconstructed images from original input
                    - prediction: Anomaly masks (discriminative network output)
                    - classification: Classification logits from severity head
                    - anomaly_mask: Ground truth anomaly masks
                    - anomaly_labels: Ground truth anomaly labels

            During inference:
                InferenceBatch: Batch containing anomaly scores and predictions
        """
        if self.training and training_phase:
            return self._forward_training(batch)
        else:
            return self._forward_inference(batch)

    def _forward_training(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass during training with CutPaste augmentation.

        Args:
            batch (torch.Tensor): Original input images

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                (reconstruction, prediction, classification)
        """
        # Apply CutPaste augmentation
        augmented_batch, anomaly_mask, _, anomaly_labels = self.synthetic_generator(batch)

        # Pass through reconstructive network (reconstruct augmented images)
        reconstruction = self.reconstructive_subnetwork(augmented_batch)

        # Concatenate reconstruction and augmented images for discriminative network
        # DRAEM expects 6 channels: [reconstructed(3) + augmented(3)]
        joined_input = torch.cat([reconstruction, augmented_batch], dim=1)

        # Pass through discriminative network to get anomaly prediction
        prediction = self.discriminative_subnetwork(joined_input)

        # Prepare input for severity head: [augmented_1ch + predicted_mask_1ch]
        # Use augmented image instead of original to provide meaningful signal
        augmented_single_ch = augmented_batch[:, :1, :, :]  # First channel of augmented image
        predicted_mask = prediction[:, 1:2, :, :]  # Anomaly channel from prediction

        severity_input = torch.cat([augmented_single_ch, predicted_mask], dim=1)

        # Pass through severity head for classification
        classification = self.severity_head(severity_input)

        return reconstruction, prediction, classification, anomaly_mask, anomaly_labels

    def _forward_inference(self, batch: torch.Tensor) -> InferenceBatch:
        """Forward pass during inference without augmentation.

        Args:
            batch (torch.Tensor): Input images

        Returns:
            InferenceBatch: Batch containing anomaly scores and predictions
        """
        # Pass through reconstructive network (no augmentation during inference)
        reconstruction = self.reconstructive_subnetwork(batch)

        # Concatenate reconstruction and original images for discriminative network
        joined_input = torch.cat([reconstruction, batch], dim=1)

        # Pass through discriminative network
        prediction = self.discriminative_subnetwork(joined_input)

        # Prepare input for severity head
        original_single_ch = batch[:, :1, :, :]
        predicted_mask = prediction[:, 1:2, :, :]
        severity_input = torch.cat([original_single_ch, predicted_mask], dim=1)

        # Get classification probabilities
        classification_logits = self.severity_head(severity_input)
        classification_probs = torch.softmax(classification_logits, dim=1)

        # Extract anomaly scores (use softmax probabilities for anomaly class)
        anomaly_scores = classification_probs[:, 1]  # Probability of anomaly class

        # Create anomaly maps (use predicted mask as anomaly map)
        anomaly_maps = predicted_mask.squeeze(1)  # Remove channel dimension

        # Create predictions (binary threshold at 0.5)
        predictions = (anomaly_scores > 0.5).float()

        return InferenceBatch(
            anomaly_map=anomaly_maps,
            pred_score=anomaly_scores,
            pred_label=predictions.long(),
            pred_mask=anomaly_maps,  # Use same as anomaly_map
        )

    def get_model_config(self) -> dict:
        """Get configuration information for the model.

        Returns:
            dict: Model configuration including all parameters
        """
        return {
            "model_name": "DraemCutPasteClf",
            "reconstructive_network": "ReconstructiveSubNetwork",
            "discriminative_network": "DiscriminativeSubNetwork",
            "severity_head": self.severity_head.get_config(),
            "synthetic_generator": self.synthetic_generator.get_config_info(),
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


