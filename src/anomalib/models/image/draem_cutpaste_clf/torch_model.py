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

    Args:
        sspcab (bool, optional): Enable SSPCAB training in reconstructive network.
            Defaults to ``False``.
        image_size (tuple[int, int], optional): Expected input size for severity head.
            Used for calculating FC layer dimensions. Defaults to ``(256, 256)``.
        severity_dropout (float, optional): Dropout rate for severity head. Defaults to ``0.3``.
        severity_input_channels (str, optional): Channels to use for severity head input.
            Options: "original", "mask", "recon", "original+mask", "original+recon", 
            "mask+recon", "original+mask+recon". Defaults to ``"original+mask+recon"``.

        # CutPaste generator parameters
        cut_w_range (tuple[int, int], optional): Range of patch widths. Defaults to ``(10, 80)``.
        cut_h_range (tuple[int, int], optional): Range of patch heights. Defaults to ``(1, 2)``.
        a_fault_start (float, optional): Minimum fault amplitude. Defaults to ``1.0``.
        a_fault_range_end (float, optional): Maximum fault amplitude. Defaults to ``10.0``.
        augment_probability (float, optional): Probability of applying augmentation. Defaults to ``0.5``.

    Example:
        >>> model = DraemCutPasteModel(
        ...     sspcab=False,
        ...     image_size=(256, 256),
        ... )
        >>> input_tensor = torch.randn(4, 3, 256, 256)
        >>> reconstruction, prediction, classification = model(input_tensor)
    """

    def __init__(
        self,
        sspcab: bool = False,
        image_size: tuple[int, int] = (256, 256),
        severity_dropout: float = 0.3,
        severity_input_channels: str = "original+mask+recon",
        # CutPaste generator parameters
        cut_w_range: tuple[int, int] = (10, 80),
        cut_h_range: tuple[int, int] = (1, 2),
        a_fault_start: float = 1.0,
        a_fault_range_end: float = 10.0,
        augment_probability: float = 0.5,
    ) -> None:
        super().__init__()

        # Store severity input configuration
        self.severity_input_channels = severity_input_channels
        
        # Calculate number of input channels for severity head
        self.severity_in_channels = self._calculate_severity_in_channels(severity_input_channels)

        # Core DRAEM networks (reuse from original implementation)
        self.reconstructive_subnetwork = ReconstructiveSubNetwork(sspcab=sspcab)
        self.discriminative_subnetwork = DiscriminativeSubNetwork(in_channels=6, out_channels=2)

        # New: Severity head for classification (dynamic channel input)
        self.severity_head = SeverityHead(
            in_channels=self.severity_in_channels,
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
            validation_enabled=True
        )

    def _calculate_severity_in_channels(self, severity_input_channels: str) -> int:
        """Calculate number of input channels based on configuration.
        
        Args:
            severity_input_channels: Channel configuration string
            
        Returns:
            Number of input channels for severity head
        """
        channel_count = 0
        if "original" in severity_input_channels:
            channel_count += 1
        if "mask" in severity_input_channels:
            channel_count += 1
        if "recon" in severity_input_channels:
            channel_count += 1
        
        if channel_count == 0:
            raise ValueError(f"Invalid severity_input_channels: {severity_input_channels}")
            
        return channel_count

    def _create_severity_input(
        self, 
        original_single_ch: torch.Tensor,
        predicted_mask: torch.Tensor, 
        reconstruction_single_ch: torch.Tensor
    ) -> torch.Tensor:
        """Create severity head input based on configuration.
        
        Args:
            original_single_ch: Single channel original image
            predicted_mask: Predicted anomaly mask
            reconstruction_single_ch: Single channel reconstructed image
            
        Returns:
            Concatenated tensor for severity head input
        """
        inputs = []
        
        if "original" in self.severity_input_channels:
            inputs.append(original_single_ch)
        if "mask" in self.severity_input_channels:
            inputs.append(predicted_mask)
        if "recon" in self.severity_input_channels:
            inputs.append(reconstruction_single_ch)
            
        return torch.cat(inputs, dim=1)

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
        augmented_batch, anomaly_mask, anomaly_labels = self.synthetic_generator(batch)

        # Pass through reconstructive network (reconstruct augmented images)
        reconstruction = self.reconstructive_subnetwork(augmented_batch)

        # Concatenate reconstruction and augmented images for discriminative network
        # DRAEM expects 6 channels: [reconstructed(3) + augmented(3)]
        joined_input = torch.cat([reconstruction, augmented_batch], dim=1)

        # Pass through discriminative network to get anomaly prediction
        prediction = self.discriminative_subnetwork(joined_input)

        # Prepare input for severity head based on configuration
        # Use augmented image instead of original to provide meaningful signal
        augmented_single_ch = augmented_batch[:, :1, :, :]  # First channel of augmented image
        predicted_mask = prediction[:, 1:2, :, :]  # Anomaly channel from prediction
        reconstruction_single_ch = reconstruction[:, :1, :, :]  # First channel of reconstruction

        severity_input = self._create_severity_input(
            augmented_single_ch, predicted_mask, reconstruction_single_ch
        )

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

        # Prepare input for severity head based on configuration
        original_single_ch = batch[:, :1, :, :]
        predicted_mask = prediction[:, 1:2, :, :]
        reconstruction_single_ch = reconstruction[:, :1, :, :]  # First channel of reconstruction
        
        severity_input = self._create_severity_input(
            original_single_ch, predicted_mask, reconstruction_single_ch
        )

        # Get classification probabilities
        classification_logits = self.severity_head(severity_input)
        classification_probs = torch.softmax(classification_logits, dim=1)

        # Extract anomaly scores (use softmax probabilities for anomaly class)
        anomaly_scores = classification_probs[:, 1]  # Probability of anomaly class

        # Create anomaly maps (apply softmax like DRAEM)
        anomaly_maps = torch.softmax(prediction, dim=1)[:, 1, ...]  # Softmax + anomaly channel

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


