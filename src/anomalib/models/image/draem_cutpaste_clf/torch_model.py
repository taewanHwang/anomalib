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
    3. 1-channel optimization for grayscale data

    The model automatically extracts the first channel from input data, making it optimized
    for grayscale datasets while maintaining compatibility with multi-channel inputs.

    Args:
        sspcab (bool, optional): Enable SSPCAB training in reconstructive network.
            Defaults to ``False``.
        image_size (tuple[int, int], optional): Expected input size for severity head.
            Used for calculating FC layer dimensions. Defaults to ``(256, 256)``.
        severity_dropout (float, optional): Dropout rate for severity head. Defaults to ``0.3``.
        severity_input_channels (str, optional): Channels to use for severity head input.
            Options: "original", "mask", "original+mask".
            Defaults to ``"original+mask"``.
        detach_mask (bool, optional): Whether to detach mask gradient during training.
            If True, prevents CE loss gradients from flowing to discriminative network.
            Defaults to ``True``.

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
        >>> # Input can be 1-channel or 3-channel (first channel will be used)
        >>> input_tensor = torch.randn(4, 1, 256, 256)  # 1-channel grayscale
        >>> # or input_tensor = torch.randn(4, 3, 256, 256)  # 3-channel (uses first)
        >>> reconstruction, prediction, classification = model(input_tensor)
    """

    def __init__(
        self,
        sspcab: bool = False,
        image_size: tuple[int, int] = (256, 256),
        severity_dropout: float = 0.3,
        severity_input_channels: str = "original+mask",
        detach_mask: bool = True,
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
        self.detach_mask = detach_mask
        
        # Calculate number of input channels for severity head
        self.severity_in_channels = self._calculate_severity_in_channels(severity_input_channels)

        # Core DRAEM networks (1-channel configuration)
        self.reconstructive_subnetwork = ReconstructiveSubNetwork(
            in_channels=1, out_channels=1, sspcab=sspcab
        )
        self.discriminative_subnetwork = DiscriminativeSubNetwork(
            in_channels=2, out_channels=2, base_width=128
        )

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

        # Initialize weights for better training stability
        # self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization for ReLU networks.

        - Conv/ConvTranspose layers: Kaiming initialization (ReLU activation)
        - Linear layers: Kaiming initialization (ReLU activation, fan_in mode)
        - BatchNorm: weights=1, bias=0

        Note: ConvTranspose2d is critical for decoder stability in ReconstructiveSubNetwork.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                # Kaiming initialization for conv/deconv layers (ReLU activation)
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Kaiming initialization for linear layers (ReLU activation)
                # Use fan_in mode for forward pass stability
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                # BatchNorm initialization
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _calculate_severity_in_channels(self, severity_input_channels: str) -> int:
        """Calculate number of input channels based on configuration.

        Supported channels: original, mask

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

        if channel_count == 0:
            raise ValueError(f"Invalid severity_input_channels: {severity_input_channels}")

        return channel_count


    def _create_severity_input(
        self,
        original_single_ch: torch.Tensor,
        prediction: torch.Tensor,
        reconstruction: torch.Tensor
    ) -> torch.Tensor:
        """Create severity head input with channel-wise normalization and optional detach.

        Detach prevents CE loss gradients from flowing to reconstructive/discriminative subnets,
        allowing each subnet to focus on its dedicated loss (MSE/SSIM/Focal).

        Supported input channels:
        - "original": Original single channel image [0, 1] bounded
        - "mask": Anomaly probability mask from discriminative network [0, 1] bounded

        Args:
            original_single_ch: Single channel original image (already 0~1 normalized)
            prediction: Full prediction tensor from discriminative network (logit values)
            reconstruction: Single channel reconstructed image (raw output, unbounded)

        Returns:
            Concatenated tensor for severity head input
        """
        inputs = []

        if "original" in self.severity_input_channels:
            # Original is already normalized to [0, 1]
            inputs.append(original_single_ch)

        if "mask" in self.severity_input_channels:
            # Convert logit prediction to probability using softmax (0~1 range)
            mask_normalized = torch.softmax(prediction, dim=1)[:, 1:2, :, :]
            # Detach during training to prevent CE gradients flowing to discriminative subnet
            if self.training and self.detach_mask:
                mask_normalized = mask_normalized.detach()
            inputs.append(mask_normalized)

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

        # Extract single channel for 1-channel model
        augmented_single_ch = augmented_batch[:, :1, :, :]

        # Pass through reconstructive network (reconstruct augmented images)
        reconstruction = self.reconstructive_subnetwork(augmented_single_ch)
        # reconstruction shape: (batch_size, 1, height, width), range: unbounded (raw network output)
        # expected range after training: ~[0, 1] (should converge to input range)

        # Concatenate augmented and reconstruction for discriminative network
        # Follow DRAEM convention: [original, reconstruction]
        joined_input = torch.cat([augmented_single_ch, reconstruction], dim=1)

        # Pass through discriminative network to get anomaly prediction
        prediction = self.discriminative_subnetwork(joined_input)
        # prediction shape: (batch_size, 2, height, width)
        # meaning: pixel-wise logits for [normal, anomaly] classes at each spatial location
        # range: unbounded logits (before softmax)
        # expected after training: softmax(prediction)[:, 1] will be ~0 for normal pixels, ~1 for anomaly pixels

        # Prepare input for severity head based on configuration
        # Use augmented image instead of original to provide meaningful signal
        severity_input = self._create_severity_input(
            augmented_single_ch, prediction, reconstruction
        )
        # severity_input shape: (batch_size, severity_in_channels, height, width)
        # range: [0, 1] for normalized channels, unbounded for residual

        # Pass through severity head for classification
        classification = self.severity_head(severity_input)
        # classification shape: (batch_size, 2)
        # meaning: image-level logits for [normal, anomaly] classes
        # range: unbounded logits (before softmax)
        # expected after training: softmax(classification)[:, 1] will be ~0 for normal images, ~1 for anomaly images

        return reconstruction, prediction, classification, anomaly_mask, anomaly_labels

    def _forward_inference(self, batch: torch.Tensor) -> InferenceBatch:
        """Forward pass during inference without augmentation.

        Args:
            batch (torch.Tensor): Input images

        Returns:
            InferenceBatch: Batch containing anomaly scores and predictions
        """
        # Extract single channel for 1-channel model
        batch_single_ch = batch[:, :1, :, :]
        # batch_single_ch shape: (batch_size, 1, height, width), range: [0, 1] normalized

        # Pass through reconstructive network (no augmentation during inference)
        reconstruction = self.reconstructive_subnetwork(batch_single_ch)
        # reconstruction shape: (batch_size, 1, height, width), range: unbounded (raw network output)
        # expected range after training: ~[0, 1] (should converge to input range)

        # Concatenate original and reconstruction for discriminative network
        # Follow DRAEM convention: [original, reconstruction]
        joined_input = torch.cat([batch_single_ch, reconstruction], dim=1)

        # Pass through discriminative network
        prediction = self.discriminative_subnetwork(joined_input)
        # prediction shape: (batch_size, 2, height, width)
        # meaning: pixel-wise logits for [normal, anomaly] classes at each spatial location
        # range: unbounded logits (before softmax)
        # expected after training: softmax(prediction)[:, 1] will be ~0 for normal pixels, ~1 for anomaly pixels

        # Prepare input for severity head based on configuration
        severity_input = self._create_severity_input(
            batch_single_ch, prediction, reconstruction
        )
        # severity_input shape: (batch_size, severity_in_channels, height, width)
        # range: [0, 1] for normalized channels, unbounded for residual

        # Get classification probabilities
        classification_logits = self.severity_head(severity_input)
        # classification_logits shape: (batch_size, 2)
        # meaning: image-level logits for [normal, anomaly] classes
        # range: unbounded logits (before softmax)
        classification_probs = torch.softmax(classification_logits, dim=1)
        # classification_probs shape: (batch_size, 2), range: [0, 1] probabilities

        # Extract anomaly scores (use softmax probabilities for anomaly class)
        anomaly_scores = classification_probs[:, 1]  # Probability of anomaly class
        # anomaly_scores shape: (batch_size,), range: [0, 1]

        # Create anomaly maps (apply softmax like DRAEM)
        anomaly_maps = torch.softmax(prediction, dim=1)[:, 1, ...]  # Softmax + anomaly channel
        # anomaly_maps shape: (batch_size, height, width), range: [0, 1]

        # Create predictions (binary threshold at 0.5)
        predictions = (anomaly_scores > 0.5).float()
        # predictions shape: (batch_size,), range: {0.0, 1.0}

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
            "model_type": "1-channel_optimized",
            "reconstructive_network": "ReconstructiveSubNetwork (1ch)",
            "discriminative_network": "DiscriminativeSubNetwork (2ch)",
            "severity_head": self.severity_head.get_config(),
            "synthetic_generator": self.synthetic_generator.get_config_info(),
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


