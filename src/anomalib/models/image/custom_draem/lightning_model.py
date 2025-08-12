# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom DRAEM Lightning Model.

This module implements the Lightning wrapper for Custom DRAEM model designed 
specifically for HDMAP datasets with fault severity prediction capabilities.

The Custom DRAEM model extends the original DRAEM with:
1. Support for 1-channel grayscale images (HDMAP format)
2. Rectangular patch-based synthetic fault generation
3. Fault Severity Prediction Sub-Network
4. Multi-task learning (reconstruction + segmentation + severity)
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .loss import CustomDraemLoss
from .synthetic_generator import HDMAPCutPasteSyntheticGenerator
from .torch_model import CustomDraemModel

__all__ = ["CustomDraem"]


class CustomDraem(AnomalibModule):
    """Custom DRAEM Lightning Model.
    
    Extended DRAEM model with fault severity prediction for HDMAP datasets.
    
    Args:
        severity_max (float, optional): Maximum severity value for prediction.
            Defaults to ``10.0``.
        severity_input_mode (str, optional): Input mode for severity network.
            Options: "discriminative_only", "with_original", "with_reconstruction",
            "with_error_map", "multi_modal". Defaults to ``"discriminative_only"``.
        patch_ratio_range (tuple, optional): Range of patch aspect ratios.
            Values >1.0 for portrait, <1.0 for landscape, 1.0 for square.
            Defaults to ``(2.0, 4.0)``.
        patch_width_range (tuple, optional): Range of patch widths in pixels
            (based on 256x256 image). Defaults to ``(20, 80)``.
        patch_count (int, optional): Number of patches to generate.
            Defaults to ``1``.
        anomaly_probability (float, optional): Probability of applying synthetic fault 
            generation. Value between 0.0 and 1.0. Defaults to ``0.5``.
        reconstruction_weight (float, optional): Weight for reconstruction loss.
            Defaults to ``1.0``.
        segmentation_weight (float, optional): Weight for segmentation loss.
            Defaults to ``1.0``.
        severity_weight (float, optional): Weight for severity loss.
            Defaults to ``0.5``.
            
    Example:
        >>> from anomalib.models.image import CustomDraem
        >>> model = CustomDraem(
        ...     severity_max=10.0,
        ...     severity_input_mode="with_original",
        ...     patch_ratio_range=(2.0, 4.0),
        ...     patch_count=2
        ... )
    """

    def __init__(
        self,
        severity_max: float = 10.0,
        severity_input_mode: str = "discriminative_only",
        patch_ratio_range: tuple[float, float] = (2.0, 4.0),
        patch_width_range: tuple[int, int] = (20, 80),
        patch_count: int = 1,
        anomaly_probability: float = 0.5,
        reconstruction_weight: float = 1.0,
        segmentation_weight: float = 1.0,
        severity_weight: float = 0.5,
    ) -> None:
        super().__init__()

        # Store configuration
        self.severity_max = severity_max
        self.severity_input_mode = severity_input_mode
        self.patch_ratio_range = patch_ratio_range
        self.patch_width_range = patch_width_range
        self.patch_count = patch_count
        self.anomaly_probability = anomaly_probability
        
        # Initialize synthetic fault generator
        self.augmenter = HDMAPCutPasteSyntheticGenerator(
            patch_width_range=patch_width_range,
            patch_ratio_range=patch_ratio_range,
            severity_max=severity_max,
            patch_count=patch_count,
            probability=anomaly_probability,
        )
        
        # Initialize model components
        self.model = CustomDraemModel(
            severity_max=severity_max,
            severity_input_mode=severity_input_mode
        )
        
        # Initialize loss function
        self.loss = CustomDraemLoss(
            reconstruction_weight=reconstruction_weight,
            segmentation_weight=segmentation_weight,
            severity_weight=severity_weight
        )

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return Custom DRAEM trainer arguments."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of Custom DRAEM.
        
        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform the training step of Custom DRAEM.
        
        For each batch, the training step:
        1. Generates synthetic faults using HDMAPCutPasteSyntheticGenerator
        2. Passes augmented images through the model
        3. Computes multi-task loss (reconstruction + segmentation + severity)
        
        Args:
            batch (Batch): Input batch containing images.
            
        Returns:
            STEP_OUTPUT: Loss value for backpropagation.
        """
        del args, kwargs  # These variables are not used.
        
        input_image = batch.image
        
        # Generate synthetic faults using HDMAPCutPasteSyntheticGenerator
        synthetic_image, fault_mask, severity_map, severity_label = self.augmenter(input_image)
        
        # Pass synthetic image through the model (training mode)
        # Note: Model automatically detects training mode via self.training
        model_output = self.model(synthetic_image)
        reconstruction, prediction, severity_pred = model_output
        
        # Ensure severity prediction shape matches target shape
        severity_pred = severity_pred.squeeze(-1)  # (B, 1) -> (B,)
        
        # Compute multi-task loss
        loss = self.loss(
            input_image=input_image,           # Original image
            reconstruction=reconstruction,      # Reconstructed image
            anomaly_mask=fault_mask,           # Ground truth fault mask
            prediction=prediction,             # Anomaly prediction logits
            severity_gt=severity_label,        # Ground truth severity
            severity_pred=severity_pred        # Predicted severity
        )
        
        # Log training loss
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        
        # Log individual loss components for monitoring
        individual_losses = self.loss.get_individual_losses(
            input_image, reconstruction, fault_mask, prediction, severity_label, severity_pred
        )
        self.log("train_l2_loss", individual_losses["l2_loss"].item(), on_epoch=True, logger=True)
        self.log("train_ssim_loss", individual_losses["ssim_loss"].item(), on_epoch=True, logger=True) 
        self.log("train_focal_loss", individual_losses["focal_loss"].item(), on_epoch=True, logger=True)
        self.log("train_severity_loss", individual_losses["severity_loss"].item(), on_epoch=True, logger=True)
        
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step of Custom DRAEM.
        
        During validation, the model processes real images (no synthetic generation)
        and outputs anomaly predictions and severity estimates. This simulates
        real-world inference where no ground truth is available.
        
        Args:
            batch (Batch): Input batch containing images.
            
        Returns:
            STEP_OUTPUT: Dictionary containing predictions and ground truth.
        """
        del args, kwargs  # These variables are not used.
        
        # For validation, use real images without synthetic fault generation
        input_image = batch.image
        
        # Pass real image through the model (inference mode)
        # Note: Model automatically detects eval mode via self.training
        model_output = self.model(input_image)
        
        # Extract predictions for evaluation
        pred_score = model_output.pred_score      # Image-level anomaly score
        anomaly_map = model_output.anomaly_map    # Pixel-level anomaly map  
        severity_pred = model_output.pred_label   # Predicted severity
        
        # Log validation metrics if available
        if hasattr(batch, 'mask') and batch.mask is not None:
            # If ground truth masks are available, log some metrics
            mask_gt = batch.mask
            mask_coverage = (mask_gt > 0).float().mean()
            pred_coverage = (anomaly_map > 0.5).float().mean()
            
            self.log("val_mask_coverage", mask_coverage.item(), on_epoch=True, logger=True)
            self.log("val_pred_coverage", pred_coverage.item(), on_epoch=True, logger=True)
        
        # Log predicted severity statistics
        self.log("val_severity_mean", severity_pred.mean().item(), on_epoch=True, logger=True)
        self.log("val_severity_std", severity_pred.std().item(), on_epoch=True, logger=True)
        self.log("val_anomaly_score_mean", pred_score.mean().item(), on_epoch=True, logger=True)
        
        # Return predictions in the format expected by anomalib
        # Create a new object with original batch data + predictions
        class ValidationOutput:
            def __init__(self, original_batch, predictions):
                # Copy original batch attributes
                for attr_name in dir(original_batch):
                    if not attr_name.startswith('_'):
                        setattr(self, attr_name, getattr(original_batch, attr_name))
                # Add predictions
                for key, value in predictions.items():
                    setattr(self, key, value)
        
        prediction = {
            "pred_score": pred_score,
            "anomaly_map": anomaly_map,
            "pred_label": severity_pred,  # Custom: severity prediction
        }
        
        return ValidationOutput(batch, prediction)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers for Custom DRAEM.
        
        Returns:
            torch.optim.Optimizer: Adam optimizer with specified learning rate.
        """
        return torch.optim.Adam(
            params=self.parameters(),
            lr=0.0001,  # Default learning rate
            weight_decay=1e-5
        )
