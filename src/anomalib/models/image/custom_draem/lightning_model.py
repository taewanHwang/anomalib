"""Custom DRAEM Lightning Model.

Lightning wrapper for Custom DRAEM model with fault severity prediction for HDMAP datasets.

Author: Taewan Hwang
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
from .adaptive_loss import AdaptiveCustomDraemLoss
from .synthetic_generator import HDMAPCutPasteSyntheticGenerator
from .torch_model import CustomDraemModel

__all__ = ["CustomDraem"]


class CustomDraem(AnomalibModule):
    """Custom DRAEM Lightning Model.
    
    Extended DRAEM model with fault severity prediction for HDMAP datasets.
    
    Args:
        sspcab (bool, optional): Enable SSPCAB training. Defaults to ``False``.
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
        use_adaptive_loss (bool, optional): Use adaptive multi-task loss with
            uncertainty weighting and progressive training. Defaults to ``True``.
        warmup_epochs (int, optional): Number of warmup epochs focusing on
            reconstruction before ramping up other tasks. Defaults to ``5``.
        optimizer (str, optional): Optimizer type ("adam", "adamw", "sgd").
            Defaults to ``"adam"``.
        learning_rate (float, optional): Learning rate for optimizer.
            Defaults to ``1e-4``.
            
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
        sspcab: bool = False,
        severity_max: float = 10.0,
        severity_input_mode: str = "discriminative_only",
        patch_ratio_range: tuple[float, float] = (2.0, 4.0),
        patch_width_range: tuple[int, int] = (20, 80),
        patch_count: int = 1,
        anomaly_probability: float = 0.5,
        reconstruction_weight: float = 1.0,
        segmentation_weight: float = 1.0,
        severity_weight: float = 0.5,
        use_adaptive_loss: bool = True,
        warmup_epochs: int = 5,
        optimizer: str = "adam",
        learning_rate: float = 1e-4,
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
            sspcab=sspcab,
            severity_max=severity_max,
            severity_input_mode=severity_input_mode
        )
        
        # Initialize loss function
        if use_adaptive_loss:
            self.loss = AdaptiveCustomDraemLoss(
                warmup_epochs=warmup_epochs,
                initial_weights={
                    "reconstruction": reconstruction_weight,
                    "segmentation": segmentation_weight,
                    "severity": severity_weight
                },
                use_uncertainty_weighting=True,
                use_dwa=False  # Start with uncertainty weighting
            )
        else:
            self.loss = CustomDraemLoss(
                reconstruction_weight=reconstruction_weight,
                segmentation_weight=segmentation_weight,
                severity_weight=severity_weight
            )
        self.use_adaptive_loss = use_adaptive_loss
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return Custom DRAEM trainer arguments."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer for training."""
        if self.optimizer_name.lower() == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name.lower() == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        elif self.optimizer_name.lower() == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}. "
                           f"Supported: 'adam', 'adamw', 'sgd'.")

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of Custom DRAEM.
        
        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform the training step of Custom DRAEM."""
        del args, kwargs  # These variables are not used.
        
        input_image = batch.image
        
        # No channel conversion needed - DRAEM backbone handles 3-channel input directly
        
        # Generate synthetic faults
        synthetic_image, fault_mask, severity_map, severity_label = self.augmenter(input_image)
        
        # Forward pass through model
        model_output = self.model(synthetic_image)
        reconstruction, prediction, severity_pred = model_output
        severity_pred = severity_pred.squeeze(-1)
        
        # Compute multi-task loss
        if self.use_adaptive_loss:
            loss = self.loss(
                input_image=input_image,
                reconstruction=reconstruction,
                anomaly_mask=fault_mask,
                prediction=prediction,
                severity_gt=severity_label,
                severity_pred=severity_pred,
                epoch=self.current_epoch
            )
        else:
            loss = self.loss(
                input_image=input_image,
                reconstruction=reconstruction,
                anomaly_mask=fault_mask,
                prediction=prediction,
                severity_gt=severity_label,
                severity_pred=severity_pred
            )
        
        # Log metrics
        # Both adaptive and standard loss return tensors
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        
        if self.use_adaptive_loss:
            individual_losses = self.loss.get_individual_losses(
                input_image, reconstruction, fault_mask, prediction, severity_label, severity_pred, self.current_epoch
            )
        else:
            individual_losses = self.loss.get_individual_losses(
                input_image, reconstruction, fault_mask, prediction, severity_label, severity_pred
            )
        self.log("train_l2_loss", individual_losses["l2_loss"].item(), on_epoch=True, logger=True)
        self.log("train_ssim_loss", individual_losses["ssim_loss"].item(), on_epoch=True, logger=True) 
        self.log("train_focal_loss", individual_losses["focal_loss"].item(), on_epoch=True, logger=True)
        self.log("train_severity_loss", individual_losses["severity_loss"].item(), on_epoch=True, logger=True)
        
        # Log adaptive weights if using adaptive loss
        if self.use_adaptive_loss:
            for key, value in individual_losses.items():
                if key.startswith("weight_") or key.startswith("uncertainty_"):
                    self.log(f"train_{key}", value.item() if hasattr(value, 'item') else value, on_epoch=True, logger=True)
        
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step of Custom DRAEM."""
        del args, kwargs  # These variables are not used.
        
        input_image = batch.image
        
        # No channel conversion needed - DRAEM backbone handles 3-channel input directly
        
        # Forward pass through model
        model_output = self.model(input_image)
        
        # Extract predictions
        pred_score = model_output.pred_score
        anomaly_map = model_output.anomaly_map
        severity_pred = model_output.pred_label
        
        # Log validation metrics
        if hasattr(batch, 'mask') and batch.mask is not None:
            mask_gt = batch.mask
            mask_coverage = (mask_gt > 0).float().mean()
            pred_coverage = (anomaly_map > 0.5).float().mean()
            
            self.log("val_mask_coverage", mask_coverage.item(), on_epoch=True, logger=True)
            self.log("val_pred_coverage", pred_coverage.item(), on_epoch=True, logger=True)
        
        self.log("val_severity_mean", severity_pred.mean().item(), on_epoch=True, logger=True)
        self.log("val_severity_std", severity_pred.std().item(), on_epoch=True, logger=True)
        self.log("val_anomaly_score_mean", pred_score.mean().item(), on_epoch=True, logger=True)
        
        # Return predictions in anomalib format
        class ValidationOutput:
            def __init__(self, original_batch, predictions):
                for attr_name in dir(original_batch):
                    if not attr_name.startswith('_'):
                        setattr(self, attr_name, getattr(original_batch, attr_name))
                for key, value in predictions.items():
                    setattr(self, key, value)
        
        prediction = {
            "pred_score": pred_score,
            "anomaly_map": anomaly_map,
            "pred_label": severity_pred,
        }
        
        return ValidationOutput(batch, prediction)

    def test_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform the test step of Custom DRAEM."""
        del args, kwargs  # These variables are not used.
        
        input_image = batch.image
        
        # No channel conversion needed - DRAEM backbone handles 3-channel input directly
        
        # Forward pass through model
        model_output = self.model(input_image)
        
        # Update batch with model predictions
        return batch.update(
            pred_score=model_output.pred_score,
            anomaly_map=model_output.anomaly_map,
            pred_label=model_output.pred_label
        )
