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

from .loss import DraemSevNetLoss, DraemSevNetLossFactory
from .synthetic_generator import HDMAPCutPasteSyntheticGenerator
from .torch_model import CustomDraemModel, DraemSevNetOutput

__all__ = ["CustomDraem"]


class CustomDraem(AnomalibModule):
    """DRAEM-SevNet Lightning Model.
    
    DRAEM with Severity Network - unified severity-aware architecture for anomaly detection.
    Combines mask prediction and severity prediction in a multi-task learning framework.
    
    Args:
        sspcab (bool, optional): Enable SSPCAB training. Defaults to ``False``.
        severity_head_mode (str, optional): SeverityHead mode.
            Options: "single_scale" (act6 only), "multi_scale" (act2~act6).
            Defaults to ``"single_scale"``.
        severity_head_hidden_dim (int, optional): Hidden dimension for SeverityHead.
            Defaults to ``128``.
        score_combination (str, optional): Method to combine mask and severity scores.
            Options: "simple_average", "weighted_average", "maximum".
            Defaults to ``"simple_average"``.
        severity_weight_for_combination (float, optional): Weight for severity score
            in weighted_average combination. Defaults to ``0.5``.
        patch_ratio_range (tuple, optional): Range of patch aspect ratios.
            Values >1.0 for portrait, <1.0 for landscape, 1.0 for square.
            Defaults to ``(2.0, 4.0)``.
        patch_width_range (tuple, optional): Range of patch widths in pixels
            (scales with input image size). Defaults to ``(20, 80)``.
        patch_count (int, optional): Number of patches to generate.
            Defaults to ``1``.
        anomaly_probability (float, optional): Probability of applying synthetic fault 
            generation. Value between 0.0 and 1.0. Defaults to ``0.5``.
        severity_weight (float, optional): Weight λ for severity loss in
            L = L_draem + λ * L_severity. Defaults to ``0.5``.
        severity_loss_type (str, optional): Type of severity loss.
            Options: "mse", "smooth_l1". Defaults to ``"mse"``.
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
        severity_head_mode: str = "single_scale",
        severity_head_hidden_dim: int = 128,
        score_combination: str = "simple_average",
        severity_weight_for_combination: float = 0.5,
        patch_ratio_range: tuple[float, float] = (2.0, 4.0),
        patch_width_range: tuple[int, int] = (20, 80),
        patch_count: int = 1,
        anomaly_probability: float = 0.5,
        severity_weight: float = 0.5,
        severity_loss_type: str = "mse",
        severity_max: float = 1.0,
        optimizer: str = "adam",
        learning_rate: float = 1e-4,
    ) -> None:
        super().__init__()

        # Store DRAEM-SevNet configuration
        self.severity_head_mode = severity_head_mode
        self.score_combination = score_combination
        self.patch_ratio_range = patch_ratio_range
        self.patch_width_range = patch_width_range
        self.patch_count = patch_count
        self.anomaly_probability = anomaly_probability
        self.severity_max = severity_max
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        
        # Initialize synthetic fault generator with configurable severity_max
        self.augmenter = HDMAPCutPasteSyntheticGenerator(
            patch_width_range=patch_width_range,
            patch_ratio_range=patch_ratio_range,
            severity_max=self.severity_max,
            patch_count=patch_count,
            probability=anomaly_probability,
        )
        
        # Initialize DRAEM-SevNet model
        self.model = CustomDraemModel(
            sspcab=sspcab,
            severity_head_mode=severity_head_mode,
            severity_head_hidden_dim=severity_head_hidden_dim,
            score_combination=score_combination,
            severity_weight_for_combination=severity_weight_for_combination
        )
        
        # Initialize DRAEM-SevNet loss function
        self.loss = DraemSevNetLoss(
            severity_weight=severity_weight,
            severity_loss_type=severity_loss_type
        )
        
        # Initialize collections for validation metrics
        self._val_predictions = []
        self._val_labels = []
        self._val_mask_scores = []
        self._val_severity_scores = []
        self._val_final_scores = []

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
        """Perform the training step of DRAEM-SevNet."""
        del args, kwargs  # These variables are not used.
        
        input_image = batch.image
        
        # Generate synthetic faults
        synthetic_image, fault_mask, severity_map, severity_label = self.augmenter(input_image)
        
        # Forward pass through DRAEM-SevNet model
        reconstruction, mask_logits, severity_score = self.model(synthetic_image)
        
        # Compute DRAEM-SevNet loss: L = L_draem + λ * L_severity
        loss = self.loss(
            input_image=input_image,
            reconstruction=reconstruction,
            anomaly_mask=fault_mask,
            prediction=mask_logits,
            severity_gt=severity_label,
            severity_pred=severity_score
        )
        
        # Log training metrics
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        
        # Log individual loss components for analysis
        individual_losses = self.loss.get_individual_losses(
            input_image, reconstruction, fault_mask, mask_logits, severity_label, severity_score
        )
        self.log("train_l2_loss", individual_losses["l2_loss"].item(), on_epoch=True, logger=True)
        self.log("train_ssim_loss", individual_losses["ssim_loss"].item(), on_epoch=True, logger=True) 
        self.log("train_focal_loss", individual_losses["focal_loss"].item(), on_epoch=True, logger=True)
        self.log("train_draem_loss", individual_losses["draem_loss"].item(), on_epoch=True, logger=True)
        self.log("train_severity_loss", individual_losses["severity_loss"].item(), on_epoch=True, logger=True)
        
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step of DRAEM-SevNet."""
        del args, kwargs  # These variables are not used.
        
        input_image = batch.image
        labels = batch.gt_label
        
        # Forward pass through DRAEM-SevNet model (inference mode)
        model_output = self.model(input_image)
        
        # Extract all DRAEM-SevNet outputs
        if isinstance(model_output, DraemSevNetOutput):
            mask_score = model_output.mask_score
            severity_score = model_output.severity_score  
            final_score = model_output.final_score
            anomaly_map = model_output.anomaly_map
        else:
            # Fallback for compatibility (shouldn't happen in inference mode)
            reconstruction, mask_logits, severity_score = model_output
            mask_score = torch.amax(torch.softmax(mask_logits, dim=1)[:, 1, ...], dim=(-2, -1))
            final_score = (mask_score + severity_score) / 2.0
            anomaly_map = torch.softmax(mask_logits, dim=1)[:, 1, ...]
        
        # Store predictions and labels for multi-AUROC calculation
        self._val_mask_scores.extend(mask_score.cpu().numpy())
        self._val_severity_scores.extend(severity_score.cpu().numpy())
        self._val_final_scores.extend(final_score.cpu().numpy())
        self._val_labels.extend(labels.cpu().numpy())
        
        # Log basic validation metrics
        self.log("val_mask_score_mean", mask_score.mean().item(), on_epoch=True, logger=True)
        self.log("val_severity_score_mean", severity_score.mean().item(), on_epoch=True, logger=True)
        self.log("val_final_score_mean", final_score.mean().item(), on_epoch=True, logger=True)
        
        return {"final_score": final_score}

    def on_validation_epoch_end(self) -> None:
        """DRAEM-SevNet validation epoch 종료 시 multi-AUROC 계산 및 로깅.
        
        Mask AUROC, Severity AUROC, Combined AUROC를 각각 계산하여
        early stopping 및 성능 분석에 활용합니다.
        """
        if (len(self._val_mask_scores) > 0 and len(self._val_severity_scores) > 0 and 
            len(self._val_final_scores) > 0 and len(self._val_labels) > 0):
            
            try:
                from sklearn.metrics import roc_auc_score
                import numpy as np
                
                # Convert to numpy arrays
                mask_scores = np.array(self._val_mask_scores)
                severity_scores = np.array(self._val_severity_scores)
                final_scores = np.array(self._val_final_scores)
                labels = np.array(self._val_labels)
                
                # Check if we have both classes for AUROC calculation
                unique_labels = np.unique(labels)
                
                if len(unique_labels) >= 2:
                    # Calculate individual AUROCs
                    mask_auroc = roc_auc_score(labels, mask_scores)
                    severity_auroc = roc_auc_score(labels, severity_scores)
                    combined_auroc = roc_auc_score(labels, final_scores)
                    
                    # Log all AUROC metrics
                    self.log("val_mask_AUROC", mask_auroc, on_epoch=True, prog_bar=True, logger=True)
                    self.log("val_severity_AUROC", severity_auroc, on_epoch=True, prog_bar=True, logger=True)
                    self.log("val_combined_AUROC", combined_auroc, on_epoch=True, prog_bar=True, logger=True)
                    
                    # Keep val_image_AUROC for backward compatibility (use combined score)
                    self.log("val_image_AUROC", combined_auroc, on_epoch=True, prog_bar=True, logger=True)
                    
                else:
                    # Single class case - set all to random performance
                    default_auroc = 0.5
                    self.log("val_mask_AUROC", default_auroc, on_epoch=True, prog_bar=True, logger=True)
                    self.log("val_severity_AUROC", default_auroc, on_epoch=True, prog_bar=True, logger=True)
                    self.log("val_combined_AUROC", default_auroc, on_epoch=True, prog_bar=True, logger=True)
                    self.log("val_image_AUROC", default_auroc, on_epoch=True, prog_bar=True, logger=True)
                    
            except Exception as e:
                # AUROC calculation failed - use default values
                default_auroc = 0.5
                self.log("val_mask_AUROC", default_auroc, on_epoch=True, prog_bar=True, logger=True)
                self.log("val_severity_AUROC", default_auroc, on_epoch=True, prog_bar=True, logger=True)
                self.log("val_combined_AUROC", default_auroc, on_epoch=True, prog_bar=True, logger=True)
                self.log("val_image_AUROC", default_auroc, on_epoch=True, prog_bar=True, logger=True)
        
        # Reset collections for next epoch
        self._val_mask_scores = []
        self._val_severity_scores = []
        self._val_final_scores = []
        self._val_labels = []

    def test_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform the test step of Custom DRAEM."""
        del args, kwargs  # These variables are not used.
        
        input_image = batch.image
        
        # No channel conversion needed - DRAEM backbone handles 3-channel input directly
        
        # Forward pass through model
        model_output = self.model(input_image)
        
        # Update batch with model predictions
        # DraemSevNetOutput uses final_score instead of pred_score
        pred_score = getattr(model_output, 'final_score', getattr(model_output, 'pred_score', None))
        
        # Generate pred_label from pred_score (threshold will be applied by post_processor)
        pred_label = (pred_score > 0.5).int() if pred_score is not None else None
        
        return batch.update(
            pred_score=pred_score,
            anomaly_map=model_output.anomaly_map,
            pred_label=pred_label
        )
