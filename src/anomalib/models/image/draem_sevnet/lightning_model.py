"""Custom DRAEM Lightning Model.

Lightning wrapper for Custom DRAEM model with fault severity prediction for HDMAP datasets.

Author: Taewan Hwang
"""

from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT


from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import AUROC, Evaluator
from anomalib.models.components import AnomalibModule
from .loss import DraemSevNetLoss
from .synthetic_generator import HDMAPCutPasteSyntheticGenerator
from .torch_model import DraemSevNetModel, DraemSevNetOutput

__all__ = ["DraemSevNet"]


class DraemSevNet(AnomalibModule):
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
        severity_head_pooling_type (str, optional): Pooling type for SeverityHead.
            Options: "gap" (Global Average Pooling), "spatial_aware" (Spatial-Aware).
            Defaults to ``"gap"``.
        severity_head_spatial_size (int, optional): Spatial size to preserve in spatial_aware mode.
            Defaults to ``4``.
        severity_head_use_spatial_attention (bool, optional): Use spatial attention mechanism.
            Defaults to ``True``.
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
        >>> from anomalib.models.image import DraemSevNet
        >>> model = DraemSevNet(
        ...     severity_max=10.0,
        ...     severity_head_mode="multi_scale",
        ...     score_combination="weighted_average",
        ...     severity_head_pooling_type="spatial_aware",
        ...     severity_head_spatial_size=4,
        ...     severity_head_use_spatial_attention=True,
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
        # 🆕 Spatial-Aware SeverityHead 설정
        severity_head_pooling_type: str = "gap",
        severity_head_spatial_size: int = 4,
        severity_head_use_spatial_attention: bool = True,
        patch_ratio_range: tuple[float, float] = (2.0, 4.0),
        patch_width_range: tuple[int, int] = (20, 80),
        patch_count: int = 1,
        anomaly_probability: float = 0.5,
        severity_weight: float = 0.5,
        severity_loss_type: str = "mse",
        severity_max: float = 1.0,
        optimizer: str = "adam",
        learning_rate: float = 1e-4,
        evaluator: Evaluator | bool = True,
    ) -> None:
        # Create evaluator with explicit AUROC metric with test_ prefix (same as DRAEM and PatchCore)
        if evaluator is True:
            val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
            test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
            evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])
        
        super().__init__(evaluator=evaluator)

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
        self.model = DraemSevNetModel(
            sspcab=sspcab,
            severity_head_mode=severity_head_mode,
            severity_head_hidden_dim=severity_head_hidden_dim,
            score_combination=score_combination,
            severity_weight_for_combination=severity_weight_for_combination,
            # 🆕 Spatial-Aware SeverityHead 설정 전달
            severity_head_pooling_type=severity_head_pooling_type,
            severity_head_spatial_size=severity_head_spatial_size,
            severity_head_use_spatial_attention=severity_head_use_spatial_attention
        )
        
        # Initialize DRAEM-SevNet loss function
        self.loss = DraemSevNetLoss(
            severity_weight=severity_weight,
            severity_loss_type=severity_loss_type
        )
        

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
        """Perform the validation step of DRAEM-SevNet.
        
        Computes both inference outputs for metrics and validation loss for early stopping.
        """
        del args, kwargs  # These variables are not used.
        
        input_image = batch.image
        
        with torch.no_grad():
            # For validation loss calculation, generate synthetic anomalies (same as training)
            synthetic_image, fault_mask, severity_map, severity_label = self.augmenter(input_image)
            
            # Temporarily set model to training mode for loss calculation
            self.model.train()
            reconstruction, mask_logits, predicted_severity = self.model(synthetic_image)
            
            # Compute validation loss
            val_loss = self.loss(
                input_image=input_image,
                reconstruction=reconstruction,
                anomaly_mask=fault_mask,
                prediction=mask_logits,
                severity_gt=severity_label,
                severity_pred=predicted_severity
            )
            
            # Log validation loss for early stopping
            self.log("val_loss", val_loss.item(), on_epoch=True, prog_bar=True, logger=True)
            
            # Set back to eval mode for inference outputs
            self.model.eval()
            # For metrics calculation, use clean image prediction
            model_output = self.model(input_image)
        
        # Return updated batch - Evaluator will handle metrics automatically (same as test_step)
        # Use same logic as test_step for consistent metrics
        pred_score = getattr(model_output, 'final_score', getattr(model_output, 'pred_score', None))
        
        # Generate pred_label from pred_score (threshold will be applied by post_processor)
        pred_label = (pred_score > 0.5).int() if pred_score is not None else None
        
        return batch.update(
            pred_score=pred_score,
            anomaly_map=model_output.anomaly_map,
            pred_label=pred_label
        )


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
