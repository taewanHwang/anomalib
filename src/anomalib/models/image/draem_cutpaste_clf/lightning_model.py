"""Lightning Model for DRAEM CutPaste Classification.

This module implements the PyTorch Lightning wrapper for DraemCutPasteModel,
providing training, validation, and testing logic with integrated loss functions.

Based on the original DRAEM lightning model with extensions for CutPaste and classification.
"""

from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import AUROC, Evaluator
from anomalib.models.components import AnomalibModule

from .loss import DraemCutPasteLoss
from .torch_model import DraemCutPasteModel

__all__ = ["DraemCutPasteClf"]


class DraemCutPasteClf(AnomalibModule):
    """DRAEM CutPaste Classification Lightning Model.

    This model extends DRAEM with CutPaste augmentation and CNN-based classification
    for binary anomaly detection. Combines reconstruction, mask prediction, and
    classification losses for end-to-end training.

    Args:
        # Model architecture parameters
        sspcab (bool, optional): Enable SSPCAB training in reconstructive network.
            Defaults to ``False``.
        image_size (tuple[int, int], optional): Expected input size for severity head.
            Defaults to ``(256, 256)``.
        severity_dropout (float, optional): Dropout rate for severity head.
            Defaults to ``0.3``.

        # CutPaste augmentation parameters
        cut_w_range (tuple[int, int], optional): Range of patch widths.
            Defaults to ``(10, 80)``.
        cut_h_range (tuple[int, int], optional): Range of patch heights.
            Defaults to ``(1, 2)``.
        a_fault_start (float, optional): Minimum fault amplitude.
            Defaults to ``1.0``.
        a_fault_range_end (float, optional): Maximum fault amplitude.
            Defaults to ``10.0``.
        augment_probability (float, optional): Probability of applying augmentation.
            Defaults to ``0.5``.
        norm (bool, optional): Enable normalization in CutPaste augmentation.
            Defaults to ``True``.

        # Loss function parameters
        clf_weight (float, optional): Weight for classification loss.
            Defaults to ``1.0``. (L2, SSIM, Focal weights use DRAEM defaults)

        # Standard anomalib parameters
        evaluator (Evaluator | bool, optional): Evaluator instance.
            Defaults to ``True``.

    Example:
        >>> model = DraemCutPasteClf(
        ...     image_size=(256, 256),
        ...     norm=True,
        ...     clf_weight=1.0
        ... )
        >>> # Training with Lightning trainer
        >>> trainer = pl.Trainer()
        >>> trainer.fit(model, train_dataloader, val_dataloader)
    """

    def __init__(
        self,
        # Model architecture parameters
        sspcab: bool = False,
        image_size: tuple[int, int] = (256, 256),
        severity_dropout: float = 0.3,

        # CutPaste augmentation parameters
        cut_w_range: tuple[int, int] = (10, 80),
        cut_h_range: tuple[int, int] = (1, 2),
        a_fault_start: float = 1.0,
        a_fault_range_end: float = 10.0,
        augment_probability: float = 0.5,
        norm: bool = True,

        # Loss function parameters
        clf_weight: float = 1.0,

        # Standard anomalib parameters
        evaluator: Evaluator | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=False,
            post_processor=False,
            evaluator=evaluator,
            visualizer=True
        )

        # Store parameters for model creation
        self.sspcab = sspcab
        self.image_size = image_size
        self.severity_dropout = severity_dropout
        self.cut_w_range = cut_w_range
        self.cut_h_range = cut_h_range
        self.a_fault_start = a_fault_start
        self.a_fault_range_end = a_fault_range_end
        self.augment_probability = augment_probability
        self.norm = norm

        # Loss weights
        self.clf_weight = clf_weight

        # Model and loss will be created in setup()
        self.model: DraemCutPasteModel
        self.loss: DraemCutPasteLoss

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Get trainer arguments specific to this model.

        Returns:
            dict[str, Any]: Trainer arguments
        """
        return {"gradient_clip_val": 0.5, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Learning type of the model.

        Returns:
            LearningType: Learning type (ONE_CLASS)
        """
        return LearningType.ONE_CLASS

    def configure_model(self) -> None:
        """Configure the PyTorch model and loss function."""
        self.model = DraemCutPasteModel(
            sspcab=self.sspcab,
            image_size=self.image_size,
            severity_dropout=self.severity_dropout,
            cut_w_range=self.cut_w_range,
            cut_h_range=self.cut_h_range,
            a_fault_start=self.a_fault_start,
            a_fault_range_end=self.a_fault_range_end,
            augment_probability=self.augment_probability,
            norm=self.norm,
        )

        self.loss = DraemCutPasteLoss(
            clf_weight=self.clf_weight,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer.

        Returns:
            torch.optim.Optimizer: Optimizer for training
        """
        # 학습 설정 가져오기 (anomaly_trainer.py에서 설정된 값)
        training_config = getattr(self, '_training_config', {})

        # 학습률 및 옵티마이저 설정 (default 없이 명시적으로 값 가져오기)
        learning_rate = training_config['learning_rate']
        optimizer_type = training_config['optimizer'].lower()
        weight_decay = training_config['weight_decay']

        # 옵티마이저 선택
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                params=self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                params=self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"지원하지 않는 옵티마이저: {optimizer_type}")

        return optimizer

    def training_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        """Training step.

        Args:
            batch (Batch): Input batch containing images
            batch_idx (int): Batch index

        Returns:
            STEP_OUTPUT: Loss value for optimization
        """
        # Extract images from batch
        images = batch.image

        # Forward pass with CutPaste augmentation
        reconstruction, prediction, classification, anomaly_mask, anomaly_labels = self.model(images, training_phase=True)

        # Convert severity labels to binary classification labels (0=normal, 1=anomaly)
        anomaly_labels = (anomaly_labels > 0).long()  # Any non-zero severity is anomaly

        # Compute loss
        total_loss, loss_dict = self.loss(
            reconstruction=reconstruction,
            original=images,
            prediction=prediction,
            anomaly_mask=anomaly_mask,
            classification=classification,
            anomaly_labels=anomaly_labels,
        )

        # Log losses
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        for loss_name, loss_value in loss_dict.items():
            if loss_name != "total_loss":  # Avoid duplicate logging
                self.log(f"train_{loss_name}", loss_value, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        """Validation step.

        Args:
            batch (Batch): Input batch
            batch_idx (int): Batch index

        Returns:
            STEP_OUTPUT: Predictions for evaluation
        """
        # During validation, we don't apply augmentation
        predictions = self.model(batch.image, training_phase=False)

        # Combine batch information with predictions for evaluator
        # This ensures AUROC metric gets both pred_score and gt_label
        return batch.update(**predictions._asdict())

    def test_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        """Test step.

        Args:
            batch (Batch): Input batch
            batch_idx (int): Batch index

        Returns:
            STEP_OUTPUT: Predictions for evaluation
        """
        # Test step uses the same logic as validation step
        predictions = self.model(batch.image, training_phase=False)
        return batch.update(**predictions._asdict())

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT:
        """Prediction step.

        Args:
            batch (Batch): Input batch
            batch_idx (int): Batch index
            dataloader_idx (int, optional): Dataloader index. Defaults to ``0``.

        Returns:
            STEP_OUTPUT: Model predictions
        """
        return self.model(batch.image, training_phase=False)

    def get_model_config(self) -> dict[str, Any]:
        """Get complete model configuration.

        Returns:
            dict[str, Any]: Model configuration
        """
        return {
            "model_name": "DraemCutPasteClf",
            "learning_type": str(self.learning_type),

            # Architecture parameters
            "sspcab": self.sspcab,
            "image_size": self.image_size,
            "severity_dropout": self.severity_dropout,

            # CutPaste parameters
            "cut_w_range": self.cut_w_range,
            "cut_h_range": self.cut_h_range,
            "a_fault_start": self.a_fault_start,
            "a_fault_range_end": self.a_fault_range_end,
            "augment_probability": self.augment_probability,
            "norm": self.norm,

            # Loss parameters
            "clf_weight": self.clf_weight,
        }


