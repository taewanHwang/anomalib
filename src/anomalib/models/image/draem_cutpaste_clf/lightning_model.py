"""Lightning Model for DRAEM CutPaste Classification.

This module implements the PyTorch Lightning wrapper for DraemCutPasteModel,
providing training, validation, and testing logic with integrated loss functions.

Based on the original DRAEM lightning model with extensions for CutPaste and classification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.utilities import rank_zero_info
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

        # Loss function parameters
        clf_weight (float, optional): Weight for classification loss.
            Defaults to ``1.0``. (L2, SSIM, Focal weights use DRAEM defaults)
        focal_alpha (float, optional): Alpha parameter for focal loss.
            Higher values give more weight to the anomaly class. Defaults to ``0.9``.

        # Severity head configuration
        severity_input_channels (str, optional): Channels to use for severity head input.
            Options: "original", "mask", "original+mask". Defaults to ``"original+mask"``.
        detach_mask (bool, optional): Whether to detach mask gradient during training.
            If True, prevents CE loss gradients from flowing to discriminative network.
            Defaults to ``True``.

        # Standard anomalib parameters
        evaluator (Evaluator | bool, optional): Evaluator instance.
            Defaults to ``True``.

    Example:
        >>> model = DraemCutPasteClf(
        ...     image_size=(256, 256),
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

        # Loss function parameters
        clf_weight: float = 1.0,
        focal_alpha: float = 0.9,

        # Learning rate parameters
        recon_lr_multiplier: float = 1.0,  # Reconstruction network LR multiplier

        # Severity head input configuration
        severity_input_channels: str = "original+mask",
        detach_mask: bool = True,

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

        # Loss weights
        self.clf_weight = clf_weight
        self.focal_alpha = focal_alpha

        # Learning rate parameters
        self.recon_lr_multiplier = recon_lr_multiplier

        # Severity head configuration
        self.severity_input_channels = severity_input_channels
        self.detach_mask = detach_mask

        # Model and loss will be created in setup()
        self.model: DraemCutPasteModel
        self.loss: DraemCutPasteLoss
        self.validation_cutpaste_probability: float = augment_probability
        self.validation_analysis_dir: Path | None = None
        self._val_scores: list[float] = []
        self._val_labels: list[int] = []

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Get trainer arguments specific to this model.

        Returns:
            dict[str, Any]: Trainer arguments
        """
        return {}

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
            severity_input_channels=self.severity_input_channels,
            detach_mask=self.detach_mask,
            cut_w_range=self.cut_w_range,
            cut_h_range=self.cut_h_range,
            a_fault_start=self.a_fault_start,
            a_fault_range_end=self.a_fault_range_end,
            augment_probability=self.augment_probability,
        )

        self.loss = DraemCutPasteLoss(
            clf_weight=self.clf_weight,
            focal_alpha=self.focal_alpha,
        )

    def _apply_validation_cutpaste(self, batch: Batch) -> tuple[Batch, torch.Tensor]:
        """Optionally augment validation data with CutPaste anomalies."""
        images = batch.image
        device = images.device
        mask_shape = (images.size(0), 1, images.size(2), images.size(3))

        if self.validation_cutpaste_probability <= 0:
            return batch, torch.zeros(mask_shape, device=device, dtype=images.dtype)

        target_ratio = float(max(0.0, min(1.0, self.validation_cutpaste_probability)))
        generator = self.model.synthetic_generator
        original_prob = getattr(generator, "probability", 1.0)
        generator.probability = 1.0
        with torch.no_grad():
            augmented_all, fault_mask_all, _ = generator(images)
        generator.probability = original_prob

        batch_size = images.size(0)
        num_anomalies = int(round(batch_size * target_ratio))
        num_anomalies = min(num_anomalies, batch_size)

        indices = torch.randperm(batch_size, device=device)
        anomaly_idx = indices[:num_anomalies]
        augmented = images.clone()
        augmented[anomaly_idx] = augmented_all[anomaly_idx]

        final_mask = torch.zeros_like(fault_mask_all)
        final_mask[anomaly_idx] = fault_mask_all[anomaly_idx]

        anomaly_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        anomaly_labels[anomaly_idx] = 1

        rank_zero_info(
            f"✂️ Validation CutPaste 적용: {num_anomalies}/{batch_size} anomalies "
            f"(ratio={target_ratio:.2f})"
        )

        batch = batch.update(image=augmented, gt_label=anomaly_labels)
        return batch, final_mask.to(device, dtype=images.dtype)

    def configure_optimizers(self) -> dict[str, Any] | torch.optim.Optimizer:
        """Configure the optimizer and learning rate scheduler.

        Returns:
            dict[str, Any] | torch.optim.Optimizer: Optimizer and optional scheduler configuration
        """
        # 학습 설정 가져오기 (anomaly_trainer.py에서 설정된 값)
        training_config = getattr(self, '_training_config', {})

        # 학습률 및 옵티마이저 설정 (default 없이 명시적으로 값 가져오기)
        learning_rate = training_config['learning_rate']
        optimizer_type = training_config['optimizer'].lower()
        weight_decay = training_config['weight_decay']

        # Reconstruction network learning rate (increased for better reconstruction)
        recon_lr = learning_rate * self.recon_lr_multiplier

        # Parameter groups: reconstruction network vs others
        recon_params = list(self.model.reconstructive_subnetwork.parameters())
        other_params = [
            p for n, p in self.model.named_parameters()
            if not n.startswith('reconstructive_subnetwork')
        ]

        param_groups = [
            {'params': recon_params, 'lr': recon_lr, 'name': 'reconstruction'},
            {'params': other_params, 'lr': learning_rate, 'name': 'others'}
        ]

        # 옵티마이저 선택
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                params=param_groups,
                weight_decay=weight_decay,
            )
        elif optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                params=param_groups,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"지원하지 않는 옵티마이저: {optimizer_type}")

        # 스케줄러 설정 (optional)
        scheduler_config = training_config.get('scheduler', None)

        if scheduler_config is None:
            # 스케줄러 없이 옵티마이저만 반환
            return optimizer

        scheduler_type = scheduler_config.get('type', 'none').lower()

        if scheduler_type == 'none':
            return optimizer

        elif scheduler_type == 'steplr':
            # StepLR 스케줄러 설정
            step_size = scheduler_config.get('step_size', 5)
            gamma = scheduler_config.get('gamma', 0.5)

            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma
            )

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',  # epoch 단위로 step
                    'frequency': 1,
                }
            }

        elif scheduler_type == 'cosineannealinglr':
            # CosineAnnealingLR 스케줄러 설정
            max_epochs = training_config.get('max_epochs', 50)
            eta_min = scheduler_config.get('eta_min', 1e-6)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
                eta_min=eta_min
            )

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }

        elif scheduler_type == 'warmup_steplr':
            # Warmup + StepLR 스케줄러 설정
            warmup_epochs = scheduler_config.get('warmup_epochs', 2)
            step_size = scheduler_config.get('step_size', 5)
            gamma = scheduler_config.get('gamma', 0.5)

            # Warmup phase: 0 → learning_rate
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,  # Start at 10% of base LR
                end_factor=1.0,    # End at 100% of base LR
                total_iters=warmup_epochs
            )

            # Main phase: StepLR decay
            main_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma
            )

            # Combine warmup and main scheduler
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }

        else:
            print(f"Warning: Unknown scheduler type '{scheduler_type}', using optimizer only")
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
        # reconstruction: (batch_size, 1, H, W), range: unbounded → ~[0,1] after training, meaning: reconstructed image
        # prediction: (batch_size, 2, H, W), range: unbounded logits, meaning: pixel-wise [normal, anomaly] scores
        # classification: (batch_size, 2), range: unbounded logits, meaning: image-level [normal, anomaly] scores
        # anomaly_mask: (batch_size, 1, H, W), range: [0,1], meaning: ground truth anomaly regions (0=normal, 1=anomaly)
        # anomaly_labels: (batch_size,), range: [0, ∞) continuous, meaning: fault severity (0=normal, >0=anomaly severity)

        # Convert severity labels to binary classification labels (0=normal, 1=anomaly)
        anomaly_labels = (anomaly_labels > 0).long()  # Any non-zero severity is anomaly
        # anomaly_labels: (batch_size,), range: {0, 1}, meaning: binary labels (0=normal, 1=anomaly)

        # Extract single channel from original images to match reconstruction
        images_single_ch = images[:, :1, :, :]
        # images_single_ch: (batch_size, 1, H, W), range: [0, 1], meaning: first channel of input (grayscale)

        # Compute loss (with focal loss)
        total_loss, loss_dict = self.loss(
            reconstruction=reconstruction,
            original=images_single_ch,
            prediction=prediction,
            anomaly_mask=anomaly_mask,
            classification=classification,
            anomaly_labels=anomaly_labels,
            use_focal_loss=True,  # Training has pixel-level GT
        )

        # Log losses
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        for loss_name, loss_value in loss_dict.items():
            if loss_name != "total_loss":  # Avoid duplicate logging
                self.log(f"train_{loss_name}", loss_value, on_step=True, on_epoch=True)

        # Log learning rates for each parameter group
        optimizer = self.trainer.optimizers[0]
        for idx, param_group in enumerate(optimizer.param_groups):
            group_name = param_group.get('name', f'group_{idx}')
            self.log(f"lr/{group_name}", param_group['lr'], on_step=True, on_epoch=True)

        # Gradient Norm Monitoring (log every 100 steps for efficiency)
        if batch_idx % 100 == 0:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    self.log(f"grad_norm/{name}", grad_norm, on_step=True, on_epoch=False)

        return total_loss

    def validation_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        """Validation step.

        Args:
            batch (Batch): Input batch
            batch_idx (int): Batch index

        Returns:
            STEP_OUTPUT: Predictions for evaluation
        """
        batch, anomaly_mask = self._apply_validation_cutpaste(batch)
        predictions = self.model(batch.image, training_phase=False)

        # Calculate validation loss on potentially augmented data
        images = batch.image
        images_single_ch = images[:, :1, :, :]
        anomaly_labels = batch.gt_label.long()
        
        with torch.no_grad():
            reconstruction = self.model.reconstructive_subnetwork(images_single_ch)
            joined_input = torch.cat([images_single_ch, reconstruction], dim=1)
            prediction_logits = self.model.discriminative_subnetwork(joined_input)
            severity_input = self.model._create_severity_input(
                images_single_ch, prediction_logits, reconstruction
            )
            classification = self.model.severity_head(severity_input)
        
        if self.validation_cutpaste_probability <= 0:
            batch_size = images.shape[0]
            device = images.device
            anomaly_mask = torch.zeros(batch_size, 1, *images.shape[2:], device=device)

        val_total_loss, val_loss_dict = self.loss(
            reconstruction=reconstruction,
            original=images_single_ch,
            prediction=prediction_logits,
            anomaly_mask=anomaly_mask,
            classification=classification,
            anomaly_labels=anomaly_labels,
            use_focal_loss=False,  # Validation has no pixel-level GT mask
        )
        
        # Log validation losses
        self.log("val_loss", val_total_loss, on_step=False, on_epoch=True, prog_bar=True)
        for loss_name, loss_value in val_loss_dict.items():
            if loss_name != "total_loss":
                self.log(f"val_{loss_name}", loss_value, on_step=False, on_epoch=True)

        # Combine batch information with predictions for evaluator
        # This ensures AUROC metric gets both pred_score and gt_label
        pred_scores = predictions.pred_score.detach().cpu().tolist()
        labels = batch.gt_label.detach().cpu().tolist()
        self._val_scores.extend(pred_scores)
        self._val_labels.extend(labels)

        return batch.update(**predictions._asdict())

    def on_validation_epoch_start(self) -> None:
        self._val_scores = []
        self._val_labels = []

    def on_validation_epoch_end(self) -> None:
        if not self._val_scores:
            return

        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
        import json

        preds = [1 if s >= 0.5 else 0 for s in self._val_scores]
        precision = precision_score(self._val_labels, preds, zero_division=0)
        recall = recall_score(self._val_labels, preds, zero_division=0)
        f1 = f1_score(self._val_labels, preds, zero_division=0)
        cm_raw = confusion_matrix(self._val_labels, preds)
        cm = cm_raw.tolist()
        accuracy = ((cm_raw[0, 0] + cm_raw[1, 1]) / cm_raw.sum()) if cm_raw.sum() else 0.0
        try:
            auroc = roc_auc_score(self._val_labels, self._val_scores)
        except ValueError:
            auroc = float("nan")

        metrics = {
            "auroc": float(auroc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "accuracy": float(accuracy),
            "confusion_matrix": cm,
            "total_samples": len(self._val_labels),
            "positive_samples": int(sum(self._val_labels)),
            "negative_samples": int(len(self._val_labels) - sum(self._val_labels)),
        }

        if self.validation_analysis_dir:
            analysis_dir = Path(self.validation_analysis_dir)
            analysis_dir.mkdir(parents=True, exist_ok=True)
            val_path = analysis_dir / "val_metrics_report.json"
            with open(val_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            rank_zero_info(f"💾 Validation metrics 저장: {val_path}")

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

            # Loss parameters
            "clf_weight": self.clf_weight,
            "focal_alpha": self.focal_alpha,

            # Severity head configuration
            "severity_input_channels": self.severity_input_channels,
            "detach_mask": self.detach_mask,
        }
