"""DRAEM CutPaste Lightning model implementation.

DRAEM with CutPaste augmentation (without severity head).
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

__all__ = ["DraemCutPaste"]


class DraemCutPaste(AnomalibModule):
    """DRAEM CutPaste Lightning Module.

    DRAEM with CutPaste augmentation, using DRAEM's original anomaly scoring.

    Args:
        enable_sspcab (bool, optional): Enable SSPCAB training.
            Defaults to ``False``.

        # CutPaste generator parameters
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
        focal_alpha (float, optional): Alpha parameter for focal loss.
            Higher values give more weight to the anomaly class. Defaults to ``0.9``.

        evaluator (Evaluator | bool, optional): Evaluator instance or flag to use default.
            Defaults to ``True``.
    """

    def __init__(
        self,
        enable_sspcab: bool = False,
        # CutPaste generator parameters
        cut_w_range: tuple[int, int] = (10, 80),
        cut_h_range: tuple[int, int] = (1, 2),
        a_fault_start: float = 1.0,
        a_fault_range_end: float = 10.0,
        augment_probability: float = 0.5,
        # Loss parameters
        focal_alpha: float = 0.9,
        # Standard anomalib parameters
        evaluator: Evaluator | bool = True,
    ) -> None:
        if evaluator is True:
            # Create evaluator with explicit AUROC metric
            val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
            test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
            evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])

        super().__init__(
            pre_processor=False,
            post_processor=False,
            evaluator=evaluator,
            visualizer=True
        )

        # Store hyperparameters
        self.sspcab = enable_sspcab
        self.cut_w_range = cut_w_range
        self.cut_h_range = cut_h_range
        self.a_fault_start = a_fault_start
        self.a_fault_range_end = a_fault_range_end
        self.augment_probability = augment_probability
        self.focal_alpha = focal_alpha

        # Model and loss will be created in setup()
        self.model: DraemCutPasteModel
        self.loss: DraemCutPasteLoss

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Get trainer arguments specific to this model.

        Returns:
            dict[str, Any]: Trainer arguments
        """
        # Note: max_epochs will be overridden by Engine config
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type (ONE_CLASS)
        """
        return LearningType.ONE_CLASS

    def setup(self, stage: str | None = None) -> None:
        """Setup model and loss.

        Args:
            stage: Current stage (fit, validate, test, predict)
        """
        del stage  # Unused

        self.model = DraemCutPasteModel(
            sspcab=self.sspcab,
            cut_w_range=self.cut_w_range,
            cut_h_range=self.cut_h_range,
            a_fault_start=self.a_fault_start,
            a_fault_range_end=self.a_fault_range_end,
            augment_probability=self.augment_probability,
        )

        self.loss = DraemCutPasteLoss(focal_alpha=self.focal_alpha)

    def configure_optimizers(self) -> dict[str, Any] | torch.optim.Optimizer:
        """Configure the optimizer and learning rate scheduler.

        Returns:
            dict[str, Any] | torch.optim.Optimizer: Optimizer and optional scheduler configuration
        """
        # 학습 설정 가져오기 (anomaly_trainer.py에서 설정된 값)
        training_config = getattr(self, '_training_config', {})

        # 학습률 및 옵티마이저 설정 (default 없이 명시적으로 값 가져오기)
        learning_rate = training_config.get('learning_rate', 0.0001)
        optimizer_type = training_config.get('optimizer', 'adam').lower()
        weight_decay = training_config.get('weight_decay', 1e-5)

        # 옵티마이저 선택
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                params=self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                params=self.model.parameters(),
                lr=learning_rate,
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

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform training step.

        Args:
            batch: Input batch
            *args: Additional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            STEP_OUTPUT: Training step outputs
        """
        del args, kwargs  # Unused

        input_image = batch.image

        # Forward pass with CutPaste augmentation
        reconstruction, prediction, anomaly_mask, anomaly_labels = self.model(
            input_image, training_phase=True
        )

        # Extract single channel (model output is 1-channel)
        input_single_ch = input_image[:, :1, :, :]

        # Compute loss (with focal loss)
        loss = self.loss(input_single_ch, reconstruction, anomaly_mask, prediction, use_focal_loss=True)

        # Log individual loss components
        l2_loss = self.loss.l2_loss(reconstruction, input_single_ch)
        ssim_loss = self.loss.ssim_loss(reconstruction, input_single_ch) * 2
        focal_loss = self.loss.focal_loss(prediction, anomaly_mask.squeeze(1).long())

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_l2", l2_loss, on_step=False, on_epoch=True)
        self.log("train_loss_ssim", ssim_loss, on_step=False, on_epoch=True)
        self.log("train_loss_focal", focal_loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform validation step.

        Args:
            batch: Input batch
            *args: Additional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            STEP_OUTPUT: Validation step outputs
        """
        del args, kwargs  # Unused

        input_image = batch.image

        # Calculate validation loss on real (non-augmented) data
        with torch.no_grad():
            # Get reconstruction and prediction directly from subnetworks
            batch_single_ch = input_image[:, :1, :, :]
            reconstruction = self.model.reconstructive_subnetwork(batch_single_ch)
            concatenated_inputs = torch.cat([batch_single_ch, reconstruction], axis=1)
            prediction = self.model.discriminative_subnetwork(concatenated_inputs)

            # Create dummy anomaly mask (validation has no pixel-level GT)
            batch_size = input_image.shape[0]
            device = input_image.device
            anomaly_mask = torch.zeros(batch_size, 1, *input_image.shape[2:], device=device)

            # Compute validation loss (without focal loss)
            val_loss = self.loss(batch_single_ch, reconstruction, anomaly_mask, prediction, use_focal_loss=False)

            # Compute individual loss components
            val_loss_l2 = self.loss.l2_loss(reconstruction, batch_single_ch)
            val_loss_ssim = self.loss.ssim_loss(reconstruction, batch_single_ch) * 2

            # Log validation losses
            self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_loss_l2", val_loss_l2, on_step=False, on_epoch=True)
            self.log("val_loss_ssim", val_loss_ssim, on_step=False, on_epoch=True)

            # Get inference output for evaluator
            anomaly_map = torch.softmax(prediction, dim=1)[:, 1, ...]
            pred_score = torch.amax(anomaly_map, dim=(-2, -1))
            from anomalib.data import InferenceBatch
            inference_output = InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

        # Use InferenceBatch for evaluator
        return batch.update(**inference_output._asdict())

    def test_step(self, batch: Batch, batch_idx: int, *args, **kwargs) -> STEP_OUTPUT:
        """Perform test step.

        Args:
            batch: Input batch
            batch_idx: Batch index (unused)
            *args: Additional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            STEP_OUTPUT: Test step outputs
        """
        del args, kwargs, batch_idx  # Unused

        # Inference mode
        predictions = self.model(batch.image, training_phase=False)

        return batch.update(**predictions._asdict())
