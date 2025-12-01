"""Lightning model for UniNet."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .components import UniNetLoss
from .torch_model import UniNetModel


class UniNet(AnomalibModule):
    """UniNet model for anomaly detection.

    Args:
        student_backbone (str): The backbone model to use for the student network. Defaults to "wide_resnet50_2".
        teacher_backbone (str): The backbone model to use for the teacher network. Defaults to "wide_resnet50_2".
        temperature (float): Temperature parameter used for contrastive loss. Controls the temperature of the student
            and teacher similarity computation. Defaults to 0.1.
        pre_processor (PreProcessor | bool, optional): Preprocessor instance or bool flag. Defaults to True.
        post_processor (PostProcessor | bool, optional): Postprocessor instance or bool flag. Defaults to True.
        evaluator (Evaluator | bool, optional): Evaluator instance or bool flag. Defaults to True.
        visualizer (Visualizer | bool, optional): Visualizer instance or bool flag. Defaults to True.
    """

    def __init__(
        self,
        student_backbone: str = "wide_resnet50_2",
        teacher_backbone: str = "wide_resnet50_2",
        temperature: float = 2.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.loss = UniNetLoss(temperature=temperature)  # base class expects self.loss in lightning module
        self.model = UniNetModel(student_backbone=student_backbone, teacher_backbone=teacher_backbone, loss=self.loss)

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a training step of UniNet."""
        del args, kwargs  # These variables are not used.

        loss = self.model(images=batch.image, masks=batch.gt_mask, labels=batch.gt_label)

        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a validation step of UniNet."""
        del args, kwargs  # These variables are not used.
        if batch.image is None or not isinstance(batch.image, torch.Tensor):
            msg = "Expected batch.image to be a tensor, but got None or non-tensor type"
            raise ValueError(msg)

        # UniNet model returns InferenceBatch during validation (self.training=False)
        # So we temporarily set training mode to compute loss
        self.model.train()
        loss = self.model(images=batch.image, masks=batch.gt_mask, labels=batch.gt_label)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Switch back to eval mode for predictions
        self.model.eval()
        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict]]:
        """Configure optimizers with warm-up + cosine annealing scheduler.

        Returns:
            tuple[list[Optimizer], list[dict]]: Optimizers and scheduler configs.
        """
        optimizer = torch.optim.AdamW(
            [
                {"params": self.model.student.parameters()},
                {"params": self.model.bottleneck.parameters()},
                {"params": self.model.dfs.parameters()},
                {"params": self.model.teachers.target_teacher.parameters(), "lr": self.learning_rate * 0.0002},
            ],
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
            eps=1e-10,
            amsgrad=True,
        )

        # Warm-up scheduler: start from 0.1x lr and linearly increase to full lr
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_epochs,
        )

        # Cosine annealing after warm-up
        max_epochs = self.trainer.max_epochs if self.trainer.max_epochs else 30
        cosine_epochs = max_epochs - self.warmup_epochs
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=self.learning_rate * 0.01,  # minimum lr = 1% of peak lr
        )

        # Combine warm-up + cosine annealing
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs],
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Does not require any trainer arguments."""
        return {}

    @property
    def learning_type(self) -> LearningType:
        """The model uses one-class learning."""
        return LearningType.ONE_CLASS
