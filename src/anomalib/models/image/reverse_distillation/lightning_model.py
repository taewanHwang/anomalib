# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Anomaly Detection via Reverse Distillation from One-Class Embedding.

This module implements the Reverse Distillation model for anomaly detection as described in
`Deng et al. (2022) <https://arxiv.org/abs/2201.10703v2>`_.

The model consists of:
- A pre-trained encoder (e.g. ResNet) that extracts multi-scale features
- A bottleneck layer that compresses features into a compact representation
- A decoder that reconstructs features back to the original feature space
- A scoring mechanism based on reconstruction error

Example:
    >>> from anomalib.models import ReverseDistillation
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.engine import Engine

    >>> # Initialize model and data
    >>> datamodule = MVTecAD()
    >>> model = ReverseDistillation(
    ...     backbone="wide_resnet50_2",
    ...     layers=["layer1", "layer2", "layer3"]
    ... )

    >>> # Train using the Engine
    >>> engine = Engine()
    >>> engine.fit(model=model, datamodule=datamodule)

    >>> # Get predictions
    >>> predictions = engine.predict(model=model, datamodule=datamodule)

See Also:
    - :class:`ReverseDistillation`: Lightning implementation of the model
    - :class:`ReverseDistillationModel`: PyTorch implementation of the model
    - :class:`ReverseDistillationLoss`: Loss function for training
"""

from collections.abc import Sequence
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .anomaly_map import AnomalyMapGenerationMode
from .loss import ReverseDistillationLoss
from .torch_model import ReverseDistillationModel


class ReverseDistillation(AnomalibModule):
    """PL Lightning Module for Reverse Distillation Algorithm.

    Args:
        backbone (str): Backbone of CNN network
            Defaults to ``wide_resnet50_2``.
        layers (list[str]): Layers to extract features from the backbone CNN
            Defaults to ``["layer1", "layer2", "layer3"]``.
        anomaly_map_mode (AnomalyMapGenerationMode, optional): Mode to generate anomaly map.
            Defaults to ``AnomalyMapGenerationMode.ADD``.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
            Defaults to ``True``.
        pre_processor (PreProcessor, optional): Pre-processor for the model.
            This is used to pre-process the input data before it is passed to the model.
            Defaults to ``None``.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer1", "layer2", "layer3"),
        anomaly_map_mode: AnomalyMapGenerationMode = AnomalyMapGenerationMode.ADD,
        pre_trained: bool = True,
        learning_rate: float = 0.005,
        weight_decay: float = 0.0,
        lr_scheduler: str | None = None,
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
        if self.input_size is None:
            msg = "Input size is required for Reverse Distillation model."
            raise ValueError(msg)

        self.backbone = backbone
        self.pre_trained = pre_trained
        self.layers = layers
        self.anomaly_map_mode = anomaly_map_mode
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler

        self.model = ReverseDistillationModel(
            backbone=self.backbone,
            pre_trained=self.pre_trained,
            layers=self.layers,
            input_size=self.input_size,
            anomaly_map_mode=self.anomaly_map_mode,
        )
        self.loss = ReverseDistillationLoss()

    def configure_optimizers(self) -> dict | optim.Adam:
        """Configure optimizers for decoder and bottleneck.

        Returns:
            Optimizer: Adam optimizer for each decoder, optionally with scheduler
        """
        optimizer = optim.Adam(
            params=list(self.model.decoder.parameters()) + list(self.model.bottleneck.parameters()),
            lr=self.learning_rate,
            betas=(0.5, 0.99),
            weight_decay=self.weight_decay,
        )

        if self.lr_scheduler is None:
            return optimizer

        # Configure scheduler based on type
        if self.lr_scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if self.trainer.max_epochs != -1 else 100,
                eta_min=self.learning_rate * 0.01
            )
        elif self.lr_scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.trainer.max_epochs // 3 if self.trainer.max_epochs != -1 else 30,
                gamma=0.1
            )
        elif self.lr_scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",  # Monitor val_loss for plateau
                }
            }
        else:
            return optimizer

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def configure_evaluator() -> Evaluator:
        """Configure the evaluator for Reverse Distillation model.

        Only includes essential AUROC metrics to avoid missing field errors.
        Reverse Distillation outputs pred_score and anomaly_map but not pred_label/pred_mask.

        Returns:
            Evaluator: Configured evaluator with essential validation and test metrics
        """
        from anomalib.metrics import AUROC
        
        # Validation metrics (for early stopping) - only essential AUROC
        val_image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
        val_metrics = [val_image_auroc]
        
        # Test metrics - only essential AUROC
        image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
        pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False)
        test_metrics = [image_auroc, pixel_auroc]
        
        return Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a training step of Reverse Distillation Model.

        Features are extracted from three layers of the Encoder model. These are passed to the bottleneck layer
        that are passed to the decoder network. The loss is then calculated based on the cosine similarity between the
        encoder and decoder features.

        Args:
          batch (batch: Batch): Input batch
          args: Additional arguments.
          kwargs: Additional keyword arguments.

        Returns:
          Feature Map
        """
        del args, kwargs  # These variables are not used.

        loss = self.loss(*self.model(batch.image))
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)

        # Log current learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", current_lr, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a validation step of Reverse Distillation Model.

        Performs inference in eval mode for consistent AUROC calculation.
        The model automatically switches to inference mode during validation,
        computing anomaly maps and scores for metric evaluation.

        Args:
          batch (Batch): Input batch
          args: Additional arguments.
          kwargs: Additional keyword arguments.

        Returns:
          Dictionary containing images, anomaly maps, true labels and masks.
          These are required for metric calculation by the Evaluator.
        """
        del args, kwargs  # These variables are not used.

        # Compute validation loss for monitoring and scheduler
        # Temporarily set model to training mode to get features
        self.model.train()
        encoder_features, decoder_features = self.model(batch.image)
        val_loss = self.loss(encoder_features, decoder_features)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Switch back to eval mode for predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(batch.image)

        # Return updated batch - Evaluator will handle metrics automatically
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return Reverse Distillation trainer arguments."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS
