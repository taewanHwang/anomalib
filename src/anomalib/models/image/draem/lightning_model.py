# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DRÆM.

A discriminatively trained reconstruction embedding for surface anomaly
detection.

Paper https://arxiv.org/abs/2108.07610

This module implements the DRÆM model for surface anomaly detection. DRÆM uses a
discriminatively trained reconstruction embedding approach to detect anomalies by
comparing input images with their reconstructions.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torchvision.transforms.v2 import Compose, Resize

from anomalib import LearningType
from anomalib.data import Batch, InferenceBatch
from anomalib.data.utils import DownloadInfo, download_and_extract
from anomalib.data.utils.generators.perlin import PerlinAnomalyGenerator
from anomalib.metrics import AUROC, Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .loss import DraemLoss
from .torch_model import DraemModel

__all__ = ["Draem"]

DTD_DOWNLOAD_INFO = DownloadInfo(
    name="dtd-r1.0.1.tar.gz",
    url="https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
    hashsum="e42855a52a4950a3b59612834602aa253914755c95b0cff9ead6d07395f8e205",
)


class Draem(AnomalibModule):
    """DRÆM.

    A discriminatively trained reconstruction embedding for
    surface anomaly detection.

    The model consists of two main components:
    1. A reconstruction network that learns to reconstruct normal images
    2. A discriminative network that learns to identify anomalous regions

    Args:
        dtd_dir (Path | str): Directory path for the DTD dataset for anomaly deneration.
            Defaults to ``./datasets/dtd``.
        enable_sspcab (bool, optional): Enable SSPCAB training.
            Defaults to ``False``.
        sspcab_lambda (float, optional): Weight factor for SSPCAB loss.
            Defaults to ``0.1``.
        anomaly_source_path (str | None, optional): Path to directory containing
            anomaly source images. If ``None``, random noise is used.
            Defaults to ``None``.
        beta (float | tuple[float, float], optional): Blend factor for anomaly
            generation. If tuple, represents range for random sampling.
            Defaults to ``(0.1, 1.0)``.
        pre_processor (PreProcessor | bool, optional): Pre-processor instance or
            flag to use default.
            Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post-processor instance
            or flag to use default.
            Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Evaluator instance or flag to
            use default.
            Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualizer instance or flag to
            use default.
            Defaults to ``True``.
    """

    def __init__(
        self,
        dtd_dir: Path | str = "./datasets/dtd",
        enable_sspcab: bool = False,
        sspcab_lambda: float = 0.1,
        beta: float | tuple[float, float] = (0.1, 1.0),
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        if evaluator is True:
            # Create evaluator with explicit AUROC metric
            val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
            test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
            evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])
        
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
        dtd_dir = Path(dtd_dir)
        if not dtd_dir.is_dir():
            download_and_extract(dtd_dir, DTD_DOWNLOAD_INFO)
        self.augmenter = PerlinAnomalyGenerator(anomaly_source_path=dtd_dir, blend_factor=beta)
        self.model = DraemModel(sspcab=enable_sspcab)
        self.loss = DraemLoss()
        self.sspcab = enable_sspcab

        if self.sspcab:
            self.sspcab_activations: dict = {}
            self.setup_sspcab()
            self.sspcab_loss = nn.MSELoss()
            self.sspcab_lambda = sspcab_lambda

    def setup_sspcab(self) -> None:
        """Set up SSPCAB forward hooks.

        Prepares the model for SSPCAB training by adding forward hooks to capture
        layer activations from specific points in the network.
        """

        def get_activation(name: str) -> Callable:
            """Create a hook function to retrieve layer activations.

            Args:
                name (str): Identifier for storing the activation in the
                    activation dictionary.

            Returns:
                Callable: Hook function that stores layer activations.
            """

            def hook(_, __, output: torch.Tensor) -> None:  # noqa: ANN001
                """Store layer activations during forward pass.

                Args:
                    _: Unused module argument.
                    __: Unused input argument.
                    output (torch.Tensor): Output tensor from the layer.
                """
                self.sspcab_activations[name] = output

            return hook

        self.model.reconstructive_subnetwork.encoder.mp4.register_forward_hook(get_activation("input"))
        self.model.reconstructive_subnetwork.encoder.block5.register_forward_hook(get_activation("output"))

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform training step for DRAEM.

        The step consists of:
        1. Generating simulated anomalies
        2. Computing reconstructions and predictions
        3. Calculating the loss

        Args:
            batch (Batch): Input batch containing images and metadata.
            args: Additional positional arguments (unused).
            kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Dictionary containing the training loss.
        """
        del args, kwargs  # These variables are not used.

        input_image = batch.image
        # Apply corruption to input image
        augmented_image, anomaly_mask = self.augmenter(input_image)
        # Generate model prediction
        reconstruction, prediction = self.model(augmented_image)
        # Compute loss (with focal loss - training has pixel-level GT)
        loss = self.loss(input_image, reconstruction, anomaly_mask, prediction, use_focal_loss=True)

        # Compute individual loss components for detailed logging
        loss_l2 = self.loss.l2_loss(reconstruction, input_image)
        loss_ssim = self.loss.ssim_loss(reconstruction, input_image) * 2  # DRAEM multiplies by 2
        loss_focal = self.loss.focal_loss(prediction, anomaly_mask.squeeze(1).long())
        
        # Clamp focal loss to prevent negative values (numerical instability)
        loss_focal = torch.clamp(loss_focal, min=0.0)

        if self.sspcab:
            sspcab_loss_val = self.sspcab_lambda * self.sspcab_loss(
                self.sspcab_activations["input"],
                self.sspcab_activations["output"],
            )
            loss += sspcab_loss_val
            self.log("train_sspcab_loss", sspcab_loss_val, on_step=False, on_epoch=True)

        # Log total loss
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        
        # Log individual loss components
        self.log("train_loss_l2", loss_l2, on_step=False, on_epoch=True)
        self.log("train_loss_ssim", loss_ssim, on_step=False, on_epoch=True)
        self.log("train_loss_focal", loss_focal, on_step=False, on_epoch=True)
        
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform validation step for DRAEM.

        Args:
            batch (Batch): Input batch containing images and metadata.
            args: Additional positional arguments (unused).
            kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Dictionary containing predictions and metadata.
        """
        del args, kwargs  # These variables are not used.

        input_image = batch.image

        # Calculate validation loss on real (non-augmented) data
        with torch.no_grad():
            # Get reconstruction and prediction directly from subnetworks
            reconstruction = self.model.reconstructive_subnetwork(input_image)
            concatenated_inputs = torch.cat([input_image, reconstruction], axis=1)
            prediction = self.model.discriminative_subnetwork(concatenated_inputs)

            # Create dummy anomaly mask (validation has no pixel-level GT)
            batch_size = input_image.shape[0]
            device = input_image.device
            anomaly_mask = torch.zeros(batch_size, 1, *input_image.shape[2:], device=device) # dummy anomaly mask, it will be ignored for loss calculation

            # Compute validation loss (without focal loss - validation has no pixel-level GT)
            val_loss = self.loss(input_image, reconstruction, anomaly_mask, prediction, use_focal_loss=False)

            # Compute individual loss components for detailed logging
            val_loss_l2 = self.loss.l2_loss(reconstruction, input_image)
            val_loss_ssim = self.loss.ssim_loss(reconstruction, input_image) * 2

            # Log validation losses
            self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_loss_l2", val_loss_l2, on_step=False, on_epoch=True)
            self.log("val_loss_ssim", val_loss_ssim, on_step=False, on_epoch=True)

            # Get inference output for evaluator (compute anomaly_map and pred_score)
            anomaly_map = torch.softmax(prediction, dim=1)[:, 1, ...]
            pred_score = torch.amax(anomaly_map, dim=(-2, -1))
            inference_output = InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

        # Use InferenceBatch directly - let Lightning's Evaluator handle AUROC calculation
        return batch.update(**inference_output._asdict())

    def test_step(self, batch: Batch, batch_idx: int, *args, **kwargs) -> STEP_OUTPUT:
        """Perform test step for DRAEM.
        
        Ensures identical inference behavior as validation_step for consistency.
        
        Args:
            batch (Batch): Input batch containing images and metadata.
            batch_idx (int): Index of the batch.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).
            
        Returns:
            STEP_OUTPUT: Updated batch with model predictions.
        """
        del args, kwargs, batch_idx  # These variables are not used.
        
        # 🔧 Identical inference logic as validation_step
        # Note: Lightning automatically sets model.eval() before test_step
        with torch.no_grad():
            prediction = self.model(batch.image)
        
        # Return updated batch - Evaluator will handle metrics automatically
        return batch.update(**prediction._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Get DRÆM-specific trainer arguments.

        Returns:
            dict[str, Any]: Dictionary containing trainer arguments:
                - gradient_clip_val: ``0``
                - num_sanity_val_steps: ``0``
        """
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer and learning rate scheduler.

        Returns:
            tuple[list[Adam], list[MultiStepLR]]: Tuple containing optimizer and
                scheduler lists.
        """
        # 학습 설정 가져오기 (anomaly_trainer.py에서 설정된 값)
        training_config = getattr(self, '_training_config', {})

        # 학습률 및 옵티마이저 설정
        learning_rate = training_config.get('learning_rate', 0.0001)
        optimizer_type = training_config.get('optimizer', 'adamw').lower()
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
            return [optimizer], []

        scheduler_type = scheduler_config.get('type', 'none').lower()

        if scheduler_type == 'none':
            return [optimizer], []

        elif scheduler_type == 'steplr':
            # StepLR 스케줄러 설정
            step_size = scheduler_config.get('step_size', 5)
            gamma = scheduler_config.get('gamma', 0.5)

            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma
            )
            return [optimizer], [scheduler]

        elif scheduler_type == 'cosineannealinglr':
            # CosineAnnealingLR 스케줄러 설정
            max_epochs = training_config.get('max_epochs', 50)
            eta_min = scheduler_config.get('eta_min', 1e-6)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
                eta_min=eta_min
            )
            return [optimizer], [scheduler]

        elif scheduler_type == 'multisteplr':
            # MultiStepLR 스케줄러 설정
            milestones = scheduler_config.get('milestones', [400, 600])
            gamma = scheduler_config.get('gamma', 0.1)

            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=gamma
            )
            return [optimizer], [scheduler]

        else:
            print(f"Warning: Unknown scheduler type '{scheduler_type}', using optimizer only")
            return [optimizer], []

    @property
    def learning_type(self) -> LearningType:
        """Get the learning type of the model.

        Returns:
            LearningType: The learning type (``LearningType.ONE_CLASS``).
        """
        return LearningType.ONE_CLASS

    @classmethod
    def configure_pre_processor(cls, image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Configure default pre-processor for DRÆM.

        Note:
            Imagenet normalization is not used in this model.

        Args:
            image_size (tuple[int, int] | None, optional): Target image size.
                Defaults to ``(256, 256)``.

        Returns:
            PreProcessor: Configured pre-processor with resize transform.
        """
        image_size = image_size or (256, 256)
        transform = Compose([Resize(image_size, antialias=True)])
        return PreProcessor(transform=transform)
