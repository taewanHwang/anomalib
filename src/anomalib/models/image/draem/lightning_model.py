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
from anomalib.data import Batch
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
        # Compute loss
        loss = self.loss(input_image, reconstruction, anomaly_mask, prediction)

        if self.sspcab:
            loss += self.sspcab_lambda * self.sspcab_loss(
                self.sspcab_activations["input"],
                self.sspcab_activations["output"],
            )

        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform validation step for DRAEM.

        Uses softmax predictions of the anomalous class as anomaly maps.
        Performs inference in eval mode for consistent AUROC calculation.

        Args:
            batch (Batch): Input batch containing images and metadata.
            args: Additional positional arguments (unused).
            kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Dictionary containing predictions and metadata.
        """
        del args, kwargs  # These variables are not used.
        
        # Note: Lightning automatically sets model.eval() before validation_step
        with torch.no_grad():
            prediction = self.model(batch.image)
        
        # Return updated batch - Evaluator will handle metrics automatically
        return batch.update(**prediction._asdict())

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
        # config에서 학습 파라미터 가져오기 (기본값 사용)
        config = getattr(self, '_training_config', {})
        learning_rate = config.get('learning_rate', 0.0001)
        weight_decay = config.get('weight_decay', 0.0)
        scheduler_type = config.get('scheduler', 'multistep')
        optimizer_type = config.get('optimizer', 'adamw')  # 기본값을 AdamW로 변경
        
        # 옵티마이저 선택
        if optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                params=self.model.parameters(), 
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(
                params=self.model.parameters(), 
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            # 기본값: AdamW
            optimizer = torch.optim.AdamW(
                params=self.model.parameters(), 
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        # 스케줄러 선택
        if scheduler_type == 'cosine':
            # Cosine Annealing 스케줄러
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=config.get('max_epochs', 50),
                eta_min=learning_rate * 0.01  # 최소 학습률을 1%로 설정
            )
        elif scheduler_type == 'warmup_cosine':
            # Warmup + Cosine (간단 구현: 처음 몇 에포크는 linear warmup)
            warmup_epochs = config.get('warmup_epochs', 5)
            total_epochs = config.get('max_epochs', 50)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=total_epochs - warmup_epochs,
                eta_min=learning_rate * 0.01
            )
        else:
            # 기본 MultiStepLR 스케줄러
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, 
                milestones=[400, 600], 
                gamma=0.1
            )
        
        return [optimizer], [scheduler]

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
