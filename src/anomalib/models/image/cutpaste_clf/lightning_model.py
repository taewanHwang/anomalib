"""CutPaste Classifier - PyTorch Lightning Model.

This module implements a baseline comparison model using only CutPaste augmentation
and a simple CNN classifier (Table B1 structure) without reconstruction or localization.

Compared to DRAEM CutPaste:
- No reconstruction network
- No localization (anomaly map)
- Only binary classification with CutPaste augmentation

This serves as a baseline to demonstrate the contribution of adding
reconstruction and localization to the CutPaste approach.
"""

import torch
import torch.nn as nn
from torch import optim
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator, AUROC
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model import CutPasteClf

# Import CutPaste augmentation from draem_cutpaste_clf
from anomalib.models.image.draem_cutpaste_clf.synthetic_generator_v2 import CutPasteSyntheticGenerator


class CutPasteClassifier(AnomalibModule):
    """CutPaste + Simple CNN Classifier (Baseline Model).

    Pure classification approach using only CutPaste augmentation
    and a simple CNN classifier (Table B1 structure).

    Args:
        image_size: Input image size (H, W)
        cut_w_range: CutPaste width range (min, max)
        cut_h_range: CutPaste height range (min, max)
        a_fault_start: Anomaly severity start value (typically 1)
        a_fault_range_end: Anomaly severity end value (max severity + 1)
        augment_probability: Probability of applying augmentation during training
        evaluator: Metrics evaluator
        pre_processor: Pre-processor for input data
        post_processor: Post-processor for model outputs
        visualizer: Visualizer for results
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (128, 128),
        cut_w_range: tuple[int, int] = (2, 16),
        cut_h_range: tuple[int, int] = (2, 16),
        a_fault_start: int = 1,
        a_fault_range_end: int = 11,
        augment_probability: float = 0.5,
        evaluator: Evaluator | bool = True,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.image_size = image_size
        self.cut_w_range = cut_w_range
        self.cut_h_range = cut_h_range
        self.a_fault_start = a_fault_start
        self.a_fault_range_end = a_fault_range_end
        self.augment_probability = augment_probability

        # Simple CNN Model (Table B1)
        self.model = CutPasteClf(image_size=image_size)

        # CutPaste Synthetic Generator
        self.synthetic_generator = CutPasteSyntheticGenerator(
            cut_w_range=cut_w_range,
            cut_h_range=cut_h_range,
            a_fault_start=a_fault_start,
            a_fault_range_end=a_fault_range_end,
            probability=augment_probability,  # Note: uses 'probability' not 'augment_probability'
        )

        # Loss: CrossEntropyLoss for binary classification
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Training step with CutPaste augmentation.

        Args:
            batch: Input batch containing images
            args: Additional arguments
            kwargs: Additional keyword arguments

        Returns:
            Dictionary containing loss value
        """
        del args, kwargs  # Unused

        images = batch.image  # [B, 3, H, W]

        # Debug: Print input image size (only once)
        if not hasattr(self, '_input_size_printed'):
            print(f"\n🔍 DEBUG: Input images shape from batch: {images.shape}")
            self._input_size_printed = True

        # Apply CutPaste augmentation
        # CutPasteSyntheticGenerator returns 3 values:
        # - augmented_images: [B, 3, H, W]
        # - fault_mask: [B, 1, H, W] (not used for classification)
        # - severity_labels: [B] (0=Normal, 1-10=Anomaly severity)
        augmented_images, _, severity_labels = self.synthetic_generator(images)

        # Debug: Print augmented image size (only once)
        if not hasattr(self, '_aug_size_printed'):
            print(f"🔍 DEBUG: Augmented images shape: {augmented_images.shape}")
            self._aug_size_printed = True

        # Forward pass
        logits = self.model(augmented_images)  # [B, 2]

        # Convert severity labels to binary (0=Normal, 1=Anomaly)
        binary_labels = (severity_labels > 0).long()  # [B]

        # Compute loss
        loss = self.criterion(logits, binary_labels)

        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Validation step.

        Args:
            batch: Input batch
            args: Additional arguments
            kwargs: Additional keyword arguments

        Returns:
            Batch with predictions
        """
        del args, kwargs  # Unused

        # Forward pass
        logits = self.model(batch.image)  # [B, 2]

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)  # [B, 2]
        pred_scores = probs[:, 1]  # [B] - Anomaly probability

        # Update batch with predictions
        batch.pred_score = pred_scores
        batch.pred_label = (pred_scores > 0.5).long()

        return batch

    def configure_optimizers(self) -> dict | optim.Optimizer:
        """Configure optimizer and learning rate scheduler.

        Returns:
            dict | Optimizer: Optimizer and optional scheduler configuration
        """
        if hasattr(self, '_training_config'):
            lr = self._training_config.get('learning_rate', 0.0001)
            optimizer_name = self._training_config.get('optimizer', 'adam')
            weight_decay = self._training_config.get('weight_decay', 0.0)

            if optimizer_name == 'adamw':
                optimizer = optim.AdamW(
                    self.parameters(),
                    lr=lr,
                    weight_decay=weight_decay
                )
            else:
                optimizer = optim.Adam(
                    self.parameters(),
                    lr=lr,
                    weight_decay=weight_decay
                )

            # Scheduler (optional)
            scheduler_config = self._training_config.get('scheduler', None)
            if scheduler_config:
                scheduler_type = scheduler_config.get('type', 'steplr')
                if scheduler_type == 'steplr':
                    scheduler = optim.lr_scheduler.StepLR(
                        optimizer,
                        step_size=scheduler_config.get('step_size', 10),
                        gamma=scheduler_config.get('gamma', 0.8)
                    )
                    # Return dictionary format for Lightning
                    return {
                        'optimizer': optimizer,
                        'lr_scheduler': {
                            'scheduler': scheduler,
                            'interval': 'epoch',
                            'frequency': 1,
                        }
                    }

            return optimizer

        # Default optimizer
        return optim.Adam(self.parameters(), lr=0.0001)

    @property
    def trainer_arguments(self) -> dict:
        """Get trainer arguments specific to this model.

        Returns:
            dict: Trainer arguments (empty for this model)
        """
        return {}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type.

        Returns:
            Learning type (ONE_CLASS for anomaly detection)
        """
        return LearningType.ONE_CLASS
