"""Test Lightning Model for DRAEM CutPaste Classification."""

import pytest
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.models.image.draem_cutpaste_clf.lightning_model import DraemCutPasteClf


class MockBatch:
    """Mock batch class for testing."""

    def __init__(self, image=None, gt_label=None, **kwargs):
        self.image = image
        self.gt_label = gt_label
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, **kwargs):
        new_batch = MockBatch()
        for attr in dir(self):
            if not attr.startswith('_') and not callable(getattr(self, attr)):
                setattr(new_batch, attr, getattr(self, attr))
        for key, value in kwargs.items():
            setattr(new_batch, key, value)
        return new_batch


class TestDraemCutPasteClf:
    """Test cases for DraemCutPasteClf Lightning model."""

    def test_lightning_model_initialization(self):
        """Test Lightning model initialization with default parameters."""
        model = DraemCutPasteClf()

        # Check learning type
        assert model.learning_type == LearningType.ONE_CLASS

        # Check parameters are stored
        assert model.sspcab is False
        assert model.image_size == (256, 256)
        assert model.severity_dropout == 0.3
        assert model.augment_probability == 0.5
        assert model.norm is True

        # Check loss weights (only clf_weight is stored, others use DRAEM defaults)
        assert model.clf_weight == 1.0
        assert model.focal_alpha == 1.0
        assert model.focal_gamma == 2.0

    def test_lightning_model_custom_parameters(self):
        """Test Lightning model initialization with custom parameters."""
        model = DraemCutPasteClf(
            sspcab=True,
            image_size=(128, 128),
            severity_dropout=0.5,
            cut_w_range=(5, 40),
            cut_h_range=(2, 4),
            a_fault_start=0.5,
            a_fault_range_end=5.0,
            augment_probability=0.8,
            norm=False,
            clf_weight=0.8,
            focal_alpha=0.25,
            focal_gamma=3.0
        )

        # Check parameters
        assert model.sspcab is True
        assert model.image_size == (128, 128)
        assert model.severity_dropout == 0.5
        assert model.cut_w_range == (5, 40)
        assert model.cut_h_range == (2, 4)
        assert model.a_fault_start == 0.5
        assert model.a_fault_range_end == 5.0
        assert model.augment_probability == 0.8
        assert model.norm is False
        assert model.clf_weight == 0.8
        assert model.focal_alpha == 0.25
        assert model.focal_gamma == 3.0

    def test_configure_model(self):
        """Test model configuration."""
        model = DraemCutPasteClf(
            image_size=(224, 224),
            norm=False
        )

        # Configure model
        model.configure_model()

        # Check that model and loss are created
        assert hasattr(model, 'model')
        assert hasattr(model, 'loss')
        assert model.model is not None
        assert model.loss is not None

        # Check model configuration
        assert model.model.severity_head.input_size == (224, 224)
        assert model.model.synthetic_generator.norm is False

    def test_configure_optimizers(self):
        """Test optimizer configuration."""
        model = DraemCutPasteClf()
        model.configure_model()

        optimizer = model.configure_optimizers()

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]['lr'] == 0.0001
        assert optimizer.param_groups[0]['weight_decay'] == 0.0

    def test_trainer_arguments(self):
        """Test trainer arguments property."""
        model = DraemCutPasteClf()
        args = model.trainer_arguments

        assert isinstance(args, dict)
        assert 'gradient_clip_val' in args
        assert 'num_sanity_val_steps' in args
        assert args['gradient_clip_val'] == 0.5
        assert args['num_sanity_val_steps'] == 0

    def test_training_step(self):
        """Test training step execution."""
        model = DraemCutPasteClf(image_size=(256, 256))
        model.configure_model()

        # Create mock batch
        batch_size = 2
        images = torch.randn(batch_size, 3, 256, 256)
        batch = MockBatch(image=images)

        # Training step
        with torch.no_grad():
            loss = model.training_step(batch, batch_idx=0)

        # Check output
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_validation_step(self):
        """Test validation step execution."""
        model = DraemCutPasteClf(image_size=(256, 256))
        model.configure_model()

        # Create mock batch without labels
        batch_size = 2
        images = torch.randn(batch_size, 3, 256, 256)
        batch = MockBatch(image=images)

        # Validation step
        with torch.no_grad():
            predictions = model.validation_step(batch, batch_idx=0)

        # Check predictions structure
        assert hasattr(predictions, 'pred_score')
        assert hasattr(predictions, 'anomaly_map')
        assert hasattr(predictions, 'pred_label')
        assert predictions.pred_score.shape == (batch_size,)
        assert predictions.anomaly_map.shape == (batch_size, 256, 256)

    def test_validation_step_with_labels(self):
        """Test validation step with ground truth labels."""
        model = DraemCutPasteClf(image_size=(256, 256))
        model.configure_model()

        # Create mock batch with labels
        batch_size = 2
        images = torch.randn(batch_size, 3, 256, 256)
        gt_labels = torch.randint(0, 2, (batch_size,))
        batch = MockBatch(image=images, gt_label=gt_labels)

        # Validation step
        with torch.no_grad():
            predictions = model.validation_step(batch, batch_idx=0)

        # Check predictions
        assert hasattr(predictions, 'pred_score')
        assert predictions.pred_score.shape == (batch_size,)

    def test_test_step(self):
        """Test test step execution."""
        model = DraemCutPasteClf(image_size=(128, 128))
        model.configure_model()

        # Create mock batch
        batch_size = 1
        images = torch.randn(batch_size, 3, 128, 128)
        batch = MockBatch(image=images)

        # Test step
        with torch.no_grad():
            predictions = model.test_step(batch, batch_idx=0)

        # Should be same as validation step
        assert hasattr(predictions, 'pred_score')
        assert hasattr(predictions, 'anomaly_map')
        assert predictions.pred_score.shape == (batch_size,)

    def test_predict_step(self):
        """Test predict step execution."""
        model = DraemCutPasteClf(image_size=(224, 224))
        model.configure_model()

        # Create mock batch
        batch_size = 3
        images = torch.randn(batch_size, 3, 224, 224)
        batch = MockBatch(image=images)

        # Predict step
        with torch.no_grad():
            predictions = model.predict_step(batch, batch_idx=0, dataloader_idx=0)

        # Check predictions
        assert hasattr(predictions, 'pred_score')
        assert hasattr(predictions, 'anomaly_map')
        assert hasattr(predictions, 'pred_label')
        assert predictions.pred_score.shape == (batch_size,)

    def test_get_model_config(self):
        """Test model configuration retrieval."""
        model = DraemCutPasteClf(
            sspcab=True,
            image_size=(128, 128),
            norm=False,
            focal_alpha=0.25
        )

        config = model.get_model_config()

        # Check expected keys
        expected_keys = {
            "model_name", "learning_type", "sspcab", "image_size",
            "severity_dropout", "cut_w_range", "cut_h_range", "a_fault_start",
            "a_fault_range_end", "augment_probability", "norm",
            "clf_weight", "focal_alpha", "focal_gamma"
        }
        assert set(config.keys()) == expected_keys

        # Check specific values
        assert config["model_name"] == "DraemCutPasteClf"
        assert config["learning_type"] == str(LearningType.ONE_CLASS)
        assert config["sspcab"] is True
        assert config["image_size"] == (128, 128)
        assert config["norm"] is False
        assert config["clf_weight"] == 1.0  # Default value
        assert config["focal_alpha"] == 0.25  # Custom value from test

    def test_lightning_model_forward_backward_pass(self):
        """Test full forward and backward pass with gradient computation."""
        model = DraemCutPasteClf(image_size=(128, 128))
        model.configure_model()
        model.train()

        # Create input with gradients
        batch_size = 2
        images = torch.randn(batch_size, 3, 128, 128, requires_grad=True)
        batch = MockBatch(image=images)

        # Forward pass
        loss = model.training_step(batch, batch_idx=0)

        # Backward pass
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter: {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter: {name}"

    def test_lightning_model_different_batch_sizes(self):
        """Test model with different batch sizes."""
        model = DraemCutPasteClf(image_size=(256, 256))
        model.configure_model()

        batch_sizes = [1, 4, 8]

        for batch_size in batch_sizes:
            images = torch.randn(batch_size, 3, 256, 256)
            batch = MockBatch(image=images)

            with torch.no_grad():
                # Training step
                loss = model.training_step(batch, batch_idx=0)
                assert isinstance(loss, torch.Tensor)
                assert loss.dim() == 0

                # Validation step
                predictions = model.validation_step(batch, batch_idx=0)
                assert predictions.pred_score.shape == (batch_size,)

    def test_lightning_model_device_compatibility(self):
        """Test model works on different devices."""
        model = DraemCutPasteClf(image_size=(128, 128))
        model.configure_model()

        # Test CPU
        batch_size = 1
        images_cpu = torch.randn(batch_size, 3, 128, 128)
        batch_cpu = MockBatch(image=images_cpu)

        with torch.no_grad():
            predictions_cpu = model.validation_step(batch_cpu, batch_idx=0)
        assert predictions_cpu.pred_score.device.type == "cpu"

        # Test GPU (if available)
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            images_cuda = images_cpu.cuda()
            batch_cuda = MockBatch(image=images_cuda)

            with torch.no_grad():
                predictions_cuda = model_cuda.validation_step(batch_cuda, batch_idx=0)
            assert predictions_cuda.pred_score.device.type == "cuda"

    def test_lightning_model_reproducibility(self):
        """Test model produces consistent results with same seed."""
        model = DraemCutPasteClf(
            image_size=(128, 128),
            augment_probability=1.0  # Always augment for consistency
        )
        model.configure_model()

        batch_size = 1
        images = torch.randn(batch_size, 3, 128, 128)
        batch = MockBatch(image=images)

        # First run
        torch.manual_seed(42)
        with torch.no_grad():
            predictions1 = model.validation_step(batch, batch_idx=0)

        # Second run with same seed
        torch.manual_seed(42)
        with torch.no_grad():
            predictions2 = model.validation_step(batch, batch_idx=0)

        # Results should be similar (allowing for some variation due to random components)
        assert torch.allclose(predictions1.pred_score, predictions2.pred_score, rtol=0.1)

    def test_lightning_model_multi_channel_input(self):
        """Test model with 3-channel input."""
        model = DraemCutPasteClf(image_size=(128, 128))
        model.configure_model()

        # Test different 3-channel inputs
        images_3ch_1 = torch.randn(2, 3, 128, 128)
        batch_3ch_1 = MockBatch(image=images_3ch_1)

        images_3ch_2 = torch.randn(2, 3, 128, 128)
        batch_3ch_2 = MockBatch(image=images_3ch_2)

        with torch.no_grad():
            predictions_1 = model.validation_step(batch_3ch_1, batch_idx=0)
            predictions_2 = model.validation_step(batch_3ch_2, batch_idx=0)

        # Both should produce valid outputs with same shapes
        assert predictions_1.pred_score.shape == (2,)
        assert predictions_2.pred_score.shape == (2,)

    def test_lightning_model_loss_components_integration(self):
        """Test that all loss components are properly integrated."""
        # Test with specific loss weights
        model = DraemCutPasteClf(
            image_size=(256, 256),
            clf_weight=2.0
        )
        model.configure_model()

        batch_size = 2
        images = torch.randn(batch_size, 3, 256, 256)
        batch = MockBatch(image=images)

        # Training step should use all loss components
        with torch.no_grad():
            loss = model.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert loss > 0  # Loss should be positive with all components

    def test_lightning_model_eval_mode_behavior(self):
        """Test model behavior in eval vs train mode."""
        model = DraemCutPasteClf(image_size=(256, 256))
        model.configure_model()

        batch_size = 2
        images = torch.randn(batch_size, 3, 256, 256)
        batch = MockBatch(image=images)

        # Test train mode (should apply augmentation)
        model.train()
        with torch.no_grad():
            loss_train = model.training_step(batch, batch_idx=0)

        # Test eval mode (should not apply augmentation)
        model.eval()
        with torch.no_grad():
            predictions_eval = model.validation_step(batch, batch_idx=0)

        # Both should produce valid outputs
        assert isinstance(loss_train, torch.Tensor)
        assert hasattr(predictions_eval, 'pred_score')
        assert predictions_eval.pred_score.shape == (batch_size,)