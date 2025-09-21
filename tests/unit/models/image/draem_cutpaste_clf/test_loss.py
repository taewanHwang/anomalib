"""Test Loss Functions for DRAEM CutPaste Classification."""

import pytest
import torch

from anomalib.models.image.draem_cutpaste_clf.loss import DraemCutPasteLoss


class TestDraemCutPasteLoss:
    """Test cases for DraemCutPasteLoss."""

    def test_loss_initialization(self):
        """Test loss initialization with default parameters."""
        loss_fn = DraemCutPasteLoss()

        # Check default parameters
        assert loss_fn.clf_weight == 1.0
        assert hasattr(loss_fn, 'l2_loss')  # Inherited from DraemLoss
        assert hasattr(loss_fn, 'ssim_loss')  # Inherited from DraemLoss
        assert hasattr(loss_fn, 'focal_loss')  # Inherited from DraemLoss
        assert hasattr(loss_fn, 'clf_loss')  # New classification loss

    def test_loss_custom_parameters(self):
        """Test loss initialization with custom parameters."""
        loss_fn = DraemCutPasteLoss(
            clf_weight=2.0,
            focal_alpha=0.5,
            focal_gamma=3.0
        )

        assert loss_fn.clf_weight == 2.0
        # Check that focal loss was updated with custom parameters
        assert hasattr(loss_fn.focal_loss, 'alpha')
        assert hasattr(loss_fn.focal_loss, 'gamma')

    @pytest.mark.parametrize("batch_size,height,width", [
        (1, 128, 128),
        (4, 256, 256),
        (2, 224, 224),
    ])
    def test_loss_forward_pass(self, batch_size, height, width):
        """Test forward pass with different input shapes."""
        loss_fn = DraemCutPasteLoss(clf_weight=1.0)

        # Create test inputs
        reconstruction = torch.randn(batch_size, 3, height, width)
        original = torch.randn(batch_size, 3, height, width)
        prediction = torch.randn(batch_size, 2, height, width)
        anomaly_mask = torch.randint(0, 2, (batch_size, 1, height, width)).float()
        classification = torch.randn(batch_size, 2)
        anomaly_labels = torch.randint(0, 2, (batch_size,))

        # Forward pass
        total_loss, loss_dict = loss_fn(
            reconstruction=reconstruction,
            original=original,
            prediction=prediction,
            anomaly_mask=anomaly_mask,
            classification=classification,
            anomaly_labels=anomaly_labels
        )

        # Check outputs
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.dim() == 0  # Scalar
        assert torch.isfinite(total_loss)
        assert total_loss >= 0

        # Check loss dictionary
        expected_keys = {"loss_l2", "loss_ssim", "loss_focal", "loss_clf", "loss_base", "total_loss"}
        assert set(loss_dict.keys()) == expected_keys

        for key, value in loss_dict.items():
            assert isinstance(value, torch.Tensor)
            assert value.dim() == 0  # Scalar
            assert torch.isfinite(value)
            assert value >= 0

    def test_loss_component_weights(self):
        """Test that classification loss weight is applied correctly."""
        # Test with zero classification weight
        loss_fn_no_clf = DraemCutPasteLoss(clf_weight=0.0)
        loss_fn_with_clf = DraemCutPasteLoss(clf_weight=1.0)

        batch_size = 2
        reconstruction = torch.randn(batch_size, 3, 128, 128)
        original = torch.randn(batch_size, 3, 128, 128)
        prediction = torch.randn(batch_size, 2, 128, 128)
        anomaly_mask = torch.randint(0, 2, (batch_size, 1, 128, 128)).float()
        classification = torch.randn(batch_size, 2)
        anomaly_labels = torch.randint(0, 2, (batch_size,))

        # Forward pass with zero classification weight
        total_loss_no_clf, loss_dict_no_clf = loss_fn_no_clf(
            reconstruction, original, prediction, anomaly_mask, classification, anomaly_labels
        )

        # Forward pass with classification weight
        total_loss_with_clf, loss_dict_with_clf = loss_fn_with_clf(
            reconstruction, original, prediction, anomaly_mask, classification, anomaly_labels
        )

        # Base loss should be the same
        assert torch.allclose(loss_dict_no_clf["loss_base"], loss_dict_with_clf["loss_base"], rtol=1e-5)

        # Total loss with classification should be larger (if clf_loss > 0)
        if loss_dict_with_clf["loss_clf"] > 0:
            assert total_loss_with_clf >= total_loss_no_clf

    def test_loss_gradient_flow(self):
        """Test that gradients flow through all loss components."""
        loss_fn = DraemCutPasteLoss(clf_weight=1.0)

        batch_size = 1
        reconstruction = torch.randn(batch_size, 3, 128, 128, requires_grad=True)
        original = torch.randn(batch_size, 3, 128, 128, requires_grad=True)
        prediction = torch.randn(batch_size, 2, 128, 128, requires_grad=True)
        anomaly_mask = torch.randint(0, 2, (batch_size, 1, 128, 128)).float()
        classification = torch.randn(batch_size, 2, requires_grad=True)
        anomaly_labels = torch.randint(0, 2, (batch_size,))

        # Forward and backward pass
        total_loss, loss_dict = loss_fn(
            reconstruction, original, prediction, anomaly_mask, classification, anomaly_labels
        )
        total_loss.backward()

        # Check gradients exist
        assert reconstruction.grad is not None
        assert original.grad is not None
        assert prediction.grad is not None
        assert classification.grad is not None

        # Check gradients are finite
        assert torch.isfinite(reconstruction.grad).all()
        assert torch.isfinite(original.grad).all()
        assert torch.isfinite(prediction.grad).all()
        assert torch.isfinite(classification.grad).all()

    def test_loss_different_data_types(self):
        """Test loss with different input data types."""
        loss_fn = DraemCutPasteLoss()

        batch_size = 2
        reconstruction = torch.randn(batch_size, 3, 128, 128).float()
        original = torch.randn(batch_size, 3, 128, 128).float()
        prediction = torch.randn(batch_size, 2, 128, 128).float()

        # Test different anomaly_mask types
        anomaly_mask_float = torch.randint(0, 2, (batch_size, 1, 128, 128)).float()
        anomaly_mask_long = torch.randint(0, 2, (batch_size, 1, 128, 128)).long()

        classification = torch.randn(batch_size, 2).float()

        # Test different anomaly_labels types
        anomaly_labels_int = torch.randint(0, 2, (batch_size,)).int()
        anomaly_labels_long = torch.randint(0, 2, (batch_size,)).long()

        # Test combinations
        combinations = [
            (anomaly_mask_float, anomaly_labels_int),
            (anomaly_mask_float, anomaly_labels_long),
            (anomaly_mask_long, anomaly_labels_int),
            (anomaly_mask_long, anomaly_labels_long),
        ]

        for mask, labels in combinations:
            total_loss, loss_dict = loss_fn(
                reconstruction, original, prediction, mask, classification, labels
            )

            assert isinstance(total_loss, torch.Tensor)
            assert torch.isfinite(total_loss)

    def test_loss_device_compatibility(self):
        """Test loss works on different devices."""
        loss_fn = DraemCutPasteLoss()

        batch_size = 1
        # Test CPU
        reconstruction_cpu = torch.randn(batch_size, 3, 128, 128)
        original_cpu = torch.randn(batch_size, 3, 128, 128)
        prediction_cpu = torch.randn(batch_size, 2, 128, 128)
        anomaly_mask_cpu = torch.randint(0, 2, (batch_size, 1, 128, 128)).float()
        classification_cpu = torch.randn(batch_size, 2)
        anomaly_labels_cpu = torch.randint(0, 2, (batch_size,))

        total_loss_cpu, loss_dict_cpu = loss_fn(
            reconstruction_cpu, original_cpu, prediction_cpu,
            anomaly_mask_cpu, classification_cpu, anomaly_labels_cpu
        )

        assert total_loss_cpu.device.type == "cpu"

        # Test GPU if available
        if torch.cuda.is_available():
            loss_fn_cuda = loss_fn.cuda()
            reconstruction_cuda = reconstruction_cpu.cuda()
            original_cuda = original_cpu.cuda()
            prediction_cuda = prediction_cpu.cuda()
            anomaly_mask_cuda = anomaly_mask_cpu.cuda()
            classification_cuda = classification_cpu.cuda()
            anomaly_labels_cuda = anomaly_labels_cpu.cuda()

            total_loss_cuda, loss_dict_cuda = loss_fn_cuda(
                reconstruction_cuda, original_cuda, prediction_cuda,
                anomaly_mask_cuda, classification_cuda, anomaly_labels_cuda
            )

            assert total_loss_cuda.device.type == "cuda"

    def test_get_config(self):
        """Test get_config method."""
        loss_fn = DraemCutPasteLoss(
            clf_weight=2.0,
            focal_alpha=0.5,
            focal_gamma=3.0
        )

        config = loss_fn.get_config()

        # Check expected keys
        expected_keys = {"base_loss", "clf_weight", "focal_alpha", "focal_gamma"}
        assert set(config.keys()) == expected_keys

        # Check values
        assert config["base_loss"] == "DraemLoss"
        assert config["clf_weight"] == 2.0
        assert config["focal_alpha"] == 0.5
        assert config["focal_gamma"] == 3.0