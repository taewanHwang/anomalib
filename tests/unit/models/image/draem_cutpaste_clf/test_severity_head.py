"""Test Severity Head for DRAEM CutPaste Classification."""

import pytest
import torch

from anomalib.models.image.draem_cutpaste_clf.severity_head import SeverityHead


class TestSeverityHead:
    """Test cases for SeverityHead."""

    @pytest.fixture
    def severity_head_configs(self):
        """Test configurations for different input sizes."""
        return [
            {"input_size": (256, 256), "batch_size": 4},
            {"input_size": (224, 224), "batch_size": 2},
            {"input_size": (128, 128), "batch_size": 1},
            {"input_size": (95, 95), "batch_size": 3},
        ]

    def test_severity_head_initialization(self):
        """Test SeverityHead initialization with default parameters."""
        model = SeverityHead()

        assert model.in_channels == 2
        assert model.dropout_rate == 0.3
        assert model.input_size == (256, 256)

        # Check layer existence
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'conv2')
        assert hasattr(model, 'conv3')
        assert hasattr(model, 'conv4')
        assert hasattr(model, 'fc1')
        assert hasattr(model, 'fc2')
        assert hasattr(model, 'fc3')

    def test_severity_head_custom_params(self):
        """Test SeverityHead initialization with custom parameters."""
        model = SeverityHead(
            in_channels=3,
            dropout_rate=0.5,
            input_size=(128, 128)
        )

        assert model.in_channels == 3
        assert model.dropout_rate == 0.5
        assert model.input_size == (128, 128)

    @pytest.mark.parametrize("input_size,batch_size", [
        ((256, 256), 4),
        ((224, 224), 2),
        ((128, 128), 1),
        ((95, 95), 3),
    ])
    def test_severity_head_forward_pass(self, input_size, batch_size):
        """Test forward pass with different input sizes."""
        h, w = input_size
        model = SeverityHead(in_channels=2, input_size=input_size)

        # Create test input (2-channel: original + mask)
        test_input = torch.randn(batch_size, 2, h, w)

        # Forward pass
        with torch.no_grad():
            output = model(test_input)

        # Check output shape
        assert output.shape == (batch_size, 2)  # Binary classification

        # Check output is finite
        assert torch.isfinite(output).all()

    def test_severity_head_output_range(self):
        """Test that output logits can be converted to probabilities."""
        model = SeverityHead(input_size=(256, 256))
        test_input = torch.randn(2, 2, 256, 256)

        with torch.no_grad():
            logits = model(test_input)
            probabilities = torch.softmax(logits, dim=1)

        # Check probability properties
        assert probabilities.shape == (2, 2)
        assert torch.all(probabilities >= 0)
        assert torch.all(probabilities <= 1)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))

    def test_severity_head_training_mode(self):
        """Test model behavior in training vs eval mode."""
        model = SeverityHead(input_size=(256, 256), dropout_rate=0.5)
        test_input = torch.randn(4, 2, 256, 256)

        # Test training mode (with dropout)
        model.train()
        with torch.no_grad():
            output_train = model(test_input)

        # Test eval mode (without dropout)
        model.eval()
        with torch.no_grad():
            output_eval = model(test_input)

        # Outputs should have same shape but potentially different values
        assert output_train.shape == output_eval.shape
        assert output_train.shape == (4, 2)

    def test_severity_head_parameter_count(self):
        """Test parameter count is reasonable."""
        model = SeverityHead(input_size=(256, 256))

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Should have parameters
        assert total_params > 0
        assert trainable_params == total_params  # All parameters should be trainable

        # Should be reasonable number (not too large)
        assert total_params < 100_000_000  # Less than 100M parameters

    def test_severity_head_get_config(self):
        """Test get_config method returns correct configuration."""
        model = SeverityHead(
            in_channels=2,
            dropout_rate=0.4,
            input_size=(224, 224)
        )

        config = model.get_config()

        expected_keys = {
            "in_channels", "dropout_rate", "input_size",
            "architecture", "num_classes", "conv_channels", "fc_layers"
        }
        assert set(config.keys()) == expected_keys
        assert config["in_channels"] == 2
        assert config["dropout_rate"] == 0.4
        assert config["input_size"] == (224, 224)
        assert config["num_classes"] == 2
        assert config["architecture"] == "CNN_simple_2ch"

    def test_severity_head_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = SeverityHead(input_size=(256, 256))
        test_input = torch.randn(2, 2, 256, 256, requires_grad=True)
        target = torch.randint(0, 2, (2,))

        # Forward pass
        output = model(test_input)
        loss = torch.nn.functional.cross_entropy(output, target)

        # Backward pass
        loss.backward()

        # Check that gradients exist for model parameters
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    @pytest.mark.parametrize("dropout_rate", [0.0, 0.3, 0.7, 1.0])
    def test_severity_head_dropout_rates(self, dropout_rate):
        """Test model with different dropout rates."""
        model = SeverityHead(input_size=(256, 256), dropout_rate=dropout_rate)
        test_input = torch.randn(1, 2, 256, 256)

        with torch.no_grad():
            output = model(test_input)

        assert output.shape == (1, 2)
        assert torch.isfinite(output).all()

    def test_severity_head_device_compatibility(self):
        """Test model works on different devices."""
        model = SeverityHead(input_size=(128, 128))

        # Test CPU
        test_input_cpu = torch.randn(1, 2, 128, 128)
        with torch.no_grad():
            output_cpu = model(test_input_cpu)
        assert output_cpu.device.type == "cpu"

        # Test GPU (if available)
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            test_input_cuda = test_input_cpu.cuda()
            with torch.no_grad():
                output_cuda = model_cuda(test_input_cuda)
            assert output_cuda.device.type == "cuda"