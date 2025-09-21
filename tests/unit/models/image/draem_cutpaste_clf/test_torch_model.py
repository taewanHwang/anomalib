"""Test PyTorch Model for DRAEM CutPaste Classification."""

import pytest
import torch

from anomalib.data import InferenceBatch
from anomalib.models.image.draem_cutpaste_clf.torch_model import DraemCutPasteModel


class TestDraemCutPasteModel:
    """Test cases for DraemCutPasteModel."""

    @pytest.fixture
    def model_configs(self):
        """Test configurations for different model settings."""
        return [
            {
                "sspcab": False,
                "image_size": (256, 256),
                "norm": True,
                "name": "Default 256x256 with norm"
            },
            {
                "sspcab": True,
                "image_size": (224, 224),
                "norm": False,
                "name": "SSPCAB 224x224 without norm"
            },
            {
                "sspcab": False,
                "image_size": (128, 128),
                "cut_w_range": (5, 40),
                "cut_h_range": (1, 3),
                "a_fault_start": 0.5,
                "a_fault_range_end": 5.0,
                "augment_probability": 0.8,
                "norm": True,
                "name": "Custom small model"
            }
        ]

    def test_model_initialization(self):
        """Test model initialization with default parameters."""
        model = DraemCutPasteModel()

        # Check that all sub-networks exist
        assert hasattr(model, 'reconstructive_subnetwork')
        assert hasattr(model, 'discriminative_subnetwork')
        assert hasattr(model, 'severity_head')
        assert hasattr(model, 'synthetic_generator')

        # Check default parameters
        assert model.severity_head.input_size == (256, 256)
        assert model.synthetic_generator.norm is True
        assert model.synthetic_generator.probability == 0.5

    def test_model_custom_parameters(self):
        """Test model initialization with custom parameters."""
        model = DraemCutPasteModel(
            sspcab=True,
            image_size=(128, 128),
            severity_dropout=0.5,
            cut_w_range=(5, 30),
            cut_h_range=(2, 4),
            a_fault_start=0.1,
            a_fault_range_end=2.0,
            augment_probability=0.8,
            norm=False
        )

        # Check custom parameters
        assert model.severity_head.input_size == (128, 128)
        assert model.severity_head.dropout_rate == 0.5
        assert model.synthetic_generator.cut_w_range == (5, 30)
        assert model.synthetic_generator.cut_h_range == (2, 4)
        assert model.synthetic_generator.a_fault_start == 0.1
        assert model.synthetic_generator.a_fault_range_end == 2.0
        assert model.synthetic_generator.probability == 0.8
        assert model.synthetic_generator.norm is False

    @pytest.mark.parametrize("batch_size,height,width", [
        (1, 256, 256),
        (4, 224, 224),
        (2, 128, 128),
    ])
    def test_model_training_forward_pass(self, batch_size, height, width):
        """Test forward pass during training phase."""
        model = DraemCutPasteModel(
            image_size=(height, width),
            augment_probability=0.8  # Higher probability for testing
        )
        model.train()

        # Test input (3-channel RGB)
        test_input = torch.randn(batch_size, 3, height, width)

        # Training forward pass
        with torch.no_grad():
            reconstruction, prediction, classification = model(test_input, training_phase=True)

        # Check output shapes
        assert reconstruction.shape == test_input.shape  # Same as input
        assert prediction.shape == (batch_size, 2, height, width)  # 2-class prediction
        assert classification.shape == (batch_size, 2)  # Binary classification

        # Check outputs are finite
        assert torch.isfinite(reconstruction).all()
        assert torch.isfinite(prediction).all()
        assert torch.isfinite(classification).all()

    @pytest.mark.parametrize("batch_size,height,width", [
        (1, 256, 256),
        (4, 224, 224),
        (2, 128, 128),
    ])
    def test_model_inference_forward_pass(self, batch_size, height, width):
        """Test forward pass during inference phase."""
        model = DraemCutPasteModel(image_size=(height, width))
        model.eval()

        # Test input (3-channel RGB)
        test_input = torch.randn(batch_size, 3, height, width)

        # Inference forward pass
        with torch.no_grad():
            output = model(test_input, training_phase=False)

        # Check output is InferenceBatch
        assert isinstance(output, InferenceBatch)

        # Check output shapes
        assert output.anomaly_map.shape == (batch_size, height, width)
        assert output.pred_score.shape == (batch_size,)
        assert output.pred_label.shape == (batch_size,)
        assert output.pred_mask.shape == (batch_size, height, width)

        # Check output ranges and properties
        assert torch.all(output.pred_score >= 0) and torch.all(output.pred_score <= 1)
        assert torch.all((output.pred_label == 0) | (output.pred_label == 1))
        assert torch.isfinite(output.anomaly_map).all()

    def test_model_training_vs_eval_mode(self):
        """Test that model behaves differently in training vs eval mode."""
        model = DraemCutPasteModel(image_size=(256, 256))
        test_input = torch.randn(2, 3, 256, 256)

        # Training mode
        model.train()
        with torch.no_grad():
            train_outputs = model(test_input, training_phase=True)

        # Eval mode
        model.eval()
        with torch.no_grad():
            eval_outputs = model(test_input, training_phase=False)

        # Training should return tuple, eval should return InferenceBatch
        assert isinstance(train_outputs, tuple) and len(train_outputs) == 3
        assert isinstance(eval_outputs, InferenceBatch)

    def test_model_gradient_flow(self):
        """Test that gradients flow properly through all sub-networks."""
        model = DraemCutPasteModel(image_size=(128, 128))
        model.train()

        # Create input with gradients
        test_input = torch.randn(2, 3, 128, 128, requires_grad=True)
        target_reconstruction = torch.randn(2, 3, 128, 128)
        target_prediction = torch.randint(0, 2, (2, 1, 128, 128)).long()
        target_classification = torch.randint(0, 2, (2,)).long()

        # Forward pass
        reconstruction, prediction, classification = model(test_input, training_phase=True)

        # Compute losses
        recon_loss = torch.nn.functional.mse_loss(reconstruction, target_reconstruction)
        pred_loss = torch.nn.functional.cross_entropy(prediction, target_prediction.squeeze(1))
        clf_loss = torch.nn.functional.cross_entropy(classification, target_classification)
        total_loss = recon_loss + pred_loss + clf_loss

        # Backward pass
        total_loss.backward()

        # Check gradients exist for all sub-networks
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter: {name}"

    def test_model_parameter_count(self):
        """Test that model has reasonable number of parameters."""
        model = DraemCutPasteModel(image_size=(256, 256))

        config = model.get_model_config()
        total_params = config["total_parameters"]
        trainable_params = config["trainable_parameters"]

        # Should have parameters
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params

        # Should be reasonable number (not too large for test)
        assert total_params < 500_000_000  # Less than 500M parameters

    def test_model_get_config(self):
        """Test get_model_config method."""
        model = DraemCutPasteModel(
            sspcab=True,
            image_size=(224, 224),
            norm=False,
            augment_probability=0.7
        )

        config = model.get_model_config()

        # Check expected keys exist
        expected_keys = {
            "model_name", "reconstructive_network", "discriminative_network",
            "severity_head", "synthetic_generator", "total_parameters", "trainable_parameters"
        }
        assert set(config.keys()) == expected_keys

        # Check values
        assert config["model_name"] == "DraemCutPasteClf"
        assert "severity_head" in config
        assert "synthetic_generator" in config
        assert isinstance(config["total_parameters"], int)
        assert isinstance(config["trainable_parameters"], int)

    def test_model_with_different_augment_probabilities(self):
        """Test model with different augmentation probabilities."""
        test_input = torch.randn(4, 3, 256, 256)

        # Test with no augmentation (probability=0)
        model_no_aug = DraemCutPasteModel(augment_probability=0.0)
        model_no_aug.train()
        with torch.no_grad():
            recon_no, pred_no, clf_no = model_no_aug(test_input, training_phase=True)

        # Test with full augmentation (probability=1.0)
        model_full_aug = DraemCutPasteModel(augment_probability=1.0)
        model_full_aug.train()
        with torch.no_grad():
            recon_full, pred_full, clf_full = model_full_aug(test_input, training_phase=True)

        # Outputs should have same shapes but potentially different values
        assert recon_no.shape == recon_full.shape
        assert pred_no.shape == pred_full.shape
        assert clf_no.shape == clf_full.shape

    def test_model_multi_channel_input_consistency(self):
        """Test that model handles 3-channel input consistently."""
        model = DraemCutPasteModel(image_size=(128, 128))

        # Test with different 3-channel inputs
        input_3ch_1 = torch.randn(2, 3, 128, 128)
        input_3ch_2 = torch.randn(2, 3, 128, 128)

        model.eval()
        with torch.no_grad():
            output_1 = model(input_3ch_1, training_phase=False)
            output_2 = model(input_3ch_2, training_phase=False)

        # Outputs should have consistent shapes (batch size and spatial dimensions)
        assert output_1.pred_score.shape == output_2.pred_score.shape
        assert output_1.anomaly_map.shape == output_2.anomaly_map.shape

    def test_model_device_compatibility(self):
        """Test model works on different devices."""
        model = DraemCutPasteModel(image_size=(128, 128))

        # Test CPU
        test_input_cpu = torch.randn(1, 3, 128, 128)
        model.eval()
        with torch.no_grad():
            output_cpu = model(test_input_cpu, training_phase=False)

        assert output_cpu.pred_score.device.type == "cpu"
        assert output_cpu.anomaly_map.device.type == "cpu"

        # Test GPU (if available)
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            test_input_cuda = test_input_cpu.cuda()

            with torch.no_grad():
                output_cuda = model_cuda(test_input_cuda, training_phase=False)

            assert output_cuda.pred_score.device.type == "cuda"
            assert output_cuda.anomaly_map.device.type == "cuda"

    def test_model_reproducibility_with_seed(self):
        """Test that model produces consistent results with same seed."""
        model = DraemCutPasteModel(
            image_size=(128, 128),
            augment_probability=1.0  # Always augment for consistency
        )
        test_input = torch.randn(1, 3, 128, 128)

        # First run
        torch.manual_seed(42)
        model.eval()
        with torch.no_grad():
            output1 = model(test_input, training_phase=False)

        # Second run with same seed
        torch.manual_seed(42)
        model.eval()
        with torch.no_grad():
            output2 = model(test_input, training_phase=False)

        # Results should be identical (or very close due to floating point)
        assert torch.allclose(output1.pred_score, output2.pred_score, rtol=1e-5)

    def test_model_inference_batch_properties(self):
        """Test that InferenceBatch output has correct properties."""
        model = DraemCutPasteModel(image_size=(256, 256))
        test_input = torch.randn(3, 3, 256, 256)

        model.eval()
        with torch.no_grad():
            output = model(test_input, training_phase=False)

        # Check that anomaly_map and pred_mask are the same (as per implementation)
        assert torch.equal(output.anomaly_map, output.pred_mask)

        # Check that pred_label are binary (0 or 1)
        unique_labels = torch.unique(output.pred_label)
        assert torch.all((unique_labels == 0) | (unique_labels == 1))

        # Check that pred_score are probabilities [0, 1]
        assert torch.all(output.pred_score >= 0)
        assert torch.all(output.pred_score <= 1)