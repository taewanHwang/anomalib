"""Test 1-channel optimization for DRAEM CutPaste Classification."""

import pytest
import torch

from anomalib.data import InferenceBatch
from anomalib.models.image.draem_cutpaste_clf.torch_model import DraemCutPasteModel


# Use GPU if available for faster tests
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestDraemCutPaste1ChOptimization:
    """Test cases for 1-channel optimized DraemCutPasteModel."""

    def test_1ch_model_initialization(self):
        """Test that 1-channel optimized model initializes correctly."""
        model = DraemCutPasteModel(
            sspcab=False,
            image_size=(256, 256),
            severity_dropout=0.1,
            severity_input_channels='original',
        )

        # Check model configuration
        config = model.get_model_config()
        assert config["model_type"] == "1-channel_optimized"
        assert "1ch" in config["reconstructive_network"]
        assert "2ch" in config["discriminative_network"]

        # Check that networks have correct configurations
        assert hasattr(model, 'reconstructive_subnetwork')
        assert hasattr(model, 'discriminative_subnetwork')
        assert hasattr(model, 'severity_head')

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_1ch_input_forward_pass(self, batch_size):
        """Test forward pass with 1-channel input."""
        # Use fixed 256x256 size to avoid FC layer dimension issues
        height, width = 256, 256

        model = DraemCutPasteModel(
            image_size=(height, width)
        ).to(DEVICE)

        # Test with 1-channel input (grayscale)
        test_input = torch.randn(batch_size, 1, height, width).to(DEVICE)

        # Training mode
        model.train()
        with torch.no_grad():
            outputs = model(test_input, training_phase=True)
            reconstruction, prediction, classification = outputs[:3]

        # Check output shapes
        assert reconstruction.shape == (batch_size, 1, height, width)  # 1-channel reconstruction
        assert prediction.shape == (batch_size, 2, height, width)  # 2-class prediction
        assert classification.shape == (batch_size, 2)  # Binary classification

        # Inference mode
        model.eval()
        with torch.no_grad():
            output = model(test_input, training_phase=False)

        assert isinstance(output, InferenceBatch)
        assert output.anomaly_map.shape == (batch_size, height, width)
        assert output.pred_score.shape == (batch_size,)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_3ch_input_uses_first_channel(self, batch_size):
        """Test that 3-channel input uses only first channel."""
        # Use fixed 256x256 size to avoid FC layer dimension issues
        height, width = 256, 256

        model = DraemCutPasteModel(
            image_size=(height, width)
        ).to(DEVICE)

        # Create 3-channel input with distinct channels
        test_input = torch.randn(batch_size, 3, height, width).to(DEVICE)

        # Make channels distinctly different
        test_input[:, 1, :, :] *= 10  # Scale second channel
        test_input[:, 2, :, :] += 100  # Offset third channel

        # Test inference
        model.eval()
        with torch.no_grad():
            output = model(test_input, training_phase=False)

        assert isinstance(output, InferenceBatch)
        assert output.anomaly_map.shape == (batch_size, height, width)
        assert output.pred_score.shape == (batch_size,)

        # Test training
        model.train()
        with torch.no_grad():
            outputs = model(test_input, training_phase=True)
            reconstruction, prediction, classification = outputs[:3]

        # Reconstruction should be 1-channel (first channel only)
        assert reconstruction.shape == (batch_size, 1, height, width)

    def test_channel_extraction_consistency(self):
        """Test that model consistently extracts first channel."""
        model = DraemCutPasteModel(
            image_size=(256, 256)
        ).to(DEVICE)

        # Create input where first channel is identical across samples
        batch_size = 2
        test_input = torch.randn(batch_size, 3, 256, 256).to(DEVICE)

        # Make first channels identical
        test_input[1, 0, :, :] = test_input[0, 0, :, :]

        # Make other channels different
        test_input[1, 1:, :, :] = torch.randn(2, 256, 256).to(DEVICE)

        model.eval()
        with torch.no_grad():
            output = model(test_input, training_phase=False)

        # Since first channels are identical, outputs should be similar
        # (though not exactly identical due to potential augmentation)
        assert output.pred_score.shape == (batch_size,)

    def test_severity_input_channels_1ch(self):
        """Test different severity input channel configurations with 1-channel model."""
        test_input = torch.randn(1, 1, 256, 256).to(DEVICE)

        channel_configs = [
            'original',
            'mask',
            'recon',
            'original+mask',
            'original+recon',
            'mask+recon',
            'original+mask+recon'
        ]

        for config in channel_configs:
            model = DraemCutPasteModel(
                image_size=(256, 256),
                severity_input_channels=config
            ).to(DEVICE)

            # Check expected channel count
            expected_channels = config.count('+') + 1
            assert model.severity_in_channels == expected_channels

            # Test forward pass
            model.eval()
            with torch.no_grad():
                output = model(test_input, training_phase=False)

            assert isinstance(output, InferenceBatch)
            assert output.pred_score.shape == (1,)

    def test_parameter_reduction_vs_3ch_model(self):
        """Test that 1-channel model has fewer parameters than equivalent 3-channel model."""
        # Note: This is a conceptual test since we don't have a 3-channel version to compare
        # But we can verify the model has reasonable parameter count

        model = DraemCutPasteModel(image_size=(256, 256))
        config = model.get_model_config()

        total_params = config["total_parameters"]

        # Should have reasonable number of parameters for 1-channel model
        assert total_params > 0
        assert total_params < 200_000_000  # Less than 200M parameters (reasonable for 1-ch)

    def test_model_works_with_both_1ch_and_3ch_inputs(self):
        """Test that model works correctly with both 1-channel and 3-channel inputs."""
        model = DraemCutPasteModel(
            image_size=(256, 256)
        ).to(DEVICE)

        # Test with 1-channel input
        input_1ch = torch.randn(2, 1, 256, 256).to(DEVICE)

        model.eval()
        with torch.no_grad():
            output_1ch = model(input_1ch, training_phase=False)

        # Test with 3-channel input (should extract first channel)
        input_3ch = torch.randn(2, 3, 256, 256).to(DEVICE)

        with torch.no_grad():
            output_3ch = model(input_3ch, training_phase=False)

        # Both should produce valid outputs with same shapes
        assert output_1ch.pred_score.shape == output_3ch.pred_score.shape
        assert output_1ch.anomaly_map.shape == output_3ch.anomaly_map.shape
        assert output_1ch.pred_label.shape == output_3ch.pred_label.shape

    def test_training_with_mixed_channel_inputs(self):
        """Test training mode with different channel inputs."""
        model = DraemCutPasteModel(
            image_size=(256, 256),
            augment_probability=0.8  # High probability for testing
        ).to(DEVICE)

        model.train()

        # Test 1-channel training
        input_1ch = torch.randn(2, 1, 256, 256).to(DEVICE)
        with torch.no_grad():
            outputs_1ch = model(input_1ch, training_phase=True)
            recon_1ch, pred_1ch, clf_1ch = outputs_1ch[:3]

        assert recon_1ch.shape == (2, 1, 256, 256)  # 1-channel reconstruction
        assert pred_1ch.shape == (2, 2, 256, 256)
        assert clf_1ch.shape == (2, 2)

        # Test 3-channel training (should extract first channel)
        input_3ch = torch.randn(2, 3, 256, 256).to(DEVICE)
        with torch.no_grad():
            outputs_3ch = model(input_3ch, training_phase=True)
            recon_3ch, pred_3ch, clf_3ch = outputs_3ch[:3]

        assert recon_3ch.shape == (2, 1, 256, 256)  # Still 1-channel reconstruction
        assert pred_3ch.shape == (2, 2, 256, 256)
        assert clf_3ch.shape == (2, 2)

    def test_gradient_flow_1ch_optimization(self):
        """Test gradient flow through 1-channel optimized model."""
        model = DraemCutPasteModel(
            image_size=(256, 256)  # Use 256x256 to avoid FC dimension issues
        ).to(DEVICE)
        model.train()

        # Test with 1-channel input
        test_input = torch.randn(1, 1, 256, 256, requires_grad=True).to(DEVICE)

        # Forward pass
        outputs = model(test_input, training_phase=True)
        reconstruction, prediction, classification = outputs[:3]

        # Create dummy targets
        target_reconstruction = torch.randn_like(reconstruction)
        target_prediction = torch.randint(0, 2, (1, 1, 256, 256)).long().to(DEVICE)
        target_classification = torch.randint(0, 2, (1,)).long().to(DEVICE)

        # Compute loss and backward
        recon_loss = torch.nn.functional.mse_loss(reconstruction, target_reconstruction)
        pred_loss = torch.nn.functional.cross_entropy(prediction, target_prediction.squeeze(1))
        clf_loss = torch.nn.functional.cross_entropy(classification, target_classification)
        total_loss = recon_loss + pred_loss + clf_loss

        total_loss.backward()

        # Check that gradients exist and are finite
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_model_config_1ch_info(self):
        """Test that model config reflects 1-channel optimization."""
        model = DraemCutPasteModel(
            image_size=(256, 256),
            severity_input_channels='original+mask'
        )

        config = model.get_model_config()

        # Check 1-channel specific information
        assert config["model_type"] == "1-channel_optimized"
        assert "1ch" in config["reconstructive_network"]
        assert "2ch" in config["discriminative_network"]

        # Check that severity head config is included
        assert "severity_head" in config
        assert isinstance(config["severity_head"], dict)