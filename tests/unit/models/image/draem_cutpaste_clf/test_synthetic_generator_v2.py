"""Test CutPaste Synthetic Generator for DRAEM CutPaste Classification."""

import pytest
import torch

from anomalib.models.image.draem_cutpaste_clf.synthetic_generator_v2 import CutPasteSyntheticGenerator


class TestCutPasteSyntheticGenerator:
    """Test cases for CutPasteSyntheticGenerator."""

    @pytest.fixture
    def generator_configs(self):
        """Test configurations for different generator settings."""
        return [
            {
                "cut_w_range": (10, 80),
                "cut_h_range": (1, 2),
                "a_fault_start": 1.0,
                "a_fault_range_end": 10.0,
                "probability": 0.5,
                "norm": True,
                "name": "Default with normalization"
            },
            {
                "cut_w_range": (5, 40),
                "cut_h_range": (1, 3),
                "a_fault_start": 0.5,
                "a_fault_range_end": 5.0,
                "probability": 0.8,
                "norm": False,
                "name": "Custom without normalization"
            }
        ]

    def test_generator_initialization(self):
        """Test generator initialization with default parameters."""
        generator = CutPasteSyntheticGenerator()

        assert generator.cut_w_range == (10, 80)
        assert generator.cut_h_range == (1, 2)
        assert generator.a_fault_start == 1.0
        assert generator.a_fault_range_end == 10.0
        assert generator.probability == 0.5
        assert generator.norm is True
        assert generator.validation_enabled is True

    def test_generator_custom_parameters(self):
        """Test generator initialization with custom parameters."""
        generator = CutPasteSyntheticGenerator(
            cut_w_range=(5, 50),
            cut_h_range=(2, 4),
            a_fault_start=0.1,
            a_fault_range_end=5.0,
            probability=0.8,
            norm=False,
            validation_enabled=False
        )

        assert generator.cut_w_range == (5, 50)
        assert generator.cut_h_range == (2, 4)
        assert generator.a_fault_start == 0.1
        assert generator.a_fault_range_end == 5.0
        assert generator.probability == 0.8
        assert generator.norm is False
        assert generator.validation_enabled is False

    def test_generator_parameter_validation(self):
        """Test parameter validation raises appropriate errors."""
        # Test invalid cut width
        with pytest.raises(ValueError, match="Cut width values must be positive"):
            CutPasteSyntheticGenerator(cut_w_range=(0, 10))

        # Test invalid cut height
        with pytest.raises(ValueError, match="Cut height values must be positive"):
            CutPasteSyntheticGenerator(cut_h_range=(-1, 2))

        # Test invalid fault amplitude
        with pytest.raises(ValueError, match="Fault amplitude values must be non-negative"):
            CutPasteSyntheticGenerator(a_fault_start=-1.0)

        # Test invalid fault range
        with pytest.raises(ValueError, match="a_fault_start must be less than a_fault_range_end"):
            CutPasteSyntheticGenerator(a_fault_start=5.0, a_fault_range_end=2.0)

        # Test invalid probability
        with pytest.raises(ValueError, match="Probability must be between 0.0 and 1.0"):
            CutPasteSyntheticGenerator(probability=1.5)

    @pytest.mark.parametrize("batch_size,channels,height,width", [
        (1, 1, 256, 256),
        (4, 1, 224, 224),
        (2, 3, 128, 128),
        (1, 3, 95, 95),
    ])
    def test_generator_forward_pass(self, batch_size, channels, height, width):
        """Test forward pass with different input sizes."""
        generator = CutPasteSyntheticGenerator(probability=0.8)  # High prob for testing
        test_input = torch.randn(batch_size, channels, height, width)

        # Forward pass
        with torch.no_grad():
            synthetic_image, fault_mask, severity_map, severity_label = generator(test_input)

        # Check output shapes
        assert synthetic_image.shape == test_input.shape
        assert fault_mask.shape == (batch_size, 1, height, width)
        assert severity_map.shape == (batch_size, 1, height, width)
        assert severity_label.shape == (batch_size,)

        # Check output types and ranges
        assert torch.isfinite(synthetic_image).all()
        assert torch.all(fault_mask >= 0) and torch.all(fault_mask <= 1)
        assert torch.all(severity_map >= 0) and torch.all(severity_map <= 1)
        assert torch.all(severity_label >= 0)

    def test_generator_with_normalization(self):
        """Test generator behavior with normalization enabled."""
        generator = CutPasteSyntheticGenerator(
            probability=1.0,  # Always generate anomaly
            norm=True,
            a_fault_start=2.0,
            a_fault_range_end=5.0
        )
        test_input = torch.randn(2, 1, 128, 128)

        with torch.no_grad():
            synthetic_image, fault_mask, severity_map, severity_label = generator(test_input)

        # Check that anomalies were generated (probability=1.0)
        assert fault_mask.sum() > 0
        assert severity_label.max() > 0

        # Check severity range (should be within expected range due to normalization)
        assert torch.all(severity_label >= 0)
        assert torch.all(severity_label <= generator.a_fault_range_end)

    def test_generator_without_normalization(self):
        """Test generator behavior with normalization disabled."""
        generator = CutPasteSyntheticGenerator(
            probability=1.0,  # Always generate anomaly
            norm=False,
            a_fault_start=1.0,
            a_fault_range_end=3.0
        )
        test_input = torch.randn(2, 1, 128, 128)

        with torch.no_grad():
            synthetic_image, fault_mask, severity_map, severity_label = generator(test_input)

        # Check that anomalies were generated
        assert fault_mask.sum() > 0
        assert severity_label.max() > 0

    def test_generator_probability_zero(self):
        """Test generator with probability=0 (no anomalies)."""
        generator = CutPasteSyntheticGenerator(probability=0.0)
        test_input = torch.randn(4, 1, 256, 256)

        with torch.no_grad():
            synthetic_image, fault_mask, severity_map, severity_label = generator(test_input)

        # Should be identical to input with no anomalies
        assert torch.allclose(synthetic_image, test_input)
        assert fault_mask.sum() == 0
        assert severity_map.sum() == 0
        assert torch.all(severity_label == 0)

    def test_generator_with_patch_info(self):
        """Test generator with detailed patch information."""
        generator = CutPasteSyntheticGenerator(probability=1.0, norm=True)
        test_input = torch.randn(1, 1, 256, 256)

        with torch.no_grad():
            synthetic_image, fault_mask, severity_map, severity_label, patch_info = generator(
                test_input, return_patch_info=True
            )

        # Check patch info structure
        expected_keys = {
            "cut_w", "cut_h", "from_location_h", "from_location_w",
            "to_location_h", "to_location_w", "a_fault", "has_anomaly",
            "patch_type", "coverage_percentage", "approach",
            "patch_amplitude_scaling"
        }
        assert set(patch_info.keys()) == expected_keys
        assert patch_info["has_anomaly"] == 1
        assert patch_info["approach"] == "CutPaste with amplitude scaling"
        assert patch_info["coverage_percentage"] > 0

    def test_generator_with_patch_info_no_anomaly(self):
        """Test generator patch info when no anomaly is generated."""
        generator = CutPasteSyntheticGenerator(probability=0.0, norm=True)
        test_input = torch.randn(1, 1, 256, 256)

        with torch.no_grad():
            synthetic_image, fault_mask, severity_map, severity_label, patch_info = generator(
                test_input, return_patch_info=True
            )

        # Check patch info for normal case
        assert patch_info["has_anomaly"] == 0
        assert patch_info["coverage_percentage"] == 0.0
        assert "No fault" in patch_info["patch_type"]

    def test_generator_batch_consistency(self):
        """Test that batch processing works consistently."""
        generator = CutPasteSyntheticGenerator(probability=0.5)
        test_input = torch.randn(8, 1, 128, 128)

        with torch.no_grad():
            synthetic_image, fault_mask, severity_map, severity_label = generator(test_input)

        # Each sample should be processed independently
        assert synthetic_image.shape[0] == 8
        assert fault_mask.shape[0] == 8
        assert severity_label.shape[0] == 8

        # Some samples should have anomalies, some shouldn't (due to probability)
        # This is probabilistic, so we don't assert exact counts

    def test_generator_multi_channel_input(self):
        """Test generator with multi-channel input (processes only first channel)."""
        generator = CutPasteSyntheticGenerator(probability=1.0)
        test_input = torch.randn(2, 3, 256, 256)

        with torch.no_grad():
            synthetic_image, fault_mask, severity_map, severity_label = generator(test_input)

        # Output should maintain multi-channel structure
        assert synthetic_image.shape == (2, 3, 256, 256)
        assert fault_mask.shape == (2, 1, 256, 256)

        # Only first channel should be modified
        # Other channels should remain unchanged (this is the expected behavior)

    def test_generator_get_config_info(self):
        """Test get_config_info method."""
        generator = CutPasteSyntheticGenerator(
            cut_w_range=(5, 40),
            norm=False,
            probability=0.7
        )

        config = generator.get_config_info()

        expected_keys = {
            "cut_w_range", "cut_h_range", "a_fault_start", "a_fault_range_end",
            "probability", "validation_enabled", "approach", "version"
        }
        assert set(config.keys()) == expected_keys
        assert config["cut_w_range"] == (5, 40)
        assert config["probability"] == 0.7
        assert config["approach"] == "CutPaste with amplitude scaling"

    def test_generator_device_compatibility(self):
        """Test generator works on different devices."""
        generator = CutPasteSyntheticGenerator(probability=1.0)

        # Test CPU
        test_input_cpu = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            synthetic_cpu, mask_cpu, _, label_cpu = generator(test_input_cpu)

        assert synthetic_cpu.device.type == "cpu"
        assert mask_cpu.device.type == "cpu"
        assert label_cpu.device.type == "cpu"

        # Test GPU (if available)
        if torch.cuda.is_available():
            generator_cuda = generator.cuda()
            test_input_cuda = test_input_cpu.cuda()

            with torch.no_grad():
                synthetic_cuda, mask_cuda, _, label_cuda = generator_cuda(test_input_cuda)

            assert synthetic_cuda.device.type == "cuda"
            assert mask_cuda.device.type == "cuda"
            assert label_cuda.device.type == "cuda"

    def test_generator_image_dimension_validation(self):
        """Test that generator validates image dimensions properly."""
        # Create generator with large patch size
        generator = CutPasteSyntheticGenerator(
            cut_w_range=(100, 200),
            cut_h_range=(50, 60),
            validation_enabled=True
        )

        # Small image should fail validation
        small_input = torch.randn(1, 1, 32, 32)
        with pytest.raises(ValueError, match="too large for image"):
            generator(small_input)

        # Disable validation should work
        generator_no_validation = CutPasteSyntheticGenerator(
            cut_w_range=(100, 200),
            validation_enabled=False
        )
        # This should not raise an error (though results may be invalid)
        with torch.no_grad():
            _ = generator_no_validation(small_input)