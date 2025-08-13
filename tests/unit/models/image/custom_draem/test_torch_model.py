"""Test script for Custom DRAEM PyTorch model.

This script tests the Custom DRAEM model components to ensure they work correctly
with the integrated DRAEM backbone.

Author: Taewan Hwang
"""

import torch

from anomalib.models.image.custom_draem.torch_model import (
    CustomDraemModel, 
    FaultSeveritySubNetwork
)


def test_fault_severity_subnetwork():
    """Test FaultSeveritySubNetwork with different input modes."""
    print("🧪 Testing FaultSeveritySubNetwork...")
    
    batch_size, height, width = 2, 224, 224
    
    # Test different input modes with their expected channel counts
    test_modes = [
        ("discriminative_only", 2),
        ("with_original", 5),  # discriminative (2) + original (3)
        ("with_reconstruction", 5),  # discriminative (2) + reconstruction (3)
        ("with_error_map", 5),  # discriminative (2) + error_map (3)
        ("multi_modal", 11)  # discriminative (2) + original (3) + reconstruction (3) + error (3)
    ]
    
    for mode, expected_channels in test_modes:
        print(f"   Testing {mode} mode ({expected_channels} channels)...")
        
        # Create input with appropriate number of channels
        input_tensor = torch.randn(batch_size, expected_channels, height, width)
        
        # Initialize severity network
        severity_net = FaultSeveritySubNetwork(
            in_channels=expected_channels, 
            severity_max=10.0
        )
        severity_net.eval()
        
        # Forward pass
        with torch.no_grad():
            severity_prediction = severity_net(input_tensor)
        
        print(f"      Input shape: {input_tensor.shape}")
        print(f"      Output shape: {severity_prediction.shape}")
        print(f"      Output range: [{severity_prediction.min():.3f}, {severity_prediction.max():.3f}]")
        
        # Verify output properties
        expected_shape = (batch_size, 1)
        assert severity_prediction.shape == expected_shape, f"Shape mismatch: {severity_prediction.shape} != {expected_shape}"
        assert 0 <= severity_prediction.min() and severity_prediction.max() <= 10.0, f"Severity not in [0,10] range"
        
        print(f"      ✅ {mode} test passed!")
    
    return True


def test_custom_draem_model():
    """Test full CustomDraemModel integration with DRAEM backbone."""
    print("\n🧪 Testing CustomDraemModel with DRAEM backbone...")
    
    # Create 3-channel RGB input (new structure with DRAEM backbone)
    batch_size, channels, height, width = 2, 3, 224, 224
    input_tensor = torch.randn(batch_size, channels, height, width)
    
    # Test different severity input modes
    test_modes = ["discriminative_only", "with_original", "with_reconstruction"]
    
    for mode in test_modes:
        print(f"   Testing {mode} mode...")
        
        # Initialize model
        model = CustomDraemModel(
            sspcab=False, 
            severity_max=10.0,
            severity_input_mode=mode
        )
        
        # Test training mode (returns tuple)
        model.train()
        with torch.no_grad():
            outputs = model(input_tensor)
            reconstruction, prediction, severity = outputs
        
        print(f"      Training mode:")
        print(f"         Reconstruction: {reconstruction.shape}")
        print(f"         Prediction: {prediction.shape}")
        print(f"         Severity: {severity.shape}")
        
        # Verify training outputs
        assert reconstruction.shape == (batch_size, 3, height, width)
        assert prediction.shape == (batch_size, 2, height, width)
        assert severity.shape == (batch_size, 1)
        
        # Test inference mode (returns InferenceBatch)
        model.eval()
        with torch.no_grad():
            inference_output = model(input_tensor)
        
        print(f"      Inference mode:")
        print(f"         Pred score: {inference_output.pred_score.shape}")
        print(f"         Anomaly map: {inference_output.anomaly_map.shape}")
        
        # Verify inference outputs
        assert inference_output.pred_score.shape == (batch_size,)
        assert inference_output.anomaly_map.shape == (batch_size, height, width)
        
        print(f"      ✅ {mode} integration test passed!")
    
    return True


def test_model_parameters():
    """Test model parameter counts and memory usage."""
    print("\n📊 Model parameter analysis...")
    
    # Create model without SSPCAB
    model_no_sspcab = CustomDraemModel(severity_input_mode="discriminative_only", sspcab=False)
    
    # Create model with SSPCAB
    model_with_sspcab = CustomDraemModel(severity_input_mode="discriminative_only", sspcab=True)
    
    # Count parameters
    params_no_sspcab = sum(p.numel() for p in model_no_sspcab.parameters())
    params_with_sspcab = sum(p.numel() for p in model_with_sspcab.parameters())
    severity_params = sum(p.numel() for p in model_no_sspcab.fault_severity_subnetwork.parameters())
    
    print(f"   Model without SSPCAB: {params_no_sspcab:,} parameters")
    print(f"   Model with SSPCAB: {params_with_sspcab:,} parameters")
    print(f"   SSPCAB difference: {params_with_sspcab - params_no_sspcab:+,} parameters")
    print(f"   Severity SubNetwork: {severity_params:,} parameters")
    print(f"   Model size (no SSPCAB): ~{params_no_sspcab * 4 / 1024 / 1024:.1f} MB (float32)")
    
    return True


def test_channel_compatibility():
    """Test channel compatibility for different severity input modes."""
    print("\n🧪 Testing channel compatibility...")
    
    batch_size, height, width = 1, 224, 224
    
    # Test cases: (mode, input_channels)
    test_cases = [
        ("discriminative_only", 2),
        ("with_original", 5),
        ("with_reconstruction", 5),
        ("with_error_map", 5),
        ("multi_modal", 11)
    ]
    
    for mode, expected_channels in test_cases:
        print(f"   Testing {mode} with {expected_channels} channels...")
        
        # Create model with specific mode
        model = CustomDraemModel(severity_input_mode=mode)
        
        # Check if the model's _get_severity_input_channels matches expected
        actual_channels = model._get_severity_input_channels()
        assert actual_channels == expected_channels, f"Expected {expected_channels} channels for {mode}, got {actual_channels}"
        
        print(f"      ✅ {mode}: {actual_channels} channels (correct)")
    
    return True


def main():
    """Run all Custom DRAEM PyTorch model tests."""
    print("🚀 Testing Custom DRAEM PyTorch Model (DRAEM Backbone Integration)")
    print("=" * 70)
    
    try:
        # Test individual components
        test_fault_severity_subnetwork()
        
        # Test full model integration
        test_custom_draem_model()
        
        # Test channel compatibility
        test_channel_compatibility()
        
        # Analyze model
        test_model_parameters()
        
        print("\n" + "=" * 70)
        print("✅ All Custom DRAEM PyTorch model tests passed!")
        print("🎯 DRAEM backbone integration verified")
        print("🎯 3-channel RGB input support verified")
        print("🎯 Multi-modal severity prediction verified")
        print("🎯 Training/Inference modes verified")
        print("🎯 SSPCAB option support verified")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)