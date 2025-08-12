#!/usr/bin/env python3
"""Test script for Custom DRAEM PyTorch model.

This script tests the 1-channel grayscale Custom DRAEM model components
to ensure they work correctly with HDMAP-style data.

Usage:
    python src/anomalib/models/image/custom_draem/test_torch_model.py
"""

import sys
from pathlib import Path

import torch

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


from anomalib.models.image.custom_draem.torch_model import (
    CustomDraemModel, 
    ReconstructiveSubNetwork, 
    DiscriminativeSubNetwork, 
    FaultSeveritySubNetwork
)


def test_reconstructive_subnetwork():
    """Test ReconstructiveSubNetwork with 1-channel input."""
    print("🧪 Testing ReconstructiveSubNetwork...")
    
    # Create 1-channel grayscale input (HDMAP style)
    batch_size, channels, height, width = 4, 1, 256, 256
    input_tensor = torch.randn(batch_size, channels, height, width)
    
    # Initialize network
    reconstructive_net = ReconstructiveSubNetwork(sspcab=False)
    reconstructive_net.eval()
    
    # Forward pass
    with torch.no_grad():
        reconstruction = reconstructive_net(input_tensor)
    
    print(f"   Input shape: {input_tensor.shape}")
    print(f"   Output shape: {reconstruction.shape}")
    print(f"   Output range: [{reconstruction.min():.3f}, {reconstruction.max():.3f}]")
    
    # Verify output properties
    assert reconstruction.shape == input_tensor.shape, f"Shape mismatch: {reconstruction.shape} != {input_tensor.shape}"
    assert 0 <= reconstruction.min() and reconstruction.max() <= 1, f"Output not in [0,1] range: [{reconstruction.min()}, {reconstruction.max()}]"
    
    print("   ✅ ReconstructiveSubNetwork test passed!")
    return True


def test_discriminative_subnetwork():
    """Test DiscriminativeSubNetwork with 2-channel input."""
    print("\n🧪 Testing DiscriminativeSubNetwork...")
    
    # Create 2-channel input (original + reconstruction concatenated)
    batch_size, channels, height, width = 4, 2, 256, 256
    input_tensor = torch.randn(batch_size, channels, height, width)
    
    # Initialize network
    discriminative_net = DiscriminativeSubNetwork(in_channels=2, out_channels=2)
    discriminative_net.eval()
    
    # Forward pass
    with torch.no_grad():
        prediction = discriminative_net(input_tensor)
    
    print(f"   Input shape: {input_tensor.shape}")
    print(f"   Output shape: {prediction.shape}")
    print(f"   Output range: [{prediction.min():.3f}, {prediction.max():.3f}]")
    
    # Verify output properties
    expected_shape = (batch_size, 2, height, width)
    assert prediction.shape == expected_shape, f"Shape mismatch: {prediction.shape} != {expected_shape}"
    
    print("   ✅ DiscriminativeSubNetwork test passed!")
    return True


def test_fault_severity_subnetwork():
    """Test FaultSeveritySubNetwork with different input modes."""
    print("\n🧪 Testing FaultSeveritySubNetwork...")
    
    batch_size, height, width = 4, 256, 256
    
    # Test different input modes
    test_modes = [
        ("discriminative_only", 2),
        ("with_original", 3),
        ("with_reconstruction", 3),
        ("with_error_map", 3),
        ("multi_modal", 5)
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
    """Test full CustomDraemModel integration."""
    print("\n🧪 Testing CustomDraemModel integration...")
    
    # Create 1-channel grayscale input (HDMAP style)
    batch_size, channels, height, width = 2, 1, 256, 256
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
        
        # Test training mode
        model.train()
        with torch.no_grad():
            reconstruction, prediction, severity = model(input_tensor)
        
        print(f"      Training mode:")
        print(f"         Reconstruction: {reconstruction.shape}")
        print(f"         Prediction: {prediction.shape}")
        print(f"         Severity: {severity.shape}")
        
        # Verify training outputs
        assert reconstruction.shape == input_tensor.shape
        assert prediction.shape == (batch_size, 2, height, width)
        assert severity.shape == (batch_size, 1)
        
        # Test inference mode
        model.eval()
        with torch.no_grad():
            inference_output = model(input_tensor)
        
        print(f"      Inference mode:")
        print(f"         Pred score: {inference_output.pred_score.shape}")
        print(f"         Anomaly map: {inference_output.anomaly_map.shape}")
        print(f"         Severity (pred_label): {inference_output.pred_label.shape}")
        
        # Verify inference outputs
        assert inference_output.pred_score.shape == (batch_size,)
        assert inference_output.anomaly_map.shape == (batch_size, height, width)
        assert inference_output.pred_label.shape == (batch_size,)  # severity scores
        
        print(f"      ✅ {mode} integration test passed!")
    
    return True


def test_model_parameters():
    """Test model parameter counts and memory usage."""
    print("\n📊 Model parameter analysis...")
    
    # Create model
    model = CustomDraemModel(severity_input_mode="discriminative_only")
    
    # Count parameters for each subnetwork
    reconstructive_params = sum(p.numel() for p in model.reconstructive_subnetwork.parameters())
    discriminative_params = sum(p.numel() for p in model.discriminative_subnetwork.parameters())
    severity_params = sum(p.numel() for p in model.fault_severity_subnetwork.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"   Reconstructive SubNetwork: {reconstructive_params:,} parameters")
    print(f"   Discriminative SubNetwork: {discriminative_params:,} parameters")
    print(f"   Severity SubNetwork: {severity_params:,} parameters")
    print(f"   Total parameters: {total_params:,} parameters")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")
    
    return True


def main():
    """Run all Custom DRAEM PyTorch model tests."""
    print("🚀 Testing Custom DRAEM PyTorch Model")
    print("=" * 60)
    
    try:
        # Test individual components
        test_reconstructive_subnetwork()
        test_discriminative_subnetwork()
        test_fault_severity_subnetwork()
        
        # Test full model integration
        test_custom_draem_model()
        
        # Analyze model
        test_model_parameters()
        
        print("\n" + "=" * 60)
        print("✅ All Custom DRAEM PyTorch model tests passed!")
        print("🎯 1-channel grayscale support verified")
        print("🎯 Multi-modal severity prediction verified")
        print("🎯 Training/Inference modes verified")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
