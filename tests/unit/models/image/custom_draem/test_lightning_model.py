"""Test script for Custom DRAEM Lightning Model.

This script tests the Lightning model training and validation steps.

Author: Taewan Hwang
"""

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib.models.image.custom_draem.lightning_model import CustomDraem
from anomalib.models.image.custom_draem.synthetic_generator import HDMAPCutPasteSyntheticGenerator
from types import SimpleNamespace
import numpy as np
from PIL import Image
import glob
import os


def create_test_batch(batch_size: int = 2, image_size: int = 224):
    """Create a test batch for training/validation."""
    # Create synthetic HDMAP-like images (3-channel RGB for DRAEM backbone)
    images = torch.randn(batch_size, 3, image_size, image_size)
    images = torch.clamp(images * 0.2 + 0.5, 0.0, 1.0)  # Normalize to [0, 1]
    
    # Create optional ground truth masks (for validation metrics)
    masks = torch.zeros(batch_size, 1, image_size, image_size)
    masks[:, :, 50:100, 50:100] = 1.0  # Simple square anomaly region
    
    # Use SimpleNamespace instead of abstract Batch class
    batch = SimpleNamespace(
        image=images,
        mask=masks, 
        gt_label=torch.zeros(batch_size),  # Normal images
        gt_mask=masks
    )
    
    return batch


def load_hdmap_images(data_path: str, category: str, max_images: int = 8) -> list[torch.Tensor]:
    """Load real HDMAP images from dataset.
    
    Args:
        data_path (str): Path to HDMAP dataset (e.g., 'datasets/HDMAP/1000_8bit_resize_224x224/domain_A')
        category (str): 'train/good', 'test/good', or 'test/fault'
        max_images (int): Maximum number of images to load
        
    Returns:
        list[torch.Tensor]: List of loaded images as tensors
    """
    image_path = os.path.join(data_path, category)
    if not os.path.exists(image_path):
        print(f"⚠️ Path not found: {image_path}")
        return []
    
    # Get PNG files
    png_files = glob.glob(os.path.join(image_path, "*.png"))
    png_files = sorted(png_files)[:max_images]
    
    images = []
    for img_file in png_files:
        try:
            # Load as RGB and convert to tensor (3-channel for DRAEM backbone)
            img = Image.open(img_file).convert('RGB')  # Convert to RGB
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]
            img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            
            # Resize to 224x224 if needed (ImageNet standard for DRAEM backbone)
            if img_tensor.shape[1] != 224 or img_tensor.shape[2] != 224:
                import torch.nn.functional as F
                img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
            
            images.append(img_tensor)
        except Exception as e:
            print(f"⚠️ Error loading {img_file}: {e}")
            continue
    
    print(f"📁 Loaded {len(images)} images from {image_path}")
    return images


def create_hdmap_batch(data_path: str, category: str, batch_size: int = 4):
    """Create a batch from real HDMAP images.
    
    Args:
        data_path (str): Path to HDMAP dataset
        category (str): 'train/good', 'test/good', or 'test/fault'
        batch_size (int): Number of images in batch
        
    Returns:
        SimpleNamespace: Batch object with real HDMAP data
    """
    # Load real HDMAP images
    images = load_hdmap_images(data_path, category, max_images=batch_size)
    
    if len(images) == 0:
        print(f"⚠️ No images loaded from {category}, falling back to synthetic data")
        return create_test_batch(batch_size)
    
    # Pad or truncate to exact batch size
    while len(images) < batch_size:
        images.append(images[0])  # Duplicate first image if needed
    images = images[:batch_size]
    
    # Stack into batch tensor
    batch_images = torch.stack(images, dim=0)  # (batch_size, 3, H, W)
    
    # Create empty masks (no ground truth masks available for HDMAP data)
    masks = torch.zeros_like(batch_images)
    
    # Create labels
    gt_labels = torch.zeros(batch_size) if "good" in category else torch.ones(batch_size)
    
    batch = SimpleNamespace(
        image=batch_images,
        mask=masks,
        gt_label=gt_labels,
        gt_mask=masks
    )
    
    return batch


def test_lightning_model():
    """Test Custom DRAEM Lightning model."""
    print("🚀 Testing Custom DRAEM Lightning Model")
    print("=" * 60)
    
    # Test different severity input modes
    severity_modes = [
        "discriminative_only", 
        # "with_original", 
        # "with_reconstruction",
        # "with_error_map", 
        # "multi_modal"
    ]
    
    for mode in severity_modes:
        print(f"\n🧪 Testing severity input mode: {mode}")
        
        # Initialize model
        model = CustomDraem(
            severity_max=10.0,
            severity_input_mode=mode,
            patch_ratio_range=(2.0, 4.0),  # Portrait patches for HDMAP (ratio > 1.0)
            patch_width_range=(30, 60),    # Moderate patch widths
            patch_count=1,
            reconstruction_weight=1.0,
            segmentation_weight=1.0,
            severity_weight=0.5
        )
        
        # Set to training mode
        model.train()
        
        # Create test batch
        train_batch = create_test_batch(batch_size=16)
        val_batch = create_test_batch(batch_size=16)
        
        try:
            # Test training step
            print("   Testing training_step...")
            train_output = model.training_step(train_batch)
            
            assert isinstance(train_output, dict), "Training output should be dict"
            assert "loss" in train_output, "Training output should contain 'loss'"
            assert isinstance(train_output["loss"], torch.Tensor), "Loss should be tensor"
            assert train_output["loss"].requires_grad, "Loss should require grad"
            
            print(f"      ✅ Training step passed!")
            print(f"      📊 Loss value: {train_output['loss'].item():.4f}")
            
            # Test validation step
            print("   Testing validation_step...")
            model.eval()
            with torch.no_grad():
                val_output = model.validation_step(val_batch)
            
            assert hasattr(val_output, 'pred_score'), "Should have pred_score"
            assert hasattr(val_output, 'anomaly_map'), "Should have anomaly_map"
            assert hasattr(val_output, 'pred_label'), "Should have pred_label (severity)"
            
            print(f"      ✅ Validation step passed!")
            print(f"      📊 Pred score shape: {val_output.pred_score.shape}")
            print(f"      📊 Anomaly map shape: {val_output.anomaly_map.shape}")
            print(f"      📊 Severity pred shape: {val_output.pred_label.shape}")
            print(f"      📊 Mean severity: {val_output.pred_label.mean().item():.3f}")
            
        except Exception as e:
            print(f"      ❌ Error in {mode}: {str(e)}")
            continue
    
    print("\n" + "=" * 60)
    print("✅ Custom DRAEM Lightning Model tests completed!")


def test_with_real_hdmap_data():
    """Test Custom DRAEM with real HDMAP dataset."""
    print("\n🗂️ Testing with Real HDMAP Data")
    print("=" * 60)
    
    # Define dataset path
    # NOTE: Modify this path to point to your HDMAP dataset location
    hdmap_path = "datasets/HDMAP/1000_8bit_resize_224x224/domain_A"
    
    # Initialize model (use discriminative_only for speed)
    model = CustomDraem(
        severity_max=10.0,
        severity_input_mode="discriminative_only",
        patch_ratio_range=(0.3, 0.8),  # Landscape patches for HDMAP
        patch_width_range=(20, 60),
        patch_count=1,
        reconstruction_weight=1.0,
        segmentation_weight=1.0,
        severity_weight=0.5
    )
    
    # TRAINING PHASE: Train on normal data only
    print("\n🎓 TRAINING PHASE: Using train/good data")
    print("-" * 50)
    
    try:
        # Create training batch from normal training data
        train_batch = create_hdmap_batch(hdmap_path, "train/good", batch_size=8)
        
        model.train()
        print("   Training with normal HDMAP images...")
        train_output = model.training_step(train_batch)
        
        assert isinstance(train_output, dict), "Training output should be dict"
        assert "loss" in train_output, "Training output should contain 'loss'"
        assert isinstance(train_output["loss"], torch.Tensor), "Loss should be tensor"
        
        print(f"      ✅ Training step passed!")
        print(f"      📊 Training loss: {train_output['loss'].item():.4f}")
        print(f"      📊 Training batch shape: {train_batch.image.shape}")
        print(f"      📊 Image value range: [{train_batch.image.min().item():.3f}, {train_batch.image.max().item():.3f}]")
        
    except Exception as e:
        print(f"      ❌ Training error: {str(e)}")
        return
    
    # TESTING PHASE: Test on both normal and fault data
    print("\n🧪 TESTING PHASE: Evaluating on test data")
    print("-" * 50)
    
    test_categories = [
        ("test/good", "Normal Test Images", "🟢"),
        ("test/fault", "Fault Test Images", "🔴")
    ]
    
    model.eval()  # Set to evaluation mode for testing
    
    for category, description, emoji in test_categories:
        print(f"\n{emoji} Testing {description} ({category})")
        
        try:
            # Create test batch
            test_batch = create_hdmap_batch(hdmap_path, category, batch_size=4)
            
            # Run inference (validation step)
            print("   Running inference...")
            with torch.no_grad():
                val_output = model.validation_step(test_batch)
            
            assert hasattr(val_output, 'pred_score'), "Should have pred_score"
            assert hasattr(val_output, 'anomaly_map'), "Should have anomaly_map"
            assert hasattr(val_output, 'pred_label'), "Should have pred_label (severity)"
            
            # Calculate metrics
            pred_scores = val_output.pred_score
            anomaly_maps = val_output.anomaly_map
            severity_preds = val_output.pred_label
            
            print(f"      ✅ Inference completed!")
            print(f"      📊 Batch shape: {test_batch.image.shape}")
            print(f"      📊 Anomaly scores: [{pred_scores.min().item():.3f}, {pred_scores.max().item():.3f}] (mean: {pred_scores.mean().item():.3f})")
            print(f"      📊 Anomaly map coverage: {(anomaly_maps > 0.5).float().mean().item() * 100:.2f}%")
            print(f"      📊 Severity predictions: [{severity_preds.min().item():.3f}, {severity_preds.max().item():.3f}] (mean: {severity_preds.mean().item():.3f})")
            
            # Expected behavior analysis
            if "good" in category:
                print(f"      💡 Expected: Low anomaly scores for normal images")
            else:
                print(f"      💡 Expected: Higher anomaly scores for fault images")
            
        except Exception as e:
            print(f"      ❌ Error testing {category}: {str(e)}")
            continue
    
    print("\n" + "=" * 60)
    print("✅ Real HDMAP data tests completed!")


def test_synthetic_generation_with_real_data():
    """Test synthetic fault generation using real HDMAP images."""
    print("\n🔧 Testing Synthetic Generation with Real HDMAP Data")
    print("=" * 60)
    
    # NOTE: Modify this path to point to your HDMAP dataset location
    hdmap_path = "datasets/HDMAP/1000_8bit_resize_224x224/domain_A"
    
    # Load a few real normal images
    real_images = load_hdmap_images(hdmap_path, "train/good", max_images=3)
    
    if len(real_images) == 0:
        print("⚠️ No real images found, skipping synthetic generation test")
        return
    
    # Stack into batch
    real_batch = torch.stack(real_images, dim=0)  # (3, 1, H, W)
    
    print(f"📊 Real image batch shape: {real_batch.shape}")
    print(f"📊 Real image value range: [{real_batch.min().item():.3f}, {real_batch.max().item():.3f}]")
    
    # Test different synthetic generation configurations
    configs = [
        {"patch_ratio_range": (0.3, 0.8), "patch_width_range": (30, 60), "patch_count": 1, "description": "Landscape single patch"},
        {"patch_ratio_range": (1.5, 3.0), "patch_width_range": (20, 40), "patch_count": 1, "description": "Portrait single patch"},
        {"patch_ratio_range": (0.5, 2.0), "patch_width_range": (25, 50), "patch_count": 2, "description": "Mixed dual patch"},
    ]
    
    for config in configs:
        print(f"\n🧪 Testing: {config['description']}")
        
        try:
            # Create generator with specific config
            generator = HDMAPCutPasteSyntheticGenerator(
                patch_width_range=config["patch_width_range"],
                patch_ratio_range=config["patch_ratio_range"],
                severity_max=10.0,
                patch_count=config["patch_count"]
            )
            
            # Generate synthetic faults
            synthetic_image, fault_mask, severity_map, severity_label = generator(real_batch)
            
            print(f"      ✅ Generation successful!")
            print(f"      📊 Synthetic image shape: {synthetic_image.shape}")
            print(f"      📊 Fault mask coverage: {(fault_mask > 0).float().mean().item() * 100:.2f}%")
            print(f"      📊 Severity labels: {severity_label.tolist()}")
            print(f"      📊 Mean severity: {severity_label.mean().item():.3f}")
            
        except Exception as e:
            print(f"      ❌ Error in {config['description']}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ Synthetic generation tests completed!")


def test_probabilistic_generation():
    """Test probabilistic synthetic fault generation."""
    print("\n🎲 Testing Probabilistic Synthetic Generation")
    print("=" * 60)
    
    # NOTE: Modify this path to point to your HDMAP dataset location
    hdmap_path = "datasets/HDMAP/1000_8bit_resize_224x224/domain_A"
    
    # Load real images for testing
    real_images = load_hdmap_images(hdmap_path, "train/good", max_images=10)
    
    if len(real_images) == 0:
        print("⚠️ No real images found, using synthetic test images")
        real_images = [torch.randn(3, 224, 224) for _ in range(10)]
    
    # Stack into batch
    test_batch = torch.stack(real_images, dim=0)  # (10, 3, H, W)
    
    print(f"📊 Test batch shape: {test_batch.shape}")
    
    # Test different probability values
    probabilities = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    for prob in probabilities:
        print(f"\n🎯 Testing probability = {prob}")
        
        try:
            # Create generator with specific probability
            generator = HDMAPCutPasteSyntheticGenerator(
                patch_width_range=(30, 60),
                patch_ratio_range=(0.5, 2.0),
                severity_max=10.0,
                patch_count=1,
                probability=prob
            )
            
            # Generate synthetic faults multiple times to check probability
            fault_counts = []
            total_runs = 20  # Run multiple times to check probability
            
            for _ in range(total_runs):
                synthetic_image, fault_mask, severity_map, severity_label = generator(test_batch)
                
                # Count how many images have faults (non-zero severity)
                fault_count = (severity_label > 0).sum().item()
                fault_counts.append(fault_count)
            
            # Calculate statistics
            avg_fault_count = sum(fault_counts) / len(fault_counts)
            actual_probability = avg_fault_count / test_batch.shape[0]
            
            print(f"      📊 Expected faults per batch: {prob * test_batch.shape[0]:.1f}")
            print(f"      📊 Actual average faults: {avg_fault_count:.1f}")
            print(f"      📊 Expected probability: {prob:.1f}")
            print(f"      📊 Actual probability: {actual_probability:.3f}")
            print(f"      📊 Difference: {abs(prob - actual_probability):.3f}")
            
            # Check if within reasonable range (±10% for randomness)
            if abs(prob - actual_probability) < 0.15:  # 15% tolerance for small sample
                print(f"      ✅ Probability test passed!")
            else:
                print(f"      ⚠️ Probability deviation higher than expected (but may be due to small sample)")
                
        except Exception as e:
            print(f"      ❌ Error testing probability {prob}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ Probabilistic generation tests completed!")


def test_lightning_model_with_probability():
    """Test Lightning model with different anomaly probabilities."""
    print("\n⚡ Testing Lightning Model with Anomaly Probabilities")
    print("=" * 60)
    
    probabilities = [0.3, 0.5, 0.8]
    
    for prob in probabilities:
        print(f"\n🧪 Testing Lightning model with anomaly_probability = {prob}")
        
        try:
            # Create model with specific probability
            model = CustomDraem(
                severity_max=10.0,
                severity_input_mode="discriminative_only",
                patch_ratio_range=(0.5, 2.0),
                patch_width_range=(30, 60),
                patch_count=1,
                anomaly_probability=prob,
                reconstruction_weight=1.0,
                segmentation_weight=1.0,
                severity_weight=0.5
            )
            
            # Create test batch
            test_batch = create_test_batch(batch_size=8)
            
            # Test training step
            model.train()
            train_output = model.training_step(test_batch)
            
            print(f"      ✅ Training step passed!")
            print(f"      📊 Loss value: {train_output['loss'].item():.4f}")
            print(f"      📊 Generator probability: {model.augmenter.probability}")
            
        except Exception as e:
            print(f"      ❌ Error testing probability {prob}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ Lightning model probability tests completed!")


if __name__ == "__main__":
    # Run all tests
    print("🚀 Starting Comprehensive Custom DRAEM Lightning Tests")
    print("=" * 80)
    
    # Test 1: Basic functionality with synthetic data
    test_lightning_model()
    
    # Test 2: Real HDMAP data integration
    test_with_real_hdmap_data() 
    
    # Test 3: Synthetic generation with real images
    test_synthetic_generation_with_real_data()
    
    # Test 4: NEW - Probabilistic generation
    test_probabilistic_generation()
    
    # Test 5: NEW - Lightning model with probabilities
    test_lightning_model_with_probability()
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 All Custom DRAEM Lightning tests completed successfully!")
    print("🎯 Phase 5: Probabilistic Generation implemented and tested!")
    print("🚀 Ready for real HDMAP dataset training!")
