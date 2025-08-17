"""Test script for HDMAPCutPasteSyntheticGenerator.

This script tests the synthetic fault generation functionality.

Run with: pytest tests/unit/models/image/draem_sevnet/test_synthetic_generator.py -v -s
Author: Taewan Hwang
"""

import os
import shutil
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*'mode' parameter is deprecated.*")

from anomalib.models.image.draem_sevnet.synthetic_generator import HDMAPCutPasteSyntheticGenerator


def load_sample_hdmap_image(image_path: str = None) -> torch.Tensor:
    """Load a sample HDMAP image for testing.
    
    Args:
        image_path (str, optional): Path to HDMAP image. If None, creates synthetic test image.
        
    Returns:
        torch.Tensor: Normalized RGB image tensor [1, 3, H, W]
    """
    if image_path and Path(image_path).exists():
        # Load real HDMAP image
        image = Image.open(image_path).convert('RGB')  # Convert to RGB
        image = image.resize((224, 224))  # Resize to ImageNet standard size
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        image_array = image_array.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    else:
        # Create synthetic test image with some patterns
        print("Creating synthetic test image (no real HDMAP path provided)")
        image_array = np.zeros((3, 224, 224))  # 3-channel RGB
        
        # Add some structured patterns (simulating HDMAP features)
        for i in range(0, 224, 28):  # Adjusted for 224x224
            image_array[:, i:i+2, :] = 0.3  # Horizontal lines
            image_array[:, :, i:i+2] = 0.3  # Vertical lines
        
        # Add some random regions with different intensities
        image_array[:, 40:80, 40:80] = 0.6  # Bright region
        image_array[:, 120:160, 120:160] = 0.2  # Dark region
        
        # Add some noise to each channel
        for c in range(3):
            noise = np.random.randn(224, 224) * 0.05
            image_array[c] = np.clip(image_array[c] + noise, 0.0, 1.0)
    
    # Convert to PyTorch tensor [1, 3, H, W]
    image_tensor = torch.from_numpy(image_array).float().unsqueeze(0)
    return image_tensor


def test_basic_generation():
    """Test basic synthetic fault generation functionality."""
    print("🧪 Testing basic synthetic fault generation...")
    
    # Load test image
    image = load_sample_hdmap_image()
    print(f"   Input image shape: {image.shape}")
    
    # Create generator with default settings
    generator = HDMAPCutPasteSyntheticGenerator(
        patch_width_range=(40, 80),        # Width 40-80 pixels
        patch_ratio_range=(0.5, 2.0),      # height/width ratio (landscape to portrait)
        severity_max=10.0,
        patch_count=1,
        probability=1.0  # Always generate faults for testing
    )
    
    # Generate synthetic fault with detailed patch information
    synthetic_image, fault_mask, severity_map, severity_label, patch_info = generator(image, return_patch_info=True)
    
    print(f"   Synthetic image shape: {synthetic_image.shape}")
    print(f"   Fault mask shape: {fault_mask.shape}")
    print(f"   Severity map shape: {severity_map.shape}")
    print(f"   Severity label: {severity_label.item():.3f}")
    print(f"   Fault mask coverage: {fault_mask.sum().item() / fault_mask.numel() * 100:.2f}%")
    
    # Print detailed patch information
    print(f"   📐 Patch Details:")
    print(f"      Width: {patch_info['patch_width']} pixels")
    print(f"      Height: {patch_info['patch_height']} pixels") 
    print(f"      Ratio (H/W): {patch_info['patch_ratio']:.3f} ({patch_info['patch_type']})")
    print(f"      Size: {patch_info['patch_size']} pixels")
    print(f"      Severity: {patch_info['severity_value']:.3f} / {generator.severity_max}")
    print(f"      Patch Count: {patch_info['patch_count']}")
    print(f"      Coverage: {patch_info['coverage_percentage']:.2f}%")
    
    # Print patch positions
    for i, (src_x, src_y, tgt_x, tgt_y) in enumerate(patch_info['patch_positions']):
        print(f"      Patch {i+1}: Cut({src_x},{src_y}) → Paste({tgt_x},{tgt_y})")
    
    # 기본 검증
    assert isinstance(image, torch.Tensor), "Input image should be tensor"
    assert isinstance(synthetic_image, torch.Tensor), "Synthetic image should be tensor"
    assert isinstance(fault_mask, torch.Tensor), "Fault mask should be tensor"
    assert isinstance(severity_map, torch.Tensor), "Severity map should be tensor"
    assert isinstance(severity_label, torch.Tensor), "Severity label should be tensor"


def test_different_patch_configurations():
    """Test different patch ratio configurations."""
    print("\n🧪 Testing different patch configurations...")
    
    image = load_sample_hdmap_image()
    
    configurations = [
        ("Landscape", (0.1, 0.3)),    # height/width < 1.0 (wider than tall)
        ("Portrait", (1.5, 3.0)),     # height/width > 1.0 (taller than wide)
        ("Square", (1.0, 1.0)),       # height/width = 1.0 (equal dimensions)
        ("Mixed", (0.5, 2.0))         # height/width mixed range
    ]
    
    results = []
    
    for config_name, patch_ratio_range in configurations:
        generator = HDMAPCutPasteSyntheticGenerator(
            patch_width_range=(32, 64),
            patch_ratio_range=patch_ratio_range,
            severity_max=10.0,
            patch_count=1,
            probability=1.0  # Always generate faults for testing
        )
        
        synthetic_image, fault_mask, severity_map, severity_label, patch_info = generator(image, return_patch_info=True)
        results.append((config_name, synthetic_image, fault_mask, severity_label))
        
        print(f"   {config_name}: severity={severity_label.item():.3f}, "
              f"coverage={fault_mask.sum().item() / fault_mask.numel() * 100:.2f}%, "
              f"ratio={patch_info['patch_ratio']:.2f}, size={patch_info['patch_size']}")
    
    # 결과 검증
    assert len(results) == len(configurations), "Should have results for all configurations"
    for config_name, _, _, _ in results:
        assert config_name in [c[0] for c in configurations], f"Unexpected config name: {config_name}"


def test_multi_patch_generation():
    """Test multi-patch generation."""
    print("\n🧪 Testing multi-patch generation...")
    
    image = load_sample_hdmap_image()
    
    for patch_count in [1, 2, 3]:
        generator = HDMAPCutPasteSyntheticGenerator(
            patch_width_range=(30, 60),
            patch_ratio_range=(0.5, 2.0),
            severity_max=10.0,
            patch_count=patch_count,
            probability=1.0  # Always generate faults for testing
        )
        
        synthetic_image, fault_mask, severity_map, severity_label, patch_info = generator(image, return_patch_info=True)
        
        print(f"   {patch_count} patches: severity={severity_label.item():.3f}, "
              f"coverage={fault_mask.sum().item() / fault_mask.numel() * 100:.2f}%, "
              f"size={patch_info['patch_size']}, positions={len(patch_info['patch_positions'])}")


def test_severity_levels():
    """Test different severity levels."""
    print("\n🧪 Testing severity levels...")
    
    image = load_sample_hdmap_image()
    
    severity_ranges = [
        ("Low", 3.0),
        ("Medium", 6.0),
        ("High", 10.0)
    ]
    
    results = []
    
    for severity_name, severity_max in severity_ranges:
        generator = HDMAPCutPasteSyntheticGenerator(
            patch_width_range=(40, 80),
            patch_ratio_range=(0.5, 2.0),
            severity_max=severity_max,
            patch_count=1,
            probability=1.0  # Always generate faults for testing
        )
        
        # Generate multiple samples to see severity variation
        severities = []
        for _ in range(5):
            _, _, _, severity_label = generator(image)
            severities.append(severity_label.item())
        
        avg_severity = np.mean(severities)
        print(f"   {severity_name} (max={severity_max}): avg_severity={avg_severity:.3f}, "
              f"range=[{min(severities):.3f}, {max(severities):.3f}]")
        
        # Store one sample for visualization with patch info
        synthetic_image, fault_mask, severity_map, severity_label, patch_info = generator(image, return_patch_info=True)
        print(f"      Sample: ratio={patch_info['patch_ratio']:.2f}, size={patch_info['patch_size']}")
        results.append((severity_name, synthetic_image, fault_mask, severity_label))
    
    # 결과 검증
    assert len(results) == len(severity_ranges), "Should have results for all severity ranges"
    for severity_name, _, _, _ in results:
        assert severity_name in [s[0] for s in severity_ranges], f"Unexpected severity name: {severity_name}"


def save_test_results(
    original_image: torch.Tensor,
    synthetic_image: torch.Tensor,
    fault_mask: torch.Tensor,
    severity_map: torch.Tensor,
    severity_label: torch.Tensor,
    output_dir: str,
    test_name: str = "test"
):
    """Save test results as individual PNG files for manual inspection.
    
    Args:
        original_image: Original input image
        synthetic_image: Generated synthetic image
        fault_mask: Binary fault mask
        severity_map: Pixel-wise severity map
        severity_label: Image-level severity value
        output_dir: Directory to save results
        test_name: Name prefix for saved files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensors to numpy arrays [0, 1] range
    original = original_image[0, 0].cpu().numpy()
    synthetic = synthetic_image[0, 0].cpu().numpy()
    mask = fault_mask[0, 0].cpu().numpy()
    severity = severity_map[0, 0].cpu().numpy()
    
    # Convert to [0, 255] uint8 for PNG saving (mode='L' removed for Pillow compatibility)
    original_img = Image.fromarray((original * 255).astype(np.uint8))
    synthetic_img = Image.fromarray((synthetic * 255).astype(np.uint8))
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    severity_img = Image.fromarray((severity * 255).astype(np.uint8))
    
    # Calculate difference map
    diff = np.abs(synthetic - original)
    diff_img = Image.fromarray((diff * 255).astype(np.uint8))
    
    # Create overlay (original with highlighted faults)
    overlay = original.copy()
    overlay[mask > 0.5] = 1.0
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
    
    # Save individual images
    severity_str = f"{severity_label.item():.3f}"
    original_img.save(f"{output_dir}/{test_name}_01_original.png")
    synthetic_img.save(f"{output_dir}/{test_name}_02_synthetic_sev{severity_str}.png")
    diff_img.save(f"{output_dir}/{test_name}_03_difference.png")
    mask_img.save(f"{output_dir}/{test_name}_04_fault_mask.png")
    severity_img.save(f"{output_dir}/{test_name}_05_severity_map.png")
    overlay_img.save(f"{output_dir}/{test_name}_06_overlay.png")
    
    print(f"   💾 Results saved to: {output_dir}/{test_name}_*.png")


def visualize_results(
    original_image: torch.Tensor,
    synthetic_image: torch.Tensor,
    fault_mask: torch.Tensor,
    severity_map: torch.Tensor,
    severity_label: torch.Tensor,
    title: str = "Synthetic Fault Generation"
):
    """Visualize the synthetic fault generation results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{title} (Severity: {severity_label.item():.3f})', fontsize=16)
    
    # Convert tensors to numpy for visualization
    # For 3-channel RGB images, convert to grayscale for visualization
    if original_image.shape[1] == 3:
        # Convert RGB to grayscale using luminance formula
        original = (0.299 * original_image[0, 0] + 0.587 * original_image[0, 1] + 0.114 * original_image[0, 2]).cpu().numpy()
        synthetic = (0.299 * synthetic_image[0, 0] + 0.587 * synthetic_image[0, 1] + 0.114 * synthetic_image[0, 2]).cpu().numpy()
    else:
        # Original 1-channel case
        original = original_image[0, 0].cpu().numpy()
        synthetic = synthetic_image[0, 0].cpu().numpy()
    
    mask = fault_mask[0, 0].cpu().numpy()
    severity = severity_map[0, 0].cpu().numpy()
    
    # Original image
    axes[0, 0].imshow(original, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Synthetic image
    axes[0, 1].imshow(synthetic, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Synthetic Image')
    axes[0, 1].axis('off')
    
    # Difference
    diff = np.abs(synthetic - original)
    im1 = axes[0, 2].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
    axes[0, 2].set_title('Difference Map')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2])
    
    # Fault mask
    axes[1, 0].imshow(mask, cmap='Reds', vmin=0, vmax=1)
    axes[1, 0].set_title('Fault Mask')
    axes[1, 0].axis('off')
    
    # Severity map
    im2 = axes[1, 1].imshow(severity, cmap='viridis', vmin=0, vmax=1)
    axes[1, 1].set_title('Severity Map')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1])
    
    # Overlay
    overlay = original.copy()
    overlay[mask > 0.5] = 1.0  # Highlight fault regions
    axes[1, 2].imshow(overlay, cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title('Fault Overlay')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig


def test_with_real_hdmap_data():
    """Test with real HDMAP data if available."""
    print("\n🧪 Testing with real HDMAP data...")
    
    # Try to find real HDMAP images
    possible_paths = [
        "./datasets/HDMAP/1000_8bit_resize_224x224/domain_A/train/good/000065.png",
        "./datasets/HDMAP/1000_8bit_resize_224x224/domain_A/train/good/000096.png",
        "./datasets/HDMAP/1000_8bit_resize_224x224/domain_B/train/good/000067.png",
        "./datasets/HDMAP/1000_8bit_resize_224x224/domain_B/train/good/000027.png",
        "./datasets/HDMAP/1000_8bit_resize_224x224/domain_C/train/good/000081.png",
        "./datasets/HDMAP/1000_8bit_resize_224x224/domain_C/train/good/000047.png",
        "./datasets/HDMAP/1000_8bit_resize_224x224/domain_D/train/good/000071.png",
        "./datasets/HDMAP/1000_8bit_resize_224x224/domain_D/train/good/000055.png",

    ]
    
    # Find all existing images
    existing_paths = []
    for path in possible_paths:
        if Path(path).exists():
            existing_paths.append(path)
    
    if existing_paths:
        print(f"   Found {len(existing_paths)} real HDMAP images")
        
        for i, real_image_path in enumerate(existing_paths):
            print(f"   📷 Processing image {i+1}/{len(existing_paths)}: {Path(real_image_path).name}")
            image = load_sample_hdmap_image(real_image_path)
            
            generator = HDMAPCutPasteSyntheticGenerator(
                patch_width_range=(64, 128),
                patch_ratio_range=(0.1, 0.2),
                severity_max=2.0,
                patch_count=1,
                probability=1.0  # Always generate faults for testing
            )
            
            synthetic_image, fault_mask, severity_map, severity_label, patch_info = generator(image, return_patch_info=True)
            
            print(f"      Severity: {severity_label.item():.3f}, "
                  f"Coverage: {fault_mask.sum().item() / fault_mask.numel() * 100:.2f}%, "
                  f"Ratio: {patch_info['patch_ratio']:.2f}, "
                  f"Size: {patch_info['patch_size']}, "
                  f"Type: {patch_info['patch_type']}")
            
            # 각 이미지 처리 완료 확인
            assert isinstance(severity_label, torch.Tensor), "Severity label should be tensor"
            assert isinstance(fault_mask, torch.Tensor), "Fault mask should be tensor"
        
        print(f"   ✅ Successfully processed {len(existing_paths)} real HDMAP images")
    else:
        print("   No real HDMAP data found, skipping real data test")


def main():
    """Run all tests for HDMAPCutPasteSyntheticGenerator."""
    print("🚀 Testing HDMAPCutPasteSyntheticGenerator")
    print("=" * 60)
    
    # Create output directory for test results
    output_dir = "tests/unit/models/image/draem_sevnet/test_results"
    
    # Remove existing output directory if it exists
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
        print(f"🗑️ Removed existing directory: {output_dir}/")
    
    print(f"📁 Test results will be saved to: {output_dir}/")
    
    try:
        # Test 1: Basic generation
        print("\n📊 Running basic generation test...")
        test_basic_generation()
        
        # Create sample data for further tests
        image = load_sample_hdmap_image()
        generator = HDMAPCutPasteSyntheticGenerator(
            patch_width_range=(40, 80),
            patch_ratio_range=(0.5, 2.0),
            severity_max=10.0,
            patch_count=1,
            probability=1.0  # Always generate faults for testing
        )
        # generator() returns 4 values: synthetic_image, fault_mask, severity_map, severity_label
        synthetic, mask, severity_map, severity_label = generator(image)
        original = image  # Use original input image
        
        # Save basic test results
        save_test_results(original, synthetic, mask, severity_map, severity_label, 
                         output_dir, "basic_generation")
        
        # Test 2: Different patch configurations
        print("\n📊 Running patch configuration tests...")
        test_different_patch_configurations()
        
        # Test 3: Multi-patch generation
        print("\n📊 Running multi-patch tests...")
        image = load_sample_hdmap_image()
        
        for patch_count in [1, 2, 3]:
            generator = HDMAPCutPasteSyntheticGenerator(
                patch_width_range=(30, 60),
                patch_ratio_range=(0.5, 2.0),
                severity_max=10.0,
                patch_count=patch_count,
                probability=1.0  # Always generate faults for testing
            )
            
            synthetic_image, fault_mask, severity_map, severity_label = generator(image)
            
            print(f"   {patch_count} patches: severity={severity_label.item():.3f}, "
                  f"coverage={fault_mask.sum().item() / fault_mask.numel() * 100:.2f}%")
            
            # Save multi-patch results
            save_test_results(image, synthetic_image, fault_mask, severity_map, severity_label,
                             output_dir, f"multipatch_{patch_count}")
        
        # Test 4: Severity levels
        print("\n📊 Running severity level tests...")
        severity_ranges = [("low", 3.0), ("medium", 6.0), ("high", 10.0)]
        
        for severity_name, severity_max in severity_ranges:
            generator = HDMAPCutPasteSyntheticGenerator(
                patch_width_range=(40, 80),
                patch_ratio_range=(0.5, 2.0),
                severity_max=severity_max,
                patch_count=1,
                probability=1.0  # Always generate faults for testing
            )
            
            synthetic_image, fault_mask, severity_map, severity_label = generator(image)
            
            print(f"   {severity_name} severity (max={severity_max}): "
                  f"actual={severity_label.item():.3f}")
            
            # Save severity test results
            save_test_results(image, synthetic_image, fault_mask, severity_map, severity_label,
                             output_dir, f"severity_{severity_name}")
        
        # Test 5: Real HDMAP data (if available)
        print("\n📊 Running real HDMAP data tests...")
        possible_paths = [
            "./datasets/HDMAP/1000_8bit_resize_224x224/domain_A/train/good/000065.png",
            "./datasets/HDMAP/1000_8bit_resize_224x224/domain_A/train/good/000096.png",
            "./datasets/HDMAP/1000_8bit_resize_224x224/domain_B/train/good/000067.png",
            "./datasets/HDMAP/1000_8bit_resize_224x224/domain_B/train/good/000027.png",
            "./datasets/HDMAP/1000_8bit_resize_224x224/domain_C/train/good/000081.png",
            "./datasets/HDMAP/1000_8bit_resize_224x224/domain_C/train/good/000047.png",
            "./datasets/HDMAP/1000_8bit_resize_224x224/domain_D/train/good/000071.png",
            "./datasets/HDMAP/1000_8bit_resize_224x224/domain_D/train/good/000055.png",
        ]
        
        # Find existing images and save results
        existing_paths = [path for path in possible_paths if Path(path).exists()]
        
        if existing_paths:
            print(f"   Found {len(existing_paths)} real HDMAP images")
            
            for i, real_image_path in enumerate(existing_paths):  # Process all available HDMAP images
                print(f"   📷 Processing image {i+1}: {Path(real_image_path).name}")
                real_image = load_sample_hdmap_image(real_image_path)
                
                generator = HDMAPCutPasteSyntheticGenerator(
                    patch_width_range=(64, 128),
                    patch_ratio_range=(0.1, 0.2),
                    severity_max=0.5,
                    patch_count=1,
                    probability=1.0  # Always generate faults for testing
                )
                
                real_synthetic, real_mask, real_sev_map, real_sev_label = generator(real_image)
                
                # Create identifier from path
                path_parts = Path(real_image_path).parts
                identifier = f"{path_parts[-5]}_{path_parts[-4]}_{path_parts[-2]}"  # resize_domain_X_good
                identifier = identifier.replace("1000_8bit_", "").replace("domain_", "")
                
                print(f"      Severity: {real_sev_label.item():.3f}, "
                      f"Coverage: {real_mask.sum().item() / real_mask.numel() * 100:.2f}%")
                
                # Save real HDMAP results
                save_test_results(real_image, real_synthetic, real_mask, real_sev_map, real_sev_label,
                                 output_dir, f"real_hdmap_{identifier}_{Path(real_image_path).stem}")
        else:
            print("   No real HDMAP data found, skipping real data test")
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print(f"📁 Check {output_dir}/ folder for PNG results")
        print(f"📄 File naming: [test_name]_[01-06]_[description].png")
        print("   01: Original, 02: Synthetic, 03: Difference")
        print("   04: Fault Mask, 05: Severity Map, 06: Overlay")
        
        # 테스트 완료 검증
        assert Path(output_dir).exists(), "Output directory should exist"
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise  # re-raise to fail the test


if __name__ == "__main__":
    results = main()
    
    # Show plots if running directly
    try:
        plt.show()
    except:
        print("Note: matplotlib display not available in this environment")


# pytest로 실행 시 자동으로 실행되는 테스트 함수
def test_synthetic_generator_functionality():
    """Synthetic generator 기능 통합 테스트"""
    print("\n🧪 Synthetic Generator Test Suite")
    print("=" * 60)
    print("Testing HDMAPCutPasteSyntheticGenerator functionality...")
    
    # 메인 테스트 실행
    main()
    
    print("\n✅ All synthetic generator tests passed!")


if __name__ == "__main__":
    print("\n🧪 Synthetic Generator Test Suite")
    print("=" * 60)
    print("To run as pytest:")
    print("pytest tests/unit/models/image/draem_sevnet/test_synthetic_generator.py -v -s")
    print("\nRunning direct execution...")
    main()
