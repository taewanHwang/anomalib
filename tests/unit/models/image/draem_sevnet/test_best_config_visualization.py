"""Test script for visualizing synthetic faults from best performing configuration.

This script generates synthetic faults using the best performing configuration:
domainA_draem_sevnet_arch_04 with AUROC 0.898878

Configuration:
- model_type: draem_sevnet
- source_domain: domain_A
- severity_head_mode: single_scale
- severity_head_hidden_dim: 512
- severity_head_pooling_type: gap
- patch_ratio_range: [0.3, 0.7]
- patch_width_range: [20, 60]
- patch_count: 1
- anomaly_probability: 0.5
- severity_weight: 0.5
- severity_loss_type: mse
- severity_max: 1.0

Run with: pytest tests/unit/models/image/draem_sevnet/test_best_config_visualization.py -v -s
Author: Taewan Hwang
"""

import os
import random
import shutil
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*'mode' parameter is deprecated.*")

# Add hdmap experiment utils to path
sys.path.append('/mnt/ex-disk/taewan.hwang/study/anomalib/examples/hdmap')
from experiment_utils import create_single_domain_datamodule

from anomalib.models.image.draem_sevnet.synthetic_generator import HDMAPCutPasteSyntheticGenerator


def load_hdmap_samples_using_datamodule(domain: str = "domain_A", num_samples: int = 10, seed: int = 42) -> List[Tuple[str, torch.Tensor]]:
    """Load random samples from HDMAP using existing datamodule infrastructure.
    
    Args:
        domain: HDMAP domain to load from
        num_samples: Number of samples to load
        seed: Random seed for reproducible sampling
        
    Returns:
        List of (filename, image_tensor) tuples
    """
    print(f"\n📷 Loading HDMAP {domain} samples using datamodule...")
    
    # Use existing common function to create datamodule
    datamodule = create_single_domain_datamodule(
        domain=domain,
        batch_size=1,  # Load one at a time
        image_size="224x224",
        val_split_ratio=0.1,
        num_workers=1,
        seed=seed
    )
    
    # Setup datamodule
    datamodule.setup()
    
    print(f"   Dataset stats:")
    print(f"   - Train samples: {len(datamodule.train_data)}")
    print(f"   - Val samples: {len(datamodule.val_data) if datamodule.val_data else 0}")
    print(f"   - Test samples: {len(datamodule.test_data)}")
    
    # Get train dataset for good samples
    train_dataset = datamodule.train_data
    
    # Set seed for reproducible sampling
    random.seed(seed)
    num_samples = min(num_samples, len(train_dataset))
    selected_indices = random.sample(range(len(train_dataset)), num_samples)
    
    print(f"   Selected {num_samples} samples from {len(train_dataset)} train images")
    
    samples = []
    for i, idx in enumerate(selected_indices):
        sample = train_dataset[idx]
        
        # ImageItem object has .image attribute, not dictionary access
        image_tensor = sample.image.unsqueeze(0)  # Add batch dimension
        
        # Try to get original filename if available, otherwise use index
        if hasattr(sample, 'image_path') and sample.image_path:
            filename = Path(sample.image_path).name
        else:
            filename = f"train_sample_{idx:06d}.png"
        
        samples.append((filename, image_tensor))
        print(f"   Loaded sample {i+1}/{num_samples}: {filename} -> {image_tensor.shape}")
    
    return samples


def create_best_config_generator() -> HDMAPCutPasteSyntheticGenerator:
    """Create synthetic generator with best performing configuration.
    
    Returns:
        Configured HDMAPCutPasteSyntheticGenerator
    """
    generator = HDMAPCutPasteSyntheticGenerator(
        patch_width_range=(20, 60),      # From best config
        patch_ratio_range=(0.3, 0.7),   # From best config  
        severity_max=1.0,                # From best config
        patch_count=1,                   # From best config
        probability=1.0                  # Always generate for visualization
    )
    
    print("Created generator with best performing configuration:")
    print(f"  - patch_width_range: (20, 60)")
    print(f"  - patch_ratio_range: (0.3, 0.7)")
    print(f"  - severity_max: 1.0")
    print(f"  - patch_count: 1")
    print(f"  - probability: 1.0")
    
    return generator


def visualize_synthetic_fault(
    original_image: torch.Tensor,
    synthetic_image: torch.Tensor,
    fault_mask: torch.Tensor,
    severity_map: torch.Tensor,
    severity_label: torch.Tensor,
    patch_info: dict,
    filename: str
) -> plt.Figure:
    """Visualize synthetic fault generation results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{filename} - Severity: {severity_label.item():.3f}', fontsize=16)
    
    # Convert RGB tensors to grayscale for visualization
    if original_image.shape[1] == 3:
        # Convert RGB to grayscale using luminance formula
        original = (0.299 * original_image[0, 0] + 0.587 * original_image[0, 1] + 0.114 * original_image[0, 2]).cpu().numpy()
        synthetic = (0.299 * synthetic_image[0, 0] + 0.587 * synthetic_image[0, 1] + 0.114 * synthetic_image[0, 2]).cpu().numpy()
    else:
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
    
    # Difference map
    diff = np.abs(synthetic - original)
    im1 = axes[0, 2].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
    axes[0, 2].set_title('Difference Map')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Fault mask
    axes[1, 0].imshow(mask, cmap='Reds', vmin=0, vmax=1)
    axes[1, 0].set_title('Fault Mask')
    axes[1, 0].axis('off')
    
    # Severity map
    im2 = axes[1, 1].imshow(severity, cmap='viridis', vmin=0, vmax=1)
    axes[1, 1].set_title('Severity Map')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Overlay (original with highlighted faults)
    overlay = original.copy()
    overlay[mask > 0.5] = 1.0
    axes[1, 2].imshow(overlay, cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title('Fault Overlay')
    axes[1, 2].axis('off')
    
    # Add patch information as text
    info_text = f"Patch Info:\n"
    info_text += f"Width: {patch_info['patch_width']}px\n"
    info_text += f"Height: {patch_info['patch_height']}px\n"
    info_text += f"Ratio: {patch_info['patch_ratio']:.3f}\n"
    info_text += f"Type: {patch_info['patch_type']}\n"
    info_text += f"Size: {patch_info['patch_size']}px\n"
    info_text += f"Coverage: {patch_info['coverage_percentage']:.2f}%"
    
    fig.text(0.02, 0.02, info_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    return fig


def save_individual_images(
    original_image: torch.Tensor,
    synthetic_image: torch.Tensor,
    fault_mask: torch.Tensor,
    severity_map: torch.Tensor,
    severity_label: torch.Tensor,
    output_dir: str,
    filename: str
):
    """Save individual result images as PNG files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert RGB to grayscale for saving
    if original_image.shape[1] == 3:
        original = (0.299 * original_image[0, 0] + 0.587 * original_image[0, 1] + 0.114 * original_image[0, 2]).cpu().numpy()
        synthetic = (0.299 * synthetic_image[0, 0] + 0.587 * synthetic_image[0, 1] + 0.114 * synthetic_image[0, 2]).cpu().numpy()
    else:
        original = original_image[0, 0].cpu().numpy()
        synthetic = synthetic_image[0, 0].cpu().numpy()
    
    mask = fault_mask[0, 0].cpu().numpy()
    severity = severity_map[0, 0].cpu().numpy()
    
    # Convert to [0, 255] uint8
    original_img = Image.fromarray((original * 255).astype(np.uint8))
    synthetic_img = Image.fromarray((synthetic * 255).astype(np.uint8))
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    severity_img = Image.fromarray((severity * 255).astype(np.uint8))
    
    # Create difference and overlay
    diff = np.abs(synthetic - original)
    diff_img = Image.fromarray((diff * 255).astype(np.uint8))
    
    overlay = original.copy()
    overlay[mask > 0.5] = 1.0
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
    
    # Save with severity in filename
    base_name = Path(filename).stem
    severity_str = f"{severity_label.item():.3f}".replace(".", "")
    
    original_img.save(f"{output_dir}/{base_name}_sev{severity_str}_01_original.png")
    synthetic_img.save(f"{output_dir}/{base_name}_sev{severity_str}_02_synthetic.png")
    diff_img.save(f"{output_dir}/{base_name}_sev{severity_str}_03_difference.png")
    mask_img.save(f"{output_dir}/{base_name}_sev{severity_str}_04_mask.png")
    severity_img.save(f"{output_dir}/{base_name}_sev{severity_str}_05_severity.png")
    overlay_img.save(f"{output_dir}/{base_name}_sev{severity_str}_06_overlay.png")


def test_best_config_visualization():
    """Main test function for visualizing best configuration synthetic faults."""
    
    print("🚀 Testing Best Configuration Synthetic Fault Visualization")
    print("=" * 70)
    print("Configuration: domainA_draem_sevnet_arch_04 (AUROC: 0.898878)")
    print("Dataset: HDMAP domain_A good samples")
    print()
    
    # Output directory (relative to current test file)
    output_dir = Path(__file__).parent / "test_results" / "best_config_viz"
    
    # Clean output directory
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
        print(f"🗑️ Cleaned existing output directory")
    
    print(f"📁 Output directory: {output_dir}")
    print()
    
    try:
        # Load 10 random samples from domain A using common datamodule
        samples = load_hdmap_samples_using_datamodule(domain="domain_A", num_samples=10, seed=42)
        print(f"✅ Loaded {len(samples)} samples")
        print()
        
        # Create generator with best configuration
        print("⚙️ Creating synthetic fault generator...")
        generator = create_best_config_generator()
        print("✅ Generator created")
        print()
        
        # Process each sample
        print("🎨 Generating synthetic faults...")
        results = []
        
        for i, (filename, image) in enumerate(samples):
            print(f"  Processing {i+1}/{len(samples)}: {filename}")
            
            # Generate synthetic fault with detailed info
            synthetic_image, fault_mask, severity_map, severity_label, patch_info = generator(
                image, return_patch_info=True
            )
            
            print(f"    Severity: {severity_label.item():.3f}")
            print(f"    Coverage: {fault_mask.sum().item() / fault_mask.numel() * 100:.2f}%")
            print(f"    Patch size: {patch_info['patch_width']}×{patch_info['patch_height']} ({patch_info['patch_type']})")
            print(f"    Patch ratio: {patch_info['patch_ratio']:.3f}")
            
            # Save individual images
            save_individual_images(
                image, synthetic_image, fault_mask, severity_map, severity_label,
                output_dir, filename
            )
            
            # Create visualization plot
            fig = visualize_synthetic_fault(
                image, synthetic_image, fault_mask, severity_map, severity_label,
                patch_info, filename
            )
            
            # Save plot
            plot_name = f"{Path(filename).stem}_sev{severity_label.item():.3f}_plot.png".replace(".", "")
            fig.savefig(f"{output_dir}/{plot_name}", dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            results.append({
                'filename': filename,
                'severity': severity_label.item(),
                'coverage': fault_mask.sum().item() / fault_mask.numel() * 100,
                'patch_info': patch_info
            })
            
            print(f"    ✅ Saved results for {filename}")
            print()
        
        # Print summary
        print("📊 Summary:")
        severities = [r['severity'] for r in results]
        coverages = [r['coverage'] for r in results]
        
        print(f"  Samples processed: {len(results)}")
        print(f"  Severity range: {min(severities):.3f} - {max(severities):.3f}")
        print(f"  Average severity: {np.mean(severities):.3f}")
        print(f"  Coverage range: {min(coverages):.2f}% - {max(coverages):.2f}%")
        print(f"  Average coverage: {np.mean(coverages):.2f}%")
        print()
        
        # Print file organization
        print("📄 Generated files:")
        print("  Individual images: [sample]_sev[XXX]_[01-06]_[type].png")
        print("    01: Original image")
        print("    02: Synthetic image with fault")
        print("    03: Difference map")
        print("    04: Fault mask")
        print("    05: Severity map")
        print("    06: Overlay (fault highlighted)")
        print("  Visualization plots: [sample]_sev[XXX]_plot.png")
        print()
        
        print("=" * 70)
        print("✅ Best configuration visualization test completed successfully!")
        print(f"📁 Check {output_dir}/ for all generated files")
        print()
        
        # Verify outputs exist
        output_path = Path(output_dir)
        assert output_path.exists(), "Output directory should exist"
        
        png_files = list(output_path.glob("*.png"))
        expected_files = len(samples) * 7  # 6 individual images + 1 plot per sample
        assert len(png_files) == expected_files, f"Expected {expected_files} files, got {len(png_files)}"
        
        print(f"✅ Verification: {len(png_files)} files generated as expected")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_best_config_visualization()