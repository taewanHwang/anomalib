"""Test CutPaste augmentation visualization for DRAEM CutPaste Classification.

This test generates augmented images using the CutPaste generator and saves them
as PNG files for visual inspection.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import pytest
from PIL import Image

from anomalib.data.datasets.image.hdmap import HDMAPDataset
from anomalib.data.utils import Split
from anomalib.models.image.draem_cutpaste_clf.synthetic_generator_v2 import CutPasteSyntheticGenerator


class TestCutPasteVisualization:
    """Test class for visualizing CutPaste augmentation results."""

    @pytest.fixture
    def config_data(self):
        """Load experiment configuration."""
        config_path = Path("examples/hdmap/single_domain/base-exp_draem_cutpaste_clf_debug.json")
        with open(config_path) as f:
            data = json.load(f)
        return data["experiment_conditions"]

    @pytest.fixture
    def output_dir(self):
        """Create output directory for visualization images."""
        output_path = Path("tests/unit/models/image/draem_cutpaste_clf/cutpaste_visualization")
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    @pytest.fixture
    def hdmap_dataset(self):
        """Load HDMAP dataset for real data testing."""
        dataset_root = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/10000_tiff_original_minmax")

        if not dataset_root.exists():
            pytest.skip(f"HDMAP dataset not found at {dataset_root}")

        dataset = HDMAPDataset(
            root=dataset_root,
            domain="domain_A",
            split=Split.TRAIN,  # Use training data (good images)
            target_size=(95, 95),  # Resize to 256x256 for consistent testing
            resize_method="resize"
        )

        return dataset

    def load_real_images(self, dataset, num_images=10):
        """Load real images from HDMAP dataset.

        Args:
            dataset: HDMAP dataset instance
            num_images (int): Number of images to load

        Returns:
            list[torch.Tensor]: List of image tensors
        """
        images = []
        indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))

        for idx in indices:
            sample = dataset[idx]
            image = sample.image  # ImageItem.image attribute
            images.append(image.unsqueeze(0))  # Add batch dimension

        return images


    def tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image for saving.

        Args:
            tensor (torch.Tensor): Image tensor (C, H, W) or (1, C, H, W)

        Returns:
            PIL.Image: Converted PIL image
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)  # Remove batch dimension

        # Clamp values to [0, 1]
        tensor = torch.clamp(tensor, 0, 1)

        # Handle different channel cases
        if tensor.shape[0] == 1:  # Single channel (grayscale)
            # Convert to 2D array for grayscale
            numpy_image = tensor.squeeze(0).numpy()
            numpy_image = (numpy_image * 255).astype(np.uint8)
            return Image.fromarray(numpy_image, mode='L')  # 'L' for grayscale
        else:  # Multi-channel (RGB)
            numpy_image = tensor.permute(1, 2, 0).numpy()
            numpy_image = (numpy_image * 255).astype(np.uint8)
            return Image.fromarray(numpy_image)

    @pytest.mark.parametrize("config_idx", [0, 1, 2])
    def test_cutpaste_augmentation_visualization(self, config_data, output_dir, hdmap_dataset, config_idx):
        """Test CutPaste augmentation and save visualization images using real HDMAP data.

        Args:
            config_data: Experiment configuration data
            output_dir: Output directory for saving images
            hdmap_dataset: HDMAP dataset fixture
            config_idx: Index of configuration to test (0, 1, 2)
        """
        config = config_data[config_idx]["config"]
        exp_name = config_data[config_idx]["name"]

        print(f"\n=== Testing {exp_name} with Real HDMAP Data ===")
        print(f"Config: {config['_comment_cutpaste_augmentation']}")
        print(f"Dataset size: {len(hdmap_dataset)} images")

        # Create CutPaste generator with config parameters
        generator = CutPasteSyntheticGenerator(
            cut_w_range=tuple(config["cut_w_range"]),
            cut_h_range=tuple(config["cut_h_range"]),
            a_fault_start=config["a_fault_start"],
            a_fault_range_end=config["a_fault_range_end"],
            probability=1.0,  # Force augmentation for visualization
            norm=config["norm"]
        )

        # Set random seed for reproducibility
        torch.manual_seed(42 + config_idx)
        random.seed(42 + config_idx)

        # Create experiment output directory
        exp_output_dir = output_dir / f"{exp_name}_real_data"
        exp_output_dir.mkdir(exist_ok=True)

        print(f"Generating augmented images with real HDMAP data...")
        print(f"- cut_w_range: {config['cut_w_range']}")
        print(f"- cut_h_range: {config['cut_h_range']}")
        print(f"- a_fault_start: {config['a_fault_start']}")
        print(f"- a_fault_range_end: {config['a_fault_range_end']}")
        print(f"- norm: {config['norm']}")

        # Load real images from HDMAP dataset
        real_images = self.load_real_images(hdmap_dataset, num_images=10)
        print(f"✅ Loaded {len(real_images)} real images")

        augmented_count = 0

        for i, original_image in enumerate(real_images):
            # Resize to target size if needed
            target_height, target_width = config["image_size"]
            if original_image.shape[2] != target_height or original_image.shape[3] != target_width:
                original_image = torch.nn.functional.interpolate(
                    original_image,
                    size=(target_height, target_width),
                    mode='bilinear',
                    align_corners=False
                )

            # Apply augmentation with patch info
            augmented_image, fault_mask, severity_map, severity_label, patch_info = generator(
                original_image, return_patch_info=True
            )

            # Debug: Print image information
            print(f"    Image {i+1}:")
            print(f"      Original shape: {original_image.shape}")
            print(f"      Original range: [{original_image.min():.4f}, {original_image.max():.4f}]")
            print(f"      Augmented shape: {augmented_image.shape}")
            print(f"      Augmented range: [{augmented_image.min():.4f}, {augmented_image.max():.4f}]")
            print(f"      Mask shape: {fault_mask.shape}")
            print(f"      Severity shape: {severity_map.shape}")
            print(f"      Severity label: {severity_label.item():.4f}")
            print(f"      Patch info:")
            print(f"        From: ({patch_info['from_location_h']}, {patch_info['from_location_w']})")
            print(f"        To: ({patch_info['to_location_h']}, {patch_info['to_location_w']})")
            print(f"        Size: {patch_info['cut_w']}x{patch_info['cut_h']}")
            print(f"        Amplitude: {patch_info['a_fault']:.4f}")
            print(f"        Coverage: {patch_info['coverage_percentage']:.2f}%")

            # Always save since we forced augmentation
            augmented_count += 1

            # Save original image
            original_pil = self.tensor_to_pil(original_image)
            original_path = exp_output_dir / f"original_{augmented_count:02d}.png"
            original_pil.save(original_path)
            print(f"      Saved original: {original_pil.mode} mode, size: {original_pil.size}")

            # Save augmented image
            augmented_pil = self.tensor_to_pil(augmented_image)
            augmented_path = exp_output_dir / f"augmented_{augmented_count:02d}.png"
            augmented_pil.save(augmented_path)
            print(f"      Saved augmented: {augmented_pil.mode} mode, size: {augmented_pil.size}")

            # Save fault mask
            mask_tensor = fault_mask.squeeze(0).repeat(3, 1, 1)  # Convert to 3-channel for visualization
            mask_pil = self.tensor_to_pil(mask_tensor)
            mask_path = exp_output_dir / f"mask_{augmented_count:02d}.png"
            mask_pil.save(mask_path)

            # Save severity map
            severity_tensor = severity_map.squeeze(0).repeat(3, 1, 1)  # Convert to 3-channel for visualization
            severity_pil = self.tensor_to_pil(severity_tensor)
            severity_path = exp_output_dir / f"severity_{augmented_count:02d}.png"
            severity_pil.save(severity_path)

            print(f"  Saved real image set {augmented_count}/10")

        print(f"✅ Generated {augmented_count} augmented image sets with real HDMAP data")
        print(f"✅ Saved to: {exp_output_dir}")

        # Verify files were created
        original_files = list(exp_output_dir.glob("original_*.png"))
        augmented_files = list(exp_output_dir.glob("augmented_*.png"))
        mask_files = list(exp_output_dir.glob("mask_*.png"))
        severity_files = list(exp_output_dir.glob("severity_*.png"))

        print(f"Debug: Expected {augmented_count} files, found:")
        print(f"  Original: {len(original_files)}")
        print(f"  Augmented: {len(augmented_files)}")
        print(f"  Mask: {len(mask_files)}")
        print(f"  Severity: {len(severity_files)}")

        assert len(original_files) == augmented_count
        assert len(augmented_files) == augmented_count
        assert len(mask_files) == augmented_count
        assert len(severity_files) == augmented_count

        # Create comparison image (side-by-side)
        if augmented_count > 0:
            self._create_comparison_image(exp_output_dir, augmented_count, exp_name + "_real_data")

    def _create_comparison_image(self, exp_output_dir, num_images, exp_name):
        """Create a comparison image showing original, augmented, and mask side by side.

        Args:
            exp_output_dir: Output directory path
            num_images: Number of image sets
            exp_name: Experiment name
        """
        # Load first image set for comparison
        original_path = exp_output_dir / "original_01.png"
        augmented_path = exp_output_dir / "augmented_01.png"
        mask_path = exp_output_dir / "mask_01.png"
        severity_path = exp_output_dir / "severity_01.png"

        if all(p.exists() for p in [original_path, augmented_path, mask_path, severity_path]):
            original = Image.open(original_path)
            augmented = Image.open(augmented_path)
            mask = Image.open(mask_path)
            severity = Image.open(severity_path)

            # Create side-by-side comparison (4 images: original, augmented, mask, severity)
            width, height = original.size
            comparison = Image.new('RGB', (width * 4, height))
            comparison.paste(original, (0, 0))
            comparison.paste(augmented, (width, 0))
            comparison.paste(mask, (width * 2, 0))
            comparison.paste(severity, (width * 3, 0))

            # Save comparison
            comparison_path = exp_output_dir / f"{exp_name}_comparison.png"
            comparison.save(comparison_path)

            print(f"✅ Created comparison image: {comparison_path}")

    def test_cutpaste_parameters_summary(self, config_data, output_dir):
        """Create a summary of all CutPaste parameters for easy comparison."""
        summary_path = output_dir / "cutpaste_parameters_summary.txt"

        with open(summary_path, 'w') as f:
            f.write("DRAEM CutPaste Classification - Augmentation Parameters Summary\n")
            f.write("=" * 65 + "\n\n")

            for i, condition in enumerate(config_data):
                config = condition["config"]
                f.write(f"{condition['name']}:\n")
                f.write(f"  Description: {condition['description']}\n")
                f.write(f"  cut_w_range: {config['cut_w_range']}\n")
                f.write(f"  cut_h_range: {config['cut_h_range']}\n")
                f.write(f"  a_fault_start: {config['a_fault_start']}\n")
                f.write(f"  a_fault_range_end: {config['a_fault_range_end']}\n")
                f.write(f"  augment_probability: {config['augment_probability']}\n")
                f.write(f"  norm: {config['norm']}\n")
                f.write(f"  clf_weight: {config['clf_weight']}\n")
                f.write("\n")

        print(f"✅ Created parameter summary: {summary_path}")


    def test_real_data_summary(self, output_dir):
        """Create a summary of real HDMAP data augmentation results."""
        summary_path = output_dir / "real_data_summary.txt"

        real_data_dirs = list(output_dir.glob("*_real_data"))

        with open(summary_path, 'w') as f:
            f.write("DRAEM CutPaste Classification - Real HDMAP Data Augmentation Summary\n")
            f.write("=" * 75 + "\n\n")

            f.write(f"Dataset: HDMAP Domain A (Training Data)\n")
            f.write(f"Dataset Path: /mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/10000_tiff_original_minmax\n")
            f.write(f"Total Experiments: {len(real_data_dirs)}\n\n")

            for exp_dir in sorted(real_data_dirs):
                exp_name = exp_dir.name.replace("_real_data", "")
                f.write(f"{exp_name}:\n")

                # Count files
                original_files = list(exp_dir.glob("original_*.png"))
                augmented_files = list(exp_dir.glob("augmented_*.png"))
                mask_files = list(exp_dir.glob("mask_*.png"))
                severity_files = list(exp_dir.glob("severity_*.png"))
                comparison_files = list(exp_dir.glob("*_comparison.png"))

                f.write(f"  Directory: {exp_dir.name}\n")
                f.write(f"  Original Images: {len(original_files)}\n")
                f.write(f"  Augmented Images: {len(augmented_files)}\n")
                f.write(f"  Fault Masks: {len(mask_files)}\n")
                f.write(f"  Severity Maps: {len(severity_files)}\n")
                f.write(f"  Comparison Images: {len(comparison_files)}\n")
                f.write(f"  Total Files: {len(list(exp_dir.glob('*.png')))}\n")
                f.write("\n")

            f.write("File Types Generated:\n")
            f.write("- original_XX.png: Real HDMAP images from dataset\n")
            f.write("- augmented_XX.png: CutPaste augmented versions\n")
            f.write("- mask_XX.png: Fault masks showing augmentation locations\n")
            f.write("- severity_XX.png: Severity maps showing fault intensity\n")
            f.write("- *_comparison.png: Side-by-side comparison (Original|Augmented|Mask|Severity)\n")

        print(f"✅ Created real data summary: {summary_path}")