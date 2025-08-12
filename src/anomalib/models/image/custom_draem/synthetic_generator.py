# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""HDMAP-specific synthetic fault generator for Custom DRAEM.

This module provides functionality to generate synthetic faults specifically designed
for HDMAP datasets using rectangular patch-based cut-paste approach. Unlike the 
original DRAEM's Perlin noise-based generation, this generator:

1. Uses rectangular patches cut from the same image
2. Supports configurable aspect ratios (landscape/portrait/square)
3. Generates continuous severity labels
4. Validates patch boundaries automatically
5. Supports multi-patch generation with identical properties

Example:
    >>> from anomalib.models.image.custom_draem.synthetic_generator import HDMAPCutPasteSyntheticGenerator
    >>> import torch
    >>> 
    >>> generator = HDMAPCutPasteSyntheticGenerator(
    ...     patch_ratio_range=(2.0, 4.0),  # Landscape patches
    ...     patch_size_range=(20, 80),     # 20-80 pixels
    ...     severity_max=10.0,             # 0-10 severity range
    ...     patch_count=1                  # Single patch
    ... )
    >>> 
    >>> # Generate synthetic fault
    >>> image = torch.randn(1, 256, 256)  # 1-channel grayscale
    >>> synthetic_image, fault_mask, severity_map, severity_label = generator(image)
"""

import random
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class HDMAPCutPasteSyntheticGenerator(nn.Module):
    """HDMAP-specific synthetic fault generator using cut-paste approach.
    
    Generates synthetic faults by cutting rectangular patches from the same image
    and pasting them at different locations with varying intensity modifications.
    
    Args:
        patch_width_range (tuple[int, int], optional): Range of patch widths in pixels.
            Defaults to ``(30, 100)``.
        patch_ratio_range (tuple[float, float], optional): Range of height/width ratios.
            Values >1.0 create portrait patches (taller), <1.0 create landscape patches 
            (wider), 1.0 creates square patches. Defaults to ``(0.3, 3.0)``.
        severity_max (float, optional): Maximum severity value for continuous labels.
            Defaults to ``10.0``.
        patch_count (int, optional): Number of patches to generate per image.
            All patches use identical properties. Defaults to ``1``.
        probability (float, optional): Probability of applying synthetic fault generation.
            Value between 0.0 and 1.0. If random value > probability, returns original
            image with empty masks. Defaults to ``0.5``.
        validation_enabled (bool, optional): Enable automatic boundary validation.
            Defaults to ``True``.
            
    Example:
        >>> generator = HDMAPCutPasteSyntheticGenerator(
        ...     patch_width_range=(40, 80),
        ...     patch_ratio_range=(1.5, 2.5),  # Portrait patches (taller)
        ...     severity_max=5.0,
        ...     patch_count=2
        ... )
        >>> synthetic_image, fault_mask, severity_map, severity_label = generator(image)
        
    Note:
        - Patch dimensions are automatically validated to prevent boundary overflow
        - Multi-patch generation uses identical severity, ratio, and size for all patches
        - Severity values are sampled uniformly from [0, severity_max]
        - Cut-paste locations are selected randomly within valid boundaries
    """
    
    def __init__(
        self,
        patch_width_range: tuple[int, int] = (30, 100),
        patch_ratio_range: tuple[float, float] = (0.3, 3.0),
        severity_max: float = 10.0,
        patch_count: int = 1,
        probability: float = 0.5,
        validation_enabled: bool = True,
    ) -> None:
        super().__init__()
        
        self.patch_width_range = patch_width_range
        self.patch_ratio_range = patch_ratio_range
        self.severity_max = severity_max
        self.patch_count = patch_count
        self.probability = probability
        self.validation_enabled = validation_enabled
        
        # Validate input parameters
        self._validate_parameters()
        
    def _validate_parameters(self) -> None:
        """Validate generator parameters."""
        if self.patch_ratio_range[0] <= 0 or self.patch_ratio_range[1] <= 0:
            raise ValueError("Patch ratio values must be positive")
            
        if self.patch_width_range[0] <= 0 or self.patch_width_range[1] <= 0:
            raise ValueError("Patch width values must be positive")
            
        if self.severity_max <= 0:
            raise ValueError("Severity max must be positive")
            
        if self.patch_count <= 0:
            raise ValueError("Patch count must be positive")
            
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("Probability must be between 0.0 and 1.0")
    
    def forward(
        self, 
        image: torch.Tensor,
        return_patch_info: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Generate synthetic faults using cut-paste approach.
        
        Args:
            image (torch.Tensor): Input grayscale image of shape 
                ``(batch_size, 1, height, width)`` or ``(1, height, width)``
            return_patch_info (bool, optional): Whether to return detailed patch information.
                Defaults to ``False``.
                
        Returns:
            tuple containing:
                - synthetic_image (torch.Tensor): Image with synthetic faults
                - fault_mask (torch.Tensor): Binary mask indicating fault locations
                - severity_map (torch.Tensor): Pixel-wise severity map
                - severity_label (torch.Tensor): Image-level severity value
                - patch_info (dict, optional): Detailed patch information if return_patch_info=True
                
        Note:
            All patches in multi-patch generation share the same properties
            to reduce complexity and ensure consistent experimental conditions.
            
            When return_patch_info=True, the patch_info dict contains:
                - patch_ratio: Actual aspect ratio used
                - patch_size: (width, height) in pixels
                - base_patch_size: Base size before ratio application
                - severity_value: Actual severity value used
                - patch_positions: List of (src_x, src_y, tgt_x, tgt_y) for each patch
                - patch_count: Number of patches generated
        """
        # Ensure correct input dimensions
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension
            
        batch_size, channels, height, width = image.shape
        
        # Validate image dimensions
        if self.validation_enabled:
            self._validate_image_dimensions(height, width)
        
        # Initialize outputs
        synthetic_images = []
        fault_masks = []
        severity_maps = []
        severity_labels = []
        all_patch_info = []
        
        for i in range(batch_size):
            # Generate synthetic fault for single image
            if return_patch_info:
                synthetic_img, fault_mask, severity_map, severity_label, patch_info = self._generate_single_fault(
                    image[i:i+1], return_patch_info=True
                )
                all_patch_info.append(patch_info)
            else:
                synthetic_img, fault_mask, severity_map, severity_label = self._generate_single_fault(
                    image[i:i+1], return_patch_info=False
                )
            
            synthetic_images.append(synthetic_img)
            fault_masks.append(fault_mask)
            severity_maps.append(severity_map)
            severity_labels.append(severity_label)
        
        # Stack results
        synthetic_image = torch.cat(synthetic_images, dim=0)
        fault_mask = torch.cat(fault_masks, dim=0)
        severity_map = torch.cat(severity_maps, dim=0)
        severity_label = torch.stack(severity_labels, dim=0)  # stack scalars to create [batch_size] tensor
        
        if return_patch_info:
            # For batch size > 1, return list of patch_info dicts
            if batch_size == 1:
                return synthetic_image, fault_mask, severity_map, severity_label, all_patch_info[0]
            else:
                return synthetic_image, fault_mask, severity_map, severity_label, all_patch_info
        else:
            return synthetic_image, fault_mask, severity_map, severity_label
    
    def _validate_image_dimensions(self, height: int, width: int) -> None:
        """Validate that patch sizes are compatible with image dimensions."""
        max_patch_size = max(self.patch_width_range)
        
        if max_patch_size >= min(height, width):
            raise ValueError(
                f"Maximum patch size {max_patch_size} is too large for image "
                f"dimensions {height}x{width}"
            )
    
    def _generate_single_fault(
        self, 
        image: torch.Tensor,
        return_patch_info: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Generate synthetic fault for a single image using cut-paste approach.
        
        Args:
            image (torch.Tensor): Input image of shape (1, channels, height, width)
            return_patch_info (bool): Whether to return detailed patch information
            
        Returns:
            tuple: (synthetic_image, fault_mask, severity_map, severity_label[, patch_info])
        """
        batch_size, channels, height, width = image.shape
        
        # Probabilistic fault generation - same logic as DRAEM's PerlinAnomalyGenerator
        if torch.rand(1, device=image.device) > self.probability:
            # Return original image with empty masks (no fault generation)
            empty_mask = torch.zeros_like(image)
            zero_severity = torch.tensor(0.0, dtype=torch.float32, device=image.device)
            
            if return_patch_info:
                patch_info = {
                    "patch_ratio": 0.0,
                    "patch_size": (0, 0),
                    "patch_width": 0,
                    "patch_height": 0,
                    "severity_value": 0.0,
                    "patch_positions": [],
                    "patch_count": 0,
                    "patch_type": "No fault (probability skip)",
                    "coverage_percentage": 0.0
                }
                return image, empty_mask, empty_mask, zero_severity, patch_info
            else:
                return image, empty_mask, empty_mask, zero_severity
        
        # Sample patch parameters (shared across all patches)
        patch_width = random.randint(*self.patch_width_range)
        patch_ratio = random.uniform(*self.patch_ratio_range)  # height/width ratio
        severity_value = random.uniform(0, self.severity_max)
        
        # Calculate patch dimensions using simple formula: height = width * ratio
        patch_height = int(patch_width * patch_ratio)
        
        # Ensure patch fits within image boundaries
        patch_width = min(patch_width, width - 1)
        patch_height = min(patch_height, height - 1)
        
        # Initialize outputs
        synthetic_image = image.clone()
        fault_mask = torch.zeros_like(image)
        severity_map = torch.zeros_like(image)
        
        # Initialize patch info collection
        patch_positions = []
        
        # Generate patches
        for _ in range(self.patch_count):
            # 1. Select random source location (where to cut from)
            src_x = random.randint(0, width - patch_width)
            src_y = random.randint(0, height - patch_height)
            
            # 2. Select random target location (where to paste to)
            # Ensure target is different from source
            max_attempts = 10
            for _ in range(max_attempts):
                tgt_x = random.randint(0, width - patch_width)
                tgt_y = random.randint(0, height - patch_height)
                
                # Check if target overlaps significantly with source
                if not self._locations_overlap(src_x, src_y, tgt_x, tgt_y, patch_width, patch_height):
                    break
            
            # 3. Extract patch from source location
            source_patch = synthetic_image[0, :, src_y:src_y+patch_height, src_x:src_x+patch_width].clone()
            
            # 4. Apply severity modification to the patch
            modified_patch = self._apply_severity_modification(source_patch, severity_value)
            
            # 5. Paste modified patch to target location
            synthetic_image[0, :, tgt_y:tgt_y+patch_height, tgt_x:tgt_x+patch_width] = modified_patch
            
            # 6. Update masks
            fault_mask[0, :, tgt_y:tgt_y+patch_height, tgt_x:tgt_x+patch_width] = 1.0
            severity_map[0, :, tgt_y:tgt_y+patch_height, tgt_x:tgt_x+patch_width] = severity_value / self.severity_max
            
            # 7. Store patch position info
            patch_positions.append((src_x, src_y, tgt_x, tgt_y))
        
        # Image-level severity (max of all patches) - shape [1] for single image
        severity_label = torch.tensor(severity_value, dtype=torch.float32, device=image.device)
        
        if return_patch_info:
            # Create detailed patch information dictionary
            patch_info = {
                "patch_ratio": patch_ratio,
                "patch_size": (patch_width, patch_height),
                "patch_width": patch_width,
                "patch_height": patch_height,
                "severity_value": severity_value,
                "patch_positions": patch_positions,
                "patch_count": self.patch_count,
                "patch_type": self._get_patch_type_description(patch_ratio),
                "coverage_percentage": (fault_mask.sum().item() / fault_mask.numel()) * 100.0
            }
            return synthetic_image, fault_mask, severity_map, severity_label, patch_info
        else:
            return synthetic_image, fault_mask, severity_map, severity_label
    
    def _locations_overlap(
        self, 
        src_x: int, src_y: int, 
        tgt_x: int, tgt_y: int, 
        patch_width: int, patch_height: int,
        overlap_threshold: float = 0.3
    ) -> bool:
        """Check if source and target locations overlap significantly."""
        # Calculate intersection area
        left = max(src_x, tgt_x)
        right = min(src_x + patch_width, tgt_x + patch_width)
        top = max(src_y, tgt_y)
        bottom = min(src_y + patch_height, tgt_y + patch_height)
        
        if left >= right or top >= bottom:
            return False  # No overlap
        
        overlap_area = (right - left) * (bottom - top)
        patch_area = patch_width * patch_height
        overlap_ratio = overlap_area / patch_area
        
        return overlap_ratio > overlap_threshold
    
    def _apply_severity_modification(
        self, 
        patch: torch.Tensor, 
        severity_value: float
    ) -> torch.Tensor:
        """Apply severity-based modification to the patch.
        
        Args:
            patch (torch.Tensor): Original patch of shape (channels, height, width)
            severity_value (float): Severity value (0 to severity_max)
            
        Returns:
            torch.Tensor: Modified patch with severity-based changes
        """
        
        intensity_factor = 1.0 + severity_value
        modified_patch = torch.clamp(patch * intensity_factor, 0.0, 1.0)
        
        return modified_patch
    
    def _get_patch_type_description(self, patch_ratio: float) -> str:
        """Get human-readable description of patch type based on height/width ratio."""
        if patch_ratio > 1.0:
            return f"Portrait (H:W = {patch_ratio:.2f}:1)"
        elif patch_ratio < 1.0:
            return f"Landscape (H:W = {patch_ratio:.2f}:1)"
        else:
            return "Square (H:W = 1:1)"
    
    def get_patch_info(self) -> dict:
        """Get information about current patch configuration.
        
        Returns:
            dict: Configuration information including:
                - patch_ratio_range: Aspect ratio range
                - patch_size_range: Size range in pixels
                - severity_max: Maximum severity value
                - patch_count: Number of patches per image
                - patch_type: Description of patch orientation
        """
        # Determine patch type description
        min_ratio, max_ratio = self.patch_ratio_range
        if min_ratio > 1.0:
            patch_type = "Landscape (wider than tall)"
        elif max_ratio < 1.0:
            patch_type = "Portrait (taller than wide)"
        elif min_ratio <= 1.0 <= max_ratio:
            patch_type = "Mixed (landscape, square, portrait)"
        else:
            patch_type = "Square (equal dimensions)"
            
        return {
            "patch_width_range": self.patch_width_range,
            "patch_ratio_range": self.patch_ratio_range,
            "severity_max": self.severity_max,
            "patch_count": self.patch_count,
            "probability": self.probability,
            "patch_type": patch_type,
            "validation_enabled": self.validation_enabled,
        }
