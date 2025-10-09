"""CutPaste-based synthetic fault generator for DRAEM CutPaste Classification.

This module implements the CutPaste augmentation approach from the original DRAME_CutPaste
implementation, adapted for anomalib integration.

Based on DRAME_CutPaste/utils/utils_data_loader_v2.py and synthetic_generator_v2.py
"""

import random

import torch
from torch import nn


class CutPasteSyntheticGenerator(nn.Module):
    """Proper CutPaste-based synthetic fault generator.

    This implementation performs true CutPaste augmentation:
    - Cuts a patch from one location in the image
    - Pastes it to a different non-overlapping location
    - No artificial amplitude scaling or color changes
    - Preserves original image content structure

    Args:
        cut_w_range (tuple[int, int], optional): Range of patch widths in pixels.
            Defaults to ``(10, 80)``.
        cut_h_range (tuple[int, int], optional): Range of patch heights in pixels.
            Defaults to ``(1, 2)``.
        a_fault_start (float, optional): Minimum multiplier for severity calculation.
            Defaults to ``1.0``.
        a_fault_range_end (float, optional): Maximum multiplier for severity calculation.
            Defaults to ``10.0``.
        probability (float, optional): Probability of applying CutPaste augmentation.
            Value between 0.0 and 1.0. Defaults to ``0.5``.
        validation_enabled (bool, optional): Enable automatic boundary validation.
            Defaults to ``True``.

    Example:
        >>> generator = CutPasteSyntheticGenerator(
        ...     cut_w_range=(10, 80),
        ...     cut_h_range=(1, 2),
        ...     probability=0.5
        ... )
        >>> synthetic_image, fault_mask, severity_label = generator(image)
    """

    def __init__(
        self,
        cut_w_range: tuple[int, int] = (10, 80),
        cut_h_range: tuple[int, int] = (1, 2),
        a_fault_start: float = 1.0,
        a_fault_range_end: float = 10.0,
        probability: float = 0.5,
        norm: bool = True,
        validation_enabled: bool = True,
    ) -> None:
        super().__init__()

        self.cut_w_range = cut_w_range
        self.cut_h_range = cut_h_range
        self.a_fault_start = a_fault_start
        self.a_fault_range_end = a_fault_range_end
        self.probability = probability
        self.norm = norm
        self.validation_enabled = validation_enabled
        
        print(f"CutPasteSyntheticGenerator initialized with parameters: norm={self.norm}")

        # Validate input parameters
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate generator parameters."""
        if self.cut_w_range[0] <= 0 or self.cut_w_range[1] <= 0:
            raise ValueError("Cut width values must be positive")

        if self.cut_h_range[0] <= 0 or self.cut_h_range[1] <= 0:
            raise ValueError("Cut height values must be positive")

        if self.a_fault_start < 0 or self.a_fault_range_end < 0:
            raise ValueError("Fault amplitude values must be non-negative")

        if self.a_fault_start >= self.a_fault_range_end:
            raise ValueError("a_fault_start must be less than a_fault_range_end")

        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("Probability must be between 0.0 and 1.0")

    def forward(
        self,
        image: torch.Tensor,
        return_patch_info: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Generate synthetic faults using CutPaste approach.

        Args:
            image (torch.Tensor): Input image of shape
                ``(batch_size, channels, height, width)`` or ``(channels, height, width)``
                For multi-channel input, only first channel is processed.
            return_patch_info (bool, optional): Whether to return detailed patch information.
                Defaults to ``False``.

        Returns:
            tuple containing:
                - synthetic_image (torch.Tensor): Image with synthetic faults (or original if no fault)
                - fault_mask (torch.Tensor): Binary mask indicating fault locations
                - severity_label (torch.Tensor): Image-level severity value (fault amplitude)
                - patch_info (dict, optional): Detailed patch information if return_patch_info=True
        """
        # Ensure correct input dimensions
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension

        batch_size, _, height, width = image.shape

        # Process all channels (changed from original single-channel processing)
        multi_channel_image = image

        # Validate image dimensions
        if self.validation_enabled:
            self._validate_image_dimensions(height, width)

        # Initialize outputs
        synthetic_images = []
        fault_masks = []
        severity_labels = []
        all_patch_info = []

        for i in range(batch_size):
            # Generate synthetic fault for multi-channel image
            if return_patch_info:
                synthetic_multi_ch, fault_mask, severity_label, patch_info = self._generate_single_fault(
                    multi_channel_image[i:i+1], return_patch_info=True
                )
                all_patch_info.append(patch_info)
            else:
                synthetic_multi_ch, fault_mask, severity_label = self._generate_single_fault(
                    multi_channel_image[i:i+1], return_patch_info=False
                )

            # Use the multi-channel synthetic result directly
            synthetic_img = synthetic_multi_ch

            synthetic_images.append(synthetic_img)
            fault_masks.append(fault_mask)
            severity_labels.append(severity_label)

        # Stack results
        synthetic_image = torch.cat(synthetic_images, dim=0)
        fault_mask = torch.cat(fault_masks, dim=0)
        severity_label = torch.stack(severity_labels, dim=0)

        if return_patch_info:
            if batch_size == 1:
                return synthetic_image, fault_mask, severity_label, all_patch_info[0]
            else:
                return synthetic_image, fault_mask, severity_label, all_patch_info
        else:
            return synthetic_image, fault_mask, severity_label

    def _validate_image_dimensions(self, height: int, width: int) -> None:
        """Validate that patch sizes are compatible with image dimensions."""
        max_cut_w = max(self.cut_w_range)
        max_cut_h = max(self.cut_h_range)

        if max_cut_w >= width:
            raise ValueError(
                f"Maximum cut width {max_cut_w} is too large for image width {width}"
            )
        if max_cut_h >= height:
            raise ValueError(
                f"Maximum cut height {max_cut_h} is too large for image height {height}"
            )

    def _generate_single_fault(
        self,
        image: torch.Tensor,
        return_patch_info: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Generate synthetic fault for a single image using CutPaste approach.

        Args:
            image (torch.Tensor): Input image of shape (1, channels, height, width)
            return_patch_info (bool): Whether to return detailed patch information

        Returns:
            tuple: (synthetic_image, fault_mask, severity_label[, patch_info])
        """
        batch_size, _, height, width = image.shape

        # Random choice for anomaly generation
        no_anomaly = torch.rand(1, device=image.device).item()

        if no_anomaly > self.probability:  # normal data
            # Return original image with empty masks
            empty_mask = torch.zeros((batch_size, 1, height, width), device=image.device, dtype=image.dtype)
            zero_severity = torch.tensor(0.0, dtype=torch.float32, device=image.device)

            if return_patch_info:
                patch_info = {
                    "cut_w": 0,
                    "cut_h": 0,
                    "from_location_h": 0,
                    "from_location_w": 0,
                    "to_location_h": 0,
                    "to_location_w": 0,
                    "a_fault": 0.0,
                    "has_anomaly": 0,
                    "patch_type": "No fault (normal data)",
                    "coverage_percentage": 0.0,
                }
                return image, empty_mask, zero_severity, patch_info
            else:
                return image, empty_mask, zero_severity

        # Anomaly data creation
        # Sample cut region dimensions
        cut_h = random.randint(*self.cut_h_range)
        cut_w = random.randint(*self.cut_w_range)

        # Ensure patch fits within image boundaries
        cut_w = min(cut_w, width - 1)
        cut_h = min(cut_h, height - 1)

        # Sample from location
        from_location_h = random.randint(0, height - cut_h)
        from_location_w = random.randint(0, width - cut_w)

        # Sample to location (paste location) - ensure non-overlapping
        # Calculate valid range for to_location to avoid overlap
        valid_h_ranges = []
        valid_w_ranges = []

        # For height: valid ranges are [0, from_location_h - cut_h] and [from_location_h + cut_h, height - cut_h]
        if from_location_h - cut_h >= 0:
            valid_h_ranges.append((0, from_location_h - cut_h))
        if from_location_h + cut_h <= height - cut_h:
            valid_h_ranges.append((from_location_h + cut_h, height - cut_h))

        # For width: valid ranges are [0, from_location_w - cut_w] and [from_location_w + cut_w, width - cut_w]
        if from_location_w - cut_w >= 0:
            valid_w_ranges.append((0, from_location_w - cut_w))
        if from_location_w + cut_w <= width - cut_w:
            valid_w_ranges.append((from_location_w + cut_w, width - cut_w))

        # If no valid non-overlapping position exists, just use random position
        if not valid_h_ranges or not valid_w_ranges:
            to_location_h = random.randint(0, height - cut_h)
            to_location_w = random.randint(0, width - cut_w)
        else:
            # Choose random valid range and position within it
            h_range = random.choice(valid_h_ranges)
            w_range = random.choice(valid_w_ranges)
            to_location_h = random.randint(h_range[0], h_range[1])
            to_location_w = random.randint(w_range[0], w_range[1])

        # Extract patch from source location (all channels)
        patch = image[0, :, from_location_h:from_location_h + cut_h, from_location_w:from_location_w + cut_w].clone()

        # Apply normalization if enabled (similar to utils_data_loader_v2.py)
        if self.norm:
            # NaN 방지를 위한 안전한 정규화
            max_val = torch.max(torch.abs(patch))
            if max_val > 0:
                patch = patch / max_val
            # else: 모든 값이 0인 경우 그대로 유지

        # Sample fault amplitude and apply to patch (all channels equally)
        a_fault = random.uniform(self.a_fault_start, self.a_fault_range_end)
        augmented_patch = patch * a_fault

        # DEBUG: Print shapes and values for debugging
        # print(f"DEBUG - patch shape: {patch.shape}, channels equal: {torch.allclose(patch[0], patch[1]) and torch.allclose(patch[1], patch[2])}")
        # print(f"DEBUG - augmented_patch shape: {augmented_patch.shape}, channels equal: {torch.allclose(augmented_patch[0], augmented_patch[1]) and torch.allclose(augmented_patch[1], augmented_patch[2])}")
        # print(f"DEBUG - amplitude: {a_fault:.4f}")
        # print(f"DEBUG - patch means: R={patch[0].mean():.6f}, G={patch[1].mean():.6f}, B={patch[2].mean():.6f}")
        # print(f"DEBUG - augmented means: R={augmented_patch[0].mean():.6f}, G={augmented_patch[1].mean():.6f}, B={augmented_patch[2].mean():.6f}")

        # Create augmented image with CutPaste (addition instead of replacement)
        synthetic_image = image.clone()
        # 패치를 치환하는 대신 추가
        synthetic_image[0, :, to_location_h:to_location_h+cut_h, to_location_w:to_location_w+cut_w] += augmented_patch

        # Create fault mask
        fault_mask = torch.zeros((batch_size, 1, height, width), device=image.device, dtype=image.dtype)

        # Mark fault location
        fault_mask[0, 0, to_location_h:to_location_h+cut_h, to_location_w:to_location_w+cut_w] = 1.0

        # Image-level severity (fault amplitude value)
        severity_label = torch.tensor(a_fault, dtype=torch.float32, device=image.device)

        if return_patch_info:
            # Create detailed patch information dictionary
            patch_info = {
                "cut_w": cut_w,
                "cut_h": cut_h,
                "from_location_h": from_location_h,
                "from_location_w": from_location_w,
                "to_location_h": to_location_h,
                "to_location_w": to_location_w,
                "a_fault": a_fault,
                "has_anomaly": 1,
                "patch_type": f"CutPaste patch ({cut_w}x{cut_h})",
                "coverage_percentage": (fault_mask.sum().item() / fault_mask.numel()) * 100.0,
                "approach": "CutPaste with amplitude scaling",
                "patch_amplitude_scaling": a_fault
            }
            return synthetic_image, fault_mask, severity_label, patch_info
        else:
            return synthetic_image, fault_mask, severity_label

    def get_config_info(self) -> dict:
        """Get information about current generator configuration.

        Returns:
            dict: Configuration information including all parameters
        """
        return {
            "cut_w_range": self.cut_w_range,
            "cut_h_range": self.cut_h_range,
            "a_fault_start": self.a_fault_start,
            "a_fault_range_end": self.a_fault_range_end,
            "probability": self.probability,
            "validation_enabled": self.validation_enabled,
            "approach": "CutPaste with amplitude scaling",
            "version": "v2_cutpaste_amplitude"
        }


