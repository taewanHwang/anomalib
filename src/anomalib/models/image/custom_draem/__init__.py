"""Custom DRAEM model.

Custom DRAEM extends the original DRAEM model with an additional Fault Severity 
Prediction Sub-Network designed specifically for HDMAP datasets. The model uses 
a triple-branch architecture:

1. Reconstructive Sub-Network: Image reconstruction (1-channel grayscale)
2. Discriminative Sub-Network: Anomaly segmentation  
3. Fault Severity Sub-Network: Continuous severity prediction

Key Features:
- Rectangular patch-based synthetic fault generation with probabilistic control
- Multi-modal input combinations for severity prediction (5 modes)
- Support for 256x256 grayscale HDMAP images
- Configurable severity range and patch parameters
- Probabilistic anomaly generation (0-100% control)

Examples:
    Basic usage with default settings:
    
    >>> from anomalib.models.image import CustomDraem
    >>> model = CustomDraem()
    >>> # Uses: severity_max=10.0, anomaly_probability=0.5, discriminative_only mode
    
    Conservative training setup (30% anomalies):
    
    >>> model = CustomDraem(
    ...     severity_max=5.0,
    ...     severity_input_mode="discriminative_only",
    ...     anomaly_probability=0.3,
    ...     patch_ratio_range=(1.5, 3.0),  # Portrait patches
    ...     patch_width_range=(20, 40),    # Small patches
    ...     reconstruction_weight=1.0,
    ...     segmentation_weight=1.0,
    ...     severity_weight=0.3
    ... )
    
    Intensive training setup (80% anomalies) with multi-modal severity:
    
    >>> model = CustomDraem(
    ...     severity_max=15.0,
    ...     severity_input_mode="multi_modal",  # Uses all 5 channels
    ...     anomaly_probability=0.8,
    ...     patch_ratio_range=(0.3, 0.8),  # Landscape patches  
    ...     patch_width_range=(40, 80),    # Larger patches
    ...     patch_count=2,                 # Multiple patches
    ...     reconstruction_weight=0.8,
    ...     segmentation_weight=1.2,
    ...     severity_weight=0.7
    ... )
    
    Ablation study - testing different severity input modes:
    
    >>> # Mode 1: Baseline (discriminative only)
    >>> model_baseline = CustomDraem(severity_input_mode="discriminative_only")
    >>> 
    >>> # Mode 2: With original image information
    >>> model_with_orig = CustomDraem(severity_input_mode="with_original")
    >>> 
    >>> # Mode 3: With reconstruction information  
    >>> model_with_recon = CustomDraem(severity_input_mode="with_reconstruction")
    >>> 
    >>> # Mode 4: With explicit error map
    >>> model_with_error = CustomDraem(severity_input_mode="with_error_map")
    >>> 
    >>> # Mode 5: Maximum information fusion
    >>> model_multimodal = CustomDraem(severity_input_mode="multi_modal")
    
    Synthetic fault generator usage:
    
    >>> from anomalib.models.image.custom_draem import HDMAPCutPasteSyntheticGenerator
    >>> generator = HDMAPCutPasteSyntheticGenerator(
    ...     patch_width_range=(30, 60),
    ...     patch_ratio_range=(0.5, 2.0),
    ...     severity_max=10.0,
    ...     patch_count=1,
    ...     probability=0.5  # 50% chance of anomaly generation
    ... )
    >>> synthetic_image, fault_mask, severity_map, severity_label = generator(image_batch)

The model can be used with HDMAP datasets and supports various domain transfer
learning scenarios. All examples above support both training and inference modes.

Training Tips:
    - Start with discriminative_only mode for baseline performance
    - Use anomaly_probability=0.5 for balanced training  
    - Adjust severity_weight (0.3-0.7) based on your task importance
    - Portrait patches (ratio > 1.0) work well for road defects
    - Landscape patches (ratio < 1.0) work well for lane anomalies

Notes:
    The model implementation is available in the ``lightning_model`` module.
    Synthetic fault generation is handled by ``HDMAPCutPasteSyntheticGenerator``.
    All parameters support dynamic adjustment for hyperparameter tuning.

See Also:
    :class:`anomalib.models.image.custom_draem.lightning_model.CustomDraem`:
        Lightning implementation of the Custom DRAEM model.
    :class:`anomalib.models.image.custom_draem.synthetic_generator.HDMAPCutPasteSyntheticGenerator`:
        Synthetic fault generator for HDMAP data.
    :class:`anomalib.models.image.custom_draem.torch_model.CustomDraemModel`:
        PyTorch implementation of the Custom DRAEM model.
"""

from .lightning_model import CustomDraem
from .torch_model import CustomDraemModel, ReconstructiveSubNetwork, DiscriminativeSubNetwork, FaultSeveritySubNetwork
from .synthetic_generator import HDMAPCutPasteSyntheticGenerator

__all__ = [
    "CustomDraem",
    "CustomDraemModel", 
    "ReconstructiveSubNetwork",
    "DiscriminativeSubNetwork", 
    "FaultSeveritySubNetwork",
    "HDMAPCutPasteSyntheticGenerator"
]
