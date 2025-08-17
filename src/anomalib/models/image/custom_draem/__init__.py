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
- Support for any square RGB images (NxN dimensions)
- Configurable severity range and patch parameters
- Probabilistic anomaly generation (0-100% control)

Examples:
    Basic usage with default settings:
    
    >>> from anomalib.models.image import CustomDraem
    >>> model = CustomDraem()
    >>> # Uses: severity_max=1.0, anomaly_probability=0.5, single_scale mode
    
    Conservative training setup (30% anomalies):
    
    >>> model = CustomDraem(
    ...     severity_max=5.0,
    ...     severity_head_mode="single_scale",
    ...     score_combination="simple_average",
    ...     anomaly_probability=0.3,
    ...     patch_ratio_range=(1.5, 3.0),  # Portrait patches
    ...     patch_width_range=(20, 40),    # Small patches
    ...     severity_weight=0.3
    ... )
    
    Intensive training setup (80% anomalies) with multi-scale severity:
    
    >>> model = CustomDraem(
    ...     severity_max=15.0,
    ...     severity_head_mode="multi_scale",  # Uses multi-resolution features
    ...     score_combination="weighted_average",
    ...     severity_weight_for_combination=0.7,
    ...     anomaly_probability=0.8,
    ...     patch_ratio_range=(0.3, 0.8),  # Landscape patches  
    ...     patch_width_range=(40, 80),    # Larger patches
    ...     patch_count=2,                 # Multiple patches
    ...     severity_weight=0.7
    ... )
    
    Ablation study - testing different severity head configurations:
    
    >>> # Configuration 1: Single-scale with simple average
    >>> model_baseline = CustomDraem(
    ...     severity_head_mode="single_scale",
    ...     score_combination="simple_average"
    ... )
    >>> 
    >>> # Configuration 2: Multi-scale with simple average
    >>> model_multi_scale = CustomDraem(
    ...     severity_head_mode="multi_scale",
    ...     score_combination="simple_average"
    ... )
    >>> 
    >>> # Configuration 3: Single-scale with weighted combination
    >>> model_weighted = CustomDraem(
    ...     severity_head_mode="single_scale",
    ...     score_combination="weighted_average",
    ...     severity_weight_for_combination=0.7
    ... )
    >>> 
    >>> # Configuration 4: Maximum information fusion
    >>> model_maximum = CustomDraem(
    ...     severity_head_mode="multi_scale",
    ...     score_combination="maximum"
    ... )
    
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
    - Start with single_scale mode for baseline performance
    - Use multi_scale mode for maximum feature information
    - Use anomaly_probability=0.5 for balanced training  
    - Adjust severity_weight (0.3-0.7) based on your task importance
    - Try weighted_average score combination for flexible control
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
from .torch_model import CustomDraemModel, DiscriminativeSubNetwork, DraemSevNetOutput
from .severity_head import SeverityHead, SeverityHeadFactory
from .loss import DraemSevNetLoss, DraemSevNetLossFactory
from .synthetic_generator import HDMAPCutPasteSyntheticGenerator

__all__ = [
    "CustomDraem",
    "CustomDraemModel",
    "DiscriminativeSubNetwork", 
    "DraemSevNetOutput",
    "SeverityHead",
    "SeverityHeadFactory",
    "DraemSevNetLoss",
    "DraemSevNetLossFactory",
    "HDMAPCutPasteSyntheticGenerator"
]
