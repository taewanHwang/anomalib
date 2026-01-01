"""FE-CLIP Model for Zero-Shot Anomaly Detection.

This module provides the FE-CLIP model implementation for zero-shot anomaly
detection and segmentation using frequency-enhanced CLIP features.

The model injects frequency information into CLIP's visual encoder through
two complementary adapters:
- FFE (Frequency-aware Feature Extraction): DCT-based frequency extraction
- LFS (Local Frequency Statistics): Local frequency pattern statistics

Example:
    >>> from anomalib.models.image import FEClip
    >>> model = FEClip()  # doctest: +SKIP
    >>> # Train with MVTec AD
    >>> engine.fit(model=model, datamodule=datamodule)  # doctest: +SKIP

Paper:
    FE-CLIP: Frequency Enhanced CLIP Model for Zero-Shot Anomaly Detection
    https://arxiv.org/abs/...
"""

from .lightning_model import FEClip
from .torch_model import FEClipModel

__all__ = ["FEClip", "FEClipModel"]
