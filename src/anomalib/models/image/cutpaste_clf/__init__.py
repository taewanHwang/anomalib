"""CutPaste Classifier - Baseline Comparison Model.

This module implements a baseline model using only CutPaste augmentation
and a simple CNN classifier (Table B1 structure) without reconstruction
or localization.

Example:
    >>> from anomalib.models.image import CutPasteClassifier
    >>> from anomalib.data import HDMAPDataModule
    >>> from anomalib.engine import Engine
    >>>
    >>> # Initialize model
    >>> model = CutPasteClassifier(
    ...     image_size=(128, 128),
    ...     cut_w_range=(2, 16),
    ...     cut_h_range=(2, 16),
    ...     a_fault_start=1,
    ...     a_fault_range_end=11
    ... )
    >>>
    >>> # Train using the Engine
    >>> datamodule = HDMAPDataModule()
    >>> engine = Engine()
    >>> engine.fit(model=model, datamodule=datamodule)
"""

from .lightning_model import CutPasteClassifier

__all__ = ["CutPasteClassifier"]
