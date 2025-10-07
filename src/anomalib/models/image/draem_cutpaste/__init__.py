"""DRAEM CutPaste model for anomaly detection.

DRAEM with CutPaste augmentation (without severity head).
This is an intermediate validation step between DRAEM and DRAEM CutPaste Clf.
"""

from .lightning_model import DraemCutPaste
from .torch_model import DraemCutPasteModel

__all__ = ["DraemCutPaste", "DraemCutPasteModel"]
