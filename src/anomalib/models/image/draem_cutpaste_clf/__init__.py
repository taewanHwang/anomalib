"""DRAEM CutPaste Classification model.

This module implements DRAEM with CutPaste augmentation and CNN classification head.
Based on the original DRAME_CutPaste implementation with improvements for anomalib integration.

Paper: https://arxiv.org/abs/2108.04456
Original implementation: DRAME_CutPaste project
"""

from .lightning_model import DraemCutPasteClf

__all__ = ["DraemCutPasteClf"]