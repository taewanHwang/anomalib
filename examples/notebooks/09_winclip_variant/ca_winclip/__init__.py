"""Condition-Aware WinCLIP (CA-WinCLIP) module.

This module provides condition-aware anomaly detection using WinCLIP with
multiple reference banks. The key idea is to maintain separate reference
banks for different conditions (e.g., cold/warm) and automatically select
the appropriate bank for each test image.

Main components:
- MultiConditionGating: CLIP embedding-based Top-K similarity gating
- P90IntensityGating: Intensity-based p90 percentile gating (recommended for HDMAP)
- OracleGating: Ground truth-based bank selection for evaluation
- ConditionAwareWinCLIP: Wrapper that combines gating with WinCLIP scoring
- BankEmbeddings: Container for a bank's cached embeddings

Gating Performance (HDMAP dataset):
- P90IntensityGating: 96.7% accuracy (recommended)
- MultiConditionGating (CLIP): 88.8% accuracy

Example:
    >>> from anomalib.models.image.winclip.torch_model import WinClipModel
    >>> from ca_winclip import ConditionAwareWinCLIP, P90IntensityGating
    >>>
    >>> # Initialize base model with text embeddings only
    >>> base_model = WinClipModel()
    >>> base_model.setup("industrial sensor data", None)
    >>>
    >>> # Create reference banks (cold/warm)
    >>> reference_banks = {"cold": cold_refs, "warm": warm_refs}
    >>>
    >>> # Create P90 gating (recommended)
    >>> gating = P90IntensityGating(domain="domain_C")
    >>>
    >>> # Create CA-WinCLIP with P90 gating
    >>> ca_model = ConditionAwareWinCLIP(base_model, reference_banks, gating=gating)
    >>>
    >>> # Run inference
    >>> pixel_scores, selected_banks, gating_details = ca_model.forward(test_batch)
"""

from .condition_aware_model import BankEmbeddings, ConditionAwareWinCLIP
from .gating import (
    InverseGating,
    MultiConditionGating,
    OracleGating,
    P90IntensityGating,
    RandomGating,
)

__all__ = [
    "BankEmbeddings",
    "ConditionAwareWinCLIP",
    "InverseGating",
    "MultiConditionGating",
    "OracleGating",
    "P90IntensityGating",
    "RandomGating",
]
