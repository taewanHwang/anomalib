"""Dinomaly variants for HDMAP experiments.

This module contains various modifications to the original Dinomaly model
for improved anomaly detection on the HDMAP dataset.

Available variants:
- DinomalyGEM: GEM (Generalized Mean) pooling with training integration (v2)
  - Uses CosineHardMiningGEMLoss to apply GEM during training
  - GEM aggregates scale-wise distance maps before hard mining
  - Result: No significant improvement over baseline (HDMAP)

- DinomalyTopK: Top-q% Loss for tail-focused learning
  - Uses CosineTopKLoss to focus on top q% of distance values
  - Aligns training objective with evaluation metric (TPR@low FPR)
  - Result: Domain_C TPR@1% improved from 76% to 80% with q=2%

- DinomalyHorizontal: Horizontal Segment Dropout for direction-aware regularization
  - Uses HorizontalSegmentDropout in bottleneck MLP
  - Drops consecutive tokens within rows to suppress horizontal reconstruction
  - Target: Improve domain_C TPR@1% from 80% to 82%+ (with physics prior)
"""

from anomalib.models.image.dinomaly_variants.gem_loss import CosineHardMiningGEMLoss
from anomalib.models.image.dinomaly_variants.gem_pooling import DinomalyGEM, DinomalyGEMModel
from anomalib.models.image.dinomaly_variants.topk_loss import CosineTopKLoss
from anomalib.models.image.dinomaly_variants.topk_model import DinomalyTopK, DinomalyTopKModel
from anomalib.models.image.dinomaly_variants.horizontal_dropout import (
    HorizontalSegmentDropout,
    HorizontalSegmentMLP,
)
from anomalib.models.image.dinomaly_variants.horizontal_model import (
    DinomalyHorizontal,
    DinomalyHorizontalModel,
)
from anomalib.models.image.dinomaly_variants.horizontal_topk_model import (
    DinomalyHorizontalTopK,
    DinomalyHorizontalTopKModel,
)
from anomalib.models.image.dinomaly_variants.kv_dropout import (
    LinearAttentionWithVDropout,
    VRowSegmentDropout,
)
from anomalib.models.image.dinomaly_variants.kv_dropout_model import (
    DinomalyKVDropout,
    DinomalyKVDropoutModel,
)

__all__ = [
    # GEM variants
    "CosineHardMiningGEMLoss",
    "DinomalyGEM",
    "DinomalyGEMModel",
    # TopK variants
    "CosineTopKLoss",
    "DinomalyTopK",
    "DinomalyTopKModel",
    # Horizontal Dropout variants
    "HorizontalSegmentDropout",
    "HorizontalSegmentMLP",
    "DinomalyHorizontal",
    "DinomalyHorizontalModel",
    # Horizontal + TopK variants
    "DinomalyHorizontalTopK",
    "DinomalyHorizontalTopKModel",
    # K/V Dropout variants (v4)
    "VRowSegmentDropout",
    "LinearAttentionWithVDropout",
    "DinomalyKVDropout",
    "DinomalyKVDropoutModel",
]
