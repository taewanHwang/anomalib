"""Dinomaly variants for HDMAP experiments.

This module contains various modifications to the original Dinomaly model
for improved anomaly detection on the HDMAP dataset.

Available variants:
- DinomalyGEM: GEM (Generalized Mean) pooling with training integration (v2)
  - Uses CosineHardMiningGEMLoss to apply GEM during training
  - GEM aggregates scale-wise distance maps before hard mining
"""

from anomalib.models.image.dinomaly_variants.gem_loss import CosineHardMiningGEMLoss
from anomalib.models.image.dinomaly_variants.gem_pooling import DinomalyGEM, DinomalyGEMModel

__all__ = ["CosineHardMiningGEMLoss", "DinomalyGEM", "DinomalyGEMModel"]
