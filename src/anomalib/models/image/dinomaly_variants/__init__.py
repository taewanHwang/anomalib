"""Dinomaly variants for HDMAP experiments.

This module contains various modifications to the original Dinomaly model
for improved anomaly detection on the HDMAP dataset.

Available variants:
- DinomalyGEM: GEM (Generalized Mean) pooling instead of max pooling
"""

from anomalib.models.image.dinomaly_variants.gem_pooling import DinomalyGEM, DinomalyGEMModel

__all__ = ["DinomalyGEM", "DinomalyGEMModel"]
