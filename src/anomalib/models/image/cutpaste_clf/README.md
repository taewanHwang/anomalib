# CutPaste Classifier

**Baseline Comparison Model**: CutPaste Augmentation + Simple CNN Classifier

## Overview

This model serves as a **baseline comparison** for DRAEM CutPaste. It uses only:
- **CutPaste augmentation** for synthetic anomaly generation
- **Simple CNN classifier** (Table B1 architecture) for binary classification

**Key Differences from DRAEM CutPaste:**
- ❌ No reconstruction network
- ❌ No localization (anomaly map)
- ✅ Only binary classification

## Architecture

### Table B1: Baseline CNN Structure

| Layer | Parameters |
|-------|------------|
| Convolution | Size=5, Ch=32, Stride=1, Padding=2 |
| ReLU | - |
| Pooling | Size=3, Stride=2, Padding=1 |
| Convolution | Size=5, Ch=48, Stride=1, Padding=2 |
| ReLU | - |
| Pooling | Size=3, Stride=2, Padding=1 |
| Fully connected | Node=100 |
| ReLU | - |
| Fully connected | Node=100 |
| ReLU | - |
| Softmax | Output=2 |

## Training

The model is trained with:
- **CutPaste augmentation**: Synthetic anomaly generation
- **Binary CrossEntropyLoss**: Normal (0) vs Anomaly (1)
- **Augmentation probability**: 0.5 (default)

## Experiments

This model is used for ablation studies with:
- **Severity levels**: 1-10 (a_fault_range_end)
- **Patch sizes**: (2-16) × (2-16) pixels
- **Learning rates**: 1e-4, 5e-4, 1e-3

## Usage

```python
from anomalib.models.image import CutPasteClassifier

model = CutPasteClassifier(
    image_size=(128, 128),
    cut_w_range=(2, 16),
    cut_h_range=(2, 16),
    a_fault_start=1,
    a_fault_range_end=11,  # Severity: 1-10
    augment_probability=0.5
)
```

## Comparison

| Model | Reconstruction | Localization | Classification |
|-------|----------------|--------------|----------------|
| **CutPaste Classifier** | ❌ | ❌ | ✅ |
| **DRAEM CutPaste** | ✅ | ✅ | ❌ |
| **DRAEM CutPaste Clf** | ✅ | ✅ | ✅ |

This baseline demonstrates the contribution of adding reconstruction and
localization to the pure classification approach.
