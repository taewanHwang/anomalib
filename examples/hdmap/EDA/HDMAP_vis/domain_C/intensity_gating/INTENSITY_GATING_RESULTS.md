# Mean Intensity-based Gating Analysis

## Domain: domain_C
## Samples per group: 500

## Optimal Threshold: 0.2385

## Per-Group Results

| Group | Accuracy | Mean Intensity |
|-------|----------|----------------|
| fault/cold | 91.2% | 0.2158 |
| fault/warm | 100.0% | 0.3099 |
| good/cold | 100.0% | 0.1904 |
| good/warm | 98.2% | 0.2654 |
| **OVERALL** | **97.4%** | - |

## Comparison

| Method | Accuracy |
|--------|----------|
| **Mean Intensity** | **97.4%** |
| CLIP Confidence | 88.8% |
| CLIP Global | 87.5% |
| FFT Best | 64.8% |
