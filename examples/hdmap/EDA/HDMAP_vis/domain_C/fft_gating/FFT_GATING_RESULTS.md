# FFT-based Gating Analysis Results

## Domain: domain_C
## Samples per group: 50

## Results

| Group | cosine | correlation | radial_cosine | band_cosine | l2_distance |
|-------|-----|-----|-----|-----|-----|
| fault/cold | 12.0% | 98.0% | 12.0% | 74.0% | 12.0% |
| fault/warm | 100.0% | 4.0% | 0.0% | 0.0% | 100.0% |
| good/cold | 46.9% | 98.0% | 67.3% | 73.5% | 46.9% |
| good/warm | 100.0% | 18.0% | 62.0% | 62.0% | 100.0% |
| **OVERALL** | **64.8%** | **54.3%** | **35.2%** | **52.3%** | **64.8%** |

## Best Method: cosine (64.8%)

## Comparison with CLIP-based Gating
- CLIP Global: 87.5%
- CLIP Confidence: 88.8%
- FFT Best (cosine): 64.8%
