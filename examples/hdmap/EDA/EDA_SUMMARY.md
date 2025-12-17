# HDMAP Adaptive Dropout Feature EDA Summary

**Date**: 2025-12-16
**Objective**: Find a suitable feature for adaptive bottleneck dropout in Dinomaly model
**Dataset**: HDMAP PNG (4 domains: A, B, C, D)

---

## Background

### Adaptive Dropout 가설
- 정상 샘플이 너무 규칙적(regular)일수록 bottleneck에서 더 강하게 dropout을 걸어 모델이 "정상 패턴을 암기해서 collapse/overfit"하는 것을 방지
- 필요한 metric: 정상 샘플 내부에서 '규칙성/방향성/대역 집중' 같은 overfit 위험도를 측정하는 지표

### 평가 기준
- **Cohen's d** (효과 크기):
  - |d| > 0.8: Strong separation
  - |d| > 0.5: Medium separation
  - |d| < 0.5: Weak separation
- **방향 일관성**: 모든 도메인에서 동일한 방향 (Normal > Anomaly 또는 Normal < Anomaly)

---

## 1. Orientation Entropy (OE) - Spatial Domain

### 개념
- Sobel gradient로 edge orientation 계산
- Orientation histogram의 entropy 측정
- Low entropy = 강한 방향성 (규칙적), High entropy = 약한 방향성 (복잡)

### 구현
- `src/anomalib/models/image/dinomaly/adaptive_dropout.py`
- `compute_orientation_entropy_batch()` 함수

### 결과

| Domain | Normal Mean | Anomaly Mean | Diff | Cohen's d | Judgment |
|--------|-------------|--------------|------|-----------|----------|
| domain_A | 0.545 | ~0.55 | ~+0.005 | ~0.1 | **Weak** |
| domain_B | 0.532 | ~0.54 | ~+0.008 | ~0.2 | **Weak** |
| domain_C | 0.555 | ~0.56 | ~+0.005 | ~0.1 | **Weak** |
| domain_D | 0.543 | ~0.55 | ~+0.007 | ~0.15 | **Weak** |

### 결론
- **부적합**: 분포 겹침이 심하고 separation이 약함
- 실험에서 adaptive dropout 효과 없음 확인됨

---

## 2. RSE (Radial Spectral Energy) - Frequency Domain

### 개념
- FFT power spectrum에서 radial binning으로 주파수 대역별 에너지 계산
- 특정 주파수 대역의 에너지 집중도 측정

### 결과
- **부적합**: EDA 레벨에서 Normal/Anomaly 분리 불충분
- Radial averaging이 방향 정보를 소실시킴

### 결론
- HDMAP의 핵심이 "방향성(가로/대각)"인데, radial binning은 이를 무시함
- 방향성 정보를 보존하는 다른 방법 필요

---

## 3. SFM (Spectral Flatness Measure) - Frequency Domain

### 개념
- 2D FFT power spectrum의 Geometric Mean / Arithmetic Mean
- Low SFM = 강한 피크 (규칙적/tonal), High SFM = 평탄 (노이즈성)

### 구현
- `examples/hdmap/EDA/sfm_eda.py`
- `examples/hdmap/EDA/sfm_visualization.py`

### 결과

| Domain | Normal Mean | Anomaly Mean | Diff | Cohen's d | Judgment |
|--------|-------------|--------------|------|-----------|----------|
| domain_A | 0.2009 | 0.2152 | +0.0143 | +0.34 | **Weak** |
| domain_B | 0.2487 | 0.2235 | -0.0252 | -0.46 | **Weak** |
| domain_C | 0.1831 | 0.2039 | +0.0208 | +0.49 | **Weak** |
| domain_D | 0.2404 | 0.2210 | -0.0194 | -0.34 | **Weak** |

### 문제점
1. **분포 겹침 심함**: Normal과 Anomaly 분포가 거의 완전히 겹침
2. **방향 불일치**: domain_A, C는 +, domain_B, D는 - (sign flip)
3. **전역 SFM의 한계**: 국소/방향/희소 변화 성격의 결함에 반응 약함

### 결론
- **부적합**: 전역 SFM은 adaptive dropout metric으로 불안정
- 살리려면 directional/band-limited 형태로 재정의 필요

### 결과 파일
```
examples/hdmap/EDA/sfm_results/
├── sfm_distributions.png
├── sfm_boxplot.png
├── sfm_results.json
└── domain_*/
    ├── comparison_*.png
    └── *_process.png
```

---

## 4. APE (Angular Power Entropy) - Frequency Domain ✅ **추천**

### 개념
- 2D FFT power spectrum에서 반지름(r)은 무시하고 각도(θ) 방향으로만 에너지 적분
- Angular energy distribution의 entropy 계산
- Low APE = 강한 방향성 (에너지가 특정 각도에 집중), High APE = isotropic (모든 방향 균일)

### 구현
- `examples/hdmap/EDA/ape_eda.py`
- `examples/hdmap/EDA/ape_visualization.py`

### 핵심 함수
```python
def angular_power_entropy_2d(
    x,
    num_angle_bins: int = 36,    # 10° resolution
    r_min_ratio: float = 0.05,   # DC 제외
    r_max_ratio: float = 0.95,   # 코너 제외
) -> float:
    # 1. 2D FFT → Power Spectrum
    # 2. 각도별 에너지 적분 → p(θ)
    # 3. Shannon Entropy 계산 → H(θ) / H_max
    return ape  # [0, 1]
```

### 결과

| Domain | Normal Mean | Anomaly Mean | Diff | Cohen's d | Judgment |
|--------|-------------|--------------|------|-----------|----------|
| domain_A | 0.7767 | 0.8502 | **+0.0734** | **+2.43** | **Strong** |
| domain_B | 0.7125 | 0.8549 | **+0.1424** | **+4.34** | **Strong** |
| domain_C | 0.8662 | 0.8887 | **+0.0225** | **+1.73** | **Strong** |
| domain_D | 0.8159 | 0.8874 | **+0.0714** | **+2.54** | **Strong** |

### 장점
1. **Strong separation**: 모든 도메인에서 Cohen's d > 1.7
2. **방향 일관성**: 모든 도메인에서 Anomaly APE > Normal APE
3. **분포 분리 명확**: 히스토그램에서 Normal/Anomaly 분포가 명확히 분리
4. **HDMAP 특성에 적합**: 방향성(가로/대각 라인)을 직접 측정

### 해석
- **Normal (APE 낮음)**: 에너지가 특정 방향(예: 180°/수평)에 집중 → 규칙적 패턴
- **Anomaly (APE 높음)**: 에너지가 여러 방향으로 분산 → 결함으로 isotropic

### Adaptive Dropout 적용 방안
```
APE ↓ (방향성 강함, 규칙적) → Dropout ↑ (overfit 방지)
APE ↑ (isotropic, 복잡)    → Dropout ↓
```

### 결과 파일
```
examples/hdmap/EDA/results/ape/
├── ape_distributions.png
├── ape_boxplot.png
├── ape_results.json
└── domain_*/
    ├── comparison_*.png
    └── *_process.png
```

---

## Summary Comparison

| Feature | Type | Cohen's d Range | Direction Consistency | Judgment |
|---------|------|-----------------|----------------------|----------|
| OE (Orientation Entropy) | Spatial | 0.1 ~ 0.2 | Yes | **Weak** |
| RSE (Radial Spectral Energy) | Frequency | N/A | N/A | **Not Suitable** |
| SFM (Spectral Flatness) | Frequency | 0.34 ~ 0.49 | **No** (sign flip) | **Weak** |
| **APE (Angular Power Entropy)** | Frequency | **1.73 ~ 4.34** | **Yes** | **Strong** ✅ |

---

## Next Steps

### 1. APE 기반 Adaptive Dropout 구현
```python
# 기존 OE → APE로 교체
def ape_to_dropout_prob(ape, base_dropout=0.3, sensitivity=4.0, normal_ape=0.78):
    # APE가 낮을수록 (더 방향성 강함) → Dropout 높게
    # APE가 높을수록 (더 isotropic) → Dropout 낮게
    deviation = normal_ape - ape  # APE < normal_ape면 positive
    ...
```

### 2. Domain-specific Normal APE 값
| Domain | Normal APE Mean |
|--------|-----------------|
| domain_A | 0.777 |
| domain_B | 0.713 |
| domain_C | 0.866 |
| domain_D | 0.816 |

### 3. 실험 계획
- APE 기반 adaptive dropout 구현
- 기존 fixed dropout과 비교 실험
- Long training (3000+ steps)에서 overfitting 방지 효과 검증

---

## File Structure

```
examples/hdmap/EDA/
├── EDA_SUMMARY.md              # This file
├── sfm_eda.py                  # SFM EDA script
├── sfm_visualization.py        # SFM visualization
├── ape_eda.py                  # APE EDA script
├── ape_visualization.py        # APE visualization
├── sfm_results/                # SFM results (deprecated)
│   ├── sfm_distributions.png
│   ├── sfm_boxplot.png
│   └── domain_*/
└── results/
    └── ape/                    # APE results (recommended)
        ├── ape_distributions.png
        ├── ape_boxplot.png
        ├── ape_results.json
        └── domain_*/
```

---

## References

1. **Orientation Entropy**: Spatial domain gradient-based regularity measure
2. **Spectral Flatness Measure (SFM)**: Audio signal processing metric adapted to 2D
3. **Angular Power Entropy (APE)**: Frequency domain directional energy distribution entropy

---

*Last updated: 2025-12-16*
