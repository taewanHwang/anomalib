# Feature-based Gating Analysis for CA-WinCLIP

## 1. 배경 및 목적

### 1.1 문제 정의
CA-WinCLIP (Condition-Aware WinCLIP)은 Cold/Warm 조건별로 별도 reference bank를 구성하여 anomaly detection 성능을 개선하는 모델이다. 이를 위해 test 이미지가 Cold인지 Warm인지 판단하는 **Gating 메커니즘**이 필요하다.

### 1.2 기존 접근법: CLIP 기반 Gating
- Global embedding similarity: 87.5%
- Confidence-based (highest margin): 88.8%
- **문제**: fault/cold에서 60%, good/warm에서 40-70%로 불안정

### 1.3 목표
- 단순하면서 높은 정확도의 Gating 방법 탐색
- 정상 데이터만으로 threshold 설정 가능한 방법

---

## 2. 분석 방법론

### 2.1 테스트 환경
- **데이터셋**: HDMAP (4개 도메인: A, B, C, D)
- **각 도메인 구성**: 2000개 test 이미지
  - fault/cold: indices 0-499
  - fault/warm: indices 500-999
  - good/cold: indices 1000-1499
  - good/warm: indices 1500-1999

### 2.2 분석한 Gating 방법들

| 카테고리 | 방법 | 설명 |
|----------|------|------|
| CLIP 기반 | Global Embedding | CLIP global feature 유사도 |
| CLIP 기반 | Patch Mean/Median | CLIP patch feature 유사도 평균/중앙값 |
| CLIP 기반 | Confidence | 가장 높은 margin을 가진 방법 선택 |
| FFT 기반 | Cosine/Correlation | 2D FFT magnitude 유사도 |
| FFT 기반 | Radial/Band | FFT radial profile, H/V band 유사도 |
| **Intensity 기반** | **mean, median, p10-p90** | **이미지 픽셀 intensity 통계** |

---

## 3. FFT 기반 Gating 결과

### 3.1 Domain C 결과

| Method | fault/cold | fault/warm | good/cold | good/warm | Overall |
|--------|------------|------------|-----------|-----------|---------|
| cosine | 12.0% | 100.0% | 46.9% | 100.0% | 64.8% |
| correlation | 98.0% | 4.0% | 98.0% | 18.0% | 54.3% |
| radial_cosine | 12.0% | 0.0% | 67.3% | 62.0% | 35.2% |
| band_cosine | 74.0% | 0.0% | 73.5% | 62.0% | 52.3% |

### 3.2 결론
- **FFT 기반 Gating은 효과 없음** (최대 64.8%)
- CLIP 기반 (88.8%)보다 훨씬 낮음
- 주파수 도메인 특성만으로는 Cold/Warm 구분 어려움

---

## 4. Intensity 기반 Gating 결과

### 4.1 테스트한 Features

| Feature | 설명 |
|---------|------|
| mean | 전체 픽셀 평균 |
| median | 전체 픽셀 중앙값 |
| std | 전체 픽셀 표준편차 |
| min/max | 최소/최대 픽셀 값 |
| p10, p25, p75, p90 | 10%, 25%, 75%, 90% percentile |
| range | max - min |
| iqr | p75 - p25 (Interquartile Range) |

### 4.2 전체 도메인 결과 (Overall Accuracy)

| Rank | Feature | Domain A | Domain B | Domain C | Domain D | 평균 |
|------|---------|----------|----------|----------|----------|------|
| 1 | **p90** | 100.0% | 100.0% | **97.9%** | 100.0% | **99.5%** |
| 2 | p75 | 100.0% | 100.0% | 97.6% | 100.0% | 99.4% |
| 3 | mean | 100.0% | 100.0% | 97.4% | 99.9% | 99.3% |
| 4 | median | 100.0% | 100.0% | 97.2% | 99.9% | 99.3% |
| 5 | std | 100.0% | 100.0% | 96.3% | 99.1% | 98.9% |
| 6 | p25 | 100.0% | 100.0% | 95.8% | 99.5% | 98.8% |
| 7 | iqr | 100.0% | 100.0% | 95.8% | 95.5% | 97.8% |
| 8 | p10 | 100.0% | 100.0% | 95.3% | 98.0% | 98.3% |
| 9 | max | 100.0% | 100.0% | 88.1% | 87.1% | 93.8% |
| 10 | range | 100.0% | 99.8% | 86.5% | 85.4% | 92.9% |
| 11 | min | 100.0% | 99.6% | 86.5% | 89.8% | 94.0% |

### 4.3 핵심 발견

1. **p90이 가장 안정적**: 모든 도메인에서 97.9% 이상
2. **상위 percentile이 효과적**: p90 > p75 > mean > median
3. **min, max, range는 노이즈에 취약**
4. **Domain C가 가장 어려움**: Cold/Warm 간 intensity gap이 가장 작음

### 4.4 vs CLIP Gating 비교

| Method | Domain A | Domain B | Domain C | Domain D | 평균 |
|--------|----------|----------|----------|----------|------|
| **p90 Intensity** | **100.0%** | **100.0%** | **97.9%** | **100.0%** | **99.5%** |
| CLIP Confidence | 88.8% | 88.8% | 88.8% | 88.8% | 88.8% |
| CLIP Global | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% |

**p90 Intensity가 CLIP 대비 +10.7%p 개선!**

---

## 5. Threshold 설정 전략

### 5.1 가정
- 정상(good) 데이터에 Cold/Warm 라벨이 있음
- Threshold는 정상 데이터만으로 설정

### 5.2 비교한 전략

| 전략 | 수식 | 설명 |
|------|------|------|
| max(good/cold) | `max(p90 of good/cold)` | Cold 정상의 최대값 |
| mean+3std | `mean(good/cold) + 3*std(good/cold)` | Cold 정상의 3-sigma |
| **midpoint** | `(max(good/cold) + min(good/warm)) / 2` | **Cold max와 Warm min의 중간** |

### 5.3 Threshold 값 (p90 기준)

| Domain | max(g/c) | min(g/w) | midpoint | mean+3std |
|--------|----------|----------|----------|-----------|
| A | 0.2323 | 0.3648 | **0.2985** | 0.2542 |
| B | 0.2542 | 0.3714 | **0.3128** | 0.2811 |
| C | 0.3007 | 0.3172 | **0.3089** | 0.3199 |
| D | 0.2784 | 0.3054 | **0.2919** | 0.2997 |

### 5.4 Threshold 전략별 Gating 정확도

#### Overall Accuracy

| Domain | midpoint | mean+3std | Winner |
|--------|----------|-----------|--------|
| A | **100.0%** | 95.3% | midpoint |
| B | **100.0%** | 100.0% | tie |
| C | 94.3% | **96.9%** | mean+3std |
| D | **92.5%** | 89.8% | midpoint |
| **평균** | **96.7%** | 95.5% | **midpoint** |

#### Per-Group Accuracy (midpoint)

| Domain | fault/cold | fault/warm | good/cold | good/warm | Overall |
|--------|------------|------------|-----------|-----------|---------|
| A | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| B | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| C | 77.2% | 100.0% | 100.0% | 100.0% | 94.3% |
| D | 100.0% | 69.8% | 100.0% | 100.0% | 92.5% |

#### Per-Group Accuracy (mean+3std)

| Domain | fault/cold | fault/warm | good/cold | good/warm | Overall |
|--------|------------|------------|-----------|-----------|---------|
| A | 81.2% | 100.0% | 100.0% | 100.0% | 95.3% |
| B | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| C | 88.4% | 100.0% | 100.0% | 99.2% | 96.9% |
| D | 100.0% | 59.0% | 100.0% | 100.0% | 89.8% |

### 5.5 Threshold 전략 결론

| 지표 | 추천 전략 | 이유 |
|------|-----------|------|
| **Overall 평균** | **midpoint** | 96.7% vs 95.5% |
| **fault/cold 평균** | **midpoint** | 94.3% vs 92.4% |
| Domain C 단독 | mean+3std | 96.9% vs 94.3% |

---

## 6. 최종 권장 Gating 알고리즘

### 6.1 선택된 방법
```
Feature: p90 (90th percentile of image pixels)
Threshold: midpoint = (max(good/cold p90) + min(good/warm p90)) / 2
Decision: if image_p90 <= threshold → Cold, else → Warm
```

### 6.2 Gating 로직
```python
import numpy as np

def compute_p90(image: np.ndarray) -> float:
    """Compute 90th percentile of image pixels."""
    return np.percentile(image, 90)

def compute_threshold(good_cold_images: List, good_warm_images: List) -> float:
    """Compute midpoint threshold from normal images."""
    cold_p90_max = max(compute_p90(img) for img in good_cold_images)
    warm_p90_min = min(compute_p90(img) for img in good_warm_images)
    return (cold_p90_max + warm_p90_min) / 2

def select_condition(image: np.ndarray, threshold: float) -> str:
    """Select cold or warm condition based on p90."""
    p90 = compute_p90(image)
    return "cold" if p90 <= threshold else "warm"
```

### 6.3 도메인별 권장 Threshold

| Domain | Threshold (midpoint) |
|--------|---------------------|
| A | 0.2985 |
| B | 0.3128 |
| C | 0.3089 |
| D | 0.2919 |

### 6.4 기대 성능

| Metric | 값 |
|--------|-----|
| Overall Gating Accuracy | **96.7%** (평균) |
| vs CLIP Confidence | **+7.9%p** |
| 구현 복잡도 | 매우 단순 (CLIP 불필요) |
| 추론 속도 | 매우 빠름 (단순 percentile 계산) |

---

## 7. 한계 및 추가 고려사항

### 7.1 한계
1. **Domain C의 fault/cold**: 77.2%로 다소 낮음
   - Fault 이미지의 p90이 normal보다 높아서 Warm으로 오분류
   - 결함이 밝은 영역을 생성하기 때문

2. **Domain D의 fault/warm**: 69.8%로 낮음
   - Gap이 좁아서 일부 fault/warm이 Cold로 오분류

### 7.2 개선 방향
1. **도메인별 threshold 최적화**: 각 도메인에 맞는 threshold 사용
2. **Hybrid 접근**: Intensity + CLIP 결합
3. **Adaptive threshold**: 데이터 분포에 따라 동적 조정

---

## 8. 관련 파일

### 8.1 분석 스크립트
- `eda_intensity_gating.py`: Mean intensity gating 분석
- `eda_intensity_features_gating.py`: 모든 intensity feature 비교
- `eda_fft_gating.py`: FFT 기반 gating 분석
- `eda_p90_threshold_analysis.py`: Threshold 전략 분석
- `eda_p90_threshold_comparison.py`: midpoint vs mean+3std 비교

### 8.2 결과 디렉토리
- `domain_*/intensity_features/`: 도메인별 intensity feature 결과
- `domain_*/fft_gating/`: 도메인별 FFT gating 결과
- `domain_*/intensity_gating/`: 도메인별 intensity gating 결과

---

## 9. 결론

1. **p90 + midpoint threshold**가 가장 효과적인 Gating 방법
2. CLIP 기반 대비 **+7.9%p 개선** (96.7% vs 88.8%)
3. **구현이 매우 단순**하고 CLIP 모델 불필요
4. 정상 데이터의 Cold/Warm 라벨만으로 threshold 설정 가능

---

*분석 날짜: 2025-12-27*
