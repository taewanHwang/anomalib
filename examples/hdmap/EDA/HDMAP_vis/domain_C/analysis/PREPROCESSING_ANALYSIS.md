# Preprocessing Analysis for HDMAP domain_C Cold Anomaly Detection

## Overview

CA-WinCLIP Oracle 실험에서 Cold-only AUROC가 ~77%로 낮은 원인을 분석하기 위해,
fault/cold vs good/cold 이미지 간 차이를 극대화하는 전처리 방법을 탐색함.

## 분석 조건

- **Domain**: domain_C
- **샘플 수**: fault/cold 50개, good/cold 50개
- **분석 대상 전처리**: 8가지 방법
- **평가 지표**: Cohen's d (효과 크기), t-test p-value, Overlap 비율

## 결과

### 전처리 방법 비교 (Cohen's d 순위)

| 순위 | 방법 | Cohen's d | p-value | Overlap | 해석 |
|------|------|-----------|---------|---------|------|
| 1 | **CLAHE** | **1.4429** | 1.63e-10 | 0.3310 | 강한 효과 |
| 2 | Original | 1.2489 | 1.46e-08 | 0.3761 | 강한 효과 |
| 3 | Row_Diff | 0.4217 | 3.95e-02 | 0.8505 | 약한 효과 |
| 4 | Sobel_V | 0.3796 | 6.32e-02 | 0.8561 | 미미한 효과 |
| 5 | Horizontal_HP | 0.3026 | 0.137 | 0.8290 | 미미한 효과 |
| 6 | Sobel_H | 0.3026 | 0.137 | 0.8290 | 미미한 효과 |
| 7 | Contrast | 0.1140 | 0.574 | 0.8231 | 효과 없음 |
| 8 | Laplacian | 0.0642 | 0.751 | 0.8589 | 효과 없음 |

### Cohen's d 해석 기준
- |d| < 0.2: 효과 없음
- 0.2 <= |d| < 0.5: 작은 효과
- 0.5 <= |d| < 0.8: 중간 효과
- |d| >= 0.8: 큰 효과

## 핵심 발견

### 1. CLAHE가 가장 효과적
- Cohen's d = 1.44 (강한 효과)
- Overlap = 33.1% (두 그룹이 잘 분리됨)
- Adaptive histogram equalization이 미세한 intensity 차이를 강조

### 2. Edge Detection 방법들은 역효과
- Sobel, Laplacian, Horizontal High-Pass 모두 낮은 효과
- fault/good 모두 유사한 edge 패턴을 가짐
- Edge만 추출하면 오히려 구분이 어려워짐

### 3. 원본 이미지도 상당한 분리도
- Cohen's d = 1.25 (강한 효과)
- **문제**: WinCLIP이 이 차이를 포착하지 못함
- 원본에서도 차이가 있지만, CLIP visual encoder가 미세한 차이를 놓침

### 4. Contrast Stretching은 효과 없음
- 단순 contrast 강화로는 부족
- Adaptive (지역적) 접근이 필요 → CLAHE

## FFT 분석 결과

2D FFT를 통해 주파수 도메인에서 fault/cold vs good/cold 비교:

- **Horizontal band 차이**: 0.1604
- **Vertical band 차이**: 0.1942

주파수 도메인에서도 두 그룹 간 차이가 존재함을 확인.

## 시사점

### WinCLIP 개선 방향

1. **CLAHE 전처리 적용**
   - WinCLIP 입력 전에 CLAHE 적용
   - Cohen's d: 1.25 → 1.44 (+15% 개선)
   - Overlap: 37.6% → 33.1% 감소

2. **FFT 기반 Gating 가능성**
   - 주파수 도메인 특성으로 Cold/Warm 구분
   - Global embedding 대신 FFT 특성 사용

### 한계

- CLAHE가 WinCLIP AUROC를 실제로 개선하는지는 실험 필요
- FFT 기반 gating의 효과도 별도 검증 필요

## 생성된 파일

- `preprocessing_comparison_metrics.png`: 전처리 방법별 메트릭 비교
- `preprocessing_top_distributions.png`: 상위 방법들의 분포 히스토그램
- `fft_analysis.png`: FFT 스펙트럼 분석
- `fft_individual_comparison.png`: 개별 샘플 FFT 비교

## 날짜

2025-12-27
