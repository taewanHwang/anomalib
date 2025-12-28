# WinCLIP HDMAP 사후 분석: Cold Start vs Warmed Up 조건 분석

**분석 일자**: 2025-12-27

## 배경

WinCLIP HDMAP 실험 결과 분석 중, 테스트 데이터의 anomaly score 분포에서 뚜렷한 패턴이 발견됨.

### 데이터 구조
- 테스트 데이터: 2000개 샘플
  - Fault: 1000개 (index 0-999)
    - Cold Start: 500개 (index 0-499)
    - Warmed Up: 500개 (index 500-999)
  - Normal: 1000개 (index 1000-1999)
    - Cold Start: 500개 (index 0-499 in good folder)
    - Warmed Up: 500개 (index 500-999 in good folder)

### 초기 관찰
Score 시계열 플롯에서 4개의 뚜렷한 영역이 관찰됨:
- 0-500: ~0.6 (Fault Cold)
- 500-1000: ~0.9 (Fault Warm)
- 1000-1500: ~0.4 (Normal Cold)
- 1500-2000: ~0.6 (Normal Warm)

---

## 가설

### 핵심 가설
> **Cold Start 조건에서 진동 진폭(amplitude)이 작아 HDMAP 이미지의 intensity가 낮고, 이로 인해 결함 패턴이 존재하더라도 anomaly score가 낮게 나온다.**

### 세부 가설
1. Cold Start 이미지는 Warmed Up 대비 전반적인 pixel intensity가 낮음
2. WinCLIP은 absolute pixel values에 민감하여 intensity가 낮은 이미지에서 낮은 score 출력
3. 결과적으로 Fault Cold와 Normal Warm의 score가 겹쳐 분류 성능 저하

---

## 검증 결과

### 1. Condition별 AUROC 분석 (Domain B, Zero-shot)

| 비교 조건 | AUROC | 해석 |
|-----------|-------|------|
| All (Mixed) | 81.32% | 기존 보고 성능 |
| **Warm Only** | **92.08%** | 우수한 분별력 |
| Cold Only | 72.33% | 보통 |
| Warm Fault vs Cold Normal | **94.02%** | 최고 분별력 |
| **Cold Fault vs Warm Normal** | **66.86%** | **최악 - 겹침!** |

### 2. 전체 도메인 AUROC 비교 (Zero-shot, Baseline)

| Domain | All | Cold Only | Warm Only | Cold Fault vs Warm Normal | Warm Fault vs Cold Normal |
|--------|-----|-----------|-----------|---------------------------|---------------------------|
| A | 81.93% | 80.46% | 84.49% | 75.04% | 87.73% |
| B | 81.32% | 72.33% | **92.08%** | **66.86%** | **94.02%** |
| C | 73.05% | 72.50% | 75.53% | **64.25%** | 79.93% |
| D | 80.86% | 71.06% | **91.70%** | 68.69% | **91.99%** |
| **Mean** | **79.29%** | **74.09%** | **85.95%** | **68.71%** | **88.42%** |

**핵심 발견**: Cold Fault vs Warm Normal은 모든 도메인에서 최저 AUROC (64-75%)

### 3. 이미지 Intensity 통계 (Domain C)

| Condition | Mean Intensity | Dynamic Range (P95-P5) |
|-----------|----------------|------------------------|
| Fault Cold | 0.216 ± 0.016 | 0.178 ± 0.023 |
| **Fault Warm** | **0.310 ± 0.020** | **0.286 ± 0.021** |
| Normal Cold | 0.190 ± 0.015 | 0.147 ± 0.023 |
| Normal Warm | 0.265 ± 0.017 | 0.237 ± 0.020 |

### 4. Cold vs Warm 비율

| Label | Mean Intensity (Warm/Cold) | Dynamic Range (Warm/Cold) |
|-------|---------------------------|--------------------------|
| **FAULT** | **1.44x** | **1.61x** |
| **NORMAL** | **1.39x** | **1.61x** |

### 5. Anomaly Score와 Image Intensity 상관관계

| 지표 | Pearson r | p-value |
|------|-----------|---------|
| Score vs Mean Intensity | **0.359** | 9.08e-62 |
| Score vs Dynamic Range | **0.336** | 8.51e-54 |

**Condition별 상관관계 (Score vs Mean Intensity):**
| Condition | r | 해석 |
|-----------|---|------|
| normal_cold | **0.411** | 가장 강함 - Cold에서 intensity가 score 결정 |
| fault_cold | 0.225 | 양의 상관 |
| fault_warm | -0.180 | 약한 음의 상관 |
| normal_warm | -0.039 | 무관 |

---

## 시각적 증거

### 이미지 비교 (Domain C, 1-shot)

**Fault Warm (000697, 000568)**:
- 원본 이미지: 밝은 가로 밴드 (horizontal bands) 명확히 보임
- Auto-scale anomaly map: 결함 영역에 빨간 hot spot 검출
- 0-1 scale anomaly map: 중간 정도의 신호

**Fault Cold (000367, 000311)**:
- 원본 이미지: 가로 밴드가 있지만 **대비가 낮음**
- Auto-scale anomaly map: **빨간 hot spot이 여전히 보임!** (패턴 자체는 검출됨)
- 0-1 scale anomaly map: **거의 파란색** (절대적 anomaly score 낮음)

**핵심 관찰**:
> Fault Cold 샘플에서 Auto-scale에서는 결함이 검출되지만, 0-1 fixed scale에서는 신호가 매우 약함.
> 이는 **패턴 자체는 인식했지만 absolute score가 낮음**을 의미.

---

## 결론

### 가설 검증 완료 ✅

1. **Cold Start 이미지는 amplitude가 1.4-1.6배 낮음**
   - 설비 cold start 시 진동 진폭이 작음
   - HDMAP 변환 시 pixel intensity가 낮아짐

2. **WinCLIP은 absolute pixel values에 민감**
   - Anomaly score와 image intensity 간 유의미한 양의 상관관계 (r=0.36)
   - Cold 조건에서 intensity가 score를 더 강하게 결정 (r=0.41)

3. **결과적으로 Fault Cold ≈ Normal Warm 겹침**
   - Fault Cold: 결함 패턴 있지만 intensity 낮음 → score 중간
   - Normal Warm: 결함 없지만 intensity 높음 → score 중간
   - 두 조건의 score 분포가 겹쳐 분류 성능 저하

---

## 개선 방안 제안

### Scale-Awareness 제거/완화 방법

| 방법 | 설명 | 난이도 | 기대 효과 |
|------|------|--------|----------|
| **A. Image Normalization** | 입력 이미지를 per-image min-max 또는 z-score 정규화 | 쉬움 | 높음 |
| **B. Score Normalization** | 후처리로 score를 intensity로 나눔 | 쉬움 | 중간 |
| **C. Contrast Enhancement** | CLAHE 등 대비 향상 전처리 | 중간 | 중간 |
| **D. Multi-scale Augmentation** | 다양한 contrast level로 augment하여 학습 | 어려움 | 높음 |
| **E. Condition-aware Model** | Cold/Warm 상태를 추가 입력으로 제공 | 어려움 | 높음 |

### 권장 순서
1. **A (Per-image Normalization)**: 가장 간단하고 효과적일 가능성 높음
2. **B (Score Normalization)**: 모델 수정 없이 후처리로 가능
3. **C (CLAHE)**: 이미지 전처리 파이프라인에 추가

---

## 개선 방안 A 실험: Per-Image Normalization

**실험 일자**: 2025-12-27

### 실험 목적
Cold/Warm 간 intensity 차이를 per-image normalization으로 제거하여 성능 개선 시도

### 실험 방법

3가지 per-image normalization 방법 테스트:

| 방법 | 수식 | 설명 |
|------|------|------|
| **minmax** | `(x - min) / (max - min)` | 전체 범위를 [0,1]로 정규화 |
| **robust (p5/p95)** | `(x - p5) / (p95 - p5)` | 극단값에 강건한 정규화 |
| **robust_soft (p1/p99)** | `(x - p1) / (p99 - p1)` | 더 부드러운 정규화 (stretch 감소) |

### 실험 결과 (Domain C, Zero-shot)

| Method | All AUROC | Cold Only | Warm Only | Cold F vs Warm N |
|--------|-----------|-----------|-----------|------------------|
| **Baseline** | **73.05%** | **72.50%** | 75.53% | **64.25%** |
| minmax | 69.24% | 56.74% | 81.16% | 58.92% |
| robust (p5/p95) | 66.43% | 44.87% | 85.62% | 53.42% |
| robust_soft (p1/p99) | 71.57% | 50.44% | **91.66%** | 53.08% |

### 실패 원인 분석

#### 1. Noise Amplification 문제

Per-image normalization이 **Cold 이미지에서 noise를 크게 증폭**시킴:

```
Noise 증폭 분석 (robust p5/p95):

Cold 이미지 (idx=327):
  Original noise: 0.0439
  Normalized noise: 0.2415
  Amplification: 5.51x  ← 크게 증폭!

Warm 이미지 (idx=625):
  Original noise: 0.0885
  Normalized noise: 0.2411
  Amplification: 2.72x

결론: Cold는 Warm보다 2.02배 더 noise가 증폭됨
```

#### 2. 원인 메커니즘

```
Cold 이미지 특성:
- Dynamic range가 작음 (p95-p5 ≈ 0.15)
- [0.15 range] → [1.0 range] = 6.7배 stretch
- Fault 패턴과 noise가 함께 증폭
- Signal-to-Noise Ratio (SNR) 저하
- CLIP이 noise를 texture로 오인식

Warm 이미지:
- Dynamic range가 큼 (p95-p5 ≈ 0.29)
- Stretch가 상대적으로 적음 (3.4배)
- SNR 유지, 성능 향상
```

#### 3. WinCLIP 전처리와의 상호작용

WinCLIP은 CLIP 전역 정규화를 사용:
```python
Normalize(mean=(0.48, 0.46, 0.41), std=(0.27, 0.26, 0.28))
```

이 전역 정규화 자체는 문제가 아님 (다른 데이터셋에서도 잘 작동).
문제는 **per-image normalization이 local contrast/noise를 변화**시키는 것.

### 시각적 증거

#### Normalization 효과 비교
`examples/hdmap/EDA/HDMAP_vis/results/`

| 파일 | 설명 |
|------|------|
| `normalization_effect_domainC_fault_*.png` | Fault 이미지: Original vs MinMax vs Robust |
| `normalization_effect_domainC_good_*.png` | Good 이미지: Original vs MinMax vs Robust |

**시각화 구성**:
- Row 1: Original (Cold 어둡고, Warm 밝음)
- Row 2: MinMax normalized (비율 역전됨)
- Row 3: Robust normalized (intensity 동일해짐)

#### Noise Amplification 분석
| 파일 | 설명 |
|------|------|
| `noise_amplification_domainC_*.png` | Cold/Warm noise 증폭 비교 |

**시각화 구성**:
- Row 1: 원본 vs 정규화 이미지
- Row 2: Local noise map (Cold에서 더 빨갛게 = noise 많음)
- Row 3: Histogram 분포
- Row 4: Line profile (Cold 정규화 후 진폭 크게 증가)

### 결론

❌ **Per-image Normalization은 부적합**

| 장점 | 단점 |
|------|------|
| Warm Only AUROC 향상 (75% → 92%) | Cold Only AUROC 급락 (72% → 50%) |
| Cold/Warm intensity 차이 제거 | Noise 증폭으로 SNR 저하 |
| | Cold Fault vs Warm Normal 개선 안됨 |

### 다음 단계

**방법 B (Score 후처리 정규화)** 시도 권장:
- 원본 이미지 그대로 사용 (noise 증폭 없음)
- CLIP이 정상적으로 feature 추출
- 출력 score만 intensity로 보정

```python
# 예시
normalized_score = score / image_mean_intensity
# 또는
normalized_score = score - alpha * image_mean_intensity
```

---

## 분석 스크립트

분석에 사용된 스크립트들:

```
examples/hdmap/EDA/winclip_baseline_post_analysis/
├── analyze_cold_warm.py          # Cold/Warm 조건별 AUROC 분석
├── analyze_image_intensity.py    # 이미지 intensity 통계 분석
├── analyze_score_vs_intensity.py # Score-Intensity 상관관계 분석
└── domain_C_*.png                # 시각화 결과
```

### 사용 예시

```bash
# Cold/Warm 조건별 분석
python examples/hdmap/EDA/winclip_baseline_post_analysis/analyze_cold_warm.py \
    --csv results/winclip_hdmap_baseline/20251226_141441/scores/domain_B_zero_shot_scores.csv

# 이미지 intensity 분석
python examples/hdmap/EDA/winclip_baseline_post_analysis/analyze_image_intensity.py \
    --domain domain_C

# Score-Intensity 상관관계 분석
python examples/hdmap/EDA/winclip_baseline_post_analysis/analyze_score_vs_intensity.py \
    --score-csv results/.../domain_C_zero_shot_scores.csv \
    --intensity-csv examples/.../domain_C_intensity_stats.csv
```

---

## 시각화 파일 목록

분석 결과 시각화 파일 경로 (`examples/hdmap/EDA/winclip_baseline_post_analysis/`):

### Score by Condition (4-panel: scatter, boxplot, histogram)
- `domain_A_zero_shot_scores_score_by_condition.png`
- `domain_B_zero_shot_scores_score_by_condition.png`
- `domain_C_zero_shot_scores_score_by_condition.png`
- `domain_D_zero_shot_scores_score_by_condition.png`

### Cross Condition Analysis (Cold Fault vs Warm Normal 겹침)
- `domain_A_zero_shot_scores_cross_condition.png`
- `domain_B_zero_shot_scores_cross_condition.png`
- `domain_C_zero_shot_scores_cross_condition.png`
- `domain_D_zero_shot_scores_cross_condition.png`

### ROC Curve Comparison (All/Cold/Warm/Cross 비교)
- `domain_A_zero_shot_scores_roc_comparison.png`
- `domain_B_zero_shot_scores_roc_comparison.png`
- `domain_C_zero_shot_scores_roc_comparison.png`
- `domain_D_zero_shot_scores_roc_comparison.png`

### Image Intensity Analysis
- `domain_C_intensity_by_condition.png` - Condition별 이미지 intensity 분포

### Score vs Intensity Correlation
- `domain_C_zero_shot_scores_vs_intensity.png` - Anomaly score와 image intensity 상관관계

### CSV 데이터
- `domain_C_intensity_stats.csv` - 이미지별 intensity 통계

---

## Per-Image Normalization 시각화 파일

분석 결과 시각화 파일 경로 (`examples/hdmap/EDA/HDMAP_vis/results/`):

### Normalization Effect Comparison
- `normalization_effect_domainC_fault_20251227_052502.png` - Fault 이미지 Original/MinMax/Robust 비교
- `normalization_effect_domainC_good_20251227_052504.png` - Good 이미지 Original/MinMax/Robust 비교

### Noise Amplification Analysis
- `noise_amplification_domainC_20251227_054330.png` - Cold/Warm noise 증폭 비교

### Per-Image Normalization 실험 결과
실험 결과 파일 경로 (`results/winclip_hdmap_normalized/`):

| 디렉토리 | 내용 |
|----------|------|
| `20251227_051036_minmax/` | MinMax normalization 결과 |
| `20251227_051223_robust/` | Robust (p5/p95) normalization 결과 |
| `20251227_054651_robust_soft/` | Robust Soft (p1/p99) normalization 결과 |

각 디렉토리 구조:
```
results/winclip_hdmap_normalized/YYYYMMDD_HHMMSS_method/
├── experiment_settings.json    # 실험 설정
├── summary.json                # 결과 요약
├── scores/
│   └── domain_C_zero_shot_scores.csv  # 예측 score CSV
└── visualizations/
    └── domain_C/
        └── score_distribution.png     # Score 분포 시각화
```

---

## 참고

- WinCLIP baseline 실험 결과: `examples/notebooks/09_winclip_variant/winclip_hdmap.md`
- 분석 시각화 결과: `examples/hdmap/EDA/winclip_baseline_post_analysis/`
