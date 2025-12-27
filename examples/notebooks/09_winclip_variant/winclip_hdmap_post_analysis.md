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

## 참고

- WinCLIP baseline 실험 결과: `examples/notebooks/09_winclip_variant/winclip_hdmap.md`
- 분석 시각화 결과: `examples/hdmap/EDA/winclip_baseline_post_analysis/`
