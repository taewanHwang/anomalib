# Dinomaly Multi-Class Experiments v2 for HDMAP Dataset

## Overview

이 문서는 Dinomaly의 HDMAP 데이터셋 실험을 체계적으로 기록합니다.
각 Method는 독립적인 모델과 스크립트로 구현되어 있습니다.

## Experiment Environment

- **GPU**: NVIDIA GPU (CUDA 지원)
- **데이터셋**: `/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_tiff_minmax`
- **로그 경로**: `/mnt/ex-disk/taewan.hwang/study/anomalib/logs/`

### 데이터 로딩 주의사항

**반드시 체크리스트 확인 후 실험 진행** (상세: v1 문서 참조)

- [ ] TIFF float32 로딩 (NO clipping)
- [ ] transforms.v2 사용
- [ ] Train-Test 전처리 일치

---

## Evaluation Metrics

각 실험에서 다음 지표들을 측정합니다:

| 지표 | 설명 |
|------|------|
| **AUROC** | Area Under ROC Curve |
| **TPR@FPR=1%** | FPR 1%에서의 True Positive Rate |
| **TPR@FPR=5%** | FPR 5%에서의 True Positive Rate |
| **Precision** | 정밀도 (optimal threshold 기준) |
| **Recall** | 재현율 (optimal threshold 기준) |
| **F1 Score** | F1 점수 |
| **Accuracy** | 정확도 (confusion matrix 기반) |

### 통계적 유의성 검정

Method간 비교 시 **Paired t-test** 수행:
```python
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(baseline_scores, method_scores)
# p < 0.05면 통계적으로 유의미한 차이
```

---

## Method Overview

| Method | 설명 | 모델 위치 | 스크립트 |
|--------|------|----------|----------|
| Baseline | 원본 Dinomaly | `anomalib.models.image.dinomaly` | `dinomaly_baseline.py` |
| Method 1 (GEM) | GEM Pooling | `dinomaly_variants/gem_pooling.py` | `dinomaly_gem.py` |
| Method 3 (Focal) | Focal Loss | `dinomaly_variants/focal_loss.py` | `dinomaly_focal.py` |
| Method 5-A (Aux) | Auxiliary Classifier | `dinomaly_variants/aux_classifier.py` | `dinomaly_aux.py` |
| Method 6-A (Scale) | Scale-wise Weighting | `dinomaly_variants/scale_weighting.py` | `dinomaly_scale.py` |

---

## Experiment 1: Baseline (원본 Dinomaly)

### 실험 설정
- **모델**: 원본 Dinomaly (수정 없음)
- **Seeds**: 42, 43, 44, 123, 456 (5회 반복)
- **결과 폴더**: `results/dinomaly_baseline/`

### 실행 명령어

```bash
# 5 seeds 병렬 실행 (3초 간격)
for seed in 42 43 44 123 456; do
    gpu_id=$((seed % 5))  # GPU 0-4 순환
    nohup python examples/notebooks/dinomaly_baseline.py \
        --mode multiclass \
        --max-steps 10000 \
        --seed $seed \
        --gpu $gpu_id \
        --result-dir results/dinomaly_baseline \
        > logs/baseline_seed${seed}.log 2>&1 &
    sleep 3
done
```

### 결과: AUROC

| Seed | Domain A | Domain B | Domain C | Domain D | Mean |
|------|----------|----------|----------|----------|------|
| 42 | - | - | - | - | - |
| 43 | - | - | - | - | - |
| 44 | - | - | - | - | - |
| 123 | - | - | - | - | - |
| 456 | - | - | - | - | - |
| **Mean±Std** | - | - | - | - | - |

### 결과: TPR@FPR=1%

| Seed | Domain A | Domain B | Domain C | Domain D | Mean |
|------|----------|----------|----------|----------|------|
| 42 | - | - | - | - | - |
| 43 | - | - | - | - | - |
| 44 | - | - | - | - | - |
| 123 | - | - | - | - | - |
| 456 | - | - | - | - | - |
| **Mean±Std** | - | - | - | - | - |

### 결과: TPR@FPR=5%

| Seed | Domain A | Domain B | Domain C | Domain D | Mean |
|------|----------|----------|----------|----------|------|
| 42 | - | - | - | - | - |
| 43 | - | - | - | - | - |
| 44 | - | - | - | - | - |
| 123 | - | - | - | - | - |
| 456 | - | - | - | - | - |
| **Mean±Std** | - | - | - | - | - |

---

## Experiment 2: Method 1 (GEM Pooling)

### 실험 설정
- **모델**: DinomalyGEM
- **변경점**: average pooling → GEM pooling (p=3)
- **Seeds**: 42, 43, 44, 123, 456 (5회 반복)
- **결과 폴더**: `results/dinomaly_gem/`

### 실행 명령어

```bash
# 5 seeds 병렬 실행 (3초 간격)
for seed in 42 43 44 123 456; do
    gpu_id=$((seed % 5))
    nohup python examples/notebooks/dinomaly_gem.py \
        --max-steps 10000 \
        --seed $seed \
        --gpu $gpu_id \
        --gem-p 3.0 \
        --result-dir results/dinomaly_gem \
        > logs/gem_seed${seed}.log 2>&1 &
    sleep 3
done
```

### 결과: AUROC

| Seed | Domain A | Domain B | Domain C | Domain D | Mean |
|------|----------|----------|----------|----------|------|
| 42 | - | - | - | - | - |
| 43 | - | - | - | - | - |
| 44 | - | - | - | - | - |
| 123 | - | - | - | - | - |
| 456 | - | - | - | - | - |
| **Mean±Std** | - | - | - | - | - |

### 결과: TPR@FPR=1%

| Seed | Domain A | Domain B | Domain C | Domain D | Mean |
|------|----------|----------|----------|----------|------|
| 42 | - | - | - | - | - |
| 43 | - | - | - | - | - |
| 44 | - | - | - | - | - |
| 123 | - | - | - | - | - |
| 456 | - | - | - | - | - |
| **Mean±Std** | - | - | - | - | - |

### 통계 검정 (vs Baseline)

| Domain | Δ Mean | p-value | 유의성 |
|--------|--------|---------|--------|
| A | - | - | - |
| B | - | - | - |
| C | - | - | - |
| D | - | - | - |

---

## Summary Table (All Methods)

### AUROC (Mean±Std)

| Method | Domain A | Domain B | Domain C | Domain D | Mean |
|--------|----------|----------|----------|----------|------|
| Baseline | - | - | - | - | - |
| Method 1 (GEM) | - | - | - | - | - |
| Method 3 (Focal) | - | - | - | - | - |
| Method 5-A (Aux) | - | - | - | - | - |
| Method 6-A (Scale) | - | - | - | - | - |

### TPR@FPR=1% (Mean±Std)

| Method | Domain A | Domain B | Domain C | Domain D | Mean |
|--------|----------|----------|----------|----------|------|
| Baseline | - | - | - | - | - |
| Method 1 (GEM) | - | - | - | - | - |
| Method 3 (Focal) | - | - | - | - | - |
| Method 5-A (Aux) | - | - | - | - | - |
| Method 6-A (Scale) | - | - | - | - | - |

### 통계적 유의성 (vs Baseline, p-value)

| Method | Domain A | Domain B | Domain C | Domain D |
|--------|----------|----------|----------|----------|
| Method 1 (GEM) | - | - | - | - |
| Method 3 (Focal) | - | - | - | - |
| Method 5-A (Aux) | - | - | - | - |
| Method 6-A (Scale) | - | - | - | - |

---

## Notes

- 결과 폴더 경로: `results/dinomaly_{method}/YYYYMMDD_HHMMSS_seed{N}/`
- 각 실험 후 결과 폴더명을 위 테이블에 기록할 것
- **통계적 유의성**: p < 0.05면 유의미한 차이로 판단
