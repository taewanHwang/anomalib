# Dinomaly Multi-Class Experiments v2 for HDMAP Dataset

## Overview

이 문서는 Dinomaly의 HDMAP 데이터셋 실험을 체계적으로 기록합니다.
각 Method는 독립적인 모델과 스크립트로 구현되어 있습니다.

## Experiment Environment

- **GPU**: NVIDIA GPU (CUDA 지원)
- **데이터셋**: `/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_tiff_minmax`
- **로그 경로**: `/mnt/ex-disk/taewan.hwang/study/anomalib/logs/`
- **이미지 크기**: 448 → CenterCrop 392

### 데이터 로딩 정책 (통합됨 ✅)

> **중요**: Training과 Testing 모두 동일한 `HDMAPDataset`을 사용하여 **일관된 TIFF 로딩**을 보장합니다.

#### Training & Testing (통합 방식)

| 항목 | 설정 |
|------|------|
| **데이터 모듈** | `AllDomainsHDMAPDataModule` (4개 도메인 통합 훈련) |
| **데이터셋** | `HDMAPDataset` (anomalib 내부, Training과 Testing 동일) |
| **이미지 로딩** | `tifffile.imread()` → float32 (NO clipping) |
| **Transforms** | anomalib PreProcessor (내부 처리) |
| **정규화** | ImageNet 표준 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) |

```python
# Training: AllDomainsHDMAPDataModule 사용
from anomalib.data.datamodules.image.all_domains_hdmap import AllDomainsHDMAPDataModule

datamodule = AllDomainsHDMAPDataModule(
    root=data_root,
    domains=["domain_A", "domain_B", "domain_C", "domain_D"],
    train_batch_size=16,
    eval_batch_size=16,
    val_split_mode="from_test",
    val_split_ratio=0.1,
)
```

#### HDMAPDataset TIFF 로딩 방식

```python
# HDMAPDataset.load_and_resize_image() 내부 구현
# TIFF 파일: tifffile 사용 (float32 정밀도 유지)
if image_path.lower().endswith(('.tiff', '.tif')):
    img_array = tifffile.imread(image_path).astype(np.float32)  # NO clipping
else:
    # PNG 등 기타 파일은 PIL 사용
    with Image.open(image_path) as img:
        img_array = np.array(img).astype(np.float32)
```

#### Per-Domain Evaluation

```python
# HDMAPDataset을 사용하여 동일한 로딩 방식 보장
from anomalib.data.datasets.image.hdmap import HDMAPDataset

test_dataset = HDMAPDataset(
    root=data_root,
    domain="domain_A",
    split="test",
    target_size=(448, 448),
)
```

### 체크리스트

#### 데이터 로딩
- [x] TIFF float32 로딩 (NO clipping) - `tifffile.imread()` 사용
- [x] transforms.v2 사용 - `torchvision.transforms.v2`
- [x] **Train-Test 전처리 완전 일치** - `HDMAPDataset` 통합 사용 ✅
- [x] **데이터 로딩 검증 로깅** - 학습/추론 시 값 범위 확인 (HDMAPDataset에 추가됨)

#### 학습 안정성
- [x] GPU 기반 Per-Domain 평가 - `torch.amp.autocast('cuda')` 사용
- [x] **Gradient Monitoring** - TensorBoard에 `grad/total_norm`, `grad/nan_count` 로깅
- [x] **NaN Loss 감지** - training_step에서 NaN 발생 시 경고 로깅
- [ ] ~~**Early Stopping**~~ - 성능 변화 관찰을 위해 일단 사용 안함

#### Lessons Learned (2024-12-24)

| 문제 | 원인 | 해결책 |
|------|------|--------|
| **Step 3000에서 NaN 발생** | Gradient explosion 또는 학습 불안정 | Gradient monitoring 추가, max_steps 감소 |
| **AUROC 감소 (Step 1000→3000)** | 과적합 (HDMAP 다양성 < MVTec) | max_steps=1500~2000 권장 |
| **TPR@FPR=0%** | NaN으로 ROC curve 계산 실패 | NaN 발생 시 해당 도메인 스킵 또는 경고 |
| **Baseline=GEM 동일 결과** | 100 steps로는 차이 안 나타남 | 충분한 학습 후 비교 필요 |
| **Per-Domain AUROC 불일치** | HDMAPDataset의 `target_size` 설정으로 인한 보간 방법 차이 | `target_size=None` 사용 (아래 상세 설명) |

##### Per-Domain 평가 버그 수정 (2024-12-24)

**증상**: Engine.test() AUROC = 98.61%, Per-Domain Mean = 41.76%

**근본 원인**: 이미지 리사이즈 보간 방법 불일치
- Training: Raw TIFF (31x95) → **PreProcessor bilinear** resize to 448 → CenterCrop to 392
- Per-Domain 평가 (버그): HDMAPDataset `target_size=(448, 448)` → **nearest neighbor** resize → PreProcessor (no-op)

**nearest neighbor vs bilinear** 보간은 31x95 → 448x448 업스케일링 시 완전히 다른 픽셀 값을 생성하여 다른 anomaly score 분포를 유발.

**해결책**: Per-domain 평가에서 `target_size=None` 사용
```python
# CORRECT: Let PreProcessor handle resize (same as training)
test_dataset = HDMAPDataset(root=data_root, domain=domain, split="test", target_size=None)

# WRONG: Different interpolation method than training
test_dataset = HDMAPDataset(root=data_root, domain=domain, split="test", target_size=(448, 448))
```

**수정 후 결과**:
- domain_A: 99.08%, domain_B: 99.11%, domain_C: 97.63%, domain_D: 98.23%
- Per-Domain Mean: 98.51% (Engine.test() 98.59%와 일치!)

#### 원본 Dinomaly 학습 조건 (MVTec)
```
total_iters = 10000      # MVTec 15개 카테고리
batch_size = 16
lr = 2e-3 → 2e-4         # WarmCosineScheduler
warmup_iters = 100
gradient_clip = 0.1
evaluation_interval = 5000
```

> **HDMAP 권장 설정**: MVTec(15개)보다 다양성이 낮으므로 (4개 도메인)
> - `max_steps = 1500~2000` (원본 10000의 15~20%)
> - `val_check_interval = 500` (더 자주 검증)

> **Note**: `Folder` 데이터모듈 대신 `AllDomainsHDMAPDataModule`을 사용하여
> Training과 Testing에서 **동일한 HDMAPDataset**을 사용합니다.
> 이로써 TIFF float32 로딩이 완전히 일치합니다.

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
start_gpu=0  # ← 시작 GPU 번호 수정
gpu_id=$start_gpu
for seed in 42 43 44 123 456; do
    nohup python examples/notebooks/dinomaly_baseline.py \
        --mode multiclass \
        --max-steps 3000 \
        --seed $seed \
        --gpu $gpu_id \
        --result-dir results/dinomaly_baseline \
        > logs/baseline_seed${seed}.log 2>&1 &
    gpu_id=$((gpu_id + 1))
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
- **변경점**:
  - Training: CosineHardMiningGEMLoss (scale별 distance를 GEM으로 aggregate 후 hard mining)
  - Inference: GEM pooling (p=3)으로 anomaly map aggregation
- **gem_p**: 3.0 (GEM power parameter)
- **gem_factor**: 0.3 (easy point gradient 감소 비율)
- **Seeds**: 42, 43, 44, 123, 456 (5회 반복)
- **결과 폴더**: `results/dinomaly_gem/`

### 실행 명령어

```bash
# 5 seeds 병렬 실행 (3초 간격)
start_gpu=5  # ← 시작 GPU 번호 수정
gpu_id=$start_gpu
for seed in 42 43 44 123 456; do
    nohup python examples/notebooks/dinomaly_gem.py \
        --max-steps 3000 \
        --seed $seed \
        --gpu $gpu_id \
        --gem-p 3.0 \
        --gem-factor 0.3 \
        --result-dir results/dinomaly_gem \
        > logs/gem_seed${seed}.log 2>&1 &
    gpu_id=$((gpu_id + 1))
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
