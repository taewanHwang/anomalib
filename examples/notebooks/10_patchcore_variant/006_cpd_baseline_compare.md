# 006. CPD-PatchCore Baseline 비교 (Train Reference)

## 개요

`--ref-source train` 옵션을 사용하여 **train/good 데이터**에서 메모리 뱅크를 구축한 실험 결과.
기존 `anomaly_trainer.py` 실험과 공정한 비교를 위한 설정.

---

## 실험 설정

### 공통 설정

| 항목 | 값 |
|------|-----|
| Reference source | **train/good** (k_ref=1000) |
| Test data | test/good + test/fault = 2000 per domain |
| Validation split | 1% from test (20 samples) |
| Actual test used | **1980** per domain |
| Layers | blocks.8 (1-layer) |
| Target size | 518x518 |
| Num neighbors | 9 |

### 실험 조건 비교

| 설정 | Vanilla PatchCore | exp23 재현 |
|------|-------------------|------------|
| Backbone | vit_base_patch14_dinov2 (768 dim) | vit_small_patch14_dinov2 (384 dim) |
| Coreset ratio | 1.0 (100%) | 0.01 (1%) |
| Resize method | resize_bilinear | resize |
| Feature mode | f_only (CPD OFF) | f_only (CPD OFF) |

---

## Baseline 결과 (CPD OFF)

### Overall Accuracy

| Domain | Vanilla PatchCore | exp23 재현 | 차이 |
|--------|-------------------|------------|------|
| A | **98.89%** | 93.74% | +5.15%p |
| B | **98.38%** | 97.17% | +1.21%p |
| C | **93.43%** | 76.26% | +17.17%p |
| D | **97.78%** | 95.76% | +2.02%p |
| **평균** | **97.12%** | 90.73% | **+6.39%p** |

### Overall AUROC

| Domain | Vanilla PatchCore | exp23 재현 | 차이 |
|--------|-------------------|------------|------|
| A | **99.83%** | 98.54% | +1.29%p |
| B | 99.78% | 99.72% | +0.06%p |
| C | **97.89%** | 83.14% | +14.75%p |
| D | **99.58%** | 99.30% | +0.28%p |
| **평균** | **99.27%** | 95.18% | **+4.09%p** |

### Cold/Warm 세부 결과

#### Vanilla PatchCore (vit_base, coreset 100%)

| Domain | Overall | Cold | Warm | AUROC |
|--------|---------|------|------|-------|
| A | 98.89% | 98.58% | 99.20% | 99.83% |
| B | 98.38% | 97.87% | 98.89% | 99.78% |
| C | 93.43% | 89.55% | 97.28% | 97.89% |
| D | 97.78% | 97.36% | 98.19% | 99.58% |
| **평균** | **97.12%** | 95.84% | 98.39% | **99.27%** |

#### exp23 재현 (vit_small, coreset 1%)

| Domain | Overall | Cold | Warm | AUROC |
|--------|---------|------|------|-------|
| A | 93.74% | 89.15% | 98.29% | 98.54% |
| B | 97.17% | 98.48% | 95.88% | 99.72% |
| C | 76.26% | 53.55% | 98.79% | 83.14% |
| D | 95.76% | 92.09% | 99.40% | 99.30% |
| **평균** | **90.73%** | 83.32% | 98.09% | **95.18%** |

---

## CPD 적용 결과

### Overall Accuracy 비교

| Domain | exp23 Baseline | exp23 + CPD f_ctx | exp23 + CPD Multi-k | Multi-k 개선 |
|--------|----------------|-------------------|---------------------|--------------|
| A | 93.74% | 94.55% | **94.75%** | **+1.01%p** |
| B | 97.17% | 97.98% | **98.48%** | **+1.31%p** |
| C | 76.26% | 78.03% | **79.80%** | **+3.54%p** |
| D | 95.76% | 96.77% | **97.07%** | **+1.31%p** |
| **평균** | **90.73%** | 91.83% | **92.53%** | **+1.80%p** |

### Overall AUROC 비교

| Domain | exp23 Baseline | exp23 + CPD f_ctx | exp23 + CPD Multi-k | Multi-k 개선 |
|--------|----------------|-------------------|---------------------|--------------|
| A | 98.54% | 98.76% | **98.79%** | **+0.25%p** |
| B | 99.72% | 99.87% | **99.92%** | **+0.20%p** |
| C | 83.14% | 86.46% | **87.94%** | **+4.80%p** |
| D | 99.30% | 99.62% | **99.73%** | **+0.43%p** |
| **평균** | **95.18%** | 96.18% | **96.60%** | **+1.42%p** |

### Cold/Warm 세부 결과

#### exp23 + CPD f_ctx (k=2)

| Domain | Overall | Cold | Warm | AUROC |
|--------|---------|------|------|-------|
| A | 94.55% | 90.26% | 98.79% | 98.76% |
| B | 97.98% | 98.99% | 96.98% | 99.87% |
| C | 78.03% | 58.11% | 97.79% | 86.46% |
| D | 96.77% | 94.12% | 99.40% | 99.62% |
| **평균** | **91.83%** | 85.37% | 98.24% | **96.18%** |

#### exp23 + CPD Multi-k concat (k=2,3)

| Domain | Overall | Cold | Warm | AUROC |
|--------|---------|------|------|-------|
| A | 94.75% | 91.08% | 98.39% | 98.79% |
| B | 98.48% | 97.46% | 99.50% | 99.92% |
| C | 79.80% | 61.66% | 97.79% | 87.94% |
| D | 97.07% | 94.62% | 99.50% | 99.73% |
| **평균** | **92.53%** | 86.21% | 98.80% | **96.60%** |

### CPD 효과 요약 (exp23 Baseline 대비)

| 지표 | CPD f_ctx | CPD Multi-k |
|------|-----------|-------------|
| Overall Accuracy | +1.10%p | **+1.80%p** |
| Overall AUROC | +1.00%p | **+1.42%p** |
| Cold Accuracy | +2.05%p | **+2.89%p** |
| Warm Accuracy | +0.15%p | **+0.71%p** |

---

## 분석

### 1. Backbone 영향 (vit_base vs vit_small)

- vit_base (768 dim)가 vit_small (384 dim)보다 **평균 +6.39%p** 높은 성능
- 특히 Domain C에서 **+17.17%p** 차이 발생
- Feature dimension이 클수록 anomaly discrimination 향상

### 2. Coreset Ratio 영향 (100% vs 1%)

- 1% coreset은 메모리 뱅크 크기를 99% 줄이지만 성능 저하 발생
- Domain C에서 가장 큰 영향: Cold accuracy가 53.55%로 급락
- 어려운 도메인에서 충분한 reference sample 필요

### 3. Domain C 특이성

- 모든 실험에서 가장 낮은 성능
- Cold sample이 특히 어려움 (exp23: 53.55% → CPD Multi-k: 61.66%)
- **CPD로 +3.54%p 개선** (가장 큰 개선폭)
- 그러나 여전히 Cold accuracy가 낮음 (61.66%)

### 4. Cold vs Warm 격차

| 실험 | Cold 평균 | Warm 평균 | 격차 |
|------|----------|----------|------|
| Vanilla PatchCore | 95.84% | 98.39% | 2.55%p |
| exp23 재현 | 83.32% | 98.09% | **14.77%p** |
| exp23 + CPD f_ctx | 85.37% | 98.24% | 12.87%p |
| exp23 + CPD Multi-k | 86.21% | 98.80% | **12.59%p** |

- exp23 재현에서 Cold/Warm 격차가 매우 큼
- CPD가 Cold accuracy를 개선하여 격차 축소 (14.77%p → 12.59%p)
- 그러나 vit_base (2.55%p 격차)에 비하면 여전히 큰 격차

### 5. CPD 효과 분석

- **CPD Multi-k가 f_ctx보다 일관되게 우수**: 평균 +0.70%p 추가 개선
- **Domain C에서 가장 큰 효과**: +3.54%p (baseline이 낮을수록 효과 큼)
- **Cold sample에서 더 효과적**: Cold +2.89%p vs Warm +0.71%p
- **AUROC 개선도 유의미**: 평균 +1.42%p (Domain C: +4.80%p)

---

## 실험 명령어

### Vanilla PatchCore (vit_base, coreset 100%)

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python examples/notebooks/10_patchcore_variant/006_cpd_patchcore/run_cpd_patchcore.py \
    --domain domain_A --k-ref 1000 --feature-mode f_only --ref-source train
```

### exp23 재현 (vit_small, coreset 1%)

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python examples/notebooks/10_patchcore_variant/006_cpd_patchcore/run_cpd_patchcore.py \
    --domain domain_A --k-ref 1000 --feature-mode f_only --ref-source train \
    --backbone vit_small_patch14_dinov2 --layers blocks.8 --coreset-ratio 0.01 --resize-method resize
```

### exp23 + CPD Multi-k concat

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python examples/notebooks/10_patchcore_variant/006_cpd_patchcore/run_cpd_patchcore.py \
    --domain domain_A --k-ref 1000 --ref-source train \
    --backbone vit_small_patch14_dinov2 --layers blocks.8 --coreset-ratio 0.01 --resize-method resize \
    --multik-mode concat --multik-list 2 3
```

---

## 로그 파일

| 실험 | 로그 파일 |
|------|----------|
| Vanilla PatchCore A | `logs/vanilla_patchcore_trainref_A.log` |
| Vanilla PatchCore B | `logs/vanilla_patchcore_trainref_B.log` |
| Vanilla PatchCore C | `logs/vanilla_patchcore_trainref_C.log` |
| Vanilla PatchCore D | `logs/vanilla_patchcore_trainref_D.log` |
| exp23 재현 A | `logs/exp23_repro_trainref_A.log` |
| exp23 재현 B | `logs/exp23_repro_trainref_B.log` |
| exp23 재현 C | `logs/exp23_repro_trainref_C.log` |
| exp23 재현 D | `logs/exp23_repro_trainref_D.log` |
| exp23 + CPD f_ctx A | `logs/exp23_cpd_trainref_A.log` |
| exp23 + CPD f_ctx B | `logs/exp23_cpd_trainref_B.log` |
| exp23 + CPD f_ctx C | `logs/exp23_cpd_trainref_C.log` |
| exp23 + CPD f_ctx D | `logs/exp23_cpd_trainref_D.log` |
| exp23 + CPD Multi-k A | `logs/exp23_cpd_multik_trainref_A.log` |
| exp23 + CPD Multi-k B | `logs/exp23_cpd_multik_trainref_B.log` |
| exp23 + CPD Multi-k C | `logs/exp23_cpd_multik_trainref_C.log` |
| exp23 + CPD Multi-k D | `logs/exp23_cpd_multik_trainref_D.log` |

---

## 결론

### 성능 순위 (Overall Accuracy 평균)

| 순위 | 설정 | 평균 Accuracy | 평균 AUROC |
|------|------|--------------|------------|
| 1 | **Vanilla PatchCore** (vit_base, 100%) | **97.12%** | **99.27%** |
| 2 | exp23 + CPD Multi-k | 92.53% | 96.60% |
| 3 | exp23 + CPD f_ctx | 91.83% | 96.18% |
| 4 | exp23 Baseline | 90.73% | 95.18% |

### 핵심 발견

1. **Backbone + Coreset이 CPD보다 영향이 큼**
   - vit_base + 100% coreset: 97.12%
   - vit_small + 1% coreset + CPD: 92.53%
   - 차이: **4.59%p** (backbone/coreset이 더 중요)

2. **CPD는 일관된 개선 효과 제공**
   - exp23 baseline → CPD Multi-k: **+1.80%p**
   - 특히 Domain C에서 효과적: **+3.54%p**
   - Cold sample에서 더 효과적: **+2.89%p**

3. **Domain C는 여전히 어려움**
   - 최고 성능: Vanilla PatchCore 93.43%
   - exp23 + CPD Multi-k: 79.80%
   - Cold accuracy 특히 낮음 (61.66%)

4. **실용적 권장 사항**
   - 성능 우선: **vit_base + 100% coreset** (97.12%)
   - 리소스 제한 시: **vit_small + 1% coreset + CPD Multi-k** (92.53%)
   - CPD는 baseline이 낮을수록 효과적
