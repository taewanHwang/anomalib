# 006. CPD-PatchCore 실험 결과

## 개요

CPD-PatchCore (Contextual Patch Descriptor PatchCore)는 기존 PatchCore의 patch feature에 **horizontal context**를 추가하여 thin horizontal band fault 탐지 성능을 개선하는 방법이다.

### 핵심 아이디어

```python
# 기존 PatchCore
embedding = f[i,j]  # shape: (D,)
score = min_distance(embedding, memory_bank)

# CPD-PatchCore
ctx = mean(f[i, j-k:j+k+1])  # horizontal context (2k+1 patches)
embedding = concat(f[i,j], ctx)  # shape: (2D,)
score = min_distance(embedding, memory_bank)
```

---

## 실험 설정

- **Backbone**: DINOv2 ViT-B/14 (blocks.8)
- **Input size**: 518×518
- **k_ref**: 32 (16 cold + 16 warm reference samples)
- **Aggregation methods tested**: mean, max, gaussian, median, trimmed, attention
- **Context sizes tested**: k=1 (3 patches), k=2 (5 patches), k=3 (7 patches)

---

## Domain C 결과

### Aggregation Method 비교 (k=2)

| Config | Overall Acc | Cold | Warm | AUROC | vs Baseline |
|--------|-------------|------|------|-------|-------------|
| Baseline (f_only) | 90.25% | 88.74% | 91.75% | 95.74% | - |
| **CPD mean** | **91.77%** | **89.76%** | 93.76% | **96.88%** | **+1.52%** |
| CPD trimmed | 91.62% | 88.54% | 94.67% | 96.80% | +1.37% |
| CPD gaussian | 91.16% | 88.64% | 93.66% | 96.36% | +0.91% |
| CPD attention | 90.56% | 88.54% | 92.56% | 95.72% | +0.31% |
| CPD max | 88.54% | 85.09% | 91.95% | 94.54% | -1.71% |

### Context Size 비교 (mean aggregation)

| k | Overall Acc | Cold | Warm | AUROC | vs Baseline |
|---|-------------|------|------|-------|-------------|
| 1 (3 patches) | 90.45% | 87.83% | 93.06% | 95.73% | +0.20% |
| 2 (5 patches) | **91.77%** | **89.76%** | 93.76% | 96.88% | **+1.52%** |
| 3 (7 patches) | **91.77%** | 89.45% | **94.06%** | **96.99%** | **+1.52%** |

### Feature Mode 비교 (mean, k=2)

| Mode | Overall Acc | Cold | Warm | AUROC |
|------|-------------|------|------|-------|
| f_only (baseline) | 90.25% | 88.74% | 91.75% | 95.74% |
| **f_ctx** | **91.77%** | **89.76%** | 93.76% | **96.88%** |
| f_ctx_std | 90.56% | 87.42% | 93.66% | 96.04% |

### Sampling Mode 비교 (mean, k=2)

가로 방향 결함이 불연속적으로 나타나는 특성을 고려하여, 인접 패치 대신 랜덤 샘플링 전략을 테스트함.

| Sampling Mode | Overall Acc | Cold | Warm | AUROC | vs Baseline |
|---------------|-------------|------|------|-------|-------------|
| **adjacent (default)** | **91.77%** | **89.76%** | **93.76%** | **96.88%** | **+1.52%** |
| random | 90.76% | 88.34% | 93.16% | 96.14% | +0.51% |
| distance_weighted (d=0.5) | 89.75% | 86.41% | 93.06% | 93.68% | -0.50% |
| distance_weighted (d=2.0) | 88.84% | 86.11% | 91.55% | 93.66% | -1.41% |

**Sampling Mode 설명:**
- **adjacent**: 기존 방식, j-k ~ j+k 인접 패치 사용
- **random**: 같은 row에서 2k개 패치를 완전 랜덤 선택
- **distance_weighted**: 거리 기반 확률 샘플링 (p ∝ 1/d^decay), decay가 클수록 더 로컬

**결론**: 인접 패치 샘플링이 최적. 랜덤 샘플링은 결함 패턴의 로컬 일관성 정보를 잃어 성능 저하.

### Multi-k CPD 비교 (k=2,3 결합)

단일 스케일(k=2 또는 k=3) 대신 여러 스케일의 context를 결합하여 더 풍부한 정보를 활용.

| Mode | Overall Acc | Cold | Warm | AUROC | vs Baseline | vs Single k=2 |
|------|-------------|------|------|-------|-------------|---------------|
| Single k=2 | 91.77% | 89.76% | 93.76% | 96.88% | +1.52% | - |
| Single k=3 | 91.77% | 89.45% | 94.06% | 96.99% | +1.52% | +0.00% |
| **Multi-k weighted** | 91.87% | 89.66% | 94.06% | 96.98% | +1.62% | +0.10% |
| **Multi-k concat** | **92.27%** | **90.06%** | **94.47%** | **97.31%** | **+2.02%** | **+0.50%** |

**Multi-k Mode 설명:**
- **weighted**: k=2, k=3 context를 가중 평균 (weight=0.5씩), 차원 = 2D
- **concat**: original + k=2 context + k=3 context 연결, 차원 = 3D

**결론**: Multi-k concat이 가장 좋은 성능 (92.27%, +2.02%p vs baseline). 여러 스케일의 context를 모두 보존하는 것이 더 효과적.

### Ablation Study: 차원 증가 vs 의미있는 Context

Multi-k concat의 성능 향상이 단순 차원 증가(2D→3D) 때문인지, 의미있는 horizontal context 때문인지 검증.

| Method | Dim | Overall Acc | Cold | Warm | AUROC | vs Baseline |
|--------|-----|-------------|------|------|-------|-------------|
| Baseline (f_only) | 1D | 90.25% | 88.74% | 91.75% | 95.74% | - |
| **Random 2D** | 2D | 82.83% | 79.61% | 86.02% | 81.09% | **-7.42%** |
| **Random 3D** | 3D | 82.83% | 79.61% | 86.02% | 81.09% | **-7.42%** |
| **Global 2D** | 2D | 90.96% | 88.84% | 93.06% | 95.97% | +0.71% |
| **Global 3D** | 3D | 90.91% | 87.42% | 94.37% | 96.02% | +0.66% |
| CPD mean k=2 | 2D | 91.77% | 89.76% | 93.76% | 96.88% | +1.52% |
| **Multi-k concat** | 3D | **92.27%** | **90.06%** | **94.47%** | **97.31%** | **+2.02%** |

**Ablation Mode 설명:**
- **Random**: f + random_embedding (무의미한 노이즈)
- **Global**: f + global_average_pooling (전역 정보, 위치 무관)
- **CPD**: f + horizontal_context (의미있는 로컬 context)

**핵심 결론:**
1. **Random embedding 실패 (-7.42%p)**: 단순 차원 증가는 성능을 **크게 저하**시킴
   - 2D든 3D든 동일하게 나쁨 → 차원 증가 자체는 무의미
2. **Global average는 중립 (+0.7%p)**: baseline과 비슷
   - 전역 정보는 localized fault 탐지에 도움 안됨
3. **CPD (horizontal context)만 의미있는 개선 (+2.02%p)**
   - **Multi-k concat의 성능 향상은 horizontal context 정보 덕분**
   - 단순 차원 증가가 아닌 **의미 있는 context**가 핵심

---

## 다른 도메인 결과

### Domain A

| Config | Overall Acc | Cold | Warm | AUROC | vs Baseline |
|--------|-------------|------|------|-------|-------------|
| Baseline (f_only) | 95.51% | 93.71% | 97.28% | 96.96% | - |
| **CPD mean k=2** | **95.96%** | **94.02%** | **97.89%** | **97.56%** | **+0.45%** |

### Domain B

| Config | Overall Acc | Cold | Warm | AUROC | vs Baseline |
|--------|-------------|------|------|-------|-------------|
| Baseline (f_only) | 93.08% | 90.77% | 95.37% | 93.22% | - |
| **CPD mean k=2** | **93.48%** | **91.08%** | **95.88%** | **93.54%** | **+0.40%** |

### Domain D

| Config | Overall Acc | Cold | Warm | AUROC | vs Baseline |
|--------|-------------|------|------|-------|-------------|
| Baseline (f_only) | 94.65% | 94.12% | 95.17% | 98.37% | - |
| **CPD mean k=2** | **95.35%** | **95.03%** | **95.67%** | **98.67%** | **+0.70%** |

---

## 전체 도메인 요약

| Domain | Baseline | Single k=2 | Multi-k concat | vs Baseline |
|--------|----------|------------|----------------|-------------|
| A | 95.51% | 95.96% | **96.46%** | **+0.95%** |
| B | 93.08% | 93.48% | **93.69%** | **+0.61%** |
| **C** | 90.25% | 91.77% | **92.27%** | **+2.02%** |
| D | 94.65% | 95.35% | **95.45%** | **+0.80%** |
| **Average** | **93.37%** | 94.14% | **94.47%** | **+1.10%** |

---

## Layer 확장 실험

기존 1-layer (blocks.8)에서 multi-layer로 확장했을 때의 성능 변화를 분석.

### 실험 설정

- **1-layer**: blocks.8 (기존 baseline)
- **2-layer**: blocks.8 + blocks.11
- **3-layer**: blocks.4 + blocks.8 + blocks.11

### Domain별 상세 결과

#### Domain A

| Layers | Mode | Accuracy | AUROC |
|--------|------|----------|-------|
| 1-layer (b8) | f_only | 95.51% | 96.96% |
| 2-layer (b8,b11) | f_only | 99.70% | 99.99% |
| 2-layer (b8,b11) | f_ctx | 99.60% | 100.00% |
| 3-layer (b4,b8,b11) | f_only | 99.70% | 99.99% |
| 3-layer (b4,b8,b11) | f_ctx | 99.60% | 100.00% |

#### Domain B

| Layers | Mode | Accuracy | AUROC |
|--------|------|----------|-------|
| 1-layer (b8) | f_only | 93.08% | 93.22% |
| 2-layer (b8,b11) | f_only | 99.49% | 99.99% |
| 2-layer (b8,b11) | f_ctx | **99.90%** | **100.00%** |
| 3-layer (b4,b8,b11) | f_only | 99.49% | 99.99% |
| 3-layer (b4,b8,b11) | f_ctx | **99.90%** | **100.00%** |

#### Domain C

| Layers | Mode | Accuracy | AUROC |
|--------|------|----------|-------|
| 1-layer (b8) | f_only | 90.25% | 95.74% |
| 1-layer (b8) | f_ctx | 90.45% | 95.73% |
| 2-layer (b8,b11) | f_only | 96.67% | 99.27% |
| 2-layer (b8,b11) | f_ctx | **96.77%** | **99.47%** |
| 3-layer (b4,b8,b11) | f_only | 96.67% | 99.28% |
| 3-layer (b4,b8,b11) | f_ctx | **96.77%** | **99.47%** |

#### Domain D

| Layers | Mode | Accuracy | AUROC |
|--------|------|----------|-------|
| 1-layer (b8) | f_only | 94.65% | 98.37% |
| 2-layer (b8,b11) | f_only | 98.99% | 99.96% |
| 2-layer (b8,b11) | f_ctx | **99.49%** | **99.99%** |
| 3-layer (b4,b8,b11) | f_only | 98.99% | 99.96% |
| 3-layer (b4,b8,b11) | f_ctx | **99.49%** | **99.99%** |

### Layer 확장 효과 요약 (Baseline f_only 기준)

| Domain | 1-layer | 2-layer | 3-layer | 1→2 Δ | 2→3 Δ |
|--------|---------|---------|---------|-------|-------|
| A | 95.51% | 99.70% | 99.70% | **+4.19%p** | +0.00%p |
| B | 93.08% | 99.49% | 99.49% | **+6.41%p** | +0.00%p |
| C | 90.25% | 96.67% | 96.67% | **+6.41%p** | +0.00%p |
| D | 94.65% | 98.99% | 98.99% | **+4.34%p** | +0.00%p |
| **평균** | 93.37% | 98.71% | 98.71% | **+5.34%p** | +0.00%p |

### 2-Layer + CPD (f_ctx) 성능 (k=1)

| Domain | Accuracy | AUROC | Cold Acc | Warm Acc |
|--------|----------|-------|----------|----------|
| A | 99.60% | 100.00% | 99.49% | 99.70% |
| B | **99.90%** | **100.00%** | 99.80% | 100.00% |
| C | 96.77% | 99.47% | 95.33% | 98.19% |
| D | 99.49% | 99.99% | 99.39% | 99.60% |
| **평균** | **98.94%** | **99.86%** | - | - |

### 2-Layer Context Size 비교 (Domain C)

2-layer에서도 context size에 따른 성능 변화 확인.

| Config | Accuracy | AUROC | vs k=1 |
|--------|----------|-------|--------|
| 2-layer k=1 | 96.77% | 99.47% | - |
| 2-layer k=2 | 97.07% | 99.55% | +0.30%p |
| **2-layer multi-k concat** | **97.68%** | **99.67%** | **+0.91%p** |

**결론**: 2-layer에서도 **multi-k concat이 최고 성능** (+0.91%p 추가 개선)

### 2-Layer + Multi-k CPD concat 전체 도메인 결과

| Domain | Accuracy | AUROC | Cold Acc | Warm Acc |
|--------|----------|-------|----------|----------|
| A | 99.70% | 99.996% | 99.59% | 99.80% |
| B | 99.80% | 99.998% | 99.59% | 100.00% |
| C | 97.68% | 99.67% | 96.65% | 98.69% |
| D | **99.90%** | **99.996%** | 99.80% | 100.00% |
| **평균** | **99.27%** | **99.91%** | - | - |

#### 2-layer k=1 대비 변화

| Domain | k=1 Acc | Multi-k Acc | Δ Acc |
|--------|---------|-------------|-------|
| A | 99.60% | 99.70% | **+0.10%p** |
| B | 99.90% | 99.80% | -0.10%p |
| C | 96.77% | 97.68% | **+0.91%p** |
| D | 99.49% | 99.90% | **+0.41%p** |
| **평균** | 98.94% | **99.27%** | **+0.33%p** |

**결론**:
- Domain C에서 가장 큰 개선 (+0.91%p)
- Domain D에서도 의미 있는 개선 (+0.41%p)
- Domain B는 미세하게 하락 (-0.10%p, 이미 99.90%로 높았음)
- **전체 평균 +0.33%p 개선 (98.94% → 99.27%)**

### Layer 확장 핵심 발견

1. **1→2 layer 확장 효과 극대**: 평균 **+5.34%p** 개선
   - blocks.11 (deeper layer)의 semantic feature가 핵심
   - 모든 도메인에서 4~6%p의 큰 성능 향상

2. **2→3 layer 확장은 효과 없음**: 모든 도메인에서 0.00%p
   - blocks.4 추가는 불필요한 계산 비용만 발생
   - 얕은 layer의 low-level feature는 anomaly detection에 기여 안함

3. **2-layer + CPD (f_ctx)가 최고 성능**
   - Domain A: 99.70% → 99.60% (-0.10%p)
   - Domain B: 99.49% → 99.90% (+0.41%p)
   - Domain C: 96.67% → 96.77% (+0.10%p)
   - Domain D: 98.99% → 99.49% (+0.50%p)
   - **평균: 98.71% → 98.94% (+0.23%p)**

### Layer 확장 최적 설정

```
최적: 2-layer (blocks.8 + blocks.11) + Multi-k CPD concat (k=2,3)
- 전체 평균 Accuracy: 99.27% (AUROC: 99.91%)
- 1-layer baseline (93.37%) 대비 +5.90%p 개선
- 2-layer k=1 (98.94%) 대비 +0.33%p 추가 개선
- 3-layer는 불필요 (추가 이득 없음)
```

---

## Few-shot 실험

Reference sample 수를 줄였을 때의 성능 변화 분석 (2-layer + multi-k concat 기준).

### Accuracy 비교

| Domain | 32-shot (16+16) | 8-shot (4+4) | 2-shot (1+1) | 32→8 Δ | 32→2 Δ |
|--------|-----------------|--------------|--------------|--------|--------|
| A | 99.70% | 99.39% | 99.29% | -0.31%p | -0.41%p |
| B | 99.80% | 99.70% | 99.49% | -0.10%p | -0.31%p |
| C | 97.68% | 96.46% | 95.45% | -1.22%p | -2.23%p |
| D | 99.90% | 99.19% | 99.19% | -0.71%p | -0.71%p |
| **평균** | **99.27%** | **98.69%** | **98.36%** | **-0.59%p** | **-0.92%p** |

### AUROC 비교

| Domain | 32-shot (16+16) | 8-shot (4+4) | 2-shot (1+1) | 32→8 Δ | 32→2 Δ |
|--------|-----------------|--------------|--------------|--------|--------|
| A | 99.996% | 99.99% | 99.97% | -0.01%p | -0.03%p |
| B | 99.998% | 99.99% | 99.98% | -0.01%p | -0.02%p |
| C | 99.67% | 99.47% | 98.85% | -0.20%p | -0.82%p |
| D | 99.996% | 99.97% | 99.98% | -0.03%p | -0.02%p |
| **평균** | **99.91%** | **99.86%** | **99.70%** | **-0.06%p** | **-0.22%p** |

### Few-shot 핵심 발견

1. **Domain A, B, D는 few-shot에 강건**
   - 8-shot: 평균 -0.37%p (A,B,D만)
   - 2-shot: 평균 -0.48%p (A,B,D만)
   - 99% 이상 성능 유지

2. **Domain C만 shot 감소에 민감**
   - 8-shot: -1.22%p, 2-shot: -2.23%p
   - 다른 도메인 대비 3~4배 큰 성능 하락
   - Cold sample 다양성이 더 필요한 도메인

3. **2-shot에서도 평균 98.36% 달성**
   - Reference sample 단 2개 (1 cold + 1 warm)
   - 실용적 few-shot anomaly detection 가능

### Baseline vs CPD 비교 (Few-shot)

CPD의 효과를 검증하기 위해 동일 조건에서 baseline (f_only)과 비교.

#### 8-shot Accuracy

| Domain | Baseline (f_only) | CPD multi-k | CPD 효과 |
|--------|-------------------|-------------|----------|
| A | 99.19% | 99.39% | **+0.20%p** |
| B | 98.99% | 99.70% | **+0.71%p** |
| C | 94.75% | 96.46% | **+1.71%p** |
| D | 97.88% | 99.19% | **+1.31%p** |
| **평균** | **97.70%** | **98.69%** | **+0.98%p** |

#### 2-shot Accuracy

| Domain | Baseline (f_only) | CPD multi-k | CPD 효과 |
|--------|-------------------|-------------|----------|
| A | 98.79% | 99.29% | **+0.50%p** |
| B | 98.99% | 99.49% | **+0.50%p** |
| C | 93.74% | 95.45% | **+1.71%p** |
| D | 97.68% | 99.19% | **+1.51%p** |
| **평균** | **97.30%** | **98.36%** | **+1.06%p** |

#### 32-shot Accuracy

| Domain | Baseline (f_only) | CPD multi-k | CPD 효과 |
|--------|-------------------|-------------|----------|
| A | 99.70% | 99.70% | +0.00%p |
| B | 99.49% | 99.80% | **+0.31%p** |
| C | 96.67% | 97.68% | **+1.01%p** |
| D | 98.99% | 99.90% | **+0.91%p** |
| **평균** | **98.71%** | **99.27%** | **+0.56%p** |

#### CPD 효과 분석

1. **CPD는 few-shot에서 더 효과적**
   - 32-shot: 평균 +0.56%p 개선
   - 8-shot: 평균 +0.98%p 개선
   - 2-shot: 평균 +1.06%p 개선
   - **Reference sample이 적을수록 CPD 효과 증가**

2. **Domain C, D에서 CPD 효과 극대화**
   - Domain C: +1.71%p (8-shot, 2-shot 동일)
   - Domain D: +1.31~1.51%p
   - 어려운 도메인에서 horizontal context가 더 유효

3. **CPD가 few-shot의 한계 보완**
   - 적은 reference sample로 인한 성능 저하를 context 정보로 보완
   - 2-shot baseline 97.30% → CPD 98.36%로 1%p 이상 개선

### Few-shot 권장 사항

```
실용적 설정: 8-shot (4 cold + 4 warm) + CPD multi-k concat
- 평균 Accuracy: 98.69% (baseline 97.70% 대비 +0.98%p)
- 평균 AUROC: 99.86%
- Memory bank 크기 1/4로 감소
- 추론 속도 향상 + CPD로 성능 보완
```

---

## 1-shot 실험: Cold vs Warm Reference

1-shot 설정에서 reference sample의 condition (cold/warm)이 성능에 미치는 영향 분석.

### 실험 설정

- **Layers**: 2-layer (blocks.8 + blocks.11)
- **k_ref**: 1 (single reference sample)
- **Cold reference**: file_idx가 가장 낮은 (가장 차가운) good sample 1개
- **Warm reference**: file_idx가 가장 높은 (가장 따뜻한) good sample 1개

### 1-shot Cold Reference 결과

| Domain | Method | Overall Acc | Cold Acc | Warm Acc | AUROC |
|--------|--------|-------------|----------|----------|-------|
| A | Baseline | 97.47% | 97.36% | 97.59% | 99.57% |
| A | CPD multi-k | **98.89%** | **98.88%** | **98.89%** | **99.91%** |
| B | Baseline | 98.18% | 98.38% | 97.99% | 99.87% |
| B | CPD multi-k | **98.89%** | **98.88%** | **98.89%** | **99.96%** |
| C | Baseline | 92.63% | 89.15% | 96.08% | 97.46% |
| C | CPD multi-k | **94.65%** | **92.09%** | **97.18%** | **98.33%** |
| D | Baseline | 95.45% | 95.23% | 95.67% | 99.10% |
| D | CPD multi-k | **97.88%** | **97.87%** | **97.89%** | **99.82%** |
| **평균** | Baseline | 95.93% | 95.03% | 96.83% | 99.00% |
| **평균** | CPD multi-k | **97.58%** | **96.93%** | **98.21%** | **99.51%** |

### 1-shot Warm Reference 결과

| Domain | Method | Overall Acc | Cold Acc | Warm Acc | AUROC |
|--------|--------|-------------|----------|----------|-------|
| A | Baseline | 97.78% | 97.26% | 98.29% | 99.64% |
| A | CPD multi-k | **98.89%** | **98.48%** | **99.30%** | **99.93%** |
| B | Baseline | 96.26% | 94.93% | 97.59% | 99.44% |
| B | CPD multi-k | **97.27%** | **96.35%** | **98.19%** | **99.71%** |
| C | Baseline | 89.85% | 85.19% | 94.47% | 95.96% |
| C | CPD multi-k | **93.84%** | **89.45%** | **98.19%** | **97.88%** |
| D | Baseline | 97.37% | 96.96% | 97.79% | 99.75% |
| D | CPD multi-k | **98.59%** | **97.97%** | **99.20%** | **99.95%** |
| **평균** | Baseline | 95.32% | 93.59% | 97.04% | 98.70% |
| **평균** | CPD multi-k | **97.15%** | **95.56%** | **98.72%** | **99.37%** |

### Cold vs Warm Reference 비교 (Accuracy)

| Domain | Cold Ref (Baseline) | Warm Ref (Baseline) | 더 나은 쪽 | Cold Ref (CPD) | Warm Ref (CPD) | 더 나은 쪽 |
|--------|---------------------|---------------------|------------|----------------|----------------|------------|
| A | 97.47% | 97.78% | Warm +0.31%p | 98.89% | 98.89% | 동일 |
| B | **98.18%** | 96.26% | **Cold +1.92%p** | **98.89%** | 97.27% | **Cold +1.62%p** |
| C | **92.63%** | 89.85% | **Cold +2.78%p** | **94.65%** | 93.84% | **Cold +0.81%p** |
| D | 95.45% | **97.37%** | Warm +1.92%p | 97.88% | **98.59%** | Warm +0.71%p |
| **평균** | **95.93%** | 95.32% | **Cold +0.61%p** | **97.58%** | 97.15% | **Cold +0.43%p** |

### CPD 효과 비교 (1-shot)

| Reference | Domain A | Domain B | Domain C | Domain D | 평균 |
|-----------|----------|----------|----------|----------|------|
| Cold ref | +1.42%p | +0.71%p | +2.02%p | +2.43%p | **+1.65%p** |
| Warm ref | +1.11%p | +1.01%p | +3.99%p | +1.22%p | **+1.83%p** |

### 1-shot 핵심 발견

1. **Cold reference가 평균적으로 더 효과적**
   - Baseline: Cold 95.93% vs Warm 95.32% → **Cold +0.61%p**
   - CPD: Cold 97.58% vs Warm 97.15% → **Cold +0.43%p**
   - Cold sample이 더 엄격한 기준 제공

2. **Domain별 최적 reference**
   - **Domain B, C**: Cold reference가 확실히 우수 (1.6~2.8%p 차이)
   - **Domain A**: 거의 동일 (0.31%p 차이)
   - **Domain D**: Warm reference가 우수 (0.7~1.9%p 차이)

3. **Cold/Warm test set 성능 패턴**
   - Cold ref → Cold test 성능 상승, Warm test 성능 하락
   - Warm ref → Warm test 성능 상승, Cold test 성능 하락
   - Reference의 condition이 해당 condition test에 유리

4. **CPD 효과는 1-shot에서도 일관됨**
   - Cold ref 평균: +1.65%p
   - Warm ref 평균: +1.83%p
   - **Reference sample이 적을수록 CPD 효과 증가 (2-shot +1.06%p → 1-shot +1.74%p)**

5. **Domain C의 특이성**
   - Cold ref에서 Warm test (96.08%) vs Cold test (89.15%) 격차 큼
   - Warm ref에서 그 격차가 더 확대 (94.47% vs 85.19%)
   - **Cold sample 다양성이 특히 중요한 도메인**

### 1-shot 권장 사항

```
1-shot 실용적 설정:
- Method: CPD multi-k concat + 2-layer
- Reference: Cold sample 선택 (평균 +0.43~0.61%p 우수)
- 기대 성능: 평균 97.58% (Baseline 95.93% 대비 +1.65%p)

Domain별 최적화:
- Domain A: Cold/Warm 무관 (98.89% 동일)
- Domain B: Cold 필수 (98.89% vs 97.27%)
- Domain C: Cold 권장 (94.65% vs 93.84%)
- Domain D: Warm 권장 (98.59% vs 97.88%)
```

---

## 결론

### 전체 도메인 분석

1. **CPD-PatchCore 일관된 성능 향상**: 모든 4개 도메인에서 개선 확인
   - Domain A: +0.45% (95.51% → 95.96%)
   - Domain B: +0.40% (93.08% → 93.48%)
   - **Domain C: +1.52%** (90.25% → 91.77%) - 가장 큰 개선
   - Domain D: +0.70% (94.65% → 95.35%)
   - **평균 개선: +0.77%p**

2. **Domain C에서 가장 큰 개선**: 가장 어려운 도메인(lowest baseline)에서 CPD가 가장 효과적
   - Horizontal context가 thin band fault 탐지에 특히 유효

3. **Mean aggregation이 최적**: uniform averaging이 fault signal 적분에 가장 효과적
4. **Max pooling은 역효과 (-1.71% on C)**: 개별 maximum이 context 정보를 상쇄
5. **Adjacent sampling이 최적**: 랜덤 샘플링(-0.99%p)보다 인접 패치가 효과적
   - 결함이 불연속적으로 나타나더라도 로컬 context가 더 중요
   - Distance-weighted 샘플링은 baseline보다 낮은 성능

6. **Multi-k concat이 최상의 성능**: 모든 도메인에서 일관된 추가 개선
   - Domain A: 96.46% (+0.95%p vs baseline, +0.50%p vs single k=2)
   - Domain B: 93.69% (+0.61%p vs baseline, +0.21%p vs single k=2)
   - Domain C: 92.27% (+2.02%p vs baseline, +0.50%p vs single k=2)
   - Domain D: 95.45% (+0.80%p vs baseline, +0.10%p vs single k=2)
   - **평균: 94.47% (+1.10%p vs baseline, +0.33%p vs single k=2)**

7. **Ablation으로 검증**: Multi-k concat의 개선은 **horizontal context 정보** 덕분
   - Random embedding concat: -7.42%p (차원 증가만으로는 성능 저하)
   - Global average concat: +0.7%p (전역 정보는 도움 안됨)
   - **의미있는 horizontal context만 성능 향상에 기여**

8. **2-Layer + Multi-k concat이 최종 최고 성능**
   - 2-layer에서도 multi-k concat이 추가 개선 제공
   - 전체 평균: 98.94% (k=1) → **99.27%** (multi-k concat) = **+0.33%p**
   - **1-layer baseline (93.37%) 대비 총 +5.90%p 개선**

### 최적 설정

#### 1-Layer 사용 시 (기존 방식)
- **Method**: Multi-k CPD concat (k=2, k=3)
- **Aggregation**: mean
- **Feature mode**: f_ctx (concatenate original + contexts)
- **Sampling mode**: adjacent (인접 패치)
- 단일 스케일 사용 시: k=2 또는 k=3 (동일 성능)

#### 최종 권장 설정 (2-Layer + Multi-k CPD)
- **Layers**: blocks.8 + blocks.11 (2-layer)
- **Feature mode**: f_ctx (CPD with horizontal context)
- **Multi-k mode**: concat (k=2, k=3 결합)
- **Aggregation**: mean
- **전체 평균 Accuracy**: 99.27% (AUROC: 99.91%)
- **1-layer baseline (93.37%) 대비**: **+5.90%p** 개선
- **2-layer k=1 (98.94%) 대비**: **+0.33%p** 추가 개선

---

## 실행 명령어

### 1-Layer 실험 (기존)

```bash
# Baseline
CUDA_VISIBLE_DEVICES=0 python run_cpd_patchcore.py \
    --domain domain_C --k-ref 32 --context-k 1 --aggregation mean --feature-mode f_only

# CPD-PatchCore (best config for 1-layer)
CUDA_VISIBLE_DEVICES=0 python run_cpd_patchcore.py \
    --domain domain_C --k-ref 32 --context-k 2 --aggregation mean --feature-mode f_ctx

# Multi-k concat (k=2,3 연결)
CUDA_VISIBLE_DEVICES=0 python run_cpd_patchcore.py \
    --domain domain_C --k-ref 32 --multik-mode concat --multik-list 2 3
```

### 2-Layer 실험 (최종 권장)

```bash
# 2-Layer Baseline
CUDA_VISIBLE_DEVICES=0 python run_cpd_patchcore.py \
    --domain domain_C --k-ref 32 --context-k 1 --aggregation mean --feature-mode f_only \
    --layers blocks.8 blocks.11

# 2-Layer + CPD k=1
CUDA_VISIBLE_DEVICES=0 python run_cpd_patchcore.py \
    --domain domain_C --k-ref 32 --context-k 1 --aggregation mean --feature-mode f_ctx \
    --layers blocks.8 blocks.11

# 2-Layer + Multi-k CPD concat (★ 최종 권장 설정)
CUDA_VISIBLE_DEVICES=0 python run_cpd_patchcore.py \
    --domain domain_C --k-ref 32 --multik-mode concat --multik-list 2 3 \
    --layers blocks.8 blocks.11
```

### 3-Layer 실험 (비권장 - 추가 이득 없음)

```bash
# 3-Layer Baseline
CUDA_VISIBLE_DEVICES=0 python run_cpd_patchcore.py \
    --domain domain_C --k-ref 32 --context-k 1 --aggregation mean --feature-mode f_only \
    --layers blocks.4 blocks.8 blocks.11

# 3-Layer + CPD
CUDA_VISIBLE_DEVICES=0 python run_cpd_patchcore.py \
    --domain domain_C --k-ref 32 --context-k 1 --aggregation mean --feature-mode f_ctx \
    --layers blocks.4 blocks.8 blocks.11
```

---

## 파일 구조

```
006_cpd_patchcore/
├── run_cpd_patchcore.py    # Main script
├── results/                 # Experiment results
│   ├── cpd_mean_k2_f_ctx_domain_C_ref32/
│   ├── cpd_mean_k3_f_ctx_domain_C_ref32/
│   ├── cpd_mean_k2_random_f_ctx_domain_C_ref32/
│   ├── cpd_multik_weighted_k2_3_domain_C_ref32/
│   ├── cpd_multik_concat_k2_3_domain_C_ref32/
│   └── ...
└── eda/                     # EDA artifacts
```
