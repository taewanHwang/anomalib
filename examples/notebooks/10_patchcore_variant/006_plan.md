# 006. Spatial-Aware PatchCore 개선 계획

## 배경

CA-PatchCore (Condition-Aware) 분석 결과:
- Oracle gating (100% 정확)이 mixed mode보다 **-2.83%p 낮음**
- Cold fault 패치가 warm normal에 더 유사 (Top-10 anomalous → warm: 73%)
- **결론**: Hard condition gating은 효과 없음, **Spatial-aware scoring**이 더 중요

## 핵심 문제

```
현재 PatchCore: d(f_i,j, bank) → 개별 패치 신호가 너무 약함
- 얇은 가로 결함은 per-patch feature로는 정상과 구분 어려움
- 사람은 "가로로 길게 이어지는 약한 신호"를 공간적으로 적분해서 인식
```

---

## Phase 1: CPD-PatchCore (Contextual Patch Descriptor)

### 목표
PatchCore의 kNN 입력을 **'중심 패치 + 가로 이웃 패치들'의 컨텍스트 특징**으로 변경

### 핵심 아이디어

```python
# 기존 PatchCore
embedding = f[i,j]  # shape: (D,)
score = min_distance(embedding, memory_bank)

# CPD-PatchCore
ctx = mean(f[i, j-k:j+k+1])  # horizontal context
embedding = concat(f[i,j], ctx)  # shape: (2D,)
score = min_distance(embedding, memory_bank)
```

### 기대 효과
- 얇은 가로 결함: 개별 f[i,j]는 정상과 비슷해도, 같은 row의 연속 패치들이 **같은 방향으로 미세하게 흔들리는 패턴**이 context에 적분됨
- kNN 거리 자체가 "라인 패턴"에 민감해짐

### 실험 설계

#### A. Context Size (k) 실험

| 실험 | k (context size) | Feature | 비고 |
|-----|------------------|---------|-----|
| baseline | - | f | 기존 PatchCore |
| cpd_k1 | 1 | [f \| ctx] | 3 patches (j-1, j, j+1) |
| cpd_k2 | 2 | [f \| ctx] | 5 patches |
| cpd_k3 | 3 | [f \| ctx] | 7 patches |

#### B. Context Aggregation Method 실험

| Method | 수식 | 특성 |
|--------|-----|------|
| **mean** | `ctx = mean(f[j-k:j+k+1])` | 기본형, 균일 가중 |
| **max** | `ctx = max(f[j-k:j+k+1])` | 가장 두드러진 특징 추출 |
| **gaussian** | `ctx = Σ w_i * f[j+i]` (w=gaussian) | 중심 가중, 부드러운 전이 |
| **median** | `ctx = median(f[j-k:j+k+1])` | Robust, outlier 제거 |
| **std** | `ctx = std(f[j-k:j+k+1])` | 변동성 측정 (fault=높은 std?) |

#### C. Feature Combination 실험

| 실험 | Feature | Dimension | 비고 |
|-----|---------|-----------|-----|
| f_only | f | D | baseline |
| f_ctx | [f \| ctx] | 2D | CPD 기본형 |
| f_ctx_std | [f \| ctx \| std] | 3D | 변동성 추가 |
| ctx_only | ctx | D | context만 (ablation) |

#### D. 최종 실험 매트릭스 (Phase 1)

| # | k | Aggregation | Feature | 우선순위 |
|---|---|-------------|---------|---------|
| 1 | 1 | mean | [f \| ctx] | **1순위** |
| 2 | 2 | mean | [f \| ctx] | **1순위** |
| 3 | 3 | mean | [f \| ctx] | 1순위 |
| 4 | 2 | max | [f \| ctx] | 2순위 |
| 5 | 2 | gaussian | [f \| ctx] | 2순위 |
| 6 | 2 | mean | [f \| ctx \| std] | 3순위 |

### 구현 포인트

```python
def add_horizontal_context(features, k=1):
    """Add horizontal context to patch features.

    Args:
        features: (B, D, H, W) or (N_patches, D)
        k: context radius (total 2k+1 patches)

    Returns:
        features_cpd: (B, 2D, H, W) or (N_patches, 2D)
    """
    # Horizontal average pooling
    ctx = F.avg_pool2d(features, kernel_size=(1, 2*k+1),
                       stride=1, padding=(0, k))
    return torch.cat([features, ctx], dim=1)
```

### 평가 메트릭
- Overall Accuracy, AUROC
- Cold/Warm breakdown
- vs Baseline (16,16) few-shot: 89.44%

### Phase 1 실험 결과 (Domain C, k_ref=32)

| Config | Overall Acc | Cold | Warm | AUROC | vs Baseline |
|--------|-------------|------|------|-------|-------------|
| Baseline (f_only) | 90.25% | 88.74% | 91.75% | 95.74% | - |
| **CPD mean k=1** | 90.45% | 87.83% | 93.06% | 95.73% | +0.20% |
| **CPD mean k=2** | **91.77%** | **89.76%** | 93.76% | **96.88%** | **+1.52%** |
| **CPD mean k=3** | **91.77%** | 89.45% | **94.06%** | **96.99%** | **+1.52%** |
| CPD max k=2 | 88.54% | 85.09% | 91.95% | 94.54% | -1.71% |
| CPD attention k=2 | 90.56% | 88.54% | 92.56% | 95.72% | +0.31% |
| CPD gaussian k=2 | 91.16% | 88.64% | 93.66% | 96.36% | +0.91% |
| CPD trimmed k=2 | 91.62% | 88.54% | 94.67% | 96.80% | +1.37% |
| CPD mean k=2 f_ctx_std | 90.56% | 87.42% | 93.66% | 96.04% | +0.31% |

### Phase 1 결론

1. **Best: CPD mean k=2/k=3** → 91.77% (+1.52%p)
   - 성공 기준(>90%) 달성, Phase 2 목표(>91%)도 달성!
2. **Mean aggregation이 최적** - uniform weighting이 fault signal 적분에 유리
3. **Max pooling은 실패** - 개별 maximum이 context 정보를 상쇄
4. **std 추가는 도움 안됨** - 이미 충분한 정보가 [f|ctx]에 포함

### Phase 2 필요성 재검토

Phase 1에서 이미 91.77% 달성 → Phase 2 목표(>91%) 초과!
- Residual context 실험은 optional로 진행
- Phase 3보다 **다른 도메인 검증**이 우선

---

## Phase 2: CPD + Residual Context

### 목표
CPD에 **residual (f - ctx)** 추가하여 global shift 제거

### 핵심 아이디어

```python
ctx = mean(f[i, j-k:j+k+1])
residual = f[i,j] - ctx
embedding = concat(f[i,j], ctx, residual)  # shape: (3D,)
```

### 기대 효과
- Cold/warm 구조적 차이(global shift)가 제거됨
- 국소적인 fault pattern이 더 부각됨

### 실험 설계

| 실험 | Feature | 비고 |
|-----|---------|-----|
| cpd_res_k1 | [f \| ctx \| res] | k=1, 3D feature |
| cpd_res_k2 | [f \| ctx \| res] | k=2, 3D feature |

### 주의사항
- Residual이 노이즈를 증폭시킬 수 있음
- Robust mean (median, trimmed mean) 고려

---

## Phase 3: Row-structured Matching (Optional)

### 목표
Row-level prototype을 활용한 2-stage detection

### 아이디어 A: Row Prototype Bank

```python
# 정상 이미지에서 각 row의 평균 prototype
row_prototype[i] = mean_j(f[i,j])

# 테스트: row-level anomaly 먼저 계산
row_score[i] = min_distance(row_prototype_query[i], row_bank)

# Patch score에 row score 가중
final_score[i,j] = patch_score[i,j] * (1 + alpha * row_score[i])
```

### 아이디어 B: Row-wise Weighted kNN

```python
# Query patch (i,j)의 kNN 검색 시 row similarity 반영
d_final = d_patch + lambda * d_row
```

### 구현 복잡도
- Phase 1,2 대비 높음
- Phase 1,2 실패 시 시도

---

## 실행 계획

```
Week 1: Phase 1 (CPD-PatchCore)
├── EDA: Horizontal context 효과 시각화
├── 구현: run_cpd_patchcore.py
├── 실험: k=1,2,3 on Domain C
└── 분석: vs baseline 비교

Week 2: Phase 2 (CPD + Residual) - Phase 1 결과에 따라
├── 구현: residual context 추가
├── 실험: cpd_res_k1, cpd_res_k2
└── 분석: vs Phase 1 비교

Week 3: Phase 3 (Row-structured) - Phase 2 결과에 따라
├── 구현: row prototype bank
├── 실험: row-weighted scoring
└── 최종 분석 및 논문화 준비
```

---

## 성공 기준

| Phase | 목표 | Baseline |
|-------|-----|----------|
| Phase 1 | > 90% accuracy | 89.44% (16,16) |
| Phase 2 | > 91% accuracy | Phase 1 결과 |
| Phase 3 | > 92% accuracy | Phase 2 결과 |

---

## 참고: 데이터 특성

- 원본 크기: 31×95 pixels
- DINOv2 입력: 518×518 (bilinear resize)
- Patch grid: 37×37 patches
- **결함 특성**: 가로로 얇은 band (row 단위)
- **Cold/Warm 차이**: 구조적/텍스처 차이 (intensity 7%만 설명)
