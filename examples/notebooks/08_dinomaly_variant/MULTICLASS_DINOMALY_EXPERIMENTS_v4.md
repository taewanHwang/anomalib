# Dinomaly Multi-Class Experiments v4: K/V Row-wise Dropout

## Overview

이 문서는 **Decoder K/V Row-wise Dropout** 방법론을 사용한 Dinomaly 실험을 기록합니다.
Bottleneck Horizontal Dropout이 Global Attention으로 인해 무력화되는 문제를 해결하기 위해,
**Decoder의 K/V (Key/Value) 경로에 직접 Row-wise Dropout을 적용**합니다.

## Background: 왜 K/V Dropout인가?

### v3.1 Bottleneck Dropout 실패 원인

**Expert Analysis (v3.1 Post-mortem):**

| 문제점 | 설명 | 영향 |
|--------|------|------|
| **진정한 Bottleneck 아님** | 196/784 tokens → 196/784 tokens (차원 보존) | 압축 효과 없음 |
| **Global Mixing** | 8개 Decoder Layer가 Global Attention으로 복구 | Dropout 효과 희석 |
| **Large Dropout = Regularization** | 강한 dropout → 전체 복원 품질 저하 | FP 증가 |

**핵심 통찰:**
```
❌ Bottleneck Dropout: "약간 노이즈 섞인 토큰" → Decoder가 Global Context로 복구
✅ K/V Dropout: "복원 재료(V) 자체 제거" → 복원 원천 차단
```

### Bottleneck vs K/V Dropout 비교

```
┌─────────────────────────────────────────────────────────────────┐
│  Bottleneck Dropout (v3.1):                                      │
│                                                                  │
│  Encoder → [Dropout(x)] → Decoder (8 layers)                    │
│              ↑                    ↑                              │
│        perturbation      Global Attention recovers              │
│                          from 782 remaining tokens               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  K/V Dropout (v4):                                               │
│                                                                  │
│  Encoder → Bottleneck → Decoder Layer [Attn: K, V_dropped, Q]   │
│                              ↑                                   │
│                    V tokens missing = no recovery material       │
│                    특정 row의 복원 정보 원천 차단                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Method Description

### 핵심 아이디어

**LinearAttention 내부에서 V(Value) 토큰에 Row-wise Dropout 적용**

```python
# LinearAttention 원본
q, k, v = qkv[0], qkv[1], qkv[2]
kv = torch.matmul(k.transpose(-2, -1), v)
x = torch.matmul(q, kv) * z

# K/V Dropout 적용 후
q, k, v = qkv[0], qkv[1], qkv[2]
v = V_row_segment_dropout(v)  # ← 복원 재료 자체 제거
kv = torch.matmul(k.transpose(-2, -1), v)
x = torch.matmul(q, kv) * z
```

### Safety Mechanisms (5가지 안정장치)

| # | 안정장치 | 설명 | 기본값 | Ablation 가능 |
|---|----------|------|--------|--------------|
| 1 | **V-only Masking** | K는 보존, V만 dropout | ✅ 활성화 | `v_only=True/False` |
| 2 | **Head-wise Dropout** | 일부 head에만 적용 | 50% heads | `head_ratio=0.5` |
| 3 | **Layer-wise Scheduling** | 후반 layer에만 적용 | 마지막 4개 | `apply_layers=[4,5,6,7]` |
| 4 | **Warmup Schedule** | 점진적 dropout 증가 | 200 steps | `warmup_steps=200` |
| 5 | **Row-internal Segment** | 전체 row 아닌 segment | seg_len=2 | `seg_len=2`, `row_p=0.1` |

### 파라미터

| 파라미터 | 의미 | 기본값 | 범위 | 비고 |
|---------|------|--------|------|------|
| `v_drop_p` | V dropout 확률 | **0.05** | 0.03-0.1 | 매우 보수적 시작 |
| `band_width` | Row 내 segment 길이 | **2** | 1-3 | 결함 형태 반영 |
| `row_p` | Row 선택 확률 | **0.1** | 0.05-0.2 | Bottleneck보다 낮게 |
| `head_ratio` | Dropout 적용 head 비율 | **0.5** | 0.25-1.0 | 1.0=전체 head |
| `apply_layers` | 적용할 layer indices | **[4,5,6,7]** | 마지막 4개 | 빈 리스트=적용 안함 |
| `warmup_steps` | Warmup 기간 | **200** | 0-500 | 0=즉시 적용 |
| `v_only` | V만 dropout (K 보존) | **True** | True/False | False=K+V 동시 |

---

## Architecture

### LinearAttentionWithVDropout

```python
class LinearAttentionWithVDropout(LinearAttention):
    """LinearAttention with V-only Row-wise Segment Dropout.

    Key insight: Drop V tokens → remove reconstruction material
    Unlike bottleneck dropout, this cannot be recovered by global attention.

    Safety mechanisms (all ablatable):
    1. V-only masking (preserve K for attention stability)
    2. Head-wise selective application
    3. Layer index check (apply only to specified layers)
    4. Warmup schedule (gradual increase)
    5. Row-internal segment dropout
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        # V Dropout parameters
        v_drop_p: float = 0.05,
        row_p: float = 0.1,
        band_width: int = 2,
        head_ratio: float = 0.5,
        v_only: bool = True,
        # Layer-wise control (set by model)
        apply_dropout: bool = True,
        # Warmup (managed externally)
        **kwargs,
    ):
        ...
```

### VRowSegmentDropout 모듈

```python
class VRowSegmentDropout(nn.Module):
    """Row-wise segment dropout for V (Value) tensors.

    Unlike HorizontalSegmentDropout, this operates on:
    - V tensor shape: (batch, num_heads, seq_len, head_dim)
    - Drops segments within rows across specified heads

    Key differences from bottleneck dropout:
    1. Applied inside attention (cannot be recovered by downstream layers)
    2. V-only preserves attention computation stability
    3. Head-wise control for gradual ablation
    """

    def __init__(
        self,
        side: int,          # Token grid side (28 for 392/14)
        row_p: float = 0.1,
        band_width: int = 2,
        seg_drop_p: float = 0.8,
        head_ratio: float = 0.5,
        v_only: bool = True,
    ):
        ...
```

### 전체 모델 구조

```
DinomalyKVDropout
├── encoder (frozen)
├── bottleneck (trainable, no horizontal dropout)
└── decoder (trainable)
    ├── Layer 0: DecoderViTBlock with LinearAttention (no V dropout)
    ├── Layer 1: DecoderViTBlock with LinearAttention (no V dropout)
    ├── Layer 2: DecoderViTBlock with LinearAttention (no V dropout)
    ├── Layer 3: DecoderViTBlock with LinearAttention (no V dropout)
    ├── Layer 4: DecoderViTBlock with LinearAttentionWithVDropout ✓
    ├── Layer 5: DecoderViTBlock with LinearAttentionWithVDropout ✓
    ├── Layer 6: DecoderViTBlock with LinearAttentionWithVDropout ✓
    └── Layer 7: DecoderViTBlock with LinearAttentionWithVDropout ✓
```

---

## Experiment Design

### 기준선 (Baseline)

| Config | TopK Loss | Bottleneck Horiz. | K/V Dropout | 비고 |
|--------|-----------|-------------------|-------------|------|
| **v3.1 Best** | ✅ q=2% | ✅ default | ❌ | 81.20% Domain C TPR@1% |

### v4.0 Ablation 설계

| Config | TopK | V Dropout | Head % | Layers | Warmup | 목적 |
|--------|------|-----------|--------|--------|--------|------|
| **H. V Dropout Only** | ✅ | ✅ p=0.05 | 50% | 4-7 | 200 | **Primary: V Dropout 효과** |
| **I. Strong V Drop** | ✅ | ✅ p=0.1 | 50% | 4-7 | 200 | Stronger V dropout |
| **J. All Heads** | ✅ | ✅ p=0.05 | 100% | 4-7 | 200 | Head-wise ablation |
| **K. All Layers** | ✅ | ✅ p=0.05 | 50% | 0-7 | 200 | Layer-wise ablation |
| **L. No Warmup** | ✅ | ✅ p=0.05 | 50% | 4-7 | 0 | Warmup ablation |
| **M. K+V Dropout** | ✅ | ✅ p=0.05 (K+V) | 50% | 4-7 | 200 | V-only ablation |

### 성공 기준

- **Domain C TPR@1%**: 81.20% → **83%+** (v3.1 대비 추가 개선)
- 다른 도메인 성능 유지 또는 개선
- 정상 Score 분포 안정 (FP 증가 없음)

---

## Implementation

### 파일 구조

```
src/anomalib/models/image/dinomaly_variants/
├── kv_dropout.py           # [NEW] VRowSegmentDropout 모듈
├── kv_attention.py         # [NEW] LinearAttentionWithVDropout
├── kv_dropout_model.py     # [NEW] DinomalyKVDropout 모델
└── __init__.py             # [MODIFY] export 추가

examples/notebooks/
├── dinomaly_kv_dropout.py  # [NEW] 실험 스크립트
└── MULTICLASS_DINOMALY_EXPERIMENTS_v4.md  # 이 문서
```

### 구현 순서

1. [ ] `VRowSegmentDropout` 모듈 구현
2. [ ] `LinearAttentionWithVDropout` 구현
3. [ ] `DinomalyKVDropout` 모델 구현
4. [ ] 실험 스크립트 `dinomaly_kv_dropout.py`
5. [ ] Config H 실험 (Primary)
6. [ ] Safety mechanism ablation (I-M)
7. [ ] 결과 분석 및 문서화

---

## Experiment Execution

### Config H: V Dropout (Primary)

```bash
# V-only dropout with all safety mechanisms
nohup python examples/notebooks/dinomaly_kv_dropout.py \
    --max-steps 3000 --seed 42 --gpu 0 \
    --use-topk --q-percent 2.0 \
    --v-drop-p 0.05 --row-p 0.1 --band-width 2 \
    --head-ratio 0.5 --apply-layers 4 5 6 7 \
    --warmup-steps 200 --v-only \
    --result-dir results/dinomaly_kv_dropout/config_H_primary \
    > logs/kv_config_H.log 2>&1 &
```

### Config I: Strong V Dropout

```bash
nohup python examples/notebooks/dinomaly_kv_dropout.py \
    --max-steps 3000 --seed 42 --gpu 1 \
    --use-topk --q-percent 2.0 \
    --v-drop-p 0.1 --row-p 0.1 --band-width 2 \
    --head-ratio 0.5 --apply-layers 4 5 6 7 \
    --warmup-steps 200 --v-only \
    --result-dir results/dinomaly_kv_dropout/config_I_strong \
    > logs/kv_config_I.log 2>&1 &
```

### Config J: All Heads

```bash
nohup python examples/notebooks/dinomaly_kv_dropout.py \
    --max-steps 3000 --seed 42 --gpu 2 \
    --use-topk --q-percent 2.0 \
    --v-drop-p 0.05 --row-p 0.1 --band-width 2 \
    --head-ratio 1.0 --apply-layers 4 5 6 7 \
    --warmup-steps 200 --v-only \
    --result-dir results/dinomaly_kv_dropout/config_J_all_heads \
    > logs/kv_config_J.log 2>&1 &
```

### Config K: All Layers

```bash
nohup python examples/notebooks/dinomaly_kv_dropout.py \
    --max-steps 3000 --seed 42 --gpu 3 \
    --use-topk --q-percent 2.0 \
    --v-drop-p 0.05 --row-p 0.1 --band-width 2 \
    --head-ratio 0.5 --apply-layers 0 1 2 3 4 5 6 7 \
    --warmup-steps 200 --v-only \
    --result-dir results/dinomaly_kv_dropout/config_K_all_layers \
    > logs/kv_config_K.log 2>&1 &
```

### Config L: No Warmup

```bash
nohup python examples/notebooks/dinomaly_kv_dropout.py \
    --max-steps 3000 --seed 42 --gpu 4 \
    --use-topk --q-percent 2.0 \
    --v-drop-p 0.05 --row-p 0.1 --band-width 2 \
    --head-ratio 0.5 --apply-layers 4 5 6 7 \
    --warmup-steps 0 --v-only \
    --result-dir results/dinomaly_kv_dropout/config_L_no_warmup \
    > logs/kv_config_L.log 2>&1 &
```

### Config M: K+V Dropout

```bash
nohup python examples/notebooks/dinomaly_kv_dropout.py \
    --max-steps 3000 --seed 42 --gpu 5 \
    --use-topk --q-percent 2.0 \
    --v-drop-p 0.05 --row-p 0.1 --band-width 2 \
    --head-ratio 0.5 --apply-layers 4 5 6 7 \
    --warmup-steps 200 \
    --result-dir results/dinomaly_kv_dropout/config_M_kv_dropout \
    > logs/kv_config_M.log 2>&1 &
```

---

## Theoretical Justification

### 왜 K/V Dropout이 효과적인가?

1. **복원 재료 직접 제거**
   - V = reconstruction source
   - V dropout = 복원 정보 원천 차단
   - Global Attention이 있어도 V가 없으면 복원 불가

2. **우회 불가**
   - Bottleneck: 8개 layer가 복구
   - K/V: 각 layer에서 직접 적용, layer별 복구 불가

3. **안정장치로 부작용 최소화**
   - V-only: K 보존으로 attention 패턴 유지
   - Head-wise: 일부 head만 → 전체 정보 보존
   - Layer-wise: 후반 layer → 초기 특징 추출 보존
   - Warmup: 학습 초기 안정성 보장

### 수학적 해석

```
# LinearAttention 수식
output = Q × (K^T × V) × Z

# V dropout 적용 시
V' = V ⊙ mask  (mask: row-segment dropout)
output' = Q × (K^T × V') × Z

# 의미:
# - K^T × V' = row-dropped kv interaction
# - 특정 row의 정보가 kv에 반영되지 않음
# - Q가 해당 row 정보 요청해도 V에 없음 → 복원 불가
```

---

## Expected Outcomes

### 긍정적 시나리오

| Metric | v3.1 Baseline | v4.0 Target | 비고 |
|--------|---------------|-------------|------|
| Domain C TPR@1% | 81.20% | **83%+** | +1.8%p 이상 |
| Domain C AUROC | 97.82% | **98%+** | +0.2%p |
| Mean TPR@1% | 91.15% | **92%+** | 전체 개선 |

### 위험 요소

| 위험 | 완화 방법 | 모니터링 |
|------|----------|----------|
| 정상 복원 품질 저하 | 낮은 v_drop_p, warmup | Good sample score 분포 |
| 학습 불안정 | Head-wise, layer-wise 제한 | Loss curve |
| 다른 도메인 성능 저하 | Conservative defaults | Per-domain AUROC |

---

## References

- v3 문서: `MULTICLASS_DINOMALY_EXPERIMENTS_v3.md`
- Plan 문서: `/home/taewan.hwang/.claude/plans/abstract-brewing-shannon.md`
- TopK Loss 구현: `src/anomalib/models/image/dinomaly_variants/topk_model.py`
- Horizontal Dropout 구현: `src/anomalib/models/image/dinomaly_variants/horizontal_model.py`
