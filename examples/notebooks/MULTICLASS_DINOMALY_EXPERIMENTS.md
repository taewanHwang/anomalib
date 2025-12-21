# Dinomaly Multi-Class Experiments for HDMAP Dataset

## Background

Dinomaly (ECCV 2025)는 Multi-Class Anomaly Detection을 지향하는 최신 아키텍처입니다.
MVTec AD의 15개 클래스를 하나의 모델로 학습하는 **Unified Training** 방식을 제안했습니다.

### Original Paper vs Anomalib Implementation 분석

| 구성요소 | 원본 (guojiajeremy/Dinomaly) | Anomalib 현재 구현 | 상태 |
|---------|----------------------------|-------------------|------|
| Encoder | DINOv2-reg frozen | 동일 | OK |
| Bottleneck Dropout | **고정 0.2** | 0%→90% progressive | **오류** |
| Discarding Rate (k%) | 0%→90% warmup (1000 steps) | 동일 | OK |
| Hard Mining Factor | 0.1 | 동일 | OK |
| Target Layers | [2,3,4,5,6,7,8,9] | 동일 | OK |
| Fuse Layers | [[0,1,2,3], [4,5,6,7]] | 동일 | OK |

### 핵심 발견: Bottleneck Dropout 구현 오류

원본 논문/코드에서:
- **Bottleneck Dropout = 고정 0.2** (정보 병목 역할)
- **Discarding Rate (k%) = 0%→90%** (Hard Mining에서 easy point 제거 비율)

Anomalib에서 잘못 해석:
```python
# torch_model.py:139-148
bottle_neck_mlp = DinomalyMLP(
    drop=bottleneck_dropout,       # 0% (잘못됨, 0.2여야 함)
    drop_final=bottleneck_dropout_final,  # 90% (dropout이 아닌 discarding rate임)
    ...
)
```

## HDMAP Dataset Structure

```
datasets/HDMAP/1000_png/
├── domain_A/        # 1000 training, ~400 test samples
│   ├── train/good/
│   └── test/{good,fault}/
├── domain_B/
├── domain_C/        # 가장 어려운 도메인 (horizontal line defects)
└── domain_D/
```

## Experiments

### 1. Multi-Class vs Single-Class Baseline

**목표**: Dinomaly의 multi-class 능력을 HDMAP에서 검증

```bash
# Multi-class unified training (4 domains -> 1 model)
python dinomaly_multiclass_baseline.py \
    --mode compare \
    --max-steps 10000 \
    --batch-size 16 \
    --encoder dinov2reg_vit_base_14 \
    --gpu 0

# Multi-class only
python dinomaly_multiclass_baseline.py \
    --mode multiclass \
    --max-steps 10000

# Single-class only (baseline comparison)
python dinomaly_multiclass_baseline.py \
    --mode singleclass \
    --max-steps 10000
```

### 2. Encoder Size Comparison

```bash
# Small (faster, less memory)
python dinomaly_multiclass_baseline.py \
    --mode multiclass \
    --encoder dinov2reg_vit_small_14 \
    --max-steps 10000

# Base (default, balanced)
python dinomaly_multiclass_baseline.py \
    --mode multiclass \
    --encoder dinov2reg_vit_base_14 \
    --max-steps 10000

# Large (highest capacity)
python dinomaly_multiclass_baseline.py \
    --mode multiclass \
    --encoder dinov2reg_vit_large_14 \
    --max-steps 10000
```

### 3. Training Steps Analysis

```bash
# Quick test (5000 steps)
python dinomaly_multiclass_baseline.py \
    --mode compare \
    --max-steps 5000 \
    --gpu 0

# Full training (10000 steps, matches original paper)
python dinomaly_multiclass_baseline.py \
    --mode compare \
    --max-steps 10000 \
    --gpu 0

# Extended training (15000 steps)
python dinomaly_multiclass_baseline.py \
    --mode compare \
    --max-steps 15000 \
    --gpu 0
```

## Expected Results

Based on original paper (MVTec AD, 15 classes):
- Multi-class Unified: ~96% I-AUROC (average)
- Single-class: ~97% I-AUROC (average)
- Gap: ~1% (multi-class slightly lower, but uses 1 model instead of 15)

For HDMAP (4 domains), expected:
- Single-class: Domain-specific performance varies
  - Domain A, B, D: 90-95%
  - Domain C: 70-80% (challenging horizontal line defects)
- Multi-class: Potential for knowledge transfer between domains

## Key Metrics to Track

1. **Per-Domain AUROC**: How well does the unified model perform on each domain?
2. **Overall AUROC**: Combined performance across all test samples
3. **Training Dynamics**: Loss convergence, gradient norms
4. **Domain Transfer**: Does training on all domains help harder domains (C)?

## Previous Adaptive Experiments Analysis

From `hdmap_adaptive_validation.py` experiments:
- APE-adaptive discarding showed mixed results
- Domain C remained challenging
- NaN issues occurred occasionally (fixed with gradient clipping)

These experiments tested **modifying the discarding rate** based on structure features,
which is the correct parameter to modify (not dropout).

## Next Steps

1. Run multi-class baseline experiments
2. Compare with single-class results
3. Analyze if multi-class training helps Domain C
4. Consider domain-specific fine-tuning after unified pre-training

## Output Structure

```
results/dinomaly_multiclass_baseline/
└── {timestamp}/
    ├── experiment_settings.json
    ├── multiclass_unified/
    │   ├── checkpoints/
    │   ├── tensorboard/
    │   └── results.json
    ├── singleclass_{domain}/
    │   └── results.json
    └── final_summary.json
```

## References

- Original Paper: "Dinomaly: An Effective Reconstruction-Based Anomaly Detection"
- Original Repo: https://github.com/guojiajeremy/Dinomaly
- Key file: `dinomaly_mvtec_uni.py` (Multi-class unified training)
