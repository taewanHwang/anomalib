# 004. CA-PatchCore (Condition-Aware PatchCore) Experiments

## Configuration
- Backbone: `vit_base_patch14_dinov2`
- Score: **Raw distance** (same as 003 baseline)
- Gating Modes: oracle, p90, mixed, random

### P90 Thresholds (from EDA)
- Domain A: 0.2985
- Domain B: 0.3128
- Domain C: 0.3089
- Domain D: 0.2919

---

## Results Summary (k=1)

### Domain C - Gating Mode Comparison

| Gating | Overall | Cold | Warm | Gating Acc |
|--------|---------|------|------|------------|
| mixed | 77.47% | 71.70% | 83.20% | - |
| random | 74.49% | 68.97% | 79.98% | 48.23% |
| p90 | 74.19% | 72.21% | 76.16% | 97.27% |
| oracle | 73.99% | 72.01% | 75.96% | 100% |

### All Domains - Oracle (k=1)

| Domain | Overall | Cold | Warm |
|--------|---------|------|------|
| A | 85.76% | 82.35% | 89.13% |
| B | 86.92% | 83.47% | 90.34% |
| **C** | **73.99%** | **72.01%** | **75.96%** |
| D | 83.94% | 81.14% | 86.72% |

### All Domains - P90 (k=1)

| Domain | Overall | Cold | Warm | Gating Acc |
|--------|---------|------|------|------------|
| A | 85.76% | 82.35% | 89.13% | 100% |
| B | 87.22% | 83.47% | 90.95% | 91.97% |
| **C** | **74.19%** | **72.21%** | **76.16%** | **97.27%** |
| D | 83.74% | 80.83% | 86.62% | 82.53% |

### vs Baseline (1,1) Comparison

| Method | Domain C Overall | vs Baseline |
|--------|------------------|-------------|
| Baseline (1,1) | 77.42% | - |
| CA-Mixed k=1 | 77.47% | +0.05%p |
| CA-P90 k=1 | 74.19% | -3.23%p |
| CA-Oracle k=1 | 73.99% | -3.43%p |

**Finding**: k=1에서 CA-PatchCore가 baseline보다 오히려 성능 하락. Memory bank diversity 부족 추정.

---

## Commands

### 1. Domain C - All Gating Modes (k=1) [DONE]

```bash
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py --domain domain_C --k-per-bank 1 --gating oracle > logs/004_ca_C_oracle_k1.log 2>&1 &
sleep 2
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py --domain domain_C --k-per-bank 1 --gating p90 > logs/004_ca_C_p90_k1.log 2>&1 &
sleep 2
CUDA_VISIBLE_DEVICES=2 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py --domain domain_C --k-per-bank 1 --gating mixed > logs/004_ca_C_mixed_k1.log 2>&1 &
sleep 2
CUDA_VISIBLE_DEVICES=3 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py --domain domain_C --k-per-bank 1 --gating random > logs/004_ca_C_random_k1.log 2>&1 &
```

### 2. All Domains - P90 & Oracle (k=1) [DONE]

```bash
# P90
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py --domain domain_A --k-per-bank 1 --gating p90 > logs/004_ca_A_p90_k1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py --domain domain_B --k-per-bank 1 --gating p90 > logs/004_ca_B_p90_k1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py --domain domain_C --k-per-bank 1 --gating p90 > logs/004_ca_C_p90_k1.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py --domain domain_D --k-per-bank 1 --gating p90 > logs/004_ca_D_p90_k1.log 2>&1 &

# Oracle
CUDA_VISIBLE_DEVICES=4 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py --domain domain_A --k-per-bank 1 --gating oracle > logs/004_ca_A_oracle_k1.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py --domain domain_B --k-per-bank 1 --gating oracle > logs/004_ca_B_oracle_k1.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py --domain domain_C --k-per-bank 1 --gating oracle > logs/004_ca_C_oracle_k1.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py --domain domain_D --k-per-bank 1 --gating oracle > logs/004_ca_D_oracle_k1.log 2>&1 &
```

---

### 3. Domain C - k=4 Experiments (8 samples total)

```bash
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py --domain domain_C --k-per-bank 4 --gating oracle > logs/004_ca_C_oracle_k4.log 2>&1 &
sleep 2
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py --domain domain_C --k-per-bank 4 --gating p90 > logs/004_ca_C_p90_k4.log 2>&1 &
sleep 2
CUDA_VISIBLE_DEVICES=2 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py --domain domain_C --k-per-bank 4 --gating mixed > logs/004_ca_C_mixed_k4.log 2>&1 &
sleep 2
CUDA_VISIBLE_DEVICES=3 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py --domain domain_C --k-per-bank 4 --gating random > logs/004_ca_C_random_k4.log 2>&1 &
```

### 4. Domain C - k=16 Experiments (32 samples total)

```bash
CUDA_VISIBLE_DEVICES=4 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py --domain domain_C --k-per-bank 16 --gating oracle > logs/004_ca_C_oracle_k16.log 2>&1 &
sleep 2
CUDA_VISIBLE_DEVICES=5 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py --domain domain_C --k-per-bank 16 --gating p90 > logs/004_ca_C_p90_k16.log 2>&1 &
sleep 2
CUDA_VISIBLE_DEVICES=6 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py --domain domain_C --k-per-bank 16 --gating mixed > logs/004_ca_C_mixed_k16.log 2>&1 &
sleep 2
CUDA_VISIBLE_DEVICES=7 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/004_ca_patchcore/run_ca_patchcore.py --domain domain_C --k-per-bank 16 --gating random > logs/004_ca_C_random_k16.log 2>&1 &
```

---

## Check Progress

```bash
ps aux | grep run_ca_patchcore | grep -v grep
tail -5 logs/004_*.log
```

---

## Results: Domain C - k Scaling Experiments

### Comparison Table

| k | Gating | Overall | Cold | Warm | Gating Acc | vs Baseline |
|---|--------|---------|------|------|------------|-------------|
| 1 | oracle | 73.99% | 72.01% | 75.96% | 100% | **-3.43%p** vs (1,1) 77.42% |
| 1 | p90 | 74.19% | 72.21% | 76.16% | 97.27% | **-3.23%p** vs (1,1) 77.42% |
| 1 | mixed | 77.47% | 71.70% | 83.20% | - | +0.05%p vs (1,1) 77.42% |
| 1 | random | 74.49% | 68.97% | 79.98% | 48.23% | -2.93%p |
| | | | | | | |
| 4 | oracle | 80.81% | 79.41% | 82.19% | 100% | **-3.94%p** vs (4,4) 84.75% |
| 4 | p90 | 80.81% | 79.41% | 82.19% | 97.27% | **-3.94%p** vs (4,4) 84.75% |
| 4 | mixed | 84.14% | 79.41% | 88.83% | - | -0.61%p vs (4,4) 84.75% |
| 4 | random | 81.06% | 75.25% | 86.82% | 48.23% | -3.69%p |
| | | | | | | |
| 16 | oracle | 87.42% | 87.22% | 87.63% | 100% | -2.02%p vs (16,16) 89.44% |
| 16 | p90 | 87.53% | 86.82% | 88.23% | 97.27% | -1.91%p vs (16,16) 89.44% |
| 16 | **mixed** | **90.25%** | **88.74%** | **91.75%** | - | **+0.81%p** vs (16,16) 89.44% ✓ |
| 16 | random | 86.36% | 83.67% | 89.03% | 48.23% | -3.08%p |

### Key Findings

1. **Mixed gating이 k=16에서 baseline을 초과 달성** (+0.81%p)
   - Baseline (16,16): 89.44%
   - CA-Mixed k=16: 90.25%

2. **Oracle/P90 gating은 모든 k에서 baseline 대비 성능 하락**
   - k=1: -3.43%p / -3.23%p
   - k=4: -3.94%p / -3.94%p
   - k=16: -2.02%p / -1.91%p
   - 원인 추정: cold/warm 분리로 memory bank diversity 감소

3. **k 증가에 따른 성능 개선**
   - k=1 → k=4: +6.8%p (mixed), +6.6%p (oracle)
   - k=4 → k=16: +6.1%p (mixed), +6.6%p (oracle)

4. **Mixed mode가 gating 보다 항상 우수**
   - cold/warm 분리 없이 전체 memory bank 사용이 더 효과적
   - 이는 condition 분리보다 memory diversity가 더 중요함을 시사

### Baseline Reference (003 Results)

| Config | Overall | Cold | Warm |
|--------|---------|------|------|
| (1,1) | 77.42% | 75.15% | 79.68% |
| (4,4) | 84.75% | 80.69% | 88.77% |
| (16,16) | 89.44% | 86.31% | 92.56% |

---

## Gating Modes Description

| Mode | Description | Expected Gating Acc |
|------|-------------|---------------------|
| oracle | Ground truth labels | 100% |
| p90 | P90 intensity threshold | ~97% |
| mixed | Both banks combined (no gating) | N/A |
| random | Random selection | ~50% |
