# 003. PatchCore Baseline Experiments

## Configuration
- Backbone: `vit_base_patch14_dinov2`
- Score: **Raw distance** (`enable_normalization=False`)
- Target Size: 518x518
- Resize Method: resize_bilinear

---

## Results Summary

### Full Baseline (1000 training samples)

| Domain | Overall Acc | Cold Acc | Warm Acc | AUROC |
|--------|-------------|----------|----------|-------|
| A | 99.09% | 99.09% | 99.09% | 99.85% |
| B | 99.19% | 98.99% | 99.39% | 99.82% |
| **C** | **94.75%** | **91.63%** | **97.87%** | **98.21%** |
| D | 98.99% | 98.39% | 99.60% | 99.77% |

### Few-Shot Baseline

#### (1,1) - 2 reference samples

| Domain | Overall Acc | Cold Acc | Warm Acc | AUROC |
|--------|-------------|----------|----------|-------|
| A | 87.88% | 83.87% | 91.90% | 85.31% |
| B | 88.18% | 83.77% | 92.61% | 86.65% |
| **C** | **77.42%** | **73.08%** | **81.78%** | **75.10%** |
| D | 86.11% | 83.87% | 88.36% | 85.20% |

#### (4,4) - 8 reference samples

| Domain | Overall Acc | Cold Acc | Warm Acc | AUROC |
|--------|-------------|----------|----------|-------|
| A | 91.57% | 88.61% | 94.53% | 92.77% |
| B | 90.05% | 86.39% | 93.72% | 88.11% |
| **C** | **84.75%** | **81.15%** | **88.36%** | **89.51%** |
| D | 91.36% | 90.63% | 92.11% | 91.90% |

#### (16,16) - 32 reference samples

| Domain | Overall Acc | Cold Acc | Warm Acc | AUROC |
|--------|-------------|----------|----------|-------|
| A | 95.61% | 93.75% | 97.47% | 96.96% |
| B | 93.18% | 90.93% | 95.45% | 93.25% |
| **C** | **89.44%** | **87.40%** | **91.50%** | **95.35%** |
| D | 94.65% | 94.15% | 95.14% | 98.35% |

---

## Analysis

### Domain C Performance Gap (Cold vs Warm)

| Config | Cold Acc | Warm Acc | Gap |
|--------|----------|----------|-----|
| Full | 91.63% | 97.87% | 6.24%p |
| (16,16) | 87.40% | 91.50% | 4.10%p |
| (4,4) | 81.15% | 88.36% | 7.21%p |
| (1,1) | 73.08% | 81.78% | 8.70%p |

### Few-Shot Scaling (Domain C)

| Config | Overall Acc | vs Full Gap |
|--------|-------------|-------------|
| Full (1000) | 94.75% | - |
| (16,16) | 89.44% | -5.31%p |
| (4,4) | 84.75% | -10.00%p |
| (1,1) | 77.42% | -17.33%p |

---

## Key Findings

1. **Domain C is hardest**: 94.75% (full) vs 99%+ (other domains)
2. **Cold samples underperform**: 91.63% cold vs 97.87% warm in Domain C
3. **Few-shot degrades gracefully**: (16,16) achieves 89.44% with only 32 samples
4. **CA-PatchCore baseline**: (1,1) = 77.42% → target for improvement

---

## Commands

<details>
<summary>Full Baseline Commands</summary>

```bash
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/003_baseline_patchcore/run_baseline.py --domain domain_A --mode full --backbone vit_base_patch14_dinov2 > logs/003_full_base_A.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/003_baseline_patchcore/run_baseline.py --domain domain_B --mode full --backbone vit_base_patch14_dinov2 > logs/003_full_base_B.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/003_baseline_patchcore/run_baseline.py --domain domain_C --mode full --backbone vit_base_patch14_dinov2 > logs/003_full_base_C.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/003_baseline_patchcore/run_baseline.py --domain domain_D --mode full --backbone vit_base_patch14_dinov2 > logs/003_full_base_D.log 2>&1 &
```

</details>

<details>
<summary>Few-Shot (1,1) Commands</summary>

```bash
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/003_baseline_patchcore/run_baseline.py --domain domain_A --mode fewshot --n-cold 1 --n-warm 1 --backbone vit_base_patch14_dinov2 > logs/003_fewshot_1_1_A.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/003_baseline_patchcore/run_baseline.py --domain domain_B --mode fewshot --n-cold 1 --n-warm 1 --backbone vit_base_patch14_dinov2 > logs/003_fewshot_1_1_B.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/003_baseline_patchcore/run_baseline.py --domain domain_C --mode fewshot --n-cold 1 --n-warm 1 --backbone vit_base_patch14_dinov2 > logs/003_fewshot_1_1_C.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/003_baseline_patchcore/run_baseline.py --domain domain_D --mode fewshot --n-cold 1 --n-warm 1 --backbone vit_base_patch14_dinov2 > logs/003_fewshot_1_1_D.log 2>&1 &
```

</details>

<details>
<summary>Few-Shot (4,4) Commands</summary>

```bash
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/003_baseline_patchcore/run_baseline.py --domain domain_A --mode fewshot --n-cold 4 --n-warm 4 --backbone vit_base_patch14_dinov2 > logs/003_fewshot_4_4_A.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/003_baseline_patchcore/run_baseline.py --domain domain_B --mode fewshot --n-cold 4 --n-warm 4 --backbone vit_base_patch14_dinov2 > logs/003_fewshot_4_4_B.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/003_baseline_patchcore/run_baseline.py --domain domain_C --mode fewshot --n-cold 4 --n-warm 4 --backbone vit_base_patch14_dinov2 > logs/003_fewshot_4_4_C.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/003_baseline_patchcore/run_baseline.py --domain domain_D --mode fewshot --n-cold 4 --n-warm 4 --backbone vit_base_patch14_dinov2 > logs/003_fewshot_4_4_D.log 2>&1 &
```

</details>

<details>
<summary>Few-Shot (16,16) Commands</summary>

```bash
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/003_baseline_patchcore/run_baseline.py --domain domain_A --mode fewshot --n-cold 16 --n-warm 16 --backbone vit_base_patch14_dinov2 > logs/003_fewshot_16_16_A.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/003_baseline_patchcore/run_baseline.py --domain domain_B --mode fewshot --n-cold 16 --n-warm 16 --backbone vit_base_patch14_dinov2 > logs/003_fewshot_16_16_B.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/003_baseline_patchcore/run_baseline.py --domain domain_C --mode fewshot --n-cold 16 --n-warm 16 --backbone vit_base_patch14_dinov2 > logs/003_fewshot_16_16_C.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup .venv/bin/python examples/notebooks/10_patchcore_variant/003_baseline_patchcore/run_baseline.py --domain domain_D --mode fewshot --n-cold 16 --n-warm 16 --backbone vit_base_patch14_dinov2 > logs/003_fewshot_16_16_D.log 2>&1 &
```

</details>
