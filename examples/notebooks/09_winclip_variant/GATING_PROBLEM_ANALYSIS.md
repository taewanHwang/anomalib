# CA-WinCLIP Gating Problem Analysis

## 1. Problem Context

### Dataset: HDMAP
- Industrial sensor data with **Cold/Warm conditions**
- Each domain has 2000 test samples:
  - `fault/cold`: indices 0-499 (file 0-499)
  - `fault/warm`: indices 500-999 (file 500-999)
  - `good/cold`: indices 1000-1499 (file 0-499)
  - `good/warm`: indices 1500-1999 (file 500-999)
- Cold images: low intensity (~0.19 mean)
- Warm images: high intensity (~0.28 mean)

### Model: WinCLIP (Zero-shot Anomaly Detection)
- Uses CLIP embeddings for anomaly scoring
- Problem: Cold Fault vs Warm Normal scores overlap (64.25% AUROC)

### Proposed Solution: Condition-Aware WinCLIP (CA-WinCLIP)
- Separate reference banks for Cold and Warm conditions
- **Gating mechanism** to select appropriate bank before scoring
- If gating works perfectly, each condition only compared with same-condition reference

---

## 2. Core Gating Problem

### Goal
Given a test image, determine whether it's **Cold** or **Warm** condition to select the correct reference bank.

### Available Information
- Cold reference image: `good/cold` (index 1000, file 000000.tiff)
- Warm reference image: `good/warm` (index 1999, file 000999.tiff)
- Test image embeddings (global and patch-level)

### Current Approach: Global Embedding Similarity
```python
# Compute cosine similarity between test and each reference
sim_cold = cosine_similarity(test_global_emb, cold_ref_global_emb)
sim_warm = cosine_similarity(test_global_emb, warm_ref_global_emb)
selected_bank = "cold" if sim_cold > sim_warm else "warm"
```

### Problem: Trade-off Between Condition Groups

| Group | Global | Patch Mean | Patch Median |
|-------|--------|------------|--------------|
| fault/cold | **60%** | 95% | 95% |
| fault/warm | 100% | 100% | 100% |
| good/cold | 95% | 100% | 95% |
| good/warm | 95% | **40%** | **55%** |

**Key Insight:**
- Global embedding fails on **fault/cold** (60%)
- Patch-level methods fail on **good/warm** (40-55%)
- No single method works for all groups

---

## 3. Attempted Solutions

### 3.1 Global Embedding (Baseline)
```python
selected = "cold" if sim_cold_global > sim_warm_global else "warm"
```
- **Result:** 87.5% overall
- **Problem:** fault/cold only 60%

### 3.2 Patch-level Mean
```python
# 225 patches (15x15 grid)
sim_cold_patches = cosine_sim(test_patches, cold_ref_patches)  # (225,)
sim_warm_patches = cosine_sim(test_patches, warm_ref_patches)  # (225,)
selected = "cold" if sim_cold_patches.mean() > sim_warm_patches.mean() else "warm"
```
- **Result:** 83.8% overall
- **Problem:** good/warm only 40%

### 3.3 Patch-level Median (Robust to outliers)
```python
selected = "cold" if sim_cold_patches.median() > sim_warm_patches.median() else "warm"
```
- **Result:** 86.2% overall
- **Problem:** good/warm only 55%

### 3.4 Ensemble (Majority Voting)
```python
votes = [global_selected, patch_mean_selected, patch_median_selected]
selected = "cold" if votes.count("cold") > votes.count("warm") else "warm"
```
- **Result:** 87.5% overall (same as global)
- fault/cold improved to 90%, but good/warm dropped to 60%

### 3.5 Confidence-based (Highest Margin)
```python
methods = {
    "global": (global_selected, abs(sim_cold_global - sim_warm_global)),
    "patch_mean": (patch_mean_selected, abs(sim_cold_mean - sim_warm_mean)),
    "patch_median": (patch_median_selected, abs(sim_cold_median - sim_warm_median)),
}
best_method = max(methods, key=lambda m: methods[m][1])  # highest margin
selected = methods[best_method][0]
```
- **Result:** 88.8% overall (best so far, +1.2%)
- fault/cold: 85%, good/warm: 70%

---

## 4. Detailed EDA Results

### Full Comparison Table

| Group | Global | P.Mean | P.Median | Ensemble | Confidence |
|-------|--------|--------|----------|----------|------------|
| fault/cold | 60.0% | 95.0% | 95.0% | 90.0% | 85.0% |
| fault/warm | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| good/cold | 95.0% | 100.0% | 95.0% | 100.0% | 100.0% |
| good/warm | 95.0% | 40.0% | 55.0% | 60.0% | 70.0% |
| **OVERALL** | **87.5%** | 83.8% | 86.2% | 87.5% | **88.8%** |

### Pattern Analysis

1. **Cold images (fault or good):**
   - Patch-level methods work better
   - Global embedding sometimes confuses cold with warm

2. **Warm images (fault or good):**
   - Global embedding works better
   - Patch-level methods often confuse warm with cold

3. **Why fault/cold fails with Global:**
   - Localized defects affect the global embedding
   - Defect patterns may have "warm-like" features

4. **Why good/warm fails with Patch:**
   - Warm images have similar patch similarities to both references
   - Small margins lead to wrong decisions

---

## 5. Oracle Experiment Results (Upper Bound)

Oracle 실험은 **완벽한 gating (Ground Truth 사용)**으로 CA-WinCLIP의 상한선을 측정.

### k=1 Results (bank당 1개 reference, 총 2개)

| Method | Overall | Cold-only | Warm-only | Cross (CF vs WG) | Gating Acc |
|--------|---------|-----------|-----------|------------------|------------|
| **Oracle k=1** | 84.86% | 70.84% | 99.41% | 71.01% | 100% |
| TopK k=1 | 81.13% | 64.61% | 99.35% | 62.37% | 87.25% |

### k=2 Results (bank당 2개 reference, 총 4개)

| Method | Overall | Cold-only | Warm-only | Cross (CF vs WG) | Gating Acc |
|--------|---------|-----------|-----------|------------------|------------|
| **Oracle k=2** | **87.19%** | **76.73%** | 99.61% | **73.52%** | 100% |
| TopK k=2 | 84.30% | 72.11% | 99.61% | 66.58% | 89.35% |

### Key Findings from Oracle

1. **Oracle이 Upper Bound를 보여줌:**
   - 완벽한 gating으로도 Overall ~87%, Cold-only ~77%
   - Gating 개선만으로는 한계가 있음

2. **Reference 수 증가 효과:**
   | Metric | k=1 → k=2 (Oracle) |
   |--------|-------------------|
   | Overall | 84.86% → 87.19% (+2.3%) |
   | Cold-only | 70.84% → 76.73% (+5.9%) |
   | Cross | 71.01% → 73.52% (+2.5%) |

3. **Cold-only가 핵심 병목:**
   - Oracle k=2에서도 Cold-only: **76.73%**로 낮음
   - Warm-only: 99.61% (거의 완벽)
   - **Gating 문제가 아니라 Cold 이미지에서 anomaly detection 자체가 어려움**

4. **TopK Gating Accuracy:**
   - k=1: 87.25% (EDA의 Global embedding 결과와 일치)
   - k=2: 89.35%
   - Reference 수 증가가 gating에도 약간 도움

### Implications

```
Oracle 결과가 시사하는 것:
┌─────────────────────────────────────────────────────────────┐
│ 1. Gating 100% 정확해도 Cold-only ~77% (Warm: 99%)         │
│ 2. Cold 이미지의 anomaly detection 자체가 어려운 문제       │
│ 3. Gating 개선 + Cold anomaly detection 개선 모두 필요     │
│ 4. k 증가가 Cold 성능에 효과적 (+5.9%)                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Root Cause Hypothesis

### The Asymmetric Similarity Problem

```
                    Cold Ref    Warm Ref
                    --------    --------
fault/cold          HIGH        MEDIUM      <- Defects add "warm-like" features
fault/warm          LOW         HIGH        <- Clear separation
good/cold           HIGH        LOW         <- Clear separation
good/warm           MEDIUM      HIGH        <- Patches see both as similar
```

**fault/cold issue:**
- Expected: Test is cold, so sim_cold >> sim_warm
- Reality: Localized defects add features that increase sim_warm
- Global embedding aggregates this, making sim_warm competitive

**good/warm issue with patch methods:**
- Expected: Test is warm, so sim_warm >> sim_cold
- Reality: At patch level, many patches have similar similarity to both refs
- Median/mean of small differences leads to wrong decision

---

## 7. What We Need

### Requirements for Complete Solution
1. **fault/cold accuracy >= 90%** (currently 60% with global)
2. **good/warm accuracy >= 90%** (currently 40-55% with patch)
3. **Other groups remain >= 95%**
4. **Overall accuracy >= 95%**

### Possible Approaches to Explore

1. **Adaptive Method Selection**
   - Learn when to use global vs patch based on image characteristics
   - Need a meta-classifier or heuristic

2. **Weighted Ensemble**
   - Instead of majority vote, weight methods differently
   - Possibly based on similarity magnitude or variance

3. **Different Patch Aggregation**
   - Top-K patches instead of all 225
   - Attention-weighted patches
   - Exclude boundary patches

4. **Multi-reference Approach**
   - Use multiple references per bank (k > 1)
   - May help stabilize similarity estimates

5. **Feature-level Gating**
   - Instead of similarity, use raw features
   - Train a small classifier on features

6. **Temperature/Threshold-based**
   - Different decision thresholds for different scenarios
   - Bias towards one method based on confidence

---

## 8. Code Location

```
examples/notebooks/09_winclip_variant/
├── eda_patch_gating.py           # EDA script with all methods
├── ca_winclip/
│   ├── gating.py                 # Gating implementations
│   └── condition_aware_model.py  # CA-WinCLIP wrapper
└── winclip_hdmap_ca_validation.py # Full validation script
```

### Key Functions in eda_patch_gating.py

```python
def analyze_patch_similarities(model, test_img, ref_cold_img, ref_warm_img):
    """
    Returns dict with:
    - global_cold/warm: global embedding similarities
    - patch_mean_cold/warm: patch-level mean similarities
    - patch_median_cold/warm: patch-level median similarities
    - ensemble_selected: majority vote result
    - confidence_selected: highest-margin method result
    - sim_cold_patches/sim_warm_patches: raw (225,) patch similarities
    """
```

---

## 9. Reproduction

```bash
cd /mnt/ex-disk/taewan.hwang/study/anomalib

# Run EDA with all methods
PYTHONUNBUFFERED=1 .venv/bin/python examples/notebooks/09_winclip_variant/eda_patch_gating.py \
    --gpu 0 --domain domain_C --num-samples 20

# Output: logs/eda_patch_gating_ensemble.log
```

---

## 10. Summary

### Gating Performance

**Current Best:** Confidence-based method with 88.8% overall gating accuracy

**Remaining Gating Gap:**
- fault/cold: 85% (need 90%+)
- good/warm: 70% (need 90%+)

### Oracle Upper Bound (Anomaly Detection with Perfect Gating)

| Metric | Oracle k=2 | Target |
|--------|------------|--------|
| Overall AUROC | 87.19% | 90%+ |
| Cold-only AUROC | **76.73%** | 85%+ |
| Warm-only AUROC | 99.61% | ✓ |
| Cross (CF vs WG) | 73.52% | 80%+ |

### Two Separate Problems

```
┌─────────────────────────────────────────────────────────────────┐
│ Problem 1: Gating Accuracy (현재 87-89%)                        │
│   - fault/cold에서 Global embedding이 실패 (60%)                │
│   - good/warm에서 Patch methods가 실패 (40-55%)                 │
│   - 해결: Adaptive method selection 필요                        │
├─────────────────────────────────────────────────────────────────┤
│ Problem 2: Cold Anomaly Detection (Oracle로도 77%)              │
│   - 완벽한 gating으로도 Cold-only AUROC가 낮음                  │
│   - Gating 문제가 아닌 Cold 이미지 자체의 anomaly detection 문제│
│   - 해결: Reference 수 증가, 또는 Cold-specific 개선 필요       │
└─────────────────────────────────────────────────────────────────┘
```

### Core Challenges

1. **Gating Trade-off:** No single similarity metric works for all condition combinations
   - Cold images → Patch-level methods better
   - Warm images → Global embedding better

2. **Cold Anomaly Detection:** Even with perfect gating, Cold-only AUROC is limited (~77%)
   - Increasing k helps (+5.9% from k=1 to k=2)
   - May need Cold-specific scoring adjustments
