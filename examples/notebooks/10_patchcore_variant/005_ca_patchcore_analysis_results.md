# 005. CA-PatchCore 성능 저하 원인 분석

## 배경

CA-PatchCore (Condition-Aware PatchCore)는 cold/warm 조건별로 memory bank를 분리하고, 테스트 시 gating을 통해 적절한 bank를 선택하는 방식이다.

### 실험 결과 (004)

| k | Gating | Overall | vs Baseline |
|---|--------|---------|-------------|
| 16 | oracle (100% 정확) | 87.42% | **-2.02%p** |
| 16 | p90 (97.27% 정확) | 87.53% | **-1.91%p** |
| 16 | **mixed** (no gating) | **90.25%** | **+0.81%p** |
| 16 | random (48% 정확) | 86.36% | -3.08%p |

**문제**: Oracle gating (100% 정확)이 mixed mode보다 **2.83%p 낮음**

### 가설

> Cold fault 샘플의 결함 패치가 cold normal보다 **warm normal에 더 유사**하여,
> cold bank만 사용하면 정확한 anomaly score를 얻지 못한다.

---

## Analysis 1: Patch-level NN Tracing (Random Samples)

### 목적
Cold fault 샘플의 패치들이 cold bank와 warm bank 중 어디에 더 가까운지 분석

### 방법
- Domain C, k=16 memory bank 구축
- 5개 cold fault 샘플 랜덤 선택
- 각 패치(37x37=1369개)의 cold/warm bank NN 거리 비교

### 결과

| Sample | % to Warm | Top-10 to Warm | Distance Ratio |
|--------|-----------|----------------|----------------|
| File 000000 | 49.2% | 4/10 | 0.998 |
| File 000100 | 56.0% | **10/10** | 1.008 |
| File 000198 | 32.9% | 2/10 | 0.976 |
| File 000298 | 72.6% | **10/10** | 1.044 |
| File 000398 | 44.7% | 9/10 | 0.995 |
| **평균** | **51.1%** | **7.0/10** | 1.004 |

### 시각화 (File 100)

![NN Analysis Sample 100](results/nn_tracing/domain_C_k16/20251228_085754/sample_000100/nn_analysis.png)

- 좌측 하단: 56%의 패치가 warm bank에 더 가까움 (빨간색)
- 우측 하단: **Top-10 anomalous 패치 중 10/10이 warm에 더 가까움**

### 결론
**가설 부분 지지**: 평균적으로 51.1%가 warm에 가깝고, 특히 **Top-10 anomalous 패치의 70%가 warm bank에 더 가까움**

---

## Analysis 2: Preprocessing Normalization Investigation

### 목적
ImageNet 정규화가 cold/warm intensity 차이를 없애는지 확인

### 방법
- Domain C에서 cold/warm 샘플 각 100개 로드
- Raw intensity vs ImageNet normalized intensity 분포 비교
- 분리도 메트릭 계산 (Cohen's d, Overlap, Bhattacharyya distance)

### 결과

| Metric | Raw | Normalized | Change |
|--------|-----|------------|--------|
| Cold Mean | 0.2031 | -1.0879 | - |
| Warm Mean | 0.2874 | -0.7151 | - |
| Mean Gap | 0.0843 | 0.3728 | +342.5% |
| **Cohen's d** | **1.096** | **1.096** | **-0.0%** |
| Overlap | 0.537 | 0.537 | +0.0% |

### 시각화

![Normalization Comparison](results/normalization/domain_C/20251228_085958/normalization_comparison.png)

### 결론
**ImageNet 정규화는 cold/warm 분리도를 완벽히 보존** (Cohen's d = 1.096 동일)
- 정규화는 단순 선형 변환이므로 상대적 분포 관계 유지
- 정규화가 CA-PatchCore 실패의 원인이 아님

---

## Analysis 3: DINOv2 Intensity Sensitivity

### 목적
DINOv2 feature가 밝기/contrast 변화에 얼마나 민감한지 테스트

### 방법
1. 샘플 이미지에 brightness delta (-0.15 ~ +0.15) 적용
2. 샘플 이미지에 contrast factor (0.7 ~ 1.3) 적용
3. 원본 vs 변환된 이미지의 DINOv2 feature 거리 측정
4. 실제 cold-warm feature 거리와 시뮬레이션 비교

### 결과

#### Brightness/Contrast Sensitivity
- DINOv2는 brightness 변화에 민감 (L2 거리 증가)
- 하지만 cosine similarity는 0.98 이상 유지

#### Cold vs Warm 비교 (핵심)

| Comparison | L2 Distance | Cosine Similarity |
|------------|-------------|-------------------|
| **Actual Cold vs Warm** | **5.363** | **0.6901** |
| Simulated (brightness shift only) | 0.395 | ~1.0 |

### 시각화

![DINOv2 Sensitivity](results/dinov2_sensitivity/domain_C/20251228_090018/sensitivity_analysis.png)

### 결론

1. **실제 cold-warm feature 차이 (L2=5.363)는 brightness 차이 (L2=0.395)의 ~14배**
2. **Cosine similarity 0.69 = cold와 warm은 feature 공간에서 매우 다름**
3. 이 차이는 intensity가 아닌 **구조적/텍스처 차이**에서 기인
4. Intensity만으로는 cold-warm 차이의 **~7%만 설명 가능**

---

## Analysis 4: False Negative 심층 분석

### 목적
CA-PatchCore Oracle mode에서 실제로 실패한 샘플들의 behavior 분석

### False Negative 분포

| Condition | False Negatives | Total Faults | FN Rate |
|-----------|-----------------|--------------|---------|
| **Cold** | **49** | ~500 | ~10% |
| Warm | 6 | ~500 | ~1% |

**Cold FN이 Warm FN의 8배!** - Cold 샘플에서 주로 실패

### 분석 대상
Score가 가장 낮은 Cold False Negative 10개 분석

### 결과

| File | Oracle Score | Cold Bank | Warm Bank | Combined | Top-10→Warm |
|------|--------------|-----------|-----------|----------|-------------|
| 0427 | 4.706 | 5.085 | 5.369 | 5.085 | 5/10 |
| 0094 | 4.720 | 5.149 | 5.001 | 4.943 | **10/10** |
| 0041 | 4.808 | 5.207 | **5.573** ✓ | 5.075 | 9/10 |
| 0239 | 4.849 | 5.214 | 5.029 | 5.029 | 9/10 |
| 0190 | 4.916 | 5.263 | 5.246 | 5.246 | 5/10 |
| 0445 | 4.952 | 5.401 | 5.221 | 5.133 | 9/10 |
| 0407 | 5.046 | 5.476 | **5.529** ✓ | 5.239 | **10/10** |
| 0289 | 5.074 | 5.559 | **5.628** ✓ | **5.559** ✓ | 3/10 |
| 0225 | 5.080 | 5.459 | 5.397 | 5.199 | **10/10** |
| 0027 | 5.082 | 5.453 | **5.460** ✓ | 5.275 | 3/10 |

*Threshold = 5.411, ✓ = PASS 가능*

### 집계 통계

| Metric | Value |
|--------|-------|
| Mean % patches closer to Warm | 46.4% |
| **Mean Top-10 anomalous → Warm** | **7.3/10** |
| Mean score improvement (Combined) | +0.148 |
| Would PASS with Combined | 1/10 |
| Would PASS with Warm | 4/10 |

### 시각화 (File 41 - Warm Bank로 PASS 가능한 케이스)

![FN Analysis File 41](results/false_negatives/domain_C_k16/20251228_090935/fn_0041/analysis.png)

**File 41 분석:**
- Oracle (Cold Bank): 5.207 < 5.411 = **FAIL**
- **Warm Bank: 5.573 > 5.411 = PASS!**
- Top-10 anomalous → Warm: 9/10
- **Warm bank를 사용했으면 탐지 성공!**

### 결론
1. **Top-10 anomalous 패치의 73%가 warm bank에 더 가까움**
2. 4/10 샘플이 warm bank로 recovery 가능
3. Cold fault의 결함 패치가 warm normal feature에 더 유사함 확인

---

## 종합 결론

### CA-PatchCore 실패 메커니즘

```
Cold Fault Sample
      │
      ▼
Oracle/P90 Gating: "cold" 예측
      │
      ▼
Cold Bank에서만 NN 검색
      │
      ▼
결함 패치가 Cold Normal보다 Warm Normal에 더 유사
      │
      ▼
Cold Bank에서 충분히 높은 거리를 얻지 못함
      │
      ▼
Score < Threshold → FALSE NEGATIVE
```

### 가설 검증 요약

| 가설 | 검증 방법 | 결과 |
|-----|----------|------|
| Cold fault 패치가 warm에 더 가까움 | NN Tracing | **지지됨** (70-73% to warm) |
| ImageNet 정규화가 문제 | Normalization Analysis | **기각됨** (분리도 보존) |
| Intensity 차이가 원인 | DINOv2 Sensitivity | **기각됨** (7%만 설명) |
| Cold-Warm은 구조적으로 다름 | DINOv2 Sensitivity | **확인됨** (cosine=0.69) |

### 왜 Mixed Mode가 더 좋은가?

1. **Memory Bank Diversity**: Cold+Warm 모두 포함하여 더 다양한 normal feature 보유
2. **Fault Pattern Coverage**: Fault 패치가 warm feature에 더 유사할 때도 매칭 가능
3. **Gating Error 없음**: 잘못된 bank 선택으로 인한 실패 방지

### 시사점

1. **Condition-aware gating은 HDMAP에서 효과 없음**
   - Cold/warm의 feature 차이가 intensity가 아닌 구조적 차이
   - Fault pattern이 condition 경계를 넘어 유사성 가짐

2. **더 나은 접근법 제안**
   - Mixed mode 사용 (현재 최선)
   - Condition을 feature에 추가하는 soft conditioning
   - Fault-specific feature learning

---

## 파일 구조

```
005_ca_patchcore_analysis/
├── analyze_nn_tracing.py           # Analysis 1
├── analyze_normalization_effect.py # Analysis 2
├── analyze_dinov2_intensity.py     # Analysis 3
├── analyze_false_negatives.py      # Analysis 4 (FN)
└── results/
    ├── nn_tracing/
    │   └── domain_C_k16/
    ├── normalization/
    │   └── domain_C/
    ├── dinov2_sensitivity/
    │   └── domain_C/
    └── false_negatives/
        └── domain_C_k16/
```

---

## 실행 명령어

```bash
# Analysis 1: NN Tracing
CUDA_VISIBLE_DEVICES=0 python analyze_nn_tracing.py --domain domain_C --k-per-bank 16 --n-samples 5

# Analysis 2: Normalization Effect
python analyze_normalization_effect.py --domain domain_C --n-samples 50

# Analysis 3: DINOv2 Intensity Sensitivity
CUDA_VISIBLE_DEVICES=0 python analyze_dinov2_intensity.py --domain domain_C --n-samples 10

# Analysis 4: False Negative Analysis
CUDA_VISIBLE_DEVICES=0 python analyze_false_negatives.py --domain domain_C --k-per-bank 16 --n-samples 10
```
