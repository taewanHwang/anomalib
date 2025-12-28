# Condition-Aware WinCLIP (CA-WinCLIP)

HDMAP 데이터셋에서 Cold/Warm condition에 따른 anomaly score 혼동 문제를 해결하기 위한 WinCLIP 확장.

---

## 1. 가설 (Hypothesis)

### 문제 정의
HDMAP 데이터셋에서 WinCLIP Zero-shot 성능이 제한적인 이유:
- **Cold Start** 이미지: 낮은 intensity (~0.19 mean)
- **Warmed Up** 이미지: 높은 intensity (~0.28 mean)
- 이로 인해 **Cold Fault**와 **Warm Normal**의 anomaly score가 겹침

### 핵심 가설
> Cold 이미지는 Cold reference와, Warm 이미지는 Warm reference와 비교하면
> condition 간 intensity 차이로 인한 혼동을 제거할 수 있다.

### Baseline 문제점 (Zero-shot)
| Metric | Value | 문제 |
|--------|-------|------|
| Overall AUROC | 73.05% | 낮음 |
| Cold Fault vs Warm Normal | 64.25% | **핵심 문제** - 거의 random |

---

## 2. 해결 방안 (Solution)

### CA-WinCLIP 아키텍처
```
기존 WinCLIP:
  Test Image → 모든 Reference와 비교 → Anomaly Score

CA-WinCLIP:
  Test Image → Gating (Cold/Warm bank 선택) → 선택된 Bank에서만 비교 → Score
```

### Gating 방법

#### 방법 1: P90 Intensity Gating (권장, 96.7% 정확도)
이미지의 90th percentile intensity로 Cold/Warm 구분:
- Threshold = (max(good/cold p90) + min(good/warm p90)) / 2
- p90 <= threshold → Cold
- p90 > threshold → Warm

| Domain | Threshold | Gating Accuracy |
|--------|-----------|-----------------|
| A | 0.2985 | 100.0% |
| B | 0.3128 | 100.0% |
| C | 0.3089 | 94.3% |
| D | 0.2919 | 92.5% |

#### 방법 2: Top-K Global Similarity (CLIP 기반, 88.8% 정확도)
1. 각 bank의 reference들로부터 global embedding 추출
2. Test image의 global embedding과 각 bank의 유사도 계산
3. Top-K 유사도 평균이 높은 bank 선택

### Reference 소스
- **Train 데이터에는 Cold/Warm 라벨이 없음**
- **Test/good 데이터 사용** (file index로 condition 구분)
  - Cold: file index 0-499 (0이 가장 cold)
  - Warm: file index 500-999 (999가 가장 warm)

---

## 3. 구현 내용 (Implementation)

### 파일 구조
```
examples/notebooks/09_winclip_variant/
├── ca_winclip/
│   ├── __init__.py
│   ├── gating.py                    # P90IntensityGating, MultiConditionGating, OracleGating
│   └── condition_aware_model.py     # ConditionAwareWinCLIP wrapper
├── winclip_hdmap_ca_validation.py   # 실험 스크립트
└── CA_WinCLIP_README.md             # 이 문서
```

### 핵심 클래스

#### P90IntensityGating (권장)
```python
class P90IntensityGating:
    """Intensity-based gating using p90 percentile (96.7% accuracy)."""

    DOMAIN_THRESHOLDS = {
        'domain_A': 0.2985,
        'domain_B': 0.3128,
        'domain_C': 0.3089,
        'domain_D': 0.2919,
    }

    def __init__(self, threshold=None, domain=None):
        # threshold 직접 지정 또는 domain으로 자동 설정

    def select_bank(self, image: Tensor) -> Tuple[str, Dict]:
        # p90 계산 후 cold/warm 반환
```

#### MultiConditionGating (CLIP 기반)
```python
class MultiConditionGating:
    """CLIP embedding-based Top-K similarity gating (88.8% accuracy)."""

    def __init__(self, bank_embeddings: Dict[str, Tensor], k: int = 1):
        # bank_embeddings: {"cold": (N, D), "warm": (N, D)}

    def select_bank(self, test_embedding: Tensor) -> Tuple[str, Dict[str, float]]:
        # Returns: (selected_bank_name, {bank: similarity_score})
```

#### ConditionAwareWinCLIP
```python
class ConditionAwareWinCLIP:
    """N-Bank CA-WinCLIP wrapper."""

    def __init__(self, base_model, reference_banks: Dict[str, Tensor],
                 gating=None,  # P90IntensityGating or MultiConditionGating
                 use_oracle: bool = False):

    def forward(self, batch, indices=None, images=None):
        # images: P90 gating용 raw 이미지 리스트
        # Returns: (pixel_scores, selected_banks, gating_details)
```

### 사용 예시
```python
from ca_winclip import ConditionAwareWinCLIP, P90IntensityGating

# P90 gating 초기화 (권장)
gating = P90IntensityGating(domain="domain_C")

# CA-WinCLIP 생성
ca_model = ConditionAwareWinCLIP(
    base_model,
    reference_banks={"cold": cold_refs, "warm": warm_refs},
    gating=gating
)

# 추론
scores, banks, details = ca_model.forward(batch, images=raw_images)
```

---

## 4. 실험 계획 및 명령어 (Experiments)

### 실험 설계
| 실험 | Cold Ref | Warm Ref | 총 Ref | 비교 Ref | Gating | 설명 |
|------|----------|----------|--------|----------|--------|------|
| Baseline (Zero-shot) | 0 | 0 | 0 | 0 | - | 기존 WinCLIP |
| **Baseline (Mixed)** | 1 | 1 | 2 | 2 | None | 2장 모두 사용 (핵심 baseline) |
| **Baseline (Random)** | 1 | 1 | 2 | 1 | Random | 랜덤 bank 선택 (~50%) |
| **Baseline (Inverse)** | 1 | 1 | 2 | 1 | Inverse | 항상 반대 bank (Worst case) |
| Oracle CA (k=1) | 1 | 1 | 2 | 1 | GT | Upper bound |
| P90 CA (k=1) | 1 | 1 | 2 | 1 | P90 | Intensity gating (96.7%) |
| Oracle CA (k=2) | 2 | 2 | 4 | 2 | GT | Upper bound |
| P90 CA (k=2) | 2 | 2 | 4 | 2 | P90 | Intensity gating (96.7%) |
| Oracle CA (k=4) | 4 | 4 | 8 | 4 | GT | Upper bound |
| P90 CA (k=4) | 4 | 4 | 8 | 4 | P90 | Intensity gating (96.7%) |

### Baseline 설명
- **Mixed**: cold 1장 + warm 1장을 구분 없이 모두 reference로 사용 (표준 few-shot)
  - CA-WinCLIP과 **동일한 2장**을 사용하지만 **gating 없이** 모두 비교
  - **핵심 비교 대상**: Oracle/P90 vs Mixed 차이 = gating의 효과
- **Random**: 각 test 이미지에 대해 cold/warm bank 중 랜덤 선택 → gating 50% 확률
- **Inverse**: 항상 반대 bank 선택 (cold→warm, warm→cold) → 최악의 경우

### 평가 지표
1. **Overall AUROC**: 전체 샘플
2. **Cold-only AUROC**: Cold 샘플만 (fault vs good)
3. **Warm-only AUROC**: Warm 샘플만 (fault vs good)
4. **Cross-condition AUROC**: Cold Fault vs Warm Good (**핵심 지표**)
5. **Gating Accuracy**: Oracle 대비 정확도 (TopK만)

---

## 5. 실험 명령어

### 기본 설정
```bash
cd /mnt/ex-disk/taewan.hwang/study/anomalib
```

### Baseline 실험 (Mixed, Random, Inverse)

```bash
# ============================================
# Baseline Experiments (Domain C)
# ============================================

# Mixed: 2장 모두 사용 (핵심 baseline - gating 없음)
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_C --k-per-bank 1 --gating mixed --gpu 0 \
    > logs/ca_domain_C_mixed_k1.log 2>&1 &
sleep 3
# Random bank selection (expected: ~50% gating accuracy)
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_C --k-per-bank 1 --gating random --gpu 1 \
    > logs/ca_domain_C_random_k1.log 2>&1 &
sleep 3

# Inverse bank selection (always wrong - worst case)
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_C --k-per-bank 1 --gating inverse --gpu 2 \
    > logs/ca_domain_C_inverse_k1.log 2>&1 &
```

### Domain C 실험 (k=1, 2, 4)

```bash
# ============================================
# Domain C - Oracle (Upper Bound)
# ============================================

# Oracle k=1
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_C --k-per-bank 1 --gating oracle --gpu 3 \
    > logs/ca_domain_C_oracle_k1.log 2>&1 &
sleep 3
# Oracle k=2
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_C --k-per-bank 2 --gating oracle --gpu 4 \
    > logs/ca_domain_C_oracle_k2.log 2>&1 &
sleep 3
# Oracle k=4
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_C --k-per-bank 4 --gating oracle --gpu 5 \
    > logs/ca_domain_C_oracle_k4.log 2>&1 &

# ============================================
# Domain C - P90 Intensity Gating (94.3% accuracy)
# ============================================

# P90 k=1
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_C --k-per-bank 1 --gating p90 --gpu 6 \
    > logs/ca_domain_C_p90_k1.log 2>&1 &
sleep 3

# P90 k=2
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_C --k-per-bank 2 --gating p90 --gpu 7 \
    > logs/ca_domain_C_p90_k2.log 2>&1 &
sleep 3

# P90 k=4
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_C --k-per-bank 4 --gating p90 --gpu 8 \
    > logs/ca_domain_C_p90_k4.log 2>&1 &
```

### Domain A 실험 (k=1, 2, 4)

```bash
# ============================================
# Domain A - Oracle (Upper Bound)
# ============================================

# Oracle k=1
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_A --k-per-bank 1 --gating oracle --gpu 9 \
    > logs/ca_domain_A_oracle_k1.log 2>&1 &
sleep 3

# Oracle k=2
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_A --k-per-bank 2 --gating oracle --gpu 10 \
    > logs/ca_domain_A_oracle_k2.log 2>&1 &
sleep 3

# Oracle k=4
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_A --k-per-bank 4 --gating oracle --gpu 11 \
    > logs/ca_domain_A_oracle_k4.log 2>&1 &

# ============================================
# Domain A - P90 Intensity Gating (100% accuracy)
# ============================================

# P90 k=1
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_A --k-per-bank 1 --gating p90 --gpu 12 \
    > logs/ca_domain_A_p90_k1.log 2>&1 &
sleep 3

# P90 k=2
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_A --k-per-bank 2 --gating p90 --gpu 13 \
    > logs/ca_domain_A_p90_k2.log 2>&1 &
sleep 3

# P90 k=4
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_A --k-per-bank 4 --gating p90 --gpu 14 \
    > logs/ca_domain_A_p90_k4.log 2>&1 &
```

### Domain B 실험 (k=1, 2, 4)

```bash
# ============================================
# Domain B - Oracle (Upper Bound)
# ============================================

# Oracle k=1
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_B --k-per-bank 1 --gating oracle --gpu 0 \
    > logs/ca_domain_B_oracle_k1.log 2>&1 &
sleep 3

# Oracle k=2
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_B --k-per-bank 2 --gating oracle --gpu 1 \
    > logs/ca_domain_B_oracle_k2.log 2>&1 &
sleep 3

# Oracle k=4
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_B --k-per-bank 4 --gating oracle --gpu 2 \
    > logs/ca_domain_B_oracle_k4.log 2>&1 &

# ============================================
# Domain B - P90 Intensity Gating (100% accuracy)
# ============================================

# P90 k=1
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_B --k-per-bank 1 --gating p90 --gpu 3 \
    > logs/ca_domain_B_p90_k1.log 2>&1 &
sleep 3

# P90 k=2
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_B --k-per-bank 2 --gating p90 --gpu 4 \
    > logs/ca_domain_B_p90_k2.log 2>&1 &
sleep 3

# P90 k=4
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_B --k-per-bank 4 --gating p90 --gpu 5 \
    > logs/ca_domain_B_p90_k4.log 2>&1 &
```

### Domain D 실험 (k=1, 2, 4)

```bash
# ============================================
# Domain D - Oracle (Upper Bound)
# ============================================

# Oracle k=1
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_D --k-per-bank 1 --gating oracle --gpu 6 \
    > logs/ca_domain_D_oracle_k1.log 2>&1 &
sleep 3

# Oracle k=2
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_D --k-per-bank 2 --gating oracle --gpu 7 \
    > logs/ca_domain_D_oracle_k2.log 2>&1 &
sleep 3

# Oracle k=4
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_D --k-per-bank 4 --gating oracle --gpu 8 \
    > logs/ca_domain_D_oracle_k4.log 2>&1 &

# ============================================
# Domain D - P90 Intensity Gating (92.5% accuracy)
# ============================================

# P90 k=1
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_D --k-per-bank 1 --gating p90 --gpu 9 \
    > logs/ca_domain_D_p90_k1.log 2>&1 &
sleep 3

# P90 k=2
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_D --k-per-bank 2 --gating p90 --gpu 10 \
    > logs/ca_domain_D_p90_k2.log 2>&1 &
sleep 3

# P90 k=4
nohup .venv/bin/python examples/notebooks/09_winclip_variant/winclip_hdmap_ca_validation.py \
    --domain domain_D --k-per-bank 4 --gating p90 --gpu 11 \
    > logs/ca_domain_D_p90_k4.log 2>&1 &
```

### 진행 확인 명령어
```bash
# 실시간 로그 확인
tail -f logs/ca_domain_C_oracle_k1.log

# 프로세스 확인
ps aux | grep winclip_hdmap_ca_validation | grep -v grep

# GPU 사용량 확인
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# 모든 로그 마지막 줄 확인
for log in logs/ca_domain_*.log; do echo "=== $log ===" && tail -3 "$log"; done
```

---

## 6. 예상 결과 템플릿

### Domain별 결과 요약

#### Baseline (Domain C)
| Method | k | 총 Ref | Overall | Cold Only | Warm Only | Cross | Gating Acc |
|--------|---|--------|---------|-----------|-----------|-------|------------|
| Zero-shot | 0 | 0 | 73.05% | 72.50% | 73.70% | 64.25% | - |
| **Mixed** | 1 | 2 | ___% | ___% | ___% | ___% | N/A |
| **Random** | 1 | 2 | ___% | ___% | ___% | ___% | ~50% |
| **Inverse** | 1 | 2 | ___% | ___% | ___% | ___% | 0% |

#### Domain A
| Method | k | Overall | Cold Only | Warm Only | Cross | Gating Acc |
|--------|---|---------|-----------|-----------|-------|------------|
| Oracle | 1 | ___% | ___% | ___% | ___% | 100% |
| P90 | 1 | ___% | ___% | ___% | ___% | 100.0% |
| Oracle | 2 | ___% | ___% | ___% | ___% | 100% |
| P90 | 2 | ___% | ___% | ___% | ___% | 100.0% |
| Oracle | 4 | ___% | ___% | ___% | ___% | 100% |
| P90 | 4 | ___% | ___% | ___% | ___% | 100.0% |

#### Domain B
| Method | k | Overall | Cold Only | Warm Only | Cross | Gating Acc |
|--------|---|---------|-----------|-----------|-------|------------|
| Oracle | 1 | ___% | ___% | ___% | ___% | 100% |
| P90 | 1 | ___% | ___% | ___% | ___% | 100.0% |
| Oracle | 2 | ___% | ___% | ___% | ___% | 100% |
| P90 | 2 | ___% | ___% | ___% | ___% | 100.0% |
| Oracle | 4 | ___% | ___% | ___% | ___% | 100% |
| P90 | 4 | ___% | ___% | ___% | ___% | 100.0% |

#### Domain C
| Method | k | Overall | Cold Only | Warm Only | Cross | Gating Acc |
|--------|---|---------|-----------|-----------|-------|------------|
| Oracle | 1 | ___% | ___% | ___% | ___% | 100% |
| P90 | 1 | ___% | ___% | ___% | ___% | 94.3% |
| Oracle | 2 | ___% | ___% | ___% | ___% | 100% |
| P90 | 2 | ___% | ___% | ___% | ___% | 94.3% |
| Oracle | 4 | ___% | ___% | ___% | ___% | 100% |
| P90 | 4 | ___% | ___% | ___% | ___% | 94.3% |

#### Domain D
| Method | k | Overall | Cold Only | Warm Only | Cross | Gating Acc |
|--------|---|---------|-----------|-----------|-------|------------|
| Oracle | 1 | ___% | ___% | ___% | ___% | 100% |
| P90 | 1 | ___% | ___% | ___% | ___% | 92.5% |
| Oracle | 2 | ___% | ___% | ___% | ___% | 100% |
| P90 | 2 | ___% | ___% | ___% | ___% | 92.5% |
| Oracle | 4 | ___% | ___% | ___% | ___% | 100% |
| P90 | 4 | ___% | ___% | ___% | ___% | 92.5% |

### 성공 기준
- [ ] Gated CA가 Oracle CA에 근접 (Gating Accuracy > 90%)
- [ ] Gated CA가 Baseline보다 유의미하게 우수
- [ ] Cross-condition AUROC 개선 (64% → 75%+)
- [ ] k 증가에 따른 성능 개선 확인

---

## 7. 관련 문서

### Feature Gating 분석
- `examples/hdmap/EDA/HDMAP_vis/feature_gating/FEATURE_GATING_ANALYSIS.md`
- P90, FFT, Intensity 기반 gating 비교 분석

### 이전 시도
- `winclip_hdmap_post_analysis.md` - Per-Image Normalization (실패)
- `condition_aware_winclip.md` - 설계 논의

### EDA 스크립트
- `examples/hdmap/EDA/HDMAP_vis/eda_intensity_features_gating.py`
- `examples/hdmap/EDA/HDMAP_vis/eda_p90_threshold_comparison.py`

---

## 8. 추가 실험: Patch-level 분석 및 전처리 (2024-12-28)

### 8.1 문제 분석: 왜 Cold Fault가 분별되지 않는가?

Patch-level 분석 결과, WinCLIP의 근본적인 한계 발견:

#### Visual Association Score 메커니즘
```
WinCLIP은 각 test patch를 모든 reference patch와 비교하여
max cosine similarity를 anomaly score로 사용

문제: 결함 patch도 reference의 "다른 위치"에서 유사한 patch를 찾아 높은 similarity 획득
```

#### Domain C 분석 (fault/000009.tiff)
| 위치 | 설명 | Max Similarity | Anomaly Score |
|------|------|----------------|---------------|
| 결함 영역 (row 5) | 가로 얇은 결함 | 0.74~0.80 | 0.10~0.13 |
| 정상 영역 | - | 0.75~0.85 | 0.08~0.12 |

**결론**: 결함 패치가 reference의 다른 위치에서 "유사한" 패치를 찾아 높은 similarity 획득 → 분별력 저하

### 8.2 시도한 전처리 기법들

#### 8.2.1 Resize Method 변경 (실패)

원본 이미지: 31x95 → 240x240 resize 시 왜곡 발생

| Method | Fault Anomaly | Good Anomaly | Diff | 결과 |
|--------|--------------|--------------|------|------|
| `resize` (nearest, baseline) | 0.1001 | 0.0906 | +0.0095 | Weak |
| `resize_bilinear` | 0.0752 | 0.0784 | **-0.0032** | FAILED |
| `resize_aspect_padding` | 0.0190 | 0.0186 | +0.0005 | Weak |

**결론**: Resize 방법 변경은 효과 없음. Bilinear는 오히려 결함을 smoothing하여 역효과.

#### 8.2.2 2D FFT 변환 (실패)

주파수 도메인에서 결함 특성이 두드러질 것으로 기대

| Method | Fault Anomaly | Good Anomaly | Diff | 결과 |
|--------|--------------|--------------|------|------|
| FFT + resize | 0.0997 | 0.0982 | +0.0015 | Weak |
| FFT + bilinear | 0.0778 | 0.0787 | -0.0010 | FAILED |
| FFT + aspect_padding | 0.0192 | 0.0201 | -0.0009 | FAILED |

**결론**: FFT magnitude spectrum은 모든 이미지가 유사하게 생김 (DC 성분 지배적). CLIP은 FFT 이미지에 학습되지 않음.

#### 8.2.3 Morphological Operations (미미한 개선)

결함 특성 분석:
- 결함 영역이 주변보다 **밝음** (Fault - Good = +0.0528)
- 따라서 **Dilation**이 적합 (밝은 영역 확장)

| Method | Fault Anomaly | Good Anomaly | Diff | 결과 |
|--------|--------------|--------------|------|------|
| none (baseline) | 0.0752 | 0.0784 | -0.0032 | FAILED |
| **dilation k=3 i=1** | 0.0914 | 0.0889 | **+0.0025** | Weak (Best) |
| dilation k=5 i=1 | 0.0972 | 0.1057 | -0.0085 | FAILED |
| erosion k=3 i=1 | 0.0604 | 0.0495 | +0.0109 | Moderate* |

*Erosion이 숫자상 좋아 보이지만, 실제로 결함을 제거하는 효과 (밝은 결함 축소)

**결론**: Morphological 연산으로도 근본적 한계 극복 불가.

#### 8.2.4 Gaussian Blur + Enhancement (실패)

Unsharp masking으로 edge 강조 시도

| Method | Fault | Good | Diff | 결과 |
|--------|-------|------|------|------|
| gaussian blur=3 enh=1.5 | 0.0741 | 0.0775 | -0.0034 | FAILED |
| gaussian blur=3 enh=2.0 | 0.0758 | 0.0776 | -0.0018 | FAILED |
| gaussian blur=5 enh=2.0 | 0.0783 | 0.0790 | -0.0006 | FAILED |

**결론**: CLIP embedding은 이런 미세한 변화에 민감하지 않음.

#### 8.2.5 CLAHE (실패)

Local contrast enhancement 시도

| Method | Fault | Good | Diff | 결과 |
|--------|-------|------|------|------|
| clahe clip=2.0 tile=4 | 0.0822 | 0.0860 | -0.0038 | FAILED |
| clahe clip=4.0 tile=4 | 0.0876 | 0.0919 | -0.0043 | FAILED |

### 8.3 핵심 발견 사항

1. **WinCLIP의 근본적 한계**: Visual association score는 spatial correspondence를 강제하지 않음
   - 결함 patch가 reference의 "다른 위치"에서 유사한 patch를 찾아 높은 similarity 획득

2. **CLIP 특성**: Natural image + natural language로 학습된 모델
   - HDMAP 같은 signal data 이미지에 대한 semantic 이해 부족

3. **전처리 한계**: 이미지 전처리로는 CLIP embedding space의 특성을 바꿀 수 없음

### 8.4 관련 스크립트
- `analyze_resize_methods.py` - resize 방법 비교
- `analyze_fft_patches.py` - 2D FFT 분석
- `analyze_dilation_patches.py` - morphological 연산 분석
- `analyze_patch_mapping_detail.py` - patch-level 매칭 시각화

### 8.5 시각화 결과
- `patch_analysis/domain_C/resize_methods_*.png`
- `patch_analysis/domain_C/fft_resize_methods_*.png`
- `patch_analysis/domain_C/dilation_effect_*.png`
- `patch_analysis/domain_C/defect_brightness_analysis.png`

---

## 9. 향후 실험 방향

### 9.1 Text Template 수정 (가능성: 중간~낮음)

현재 template (`prompting.py`):
```python
NORMAL_STATES = ["flawless {}", "perfect {}", "unblemished {}", ...]
ANOMALOUS_STATES = ["damaged {}", "{} with flaw", "{} with defect", ...]
```

제안하는 HDMAP 특화 template:
```python
# hdmap_prompting.py (신규 작성)
NORMAL_STATES = [
    "{}",
    "normal {} pattern",
    "stable {} signal",
    "{} with consistent intensity",
]
ANOMALOUS_STATES = [
    "abnormal {}",
    "{} with anomaly",
    "{} with irregular pattern",
    "{} with signal distortion",
]
```

**한계**: CLIP이 이런 technical term을 natural language처럼 이해할지 의문

### 9.2 Multi-scale별 분석 (가능성: 높음)

WinCLIP의 multi-scale 구조:
- **Original scale**: 각 patch 1x1 (225 patches)
- **Small scale (2)**: 2x2 sliding window (196 windows)
- **Mid scale (3)**: 3x3 sliding window (169 windows)
- **Full image**: 전체 CLS token (1)

각 scale별로 Cold/Warm 분별력 분석 필요:
```python
# 분석 코드 예시
def analyze_per_scale(model, batch):
    image_emb, window_embs, patch_embs = model.encode_image(batch)

    # Full image score
    image_score = class_scores(image_emb, text_emb)

    # Small scale (2x2)
    small_score = class_scores(window_embs[0], text_emb)

    # Mid scale (3x3)
    mid_score = class_scores(window_embs[1], text_emb)

    return {'image': image_score, 'small': small_score, 'mid': mid_score}
```

### 9.3 Spatial Correspondence 강제 (가능성: 중간)

현재 visual association score: `max(similarity to ALL reference patches)`

제안: `similarity to SAME position patch only`

```python
# 수정된 visual_association_score
def spatial_visual_association_score(embeddings, reference_embeddings):
    # embeddings: (B, N, D), reference: (K, N, D)
    # 같은 위치 patch끼리만 비교
    scores = []
    for i in range(N):
        patch_sim = cosine_similarity(embeddings[:, i], reference_embeddings[:, i])
        scores.append(patch_sim.max(dim=-1))
    return torch.stack(scores, dim=1)
