# Condition-Aware PatchCore (CA-PatchCore) 실험 계획

> 작성일: 2025-12-28
> 목표: HDMAP 데이터셋에서 PatchCore + DINO backbone에 Condition-Aware 접근법을 적용하여 성능 향상

---

## 1. 배경 및 동기

### 1.1 문제 정의

HDMAP 데이터셋의 특성:
- **4개 도메인**: domain_A, domain_B, domain_C, domain_D
- **Cold/Warm 상태**: 각 test 데이터의 앞 500개는 Cold, 뒤 500개는 Warm 상태
- **Cold/Warm 간 intensity 차이**: Cold 이미지는 낮은 intensity (~0.19 mean), Warm 이미지는 높은 intensity (~0.28 mean)
- **결함 패턴**: 기어박스 진동 신호 이미지화, 결함은 가로(horizontal) 패턴으로 나타남
- **원본 이미지 크기**: 31x95 (가로가 세로보다 3배 이상 긴 비율)

### 1.2 기존 실험 결과

#### WinCLIP Within-Condition 분석 (CLIP embedding 기반)
| Domain | Fault Mean | Good Mean | Diff | Overlap | AUROC | 평가 |
|--------|------------|-----------|------|---------|-------|------|
| A | 0.1041 | 0.0932 | +0.0109 | 2.8% | **99.2%** | Excellent |
| B | 0.1065 | 0.0915 | +0.0150 | 1.3% | **99.2%** | Excellent |
| **C** | 0.1080 | 0.0977 | +0.0103 | **66.3%** | **81.0%** | **Moderate** |
| D | 0.1152 | 0.1016 | +0.0136 | 24.0% | **97.2%** | Very Good |

**Domain C가 가장 어려운 도메인** - 결함 signal이 미세하고 분포 겹침이 66%로 높음

#### 기존 PatchCore + DINO 성능 (exp-23)
| Domain | Accuracy | 설정 |
|--------|----------|------|
| A | 96.3% | vit_small_patch14_dinov2, blocks.8, coreset_ratio=0.01 |
| B | 96.4% | 동일 |
| **C** | **79.9%** | 동일 |
| D | 91.6% | 동일 |

**참고**: SOTA 기준 Domain C accuracy = 98.2% (목표치)

### 1.3 핵심 가설

#### 가설 1: DINO Feature의 우수성
> DINO는 이미지 자체를 대규모로 학습했기 때문에 CLIP보다 미세한 결함 특성을 더 잘 추출할 수 있다.

- CLIP은 text-image alignment에 최적화되어 natural language로 표현하기 어려운 미세 패턴에 약함
- DINO는 self-supervised learning으로 visual feature 자체에 집중하여 학습됨

#### 가설 2: Condition-Aware Memory Bank의 효과
> Cold/Warm 상태에 따라 분리된 Memory Bank를 사용하면 cross-condition 혼동을 제거할 수 있다.

현재 문제:
```
Cold Fault의 patch ↔ Warm Normal의 patch가 유사한 intensity scale을 가질 수 있음
→ kNN 검색 시 Cold Fault가 Warm Normal patch와 매칭되어 낮은 anomaly score 획득
→ False Negative 증가
```

제안:
```
P90 Intensity Gating으로 test image의 Cold/Warm 상태 판별 (96.7% 정확도)
→ Cold test image는 Cold Memory Bank에서만 검색
→ Cross-condition 혼동 제거
→ Within-condition 분별력만으로 anomaly 판정
```

---

## 2. 실험 계획

### Phase 1: EDA (탐색적 데이터 분석)

#### 실험 1.0: Resize Method 비교 분석 (우선 수행)
**목적**: HDMAP의 특이한 종횡비(31x95)에 최적인 resize 방법 결정

**HDMAP DataModule의 resize_method 옵션**:
| Method | 설명 | 특징 |
|--------|------|------|
| `resize` | Nearest neighbor interpolation | 기본값, aspect ratio 무시하고 target size로 강제 리사이즈 |
| `resize_bilinear` | Bilinear interpolation | 부드러운 보간, aspect ratio 무시 |
| `resize_aspect_padding` | Aspect ratio 유지 + bilinear + padding | 비율 유지하며 최대 확대 후 검은색 패딩 |
| `black_padding` | No resize, black padding | 원본 크기 유지, 검은색 패딩만 추가 |
| `noise_padding` | No resize, noise padding | 원본 크기 유지, 노이즈 패딩 추가 |

**검토 사항**:
1. **Aspect ratio 왜곡의 영향**: 31x95 → 518x518 리사이즈 시 가로 결함이 왜곡되는 정도
2. **DINO feature 품질**: 각 resize 방법별 DINO feature의 discriminative power
3. **WinCLIP 실험과의 일관성**: 기존에 어떤 방법을 사용했는지 확인

**실험 설계**:
```python
resize_methods = [
    "resize",              # Nearest, aspect ratio 무시 (기존 exp-23 설정)
    "resize_bilinear",     # Bilinear, aspect ratio 무시
    "resize_aspect_padding"  # Aspect ratio 유지 + padding
]
```

**평가 기준**:
1. 시각적 비교: 원본 결함 패턴이 resize 후에도 보존되는지
2. DINO feature t-SNE: Fault/Good 분리도
3. PatchCore 성능: 각 resize 방법별 accuracy

**스크립트**: `examples/notebooks/10_patchcore_variant/eda_resize_methods.py`

---

#### 실험 1.1: DINO Feature 분포 분석
**목적**: DINO feature space에서 Cold/Warm, Fault/Good 분포 특성 파악

**방법**:
1. 각 도메인별로 test 이미지의 DINO feature 추출 (vit_small_patch14_dinov2)
2. t-SNE/UMAP으로 2D 시각화
3. Cold vs Warm 군집 분리도 확인
4. Within-condition (Cold-Cold, Warm-Warm)에서 Fault vs Good 분리도 확인

**기대 결과**:
- DINO feature에서도 Cold/Warm 구분이 가능한가?
- Within-condition에서 Fault/Good 분리가 CLIP보다 좋은가?

**스크립트**: `examples/notebooks/10_patchcore_variant/eda_dino_features.py`

---

#### 실험 1.2: Patch-level Feature 분석
**목적**: PatchCore가 사용하는 patch-level feature의 분포 특성 파악

**방법**:
1. DINO backbone에서 patch feature 추출 (blocks.8 layer)
2. Normal 이미지의 patch feature 분포 시각화
3. Cold Normal vs Warm Normal의 patch feature 분포 비교
4. Anomaly patch의 특성 분석 (결함 위치 patch vs 정상 위치 patch)

**기대 결과**:
- Cold/Warm 상태가 patch feature 분포에 미치는 영향 정량화
- Condition-Aware Memory Bank의 필요성 근거 확보

**스크립트**: `examples/notebooks/10_patchcore_variant/eda_patch_features.py`

---

#### 실험 1.3: Memory Bank 구성 분석
**목적**: Memory Bank 내 Cold/Warm patch 분포 확인

**방법**:
1. 기존 방식대로 전체 train 데이터로 Memory Bank 생성
2. Memory Bank 내 patch들의 intensity 분포 분석
3. Coreset sampling 후 Cold/Warm 비율 확인

**기대 결과**:
- 현재 Memory Bank가 Cold/Warm patch를 어떤 비율로 포함하는지 파악
- Condition-Aware 분리의 근거 확보

**스크립트**: `examples/notebooks/10_patchcore_variant/eda_memory_bank.py`

---

### Phase 2: Baseline 재현 및 검증

#### 실험 2.1: 기존 PatchCore 성능 재현
**목적**: exp-23 결과 재현 및 상세 분석

**설정**:
```python
{
    "backbone": "vit_small_patch14_dinov2",
    "layers": ["blocks.8"],
    "target_size": [518, 518],
    "resize_method": "resize",  # 또는 Phase 1에서 결정된 최적 방법
    "coreset_sampling_ratio": 0.01,
    "num_neighbors": 9
}
```

**평가 지표**:
- Overall Accuracy
- Cold-only Accuracy (index 0-499)
- Warm-only Accuracy (index 500-999)
- Cross-condition 혼동 분석 (Cold Fault가 Warm Normal로 분류되는 비율)

**스크립트**: `examples/notebooks/10_patchcore_variant/baseline_patchcore.py`

---

#### 실험 2.2: 다양한 Backbone/Layer 탐색 (Optional)
**목적**: DINO backbone의 최적 설정 탐색

**변수**:
- Backbone: `vit_small_patch14_dinov2`, `vit_base_patch14_dinov2`, `vit_large_patch14_dinov2`
- Layer: `blocks.4`, `blocks.8`, `blocks.11`, multi-layer (`blocks.8`, `blocks.11`)

**우선순위**: Domain C 성능 개선에 집중. Phase 3 이후 필요 시 수행.

**스크립트**: `examples/notebooks/10_patchcore_variant/explore_backbones.py`

---

### Phase 3: Condition-Aware PatchCore 구현

#### 3.1 아키텍처 설계

```
기존 PatchCore:
  Train Images → Feature Extraction → Memory Bank (Single)
  Test Image → Feature Extraction → kNN Search (전체 Memory Bank) → Anomaly Score

CA-PatchCore:
  Cold Normal Images → Feature Extraction → Cold Memory Bank
  Warm Normal Images → Feature Extraction → Warm Memory Bank

  Test Image → P90 Gating (Cold/Warm 판정)
            → 선택된 Bank에서만 kNN Search
            → Anomaly Score
```

#### 3.2 핵심 컴포넌트

**3.2.1 P90 Intensity Gating (기존 CA-WinCLIP에서 이전)**
```python
class P90IntensityGating:
    DOMAIN_THRESHOLDS = {
        'domain_A': 0.2985,  # 100.0% accuracy
        'domain_B': 0.3128,  # 100.0% accuracy
        'domain_C': 0.3089,  # 94.3% accuracy
        'domain_D': 0.2919,  # 92.5% accuracy
    }
```
- 이미지의 90th percentile intensity로 Cold/Warm 판별
- 평균 96.7% 정확도로 gating 가능

**3.2.2 Condition-Aware Memory Bank**
```python
class ConditionAwareMemoryBank:
    def __init__(self):
        self.cold_bank: torch.Tensor  # Cold normal patches
        self.warm_bank: torch.Tensor  # Warm normal patches

    def search(self, query_patches, condition: str):
        """선택된 condition의 bank에서만 kNN 검색"""
        bank = self.cold_bank if condition == "cold" else self.warm_bank
        return self._knn_search(query_patches, bank)
```

**3.2.3 CA-PatchCore Model**
```python
class CAPatchcoreModel(PatchcoreModel):
    """Condition-Aware PatchCore"""

    def __init__(self, ...):
        super().__init__(...)
        self.gating = P90IntensityGating(domain=domain)
        self.cold_memory_bank: torch.Tensor
        self.warm_memory_bank: torch.Tensor

    def forward(self, input_tensor, raw_image=None):
        # 1. Feature extraction
        embedding = self.extract_embedding(input_tensor)

        # 2. Gating (Cold/Warm 판정) - P90 사용
        if raw_image is not None:
            condition, _ = self.gating.select_bank(raw_image)
        else:
            condition = "mixed"  # fallback

        # 3. Condition-specific kNN search
        if condition == "cold":
            patch_scores, locations = self.nearest_neighbors(
                embedding, self.cold_memory_bank
            )
        elif condition == "warm":
            patch_scores, locations = self.nearest_neighbors(
                embedding, self.warm_memory_bank
            )
        else:
            # Mixed: 기존 방식
            patch_scores, locations = self.nearest_neighbors(
                embedding, self.memory_bank
            )

        return self.compute_anomaly_score(patch_scores, locations, embedding)
```

**파일 구조**:
```
src/anomalib/models/image/patchcore_variants/
├── __init__.py
├── ca_patchcore/
│   ├── __init__.py
│   ├── gating.py              # P90IntensityGating (기존 코드 재사용)
│   ├── torch_model.py         # CAPatchcoreModel
│   └── lightning_model.py     # CAPatchcore Lightning Module
```

---

### Phase 4: 실험 및 평가

#### 실험 4.1: CA-PatchCore vs Baseline 비교
**목적**: Condition-Aware 접근법의 효과 검증

**실험 설계**:
| 실험 | Cold Ref | Warm Ref | Gating | 설명 |
|------|----------|----------|--------|------|
| Baseline (Mixed) | All | All | None | 기존 PatchCore |
| Oracle CA | Cold만 | Warm만 | GT | Upper bound |
| **P90 CA** | Cold만 | Warm만 | **P90** | **제안 방법** |
| Random CA | Cold만 | Warm만 | Random | 50% 정확도 baseline |
| Inverse CA | Cold만 | Warm만 | Inverse | Worst case |

**평가 지표**:
- Overall Accuracy
- Cold-only Accuracy
- Warm-only Accuracy
- **Cross-condition 혼동률** (핵심 지표)
- AUROC, F1-Score

**스크립트**: `examples/notebooks/10_patchcore_variant/evaluate_ca_patchcore.py`

---

#### 실험 4.2: Domain C 집중 분석
**목적**: Domain C 성능 향상 검증 (79.9% → 목표 98.2%)

**추가 분석**:
1. Cold Fault 중 False Negative 케이스 분석
2. Warm Normal 중 False Positive 케이스 분석
3. P90 Gating 오류가 성능에 미치는 영향 분석

**스크립트**: `examples/notebooks/10_patchcore_variant/analyze_domain_c.py`

---

#### 실험 4.3: Ablation Study
**목적**: 각 컴포넌트의 기여도 분석

**변수**:
1. Gating 정확도 영향: Oracle vs P90 vs Random
2. Memory Bank 분리 방식: Cold/Warm vs Single
3. Coreset sampling ratio: 0.01, 0.1, 0.5
4. Resize method: resize vs resize_bilinear vs resize_aspect_padding

**스크립트**: `examples/notebooks/10_patchcore_variant/ablation_study.py`

---

## 3. 구현 순서 (Bottom-Up)

### 폴더 구조
```
examples/notebooks/10_patchcore_variant/
├── 000_plan.md                              # 전체 계획
├── 001_eda_resize_methods/                  # Phase 1.0 - Resize 비교
│   ├── eda_resize_methods.py
│   └── results/domain_C/
├── 001_eda_resize_methods_results.md        # 결과 정리
├── 002_eda_dino_features/                   # Phase 1.1 - DINO 분석
├── 002_eda_dino_features_results.md
├── 003_baseline_patchcore/                  # Phase 2 - Baseline
├── 003_baseline_patchcore_results.md
├── 004_ca_patchcore/                        # Phase 3/4 - CA-PatchCore
└── 004_ca_patchcore_results.md
```

### Step 1: EDA 스크립트 작성 및 실행
1. **`001_eda_resize_methods/`** - Resize method 비교 ✅ 완료
2. `002_eda_dino_features/` - DINO feature 분포 분석
3. `003_eda_patch_features/` - Patch-level 분석 (필요시)

### Step 2: Baseline 재현
1. `003_baseline_patchcore/` - 기존 성능 재현 (resize_bilinear 적용)
2. Cold/Warm별 상세 성능 분석 추가

### Step 3: CA-PatchCore 구현
1. `src/anomalib/models/image/patchcore_variants/ca_patchcore/` 디렉토리 구조 생성
2. `gating.py` - P90IntensityGating (기존 코드 재사용)
3. `torch_model.py` - CAPatchcoreModel
4. `lightning_model.py` - CAPatchcore

### Step 4: 실험 및 평가
1. `004_ca_patchcore/` - CA-PatchCore 종합 평가
2. Domain C 집중 분석
3. Ablation study

---

## 4. 성공 기준

| 지표 | 기존 Baseline | 목표 (CA-PatchCore) | SOTA 참고 |
|------|---------------|---------------------|-----------|
| Domain A | 96.3% | 97%+ | - |
| Domain B | 96.4% | 97%+ | - |
| **Domain C** | **79.9%** | **90%+** | **98.2%** |
| Domain D | 91.6% | 95%+ | - |
| Cross-condition 혼동률 | 측정 필요 | 10% 이하 | - |

**핵심 목표**: Domain C 성능을 79.9% → 90% 이상으로 개선

---

## 5. 예상 이슈 및 대응

### 이슈 1: Train 데이터에 Cold/Warm 라벨 없음
**대응**:
- Test/good 데이터에서 Cold/Warm 샘플을 reference로 사용 (WinCLIP과 동일 전략)
- 또는 P90 기반으로 Train 데이터를 자동 분류

### 이슈 2: Memory Bank 크기 증가
**대응**:
- Cold/Warm 각각 별도 coreset sampling 적용
- Total memory 사용량은 기존과 유사하게 유지

### 이슈 3: Resize method에 따른 결함 패턴 왜곡
**대응**:
- Phase 1에서 resize method별 성능 비교 수행
- Aspect ratio 유지 방식이 결함 패턴 보존에 유리할 수 있음
- 단, padding 영역이 DINO feature에 미치는 영향도 분석 필요

### 이슈 4: P90 Gating 오류의 영향 (Domain C: 94.3%)
**대응**:
- P90 gating 정확도가 94.3%인 Domain C에서 영향 분석
- 오분류 시 fallback 전략 검토 (양쪽 bank 모두 검색 후 min 선택)

---

## 6. 관련 파일

### 기존 코드 참조
- PatchCore 모델: `src/anomalib/models/image/patchcore/`
- HDMAP 데이터셋: `src/anomalib/data/datasets/image/hdmap.py`
- CA-WinCLIP Gating: `examples/notebooks/09_winclip_variant/ca_winclip/gating.py`
- 기존 실험 설정: `examples/hdmap/single_domain/exp_23_patchcore.json`

### WinCLIP 분석 결과 참조
- 최종 분석: `examples/notebooks/09_winclip_variant/WINCLIP_HDMAP_FINAL_ANALYSIS.md`
- CA-WinCLIP 설계: `examples/notebooks/09_winclip_variant/CA_WinCLIP_README.md`
- Within-condition 분석: `examples/hdmap/EDA/HDMAP_vis/domain_*/within_condition_analysis.png`
- Resize method 분석: `examples/notebooks/09_winclip_variant/analyze_resize_methods.py`

---

## 7. 실행 명령어 예시

```bash
# 001: EDA - Resize Method 비교 ✅ 완료
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
    examples/notebooks/10_patchcore_variant/001_eda_resize_methods/eda_resize_methods.py \
    --domain domain_C --max-samples 400

# 002: EDA - DINO Features (다음 단계)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
    examples/notebooks/10_patchcore_variant/002_eda_dino_features/eda_dino_features.py \
    --domain domain_C

# 003: Baseline
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
    examples/notebooks/10_patchcore_variant/003_baseline_patchcore/baseline_patchcore.py \
    --domain domain_C

# 004: CA-PatchCore 평가
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
    examples/notebooks/10_patchcore_variant/004_ca_patchcore/evaluate_ca_patchcore.py \
    --domain domain_C --gating p90
```

---

## 8. 진행 상황

### ✅ 완료
- **001_eda_resize_methods**: Resize method 비교 분석
  - 결과: `resize_bilinear`가 Cold AUROC 0.845로 가장 우수
  - 상세: `001_eda_resize_methods_results.md` 참조

- **002_eda_dino_features**: Patch-level DINO feature 분석
  - **핵심 발견**: Cold/Warm Normal 패치의 Separation Ratio가 낮음 (0.19~0.26)
  - ⚠️ Patch AUROC는 localized fault 특성상 해석 주의 필요 (낮은 값은 예상된 결과)
  - **CA-PatchCore 효과 예측**: Cold/Warm 분리가 약해 효과 제한적일 수 있음
  - 상세: `002_eda_dino_features_results.md` 참조

  | Domain | Separation Ratio | 해석 |
  |--------|------------------|------|
  | A | 0.26 | Cold/Warm 겹침 |
  | B | 0.26 | Cold/Warm 겹침 |
  | **C** | 0.22 | Cold/Warm 겹침 |
  | D | 0.19 | Cold/Warm 겹침 |

  ※ Separation Ratio < 2.0 → CA-PatchCore 효과 제한적

### 📋 다음 단계
- **003_baseline_patchcore**: 실제 PatchCore 성능 측정 (resize_bilinear 적용)
- **004_ca_patchcore**: CA-PatchCore 구현 및 평가
