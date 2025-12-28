# 002. EDA: Patch-level DINO Feature Analysis Results

> 실행일: 2025-12-28
> 대상: 전체 도메인 (A, B, C, D)
> 목적: PatchCore가 실제로 사용하는 Patch-level feature 분석

---

## 1. 실험 개요

### 1.1 분석 목적

001에서 **global feature** (이미지 전체 평균)를 분석했다면,
002에서는 **patch-level feature** (PatchCore가 실제 사용)를 분석합니다.

PatchCore 작동 방식:
1. 이미지를 37x37 = 1369개 패치로 분할
2. 각 패치의 feature 추출
3. Normal 패치들로 Memory Bank 구성
4. Test 시 각 패치와 Memory Bank의 최근접 거리로 anomaly score 계산

### 1.2 실험 설정

- **Backbone**: vit_small_patch14_dinov2
- **Layer**: blocks.8
- **Target Size**: 518x518 → 37x37 patches
- **Resize Method**: `resize_bilinear` (001에서 결정)
- **샘플 수**: 도메인당 30개 이미지 × 4 그룹 = 120개 이미지
- **패치 수**: 30 이미지 × 1369 패치 = 41,070 패치/그룹

### 1.3 분석 항목

1. **Cold vs Warm Normal 패치 분리도**: CA-PatchCore 필요성 검증
2. **Within-condition 거리 분석**: 같은 condition 내에서 Fault vs Good 패치 거리
3. **Cross-condition 혼동**: Cold Fault가 Warm Good과 혼동되는지

### 1.4 ⚠️ 분석 한계 및 주의사항

**HDMAP Fault 특성**:
- Fault 이미지는 **전체가 비정상이 아님**
- **Localized fault**: 이미지의 일부 영역(가로 밝은 선)만 결함
- 1369개 패치 중 **~95%는 normal**, **~5%만 anomalous**

**이로 인한 분석 한계**:
```
Good 이미지:  100% normal patches
Fault 이미지: ~95% normal patches + ~5% anomalous patches

→ "Fault 이미지의 모든 패치" vs "Good 이미지의 모든 패치" 비교는
  실제로는 "normal vs mostly-normal" 비교가 됨
→ AUROC ~0.5-0.6은 예상된 결과 (문제가 아님)
```

**진정한 anomaly detection 성능 평가**:
- Pixel-level GT mask가 필요 (실제 fault 영역 표시)
- HDMAP에는 mask가 없어 정확한 평가 불가
- **실제 PatchCore 성능은 003 baseline에서 측정**

---

## 2. 전체 도메인 결과

### 2.1 Within-Condition 거리 분석

| Domain | Cold Patch AUROC | Warm Patch AUROC | 001 Global Cold AUROC |
|--------|------------------|------------------|----------------------|
| **A** | 0.6585 | 0.6738 | 1.0000 |
| **B** | 0.7673 | 0.7663 | 1.0000 |
| **C** | **0.5813** | 0.6876 | 0.8454 |
| **D** | 0.6662 | 0.7861 | 0.9728 |

**⚠️ 해석 주의**:
- 이 AUROC는 "Fault 이미지의 모든 패치" vs "Good 이미지의 모든 패치"를 비교
- Fault 이미지 대부분이 normal patch이므로 **낮은 AUROC는 예상된 결과**
- **실제 PatchCore 성능과 직접 비교 불가** (PatchCore는 image-level 판정)

### 2.2 Cold vs Warm Normal 분리도 (CA-PatchCore 핵심 지표)

| Domain | Centroid Distance | Within Distance | Separation Ratio |
|--------|-------------------|-----------------|------------------|
| **A** | 0.3895 | 1.51 | 0.26 |
| **B** | 0.3943 | 1.50 | 0.26 |
| **C** | 0.3357 | 1.54 | 0.22 |
| **D** | 0.2978 | 1.54 | 0.19 |

**해석**:
- Separation Ratio = Centroid Distance / Within Distance
- Ratio < 2.0 → Cold/Warm normal 패치가 feature space에서 **많이 겹침**
- **CA-PatchCore의 효과가 제한적일 수 있음** (핵심 발견)

### 2.3 Cross-Condition Confusion 분석

| Domain | Cold Fault → Cold Good | Cold Fault → Warm Good | Closer To |
|--------|------------------------|------------------------|-----------|
| **A** | 1.6449 | 1.6675 | Cold (OK) |
| **B** | 1.7358 | 1.7524 | Cold (OK) |
| **C** | 1.5993 | 1.6041 | Cold (OK) |
| **D** | 1.6931 | 1.7251 | Cold (OK) |

**해석**:
- 모든 도메인에서 Cold Fault가 Cold Good에 더 가까움
- Cross-condition 혼동 심하지 않음
- CA-PatchCore의 주요 동기(cross-condition 혼동 방지)가 약함

---

## 3. Domain C 심층 분석 (핵심 타겟)

### 3.1 t-SNE 시각화

![Domain C Patch t-SNE](002_eda_dino_features/results/domain_C/patch_tsne.png)

**관찰**:
- **Cold vs Warm** (좌): 거의 완전히 섞여 있음
- **Good vs Fault** (중앙): 역시 완전히 섞여 있음
- **4개 그룹** (우): 모든 그룹이 겹쳐 있음

**⚠️ 이것은 문제가 아님**:
- Fault 이미지도 대부분(~95%) normal patches
- 따라서 Good/Fault 패치가 겹치는 것은 **당연한 현상**
- PatchCore는 이 중 **소수의 anomalous patches**만 탐지하면 됨

### 3.2 Cross-Condition 분석

![Domain C Analysis](002_eda_dino_features/results/domain_C/cross_condition_analysis.png)

**왼쪽 그래프 해석 (Cold vs Warm Normal Patch Separation)**:
| 지표 | 값 | 의미 |
|------|-----|------|
| Centroid Distance | 0.34 | Cold/Warm normal 중심점 간 거리 |
| Cold Within Dist | 1.54 | Cold normal 패치들의 평균 분산 |
| Warm Within Dist | 1.54 | Warm normal 패치들의 평균 분산 |
| **Separation Ratio** | **0.22** | 0.34 / 1.54 (매우 낮음) |

→ Cold와 Warm normal 패치가 feature space에서 거의 구분되지 않음
→ CA-PatchCore로 memory bank를 분리해도 효과가 제한적

**오른쪽 그래프 해석 (Within-Condition Anomaly Detection)**:
| 조건 | AUROC | Distance Separation |
|------|-------|---------------------|
| Cold | 0.581 | ~0.05 |
| Warm | 0.688 | ~0.13 |

→ ⚠️ 이 AUROC는 localized fault 특성상 **해석에 주의 필요**
→ 실제 PatchCore 성능은 003에서 측정

### 3.3 Within-Condition 상세

```
COLD: AUROC=0.5813, Separation=0.0544
      Good dist: 1.5425 ± 0.2002
      Fault dist: 1.5969 ± 0.2073
      → Fault 이미지 패치들이 Good보다 약간 더 멀리 분포

WARM: AUROC=0.6876, Separation=0.1324
      Good dist: 1.5339 ± 0.1909
      Fault dist: 1.6664 ± 0.2171
      → Cold보다 분리가 조금 더 좋음
```

---

## 4. 핵심 발견 및 해석

### 4.1 t-SNE 겹침에 대한 올바른 해석

| 현상 | 잘못된 해석 | 올바른 해석 |
|------|------------|------------|
| Good/Fault 패치 겹침 | 탐지 불가능 | **정상** (Fault 이미지도 ~95% normal patch) |
| Cold/Warm 패치 겹침 | 문제 없음 | **CA-PatchCore 효과 제한 가능성** |
| AUROC ~0.6 | 성능 나쁨 | **예상된 결과** (mostly-normal vs normal 비교) |

### 4.2 CA-PatchCore 효과 예측 (유효한 분석)

**Cold/Warm Normal 분리도 분석은 여전히 유효**:
- Separation Ratio 0.19~0.26 (모든 도메인)
- 기준 2.0 대비 매우 낮음
- **Cold memory bank와 Warm memory bank가 비슷한 영역 커버**

**결론**: CA-PatchCore의 핵심 가정(Cold/Warm normal이 다름)이 약함
→ Memory bank 분리 효과가 **제한적일 수 있음**

### 4.3 Cross-condition 혼동 분석 (유효한 분석)

모든 도메인에서:
- Cold Fault → Cold Good 거리 < Cold Fault → Warm Good 거리
- **Cross-condition 혼동이 심하지 않음**
- CA-PatchCore의 주요 동기 중 하나(혼동 방지)가 약함

---

## 5. 결론 및 권장사항

### 5.1 002 분석의 의의

| 분석 항목 | 유효성 | 결론 |
|----------|--------|------|
| Patch-level AUROC | ⚠️ 해석 주의 | Localized fault 특성상 낮은 값은 예상됨 |
| t-SNE 시각화 | ⚠️ 해석 주의 | 겹침은 정상, 문제 아님 |
| **Cold/Warm Separation Ratio** | ✅ 유효 | 0.22 → CA 효과 제한적 |
| **Cross-condition Confusion** | ✅ 유효 | 혼동 심하지 않음 |

### 5.2 CA-PatchCore 효과 예측

| 항목 | 예측 | 근거 |
|------|------|------|
| Domain A | 효과 제한적 | Separation Ratio 0.26 |
| Domain B | 효과 제한적 | Separation Ratio 0.26 |
| **Domain C** | **효과 제한적** | Separation Ratio 0.22 |
| Domain D | 효과 제한적 | Separation Ratio 0.19 |

**그러나**: 실제 효과는 003/004 실험에서 검증 필요

### 5.3 다음 단계

1. **003_baseline_patchcore**: 실제 PatchCore로 baseline 성능 측정
   - Image-level accuracy (Cold/Warm 별도)
   - 이것이 진짜 성능 지표

2. **004_ca_patchcore**: CA-PatchCore 구현 및 비교
   - 002 분석상 효과가 제한적일 것으로 예상되나
   - 실제로 검증하여 확인

---

## 6. 실행 방법

```bash
# 단일 도메인 분석
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
    examples/notebooks/10_patchcore_variant/002_eda_dino_features/eda_patch_features.py \
    --domain domain_C --n-samples 30

# 전체 도메인 분석
for domain in domain_A domain_B domain_C domain_D; do
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        examples/notebooks/10_patchcore_variant/002_eda_dino_features/eda_patch_features.py \
        --domain $domain --n-samples 30
done
```

---

## 7. 생성된 파일

```
002_eda_dino_features/
├── eda_patch_features.py
└── results/
    ├── domain_A/
    │   ├── patch_tsne.png
    │   └── cross_condition_analysis.png
    ├── domain_B/
    │   ├── patch_tsne.png
    │   └── cross_condition_analysis.png
    ├── domain_C/
    │   ├── patch_tsne.png
    │   └── cross_condition_analysis.png
    └── domain_D/
        ├── patch_tsne.png
        └── cross_condition_analysis.png
```
