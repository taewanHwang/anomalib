# 001. EDA: Resize Method Comparison Results

> 실행일: 2025-12-28
> 대상: 전체 도메인 (A, B, C, D)
> 목적: HDMAP의 특이한 종횡비(31x95 → 518x518)에 최적인 resize 방법 결정

---

## 1. 실험 개요

### 1.1 비교 대상 Resize Methods

| Method | 설명 | Aspect Ratio |
|--------|------|--------------|
| `resize` | Nearest neighbor interpolation | 무시 (강제 518x518) |
| `resize_bilinear` | Bilinear interpolation | 무시 (강제 518x518) |
| `resize_aspect_padding` | Bilinear + black padding | 유지 |

### 1.2 실험 설정

- **Backbone**: vit_small_patch14_dinov2
- **Layer**: blocks.8
- **Target Size**: 518x518
- **샘플 수**: 도메인당 400개 (cold_fault: 100, cold_good: 100, warm_fault: 100, warm_good: 100)
- **평가 방법**: DINO global feature 추출 → Good centroid 기준 거리로 AUROC 계산
  - **참고**: 이는 PatchCore (memory bank + kNN)가 아닌 간단한 EDA 수준의 분석입니다

### 1.3 Cold/Warm 인덱스 정의
- **Cold**: test 파일 인덱스 0~499 (0에 가까울수록 더 cold)
- **Warm**: test 파일 인덱스 500~999 (999에 가까울수록 더 warm)

---

## 2. 전체 도메인 결과

### 2.1 Cold AUROC (핵심 지표)

| Domain | resize | resize_bilinear | resize_aspect_padding | Best Method |
|--------|--------|-----------------|----------------------|-------------|
| **A** | 0.9863 | **1.0000** | 0.9994 | resize_bilinear |
| **B** | 0.9994 | **1.0000** | 1.0000 | resize_bilinear |
| **C** | 0.7569 | **0.8454** | 0.8197 | resize_bilinear |
| **D** | 0.8835 | 0.9728 | **0.9810** | resize_aspect_padding |

### 2.2 Warm AUROC

| Domain | resize | resize_bilinear | resize_aspect_padding |
|--------|--------|-----------------|----------------------|
| **A** | 0.9985 | **1.0000** | 1.0000 |
| **B** | 0.9997 | **1.0000** | 1.0000 |
| **C** | 0.9012 | **0.9987** | 0.9950 |
| **D** | 0.9793 | **1.0000** | 1.0000 |

### 2.3 Overall AUROC

| Domain | resize | resize_bilinear | resize_aspect_padding |
|--------|--------|-----------------|----------------------|
| **A** | 0.7376 | **1.0000** | 0.9453 |
| **B** | 0.9636 | **0.9996** | 0.9971 |
| **C** | 0.7182 | 0.9267 | **0.9293** |
| **D** | 0.7930 | 0.9912 | **0.9943** |

---

## 3. 도메인별 상세 분석

### 3.1 Domain A (가장 쉬움)

| Method | Cold AUROC | Warm AUROC | Overall AUROC |
|--------|------------|------------|---------------|
| resize | 0.9863 | 0.9985 | 0.7376 |
| **resize_bilinear** | **1.0000** | **1.0000** | **1.0000** |
| resize_aspect_padding | 0.9994 | 1.0000 | 0.9453 |

- `resize_bilinear`가 완벽한 분리 달성 (AUROC 1.0)
- Overall AUROC가 `resize`에서 낮은 이유: Cold/Warm 간 feature 혼동

### 3.2 Domain B

| Method | Cold AUROC | Warm AUROC | Overall AUROC |
|--------|------------|------------|---------------|
| resize | 0.9994 | 0.9997 | 0.9636 |
| **resize_bilinear** | **1.0000** | **1.0000** | **0.9996** |
| resize_aspect_padding | 1.0000 | 1.0000 | 0.9971 |

- Domain A와 유사하게 `resize_bilinear`가 최고 성능
- 세 방법 모두 Cold/Warm에서 거의 완벽한 분리

### 3.3 Domain C (가장 어려움 - 핵심 타겟)

| Method | Cold AUROC | Warm AUROC | Overall AUROC |
|--------|------------|------------|---------------|
| resize | 0.7569 | 0.9012 | 0.7182 |
| **resize_bilinear** | **0.8454** | **0.9987** | 0.9267 |
| resize_aspect_padding | 0.8197 | 0.9950 | **0.9293** |

- **Cold AUROC가 전 도메인 중 가장 낮음** (핵심 문제)
- `resize_bilinear`가 Cold에서 가장 좋음 (+8.85% vs resize)
- Warm은 모든 방법에서 양호 (0.9+)

### 3.4 Domain D

| Method | Cold AUROC | Warm AUROC | Overall AUROC |
|--------|------------|------------|---------------|
| resize | 0.8835 | 0.9793 | 0.7930 |
| resize_bilinear | 0.9728 | 1.0000 | 0.9912 |
| **resize_aspect_padding** | **0.9810** | **1.0000** | **0.9943** |

- 유일하게 `resize_aspect_padding`이 Cold에서 최고 성능
- 그러나 차이가 작음 (0.9810 vs 0.9728)

---

## 4. 핵심 발견

### 4.1 Domain C의 특수성

```
Domain C Cold AUROC 비교:
- resize:              0.7569 (baseline)
- resize_bilinear:     0.8454 (+11.7%)  ← 최고
- resize_aspect_padding: 0.8197 (+8.3%)
```

Domain C의 cold fault는 결함 signal이 미세하여 모든 방법에서 어려움.
그러나 `resize_bilinear`가 가장 좋은 분별력을 보임.

### 4.2 resize 방법의 문제점

`resize` (nearest neighbor)의 문제:
1. **Overall AUROC 급락**: Domain A에서 Cold/Warm 개별로는 0.98+인데 Overall은 0.74
2. 이는 Cold/Warm 간 feature가 혼동되어 cross-condition 문제 발생
3. Bilinear 보간이 DINO feature 품질에 더 유리

### 4.3 resize_aspect_padding의 한계

예상과 달리 대부분의 도메인에서 `resize_bilinear`보다 낮은 성능:
- 검은색 패딩 영역(mean=0.06)이 feature에 노이즈로 작용
- DINO가 정사각형 이미지로 학습되어 aspect ratio 왜곡에 robust

---

## 5. 결론 및 권장사항

### 5.1 최종 권장: `resize_bilinear`

| 기준 | resize | resize_bilinear | resize_aspect_padding |
|------|--------|-----------------|----------------------|
| Domain A Cold | 0.986 | **1.000** | 0.999 |
| Domain B Cold | 0.999 | **1.000** | 1.000 |
| Domain C Cold | 0.757 | **0.845** | 0.820 |
| Domain D Cold | 0.884 | 0.973 | **0.981** |
| **평균 Cold AUROC** | 0.907 | **0.955** | 0.950 |

**`resize_bilinear` 선택 이유:**
1. 4개 도메인 중 3개에서 Cold AUROC 최고
2. 핵심 문제인 Domain C에서 가장 좋은 성능 (0.845)
3. Domain D에서도 차이가 작음 (0.973 vs 0.981)

### 5.2 다음 단계 적용

- Phase 2 (Baseline PatchCore)부터 `resize_bilinear` 사용
- 기존 exp-23의 `resize` (nearest) → `resize_bilinear`로 변경

---

## 6. 시각화 결과

### 6.1 t-SNE 비교

| Domain | 시각화 파일 |
|--------|------------|
| A | `results/domain_A/tsne_comparison.png` |
| B | `results/domain_B/tsne_comparison.png` |
| C | `results/domain_C/tsne_comparison.png` |
| D | `results/domain_D/tsne_comparison.png` |

### 6.2 Visual Comparison (Domain C)

![Visual Comparison](001_eda_resize_methods/results/domain_C/visual_comparison.png)

---

## 7. 실행 방법

```bash
# 단일 도메인 분석
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
    examples/notebooks/10_patchcore_variant/001_eda_resize_methods/eda_resize_methods.py \
    --domain domain_C --max-samples 400

# 전체 도메인 분석
for domain in domain_A domain_B domain_C domain_D; do
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        examples/notebooks/10_patchcore_variant/001_eda_resize_methods/eda_resize_methods.py \
        --domain $domain --max-samples 400 --skip-visual
done
```

---

## 8. 생성된 파일

```
001_eda_resize_methods/
├── eda_resize_methods.py
└── results/
    ├── domain_A/
    │   └── tsne_comparison.png
    ├── domain_B/
    │   └── tsne_comparison.png
    ├── domain_C/
    │   ├── tsne_comparison.png
    │   └── visual_comparison.png
    └── domain_D/
        └── tsne_comparison.png
```
