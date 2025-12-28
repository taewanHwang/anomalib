# WinCLIP HDMAP 최종 분석 보고서

> 작성일: 2025-12-28
> CA-WinCLIP 실험 전 EDA 및 가능성 검증 결과

---

## 1. 실험 목적

HDMAP 데이터셋에서 WinCLIP 기반 anomaly detection의 한계를 분석하고, 개선 가능성을 탐색.

### 주요 연구 질문
1. **Text Template 수정**으로 HDMAP 결함 인식률을 높일 수 있는가?
2. **Multi-scale** 중 어떤 scale이 가장 효과적인가?
3. **Within-condition** (Cold-Cold, Warm-Warm)에서 분별력은 어떠한가?
4. **도메인별 난이도** 차이는 어떠한가?

---

## 2. Text Template 실험

### 2.1 가설
HDMAP 결함은 "horizontal bright line" 형태이므로:
- Abnormal 프롬프트: "horizontal line" 포함
- Normal 프롬프트: "vertical line" 포함

→ CLIP이 결함을 horizontal로 인식하면 분별력 향상 기대

### 2.2 실험: Contrast Level별 CLIP 인식

| Contrast | H-V Diff | Status |
|----------|----------|--------|
| Original Good (1.0x) | -0.0022 | Not detected |
| **Real Fault (1.32x)** | **-0.0065** | **Not detected** |
| 1.5x | -0.0006 | Not detected |
| 2.0x | +0.0012 | Weak |
| 3.0x | +0.0032 | Weak |
| 4.0x | +0.0037 | Weak (Max) |
| 5.0x | +0.0031 | Weak |

### 2.3 프롬프트 최적화 (8개 카테고리)

| Category | Prompts 예시 | Real Fault H-V | Artificial 5x H-V |
|----------|-------------|----------------|-------------------|
| bright | "bright horizontal line" | -0.0053 | **+0.0083** |
| texture | "horizontally striped texture" | -0.0080 | +0.0080 |
| signal | "horizontal noise pattern" | **-0.0042** | +0.0055 |
| original | "horizontal band across image" | -0.0065 | +0.0031 |
| technical | "horizontal scanline" | -0.0105 | -0.0039 |

### 2.4 결론

**Text Template 수정은 효과 없음**

- 실제 HDMAP 결함 contrast (1.32x)는 CLIP이 인식하기에 너무 미세
- 인공적으로 5x contrast 강화해도 detection threshold (0.02) 미달
- 근본 원인: CLIP은 natural image의 강한 시각 패턴에 학습됨

---

## 3. Multi-scale 분석

### 3.1 WinCLIP Scale 구조

| Scale | Grid | 윈도우 수 |
|-------|------|----------|
| Full Image (CLS) | 1x1 | 1 |
| Small (2x2 window) | 14x14 | 196 |
| Mid (3x3 window) | 13x13 | 169 |
| Patch (1x1) | 15x15 | 225 |

### 3.2 Within-condition 분별력 (Domain C)

| Scale | Cold Fault | Cold Good | Diff (F-G) |
|-------|------------|-----------|------------|
| Full Image | 0.0060 | 0.0034 | +0.0025 |
| Small (2x2) | 0.0033 | 0.0040 | -0.0007 |
| Mid (3x3) | 0.0079 | 0.0088 | -0.0010 |
| **Patch (1x1)** | 0.1001 | 0.0933 | **+0.0068** |

### 3.3 Cross-condition 분별력 (Cold Fault vs Warm Good)

| Scale | Cold Fault | Warm Good | Diff | Status |
|-------|------------|-----------|------|--------|
| **Full Image** | 0.0060 | 0.0028 | **+0.0032** | Weak (Best) |
| Small (2x2) | 0.0033 | 0.0027 | +0.0006 | Weak |
| Mid (3x3) | 0.0079 | 0.0066 | +0.0013 | Weak |
| Patch (1x1) | 0.1001 | 0.1065 | **-0.0064** | **FAILED** |

### 3.4 결론

- **Within-condition**: Patch level이 가장 효과적 (+0.0068)
- **Cross-condition**: Full Image가 유일하게 양수 (+0.0032), 하지만 약함
- **Patch level Cross-condition 실패**: Cold Fault가 Warm Good보다 낮은 score

---

## 4. Within-Condition 도메인별 분석 (핵심 결과)

### 4.1 실험 설정
- Reference: Cold (index 0)
- Cold Fault: index 0-19 (20개)
- Cold Good: index 1-20 (20개)
- Metric: Patch-level max similarity 기반 anomaly score

### 4.2 도메인별 결과

| Domain | Fault Mean | Good Mean | Diff | Overlap | **AUROC** | 평가 |
|--------|------------|-----------|------|---------|-----------|------|
| **A** | 0.1041 | 0.0932 | +0.0109 | 2.8% | **99.2%** | Excellent |
| **B** | 0.1065 | 0.0915 | +0.0150 | 1.3% | **99.2%** | Excellent |
| **C** | 0.1080 | 0.0977 | +0.0103 | **66.3%** | **81.0%** | **Moderate** |
| **D** | 0.1152 | 0.1016 | +0.0136 | 24.0% | **97.2%** | Very Good |

### 4.3 도메인별 특성

```
Domain A, B: 결함 신호 강함 → WinCLIP으로 충분히 분별 가능 (99%+)
Domain D:   결함 신호 중간 → 높은 분별력 (97%)
Domain C:   결함 신호 약함 → 제한적 분별력 (81%), 66% 분포 겹침
```

### 4.4 Domain C 분석

**가장 어려운 도메인**:
- 결함 contrast가 가장 미세 (1.32x)
- Fault/Good 분포가 크게 겹침 (66%)
- Within-condition에서도 81% AUROC로 한계
- WinCLIP 단독으로는 완벽한 분류 불가

---

## 5. 최종 결론

### 5.1 WinCLIP HDMAP 적용 한계 요약

| 문제 | 원인 | 해결 여부 |
|------|------|-----------|
| Cross-condition 혼동 | Cold/Warm intensity 차이 | **CA-WinCLIP으로 해결 가능** |
| Domain C 낮은 분별력 | 결함 신호 미세 (1.32x contrast) | **해결 어려움** |
| Text Template 한계 | CLIP의 subtle pattern 인식 불가 | **불가능** |
| Multi-scale 한계 | 모든 scale에서 cross-condition 약함 | **개선 여지 적음** |

### 5.2 CA-WinCLIP 기대 효과

| 시나리오 | 현재 (Mixed) | CA-WinCLIP (예상) |
|----------|--------------|-------------------|
| Domain A | ~99% | ~99% |
| Domain B | ~99% | ~99% |
| **Domain C** | ~81% | **~81%** (개선 없음) |
| Domain D | ~97% | ~97% |
| Cross-condition | ~50-60% | **N/A (회피됨)** |

### 5.3 권장 사항

#### 즉시 적용 가능
1. **CA-WinCLIP 구현**: P90 Intensity Gating (96.7% 정확도)으로 cross-condition 문제 해결
2. **도메인별 전략 차별화**: Domain A, B, D는 WinCLIP 효과적, Domain C는 보완 필요

#### 추가 검토 필요
1. **Domain C 대응**: Dinomaly 등 다른 모델과 앙상블, 또는 fine-tuning
2. **결함 강조 전처리**: 결함 contrast 증폭 후 WinCLIP 적용 (효과 미검증)

#### 추가 실험 불필요
- Text Template 수정: 효과 없음 확인
- Multi-scale 개별 최적화: 개선 여지 없음
- 전처리 (FFT, morphological, CLAHE): 모두 실패

---

## 6. 관련 파일

### EDA 스크립트
```
examples/hdmap/EDA/
├── test_contrast_levels.py        # Contrast별 CLIP 인식 테스트
├── test_improved_prompts.py       # 프롬프트 최적화 테스트
├── analyze_multiscale.py          # Multi-scale 분별력 분석
├── analyze_within_condition.py    # Within-condition AUROC
└── visualize_test_images.py       # 테스트 이미지 시각화
```

### 시각화 결과
```
examples/hdmap/EDA/HDMAP_vis/
├── domain_A/
│   └── within_condition_analysis.png
├── domain_B/
│   └── within_condition_analysis.png
├── domain_C/
│   ├── contrast_level_comparison.png
│   ├── improved_prompts_comparison.png
│   ├── multiscale_analysis.png
│   ├── within_condition_analysis.png
│   ├── test_images_main.png
│   ├── test_images_artificial_lines.png
│   └── test_images_defect_detail.png
└── domain_D/
    └── within_condition_analysis.png
```

### 관련 문서
- `CA_WinCLIP_README.md`: CA-WinCLIP 설계 및 구현 계획
- `condition_aware_winclip.md`: 초기 설계 논의
- `winclip_hdmap_post_analysis.md`: Per-Image Normalization 실험 (실패)

---

## 7. 핵심 수치 요약

```
┌─────────────────────────────────────────────────────────────┐
│                    HDMAP WinCLIP 분석 요약                    │
├─────────────────────────────────────────────────────────────┤
│  Within-Condition AUROC (Cross-condition 회피 시)            │
│    Domain A: 99.2%  ████████████████████████████████████ ✓   │
│    Domain B: 99.2%  ████████████████████████████████████ ✓   │
│    Domain D: 97.2%  ██████████████████████████████████░░ ✓   │
│    Domain C: 81.0%  ████████████████████████░░░░░░░░░░░░ △   │
├─────────────────────────────────────────────────────────────┤
│  Text Template 효과: ✗ 없음 (CLIP contrast 인식 한계)         │
│  Multi-scale 최적:   Full Image (Cross), Patch (Within)     │
│  CA-WinCLIP 가치:    Cross-condition 문제 해결               │
│  Domain C 한계:      본질적으로 어려움, 추가 모델 필요         │
└─────────────────────────────────────────────────────────────┘
```
