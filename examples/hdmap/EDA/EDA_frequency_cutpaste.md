# EDA: Frequency-aware Scaled CutPaste for Anomaly Detection

## 프로젝트 배경

### Repository 구조
- **경로**: `/mnt/ex-disk/taewan.hwang/study/anomalib`
- **기반**: Anomalib (Intel의 anomaly detection 라이브러리)
- **데이터셋**: HDMAP (고정밀 기어박스 이상 감지용 데이터)
  - 경로: `datasets/HDMAP/100000_tiff_minmax`
  - 4개 도메인: domain_A, domain_B, domain_C, domain_D
  - 이미지: TIFF 형식, minmax 정규화 적용

### 현재까지의 연구 성과
1. **Scaled CutPaste + DRAEM 아키텍처** 개선
   - 기존 CutPaste 대비 스케일 조절된 패치 붙여넣기
   - HDMAP 데이터에서 이상감지 성능 개선 확인

2. **주요 모델 실험 완료**
   - DRAEM CutPaste (`draem_cutpaste`, `draem_cutpaste_clf`)
   - PaDiM, PatchCore (with DINOv2 backbone)
   - CFlow, UniNet, Reverse Distillation, GANomaly 등

### 핵심 코드 파일
| 파일 | 설명 |
|------|------|
| `examples/hdmap/single_domain/anomaly_trainer.py` | 통합 모델 훈련 클래스 |
| `src/anomalib/models/image/draem_cutpaste/` | DRAEM CutPaste 모델 |
| `src/anomalib/models/image/draem_cutpaste_clf/` | DRAEM CutPaste + CNN Classification |
| `examples/hdmap/single_domain/exp_*.json` | 실험 설정 파일들 |

---

## 새로운 연구 방향: Frequency-aware CutPaste

### 동기 (Motivation)

#### Why 1: 2D FFT 기술에 대한 관심
- 이미지의 주파수 도메인 분석/조작에 대한 기술적 관심

#### Why 2: FAIR 논문의 인사이트 (freq_aware.md 참조)
FAIR (Frequency-aware Image Restoration) 논문의 핵심:

1. **문제 재정의**: Reconstruction 기반 anomaly detection의 trade-off는 **공간 도메인이 아닌 주파수 도메인 문제**

2. **핵심 관찰 (Frequency Bias)**:
   - 정상 재구성 오류 → **고주파(high-frequency)에 편향**
   - 이상 재구성 오류 → **전 주파수 대역(low~high)에 분포**

3. **해결책**: 2D FFT 기반 High-Pass Filter로 저주파 제거 → anomaly identity mapping 차단

4. **최적 필터**: 2nd-order Butterworth HPF (BHPF)가 최적
   - Ideal HPF: ringing artifact 심함
   - Gaussian HPF: 저주파 과다 보존
   - Butterworth HPF: 타협점

---

### 제안하는 실험 파이프라인

#### 2D Frequency Filter 정의
```python
# 이미지에 2D FFT 수행 후 Low/High Pass Filter 적용
image → FFT → Filter (LPF/HPF) → iFFT → filtered_image
```

#### 실험 파라미터
| 파라미터 | 옵션 |
|---------|------|
| Filter Type | Low Pass (LPF) / High Pass (HPF) |
| Cutoff Frequency | 다양한 값 (예: 10, 30, 50, 100 등) |
| Filter Shape | Butterworth / Gaussian / Ideal |
| Butterworth Order | 1, 2, 3, ... |

---

### 조합 방안 (Integration Strategies)

#### 1안: Cut → Frequency Filter → Scaled Paste
```
원본 이미지 → Cut patch 추출 → [HPF/LPF 적용] → Scale 조절 → Paste
```
- **의도**: 패치 자체의 주파수 특성 변형
- **HPF 적용**: 패치가 edge 중심 → "구조적 이상" 느낌
- **LPF 적용**: 패치가 blur → "흐릿한 얼룩" 느낌
- **장점**: 계산량 적음, 패치만 변형

#### 2안: Scaled CutPaste → Frequency Filter → Model
```
원본 이미지 → Scaled CutPaste 완성 → [전체 이미지 HPF/LPF] → Reconstruction
```
- **의도**: FAIR와 유사하게 모델 입력을 frequency domain으로 제한
- **주의점**: 모델이 filtered 이미지를 reconstruct → 원본과 비교 방식 고려 필요

#### 3안: Dual-path Frequency Mixing (추가 제안)
```
Cut patch → [HPF] → High-freq patch
Cut patch → [LPF] → Low-freq patch
→ 두 개를 조합하거나 선택적 사용
```

#### 4안: Frequency-aware Blending (추가 제안)
```
Cut patch → FFT → 주파수별 가중치 적용 → iFFT → Paste
(예: 중간 주파수 강조, 고주파/저주파 감쇠)
```

---

### 기대 효과

1. **더 Realistic한 Pseudo Anomaly 생성**
   - 주파수 특성 조절로 자연스러운 이상 패턴 생성

2. **파라미터 민감도 감소**
   - CutPaste의 크기/위치 파라미터에 덜 민감해질 가능성

3. **모델의 Frequency Bias 활용**
   - 정상/이상 간 주파수 특성 차이를 명시적으로 활용

---

## EDA 계획 (단계별)

### Phase 1: HDMAP 데이터의 주파수 특성 분석

#### 1-1. 원본 이미지의 2D FFT 주파수 스펙트럼 분석 ✅ 완료

**목표**: 원본 이미지에 2D FFT를 적용했을 때 Normal vs Fault 이미지의 주파수 특성 차이 확인

**수행 내용**:
```python
1. Normal 이미지 전체 (train + test) 로드
2. Fault 이미지 전체 (first 500 vs last 500 분리)
3. 각 이미지에 2D FFT 수행
4. 주파수 대역별 에너지 분포 분석 (5 bands: very_low ~ very_high)
5. 평균 스펙트럼 시각화 및 비교
```

**실행 스크립트**: `run_phase1_1_eda.py`
```bash
python run_phase1_1_eda.py --all_domains --parallel --n_workers 16
```

**결과 요약** (4개 도메인, 1000_tiff_minmax 데이터셋):

**Normal 에너지 분포 (%)**:
| 도메인 | very_low | low | mid | high | very_high |
|--------|----------|-----|-----|------|-----------|
| Domain A | 93.96 | 1.31 | 1.20 | 1.29 | 2.23 |
| Domain B | 93.55 | 1.47 | 1.18 | 1.60 | 2.19 |
| Domain C | 95.11 | 0.66 | 0.71 | 0.83 | 2.70 |
| Domain D | 92.03 | 1.03 | 1.19 | 1.50 | 4.26 |

**Fault - Normal 차이 (%p)**:
| 도메인 | very_low | low | mid | high | very_high |
|--------|----------|-----|-----|------|-----------|
| Domain A | -0.31 | +0.02 | -0.01 | +0.04 | **+0.26** |
| Domain B | -0.04 | -0.17 | +0.14 | -0.20 | **+0.26** |
| Domain C | **-1.35** | +0.24 | +0.20 | +0.27 | **+0.65** |
| Domain D | -0.48 | +0.31 | +0.15 | +0.07 | -0.05 |

**결론**:
- Fault 이미지는 Normal 대비 **고주파(very_high)에서 에너지 증가** (Domain D 제외)
- Fault 이미지는 Normal 대비 **DC 성분(very_low)에서 에너지 감소** (모든 도메인)
- Domain C가 가장 큰 차이를 보임 (very_low: -1.35%, very_high: +0.65%)
- 이는 원본 이미지 자체의 특성이며, FAIR 논문의 reconstruction error 분석과는 다름

**결과 위치**: `results/phase1_1/`

---

#### 1-2. CutPaste Augmentation 주파수 분석 ✅ 완료

**목표**: CutPaste augmentation이 이미지의 주파수 특성에 어떤 변화를 주는지 분석

**수행 내용**:
```python
1. Normal 이미지 로드 후 31x95로 resize (모델 입력 크기)
2. CutPaste 적용 (cut_w=10-80, cut_h=1-2, a_fault=0-2, norm=True)
3. Original vs CutPaste 이미지 FFT 비교
4. Diff 이미지 (CutPaste - Original)의 FFT 분석
```

**실행 스크립트**: `run_phase1_2_eda.py`
```bash
python run_phase1_2_eda.py --all_domains --n_samples 100
```

**CutPaste 적용 후 에너지 변화 (%p)**:
| 도메인 | very_low | low | mid | high | very_high |
|--------|----------|-----|-----|------|-----------|
| Domain A | **-4.52** | +1.69 | +1.32 | +0.91 | +0.61 |
| Domain B | **-4.74** | +1.85 | +1.40 | +0.87 | +0.63 |
| Domain C | **-5.25** | +2.01 | +1.51 | +1.06 | +0.67 |
| Domain D | **-5.13** | +2.07 | +1.52 | +1.00 | +0.55 |

**Diff 이미지 (CutPaste가 도입한 변화)의 주파수 분포 (%)**:
| 도메인 | very_low | low | mid | high | very_high |
|--------|----------|-----|-----|------|-----------|
| Domain A | 28.35 | 23.76 | 19.22 | 14.88 | 13.80 |
| Domain B | 28.30 | 23.74 | 19.25 | 14.92 | 13.79 |
| Domain C | 28.83 | 23.85 | 19.09 | 14.56 | 13.66 |
| Domain D | 27.92 | 23.39 | 18.86 | 14.65 | 15.18 |

**결론**:
1. **CutPaste → DC(very_low) 에너지 ~5%p 감소** (모든 도메인)
2. **CutPaste → low/mid 주파수 증가** (low +1.9%p, mid +1.4%p 평균)
3. **Diff 이미지는 전 주파수 대역에 고르게 분포** (very_low 28% > low 24% > mid 19% > high 15% > very_high 14%)
4. CutPaste가 도입하는 변화 자체는 저주파 성분이 많지만, 원본 대비 상대적으로는 고주파 비중 증가

**결과 위치**: `results/phase1_2/`

---

### Phase 2: Filter 효과 시각화 ✅ 완료

**목표**: CutPaste 패치에 HPF/LPF 적용 시 주파수 특성 변화 분석

**실행 스크립트**: `run_phase2_eda.py`
```bash
python run_phase2_eda.py --all_domains --n_samples 50
```

**테스트한 필터**:
- Butterworth HPF (cutoff: 0.1~0.3, order: 2)
- Gaussian HPF (cutoff: 0.1~0.3)
- Ideal HPF (cutoff: 0.1~0.3)
- Butterworth LPF (비교용)

**Diff Image의 very_high 비율 (Domain A)**:
| 필터 | very_high 비율 |
|------|---------------|
| No Filter | **14.3%** |
| bhpf_c0.1 | 53.8% |
| bhpf_c0.2 | **66.4%** |
| bhpf_c0.3 | **76.6%** |

**핵심 결론**:
1. **HPF 적용 시 very_high 집중도 대폭 증가** (14% → 66~77%)
2. **Real Fault의 주파수 특성과 유사해짐**
3. **권장 설정**: `bhpf_c0.2_o2` (Butterworth HPF, cutoff=0.2, order=2)

**결과 위치**: `results/phase2/`

---

### Phase 3: FAIR 가설 검증 (Optional)

> **주의**: FAIR 논문의 가설은 **reconstruction error**에 대한 것임.
> 원본 이미지의 주파수 특성이 아니라, 모델이 재구성한 이미지와 원본의 차이를 분석해야 함.

#### 3-1. Reconstruction Error의 주파수 분석 (TODO)
```python
# FAIR 가설 검증을 위한 분석
1. Reconstruction 모델 (DRAEM, AutoEncoder 등) 학습
2. Normal/Fault 이미지 재구성
3. reconstruction_error = original - reconstructed
4. reconstruction_error의 2D FFT 분석
5. Normal의 오류가 고주파에 편향되는지 확인
6. Fault의 오류가 전 주파수에 분포하는지 확인
```

#### 3-2. Filter 적용이 Reconstruction 난이도에 미치는 영향 (TODO)
- 동일 DRAEM 모델로 다양한 필터 적용 이미지 reconstruction
- 어떤 필터가 anomaly를 더 "어렵게" 재구성하게 하는지 측정

---

## 구현 시 참고할 코드 위치

### 현재 CutPaste 구현
```
src/anomalib/models/image/draem_cutpaste/
├── lightning_model.py    # Lightning 모델
├── torch_model.py        # PyTorch 모델
└── utils/
    └── augmenters.py     # CutPaste augmentation 구현
```

### 수정 대상 (예상)
1. `augmenters.py`의 CutPaste 클래스에 frequency filter 옵션 추가
2. 또는 별도의 `FrequencyAugmenter` 클래스 구현

---

## 참고 자료

### FAIR 논문 요약 (freq_aware.md)
- 위치: `/mnt/ex-disk/taewan.hwang/study/anomalib/examples/hdmap/single_domain/freq_aware.md`
- 핵심: 2D FFT 기반 HPF로 저주파 제거 → anomaly identity mapping 차단

### 현재 실험 설정 파일들
| 파일 | 모델 |
|------|------|
| `exp_41_draem_cp.json` | DRAEM CutPaste |
| `exp_42_draem_cp_clf.json` | DRAEM CutPaste + Classification |
| `exp_28_uninet.json` | UniNet |
| `exp_26_reverse_distillation.json` | Reverse Distillation |

---

## 다음 단계

1. **Phase 1-1 시작**: HDMAP Normal/Anomaly 이미지의 2D FFT 분석
2. 주파수 스펙트럼 시각화 및 비교
3. 결과 기반으로 어떤 주파수 대역을 타겟할지 결정
4. Phase 2로 진행: 다양한 필터 효과 시각화

---

## 세션 시작 시 Claude에게 전달할 컨텍스트

```
이 프로젝트는 anomalib 기반 HDMAP 이상감지 연구입니다.
현재까지 Scaled CutPaste + DRAEM으로 성능 개선을 확인했고,
이제 2D FFT 기반 frequency filter를 CutPaste에 통합하는 실험을 계획하고 있습니다.

먼저 EDA로:
1. HDMAP Normal/Anomaly의 주파수 특성 분석
2. 다양한 필터 (HPF/LPF, Butterworth/Gaussian/Ideal) 효과 시각화

를 수행한 후, 좋은 결과를 보이는 방향으로만 구현을 진행할 예정입니다.

참고 문서: examples/hdmap/single_domain/freq_aware.md (FAIR 논문 요약)
EDA 계획: examples/hdmap/single_domain/EDA_frequency_cutpaste.md (이 파일)
```
