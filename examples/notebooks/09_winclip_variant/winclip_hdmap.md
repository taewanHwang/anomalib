# WinCLIP HDMAP Validation

WinCLIP 모델의 HDMAP 데이터셋 성능 평가 스크립트.

## 실험 환경

### 데이터셋
- **경로**: `datasets/HDMAP/1000_tiff_minmax`
- **도메인**: 4개 (domain_A, domain_B, domain_C, domain_D)
- **포맷**: TIFF (float32, minmax normalized)
- **Task**: Classification (no pixel-level masks)

### WinCLIP 특성
- **Zero-shot (k=0)**: 텍스트 프롬프트만으로 이상 탐지 (학습 불필요)
- **Few-shot (k=1,2,4)**: Normal reference 이미지 k장 사용
- **메트릭**: Image AUROC (HDMAP은 classification task)

## 실행 방법

**모든 명령어는 anomalib 루트 디렉토리에서 실행**

### 실험 1: Baseline (GPU 0-3)
- **class_name**: `"industrial sensor data"` (기본값)
- 일반적인 산업 센서 데이터 프롬프트

```bash
# Domain A (GPU 0)
nohup python examples/notebooks/09_winclip_variant/winclip_hdmap_validation.py \
    --domains domain_A --k-shots 0 1 2 --gpu 0 \
    --result-dir ./results/winclip_hdmap_baseline \
    > logs/winclip_baseline_domainA.log 2>&1 &
sleep 3
# Domain B (GPU 1)
nohup python examples/notebooks/09_winclip_variant/winclip_hdmap_validation.py \
    --domains domain_B --k-shots 0 1 2 --gpu 1 \
    --result-dir ./results/winclip_hdmap_baseline \
    > logs/winclip_baseline_domainB.log 2>&1 &
sleep 3
# Domain C (GPU 2)
nohup python examples/notebooks/09_winclip_variant/winclip_hdmap_validation.py \
    --domains domain_C --k-shots 0 1 2 --gpu 2 \
    --result-dir ./results/winclip_hdmap_baseline \
    > logs/winclip_baseline_domainC.log 2>&1 &
sleep 3
# Domain D (GPU 3)
nohup python examples/notebooks/09_winclip_variant/winclip_hdmap_validation.py \
    --domains domain_D --k-shots 0 1 2 --gpu 3 \
    --result-dir ./results/winclip_hdmap_baseline \
    > logs/winclip_baseline_domainD.log 2>&1 &
```

### 실험 2: Partial Prior Knowledge (GPU 4-7)
- **class_name**: `"sensor fault detection heatmap"`
- 고장 탐지 목적 + heatmap 형태 암시 (구체적 패턴 정보 없음)

```bash
# Domain A (GPU 4)
nohup python examples/notebooks/09_winclip_variant/winclip_hdmap_validation.py \
    --domains domain_A --k-shots 0 1 2 --gpu 4 \
    --class-name "sensor fault detection heatmap" \
    --result-dir ./results/winclip_hdmap_partial_prior \
    > logs/winclip_partial_domainA.log 2>&1 &
sleep 3

# Domain B (GPU 5)
nohup python examples/notebooks/09_winclip_variant/winclip_hdmap_validation.py \
    --domains domain_B --k-shots 0 1 2 --gpu 5 \
    --class-name "sensor fault detection heatmap" \
    --result-dir ./results/winclip_hdmap_partial_prior \
    > logs/winclip_partial_domainB.log 2>&1 &
sleep 3

# Domain C (GPU 6)
nohup python examples/notebooks/09_winclip_variant/winclip_hdmap_validation.py \
    --domains domain_C --k-shots 0 1 2 --gpu 6 \
    --class-name "sensor fault detection heatmap" \
    --result-dir ./results/winclip_hdmap_partial_prior \
    > logs/winclip_partial_domainC.log 2>&1 &
sleep 3

# Domain D (GPU 7)
nohup python examples/notebooks/09_winclip_variant/winclip_hdmap_validation.py \
    --domains domain_D --k-shots 0 1 2 --gpu 7 \
    --class-name "sensor fault detection heatmap" \
    --result-dir ./results/winclip_hdmap_partial_prior \
    > logs/winclip_partial_domainD.log 2>&1 &
```

### 실험 3: Full Domain Knowledge (GPU 8-11)
- **class_name**: `"sensor heatmap where horizontal bands indicate equipment fault"`
- **도메인 지식**: 고장 시 가로 패턴(horizontal bands) 발생, 정상 시 균일한 패턴

```bash
# Domain A (GPU 8)
nohup python examples/notebooks/09_winclip_variant/winclip_hdmap_validation.py \
    --domains domain_A --k-shots 0 1 2 --gpu 8 \
    --class-name "sensor heatmap where horizontal bands indicate equipment fault" \
    --result-dir ./results/winclip_hdmap_full_prior \
    > logs/winclip_full_domainA.log 2>&1 &
sleep 3

# Domain B (GPU 9)
nohup python examples/notebooks/09_winclip_variant/winclip_hdmap_validation.py \
    --domains domain_B --k-shots 0 1 2 --gpu 9 \
    --class-name "sensor heatmap where horizontal bands indicate equipment fault" \
    --result-dir ./results/winclip_hdmap_full_prior \
    > logs/winclip_full_domainB.log 2>&1 &
sleep 3

# Domain C (GPU 10)
nohup python examples/notebooks/09_winclip_variant/winclip_hdmap_validation.py \
    --domains domain_C --k-shots 0 1 2 --gpu 10 \
    --class-name "sensor heatmap where horizontal bands indicate equipment fault" \
    --result-dir ./results/winclip_hdmap_full_prior \
    > logs/winclip_full_domainC.log 2>&1 &
sleep 3

# Domain D (GPU 11)
nohup python examples/notebooks/09_winclip_variant/winclip_hdmap_validation.py \
    --domains domain_D --k-shots 0 1 2 --gpu 11 \
    --class-name "sensor heatmap where horizontal bands indicate equipment fault" \
    --result-dir ./results/winclip_hdmap_full_prior \
    > logs/winclip_full_domainD.log 2>&1 &
```

### 실험 4: Expert Domain Knowledge (GPU 12-15)
- **class_name**: `"planetary gearbox vibration heatmap where bright horizontal bands indicate gear tooth defect"`
- **전문가 수준 도메인 지식**: 유성 기어박스 진동, 밝은 픽셀=강한 진동, 가로 밴드=특정 기어치 결함

```bash
# Domain A (GPU 12)
nohup python examples/notebooks/09_winclip_variant/winclip_hdmap_validation.py \
    --domains domain_A --k-shots 0 1 2 --gpu 12 \
    --class-name "planetary gearbox vibration heatmap where bright horizontal bands indicate gear tooth defect" \
    --result-dir ./results/winclip_hdmap_expert_prior \
    > logs/winclip_expert_domainA.log 2>&1 &
sleep 3

# Domain B (GPU 13)
nohup python examples/notebooks/09_winclip_variant/winclip_hdmap_validation.py \
    --domains domain_B --k-shots 0 1 2 --gpu 13 \
    --class-name "planetary gearbox vibration heatmap where bright horizontal bands indicate gear tooth defect" \
    --result-dir ./results/winclip_hdmap_expert_prior \
    > logs/winclip_expert_domainB.log 2>&1 &
sleep 3

# Domain C (GPU 14)
nohup python examples/notebooks/09_winclip_variant/winclip_hdmap_validation.py \
    --domains domain_C --k-shots 0 1 2 --gpu 14 \
    --class-name "planetary gearbox vibration heatmap where bright horizontal bands indicate gear tooth defect" \
    --result-dir ./results/winclip_hdmap_expert_prior \
    > logs/winclip_expert_domainC.log 2>&1 &
sleep 3

# Domain D (GPU 15)
nohup python examples/notebooks/09_winclip_variant/winclip_hdmap_validation.py \
    --domains domain_D --k-shots 0 1 2 --gpu 15 \
    --class-name "planetary gearbox vibration heatmap where bright horizontal bands indicate gear tooth defect" \
    --result-dir ./results/winclip_hdmap_expert_prior \
    > logs/winclip_expert_domainD.log 2>&1 &
```

### 로그 확인

```bash
# 실시간 로그 확인
tail -f logs/winclip_baseline_domainA.log

# 전체 실험 상태 확인
for log in logs/winclip_*.log; do echo "=== $(basename $log) ==="; tail -3 $log; done
```

### 프롬프트 비교 요약

| 실험 | class_name | GPU |
|------|-----------|-----|
| Baseline | `industrial sensor data` | 0-3 |
| Partial Prior | `sensor fault detection heatmap` | 4-7 |
| Full Prior | `sensor heatmap where horizontal bands indicate equipment fault` | 8-11 |
| Expert Prior | `planetary gearbox vibration heatmap where bright horizontal bands indicate gear tooth defect` | 12-15 |

**도메인 지식 수준:**
| 실험 | 제공 정보 |
|------|----------|
| Baseline | 일반적 산업 센서 데이터 |
| Partial | + 고장 탐지 목적, heatmap 형태 |
| Full | + 가로 밴드 = 장비 고장 |
| Expert | + 유성 기어박스, 밝은 픽셀 = 강한 진동, 가로 밴드 = 기어치 결함 |

## CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--domains` | `domain_A` | 테스트할 도메인 (`all`로 전체 선택) |
| `--k-shots` | `0 1 2 4` | 테스트할 k_shot 값들 |
| `--dataset-root` | `./datasets/HDMAP/1000_tiff_minmax` | HDMAP 데이터셋 경로 |
| `--result-dir` | `./results/winclip_hdmap_validation` | 결과 저장 경로 |
| `--gpu` | `0` | GPU ID |
| `--class-name` | `industrial sensor data` | WinCLIP 텍스트 프롬프트용 클래스명 |
| `--image-size` | `240` | CLIP 입력 이미지 크기 (ViT-B-16-plus-240용) |
| `--no-visualizations` | (flag) | 시각화 생성 스킵 (빠른 실행용) |

## 결과 구조

```
results/winclip_hdmap_validation/
└── 20241226_123456/              # 타임스탬프
    ├── experiment_settings.json  # 실험 설정
    ├── summary.json              # 전체 결과 요약
    ├── visualizations/           # 시각화 결과
    │   ├── all_domains_score_comparison_zero_shot.png  # 전체 도메인 비교 (2x2)
    │   ├── all_domains_score_comparison_1_shot.png
    │   ├── domain_A/
    │   │   └── score_distribution.png   # 도메인별 score 분포
    │   ├── domain_B/
    │   │   └── score_distribution.png
    │   └── ...
    ├── scores/                   # 예측 점수 CSV
    │   ├── domain_A_zero_shot_scores.csv
    │   ├── domain_A_1_shot_scores.csv
    │   └── ...
    ├── domain_A/
    │   ├── zero-shot/
    │   │   └── results.json
    │   ├── 1-shot/
    │   │   └── results.json
    │   ├── 2-shot/
    │   │   └── results.json
    │   └── 4-shot/
    │       └── results.json
    ├── domain_B/
    │   └── ...
    ├── domain_C/
    │   └── ...
    └── domain_D/
        └── ...
```

## 시각화 출력

### 4-Column Image Visualization
각 이미지별 4개 컬럼으로 구성된 시각화 (domain_A/zero-shot/images_4col/):
1. **Image**: 원본 이미지
2. **Anomaly (Auto)**: Anomaly Map overlay (이미지별 min-max 자동 스케일)
3. **Anomaly (0-1)**: Anomaly Map overlay (고정 0-1 스케일)
4. **Pred Mask**: 예측 마스크 overlay

### Per-Domain Score Distribution
각 도메인별 Good/Fault 클래스의 anomaly score 분포 히스토그램

### All Domains Score Comparison
4개 도메인을 2x2 subplot으로 비교하는 플롯

### Score CSV Export
각 이미지별 예측 점수를 CSV로 저장 (후속 분석용)

## 주의사항

### HDMAP 데이터 특성
- **Classification Task**: HDMAP은 pixel-level mask가 없으므로 Image AUROC가 주요 메트릭
- **TIFF Float32**: 데이터가 float32 precision으로 저장되어 있음 (minmax normalized)
- **4 Domains**: 각 도메인은 다른 운영 조건/센서 위치를 나타냄
- **이미지 크기**: 원본 이미지는 31x95로 매우 작아서 240x240으로 자동 리사이즈됨 (CLIP 입력 요구사항)

### WinCLIP 프롬프트
- `--class-name` 옵션으로 텍스트 프롬프트 조정 가능
- 기본값: "industrial sensor data"
- 도메인 특성에 맞게 조정 시 성능 변화 가능

### 메모리 관리
- Few-shot (k>0) 모드는 reference 이미지를 메모리에 로드
- GPU 메모리 부족 시 `eval_batch_size` 줄이기

```python
# winclip_hdmap_validation.py에서 수정
datamodule = HDMAPDataModule(
    eval_batch_size=16,  # 32에서 줄임
    ...
)
```

## 문제 해결

### CLIP 모델 다운로드 오류
```bash
# open_clip 패키지 설치 확인
pip install open-clip-torch
```

### 데이터셋 경로 오류
```bash
# 데이터셋 존재 확인
ls datasets/HDMAP/1000_tiff_minmax/

# 출력 예시:
# domain_A  domain_B  domain_C  domain_D
```

### GPU 메모리 부족
```bash
# 배치 크기 줄이기 또는 다른 GPU 사용
nohup python examples/notebooks/09_winclip_variant/winclip_hdmap_validation.py \
    --domains domain_A \
    --k-shots 0 \
    --gpu 1 \
    > logs/winclip_hdmap_gpu1.log 2>&1 &
```

---

## 실험 결과 (Baseline)

**실험 일자**: 2025-12-26

### 프롬프트 전략별 성능 비교 (Mean Image AUROC)

| 실험 | Zero-shot | 1-shot | 2-shot | class_name |
|------|-----------|--------|--------|------------|
| Baseline | 79.45% | 82.83% | 84.92% | `industrial sensor data` |
| **Partial Prior** | **88.25%** | **85.76%** | **87.62%** | `sensor fault detection heatmap` |
| Full Prior | 72.90% | 80.36% | 84.66% | `sensor heatmap where horizontal bands indicate equipment fault` |
| Expert Prior | 85.63% | 83.26% | 86.27% | `planetary gearbox vibration heatmap where bright horizontal bands indicate gear tooth defect` |

### 도메인별 상세 결과 (Image AUROC)

#### Baseline (`industrial sensor data`)
| Mode | domain_A | domain_B | domain_C | domain_D | Mean |
|------|----------|----------|----------|----------|------|
| Zero-shot | 82.23% | 81.40% | 73.74% | 80.42% | 79.45% |
| 1-shot | 81.10% | 85.88% | 81.44% | 82.91% | 82.83% |
| 2-shot | 84.38% | 89.72% | 79.11% | 86.48% | 84.92% |

#### Partial Prior (`sensor fault detection heatmap`)
| Mode | domain_A | domain_B | domain_C | domain_D | Mean |
|------|----------|----------|----------|----------|------|
| Zero-shot | 85.83% | **97.72%** | 75.29% | 94.15% | **88.25%** |
| 1-shot | 83.42% | 95.17% | 78.04% | 86.41% | 85.76% |
| 2-shot | 88.89% | 93.98% | 80.02% | 87.58% | 87.62% |

#### Full Prior (`sensor heatmap where horizontal bands indicate equipment fault`)
| Mode | domain_A | domain_B | domain_C | domain_D | Mean |
|------|----------|----------|----------|----------|------|
| Zero-shot | 62.65% | 80.61% | 67.26% | 81.07% | 72.90% |
| 1-shot | 76.88% | 82.51% | 75.23% | 86.84% | 80.36% |
| 2-shot | 79.36% | 94.16% | 77.37% | 87.74% | 84.66% |

#### Expert Prior (`planetary gearbox vibration heatmap where bright horizontal bands indicate gear tooth defect`)
| Mode | domain_A | domain_B | domain_C | domain_D | Mean |
|------|----------|----------|----------|----------|------|
| Zero-shot | 87.87% | 85.23% | 84.14% | 85.28% | 85.63% |
| 1-shot | 81.55% | 87.96% | 78.72% | 84.83% | 83.26% |
| 2-shot | 84.18% | 92.48% | 81.05% | 87.37% | 86.27% |

### 주요 발견

1. **Partial Prior가 최고 성능**: Zero-shot에서 88.25% 달성 (Baseline 대비 +8.8%p)
   - 간결한 `"sensor fault detection heatmap"` 프롬프트가 가장 효과적

2. **Full Prior는 오히려 성능 저하**: Zero-shot 72.90% (최저)
   - 구체적인 "horizontal bands indicate fault" 설명이 CLIP에게 혼란 유발

3. **Expert Prior는 중간 성능**: Zero-shot 85.63%
   - "planetary gearbox", "gear tooth" 등 전문 용어가 CLIP 학습 데이터에 부족했을 가능성

4. **Few-shot 효과**:
   - Baseline: k 증가 시 지속적 성능 향상 (+5.5%p from k=0 to k=2)
   - Partial/Expert: Zero-shot이 이미 높아 few-shot 개선 폭 제한적

### 결론

| 권장 설정 | 조건 |
|----------|------|
| `sensor fault detection heatmap` + Zero-shot | 학습 데이터 없이 최고 성능 (88.25%) |
| `sensor fault detection heatmap` + 2-shot | 정상 샘플 2장 사용 시 (87.62%) |

**핵심 인사이트**: CLIP은 도메인 특화 전문 용어보다 **일반적이지만 목적을 명시한 프롬프트**를 더 잘 이해함.
