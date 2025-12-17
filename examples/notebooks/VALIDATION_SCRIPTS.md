# Anomaly Detection Validation Scripts

MVTec-AD 및 HDMAP 데이터셋에서 UniNet과 Dinomaly(DINOv2) 모델을 검증하는 스크립트입니다.

## 사전 준비

```bash
cd /mnt/ex-disk/taewan.hwang/study/anomalib
mkdir -p logs  # 로그 디렉토리 생성
```

### 데이터셋 경로

| 데이터셋 | 경로 | 설명 |
|---------|------|------|
| MVTec-AD | `datasets/MVTecAD/` | 표준 이상 탐지 벤치마크 |
| HDMAP (PNG) | `datasets/HDMAP/1000_png/` | 원본 RGB 이미지 |
| HDMAP (FFT Magnitude) | `datasets/HDMAP/1000_png_2dfft/` | 2D FFT Magnitude Spectrum |
| HDMAP (FFT Phase) | `datasets/HDMAP/1000_png_2dfft_phase/` | 2D FFT Phase Spectrum |
| HDMAP (TIFF) | `datasets/HDMAP/1000_tiff_minmax/` | TIFF 형식 |

---

## 1. UniNet MVTec-AD Validation

**파일**: `examples/notebooks/uninet_mvtec_validation.py`

UniNet 모델이 MVTec-AD에서 논문 성능(99.9%+)을 재현하는지 검증합니다.

### 기본 사용법 (nohup)

```bash
# 단일 카테고리 (bottle)
nohup python examples/notebooks/uninet_mvtec_validation.py \
    --categories bottle --gpu 0 \
    > logs/uninet_mvtec_bottle_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 여러 카테고리
nohup python examples/notebooks/uninet_mvtec_validation.py \
    --categories bottle cable transistor --gpu 0 \
    > logs/uninet_mvtec_multi_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 전체 15개 카테고리
nohup python examples/notebooks/uninet_mvtec_validation.py \
    --categories all --gpu 0 \
    > logs/uninet_mvtec_all_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--categories` | 테스트할 카테고리 (`all` = 전체) | `bottle` |
| `--gpu` | GPU ID | `0` |
| `--epochs` | 학습 에폭 수 | `100` |
| `--batch-size` | 배치 크기 | `8` |
| `--output-dir` | 결과 저장 경로 | `./results/uninet_mvtec_validation` |

### 예시

```bash
# 빠른 테스트 (10 epochs)
nohup python examples/notebooks/uninet_mvtec_validation.py \
    --categories bottle --gpu 0 --epochs 10 \
    > logs/uninet_mvtec_bottle_quick_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# GPU 4번 사용, 전체 카테고리
nohup python examples/notebooks/uninet_mvtec_validation.py \
    --categories all --gpu 4 --epochs 100 \
    > logs/uninet_mvtec_all_gpu4_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 기대 결과

- **Image AUROC > 99%**: 논문 성능 재현 확인

---

## 2. Dinomaly (DINOv2) MVTec-AD Validation

**파일**: `examples/notebooks/dinomaly_mvtec_validation.py`

DINOv2 백본 기반 Dinomaly 모델을 MVTec-AD에서 검증합니다.

### 기본 사용법 (nohup)

```bash
# 단일 카테고리 (bottle)
nohup python examples/notebooks/dinomaly_mvtec_validation.py \
    --categories bottle --gpu 0 \
    > logs/dinomaly_mvtec_bottle_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 여러 카테고리
nohup python examples/notebooks/dinomaly_mvtec_validation.py \
    --categories bottle cable --gpu 0 \
    > logs/dinomaly_mvtec_multi_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 전체 카테고리
nohup python examples/notebooks/dinomaly_mvtec_validation.py \
    --categories all --gpu 0 \
    > logs/dinomaly_mvtec_all_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--categories` | 테스트할 카테고리 (`all` = 전체) | `bottle` |
| `--gpu` | GPU ID | `0` |
| `--max-steps` | 최대 학습 스텝 | `5000` |
| `--batch-size` | 배치 크기 | `8` |
| `--encoder` | DINOv2 인코더 크기 | `dinov2reg_vit_base_14` |
| `--output-dir` | 결과 저장 경로 | `./results/dinomaly_mvtec_validation` |

### DINOv2 인코더 옵션

| 인코더 | 파라미터 | 속도 |
|--------|---------|------|
| `dinov2reg_vit_small_14` | 작음 | 빠름 |
| `dinov2reg_vit_base_14` | 중간 | 중간 |
| `dinov2reg_vit_large_14` | 큼 | 느림 |

### 예시

```bash
# 빠른 테스트 (1000 steps, small encoder)
nohup python examples/notebooks/dinomaly_mvtec_validation.py \
    --categories bottle --gpu 0 --max-steps 1000 --encoder dinov2reg_vit_small_14 \
    > logs/dinomaly_mvtec_quick_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 전체 테스트 (large encoder)
nohup python examples/notebooks/dinomaly_mvtec_validation.py \
    --categories all --gpu 4 --encoder dinov2reg_vit_large_14 \
    > logs/dinomaly_mvtec_all_large_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 기대 결과

- **Image AUROC ≥ 99%**: UniNet과 동등 또는 상회

---

## 3. HDMAP Validation (UniNet / Dinomaly)

**파일**: `examples/notebooks/hdmap_validation.py`

HDMAP 데이터셋(PNG 변환)에서 UniNet과 Dinomaly 성능을 비교합니다.

### 기본 사용법 (nohup)

```bash
# UniNet - domain_A
nohup python examples/notebooks/hdmap_validation.py \
    --model uninet --domain domain_A --gpu 0 \
    > logs/hdmap_uninet_domainA_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Dinomaly - domain_A
nohup python examples/notebooks/hdmap_validation.py \
    --model dinomaly --domain domain_A --gpu 0 \
    > logs/hdmap_dinomaly_domainA_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# UniNet - 전체 도메인
nohup python examples/notebooks/hdmap_validation.py \
    --model uninet --domain all --gpu 0 \
    > logs/hdmap_uninet_all_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Dinomaly - 전체 도메인
nohup python examples/notebooks/hdmap_validation.py \
    --model dinomaly --domain all --gpu 0 \
    > logs/hdmap_dinomaly_all_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--model` | `uninet` 또는 `dinomaly` | **필수** |
| `--domain` | `domain_A/B/C/D` 또는 `all` | `domain_A` |
| `--gpu` | GPU ID | `0` |
| `--epochs` | UniNet 에폭 수 | `100` |
| `--max-steps` | Dinomaly 스텝 수 | `5000` |
| `--temperature` | UniNet temperature | `2.0` |
| `--encoder` | DINOv2 인코더 (Dinomaly) | `dinov2reg_vit_base_14` |
| `--batch-size` | 배치 크기 | `8` |
| `--output-dir` | 결과 저장 경로 | `./results/hdmap_validation` |
| `--dataset` | 데이터셋 타입 (`png`, `fft`, `fft_phase`) | `png` |

### 예시

```bash
# UniNet 빠른 테스트 (10 epochs)
nohup python examples/notebooks/hdmap_validation.py \
    --model uninet --domain domain_A --epochs 10 --gpu 4 \
    > logs/hdmap_uninet_quick_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Dinomaly 빠른 테스트 (1000 steps)
nohup python examples/notebooks/hdmap_validation.py \
    --model dinomaly --domain domain_A --max-steps 1000 --gpu 4 \
    > logs/hdmap_dinomaly_quick_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# UniNet 전체 도메인 (temperature 조정)
nohup python examples/notebooks/hdmap_validation.py \
    --model uninet --domain all --temperature 0.1 --gpu 4 \
    > logs/hdmap_uninet_all_temp01_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Dinomaly large encoder
nohup python examples/notebooks/hdmap_validation.py \
    --model dinomaly --domain all --encoder dinov2reg_vit_large_14 --gpu 4 \
    > logs/hdmap_dinomaly_all_large_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### HDMAP 장기 학습 (Long Training)

이전 실험 결과:
- Dinomaly Large (5000 steps): domain_A 99.11%, domain_B 99.97%, domain_C 93.05%, domain_D 99.65%
- UniNet (100 epochs): domain_A 67.59%

```bash
# Dinomaly Large - 전체 도메인, 10000 steps (약 1.5시간)
nohup python examples/notebooks/hdmap_validation.py \
    --model dinomaly --domain all --encoder dinov2reg_vit_large_14 --max-steps 10000 --gpu 4 \
    > logs/hdmap_dinomaly_all_large_10k_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Dinomaly Large - domain_C만 (가장 어려운 도메인), 15000 steps
nohup python examples/notebooks/hdmap_validation.py \
    --model dinomaly --domain domain_C --encoder dinov2reg_vit_large_14 --max-steps 15000 --gpu 5 \
    > logs/hdmap_dinomaly_domainC_large_15k_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# UniNet - 전체 도메인, 500 epochs (약 3시간)
nohup python examples/notebooks/hdmap_validation.py \
    --model uninet --domain all --epochs 500 --gpu 6 \
    > logs/hdmap_uninet_all_500ep_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# UniNet - temperature 0.1 테스트 (MVTec cable/pill/transistor 설정)
nohup python examples/notebooks/hdmap_validation.py \
    --model uninet --domain all --epochs 500 --temperature 0.1 --gpu 7 \
    > logs/hdmap_uninet_all_500ep_temp01_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### HDMAP 2D FFT 실험 (Frequency Domain)

2D FFT Magnitude Spectrum으로 변환된 이미지를 사용한 실험입니다.
주파수 도메인에서의 이상 탐지 성능을 비교합니다.

```bash
# Dinomaly Large - FFT 데이터셋, 전체 도메인, 5000 steps
nohup python examples/notebooks/hdmap_validation.py \
    --model dinomaly --domain all --encoder dinov2reg_vit_large_14 --dataset fft --gpu 4 \
    > logs/hdmap_dinomaly_all_large_fft_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Dinomaly Large - FFT 데이터셋, domain_C만 (어려운 도메인), 5000 steps
nohup python examples/notebooks/hdmap_validation.py \
    --model dinomaly --domain domain_C --encoder dinov2reg_vit_large_14 --dataset fft --gpu 5 \
    > logs/hdmap_dinomaly_domainC_large_fft_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# UniNet - FFT 데이터셋, 전체 도메인, 100 epochs
nohup python examples/notebooks/hdmap_validation.py \
    --model uninet --domain all --epochs 100 --dataset fft --gpu 6 \
    > logs/hdmap_uninet_all_fft_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# PNG vs FFT 비교 실험 (동시 실행)
# PNG (기존)
nohup python examples/notebooks/hdmap_validation.py \
    --model dinomaly --domain all --encoder dinov2reg_vit_large_14 --dataset png --gpu 4 \
    > logs/hdmap_dinomaly_all_large_png_compare_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# FFT Magnitude
nohup python examples/notebooks/hdmap_validation.py \
    --model dinomaly --domain all --encoder dinov2reg_vit_large_14 --dataset fft --gpu 5 \
    > logs/hdmap_dinomaly_all_large_fft_compare_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# FFT Phase
nohup python examples/notebooks/hdmap_validation.py \
    --model dinomaly --domain all --encoder dinov2reg_vit_large_14 --dataset fft_phase --gpu 6 \
    > logs/hdmap_dinomaly_all_large_fft_phase_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## 로그 및 프로세스 관리

### 실행 중인 프로세스 확인

```bash
# 실행 중인 Python 프로세스 확인
ps aux | grep python | grep validation

# GPU 사용량 확인
nvidia-smi
```

### 로그 실시간 확인

```bash
# 최신 로그 파일 tail
tail -f logs/$(ls -t logs/ | head -1)

# 특정 로그 파일 확인
tail -f logs/hdmap_uninet_domainA_20251214_143000.log
```

### 프로세스 종료

```bash
# PID로 종료
kill <PID>

# 특정 스크립트 모두 종료
pkill -f "hdmap_validation.py"
```

---

## 결과 확인

### 결과 디렉토리 구조

```
results/
├── uninet_mvtec_validation/
│   └── 20251214_HHMMSS/
│       ├── experiment_settings.json
│       ├── summary.json
│       └── bottle/
│           ├── results.json
│           ├── checkpoints/
│           └── UniNet/MVTecAD/bottle/v0/images/  # Heatmap 이미지
├── dinomaly_mvtec_validation/
│   └── ...
└── hdmap_validation/
    └── 20251214_HHMMSS/
        ├── experiment_settings.json
        ├── summary.json
        ├── uninet_domain_A/
        └── dinomaly_domain_A/

logs/
├── uninet_mvtec_bottle_20251214_143000.log
├── dinomaly_mvtec_all_20251214_150000.log
└── hdmap_uninet_domainA_20251214_160000.log
```

### 결과 요약 확인

```bash
# Summary JSON 확인
cat results/uninet_mvtec_validation/*/summary.json | python -m json.tool

# 최신 결과 확인
cat results/hdmap_validation/$(ls -t results/hdmap_validation/ | head -1)/summary.json | python -m json.tool

# Heatmap 이미지 위치
ls results/uninet_mvtec_validation/*/bottle/UniNet/MVTecAD/bottle/v0/images/test/
```

---

## 참고: MVTec-AD 카테고리

| 카테고리 | Temperature (UniNet) |
|---------|---------------------|
| bottle | 2.0 |
| cable | 0.1 (특수) |
| capsule | 2.0 |
| carpet | 2.0 |
| grid | 2.0 |
| hazelnut | 2.0 |
| leather | 2.0 |
| metal_nut | 2.0 |
| pill | 0.1 (특수) |
| screw | 2.0 |
| tile | 2.0 |
| toothbrush | 2.0 |
| transistor | 0.1 (특수) |
| wood | 2.0 |
| zipper | 2.0 |

---

## 벤치마크 결과 (MVTec-AD bottle 기준)

| 모델 | 백본 | Image AUROC |
|------|------|-------------|
| UniNet | wide_resnet50_2 | 99.92% |
| Dinomaly | DINOv2 base | 100.00% |

---

## HDMAP 재실험 (수정된 AUROC 계산 적용)

수정된 `hdmap_validation.py`로 AUROC를 sklearn roc_auc_score로 직접 계산합니다.

### FFT 데이터셋 (GPU 0-5)

```bash
# FFT 200 steps - all domains (GPU 0)
nohup python examples/notebooks/hdmap_validation.py \
    --model dinomaly --domain all --encoder dinov2reg_vit_large_14 --dataset fft --max-steps 200 --gpu 0 \
    > logs/hdmap_dinomaly_all_fft_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# FFT 1000 steps - all domains (GPU 1)
nohup python examples/notebooks/hdmap_validation.py \
    --model dinomaly --domain all --encoder dinov2reg_vit_large_14 --dataset fft --max-steps 1000 --gpu 1 \
    > logs/hdmap_dinomaly_all_fft_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# FFT 3000 steps - all domains (GPU 2)
nohup python examples/notebooks/hdmap_validation.py \
    --model dinomaly --domain all --encoder dinov2reg_vit_large_14 --dataset fft --max-steps 3000 --gpu 2 \
    > logs/hdmap_dinomaly_all_fft_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# FFT 5000 steps - all domains (GPU 3)
nohup python examples/notebooks/hdmap_validation.py \
    --model dinomaly --domain all --encoder dinov2reg_vit_large_14 --dataset fft --max-steps 5000 --gpu 3 \
    > logs/hdmap_dinomaly_all_fft_5000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# FFT 10000 steps - domain_C only (GPU 4)
nohup python examples/notebooks/hdmap_validation.py \
    --model dinomaly --domain domain_C --encoder dinov2reg_vit_large_14 --dataset fft --max-steps 10000 --gpu 4 \
    > logs/hdmap_dinomaly_domainC_fft_10000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# FFT 15000 steps - domain_C only (GPU 5)
nohup python examples/notebooks/hdmap_validation.py \
    --model dinomaly --domain domain_C --encoder dinov2reg_vit_large_14 --dataset fft --max-steps 15000 --gpu 5 \
    > logs/hdmap_dinomaly_domainC_fft_15000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### PNG 데이터셋 (GPU 6-11)

```bash
# PNG 200 steps - all domains (GPU 6)
nohup python examples/notebooks/hdmap_validation.py \
    --model dinomaly --domain all --encoder dinov2reg_vit_large_14 --dataset png --max-steps 200 --gpu 6 \
    > logs/hdmap_dinomaly_all_png_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# PNG 1000 steps - all domains (GPU 7)
nohup python examples/notebooks/hdmap_validation.py \
    --model dinomaly --domain all --encoder dinov2reg_vit_large_14 --dataset png --max-steps 1000 --gpu 7 \
    > logs/hdmap_dinomaly_all_png_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# PNG 3000 steps - all domains (GPU 8)
nohup python examples/notebooks/hdmap_validation.py \
    --model dinomaly --domain all --encoder dinov2reg_vit_large_14 --dataset png --max-steps 3000 --gpu 8 \
    > logs/hdmap_dinomaly_all_png_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# PNG 5000 steps - all domains (GPU 9)
nohup python examples/notebooks/hdmap_validation.py \
    --model dinomaly --domain all --encoder dinov2reg_vit_large_14 --dataset png --max-steps 5000 --gpu 9 \
    > logs/hdmap_dinomaly_all_png_5000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# PNG 10000 steps - domain_C only (GPU 10)
nohup python examples/notebooks/hdmap_validation.py \
    --model dinomaly --domain domain_C --encoder dinov2reg_vit_large_14 --dataset png --max-steps 10000 --gpu 10 \
    > logs/hdmap_dinomaly_domainC_png_10000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# PNG 15000 steps - domain_C only (GPU 11)
nohup python examples/notebooks/hdmap_validation.py \
    --model dinomaly --domain domain_C --encoder dinov2reg_vit_large_14 --dataset png --max-steps 15000 --gpu 11 \
    > logs/hdmap_dinomaly_domainC_png_15000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## 4. HDMAP Adaptive Dropout Validation

**파일**: `examples/notebooks/hdmap_adaptive_validation.py`

DinomalyAdaptive 모델의 orientation entropy 기반 adaptive dropout 효과를 검증합니다.

### 핵심 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--sensitivity` | Entropy→Dropout 민감도 (0=고정 dropout) | `4.0` |
| `--normal-entropy` | Normal 샘플 기준 entropy | `0.53` |
| `--max-steps` | 최대 학습 스텝 | `5000` |

### Domain A 실험 (9개: 3 steps × 3 sensitivity)

```bash
# ============================================================
# Sensitivity = 0 (baseline, fixed dropout)
# ============================================================

# sens=0, steps=200 (GPU 0)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 200 --sensitivity 0 --gpu 0 \
    > logs/hdmap_adaptive_domainA_sens0_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=0, steps=1000 (GPU 1)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 1000 --sensitivity 0 --gpu 1 \
    > logs/hdmap_adaptive_domainA_sens0_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=0, steps=3000 (GPU 2)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 0 --gpu 2 \
    > logs/hdmap_adaptive_domainA_sens0_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ============================================================
# Sensitivity = 4 (moderate adaptive)
# ============================================================

# sens=4, steps=200 (GPU 3)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 200 --sensitivity 4 --gpu 3 \
    > logs/hdmap_adaptive_domainA_sens4_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=4, steps=1000 (GPU 4)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 1000 --sensitivity 4 --gpu 4 \
    > logs/hdmap_adaptive_domainA_sens4_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=4, steps=3000 (GPU 5)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 4 --gpu 5 \
    > logs/hdmap_adaptive_domainA_sens4_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ============================================================
# Sensitivity = 15 (strong adaptive)
# ============================================================

# sens=15, steps=200 (GPU 6)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 200 --sensitivity 15 --gpu 6 \
    > logs/hdmap_adaptive_domainA_sens15_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=15, steps=1000 (GPU 7)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 1000 --sensitivity 15 --gpu 7 \
    > logs/hdmap_adaptive_domainA_sens15_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=15, steps=3000 (GPU 8)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 15 --gpu 8 \
    > logs/hdmap_adaptive_domainA_sens15_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Domain B 실험 (9개: 3 steps × 3 sensitivity)

```bash
# ============================================================
# Sensitivity = 0 (baseline, fixed dropout)
# ============================================================

# sens=0, steps=200 (GPU 0)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 200 --sensitivity 0 --gpu 0 \
    > logs/hdmap_adaptive_domainB_sens0_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=0, steps=1000 (GPU 1)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 1000 --sensitivity 0 --gpu 1 \
    > logs/hdmap_adaptive_domainB_sens0_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=0, steps=3000 (GPU 2)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 3000 --sensitivity 0 --gpu 2 \
    > logs/hdmap_adaptive_domainB_sens0_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ============================================================
# Sensitivity = 4 (moderate adaptive)
# ============================================================

# sens=4, steps=200 (GPU 3)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 200 --sensitivity 4 --gpu 3 \
    > logs/hdmap_adaptive_domainB_sens4_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=4, steps=1000 (GPU 4)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 1000 --sensitivity 4 --gpu 4 \
    > logs/hdmap_adaptive_domainB_sens4_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=4, steps=3000 (GPU 5)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 3000 --sensitivity 4 --gpu 5 \
    > logs/hdmap_adaptive_domainB_sens4_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ============================================================
# Sensitivity = 15 (strong adaptive)
# ============================================================

# sens=15, steps=200 (GPU 6)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 200 --sensitivity 15 --gpu 6 \
    > logs/hdmap_adaptive_domainB_sens15_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=15, steps=1000 (GPU 7)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 1000 --sensitivity 15 --gpu 7 \
    > logs/hdmap_adaptive_domainB_sens15_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=15, steps=3000 (GPU 8)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 3000 --sensitivity 15 --gpu 8 \
    > logs/hdmap_adaptive_domainB_sens15_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Domain C 실험 (9개: 3 steps × 3 sensitivity)

```bash
# ============================================================
# Sensitivity = 0 (baseline, fixed dropout)
# ============================================================

# sens=0, steps=200 (GPU 0)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 200 --sensitivity 0 --gpu 0 \
    > logs/hdmap_adaptive_domainC_sens0_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=0, steps=1000 (GPU 1)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 1000 --sensitivity 0 --gpu 1 \
    > logs/hdmap_adaptive_domainC_sens0_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=0, steps=3000 (GPU 2)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 3000 --sensitivity 0 --gpu 2 \
    > logs/hdmap_adaptive_domainC_sens0_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ============================================================
# Sensitivity = 4 (moderate adaptive)
# ============================================================

# sens=4, steps=200 (GPU 3)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 200 --sensitivity 4 --gpu 3 \
    > logs/hdmap_adaptive_domainC_sens4_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=4, steps=1000 (GPU 4)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 1000 --sensitivity 4 --gpu 4 \
    > logs/hdmap_adaptive_domainC_sens4_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=4, steps=3000 (GPU 5)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 3000 --sensitivity 4 --gpu 5 \
    > logs/hdmap_adaptive_domainC_sens4_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ============================================================
# Sensitivity = 15 (strong adaptive)
# ============================================================

# sens=15, steps=200 (GPU 6)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 200 --sensitivity 15 --gpu 6 \
    > logs/hdmap_adaptive_domainC_sens15_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=15, steps=1000 (GPU 7)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 1000 --sensitivity 15 --gpu 7 \
    > logs/hdmap_adaptive_domainC_sens15_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=15, steps=3000 (GPU 8)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 3000 --sensitivity 15 --gpu 8 \
    > logs/hdmap_adaptive_domainC_sens15_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Domain D 실험 (9개: 3 steps × 3 sensitivity)

```bash
# ============================================================
# Sensitivity = 0 (baseline, fixed dropout)
# ============================================================

# sens=0, steps=200 (GPU 0)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 200 --sensitivity 0 --gpu 0 \
    > logs/hdmap_adaptive_domainD_sens0_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=0, steps=1000 (GPU 1)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 1000 --sensitivity 0 --gpu 1 \
    > logs/hdmap_adaptive_domainD_sens0_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=0, steps=3000 (GPU 2)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 3000 --sensitivity 0 --gpu 2 \
    > logs/hdmap_adaptive_domainD_sens0_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ============================================================
# Sensitivity = 4 (moderate adaptive)
# ============================================================

# sens=4, steps=200 (GPU 3)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 200 --sensitivity 4 --gpu 3 \
    > logs/hdmap_adaptive_domainD_sens4_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=4, steps=1000 (GPU 4)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 1000 --sensitivity 4 --gpu 4 \
    > logs/hdmap_adaptive_domainD_sens4_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=4, steps=3000 (GPU 5)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 3000 --sensitivity 4 --gpu 5 \
    > logs/hdmap_adaptive_domainD_sens4_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ============================================================
# Sensitivity = 15 (strong adaptive)
# ============================================================

# sens=15, steps=200 (GPU 6)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 200 --sensitivity 15 --gpu 6 \
    > logs/hdmap_adaptive_domainD_sens15_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=15, steps=1000 (GPU 7)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 1000 --sensitivity 15 --gpu 7 \
    > logs/hdmap_adaptive_domainD_sens15_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=15, steps=3000 (GPU 8)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 3000 --sensitivity 15 --gpu 8 \
    > logs/hdmap_adaptive_domainD_sens15_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 결과 분석

```bash
# 결과 분석 스크립트 실행
python results/hdmap_adaptive_validation/analyze_results.py --path results/hdmap_adaptive_validation
```

---

## 5. Fixed Dropout 비교 실험

**목적**: Adaptive dropout 대신 고정 dropout 비율의 효과 검증

현재까지 결과에서 sensitivity=0 (fixed dropout=0.3)이 가장 좋은 성능을 보임.
더 높은 dropout이 overfitting 방지에 효과적인지 검증.

### 파라미터

| dropout | 설명 |
|---------|------|
| 0.2 | 기존 Dinomaly 기본값 |
| 0.3 | DinomalyAdaptive 기본값 |
| 0.4 | 높은 dropout |
| 0.5 | 매우 높은 dropout |

### Domain A 실험 (12개: 4 dropout × 3 steps)

```bash
# ============================================================
# Dropout = 0.2 (original Dinomaly default)
# ============================================================

# dropout=0.2, steps=200 (GPU 0)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 200 --sensitivity 0 --base-dropout 0.2 --gpu 0 \
    > logs/hdmap_fixed_domainA_drop02_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.2, steps=1000 (GPU 1)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 1000 --sensitivity 0 --base-dropout 0.2 --gpu 1 \
    > logs/hdmap_fixed_domainA_drop02_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.2, steps=3000 (GPU 2)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 0 --base-dropout 0.2 --gpu 2 \
    > logs/hdmap_fixed_domainA_drop02_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ============================================================
# Dropout = 0.3 (current DinomalyAdaptive default)
# ============================================================

# dropout=0.3, steps=200 (GPU 3)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 200 --sensitivity 0 --base-dropout 0.3 --gpu 3 \
    > logs/hdmap_fixed_domainA_drop03_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.3, steps=1000 (GPU 4)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 1000 --sensitivity 0 --base-dropout 0.3 --gpu 4 \
    > logs/hdmap_fixed_domainA_drop03_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.3, steps=3000 (GPU 5)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 0 --base-dropout 0.3 --gpu 5 \
    > logs/hdmap_fixed_domainA_drop03_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ============================================================
# Dropout = 0.4 (higher dropout)
# ============================================================

# dropout=0.4, steps=200 (GPU 6)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 200 --sensitivity 0 --base-dropout 0.4 --gpu 6 \
    > logs/hdmap_fixed_domainA_drop04_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.4, steps=1000 (GPU 7)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 1000 --sensitivity 0 --base-dropout 0.4 --gpu 7 \
    > logs/hdmap_fixed_domainA_drop04_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.4, steps=3000 (GPU 8)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 0 --base-dropout 0.4 --gpu 8 \
    > logs/hdmap_fixed_domainA_drop04_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ============================================================
# Dropout = 0.5 (very high dropout)
# ============================================================

# dropout=0.5, steps=200 (GPU 9)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 200 --sensitivity 0 --base-dropout 0.5 --gpu 9 \
    > logs/hdmap_fixed_domainA_drop05_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.5, steps=1000 (GPU 10)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 1000 --sensitivity 0 --base-dropout 0.5 --gpu 10 \
    > logs/hdmap_fixed_domainA_drop05_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.5, steps=3000 (GPU 11)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 0 --base-dropout 0.5 --gpu 11 \
    > logs/hdmap_fixed_domainA_drop05_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 전체 도메인 실험 (best dropout 찾은 후)

Domain A에서 최적 dropout을 찾은 후, 전체 도메인에 적용:

```bash
# 예: dropout=0.4가 최적이라면
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain all --max-steps 3000 --sensitivity 0 --base-dropout 0.4 --gpu 0 \
    > logs/hdmap_fixed_all_drop04_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## 6. 전체 60개 실험 재현 (수정된 pred_score 기반 AUROC)

**목적**: 기존 실험 결과를 수정된 AUROC 계산(pred_score 기반)으로 재현

**수정 사항**:
- 이전: `anomaly_map.amax()` (max 값, outlier에 민감)
- 이후: `pred_score` (top-k mean, validation AUROC와 일관)

### 실험 매트릭스 요약

| 카테고리 | 실험 수 | 설명 |
|---------|---------|------|
| Fixed Dropout (sens=0) | 36개 | 4 dropout × (domain_A,B,C,D 일부) × 3 steps |
| Adaptive (sens=4, 15) | 24개 | 2 sens × 4 domains × 3 steps |
| **합계** | **60개** | |

---

### 6.1 Fixed Dropout 실험 (Sensitivity = 0)

#### Dropout = 0.2 (6개: domain_A, domain_C × 3 steps)

```bash
# dropout=0.2, domain_A, steps=200 (GPU 0)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 200 --sensitivity 0 --base-dropout 0.2 --gpu 0 \
    > logs/rerun_fixed_domainA_drop02_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.2, domain_A, steps=1000 (GPU 1)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 1000 --sensitivity 0 --base-dropout 0.2 --gpu 1 \
    > logs/rerun_fixed_domainA_drop02_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.2, domain_A, steps=3000 (GPU 2)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 0 --base-dropout 0.2 --gpu 2 \
    > logs/rerun_fixed_domainA_drop02_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.2, domain_C, steps=200 (GPU 3)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 200 --sensitivity 0 --base-dropout 0.2 --gpu 3 \
    > logs/rerun_fixed_domainC_drop02_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.2, domain_C, steps=1000 (GPU 4)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 1000 --sensitivity 0 --base-dropout 0.2 --gpu 4 \
    > logs/rerun_fixed_domainC_drop02_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.2, domain_C, steps=3000 (GPU 5)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 3000 --sensitivity 0 --base-dropout 0.2 --gpu 5 \
    > logs/rerun_fixed_domainC_drop02_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### Dropout = 0.3 (12개: 4 domains × 3 steps)

```bash
# dropout=0.3, domain_A, steps=200 (GPU 0)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 200 --sensitivity 0 --base-dropout 0.3 --gpu 0 \
    > logs/rerun_fixed_domainA_drop03_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.3, domain_A, steps=1000 (GPU 1)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 1000 --sensitivity 0 --base-dropout 0.3 --gpu 1 \
    > logs/rerun_fixed_domainA_drop03_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.3, domain_A, steps=3000 (GPU 2)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 0 --base-dropout 0.3 --gpu 2 \
    > logs/rerun_fixed_domainA_drop03_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.3, domain_B, steps=200 (GPU 3)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 200 --sensitivity 0 --base-dropout 0.3 --gpu 3 \
    > logs/rerun_fixed_domainB_drop03_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.3, domain_B, steps=1000 (GPU 4)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 1000 --sensitivity 0 --base-dropout 0.3 --gpu 4 \
    > logs/rerun_fixed_domainB_drop03_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.3, domain_B, steps=3000 (GPU 5)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 3000 --sensitivity 0 --base-dropout 0.3 --gpu 5 \
    > logs/rerun_fixed_domainB_drop03_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.3, domain_C, steps=200 (GPU 6)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 200 --sensitivity 0 --base-dropout 0.3 --gpu 6 \
    > logs/rerun_fixed_domainC_drop03_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.3, domain_C, steps=1000 (GPU 7)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 1000 --sensitivity 0 --base-dropout 0.3 --gpu 7 \
    > logs/rerun_fixed_domainC_drop03_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.3, domain_C, steps=3000 (GPU 8)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 3000 --sensitivity 0 --base-dropout 0.3 --gpu 8 \
    > logs/rerun_fixed_domainC_drop03_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.3, domain_D, steps=200 (GPU 9)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 200 --sensitivity 0 --base-dropout 0.3 --gpu 9 \
    > logs/rerun_fixed_domainD_drop03_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.3, domain_D, steps=1000 (GPU 10)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 1000 --sensitivity 0 --base-dropout 0.3 --gpu 10 \
    > logs/rerun_fixed_domainD_drop03_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.3, domain_D, steps=3000 (GPU 11)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 3000 --sensitivity 0 --base-dropout 0.3 --gpu 11 \
    > logs/rerun_fixed_domainD_drop03_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### Dropout = 0.4 (6개: domain_A, domain_C × 3 steps)

```bash
# dropout=0.4, domain_A, steps=200 (GPU 0)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 200 --sensitivity 0 --base-dropout 0.4 --gpu 0 \
    > logs/rerun_fixed_domainA_drop04_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.4, domain_A, steps=1000 (GPU 1)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 1000 --sensitivity 0 --base-dropout 0.4 --gpu 1 \
    > logs/rerun_fixed_domainA_drop04_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.4, domain_A, steps=3000 (GPU 2)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 0 --base-dropout 0.4 --gpu 2 \
    > logs/rerun_fixed_domainA_drop04_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.4, domain_C, steps=200 (GPU 3)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 200 --sensitivity 0 --base-dropout 0.4 --gpu 3 \
    > logs/rerun_fixed_domainC_drop04_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.4, domain_C, steps=1000 (GPU 4)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 1000 --sensitivity 0 --base-dropout 0.4 --gpu 4 \
    > logs/rerun_fixed_domainC_drop04_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.4, domain_C, steps=3000 (GPU 5)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 3000 --sensitivity 0 --base-dropout 0.4 --gpu 5 \
    > logs/rerun_fixed_domainC_drop04_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### Dropout = 0.5 (6개: domain_A, domain_C × 3 steps)

```bash
# dropout=0.5, domain_A, steps=200 (GPU 0)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 200 --sensitivity 0 --base-dropout 0.5 --gpu 0 \
    > logs/rerun_fixed_domainA_drop05_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.5, domain_A, steps=1000 (GPU 1)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 1000 --sensitivity 0 --base-dropout 0.5 --gpu 1 \
    > logs/rerun_fixed_domainA_drop05_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.5, domain_A, steps=3000 (GPU 2)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 0 --base-dropout 0.5 --gpu 2 \
    > logs/rerun_fixed_domainA_drop05_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.5, domain_C, steps=200 (GPU 3)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 200 --sensitivity 0 --base-dropout 0.5 --gpu 3 \
    > logs/rerun_fixed_domainC_drop05_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.5, domain_C, steps=1000 (GPU 4)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 1000 --sensitivity 0 --base-dropout 0.5 --gpu 4 \
    > logs/rerun_fixed_domainC_drop05_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dropout=0.5, domain_C, steps=3000 (GPU 5)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 3000 --sensitivity 0 --base-dropout 0.5 --gpu 5 \
    > logs/rerun_fixed_domainC_drop05_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

### 6.2 Adaptive Dropout 실험 (Sensitivity > 0)

#### Sensitivity = 4.0 (12개: 4 domains × 3 steps)

```bash
# sens=4, domain_A, steps=200 (GPU 0)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 200 --sensitivity 4 --gpu 0 \
    > logs/rerun_adaptive_domainA_sens4_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=4, domain_A, steps=1000 (GPU 1)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 1000 --sensitivity 4 --gpu 1 \
    > logs/rerun_adaptive_domainA_sens4_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=4, domain_A, steps=3000 (GPU 2)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 4 --gpu 2 \
    > logs/rerun_adaptive_domainA_sens4_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=4, domain_B, steps=200 (GPU 3)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 200 --sensitivity 4 --gpu 3 \
    > logs/rerun_adaptive_domainB_sens4_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=4, domain_B, steps=1000 (GPU 4)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 1000 --sensitivity 4 --gpu 4 \
    > logs/rerun_adaptive_domainB_sens4_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=4, domain_B, steps=3000 (GPU 5)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 3000 --sensitivity 4 --gpu 5 \
    > logs/rerun_adaptive_domainB_sens4_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=4, domain_C, steps=200 (GPU 6)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 200 --sensitivity 4 --gpu 6 \
    > logs/rerun_adaptive_domainC_sens4_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=4, domain_C, steps=1000 (GPU 7)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 1000 --sensitivity 4 --gpu 7 \
    > logs/rerun_adaptive_domainC_sens4_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=4, domain_C, steps=3000 (GPU 8)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 3000 --sensitivity 4 --gpu 8 \
    > logs/rerun_adaptive_domainC_sens4_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=4, domain_D, steps=200 (GPU 9)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 200 --sensitivity 4 --gpu 9 \
    > logs/rerun_adaptive_domainD_sens4_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=4, domain_D, steps=1000 (GPU 10)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 1000 --sensitivity 4 --gpu 10 \
    > logs/rerun_adaptive_domainD_sens4_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=4, domain_D, steps=3000 (GPU 11)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 3000 --sensitivity 4 --gpu 11 \
    > logs/rerun_adaptive_domainD_sens4_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### Sensitivity = 15.0 (12개: 4 domains × 3 steps)

```bash
# sens=15, domain_A, steps=200 (GPU 0)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 200 --sensitivity 15 --gpu 0 \
    > logs/rerun_adaptive_domainA_sens15_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=15, domain_A, steps=1000 (GPU 1)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 1000 --sensitivity 15 --gpu 1 \
    > logs/rerun_adaptive_domainA_sens15_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=15, domain_A, steps=3000 (GPU 2)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 15 --gpu 2 \
    > logs/rerun_adaptive_domainA_sens15_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=15, domain_B, steps=200 (GPU 3)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 200 --sensitivity 15 --gpu 3 \
    > logs/rerun_adaptive_domainB_sens15_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=15, domain_B, steps=1000 (GPU 4)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 1000 --sensitivity 15 --gpu 4 \
    > logs/rerun_adaptive_domainB_sens15_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=15, domain_B, steps=3000 (GPU 5)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 3000 --sensitivity 15 --gpu 5 \
    > logs/rerun_adaptive_domainB_sens15_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=15, domain_C, steps=200 (GPU 6)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 200 --sensitivity 15 --gpu 6 \
    > logs/rerun_adaptive_domainC_sens15_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=15, domain_C, steps=1000 (GPU 7)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 1000 --sensitivity 15 --gpu 7 \
    > logs/rerun_adaptive_domainC_sens15_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=15, domain_C, steps=3000 (GPU 8)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 3000 --sensitivity 15 --gpu 8 \
    > logs/rerun_adaptive_domainC_sens15_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=15, domain_D, steps=200 (GPU 9)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 200 --sensitivity 15 --gpu 9 \
    > logs/rerun_adaptive_domainD_sens15_200steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=15, domain_D, steps=1000 (GPU 10)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 1000 --sensitivity 15 --gpu 10 \
    > logs/rerun_adaptive_domainD_sens15_1000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sens=15, domain_D, steps=3000 (GPU 11)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 3000 --sensitivity 15 --gpu 11 \
    > logs/rerun_adaptive_domainD_sens15_3000steps_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

### 6.3 병렬 실행 가이드 (16 GPU 활용)

모든 실험을 최대한 빠르게 실행하기 위한 배치 스크립트:

```bash
#!/bin/bash
# rerun_all_experiments.sh
# 60개 실험을 16 GPU에서 병렬 실행

cd /mnt/ex-disk/taewan.hwang/study/anomalib
mkdir -p logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ============================================================
# Batch 1: Fixed Dropout 0.2, 0.3 (18개, GPU 0-11 사용)
# ============================================================

# Fixed 0.2 - domain_A (GPU 0-2)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 200 --sensitivity 0 --base-dropout 0.2 --gpu 0 > logs/rerun_${TIMESTAMP}_fixed_A_d02_200.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 1000 --sensitivity 0 --base-dropout 0.2 --gpu 1 > logs/rerun_${TIMESTAMP}_fixed_A_d02_1000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 0 --base-dropout 0.2 --gpu 2 > logs/rerun_${TIMESTAMP}_fixed_A_d02_3000.log 2>&1 &

# Fixed 0.2 - domain_C (GPU 3-5)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 200 --sensitivity 0 --base-dropout 0.2 --gpu 3 > logs/rerun_${TIMESTAMP}_fixed_C_d02_200.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 1000 --sensitivity 0 --base-dropout 0.2 --gpu 4 > logs/rerun_${TIMESTAMP}_fixed_C_d02_1000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 3000 --sensitivity 0 --base-dropout 0.2 --gpu 5 > logs/rerun_${TIMESTAMP}_fixed_C_d02_3000.log 2>&1 &

# Fixed 0.3 - domain_A (GPU 6-8)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 200 --sensitivity 0 --base-dropout 0.3 --gpu 6 > logs/rerun_${TIMESTAMP}_fixed_A_d03_200.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 1000 --sensitivity 0 --base-dropout 0.3 --gpu 7 > logs/rerun_${TIMESTAMP}_fixed_A_d03_1000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 0 --base-dropout 0.3 --gpu 8 > logs/rerun_${TIMESTAMP}_fixed_A_d03_3000.log 2>&1 &

# Fixed 0.3 - domain_B (GPU 9-11)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps 200 --sensitivity 0 --base-dropout 0.3 --gpu 9 > logs/rerun_${TIMESTAMP}_fixed_B_d03_200.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps 1000 --sensitivity 0 --base-dropout 0.3 --gpu 10 > logs/rerun_${TIMESTAMP}_fixed_B_d03_1000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps 3000 --sensitivity 0 --base-dropout 0.3 --gpu 11 > logs/rerun_${TIMESTAMP}_fixed_B_d03_3000.log 2>&1 &

# Fixed 0.3 - domain_C (GPU 12-14)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 200 --sensitivity 0 --base-dropout 0.3 --gpu 12 > logs/rerun_${TIMESTAMP}_fixed_C_d03_200.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 1000 --sensitivity 0 --base-dropout 0.3 --gpu 13 > logs/rerun_${TIMESTAMP}_fixed_C_d03_1000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 3000 --sensitivity 0 --base-dropout 0.3 --gpu 14 > logs/rerun_${TIMESTAMP}_fixed_C_d03_3000.log 2>&1 &

# Fixed 0.3 - domain_D (GPU 15 + wait)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps 200 --sensitivity 0 --base-dropout 0.3 --gpu 15 > logs/rerun_${TIMESTAMP}_fixed_D_d03_200.log 2>&1 &

echo "Batch 1 started (18 experiments). Wait for completion..."
wait

# ============================================================
# Batch 2: Fixed 0.3 (domain_D 나머지) + Fixed 0.4, 0.5 (18개)
# ============================================================

# Fixed 0.3 - domain_D (나머지, GPU 0-1)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps 1000 --sensitivity 0 --base-dropout 0.3 --gpu 0 > logs/rerun_${TIMESTAMP}_fixed_D_d03_1000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps 3000 --sensitivity 0 --base-dropout 0.3 --gpu 1 > logs/rerun_${TIMESTAMP}_fixed_D_d03_3000.log 2>&1 &

# Fixed 0.4 - domain_A (GPU 2-4)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 200 --sensitivity 0 --base-dropout 0.4 --gpu 2 > logs/rerun_${TIMESTAMP}_fixed_A_d04_200.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 1000 --sensitivity 0 --base-dropout 0.4 --gpu 3 > logs/rerun_${TIMESTAMP}_fixed_A_d04_1000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 0 --base-dropout 0.4 --gpu 4 > logs/rerun_${TIMESTAMP}_fixed_A_d04_3000.log 2>&1 &

# Fixed 0.4 - domain_C (GPU 5-7)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 200 --sensitivity 0 --base-dropout 0.4 --gpu 5 > logs/rerun_${TIMESTAMP}_fixed_C_d04_200.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 1000 --sensitivity 0 --base-dropout 0.4 --gpu 6 > logs/rerun_${TIMESTAMP}_fixed_C_d04_1000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 3000 --sensitivity 0 --base-dropout 0.4 --gpu 7 > logs/rerun_${TIMESTAMP}_fixed_C_d04_3000.log 2>&1 &

# Fixed 0.5 - domain_A (GPU 8-10)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 200 --sensitivity 0 --base-dropout 0.5 --gpu 8 > logs/rerun_${TIMESTAMP}_fixed_A_d05_200.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 1000 --sensitivity 0 --base-dropout 0.5 --gpu 9 > logs/rerun_${TIMESTAMP}_fixed_A_d05_1000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 0 --base-dropout 0.5 --gpu 10 > logs/rerun_${TIMESTAMP}_fixed_A_d05_3000.log 2>&1 &

# Fixed 0.5 - domain_C (GPU 11-13)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 200 --sensitivity 0 --base-dropout 0.5 --gpu 11 > logs/rerun_${TIMESTAMP}_fixed_C_d05_200.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 1000 --sensitivity 0 --base-dropout 0.5 --gpu 12 > logs/rerun_${TIMESTAMP}_fixed_C_d05_1000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 3000 --sensitivity 0 --base-dropout 0.5 --gpu 13 > logs/rerun_${TIMESTAMP}_fixed_C_d05_3000.log 2>&1 &

echo "Batch 2 started (14 experiments). Wait for completion..."
wait

# ============================================================
# Batch 3: Adaptive Sensitivity 4.0 (12개)
# ============================================================

nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 200 --sensitivity 4 --gpu 0 > logs/rerun_${TIMESTAMP}_adapt_A_s4_200.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 1000 --sensitivity 4 --gpu 1 > logs/rerun_${TIMESTAMP}_adapt_A_s4_1000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 4 --gpu 2 > logs/rerun_${TIMESTAMP}_adapt_A_s4_3000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps 200 --sensitivity 4 --gpu 3 > logs/rerun_${TIMESTAMP}_adapt_B_s4_200.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps 1000 --sensitivity 4 --gpu 4 > logs/rerun_${TIMESTAMP}_adapt_B_s4_1000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps 3000 --sensitivity 4 --gpu 5 > logs/rerun_${TIMESTAMP}_adapt_B_s4_3000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 200 --sensitivity 4 --gpu 6 > logs/rerun_${TIMESTAMP}_adapt_C_s4_200.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 1000 --sensitivity 4 --gpu 7 > logs/rerun_${TIMESTAMP}_adapt_C_s4_1000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 3000 --sensitivity 4 --gpu 8 > logs/rerun_${TIMESTAMP}_adapt_C_s4_3000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps 200 --sensitivity 4 --gpu 9 > logs/rerun_${TIMESTAMP}_adapt_D_s4_200.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps 1000 --sensitivity 4 --gpu 10 > logs/rerun_${TIMESTAMP}_adapt_D_s4_1000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps 3000 --sensitivity 4 --gpu 11 > logs/rerun_${TIMESTAMP}_adapt_D_s4_3000.log 2>&1 &

echo "Batch 3 started (12 experiments). Wait for completion..."
wait

# ============================================================
# Batch 4: Adaptive Sensitivity 15.0 (12개)
# ============================================================

nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 200 --sensitivity 15 --gpu 0 > logs/rerun_${TIMESTAMP}_adapt_A_s15_200.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 1000 --sensitivity 15 --gpu 1 > logs/rerun_${TIMESTAMP}_adapt_A_s15_1000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 15 --gpu 2 > logs/rerun_${TIMESTAMP}_adapt_A_s15_3000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps 200 --sensitivity 15 --gpu 3 > logs/rerun_${TIMESTAMP}_adapt_B_s15_200.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps 1000 --sensitivity 15 --gpu 4 > logs/rerun_${TIMESTAMP}_adapt_B_s15_1000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps 3000 --sensitivity 15 --gpu 5 > logs/rerun_${TIMESTAMP}_adapt_B_s15_3000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 200 --sensitivity 15 --gpu 6 > logs/rerun_${TIMESTAMP}_adapt_C_s15_200.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 1000 --sensitivity 15 --gpu 7 > logs/rerun_${TIMESTAMP}_adapt_C_s15_1000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 3000 --sensitivity 15 --gpu 8 > logs/rerun_${TIMESTAMP}_adapt_C_s15_3000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps 200 --sensitivity 15 --gpu 9 > logs/rerun_${TIMESTAMP}_adapt_D_s15_200.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps 1000 --sensitivity 15 --gpu 10 > logs/rerun_${TIMESTAMP}_adapt_D_s15_1000.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps 3000 --sensitivity 15 --gpu 11 > logs/rerun_${TIMESTAMP}_adapt_D_s15_3000.log 2>&1 &

echo "Batch 4 started (12 experiments). Wait for completion..."
wait

echo "All 60 experiments completed!"
echo "Analyze results: python results/hdmap_adaptive_validation/analyze_results.py"
```

스크립트 저장 및 실행:
```bash
# 스크립트 저장
chmod +x scripts/rerun_all_experiments.sh

# 실행
./scripts/rerun_all_experiments.sh
```

---

### 6.4 결과 분석

```bash
# 결과 분석
python results/hdmap_adaptive_validation/analyze_results.py --path results/hdmap_adaptive_validation

# 상세 통계 포함
python results/hdmap_adaptive_validation/analyze_results.py --path results/hdmap_adaptive_validation --detailed
```

---

## 7. 통합 실험 (Validation History 기반)

**목적**: 동일 실험 조건에 대해 step별 별도 실험 대신, 최대 step으로 1회 실행하고 중간 validation AUROC를 모두 기록

**변경 사항**:
- 이전: 60개 실험 (20 조건 × 3 steps)
- 이후: 20개 실험 (20 조건 × 1 max_step, 중간 val history 포함)

**JSON 결과 형식**:
```json
{
  "model_type": "dinomaly_adaptive",
  "domain": "domain_A",
  "max_steps": 3000,
  "adaptive_settings": {
    "sensitivity": 4.0,
    "normal_entropy": 0.545,
    "base_dropout": 0.3
  },
  "validation_history": [
    {"step": 100, "val_image_AUROC": 0.5426},
    {"step": 200, "val_image_AUROC": 0.7522},
    {"step": 300, "val_image_AUROC": 0.8923},
    ...
    {"step": 3000, "val_image_AUROC": 0.9872}
  ],
  "test_image_AUROC": 0.9836
}
```

### 실험 매트릭스 (20개 통합)

| 카테고리 | 실험 수 | 설명 |
|---------|---------|------|
| Fixed Dropout (sens=0) | 12개 | dropout 0.2,0.3,0.4,0.5 × 3 domains (A,C + B,D for 0.3) |
| Adaptive (sens=4, 15) | 8개 | 2 sens × 4 domains |
| **합계** | **20개** | 최대 3000 steps, val history 포함 |

---

### 7.1 Fixed Dropout 실험 (12개)

```bash
# ============================================================
# Fixed Dropout Experiments (sensitivity = 0)
# Run once with max_steps=3000, validation history tracked automatically
# ============================================================

# Dropout 0.2 (domain_A, domain_C)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 0 --base-dropout 0.2 --gpu 0 \
    > logs/consolidated_fixed_A_d02_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 3000 --sensitivity 0 --base-dropout 0.2 --gpu 1 \
    > logs/consolidated_fixed_C_d02_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Dropout 0.3 (all 4 domains)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 0 --base-dropout 0.3 --gpu 2 \
    > logs/consolidated_fixed_A_d03_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 3000 --sensitivity 0 --base-dropout 0.3 --gpu 3 \
    > logs/consolidated_fixed_B_d03_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 3000 --sensitivity 0 --base-dropout 0.3 --gpu 4 \
    > logs/consolidated_fixed_C_d03_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 3000 --sensitivity 0 --base-dropout 0.3 --gpu 5 \
    > logs/consolidated_fixed_D_d03_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Dropout 0.4 (domain_A, domain_C)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 0 --base-dropout 0.4 --gpu 6 \
    > logs/consolidated_fixed_A_d04_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 3000 --sensitivity 0 --base-dropout 0.4 --gpu 7 \
    > logs/consolidated_fixed_C_d04_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Dropout 0.5 (domain_A, domain_C)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 0 --base-dropout 0.5 --gpu 8 \
    > logs/consolidated_fixed_A_d05_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 3000 --sensitivity 0 --base-dropout 0.5 --gpu 9 \
    > logs/consolidated_fixed_C_d05_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

### 7.2 Adaptive Dropout 실험 (8개)

```bash
# ============================================================
# Adaptive Dropout Experiments (sensitivity > 0)
# Domain-specific normal_entropy applied automatically
# ============================================================

# Sensitivity = 4.0 (all 4 domains)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 4 --gpu 0 \
    > logs/consolidated_adaptive_A_sens4_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 3000 --sensitivity 4 --gpu 1 \
    > logs/consolidated_adaptive_B_sens4_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 3000 --sensitivity 4 --gpu 2 \
    > logs/consolidated_adaptive_C_sens4_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 3000 --sensitivity 4 --gpu 3 \
    > logs/consolidated_adaptive_D_sens4_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Sensitivity = 15.0 (all 4 domains)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 3000 --sensitivity 15 --gpu 4 \
    > logs/consolidated_adaptive_A_sens15_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 3000 --sensitivity 15 --gpu 5 \
    > logs/consolidated_adaptive_B_sens15_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 3000 --sensitivity 15 --gpu 6 \
    > logs/consolidated_adaptive_C_sens15_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 3000 --sensitivity 15 --gpu 7 \
    > logs/consolidated_adaptive_D_sens15_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

### 7.3 전체 20개 실험 병렬 실행

```bash
#!/bin/bash
# run_consolidated_experiments.sh
# 20개 통합 실험 - 8 GPU에서 병렬 실행

cd /mnt/ex-disk/taewan.hwang/study/anomalib
mkdir -p logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAX_STEPS=3000

echo "Starting 20 consolidated experiments at ${TIMESTAMP}"

# ============================================================
# Batch 1: Fixed Dropout 0.2, 0.3 (10개, GPU 0-7 사용)
# ============================================================

# Fixed 0.2 (2개)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps $MAX_STEPS --sensitivity 0 --base-dropout 0.2 --gpu 0 > logs/cons_${TIMESTAMP}_fixed_A_d02.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps $MAX_STEPS --sensitivity 0 --base-dropout 0.2 --gpu 1 > logs/cons_${TIMESTAMP}_fixed_C_d02.log 2>&1 &

# Fixed 0.3 (4개)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps $MAX_STEPS --sensitivity 0 --base-dropout 0.3 --gpu 2 > logs/cons_${TIMESTAMP}_fixed_A_d03.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps $MAX_STEPS --sensitivity 0 --base-dropout 0.3 --gpu 3 > logs/cons_${TIMESTAMP}_fixed_B_d03.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps $MAX_STEPS --sensitivity 0 --base-dropout 0.3 --gpu 4 > logs/cons_${TIMESTAMP}_fixed_C_d03.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps $MAX_STEPS --sensitivity 0 --base-dropout 0.3 --gpu 5 > logs/cons_${TIMESTAMP}_fixed_D_d03.log 2>&1 &

# Fixed 0.4 (2개)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps $MAX_STEPS --sensitivity 0 --base-dropout 0.4 --gpu 6 > logs/cons_${TIMESTAMP}_fixed_A_d04.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps $MAX_STEPS --sensitivity 0 --base-dropout 0.4 --gpu 7 > logs/cons_${TIMESTAMP}_fixed_C_d04.log 2>&1 &

echo "Batch 1 (8 experiments) started. Waiting for completion..."
wait

# ============================================================
# Batch 2: Fixed 0.5 + Adaptive (12개)
# ============================================================

# Fixed 0.5 (2개)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps $MAX_STEPS --sensitivity 0 --base-dropout 0.5 --gpu 0 > logs/cons_${TIMESTAMP}_fixed_A_d05.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps $MAX_STEPS --sensitivity 0 --base-dropout 0.5 --gpu 1 > logs/cons_${TIMESTAMP}_fixed_C_d05.log 2>&1 &

# Adaptive sens=4 (4개)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps $MAX_STEPS --sensitivity 4 --gpu 2 > logs/cons_${TIMESTAMP}_adapt_A_s4.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps $MAX_STEPS --sensitivity 4 --gpu 3 > logs/cons_${TIMESTAMP}_adapt_B_s4.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps $MAX_STEPS --sensitivity 4 --gpu 4 > logs/cons_${TIMESTAMP}_adapt_C_s4.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps $MAX_STEPS --sensitivity 4 --gpu 5 > logs/cons_${TIMESTAMP}_adapt_D_s4.log 2>&1 &

# Adaptive sens=15 (first 2)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps $MAX_STEPS --sensitivity 15 --gpu 6 > logs/cons_${TIMESTAMP}_adapt_A_s15.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps $MAX_STEPS --sensitivity 15 --gpu 7 > logs/cons_${TIMESTAMP}_adapt_B_s15.log 2>&1 &

echo "Batch 2 (8 experiments) started. Waiting for completion..."
wait

# ============================================================
# Batch 3: Remaining Adaptive sens=15 (2개)
# ============================================================

nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps $MAX_STEPS --sensitivity 15 --gpu 0 > logs/cons_${TIMESTAMP}_adapt_C_s15.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps $MAX_STEPS --sensitivity 15 --gpu 1 > logs/cons_${TIMESTAMP}_adapt_D_s15.log 2>&1 &

echo "Batch 3 (2 experiments) started. Waiting for completion..."
wait

echo "All 20 experiments completed!"
echo "Analyze results with:"
echo "  python results/hdmap_adaptive_validation/analyze_results.py --results-dir results/hdmap_adaptive_validation"
```

---

### 7.4 결과 분석 (개선된 버전)

```bash
# 기본 분석 (validation history + test AUROC 테이블)
python results/hdmap_adaptive_validation/analyze_results.py \
    --results-dir results/hdmap_adaptive_validation

# 특정 step에서의 validation AUROC 확인
python results/hdmap_adaptive_validation/analyze_results.py \
    --results-dir results/hdmap_adaptive_validation \
    --steps 200,500,1000,2000,3000

# 도메인별 필터링
python results/hdmap_adaptive_validation/analyze_results.py \
    --results-dir results/hdmap_adaptive_validation \
    --domain domain_A

# 상세 validation history 출력
python results/hdmap_adaptive_validation/analyze_results.py \
    --results-dir results/hdmap_adaptive_validation \
    --detailed

# 도메인별 요약 통계
python results/hdmap_adaptive_validation/analyze_results.py \
    --results-dir results/hdmap_adaptive_validation \
    --summary
```

**출력 예시:**
```
==================================================
HDMAP Adaptive Dropout Validation Results
==================================================
Domain     | Sens | Drop | Val@200 | Val@500 | Val@1000 | Val@2000 | Val@3000 | Test AUROC | Best Val
----------------------------------------------------------------------------------------------------------
domain_A   |  0.0 | 0.20 |  75.22% |  91.34% |   97.06% |   98.12% |   98.72% |     98.36% |   98.72%
domain_A   |  0.0 | 0.30 |  72.15% |  89.87% |   96.45% |   97.89% |   98.45% |     98.21% |   98.45%
domain_A   |  4.0 | 0.30 |  74.56% |  90.23% |   97.12% |   98.34% |   98.89% |     98.67% |   98.89%
...
```

---

## 8. Phase 3: APE 기반 Full Comparison 실험 (OE → APE 교체 후)

**목표**: APE(Angular Power Entropy) 기반 adaptive dropout의 장기 학습(15000 steps)에서 overfitting 방지 효과 검증

**배경**:
- Phase 2 Sanity Check 결과: Fixed=98.78%, APE-Adaptive=98.69% (1000 steps, 거의 동일)
- 1000 steps에서는 overfitting이 아직 발생하지 않아 차이 미미
- 15000 steps 장기 학습에서 APE-adaptive dropout의 진가 발휘 예상 (overnight)

**Domain별 Normal APE 값** (EDA 결과):
| Domain | Normal APE Mean | 특성 |
|--------|-----------------|------|
| domain_A | 0.777 | 중간 방향성 |
| domain_B | 0.713 | 강한 방향성 (overfit 위험 높음) |
| domain_C | 0.866 | 약한 방향성 (isotropic) |
| domain_D | 0.816 | 중간-약 방향성 |

### 8.1 실험 조건

| 조건 | Sensitivity | Dropout | 설명 |
|------|-------------|---------|------|
| Fixed | 0 | 0.3 (고정) | Baseline: 항상 dropout=0.3 |
| APE-Adaptive (mild) | 4 | 0.1~0.6 (적응) | APE↓ → Dropout↑ (완만한 반응) |
| APE-Adaptive (strong) | 15 | 0.1~0.6 (적응) | APE↓ → Dropout↑ (강한 반응) |

**총 12개 실험**: 4 domains × 3 conditions

### 8.2 개별 실험 명령어

#### Fixed Dropout (Sensitivity=0, 4개)

```bash
# Fixed, domain_A, 15000 steps (GPU 0)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 15000 \
    --sensitivity 0 --base-dropout 0.3 --gpu 0 \
    > logs/ape_phase3_fixed_A_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Fixed, domain_B, 15000 steps (GPU 1)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 15000 \
    --sensitivity 0 --base-dropout 0.3 --gpu 1 \
    > logs/ape_phase3_fixed_B_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Fixed, domain_C, 15000 steps (GPU 2)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 15000 \
    --sensitivity 0 --base-dropout 0.3 --gpu 2 \
    > logs/ape_phase3_fixed_C_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Fixed, domain_D, 15000 steps (GPU 3)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 15000 \
    --sensitivity 0 --base-dropout 0.3 --gpu 3 \
    > logs/ape_phase3_fixed_D_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### APE-Adaptive Dropout (Sensitivity=4, 4개)

```bash
# APE-Adaptive sens=4, domain_A, 15000 steps (GPU 4) - normal_ape=0.777 자동 적용
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 15000 \
    --sensitivity 4 --base-dropout 0.3 --gpu 4 \
    > logs/ape_phase3_adaptive_s4_A_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# APE-Adaptive sens=4, domain_B, 15000 steps (GPU 5) - normal_ape=0.713 자동 적용
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 15000 \
    --sensitivity 4 --base-dropout 0.3 --gpu 5 \
    > logs/ape_phase3_adaptive_s4_B_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# APE-Adaptive sens=4, domain_C, 15000 steps (GPU 6) - normal_ape=0.866 자동 적용
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 15000 \
    --sensitivity 4 --base-dropout 0.3 --gpu 6 \
    > logs/ape_phase3_adaptive_s4_C_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# APE-Adaptive sens=4, domain_D, 15000 steps (GPU 7) - normal_ape=0.816 자동 적용
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 15000 \
    --sensitivity 4 --base-dropout 0.3 --gpu 7 \
    > logs/ape_phase3_adaptive_s4_D_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### APE-Adaptive Dropout (Sensitivity=15, 4개)

```bash
# APE-Adaptive sens=15, domain_A, 15000 steps - normal_ape=0.777 자동 적용
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 15000 \
    --sensitivity 15 --base-dropout 0.3 --gpu 0 \
    > logs/ape_phase3_adaptive_s15_A_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# APE-Adaptive sens=15, domain_B, 15000 steps - normal_ape=0.713 자동 적용
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 15000 \
    --sensitivity 15 --base-dropout 0.3 --gpu 1 \
    > logs/ape_phase3_adaptive_s15_B_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# APE-Adaptive sens=15, domain_C, 15000 steps - normal_ape=0.866 자동 적용
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 15000 \
    --sensitivity 15 --base-dropout 0.3 --gpu 2 \
    > logs/ape_phase3_adaptive_s15_C_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# APE-Adaptive sens=15, domain_D, 15000 steps - normal_ape=0.816 자동 적용
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 15000 \
    --sensitivity 15 --base-dropout 0.3 --gpu 3 \
    > logs/ape_phase3_adaptive_s15_D_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 8.3 일괄 실행 스크립트 (12개 실험, GPU 병렬)

```bash
#!/bin/bash
# Phase 3: APE-based Full Comparison (12 experiments, ~10 hours overnight)
# 4 Fixed + 4 sens=4 + 4 sens=15

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

echo "Phase 3: APE Full Comparison Started at $TIMESTAMP"
echo "Running 12 experiments (4 domains × 3 conditions) in parallel..."
echo "15000 steps each - overnight run"

# Fixed Dropout (sens=0) - GPU 0~3
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 15000 --sensitivity 0 --base-dropout 0.3 --gpu 0 > logs/ape_phase3_${TIMESTAMP}_fixed_A.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps 15000 --sensitivity 0 --base-dropout 0.3 --gpu 1 > logs/ape_phase3_${TIMESTAMP}_fixed_B.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 15000 --sensitivity 0 --base-dropout 0.3 --gpu 2 > logs/ape_phase3_${TIMESTAMP}_fixed_C.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps 15000 --sensitivity 0 --base-dropout 0.3 --gpu 3 > logs/ape_phase3_${TIMESTAMP}_fixed_D.log 2>&1 &

# APE-Adaptive sens=4 - GPU 4~7
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 15000 --sensitivity 4 --base-dropout 0.3 --gpu 4 > logs/ape_phase3_${TIMESTAMP}_s4_A.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps 15000 --sensitivity 4 --base-dropout 0.3 --gpu 5 > logs/ape_phase3_${TIMESTAMP}_s4_B.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 15000 --sensitivity 4 --base-dropout 0.3 --gpu 6 > logs/ape_phase3_${TIMESTAMP}_s4_C.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps 15000 --sensitivity 4 --base-dropout 0.3 --gpu 7 > logs/ape_phase3_${TIMESTAMP}_s4_D.log 2>&1 &

# APE-Adaptive sens=15 - GPU 8~11 (또는 0~3 재사용 시 wait 후 실행)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 15000 --sensitivity 15 --base-dropout 0.3 --gpu 8 > logs/ape_phase3_${TIMESTAMP}_s15_A.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps 15000 --sensitivity 15 --base-dropout 0.3 --gpu 9 > logs/ape_phase3_${TIMESTAMP}_s15_B.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 15000 --sensitivity 15 --base-dropout 0.3 --gpu 10 > logs/ape_phase3_${TIMESTAMP}_s15_C.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps 15000 --sensitivity 15 --base-dropout 0.3 --gpu 11 > logs/ape_phase3_${TIMESTAMP}_s15_D.log 2>&1 &

echo "All 12 experiments launched!"
echo "Monitor with: tail -f logs/ape_phase3_${TIMESTAMP}_*.log"
echo "Check GPU usage: watch nvidia-smi"

wait
echo "Phase 3 Complete!"
```

### 8.4 순차 실행 (GPU 2개 사용)

```bash
#!/bin/bash
# Phase 3: Sequential execution with 2 GPUs (12 experiments)
# 각 batch에서 2개씩 병렬 실행

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

echo "Phase 3 Sequential: Started at $TIMESTAMP"
echo "12 experiments (4 domains × 3 conditions), 15000 steps each"

# Batch 1: domain_A (Fixed + sens=4)
echo "Batch 1/6: domain_A Fixed + sens=4..."
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 15000 --sensitivity 0 --base-dropout 0.3 --gpu 0 > logs/ape_phase3_${TIMESTAMP}_fixed_A.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 15000 --sensitivity 4 --base-dropout 0.3 --gpu 1 > logs/ape_phase3_${TIMESTAMP}_s4_A.log 2>&1 &
wait

# Batch 2: domain_A sens=15 + domain_B Fixed
echo "Batch 2/6: domain_A sens=15 + domain_B Fixed..."
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 15000 --sensitivity 15 --base-dropout 0.3 --gpu 0 > logs/ape_phase3_${TIMESTAMP}_s15_A.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps 15000 --sensitivity 0 --base-dropout 0.3 --gpu 1 > logs/ape_phase3_${TIMESTAMP}_fixed_B.log 2>&1 &
wait

# Batch 3: domain_B (sens=4 + sens=15)
echo "Batch 3/6: domain_B sens=4 + sens=15..."
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps 15000 --sensitivity 4 --base-dropout 0.3 --gpu 0 > logs/ape_phase3_${TIMESTAMP}_s4_B.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps 15000 --sensitivity 15 --base-dropout 0.3 --gpu 1 > logs/ape_phase3_${TIMESTAMP}_s15_B.log 2>&1 &
wait

# Batch 4: domain_C (Fixed + sens=4)
echo "Batch 4/6: domain_C Fixed + sens=4..."
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 15000 --sensitivity 0 --base-dropout 0.3 --gpu 0 > logs/ape_phase3_${TIMESTAMP}_fixed_C.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 15000 --sensitivity 4 --base-dropout 0.3 --gpu 1 > logs/ape_phase3_${TIMESTAMP}_s4_C.log 2>&1 &
wait

# Batch 5: domain_C sens=15 + domain_D Fixed
echo "Batch 5/6: domain_C sens=15 + domain_D Fixed..."
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 15000 --sensitivity 15 --base-dropout 0.3 --gpu 0 > logs/ape_phase3_${TIMESTAMP}_s15_C.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps 15000 --sensitivity 0 --base-dropout 0.3 --gpu 1 > logs/ape_phase3_${TIMESTAMP}_fixed_D.log 2>&1 &
wait

# Batch 6: domain_D (sens=4 + sens=15)
echo "Batch 6/6: domain_D sens=4 + sens=15..."
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps 15000 --sensitivity 4 --base-dropout 0.3 --gpu 0 > logs/ape_phase3_${TIMESTAMP}_s4_D.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps 15000 --sensitivity 15 --base-dropout 0.3 --gpu 1 > logs/ape_phase3_${TIMESTAMP}_s15_D.log 2>&1 &
wait

echo "Phase 3 Complete! All 12 experiments finished."
```

### 8.5 결과 분석

```bash
# 전체 결과 분석
python results/hdmap_adaptive_validation/analyze_results.py \
    --results-dir results/hdmap_adaptive_validation \
    --auto-steps

# Fixed vs Adaptive 비교 (domain별) - 장기 학습 추이
python results/hdmap_adaptive_validation/analyze_results.py \
    --results-dir results/hdmap_adaptive_validation \
    --steps 1000,3000,5000,10000,15000 \
    --summary
```

### 8.6 예상 결과 및 분석 포인트

**검증 항목**:
1. **Test AUROC**: Fixed vs sens=4 vs sens=15 최종 성능 비교
2. **Val AUROC 추이**: @1000 → @5000 → @10000 → @15000 overfitting 곡선 분석
3. **Sensitivity 효과**: sens=4 (완만) vs sens=15 (강함) 비교
4. **domain_B 특수성**: 가장 낮은 normal_ape(0.713) → 가장 강한 적응 효과 기대
5. **APE/Dropout 통계**: 학습 중 dropout 분포 분석

**성공 기준**:
- APE-Adaptive(sens=4 or 15)가 15000 steps에서 Fixed 대비 overfitting 감소
- 특히 domain_B(방향성 강함)에서 효과 두드러짐
- Val AUROC peak 이후 하락 폭: Fixed > sens=4 ≥ sens=15
- sens=15가 너무 aggressive하면 underfitting 가능 → 적정 sensitivity 탐색
