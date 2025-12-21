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
# Fixed, domain_A, 30000 steps (GPU 0)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 30000 \
    --sensitivity 0 --base-dropout 0.3 --gpu 8 \
    > logs/ape_phase3_fixed_A_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Fixed, domain_B, 30000 steps (GPU 1)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 30000 \
    --sensitivity 0 --base-dropout 0.3 --gpu 9 \
    > logs/ape_phase3_fixed_B_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Fixed, domain_C, 30000 steps (GPU 2)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 30000 \
    --sensitivity 0 --base-dropout 0.3 --gpu 10 \
    > logs/ape_phase3_fixed_C_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Fixed, domain_D, 30000 steps (GPU 3)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 30000 \
    --sensitivity 0 --base-dropout 0.3 --gpu 11 \
    > logs/ape_phase3_fixed_D_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### APE-Adaptive Dropout (Sensitivity=4, 4개)

```bash
# APE-Adaptive sens=4, domain_A, 30000 steps (GPU 4) - normal_ape=0.777 자동 적용
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 30000 \
    --sensitivity 4 --base-dropout 0.3 --gpu 12 \
    > logs/ape_phase3_adaptive_s4_A_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# APE-Adaptive sens=4, domain_B, 30000 steps (GPU 5) - normal_ape=0.713 자동 적용
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 30000 \
    --sensitivity 4 --base-dropout 0.3 --gpu 13 \
    > logs/ape_phase3_adaptive_s4_B_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# APE-Adaptive sens=4, domain_C, 30000 steps (GPU 6) - normal_ape=0.866 자동 적용
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 30000 \
    --sensitivity 4 --base-dropout 0.3 --gpu 14 \
    > logs/ape_phase3_adaptive_s4_C_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# APE-Adaptive sens=4, domain_D, 30000 steps (GPU 7) - normal_ape=0.816 자동 적용
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 30000 \
    --sensitivity 4 --base-dropout 0.3 --gpu 15 \
    > logs/ape_phase3_adaptive_s4_D_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### APE-Adaptive Dropout (Sensitivity=15, 4개)

```bash
# APE-Adaptive sens=15, domain_A, 30000 steps - normal_ape=0.777 자동 적용
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_A --max-steps 30000 \
    --sensitivity 15 --base-dropout 0.3 --gpu 8 \
    > logs/ape_phase3_adaptive_s15_A_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# APE-Adaptive sens=15, domain_B, 30000 steps - normal_ape=0.713 자동 적용
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_B --max-steps 30000 \
    --sensitivity 15 --base-dropout 0.3 --gpu 9 \
    > logs/ape_phase3_adaptive_s15_B_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# APE-Adaptive sens=15, domain_C, 30000 steps - normal_ape=0.866 자동 적용
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 30000 \
    --sensitivity 15 --base-dropout 0.3 --gpu 10 \
    > logs/ape_phase3_adaptive_s15_C_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# APE-Adaptive sens=15, domain_D, 30000 steps - normal_ape=0.816 자동 적용
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_D --max-steps 30000 \
    --sensitivity 15 --base-dropout 0.3 --gpu 11 \
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
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 30000 --sensitivity 0 --base-dropout 0.3 --gpu 0 > logs/ape_phase3_${TIMESTAMP}_fixed_A.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps 30000 --sensitivity 0 --base-dropout 0.3 --gpu 1 > logs/ape_phase3_${TIMESTAMP}_fixed_B.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 30000 --sensitivity 0 --base-dropout 0.3 --gpu 2 > logs/ape_phase3_${TIMESTAMP}_fixed_C.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps 30000 --sensitivity 0 --base-dropout 0.3 --gpu 3 > logs/ape_phase3_${TIMESTAMP}_fixed_D.log 2>&1 &

# APE-Adaptive sens=4 - GPU 4~7
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 30000 --sensitivity 4 --base-dropout 0.3 --gpu 4 > logs/ape_phase3_${TIMESTAMP}_s4_A.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps 30000 --sensitivity 4 --base-dropout 0.3 --gpu 5 > logs/ape_phase3_${TIMESTAMP}_s4_B.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 30000 --sensitivity 4 --base-dropout 0.3 --gpu 6 > logs/ape_phase3_${TIMESTAMP}_s4_C.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps 30000 --sensitivity 4 --base-dropout 0.3 --gpu 7 > logs/ape_phase3_${TIMESTAMP}_s4_D.log 2>&1 &

# APE-Adaptive sens=15 - GPU 8~11 (또는 0~3 재사용 시 wait 후 실행)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 30000 --sensitivity 15 --base-dropout 0.3 --gpu 8 > logs/ape_phase3_${TIMESTAMP}_s15_A.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps 30000 --sensitivity 15 --base-dropout 0.3 --gpu 9 > logs/ape_phase3_${TIMESTAMP}_s15_B.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 30000 --sensitivity 15 --base-dropout 0.3 --gpu 10 > logs/ape_phase3_${TIMESTAMP}_s15_C.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps 30000 --sensitivity 15 --base-dropout 0.3 --gpu 11 > logs/ape_phase3_${TIMESTAMP}_s15_D.log 2>&1 &

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

---

## 9. Ablation A: Fixed Dropout Sweep

**목표**: APE-Adaptive에서 관측된 dropout_mean 근처 값들로 Fixed Dropout 실험하여, 최적 fixed dropout으로도 APE-Adaptive를 넘을 수 없음을 검증

**배경**:
- Phase 3 결과: domain_C에서 Fixed(0.3)=85.64%, APE-Adaptive(sens=15)=91.56%
- APE-Adaptive(sens=15)의 dropout_mean ≈ 0.38 관측
- "단순히 dropout 값이 높아서 좋은 것 아닌가?" 반론에 대한 검증

**실험 조건**:
| Dropout | 설명 |
|---------|------|
| 0.25 | 기존보다 낮음 |
| 0.30 | 기존 baseline (완료) |
| 0.35 | APE-Adaptive mean 근처 |
| 0.40 | APE-Adaptive mean 초과 |

### 9.1 개별 실험 명령어 (domain_C 기준)

```bash
# Fixed p=0.25, domain_C, 15000 steps
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 15000 \
    --sensitivity 0 --base-dropout 0.25 --gpu 0 \
    > logs/ablation_fixed_C_d025_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Fixed p=0.35, domain_C, 15000 steps
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 15000 \
    --sensitivity 0 --base-dropout 0.35 --gpu 1 \
    > logs/ablation_fixed_C_d035_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Fixed p=0.40, domain_C, 15000 steps
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 15000 \
    --sensitivity 0 --base-dropout 0.40 --gpu 2 \
    > logs/ablation_fixed_C_d040_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 9.2 전체 도메인 Fixed Sweep (4 domains × 3 dropout values)

```bash
#!/bin/bash
# Ablation A: Fixed Dropout Sweep (12 experiments)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

echo "Ablation A: Fixed Dropout Sweep Started at $TIMESTAMP"

# p=0.25 (4 domains)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 30000 --sensitivity 0 --base-dropout 0.25 --gpu 0 > logs/ablation_${TIMESTAMP}_fixed_A_d025.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps 30000 --sensitivity 0 --base-dropout 0.25 --gpu 1 > logs/ablation_${TIMESTAMP}_fixed_B_d025.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 30000 --sensitivity 0 --base-dropout 0.25 --gpu 2 > logs/ablation_${TIMESTAMP}_fixed_C_d025.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps 30000 --sensitivity 0 --base-dropout 0.25 --gpu 3 > logs/ablation_${TIMESTAMP}_fixed_D_d025.log 2>&1 &

# p=0.35 (4 domains)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 30000 --sensitivity 0 --base-dropout 0.35 --gpu 4 > logs/ablation_${TIMESTAMP}_fixed_A_d035.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps 30000 --sensitivity 0 --base-dropout 0.35 --gpu 5 > logs/ablation_${TIMESTAMP}_fixed_B_d035.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 30000 --sensitivity 0 --base-dropout 0.35 --gpu 6 > logs/ablation_${TIMESTAMP}_fixed_C_d035.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps 30000 --sensitivity 0 --base-dropout 0.35 --gpu 7 > logs/ablation_${TIMESTAMP}_fixed_D_d035.log 2>&1 &

# p=0.40 (4 domains)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_A --max-steps 30000 --sensitivity 0 --base-dropout 0.40 --gpu 8 > logs/ablation_${TIMESTAMP}_fixed_A_d040.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_B --max-steps 30000 --sensitivity 0 --base-dropout 0.40 --gpu 9 > logs/ablation_${TIMESTAMP}_fixed_B_d040.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 30000 --sensitivity 0 --base-dropout 0.40 --gpu 10 > logs/ablation_${TIMESTAMP}_fixed_C_d040.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_D --max-steps 30000 --sensitivity 0 --base-dropout 0.40 --gpu 11 > logs/ablation_${TIMESTAMP}_fixed_D_d040.log 2>&1 &

echo "All 12 experiments launched!"
wait
echo "Ablation A Complete!"
```

### 9.3 domain_C만 빠르게 테스트 (GPU 3개)

```bash
#!/bin/bash
# Quick Ablation A: domain_C only (3 experiments)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

echo "Quick Ablation A (domain_C): Started at $TIMESTAMP"

nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 15000 --sensitivity 0 --base-dropout 0.25 --gpu 0 > logs/ablation_${TIMESTAMP}_fixed_C_d025.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 15000 --sensitivity 0 --base-dropout 0.35 --gpu 1 > logs/ablation_${TIMESTAMP}_fixed_C_d035.log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --max-steps 15000 --sensitivity 0 --base-dropout 0.40 --gpu 2 > logs/ablation_${TIMESTAMP}_fixed_C_d040.log 2>&1 &

wait
echo "Quick Ablation A Complete!"
```

### 9.4 예상 결과

| Dropout | 예상 Test AUROC | vs APE-Adaptive(91.56%) |
|---------|-----------------|-------------------------|
| 0.25 | ~83-85% | ❌ 낮음 (underfitting) |
| 0.30 | 85.64% (실측) | ❌ 낮음 |
| 0.35 | ~87-89% | ❌ 여전히 낮음 |
| 0.40 | ~85-88% | ❌ 낮음 (과한 regularization) |

**성공 기준**: 어떤 fixed dropout도 APE-Adaptive(sens=15)의 91.56%를 넘지 못함
→ "APE 기반 적응적 조절"의 가치 입증

---

## 10. MVTec-AD Dinomaly 논문 원본 재현 실험 (Ablation Mode)

**목표**: Dinomaly 논문의 Unified Information-Control Dropout을 MVTec-AD에서 검증

**Unified Dropout Formula**:
```
dropout_p = p_min + (p_max - p_min) * p_time * p_struct
```
- `p_time = min(1.0, global_step / t_warmup)` - 시간 기반 curriculum
- `p_struct = sigmoid(α * (μ_normal - APE))` - 샘플 기반 조절

**Ablation Mode (sensitivity=0)**:
- `p_struct = 1.0` (모든 샘플에 동일 적용)
- 결과: `dropout = p_time * p_max` → 0%에서 90%로 1000 steps 동안 증가
- 이것이 **Dinomaly 논문 원본** 구현

### 10.1 대표 카테고리 선정

| 카테고리 | 유형 | 특성 |
|----------|------|------|
| bottle | Object | 투명/반사 표면, 구조적 결함 |
| cable | Object | 복잡한 구조, 다양한 결함 유형 |
| carpet | Texture | 반복 패턴, 미세 결함 |
| leather | Texture | 불규칙 텍스처, 표면 결함 |
| transistor | Object | 전자부품, 미세 구조 |

### 10.2 MVTec Ablation 실험 명령어 (5개, 10k steps)

```bash
# MVTec bottle - Ablation mode (논문 원본: 0%→90% progressive dropout)
nohup python examples/notebooks/dinomaly_mvtec_validation.py \
    --categories bottle \
    --dropout-sensitivity 0 \
    --p-max 0.9 --t-warmup 1000 \
    --max-steps 5000 --gpu 0 \
    > logs/mvtec_ablation_bottle_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# MVTec cable - Ablation mode
nohup python examples/notebooks/dinomaly_mvtec_validation.py \
    --categories cable \
    --dropout-sensitivity 0 \
    --p-max 0.9 --t-warmup 1000 \
    --max-steps 5000 --gpu 1 \
    > logs/mvtec_ablation_cable_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# MVTec carpet - Ablation mode
nohup python examples/notebooks/dinomaly_mvtec_validation.py \
    --categories carpet \
    --dropout-sensitivity 0 \
    --p-max 0.9 --t-warmup 1000 \
    --max-steps 5000 --gpu 2 \
    > logs/mvtec_ablation_carpet_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# MVTec leather - Ablation mode
nohup python examples/notebooks/dinomaly_mvtec_validation.py \
    --categories leather \
    --dropout-sensitivity 0 \
    --p-max 0.9 --t-warmup 1000 \
    --max-steps 5000 --gpu 3 \
    > logs/mvtec_ablation_leather_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# MVTec transistor - Ablation mode
nohup python examples/notebooks/dinomaly_mvtec_validation.py \
    --categories transistor \
    --dropout-sensitivity 0 \
    --p-max 0.9 --t-warmup 1000 \
    --max-steps 5000 --gpu 4 \
    > logs/mvtec_ablation_transistor_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## 11. HDMAP Dinomaly Ablation Study (Domain C)

**목표**: Dinomaly dropout 스케줄 ablation study (Domain C 기준)

**Unified Dropout Formula**:
```
dropout_p = p_min + (p_max - p_min) * p_time * p_struct

p_time = min(1.0, global_step / t_warmup)  # [0, 1] time-based
p_struct = sigmoid(sensitivity * (normal_ape - APE))  # [0, 1] sample-based
```

**공통 설정**: Domain C, max-steps=5000, p-min=0

---

### 11.1 Baseline 1: Original Dinomaly (p-max=0.2, t-warmup=1000)

Dinomaly 논문 기본 설정 (sensitivity=0 → p_struct=1.0)

```bash
# Baseline 1: Original Dinomaly settings
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --dropout-sensitivity 0 \
    --p-max 0.2 --t-warmup 1000 \
    --max-steps 5000 --gpu 0 \
    > logs/hdmap_b1_pmax02_twarm1000_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

### 11.2 Baseline 2-1: p-max Variation (t-warmup=1000 fixed)

p-max를 0.1, 0.3, 0.4로 변경하여 upper bound 민감도 테스트

```bash
# p-max = 0.1
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --dropout-sensitivity 0 \
    --p-max 0.1 --t-warmup 1000 \
    --max-steps 5000 --gpu 1 \
    > logs/hdmap_b2_pmax01_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# p-max = 0.3
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --dropout-sensitivity 0 \
    --p-max 0.3 --t-warmup 1000 \
    --max-steps 5000 --gpu 2 \
    > logs/hdmap_b2_pmax03_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# p-max = 0.4
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --dropout-sensitivity 0 \
    --p-max 0.4 --t-warmup 1000 \
    --max-steps 5000 --gpu 3 \
    > logs/hdmap_b2_pmax04_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

### 11.3 Baseline 2-2: t-warmup Variation (p-max=0.2 fixed)

t-warmup을 500, 2000, 3000으로 변경하여 warmup 속도 민감도 테스트

```bash
# t-warmup = 500 (빠른 warmup)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --dropout-sensitivity 0 \
    --p-max 0.2 --t-warmup 500 \
    --max-steps 5000 --gpu 4 \
    > logs/hdmap_b2_twarm500_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# t-warmup = 2000 (느린 warmup)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --dropout-sensitivity 0 \
    --p-max 0.2 --t-warmup 2000 \
    --max-steps 5000 --gpu 5 \
    > logs/hdmap_b2_twarm2000_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# t-warmup = 3000 (매우 느린 warmup)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --dropout-sensitivity 0 \
    --p-max 0.2 --t-warmup 3000 \
    --max-steps 5000 --gpu 6 \
    > logs/hdmap_b2_twarm3000_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

### 11.4 Baseline 3: No Warmup (t-warmup=0)

warmup 없이 처음부터 full dropout 적용

```bash
# No warmup (instant full dropout)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --dropout-sensitivity 0 \
    --p-max 0.2 --t-warmup 0 \
    --max-steps 5000 --gpu 7 \
    > logs/hdmap_b3_nowarmup_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

### 11.5 Proposed: Adaptive Dropout (APE-based)

sensitivity > 0으로 APE 기반 sample-adaptive dropout 활성화
- Domain C normal_ape = 0.866
- sensitivity=15 (strong), sensitivity=30 (very strong)
- p-max=0.2, 0.4, 0.6 (p_struct < 1.0이므로 실제 dropout이 작음 → 넓은 범위 테스트)

```bash
# sensitivity=15, p-max=0.2
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --dropout-sensitivity 15 \
    --p-max 0.2 --t-warmup 1000 \
    --max-steps 5000 --gpu 8 \
    > logs/hdmap_proposed_s15_pmax02_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sensitivity=15, p-max=0.4
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --dropout-sensitivity 15 \
    --p-max 0.4 --t-warmup 1000 \
    --max-steps 5000 --gpu 9 \
    > logs/hdmap_proposed_s15_pmax04_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sensitivity=15, p-max=0.6
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --dropout-sensitivity 15 \
    --p-max 0.6 --t-warmup 1000 \
    --max-steps 5000 --gpu 10 \
    > logs/hdmap_proposed_s15_pmax06_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sensitivity=30, p-max=0.2
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --dropout-sensitivity 30 \
    --p-max 0.2 --t-warmup 1000 \
    --max-steps 5000 --gpu 11 \
    > logs/hdmap_proposed_s30_pmax02_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sensitivity=30, p-max=0.4
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --dropout-sensitivity 30 \
    --p-max 0.4 --t-warmup 1000 \
    --max-steps 5000 --gpu 12 \
    > logs/hdmap_proposed_s30_pmax04_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# sensitivity=30, p-max=0.6
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --dropout-sensitivity 30 \
    --p-max 0.6 --t-warmup 1000 \
    --max-steps 5000 --gpu 13 \
    > logs/hdmap_proposed_s30_pmax06_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```
---

## Section 12: Quick Sanity Check (동작 확인용)

APE-Adaptive Discarding 구현 후 동작 확인을 위한 짧은 스텝 학습.
100 steps로 빠르게 돌려서 에러 없이 학습되는지 확인.

**새 CLI 파라미터:**
- `--discard-sensitivity` (기존 `--dropout-sensitivity` 대체)
- `--k-max` (기존 `--p-max` 대체)
- `--k-min` (기존 `--p-min` 대체)
- `--ema-alpha` (EMA 평활화, 새로 추가)
- `--bottleneck-dropout` (고정 dropout, 새로 추가)

```bash
# Baseline 동작 확인 (sensitivity=0)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --discard-sensitivity 0 \
    --k-max 0.9 --t-warmup 200 \
    --max-steps 500 --gpu 0 \
    > logs/sanity_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Proposed 동작 확인 (sensitivity=15)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --discard-sensitivity 15 \
    --k-max 0.9 --t-warmup 200 \
    --max-steps 500 --gpu 1 \
    > logs/sanity_proposed_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# EMA 활성화 동작 확인
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --discard-sensitivity 15 \
    --k-max 0.9 --t-warmup 200 \
    --ema-alpha 0.1 \
    --max-steps 500 --gpu 2 \
    > logs/sanity_ema_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## Section 13: APE-Adaptive Discarding 1차 실험 (Domain C)

**논문 정정 사항:**
- Dinomaly 논문의 0%→90% warmup은 **Dropout이 아닌 Discarding rate (k%)**
- Dropout: 고정 0.2 (Real-IAD는 0.4)
- Discarding rate (Eq.5의 k%): 0%→90% warmup over 1000 steps

**APE-Adaptive Discarding 공식:**
```
k_discard = k_min + (k_max - k_min) * k_time * k_struct

Where:
- k_time = min(1.0, global_step / t_warmup)  # Time-based warmup
- k_struct = sigmoid(sensitivity * (normal_ape - APE))  # APE-based factor
```

**실험 목표:** Domain C에서 APE-adaptive discarding의 효과 검증

**총 14개 실험 (16 GPU 중 14개 사용)**

---

### 13.1 Baseline 실험 (7개)

#### B0: Original Dinomaly (논문 기준)
```bash
# B0: sensitivity=0, k_max=0.9, t_warmup=1000 (GPU 0)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --discard-sensitivity 0 \
    --k-max 0.9 --t-warmup 1000 \
    --max-steps 5000 --gpu 0 \
    > logs/exp13_B0_original_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### B1: No Hard-mining (효과 분리)
```bash
# B1: sensitivity=0, k_max=0.0 (hard-mining 비활성화) (GPU 1)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --discard-sensitivity 0 \
    --k-max 0.0 --t-warmup 1000 \
    --max-steps 5000 --gpu 1 \
    > logs/exp13_B1_no_hardmining_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### B2: k_max sweep (버리는 강도 민감도)
```bash
# B2-1: k_max=0.6 (GPU 2)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --discard-sensitivity 0 \
    --k-max 0.6 --t-warmup 1000 \
    --max-steps 5000 --gpu 2 \
    > logs/exp13_B2_kmax060_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# B2-2: k_max=0.75 (GPU 3)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --discard-sensitivity 0 \
    --k-max 0.75 --t-warmup 1000 \
    --max-steps 5000 --gpu 3 \
    > logs/exp13_B2_kmax075_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### B3: t_warmup sweep (초반 학습 안정성)
```bash
# B3-1: t_warmup=0 (No warmup) (GPU 4)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --discard-sensitivity 0 \
    --k-max 0.9 --t-warmup 0 \
    --max-steps 5000 --gpu 4 \
    > logs/exp13_B3_twarm0_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# B3-2: t_warmup=500 (빠른 warmup) (GPU 5)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --discard-sensitivity 0 \
    --k-max 0.9 --t-warmup 500 \
    --max-steps 5000 --gpu 5 \
    > logs/exp13_B3_twarm500_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# B3-3: t_warmup=2000 (느린 warmup) (GPU 6)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --discard-sensitivity 0 \
    --k-max 0.9 --t-warmup 2000 \
    --max-steps 5000 --gpu 6 \
    > logs/exp13_B3_twarm2000_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

### 13.2 Proposed 실험 (7개)

#### P0: Proposed default (APE-adaptive)
```bash
# P0: sensitivity=15, k_max=0.9, t_warmup=1000 (GPU 7)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --discard-sensitivity 15 \
    --k-max 0.9 --t-warmup 1000 \
    --max-steps 5000 --gpu 7 \
    > logs/exp13_P0_proposed_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### P1: sensitivity sweep (APE 민감도)
```bash
# P1-1: sensitivity=4 (약한 adaptation) (GPU 8)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --discard-sensitivity 4 \
    --k-max 0.9 --t-warmup 1000 \
    --max-steps 5000 --gpu 8 \
    > logs/exp13_P1_sens04_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# P1-2: sensitivity=30 (강한 adaptation) (GPU 9)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --discard-sensitivity 30 \
    --k-max 0.9 --t-warmup 1000 \
    --max-steps 5000 --gpu 9 \
    > logs/exp13_P1_sens30_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### P2: k_max sweep (Proposed에서의 k_max 민감도)
```bash
# P2-1: sensitivity=15, k_max=0.6 (GPU 10)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --discard-sensitivity 15 \
    --k-max 0.6 --t-warmup 1000 \
    --max-steps 5000 --gpu 10 \
    > logs/exp13_P2_kmax060_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# P2-2: sensitivity=15, k_max=0.75 (GPU 11)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --discard-sensitivity 15 \
    --k-max 0.75 --t-warmup 1000 \
    --max-steps 5000 --gpu 11 \
    > logs/exp13_P2_kmax075_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### P3: EMA smoothing (배치 변동 안정화)
```bash
# P3: sensitivity=15, ema_alpha=0.1 (GPU 12)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --discard-sensitivity 15 \
    --k-max 0.9 --t-warmup 1000 \
    --ema-alpha 0.1 \
    --max-steps 5000 --gpu 12 \
    > logs/exp13_P3_ema01_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# P3-2: sensitivity=15, ema_alpha=0.05 (더 강한 smoothing) (GPU 13)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C \
    --discard-sensitivity 15 \
    --k-max 0.9 --t-warmup 1000 \
    --ema-alpha 0.05 \
    --max-steps 5000 --gpu 13 \
    > logs/exp13_P3_ema005_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

### 13.3 실험 요약 테이블

| ID | GPU | sensitivity | k_max | t_warmup | ema_alpha | 목적 |
|----|-----|-------------|-------|----------|-----------|------|
| **Baseline** |
| B0 | 0 | 0 | 0.9 | 1000 | 0 | Original Dinomaly (기준선) |
| B1 | 1 | 0 | 0.0 | 1000 | 0 | No hard-mining (효과 분리) |
| B2-1 | 2 | 0 | 0.6 | 1000 | 0 | k_max sweep |
| B2-2 | 3 | 0 | 0.75 | 1000 | 0 | k_max sweep |
| B3-1 | 4 | 0 | 0.9 | 0 | 0 | No warmup |
| B3-2 | 5 | 0 | 0.9 | 500 | 0 | Fast warmup |
| B3-3 | 6 | 0 | 0.9 | 2000 | 0 | Slow warmup |
| **Proposed** |
| P0 | 7 | 15 | 0.9 | 1000 | 0 | APE-adaptive default |
| P1-1 | 8 | 4 | 0.9 | 1000 | 0 | Weak adaptation |
| P1-2 | 9 | 30 | 0.9 | 1000 | 0 | Strong adaptation |
| P2-1 | 10 | 15 | 0.6 | 1000 | 0 | k_max sweep (proposed) |
| P2-2 | 11 | 15 | 0.75 | 1000 | 0 | k_max sweep (proposed) |
| P3-1 | 12 | 15 | 0.9 | 1000 | 0.1 | EMA smoothing |
| P3-2 | 13 | 15 | 0.9 | 1000 | 0.05 | Strong EMA smoothing |

---

### 13.4 All-in-one 실행 스크립트

모든 실험을 한 번에 실행:

```bash
# logs 디렉토리 생성
mkdir -p logs

# ===== Baseline 실험 =====
# B0: Original
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --discard-sensitivity 0 --k-max 0.9 --t-warmup 1000 --max-steps 5000 --gpu 0 > logs/exp13_B0_original_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# B1: No hard-mining
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --discard-sensitivity 0 --k-max 0.0 --t-warmup 1000 --max-steps 5000 --gpu 1 > logs/exp13_B1_no_hardmining_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# B2: k_max sweep
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --discard-sensitivity 0 --k-max 0.6 --t-warmup 1000 --max-steps 5000 --gpu 2 > logs/exp13_B2_kmax060_$(date +%Y%m%d_%H%M%S).log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --discard-sensitivity 0 --k-max 0.75 --t-warmup 1000 --max-steps 5000 --gpu 3 > logs/exp13_B2_kmax075_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# B3: t_warmup sweep
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --discard-sensitivity 0 --k-max 0.9 --t-warmup 0 --max-steps 5000 --gpu 4 > logs/exp13_B3_twarm0_$(date +%Y%m%d_%H%M%S).log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --discard-sensitivity 0 --k-max 0.9 --t-warmup 500 --max-steps 5000 --gpu 5 > logs/exp13_B3_twarm500_$(date +%Y%m%d_%H%M%S).log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --discard-sensitivity 0 --k-max 0.9 --t-warmup 2000 --max-steps 5000 --gpu 6 > logs/exp13_B3_twarm2000_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== Proposed 실험 =====
# P0: Default
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --discard-sensitivity 15 --k-max 0.9 --t-warmup 1000 --max-steps 5000 --gpu 7 > logs/exp13_P0_proposed_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# P1: sensitivity sweep
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --discard-sensitivity 4 --k-max 0.9 --t-warmup 1000 --max-steps 5000 --gpu 8 > logs/exp13_P1_sens04_$(date +%Y%m%d_%H%M%S).log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --discard-sensitivity 30 --k-max 0.9 --t-warmup 1000 --max-steps 5000 --gpu 9 > logs/exp13_P1_sens30_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# P2: k_max sweep (proposed)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --discard-sensitivity 15 --k-max 0.6 --t-warmup 1000 --max-steps 5000 --gpu 10 > logs/exp13_P2_kmax060_$(date +%Y%m%d_%H%M%S).log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --discard-sensitivity 15 --k-max 0.75 --t-warmup 1000 --max-steps 5000 --gpu 11 > logs/exp13_P2_kmax075_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# P3: EMA smoothing
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --discard-sensitivity 15 --k-max 0.9 --t-warmup 1000 --ema-alpha 0.1 --max-steps 5000 --gpu 12 > logs/exp13_P3_ema01_$(date +%Y%m%d_%H%M%S).log 2>&1 &
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --domain domain_C --discard-sensitivity 15 --k-max 0.9 --t-warmup 1000 --ema-alpha 0.05 --max-steps 5000 --gpu 13 > logs/exp13_P3_ema005_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "All 14 experiments started. Check logs/ for progress."
```

---

### 13.5 결과 분석

실험 완료 후 결과 분석:

```bash
# 결과 수집
python results/hdmap_adaptive_validation/analyze_results.py --auto-steps --summary

# 특정 실험 그룹만 보기
python results/hdmap_adaptive_validation/analyze_results.py --domain domain_C --detailed
```

**핵심 비교 포인트:**

1. **B0 vs B1**: Hard-mining 자체의 효과 (B1이 훨씬 나쁘면 hard-mining 필수)
2. **B0 vs B2**: k_max 민감도 (0.6~0.9 범위에서 최적값 탐색)
3. **B0 vs B3**: Warmup 민감도 (너무 빠르거나 느리면 성능 저하?)
4. **B0 vs P0**: APE-adaptive의 기본 효과 (P0이 더 좋으면 효과 있음)
5. **P0 vs P1**: Sensitivity 최적값 (Domain C에서 4/15/30 중 어떤 게 좋은지)
6. **P0 vs P2**: APE-adaptive가 k_max 의존성을 줄이는지 (robustness)
7. **P0 vs P3**: EMA smoothing이 안정성을 개선하는지

---

### 13.6 예상 결과 및 해석

| 시나리오 | 예상 | 해석 |
|----------|------|------|
| B0 ≈ P0 | APE 효과 미미 | Domain C의 APE 분포가 분리가 약해서 효과 제한적 |
| P0 > B0 | APE 효과 있음 | APE-adaptive가 학습 초점 조절에 도움 |
| P1(sens=4) > P0(sens=15) | 낮은 sensitivity가 좋음 | Domain C에서 과도한 적응은 noise |
| P3 > P0 | EMA가 중요 | 배치별 변동을 줄이는 게 안정성에 도움 |
| B2(k_max=0.6) > B0 | 낮은 k_max가 좋음 | Domain C에서 너무 강한 hard-mining은 역효과 |

---

## 14. Structure Feature 테스트 (APE vs OE)

### 14.1 개요

Pluggable structure feature 아키텍처 테스트를 위한 실험 세트.
- **APE (Angular Power Entropy)**: Frequency-domain, GPU-batch optimized
- **OE (Orientational Entropy)**: Spatial-domain, gradient-based

### 14.2 Quick Test (max-steps 1000)

10개 GPU를 활용한 빠른 검증 테스트:

| ID | GPU | Feature | Sensitivity | Domain | 목적 |
|----|-----|---------|-------------|--------|------|
| T0 | 0 | ape | 0 | domain_C | APE Baseline (no adaptive) |
| T1 | 1 | ape | 15 | domain_C | APE Proposed |
| T2 | 2 | oe | 0 | domain_C | OE Baseline (no adaptive) |
| T3 | 3 | oe | 15 | domain_C | OE Proposed |
| T4 | 4 | ape | 15 | domain_A | APE on Domain A |
| T5 | 5 | ape | 15 | domain_B | APE on Domain B |
| T6 | 6 | ape | 15 | domain_D | APE on Domain D |
| T7 | 7 | oe | 15 | domain_A | OE on Domain A |
| T8 | 8 | oe | 15 | domain_B | OE on Domain B |
| T9 | 9 | oe | 15 | domain_D | OE on Domain D |

### 14.3 실행 명령어

```bash
# logs 디렉토리 생성
mkdir -p logs

# T0: APE Baseline (Domain C)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature ape --discard-sensitivity 0 --domain domain_C --max-steps 1000 --gpu 0 > logs/exp14_T0_ape_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# T1: APE Proposed (Domain C)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature ape --discard-sensitivity 15 --domain domain_C --max-steps 1000 --gpu 1 > logs/exp14_T1_ape_proposed_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# T2: OE Baseline (Domain C)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature oe --discard-sensitivity 0 --domain domain_C --max-steps 1000 --gpu 2 > logs/exp14_T2_oe_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# T3: OE Proposed (Domain C)
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature oe --discard-sensitivity 15 --domain domain_C --max-steps 1000 --gpu 3 > logs/exp14_T3_oe_proposed_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# T4: APE on Domain A
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature ape --discard-sensitivity 15 --domain domain_A --max-steps 1000 --gpu 4 > logs/exp14_T4_ape_domainA_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# T5: APE on Domain B
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature ape --discard-sensitivity 15 --domain domain_B --max-steps 1000 --gpu 5 > logs/exp14_T5_ape_domainB_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# T6: APE on Domain D
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature ape --discard-sensitivity 15 --domain domain_D --max-steps 1000 --gpu 6 > logs/exp14_T6_ape_domainD_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# T7: OE on Domain A
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature oe --discard-sensitivity 15 --domain domain_A --max-steps 1000 --gpu 7 > logs/exp14_T7_oe_domainA_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# T8: OE on Domain B
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature oe --discard-sensitivity 15 --domain domain_B --max-steps 1000 --gpu 8 > logs/exp14_T8_oe_domainB_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# T9: OE on Domain D
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature oe --discard-sensitivity 15 --domain domain_D --max-steps 1000 --gpu 9 > logs/exp14_T9_oe_domainD_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "All 10 structure feature tests started. Check logs/ for progress."
```

### 14.4 TensorBoard 로깅 확인

APE와 OE는 동적으로 다른 메트릭 이름을 사용:

```
# APE 사용 시
train/ape_mean, train/ape_p10, train/ape_p50, train/ape_p90

# OE 사용 시
train/oe_mean, train/oe_p10, train/oe_p50, train/oe_p90

# 공통
train/k_discard, train/k_time, train/k_struct, train/dropout
```

### 14.5 분석 포인트

1. **T0 vs T1**: APE-adaptive 효과 (Domain C)
2. **T2 vs T3**: OE-adaptive 효과 (Domain C)
3. **T1 vs T3**: APE vs OE 비교 (어느 feature가 더 효과적인가?)
4. **T4-T6 vs T7-T9**: 도메인별 APE vs OE 비교

### 14.6 주의사항

- OE의 normal_value는 현재 0.5 (placeholder)로 설정됨
- 정확한 OE 실험을 위해서는 EDA로 도메인별 OE 참조값 측정 필요
- `examples/hdmap/EDA/` 디렉토리의 OE 분석 스크립트 활용

---

## 15. OE Feature Domain C 집중 실험

### 15.1 개요

OE (Orientational Entropy) 기반 adaptive discarding의 Domain C 성능 검증.
- OE normal_value는 EDA 미완료로 0.5 (placeholder) 사용
- 다양한 sensitivity 및 normal_value sweep으로 최적값 탐색

### 15.2 실험 설계 (16 GPUs)

| ID | GPU | Feature | Sens | normal_value | kMax | 목적 |
|----|-----|---------|------|--------------|------|------|
| **Baseline 비교군** |
| E0 | 0 | ape | 0 | 0.866 | 0.9 | APE Baseline (기준선) |
| E1 | 1 | ape | 15 | 0.866 | 0.9 | APE Proposed (기준선) |
| E2 | 2 | oe | 0 | - | 0.9 | OE Baseline (no adaptive) |
| **OE Sensitivity Sweep** |
| E3 | 3 | oe | 5 | 0.5 | 0.9 | Weak adaptation |
| E4 | 4 | oe | 10 | 0.5 | 0.9 | Moderate adaptation |
| E5 | 5 | oe | 15 | 0.5 | 0.9 | Default adaptation |
| E6 | 6 | oe | 20 | 0.5 | 0.9 | Strong adaptation |
| E7 | 7 | oe | 30 | 0.5 | 0.9 | Very strong adaptation |
| **OE normal_value Sweep** (sens=15 고정) |
| E8 | 8 | oe | 15 | 0.3 | 0.9 | Low reference |
| E9 | 9 | oe | 15 | 0.4 | 0.9 | Low-mid reference |
| E10 | 10 | oe | 15 | 0.6 | 0.9 | Mid-high reference |
| E11 | 11 | oe | 15 | 0.7 | 0.9 | High reference |
| E12 | 12 | oe | 15 | 0.8 | 0.9 | Very high reference |
| **OE + EMA Smoothing** |
| E13 | 13 | oe | 15 | 0.5 | 0.9 | EMA=0.1 |
| E14 | 14 | oe | 15 | 0.5 | 0.9 | EMA=0.05 |
| **OE + k_max Sweep** |
| E15 | 15 | oe | 15 | 0.5 | 0.7 | Lower k_max |

### 15.3 실행 명령어 (5000 steps)

```bash
# logs 디렉토리 생성
mkdir -p logs

# ===== Baseline 비교군 =====
# E0: APE Baseline
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature ape --discard-sensitivity 0 --domain domain_C --max-steps 5000 --gpu 0 > logs/exp15_E0_ape_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# E1: APE Proposed
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature ape --discard-sensitivity 15 --domain domain_C --max-steps 5000 --gpu 1 > logs/exp15_E1_ape_proposed_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# E2: OE Baseline
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature oe --discard-sensitivity 0 --domain domain_C --max-steps 5000 --gpu 2 > logs/exp15_E2_oe_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== OE Sensitivity Sweep =====
# E3: sens=5
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature oe --discard-sensitivity 5 --normal-value 0.5 --domain domain_C --max-steps 5000 --gpu 3 > logs/exp15_E3_oe_sens05_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# E4: sens=10
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature oe --discard-sensitivity 10 --normal-value 0.5 --domain domain_C --max-steps 5000 --gpu 4 > logs/exp15_E4_oe_sens10_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# E5: sens=15
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature oe --discard-sensitivity 15 --normal-value 0.5 --domain domain_C --max-steps 5000 --gpu 5 > logs/exp15_E5_oe_sens15_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# E6: sens=20
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature oe --discard-sensitivity 20 --normal-value 0.5 --domain domain_C --max-steps 5000 --gpu 6 > logs/exp15_E6_oe_sens20_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# E7: sens=30
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature oe --discard-sensitivity 30 --normal-value 0.5 --domain domain_C --max-steps 5000 --gpu 7 > logs/exp15_E7_oe_sens30_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== OE normal_value Sweep =====
# E8: normal_value=0.3
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature oe --discard-sensitivity 15 --normal-value 0.3 --domain domain_C --max-steps 5000 --gpu 8 > logs/exp15_E8_oe_nv03_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# E9: normal_value=0.4
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature oe --discard-sensitivity 15 --normal-value 0.4 --domain domain_C --max-steps 5000 --gpu 9 > logs/exp15_E9_oe_nv04_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# E10: normal_value=0.6
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature oe --discard-sensitivity 15 --normal-value 0.6 --domain domain_C --max-steps 5000 --gpu 10 > logs/exp15_E10_oe_nv06_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# E11: normal_value=0.7
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature oe --discard-sensitivity 15 --normal-value 0.7 --domain domain_C --max-steps 5000 --gpu 11 > logs/exp15_E11_oe_nv07_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# E12: normal_value=0.8
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature oe --discard-sensitivity 15 --normal-value 0.8 --domain domain_C --max-steps 5000 --gpu 12 > logs/exp15_E12_oe_nv08_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== OE + EMA Smoothing =====
# E13: EMA=0.1
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature oe --discard-sensitivity 15 --normal-value 0.5 --ema-alpha 0.1 --domain domain_C --max-steps 5000 --gpu 13 > logs/exp15_E13_oe_ema01_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# E14: EMA=0.05
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature oe --discard-sensitivity 15 --normal-value 0.5 --ema-alpha 0.05 --domain domain_C --max-steps 5000 --gpu 14 > logs/exp15_E14_oe_ema005_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== OE + k_max Sweep =====
# E15: k_max=0.7
nohup python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature oe --discard-sensitivity 15 --normal-value 0.5 --k-max 0.7 --domain domain_C --max-steps 5000 --gpu 15 > logs/exp15_E15_oe_kmax07_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "All 16 OE experiments started. Check logs/ for progress."
```

### 15.4 분석 포인트

1. **E0 vs E2**: APE Baseline vs OE Baseline (feature 차이만)
2. **E1 vs E5**: APE Proposed vs OE Proposed (sens=15 동일 조건)
3. **E3~E7**: OE sensitivity 최적값 탐색 (5/10/15/20/30)
4. **E8~E12**: OE normal_value 최적값 탐색 (0.3~0.8)
5. **E13~E14**: EMA smoothing 효과
6. **E15**: k_max 낮추면 OE 성능 개선되는지

### 15.5 TensorBoard 확인사항

```bash
tensorboard --logdir results/hdmap_adaptive_validation --port 6006
```

주요 메트릭:
- `train/oe_mean`: OE 평균값 (normal sample 기준 측정용)
- `train/oe_p10`, `train/oe_p90`: OE 분포 확인
- `train/k_discard`: 실제 적용된 discarding rate
- `train/k_struct`: feature-based modulation factor

### 15.6 Quick Test (1000 steps, 선택적)

전체 실험 전 빠른 검증:

```bash
# E5만 1000 step으로 빠른 테스트
python examples/notebooks/hdmap_adaptive_validation.py --model dinomaly_adaptive --structure-feature oe --discard-sensitivity 15 --normal-value 0.5 --domain domain_C --max-steps 1000 --gpu 0
```

### 15.7 결과 분석

```bash
cd results/hdmap_adaptive_validation
python analyze_results.py --auto-steps --summary --domain domain_C
```

---

## 16. OE + TIFF Dataset 실험 (TIFF 원본 기반)

### 16.1 배경

TIFF → PNG 변환 시 per-image min-max normalization이 적용되어 절대 amplitude 정보가 손실됨.
TIFF EDA 결과, OE (Orientational Entropy)가 Domain C에서 Cohen's d = 2.88로 최고 separability를 보임.

**TIFF EDA 결과 (train_good mean):**
- domain_A: OE = 0.9355, Cohen's d = 4.27
- domain_B: OE = 0.9149, Cohen's d = 10.42
- domain_C: OE = 0.9731, Cohen's d = 2.88
- domain_D: OE = 0.9520, Cohen's d = 4.87

### 16.2 새로운 CLI 옵션

```
--data-root {png,tiff}   # 데이터셋 타입 선택 (기본값: png)
--structure-feature oe   # OE feature 사용
--normal-value <float>   # TIFF EDA 결과 기반 정상값 (도메인별 자동 적용)
```

### 16.3 Sanity Check

```bash
# TIFF + OE 기본 작동 확인 (100 steps)
python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive \
  --domain domain_C \
  --max-steps 100 \
  --gpu 0 \
  --data-root tiff \
  --structure-feature oe \
  --discard-sensitivity 1.0

# 결과 예시:
# - Using TIFF dataset: .../datasets/HDMAP/1000_tiff_minmax
# - Using domain-specific OE for domain_C: 0.9731
# - OE: mean=0.8352 | k_discard: mean=0.0481
```

### 16.4 Full Experiment (5000 steps, 4 Domains)

```bash
mkdir -p logs

# ===== Domain A =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_A --max-steps 5000 \
  --gpu 0 --data-root tiff --structure-feature oe --discard-sensitivity 0 \
  > logs/tiff_oe_A_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_A --max-steps 5000 \
  --gpu 1 --data-root tiff --structure-feature oe --discard-sensitivity 15 \
  > logs/tiff_oe_A_proposed_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== Domain B =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_B --max-steps 5000 \
  --gpu 2 --data-root tiff --structure-feature oe --discard-sensitivity 0 \
  > logs/tiff_oe_B_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_B --max-steps 5000 \
  --gpu 3 --data-root tiff --structure-feature oe --discard-sensitivity 15 \
  > logs/tiff_oe_B_proposed_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== Domain C (핵심 도메인) =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
  --gpu 4 --data-root tiff --structure-feature oe --discard-sensitivity 0 \
  > logs/tiff_oe_C_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
  --gpu 5 --data-root tiff --structure-feature oe --discard-sensitivity 15 \
  > logs/tiff_oe_C_proposed_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== Domain D =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_D --max-steps 5000 \
  --gpu 6 --data-root tiff --structure-feature oe --discard-sensitivity 0 \
  > logs/tiff_oe_D_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_D --max-steps 5000 \
  --gpu 7 --data-root tiff --structure-feature oe --discard-sensitivity 15 \
  > logs/tiff_oe_D_proposed_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "8 TIFF+OE experiments started (4 domains x 2 conditions)"
```

### 16.5 Quick Validation (1000 steps)

```bash
# Domain C만 빠른 검증
python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 1000 \
  --gpu 0 --data-root tiff --structure-feature oe --discard-sensitivity 0

python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 1000 \
  --gpu 1 --data-root tiff --structure-feature oe --discard-sensitivity 15
```

### 16.6 Sensitivity Sweep (Domain C + TIFF)

```bash
# OE Sensitivity 탐색 (5, 10, 15, 20, 30)
for SENS in 5 10 15 20 30; do
  nohup python examples/notebooks/hdmap_adaptive_validation.py \
    --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
    --gpu $((SENS/5-1)) --data-root tiff --structure-feature oe \
    --discard-sensitivity $SENS \
    > logs/tiff_oe_C_sens${SENS}_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done
```

### 16.7 PNG vs TIFF 비교 실험

PNG와 TIFF 데이터셋 간 성능 비교:

```bash
# PNG (기존)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
  --gpu 0 --data-root png --structure-feature oe --discard-sensitivity 15 \
  > logs/png_oe_C_sens15_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# TIFF (신규)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
  --gpu 1 --data-root tiff --structure-feature oe --discard-sensitivity 15 \
  > logs/tiff_oe_C_sens15_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 16.8 분석 명령어

```bash
cd results/hdmap_adaptive_validation
python analyze_results.py --auto-steps --summary
```

### 16.9 주요 확인사항

1. **TIFF 데이터 로딩**: 로그에서 "Using TIFF dataset" 확인
2. **OE 정상값 적용**: "Using domain-specific OE for domain_X" 확인
3. **k_discard 동작**: TensorBoard에서 `train/k_discard` 확인
4. **PNG vs TIFF 성능 차이**: 동일 조건에서 AUROC 비교

### 16.10 OE + TIFF Domain C 집중 실험 (16 GPUs)

Domain C에 대한 OE 피처 기반 집중 하이퍼파라미터 탐색.

| Exp | GPU | sensitivity | normal_value | ema_alpha | k_max | 설명 |
|-----|-----|-------------|--------------|-----------|-------|------|
| E0 | 0 | 0 | - | 0 | 0.9 | Baseline (no adaptation) |
| E1 | 1 | 5 | 0.9731 | 0 | 0.9 | Low sensitivity |
| E2 | 2 | 10 | 0.9731 | 0 | 0.9 | Medium sensitivity |
| E3 | 3 | 15 | 0.9731 | 0 | 0.9 | Proposed (default) |
| E4 | 4 | 20 | 0.9731 | 0 | 0.9 | High sensitivity |
| E5 | 5 | 30 | 0.9731 | 0 | 0.9 | Very high sensitivity |
| E6 | 6 | 15 | 0.95 | 0 | 0.9 | Lower normal_value |
| E7 | 7 | 15 | 0.98 | 0 | 0.9 | Higher normal_value |
| E8 | 8 | 15 | 0.99 | 0 | 0.9 | Very high normal_value |
| E9 | 9 | 15 | 0.9731 | 0.05 | 0.9 | EMA smoothing (0.05) |
| E10 | 10 | 15 | 0.9731 | 0.1 | 0.9 | EMA smoothing (0.1) |
| E11 | 11 | 15 | 0.9731 | 0.2 | 0.9 | EMA smoothing (0.2) |
| E12 | 12 | 15 | 0.9731 | 0 | 0.7 | Lower k_max |
| E13 | 13 | 15 | 0.9731 | 0 | 0.8 | Medium k_max |
| E14 | 14 | 20 | 0.98 | 0.1 | 0.9 | Combined: high sens + high nv + EMA |
| E15 | 15 | 10 | 0.95 | 0.05 | 0.8 | Combined: low sens + low nv + EMA + low k_max |

```bash
mkdir -p logs

# ===== Baseline =====
# E0: Baseline (no adaptation)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
  --gpu 0 --data-root tiff --structure-feature oe --discard-sensitivity 0 \
  > logs/tiff_oe_C_E0_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== Sensitivity Sweep (E1-E5) =====
# E1: sens=5
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
  --gpu 1 --data-root tiff --structure-feature oe --discard-sensitivity 5 \
  > logs/tiff_oe_C_E1_sens05_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# E2: sens=10
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
  --gpu 2 --data-root tiff --structure-feature oe --discard-sensitivity 10 \
  > logs/tiff_oe_C_E2_sens10_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# E3: sens=15 (Proposed default)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
  --gpu 3 --data-root tiff --structure-feature oe --discard-sensitivity 15 \
  > logs/tiff_oe_C_E3_sens15_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# E4: sens=20
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
  --gpu 4 --data-root tiff --structure-feature oe --discard-sensitivity 20 \
  > logs/tiff_oe_C_E4_sens20_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# E5: sens=30
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
  --gpu 5 --data-root tiff --structure-feature oe --discard-sensitivity 30 \
  > logs/tiff_oe_C_E5_sens30_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== Normal Value Sweep (E6-E8) =====
# E6: normal_value=0.95
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
  --gpu 6 --data-root tiff --structure-feature oe --discard-sensitivity 15 --normal-value 0.95 \
  > logs/tiff_oe_C_E6_nv095_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# E7: normal_value=0.98
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
  --gpu 7 --data-root tiff --structure-feature oe --discard-sensitivity 15 --normal-value 0.98 \
  > logs/tiff_oe_C_E7_nv098_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# E8: normal_value=0.99
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
  --gpu 8 --data-root tiff --structure-feature oe --discard-sensitivity 15 --normal-value 0.99 \
  > logs/tiff_oe_C_E8_nv099_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EMA Smoothing Sweep (E9-E11) =====
# E9: ema_alpha=0.05
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
  --gpu 9 --data-root tiff --structure-feature oe --discard-sensitivity 15 --ema-alpha 0.05 \
  > logs/tiff_oe_C_E9_ema005_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# E10: ema_alpha=0.1
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
  --gpu 10 --data-root tiff --structure-feature oe --discard-sensitivity 15 --ema-alpha 0.1 \
  > logs/tiff_oe_C_E10_ema01_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# E11: ema_alpha=0.2
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
  --gpu 11 --data-root tiff --structure-feature oe --discard-sensitivity 15 --ema-alpha 0.2 \
  > logs/tiff_oe_C_E11_ema02_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== k_max Sweep (E12-E13) =====
# E12: k_max=0.7
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
  --gpu 12 --data-root tiff --structure-feature oe --discard-sensitivity 15 --k-max 0.7 \
  > logs/tiff_oe_C_E12_kmax07_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# E13: k_max=0.8
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
  --gpu 13 --data-root tiff --structure-feature oe --discard-sensitivity 15 --k-max 0.8 \
  > logs/tiff_oe_C_E13_kmax08_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== Combined Settings (E14-E15) =====
# E14: High sens + High normal_value + EMA
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
  --gpu 14 --data-root tiff --structure-feature oe \
  --discard-sensitivity 20 --normal-value 0.98 --ema-alpha 0.1 \
  > logs/tiff_oe_C_E14_combined_high_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# E15: Low sens + Low normal_value + EMA + Low k_max
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
  --gpu 15 --data-root tiff --structure-feature oe \
  --discard-sensitivity 10 --normal-value 0.95 --ema-alpha 0.05 --k-max 0.8 \
  > logs/tiff_oe_C_E15_combined_low_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "All 16 TIFF+OE Domain C experiments started on GPUs 0-15."
echo "Check logs/ directory for progress."
```

**분석 포인트:**
1. **E0 vs E3**: Baseline vs Proposed (OE adaptive 효과)
2. **E1-E5**: sensitivity 최적값 탐색
3. **E6-E8**: normal_value 튜닝 (TIFF EDA 기준 0.9731 vs 다른 값)
4. **E9-E11**: EMA smoothing 효과
5. **E12-E13**: k_max 하향 조정 효과
6. **E14-E15**: 복합 설정 탐색

**결과 확인:**
```bash
cd results/hdmap_adaptive_validation
python analyze_results.py --auto-steps --summary --domain domain_C
```

---

## 17. TIFF Baseline 성능 검증 (4 Domains × 4 Conditions)

### 17.1 목적

TIFF 데이터셋 전환 후 각 도메인별 **원본 Dinomaly baseline** 성능 및 OE/APE 피처 효과 비교.

### 17.2 실험 설계

| Domain | GPU | Model | Feature | Sens | 설명 |
|--------|-----|-------|---------|------|------|
| domain_A | 0 | dinomaly | - | - | **원본 Dinomaly Baseline** |
| domain_A | 1 | dinomaly_adaptive | oe | 15 | OE Proposed |
| domain_A | 2 | dinomaly_adaptive | ape | 15 | APE Proposed |
| domain_A | 3 | dinomaly_adaptive | oe | 30 | OE High Sens |
| domain_B | 4 | dinomaly | - | - | **원본 Dinomaly Baseline** |
| domain_B | 5 | dinomaly_adaptive | oe | 15 | OE Proposed |
| domain_B | 6 | dinomaly_adaptive | ape | 15 | APE Proposed |
| domain_B | 7 | dinomaly_adaptive | oe | 30 | OE High Sens |
| domain_C | 8 | dinomaly | - | - | **원본 Dinomaly Baseline** |
| domain_C | 9 | dinomaly_adaptive | oe | 15 | OE Proposed |
| domain_C | 10 | dinomaly_adaptive | ape | 15 | APE Proposed |
| domain_C | 11 | dinomaly_adaptive | oe | 30 | OE High Sens |
| domain_D | 12 | dinomaly | - | - | **원본 Dinomaly Baseline** |
| domain_D | 13 | dinomaly_adaptive | oe | 15 | OE Proposed |
| domain_D | 14 | dinomaly_adaptive | ape | 15 | APE Proposed |
| domain_D | 15 | dinomaly_adaptive | oe | 30 | OE High Sens |

### 17.3 실행 명령어 (5000 steps, 16 GPUs)

```bash
mkdir -p logs

# ===== Domain A (GPU 0-3) =====
# Baseline: 원본 Dinomaly (no structure feature)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly --domain domain_A --max-steps 5000 \
  --gpu 0 --data-root tiff \
  > logs/tiff_A_dinomaly_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_A --max-steps 5000 \
  --gpu 1 --data-root tiff --structure-feature oe --discard-sensitivity 15 \
  > logs/tiff_A_oe_sens15_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_A --max-steps 5000 \
  --gpu 2 --data-root tiff --structure-feature ape --discard-sensitivity 15 \
  > logs/tiff_A_ape_sens15_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_A --max-steps 5000 \
  --gpu 3 --data-root tiff --structure-feature oe --discard-sensitivity 30 \
  > logs/tiff_A_oe_sens30_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== Domain B (GPU 4-7) =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly --domain domain_B --max-steps 5000 \
  --gpu 4 --data-root tiff \
  > logs/tiff_B_dinomaly_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_B --max-steps 5000 \
  --gpu 5 --data-root tiff --structure-feature oe --discard-sensitivity 15 \
  > logs/tiff_B_oe_sens15_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_B --max-steps 5000 \
  --gpu 6 --data-root tiff --structure-feature ape --discard-sensitivity 15 \
  > logs/tiff_B_ape_sens15_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_B --max-steps 5000 \
  --gpu 7 --data-root tiff --structure-feature oe --discard-sensitivity 30 \
  > logs/tiff_B_oe_sens30_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== Domain C (GPU 8-11) =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly --domain domain_C --max-steps 5000 \
  --gpu 8 --data-root tiff \
  > logs/tiff_C_dinomaly_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
  --gpu 9 --data-root tiff --structure-feature oe --discard-sensitivity 15 \
  > logs/tiff_C_oe_sens15_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
  --gpu 10 --data-root tiff --structure-feature ape --discard-sensitivity 15 \
  > logs/tiff_C_ape_sens15_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_C --max-steps 5000 \
  --gpu 11 --data-root tiff --structure-feature oe --discard-sensitivity 30 \
  > logs/tiff_C_oe_sens30_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== Domain D (GPU 12-15) =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly --domain domain_D --max-steps 5000 \
  --gpu 12 --data-root tiff \
  > logs/tiff_D_dinomaly_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_D --max-steps 5000 \
  --gpu 13 --data-root tiff --structure-feature oe --discard-sensitivity 15 \
  > logs/tiff_D_oe_sens15_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_D --max-steps 5000 \
  --gpu 14 --data-root tiff --structure-feature ape --discard-sensitivity 15 \
  > logs/tiff_D_ape_sens15_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_adaptive --domain domain_D --max-steps 5000 \
  --gpu 15 --data-root tiff --structure-feature oe --discard-sensitivity 30 \
  > logs/tiff_D_oe_sens30_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "All 16 TIFF experiments started (4 domains × 4 conditions)."
echo "Check logs/ directory for progress."
```

### 17.4 분석 포인트

1. **도메인별 TIFF Baseline**: 원본 Dinomaly 성능
2. **OE vs APE 효과**: 각 도메인에서 어떤 피처가 더 효과적인지
3. **Sensitivity 효과**: sens=15 vs sens=30
4. **PNG vs TIFF 비교**: 기존 PNG 결과와 TIFF baseline 비교

### 17.5 결과 분석

```bash
cd results/hdmap_adaptive_validation
python analyze_results.py --auto-steps --summary
```

### 17.6 예상 결과 테이블

| Domain | Dinomaly Baseline | OE sens=15 | APE sens=15 | OE sens=30 |
|--------|-------------------|------------|-------------|------------|
| A | ? | ? | ? | ? |
| B | ? | ? | ? | ? |
| C | ? | ? | ? | ? |
| D | ? | ? | ? | ? |

---

## 18. HDMAPLoss Domain C 집중 실험 (Row-wise Loss)

### 18.1 목적

Domain C의 **수평선 결함(horizontal line defect)** 특성에 최적화된 HDMAPLoss의 하이퍼파라미터 탐색.
APE/OE 기반 structure-adaptive discarding이 Domain C에서 효과가 없었던 점을 고려하여, 
**row-wise loss** 방식으로 접근.

### 18.2 HDMAPLoss 핵심 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `row_weight` | 0.3 | Row-wise loss 컴포넌트 가중치 |
| `top_k_ratio` | 0.15 | 집중할 top-k% 행 비율 (15% = 상위 에러 행) |
| `factor` | 0.1 | Easy point gradient 감소 비율 |
| `row_var_weight` | 0.1 | Row variance regularization 가중치 |

### 18.3 실험 설계 (12 실험)

| GPU | 실험 | row_weight | top_k_ratio | factor | row_var_weight | 설명 |
|-----|------|------------|-------------|--------|----------------|------|
| 0 | EXP01 | - | - | - | - | **Dinomaly Baseline** (비교군) |
| 1 | EXP02 | 0.3 | 0.15 | 0.1 | 0.1 | **HDMAPLoss Default** |
| 2 | EXP03 | 0.1 | 0.15 | 0.1 | 0.1 | Row weight Low |
| 3 | EXP04 | 0.5 | 0.15 | 0.1 | 0.1 | Row weight High |
| 4 | EXP05 | 0.3 | 0.10 | 0.1 | 0.1 | Top-k 10% (더 적은 행 집중) |
| 5 | EXP06 | 0.3 | 0.20 | 0.1 | 0.1 | Top-k 20% |
| 6 | EXP07 | 0.3 | 0.25 | 0.1 | 0.1 | Top-k 25% (더 많은 행 집중) |
| 7 | EXP08 | 0.3 | 0.15 | 0.05 | 0.1 | Factor Low (gradient 덜 감소) |
| 8 | EXP09 | 0.3 | 0.15 | 0.2 | 0.1 | Factor High (gradient 더 감소) |
| 9 | EXP10 | 0.3 | 0.15 | 0.1 | 0.05 | Row Var Low |
| 10 | EXP11 | 0.3 | 0.15 | 0.1 | 0.2 | Row Var High |
| 11 | EXP12 | 0.5 | 0.20 | 0.1 | 0.15 | **Aggressive** (row 강조) |

### 18.4 실행 명령어 (5000 steps, 12 GPUs)

```bash
mkdir -p logs

# ===== EXP01: Dinomaly Baseline (비교군) =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly --domain domain_C --max-steps 5000 \
  --gpu 0 --data-root tiff \
  > logs/hdmap_C_exp01_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP02: HDMAPLoss Default =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 1 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.15 --factor 0.1 --row-var-weight 0.1 \
  > logs/hdmap_C_exp02_default_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP03: Row Weight Low (0.1) =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 2 --data-root tiff \
  --row-weight 0.1 --top-k-ratio 0.15 --factor 0.1 --row-var-weight 0.1 \
  > logs/hdmap_C_exp03_rw01_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP04: Row Weight High (0.5) =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 3 --data-root tiff \
  --row-weight 0.5 --top-k-ratio 0.15 --factor 0.1 --row-var-weight 0.1 \
  > logs/hdmap_C_exp04_rw05_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP05: Top-k 10% =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 4 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.10 --factor 0.1 --row-var-weight 0.1 \
  > logs/hdmap_C_exp05_topk10_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP06: Top-k 20% =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 5 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.20 --factor 0.1 --row-var-weight 0.1 \
  > logs/hdmap_C_exp06_topk20_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP07: Top-k 25% =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 6 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.25 --factor 0.1 --row-var-weight 0.1 \
  > logs/hdmap_C_exp07_topk25_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP08: Factor Low (0.05) =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 7 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.15 --factor 0.05 --row-var-weight 0.1 \
  > logs/hdmap_C_exp08_factor005_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP09: Factor High (0.2) =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 8 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.15 --factor 0.2 --row-var-weight 0.1 \
  > logs/hdmap_C_exp09_factor02_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP10: Row Var Low (0.05) =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 9 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.15 --factor 0.1 --row-var-weight 0.05 \
  > logs/hdmap_C_exp10_rowvar005_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP11: Row Var High (0.2) =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 10 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.15 --factor 0.1 --row-var-weight 0.2 \
  > logs/hdmap_C_exp11_rowvar02_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP12: Aggressive (row 강조) =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 11 --data-root tiff \
  --row-weight 0.5 --top-k-ratio 0.20 --factor 0.1 --row-var-weight 0.15 \
  > logs/hdmap_C_exp12_aggressive_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "All 12 HDMAPLoss Domain C experiments started."
echo "Check logs/ directory for progress."
```

### 18.5 분석 포인트

1. **Baseline vs HDMAPLoss Default**: Row-wise loss가 Domain C에 효과적인지
2. **Row Weight 영향**: 0.1 vs 0.3 vs 0.5 - row loss 비중
3. **Top-k Ratio 영향**: 10% vs 15% vs 20% vs 25% - 집중 행 비율
4. **Factor 영향**: gradient 감소 정도
5. **Row Variance 영향**: 정상/결함 행 분리 강도

### 18.6 결과 분석

```bash
cd results/hdmap_adaptive_validation
python analyze_results.py --auto-steps --summary --domain domain_C
```

### 18.7 예상 결과 테이블

| EXP | Model | row_weight | top_k_ratio | factor | row_var_weight | AUROC |
|-----|-------|------------|-------------|--------|----------------|-------|
| 01 | dinomaly | - | - | - | - | ? (baseline) |
| 02 | hdmap | 0.3 | 0.15 | 0.1 | 0.1 | ? |
| 03 | hdmap | 0.1 | 0.15 | 0.1 | 0.1 | ? |
| 04 | hdmap | 0.5 | 0.15 | 0.1 | 0.1 | ? |
| 05 | hdmap | 0.3 | 0.10 | 0.1 | 0.1 | ? |
| 06 | hdmap | 0.3 | 0.20 | 0.1 | 0.1 | ? |
| 07 | hdmap | 0.3 | 0.25 | 0.1 | 0.1 | ? |
| 08 | hdmap | 0.3 | 0.15 | 0.05 | 0.1 | ? |
| 09 | hdmap | 0.3 | 0.15 | 0.2 | 0.1 | ? |
| 10 | hdmap | 0.3 | 0.15 | 0.1 | 0.05 | ? |
| 11 | hdmap | 0.3 | 0.15 | 0.1 | 0.2 | ? |
| 12 | hdmap | 0.5 | 0.20 | 0.1 | 0.15 | ? |

### 18.8 성공 기준

- **목표**: Domain C baseline (약 95.98%) 대비 향상
- **최소 성공**: baseline과 동등 이상
- **의미 있는 향상**: +1% AUROC 이상

---

### 18.9 Round 1 결과 분석

| 순위 | row_weight | top_k_ratio | factor | row_var_weight | Test AUROC |
|------|------------|-------------|--------|----------------|------------|
| 1 | 0.3 | 0.15 | 0.1 | **0.05** | **96.10%** |
| 2 | 0.3 | 0.15 | 0.1 | 0.10 | 96.05% |
| 3 | 0.3 | 0.25 | 0.1 | 0.10 | 95.41% |
| - | **Baseline** | - | - | - | **95.59%** |

**핵심 발견:**
- `row_var_weight`가 가장 민감: 0.05가 최적, 낮을수록 좋은 경향
- `row_weight=0.3`이 최적, 0.1/0.5는 성능 저하
- `top_k_ratio`, `factor`는 상대적으로 둔감

---

### 18.10 Round 2: row_var_weight 미세 조정 (16 실험)

Round 1에서 `row_var_weight=0.05`가 최고 성능을 보였으므로,
더 낮은 값(0.00~0.04)과 주변 파라미터 조합을 탐색.

| GPU | 실험 | row_weight | top_k_ratio | factor | row_var_weight | 설명 |
|-----|------|------------|-------------|--------|----------------|------|
| 0 | EXP13 | 0.3 | 0.15 | 0.1 | 0.05 | **Round 1 Best** (재현성 확인) |
| 1 | EXP14 | 0.3 | 0.15 | 0.1 | 0.04 | row_var 더 낮춤 |
| 2 | EXP15 | 0.3 | 0.15 | 0.1 | 0.03 | row_var 더 낮춤 |
| 3 | EXP16 | 0.3 | 0.15 | 0.1 | 0.02 | row_var 더 낮춤 |
| 4 | EXP17 | 0.3 | 0.15 | 0.1 | 0.01 | row_var 매우 낮음 |
| 5 | EXP18 | 0.3 | 0.15 | 0.1 | 0.00 | **row_var 비활성화** |
| 6 | EXP19 | 0.25 | 0.15 | 0.1 | 0.05 | row_weight 낮춤 |
| 7 | EXP20 | 0.35 | 0.15 | 0.1 | 0.05 | row_weight 높임 |
| 8 | EXP21 | 0.3 | 0.15 | 0.08 | 0.05 | factor 낮춤 |
| 9 | EXP22 | 0.3 | 0.15 | 0.12 | 0.05 | factor 높임 |
| 10 | EXP23 | 0.3 | 0.12 | 0.1 | 0.05 | top_k 낮춤 |
| 11 | EXP24 | 0.3 | 0.18 | 0.1 | 0.05 | top_k 높임 |
| 12 | EXP25 | 0.25 | 0.15 | 0.1 | 0.03 | 조합1: rW↓ + rVar↓ |
| 13 | EXP26 | 0.35 | 0.15 | 0.1 | 0.03 | 조합2: rW↑ + rVar↓ |
| 14 | EXP27 | 0.3 | 0.15 | 0.08 | 0.03 | 조합3: factor↓ + rVar↓ |
| 15 | EXP28 | 0.3 | 0.12 | 0.1 | 0.03 | 조합4: topK↓ + rVar↓ |

### 18.11 Round 2 실행 명령어 (5000 steps, 16 GPUs)

```bash
mkdir -p logs

# ===== EXP13: Round 1 Best 재현 =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 0 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.15 --factor 0.1 --row-var-weight 0.05 \
  > logs/hdmap_C_exp13_r1best_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP14: row_var=0.04 =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 1 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.15 --factor 0.1 --row-var-weight 0.04 \
  > logs/hdmap_C_exp14_rvar004_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP15: row_var=0.03 =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 2 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.15 --factor 0.1 --row-var-weight 0.03 \
  > logs/hdmap_C_exp15_rvar003_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP16: row_var=0.02 =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 3 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.15 --factor 0.1 --row-var-weight 0.02 \
  > logs/hdmap_C_exp16_rvar002_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP17: row_var=0.01 =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 4 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.15 --factor 0.1 --row-var-weight 0.01 \
  > logs/hdmap_C_exp17_rvar001_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP18: row_var=0.00 (비활성화) =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 5 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.15 --factor 0.1 --row-var-weight 0.00 \
  > logs/hdmap_C_exp18_rvar000_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP19: row_weight=0.25 =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 6 --data-root tiff \
  --row-weight 0.25 --top-k-ratio 0.15 --factor 0.1 --row-var-weight 0.05 \
  > logs/hdmap_C_exp19_rw025_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP20: row_weight=0.35 =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 7 --data-root tiff \
  --row-weight 0.35 --top-k-ratio 0.15 --factor 0.1 --row-var-weight 0.05 \
  > logs/hdmap_C_exp20_rw035_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP21: factor=0.08 =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 8 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.15 --factor 0.08 --row-var-weight 0.05 \
  > logs/hdmap_C_exp21_factor008_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP22: factor=0.12 =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 9 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.15 --factor 0.12 --row-var-weight 0.05 \
  > logs/hdmap_C_exp22_factor012_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP23: top_k=0.12 =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 10 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.12 --factor 0.1 --row-var-weight 0.05 \
  > logs/hdmap_C_exp23_topk012_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP24: top_k=0.18 =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 11 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.18 --factor 0.1 --row-var-weight 0.05 \
  > logs/hdmap_C_exp24_topk018_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP25: 조합1 (row_weight=0.25, row_var=0.03) =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 12 --data-root tiff \
  --row-weight 0.25 --top-k-ratio 0.15 --factor 0.1 --row-var-weight 0.03 \
  > logs/hdmap_C_exp25_combo1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP26: 조합2 (row_weight=0.35, row_var=0.03) =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 13 --data-root tiff \
  --row-weight 0.35 --top-k-ratio 0.15 --factor 0.1 --row-var-weight 0.03 \
  > logs/hdmap_C_exp26_combo2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP27: 조합3 (factor=0.08, row_var=0.03) =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 14 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.15 --factor 0.08 --row-var-weight 0.03 \
  > logs/hdmap_C_exp27_combo3_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== EXP28: 조합4 (top_k=0.12, row_var=0.03) =====
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 5000 \
  --gpu 15 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.12 --factor 0.1 --row-var-weight 0.03 \
  > logs/hdmap_C_exp28_combo4_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "All 16 Round 2 experiments started (EXP13-EXP28)."
echo "Check logs/ directory for progress."
```

### 18.12 Round 2 예상 결과 테이블

| EXP | row_weight | top_k_ratio | factor | row_var_weight | AUROC | 비고 |
|-----|------------|-------------|--------|----------------|-------|------|
| 13 | 0.3 | 0.15 | 0.1 | 0.05 | ? | Round 1 Best 재현 |
| 14 | 0.3 | 0.15 | 0.1 | 0.04 | ? | row_var 미세 조정 |
| 15 | 0.3 | 0.15 | 0.1 | 0.03 | ? | row_var 미세 조정 |
| 16 | 0.3 | 0.15 | 0.1 | 0.02 | ? | row_var 미세 조정 |
| 17 | 0.3 | 0.15 | 0.1 | 0.01 | ? | row_var 매우 낮음 |
| 18 | 0.3 | 0.15 | 0.1 | 0.00 | ? | row_var 비활성화 |
| 19 | 0.25 | 0.15 | 0.1 | 0.05 | ? | row_weight 낮춤 |
| 20 | 0.35 | 0.15 | 0.1 | 0.05 | ? | row_weight 높임 |
| 21 | 0.3 | 0.15 | 0.08 | 0.05 | ? | factor 낮춤 |
| 22 | 0.3 | 0.15 | 0.12 | 0.05 | ? | factor 높임 |
| 23 | 0.3 | 0.12 | 0.1 | 0.05 | ? | top_k 낮춤 |
| 24 | 0.3 | 0.18 | 0.1 | 0.05 | ? | top_k 높임 |
| 25 | 0.25 | 0.15 | 0.1 | 0.03 | ? | 조합1 |
| 26 | 0.35 | 0.15 | 0.1 | 0.03 | ? | 조합2 |
| 27 | 0.3 | 0.15 | 0.08 | 0.03 | ? | 조합3 |
| 28 | 0.3 | 0.12 | 0.1 | 0.03 | ? | 조합4 |

### 18.13 Round 2 분석 목표

1. **row_var_weight 최적값 탐색**: 0.05 vs 0.04 vs 0.03 vs 0.02 vs 0.01 vs 0.00
2. **row_weight 민감도 재확인**: 0.25 vs 0.30 vs 0.35 (row_var=0.05 고정)
3. **조합 효과 탐색**: row_var=0.03 + 다른 파라미터 조합

### 18.14 결과 분석 명령어

```bash
cd results/hdmap_adaptive_validation
python analyze_results.py --auto-steps --summary --compare --domain domain_C
```

### 18.15 Round 3: 전체 도메인 검증 (16 GPUs, 4 domains × 4 experiments)

Round 2에서 Domain C 최적 파라미터를 도출했으므로, 이를 전체 도메인에 적용하여 일반화 성능을 검증한다.

**실험 설계:**
- 각 도메인당 4개 실험 (DINOMALY baseline + HDMAPLoss 3가지 설정)
- 총 16개 실험 (GPU 0-15 사용)
- 10,000 steps, batch_size=32, 3 repeats (통계적 유의성 확보)

**파라미터 조합:**
| 설정 | row_weight | top_k_ratio | factor | row_var_weight | 비고 |
|------|------------|-------------|--------|----------------|------|
| Baseline | - | - | - | - | DINOMALY 원본 |
| Config A | 0.3 | 0.12 | 0.1 | 0.05 | Domain C 최적 |
| Config B | 0.3 | 0.15 | 0.1 | 0.05 | top_k 원본 |
| Config C | 0.3 | 0.12 | 0.1 | 0.03 | row_var 낮춤 |

### 18.16 Round 3 실행 스크립트

```bash
# ============================================================
# ROUND 3: 전체 도메인 검증 (16 GPUs)
# 실행 시간: 약 6-8시간 (10k steps × 3 repeats × 16 experiments)
# ============================================================

mkdir -p logs

# ===== DOMAIN A (GPU 0-3) =====
# EXP_A1: DINOMALY Baseline
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly --domain domain_A --max-steps 10000 \
  --batch-size 32 --repeats 3 \
  --gpu 0 --data-root tiff \
  > logs/hdmap_A_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# EXP_A2: HDMAPLoss Config A (Domain C 최적)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_A --max-steps 10000 \
  --batch-size 32 --repeats 3 \
  --gpu 1 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.12 --factor 0.1 --row-var-weight 0.05 \
  > logs/hdmap_A_configA_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# EXP_A3: HDMAPLoss Config B (top_k 원본)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_A --max-steps 10000 \
  --batch-size 32 --repeats 3 \
  --gpu 2 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.15 --factor 0.1 --row-var-weight 0.05 \
  > logs/hdmap_A_configB_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# EXP_A4: HDMAPLoss Config C (row_var 낮춤)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_A --max-steps 10000 \
  --batch-size 32 --repeats 3 \
  --gpu 3 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.12 --factor 0.1 --row-var-weight 0.03 \
  > logs/hdmap_A_configC_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== DOMAIN B (GPU 4-7) =====
# EXP_B1: DINOMALY Baseline
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly --domain domain_B --max-steps 10000 \
  --batch-size 32 --repeats 3 \
  --gpu 4 --data-root tiff \
  > logs/hdmap_B_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# EXP_B2: HDMAPLoss Config A (Domain C 최적)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_B --max-steps 10000 \
  --batch-size 32 --repeats 3 \
  --gpu 5 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.12 --factor 0.1 --row-var-weight 0.05 \
  > logs/hdmap_B_configA_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# EXP_B3: HDMAPLoss Config B (top_k 원본)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_B --max-steps 10000 \
  --batch-size 32 --repeats 3 \
  --gpu 6 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.15 --factor 0.1 --row-var-weight 0.05 \
  > logs/hdmap_B_configB_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# EXP_B4: HDMAPLoss Config C (row_var 낮춤)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_B --max-steps 10000 \
  --batch-size 32 --repeats 3 \
  --gpu 7 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.12 --factor 0.1 --row-var-weight 0.03 \
  > logs/hdmap_B_configC_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== DOMAIN C (GPU 8-11) =====
# EXP_C1: DINOMALY Baseline
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly --domain domain_C --max-steps 10000 \
  --batch-size 32 --repeats 3 \
  --gpu 8 --data-root tiff \
  > logs/hdmap_C_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# EXP_C2: HDMAPLoss Config A (Domain C 최적)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 10000 \
  --batch-size 32 --repeats 3 \
  --gpu 9 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.12 --factor 0.1 --row-var-weight 0.05 \
  > logs/hdmap_C_configA_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# EXP_C3: HDMAPLoss Config B (top_k 원본)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 10000 \
  --batch-size 32 --repeats 3 \
  --gpu 10 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.15 --factor 0.1 --row-var-weight 0.05 \
  > logs/hdmap_C_configB_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# EXP_C4: HDMAPLoss Config C (row_var 낮춤)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_C --max-steps 10000 \
  --batch-size 32 --repeats 3 \
  --gpu 11 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.12 --factor 0.1 --row-var-weight 0.03 \
  > logs/hdmap_C_configC_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ===== DOMAIN D (GPU 12-15) =====
# EXP_D1: DINOMALY Baseline
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly --domain domain_D --max-steps 10000 \
  --batch-size 32 --repeats 3 \
  --gpu 12 --data-root tiff \
  > logs/hdmap_D_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# EXP_D2: HDMAPLoss Config A (Domain C 최적)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_D --max-steps 10000 \
  --batch-size 32 --repeats 3 \
  --gpu 13 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.12 --factor 0.1 --row-var-weight 0.05 \
  > logs/hdmap_D_configA_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# EXP_D3: HDMAPLoss Config B (top_k 원본)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_D --max-steps 10000 \
  --batch-size 32 --repeats 3 \
  --gpu 14 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.15 --factor 0.1 --row-var-weight 0.05 \
  > logs/hdmap_D_configB_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# EXP_D4: HDMAPLoss Config C (row_var 낮춤)
nohup python examples/notebooks/hdmap_adaptive_validation.py \
  --model dinomaly_hdmap --domain domain_D --max-steps 10000 \
  --batch-size 32 --repeats 3 \
  --gpu 15 --data-root tiff \
  --row-weight 0.3 --top-k-ratio 0.12 --factor 0.1 --row-var-weight 0.03 \
  > logs/hdmap_D_configC_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "All 16 Round 3 experiments started across 4 domains."
echo "Expected runtime: 6-8 hours (10k steps × 3 repeats)"
echo "Check logs/ directory for progress."
```

### 18.17 Round 3 예상 결과 테이블

| Domain | Config | row_weight | top_k | factor | row_var | AUROC (mean±std) |
|--------|--------|------------|-------|--------|---------|------------------|
| A | Baseline | - | - | - | - | ? |
| A | Config A | 0.3 | 0.12 | 0.1 | 0.05 | ? |
| A | Config B | 0.3 | 0.15 | 0.1 | 0.05 | ? |
| A | Config C | 0.3 | 0.12 | 0.1 | 0.03 | ? |
| B | Baseline | - | - | - | - | ? |
| B | Config A | 0.3 | 0.12 | 0.1 | 0.05 | ? |
| B | Config B | 0.3 | 0.15 | 0.1 | 0.05 | ? |
| B | Config C | 0.3 | 0.12 | 0.1 | 0.03 | ? |
| C | Baseline | - | - | - | - | ? |
| C | Config A | 0.3 | 0.12 | 0.1 | 0.05 | ? |
| C | Config B | 0.3 | 0.15 | 0.1 | 0.05 | ? |
| C | Config C | 0.3 | 0.12 | 0.1 | 0.03 | ? |
| D | Baseline | - | - | - | - | ? |
| D | Config A | 0.3 | 0.12 | 0.1 | 0.05 | ? |
| D | Config B | 0.3 | 0.15 | 0.1 | 0.05 | ? |
| D | Config C | 0.3 | 0.12 | 0.1 | 0.03 | ? |

### 18.18 Round 3 분석 목표

1. **일반화 검증**: Domain C 최적 파라미터가 다른 도메인에도 유효한지 확인
2. **도메인별 최적값 차이**: 도메인 특성에 따라 최적 파라미터가 다를 수 있음
3. **통계적 유의성**: 3회 반복 실험으로 mean ± std 확보
4. **Baseline 대비 개선율**: 각 도메인에서 HDMAPLoss의 효과 측정

### 18.19 Round 3 결과 분석 명령어

```bash
# 전체 도메인 결과 분석
cd results/hdmap_adaptive_validation
python analyze_results.py --auto-steps --summary --compare

# 도메인별 상세 분석
python analyze_results.py --auto-steps --summary --domain domain_A
python analyze_results.py --auto-steps --summary --domain domain_B
python analyze_results.py --auto-steps --summary --domain domain_C
python analyze_results.py --auto-steps --summary --domain domain_D
```
