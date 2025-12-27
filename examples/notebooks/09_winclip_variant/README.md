# WinCLIP Experiments

WinCLIP 모델의 MVTec-AD 데이터셋 성능 검증 및 변형 실험을 위한 스크립트 모음.

## WinCLIP 개요

WinCLIP은 CLIP 임베딩과 슬라이딩 윈도우 방식을 사용하는 zero-shot/few-shot 이상 탐지 모델.

- **Zero-shot (k=0)**: 텍스트 프롬프트만으로 이상 탐지 (학습 불필요)
- **Few-shot (k=1,2,4)**: Normal reference 이미지 k장 사용

## 파일 구조

```
09_winclip_variant/
├── README.md                      # 이 파일
├── winclip_mvtec_validation.py    # MVTec-AD 검증 스크립트
└── results/                       # 결과 저장 (gitignore)
```

## 실행 방법

**모든 명령어는 anomalib 루트 디렉토리에서 실행**

### 1. 단일 카테고리 테스트 (빠른 확인용)

```bash
# Zero-shot만 테스트
nohup python examples/notebooks/09_winclip_variant/winclip_mvtec_validation.py \
    --categories bottle \
    --k-shots 0 \
    --gpu 0 \
    > logs/winclip_bottle_0shot.log 2>&1 &

# 모든 k_shot 모드 테스트
nohup python examples/notebooks/09_winclip_variant/winclip_mvtec_validation.py \
    --categories bottle \
    --k-shots 0 1 2 4 \
    --gpu 0 \
    > logs/winclip_bottle_all.log 2>&1 &
```

### 2. 전체 카테고리 테스트 (nohup 백그라운드)

```bash
# 전체 15개 카테고리, Zero-shot
nohup python examples/notebooks/09_winclip_variant/winclip_mvtec_validation.py \
    --categories all \
    --k-shots 0 \
    --result-dir ./results/winclip_mvtec_full \
    --gpu 0 \
    > logs/winclip_validation_full.log 2>&1 &

# 전체 15개 카테고리, 모든 k_shot 모드
nohup python examples/notebooks/09_winclip_variant/winclip_mvtec_validation.py \
    --categories all \
    --k-shots 0 1 2 4 \
    --result-dir ./results/winclip_mvtec_full_all_shots \
    --gpu 0 \
    > logs/winclip_validation_full_all_shots.log 2>&1 &

# 로그 확인
tail -f logs/winclip_validation_full.log
```

### 3. 특정 카테고리만 테스트

```bash
# 텍스처 카테고리들
nohup python examples/notebooks/09_winclip_variant/winclip_mvtec_validation.py \
    --categories carpet grid leather tile wood \
    --k-shots 0 1 2 4 \
    --result-dir ./results/winclip_mvtec_textures \
    --gpu 0 \
    > logs/winclip_textures.log 2>&1 &

# 오브젝트 카테고리들
nohup python examples/notebooks/09_winclip_variant/winclip_mvtec_validation.py \
    --categories bottle cable capsule hazelnut metal_nut pill screw toothbrush transistor zipper \
    --k-shots 0 1 2 4 \
    --result-dir ./results/winclip_mvtec_objects \
    --gpu 1 \
    > logs/winclip_objects.log 2>&1 &
```

## CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--categories` | `bottle` | 테스트할 카테고리 (`all`로 전체 선택) |
| `--k-shots` | `0 1 2 4` | 테스트할 k_shot 값들 |
| `--result-dir` | `./results/winclip_mvtec_validation` | 결과 저장 경로 |
| `--gpu` | `0` | GPU ID |

## 결과 구조

```
results/winclip_mvtec_validation/
└── 20241226_123456/              # 타임스탬프
    ├── experiment_settings.json  # 실험 설정
    ├── summary.json              # 전체 결과 요약
    ├── bottle/
    │   ├── zero-shot/
    │   │   └── results.json
    │   ├── 1-shot/
    │   │   └── results.json
    │   ├── 2-shot/
    │   │   └── results.json
    │   └── 4-shot/
    │       └── results.json
    └── ...
```

## 예상 성능 (WinCLIP 논문 기준)

| Mode | Image AUROC (Avg) | Pixel AUROC (Avg) |
|------|-------------------|-------------------|
| Zero-shot | ~91.8% | ~85.1% |
| 1-shot | ~93.1% | ~87.3% |
| 2-shot | ~94.4% | ~89.1% |
| 4-shot | ~95.2% | ~90.3% |

*논문: [WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation](https://arxiv.org/abs/2303.14814)*

## 문제 해결

### CLIP 모델 다운로드 오류
```bash
# open_clip 패키지 설치 확인
pip install open-clip-torch
```

### GPU 메모리 부족
```python
# eval_batch_size 줄이기
datamodule = MVTecAD(
    eval_batch_size=16,  # 32에서 줄임
    ...
)
```
