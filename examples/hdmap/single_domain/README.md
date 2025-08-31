# Single Domain Anomaly Detection 통합 실험 가이드

## 🎯 개요

이 디렉토리는 통합된 Base 시스템을 통해 모든 anomaly detection 모델의 단일 도메인 실험을 효율적으로 관리합니다. 
개별 모델별 파일 대신 공통 Base 템플릿을 사용하여 코드 중복을 제거하고 실험 일관성을 확보합니다.

## 📁 파일 구조

```
single_domain/
├── README.md                          # 이 파일 - 사용 가이드
├── anomaly_trainer.py                 # BaseAnomalyTrainer 클래스 (핵심)
├── base-training.py                   # 메인 실행 스크립트
├── base-run.sh                        # 병렬 실험 실행 스크립트
└── base-exp_condition1.json          # 실험 조건 설정 파일
```

## 🚀 빠른 시작

### 단일 실험 실행
```bash
# 특정 실험 ID 실행 (예: ID 0번 실험)
python examples/hdmap/single_domain/base-training.py \
    --config examples/hdmap/single_domain/base-exp_condition1.json \
    --experiment-id 0 \
    --gpu-id 0
```

### 모든 실험 병렬 실행 (추천)
```bash
# 백그라운드에서 모든 실험 실행
nohup ./examples/hdmap/single_domain/base-run.sh all > training.log 2>&1 &

# 실행 상태 확인
tail -f training.log                               # 메인 스크립트 진행 상황
tail -f results/*/training_detail.log             # 개별 실험 상세 로그
tail -f results/*/domain*_single.log              # 실험별 구조화된 로그
```

## 🏗️ 시스템 아키텍처

### 핵심 컴포넌트

#### 1. BaseAnomalyTrainer (`anomaly_trainer.py`)
모든 모델의 공통 로직을 처리하는 통합 클래스:

```python
class BaseAnomalyTrainer:
    def __init__(self, config, experiment_name, session_timestamp):
        self.model_type = config["model_type"]  # 모델 타입별 Factory 분기
    
    def create_model(self):
        # Factory Pattern으로 모델 생성
        if self.model_type == "draem":
            return self._create_draem_model()
        elif self.model_type == "dinomaly":
            return self._create_dinomaly_model()
        # ...
    
    def run_experiment(self):
        # 공통 실험 실행 로직
```

#### 2. 실험 조건 설정 (`base-exp_condition1.json`)
모든 모델의 실험 설정을 JSON 형태로 통합 관리:

```json
{
  "experiment_conditions": [
    {
      "name": "domainA_draem_baseline",
      "description": "Domain A - DRAEM 기본 설정",
      "config": {
        "model_type": "draem",           // 모델 타입 지정
        "source_domain": "domain_A",
        "max_epochs": 5,
        "learning_rate": 0.0001,
        "batch_size": 16
      }
    },
    {
      "name": "domainA_dinomaly_baseline",
      "config": {
        "model_type": "dinomaly",        // 모델별 고유 설정
        "batch_size": 8,
        "encoder_name": "dinov2reg_vit_base_14"
      }
    }
  ]
}
```

### 지원 모델

현재 지원하는 anomaly detection 모델:

| 모델 | model_type | 설명 | 주요 특징 |
|------|------------|------|-----------|
| **DRAEM** | `"draem"` | Reconstruction 기반 anomaly detection | 빠른 훈련, 안정적 |
| **Dinomaly** | `"dinomaly"` | Vision Transformer 기반 with DINOv2 | 높은 성능, 메모리 집약적 |
| **PatchCore** | `"patchcore"` | Memory bank 기반 few-shot learning | 메모리 효율적, 빠른 추론 |
| **DRAEM-SevNet** | `"draem_sevnet"` | Selective feature reconstruction | 정교한 anomaly 탐지 |

## 📊 로그 구조 및 모니터링

### 로그 파일 구조
실험 실행 시 다음과 같은 계층적 로그 구조가 생성됩니다:

```
results/20250831_032243/                           # 세션 타임스탬프
├── domainA_draem_baseline_20250831_032243/        # 실험별 디렉토리
│   ├── training_detail.log                        # 상세 훈련 로그 (tqdm, 에러 등)
│   ├── domain_A_single.log                       # 구조화된 결과 로그
│   └── tensorboard_logs/                         # TensorBoard 로그
├── domainA_dinomaly_baseline_20250831_032243/
│   ├── training_detail.log
│   └── domain_A_single.log
└── ...
```

### 로그 모니터링 명령어

```bash
# 전체 실험 진행 상황
tail -f training.log

# 특정 모델의 상세 훈련 로그 (실시간 tqdm 바 확인)
tail -f results/20250831_*/domainA_dinomaly_*/training_detail.log

# 가장 최근 실험의 구조화된 로그
tail -f $(ls -t results/*/domain*_single.log | head -1)

# 모든 실험의 최종 결과 확인
grep "Image AUROC" results/*/domain*_single.log
```

## ⚙️ 실험 설정 커스터마이징

### 새로운 실험 조건 추가

`base-exp_condition1.json`에 새로운 실험 조건을 추가:

```json
{
  "name": "domainA_patchcore_optimized",
  "description": "Domain A - PatchCore 최적화된 설정",
  "config": {
    "model_type": "patchcore",
    "source_domain": "domain_A", 
    "max_epochs": 1,
    "batch_size": 32,
    "coreset_sampling_ratio": 0.1,
    "num_neighbors": 9
  }
}
```

### 모델별 주요 파라미터

#### DRAEM
- `batch_size`: 16 (권장)
- `learning_rate`: 0.0001
- `anomaly_source_path`: None (기본값 사용)

#### Dinomaly  
- `batch_size`: 8 (메모리 제약)
- `encoder_name`: `"dinov2reg_vit_base_14"`
- `target_layers`: `[2,3,4,5,6,7,8,9]`

#### PatchCore
- `batch_size`: 32 (빠른 피팅)
- `coreset_sampling_ratio`: 0.1
- `num_neighbors`: 9

#### DRAEM-SevNet
- `batch_size`: 16
- 동일한 DRAEM 파라미터 + SevNet head

## 🔧 고급 사용법

### GPU 할당 커스터마이징

`base-run.sh`에서 사용할 GPU 목록 수정:
```bash
# 사용할 GPU ID 목록 변경
AVAILABLE_GPUS=(0 1 2 3)  # 4개 GPU만 사용
```

### 실험 병렬 실행 제어

```bash
# 특정 실험만 실행
./examples/hdmap/single_domain/base-run.sh 2  # ID 2번 실험만

# 실험 진행 상황 확인
ps aux | grep base-training                   # 실행 중인 실험들
nvidia-smi                                    # GPU 사용 현황
```

### Early Stopping 전략

모든 모델에서 `val_loss`를 사용하여 Early Stopping:
- Single domain에서 validation은 모두 정상 데이터이므로 AUROC 부적절
- DRAEM 계열: ValidationLossCallback으로 수동 val_loss 계산
- 다른 모델: Lightning 기본 val_loss 사용

## 📈 결과 분석

### 실험 결과 비교

각 실험 완료 후 다음 파일들을 확인:

```bash
# JSON 결과 파일
ls results/*/result_*.json

# 최종 AUROC 성능 비교
grep -r "Image AUROC" results/*/domain*_single.log | sort -k3 -nr

# TensorBoard로 훈련 과정 시각화
tensorboard --logdir results/20250831_*/tensorboard_logs
```

### 성능 벤치마크 예시

| 모델 | Domain A AUROC | 훈련 시간 | 메모리 사용량 |
|------|----------------|-----------|---------------|
| PatchCore | 0.85+ | ~5분 | 낮음 |
| DRAEM | 0.80+ | ~30분 | 중간 |
| Dinomaly | 0.88+ | ~60분 | 높음 |
| DRAEM-SevNet | 0.82+ | ~40분 | 중간 |

## 🛠️ 문제 해결

### 일반적인 문제들

#### 1. GPU 메모리 부족
```bash
# Dinomaly 배치 사이즈 줄이기
"batch_size": 4  # 8에서 4로 변경
```

#### 2. 실험 중단됨
```bash
# 실행 중인 프로세스 확인
ps aux | grep base-training

# 특정 실험만 재실행
python base-training.py --config base-exp_condition1.json --experiment-id 2
```

#### 3. 로그 파일을 찾을 수 없음
```bash
# 최신 결과 디렉토리 확인
ls -la results/
find results -name "*.log" | head -5
```

## 🔄 마이그레이션 가이드

### 기존 개별 스크립트에서 Base 시스템으로 이전

기존의 `draem-training.py`, `dinomaly-training.py` 등을 사용하던 경우:

1. **실험 설정 이전**: 기존 파라미터를 `base-exp_condition1.json`으로 복사
2. **실행 방식 변경**: `base-training.py` 사용
3. **로그 위치 확인**: 새로운 계층적 로그 구조 적응

### 호환성 보장

Base 시스템은 기존 개별 스크립트와 동일한 결과를 보장합니다:
- 동일한 모델 파라미터
- 동일한 데이터 처리 로직  
- 동일한 평가 메트릭

## 📚 추가 리소스

- **실험 조건 예시**: `base-exp_condition1.json` 참고
- **로그 분석 도구**: `examples/hdmap/analyze_experiment_results.py`
- **TensorBoard 시각화**: `tensorboard --logdir results/*/tensorboard_logs`

---

이 통합 시스템을 통해 **단일 코드베이스**로 모든 anomaly detection 모델의 실험을 효율적으로 관리하고, 일관된 성능 비교를 수행할 수 있습니다.