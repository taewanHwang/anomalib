# Multi-Domain Anomaly Detection 통합 실험 가이드

## 🎯 개요

이 디렉토리는 통합된 Base 시스템을 통해 모든 anomaly detection 모델의 **다중 도메인 전이학습** 실험을 효율적으로 관리합니다. Single domain과 달리 source domain에서 훈련하고 multiple target domains에서 평가하여 도메인 간 전이 성능을 측정합니다.

## 📁 파일 구조

```
multi_domain/
├── README.md                          # 이 파일 - 사용 가이드
├── anomaly_trainer.py                 # MultiDomainAnomalyTrainer 클래스 (핵심)
├── base-training.py                   # 메인 실행 스크립트
├── base-run.sh                        # 병렬 실험 실행 스크립트
└── base-exp_condition1.json          # 실험 조건 설정 파일
```

## 🚀 빠른 시작

### 단일 실험 실행
```bash
# 특정 실험 ID 실행 (예: ID 0번 실험)
python examples/hdmap/multi_domain/base-training.py \
    --config examples/hdmap/multi_domain/base-exp_condition1.json \
    --experiment-id 0 \
    --gpu-id 0
```

### 모든 실험 병렬 실행 (추천)
```bash
# 백그라운드에서 모든 실험 실행
nohup ./examples/hdmap/multi_domain/base-run.sh all > multi_domain_training.log 2>&1 &

# 실행 상태 확인
tail -f multi_domain_training.log                    # 메인 스크립트 진행 상황
tail -f results/*/multi_domain_*.log                 # 개별 실험 상세 로그
tail -f results/*/training_detail.log                # 개별 실험 훈련 로그
```

## 🏗️ 시스템 아키텍처

### 핵심 컴포넌트

#### 1. MultiDomainAnomalyTrainer (`anomaly_trainer.py`)
모든 모델의 multi-domain 로직을 처리하는 통합 클래스:

```python
class MultiDomainAnomalyTrainer:
    def __init__(self, config, experiment_name, session_timestamp, experiment_dir):
        self.source_domain = config["source_domain"]       # 훈련 도메인
        self.target_domains = config["target_domains"]     # 평가 도메인들
    
    def run_experiment(self):
        # 1. Source domain에서 모델 훈련
        # 2. Source domain test 평가 (validation 역할)
        # 3. Target domains test 평가 (전이 성능 측정)
        # 4. Source/Target 시각화 결과 분리
```

#### 2. 실험 조건 설정 (`base-exp_condition1.json`)
모든 모델의 multi-domain 실험 설정을 JSON 형태로 통합 관리:

```json
{
  "experiment_conditions": [
    {
      "name": "domainA_to_BCD_draem_baseline",
      "description": "Domain A → B,C,D DRAEM 전이학습",
      "config": {
        "model_type": "draem",
        "source_domain": "domain_A",              // 훈련 도메인
        "target_domains": ["domain_B", "domain_C", "domain_D"],  // 평가 도메인들
        "max_epochs": 50,
        "learning_rate": 0.0001,
        "batch_size": 16
      }
    }
  ]
}
```

### 지원 모델

현재 지원하는 multi-domain anomaly detection 모델:

| 모델 | model_type | 전이학습 특징 | 추천 설정 |
|------|------------|---------------|-----------|
| **DRAEM** | `"draem"` | Source에서 reconstruction 학습 후 target 평가 | batch_size: 16, epochs: 50 |
| **Dinomaly** | `"dinomaly"` | DINOv2 backbone으로 안정적인 전이 | batch_size: 8, 메모리 집약적 |
| **PatchCore** | `"patchcore"` | Memory bank 기반 빠른 전이 | epochs: 1, 추론 중심 |
| **DRAEM-SevNet** | `"draem_sevnet"` | Severity head로 정교한 전이학습 | batch_size: 16, 복합 loss |

## 📊 Multi-Domain vs Single Domain 비교

### 주요 차이점

| 구분 | Single Domain | Multi Domain |
|------|---------------|--------------|
| **데이터 모듈** | `SingleDomainHDMAPDataModule` | `MultiDomainHDMAPDataModule` |
| **훈련 방식** | 1개 도메인 내 train/val 분리 | Source domain 전체로 훈련 |
| **Validation** | Source train의 일부 (val_split_ratio) | Source domain test 전체 |
| **Test** | Source domain test | Target domains test (각각) |
| **평가 메트릭** | `test_image_AUROC` (1개) | `val_image_AUROC` (source) + target별 AUROC |
| **시각화** | 단일 폴더 | source/ + targets/ 분리 |
| **결과 의미** | 도메인 내 성능 | 도메인 간 전이 성능 |

### 로그 구조 및 모니터링

실험 실행 시 다음과 같은 계층적 로그 구조가 생성됩니다:

```
results/20250831_120000/                              # 세션 타임스탬프
├── domainA_to_BCD_draem_baseline_20250831_120000/    # 실험별 디렉터리
│   ├── multi_domain_domainA_to_BCD_draem_baseline.log    # 구조화된 결과 로그
│   ├── training_detail.log                           # 상세 훈련 로그 (tqdm, 에러 등)
│   └── tensorboard_logs/                            # TensorBoard 로그
│       ├── result_*.json                            # JSON 결과 파일
│       ├── checkpoints/                             # 모델 체크포인트
│       └── visualizations/                          # 시각화 결과
│           ├── source/                              # Source domain 시각화
│           └── targets/                             # Target domains 시각화
└── ...
```

### 로그 모니터링 명령어

```bash
# 전체 실험 진행 상황
tail -f multi_domain_training.log

# 특정 모델의 상세 훈련 로그 (실시간 tqdm 바 확인)
tail -f results/20250831_*/domainA_to_BCD_*/training_detail.log

# 가장 최근 실험의 구조화된 로그
tail -f $(ls -t results/*/multi_domain_*.log | head -1)

# 모든 실험의 최종 결과 확인
grep -r "AUROC" results/*/multi_domain_*.log
grep -r "Target.*AUROC" results/*/tensorboard_logs/result_*.json
```

## ⚙️ 실험 설정 커스터마이징

### 새로운 Multi-Domain 실험 조건 추가

`base-exp_condition1.json`에 새로운 실험 조건을 추가:

```json
{
  "name": "domainB_to_ACD_patchcore_optimized",
  "description": "Domain B → A,C,D PatchCore 역전이 학습",
  "config": {
    "model_type": "patchcore",
    "source_domain": "domain_B",         // 다른 소스 도메인
    "target_domains": ["domain_A", "domain_C", "domain_D"],
    "max_epochs": 1,
    "batch_size": 32,
    "coreset_sampling_ratio": 0.15,
    "num_neighbors": 12
  }
}
```

### 모델별 Multi-Domain 최적 파라미터

#### DRAEM (Domain Transfer)
- `batch_size`: 16 (안정적 전이)
- `learning_rate`: 0.0001 (보수적)
- `max_epochs`: 50 (충분한 학습)
- `early_stopping_patience`: 10

#### Dinomaly (Vision Transformer Transfer)  
- `batch_size`: 8 (메모리 제약)
- `encoder_name`: `"dinov2reg_vit_base_14"`
- `target_layers`: `[2,3,4,5,6,7,8,9]`
- `learning_rate`: 0.0001 (안정적)

#### PatchCore (Memory Bank Transfer)
- `batch_size`: 32 (빠른 피팅)
- `max_epochs`: 1 (피팅만)
- `coreset_sampling_ratio`: 0.1 (효율성)
- `num_neighbors`: 9

#### DRAEM-SevNet (Severity-aware Transfer)
- `batch_size`: 16
- `score_combination`: `"weighted_average"`
- `severity_loss_type`: `"smooth_l1"`
- `severity_head_pooling_type`: `"gap"`

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
./examples/hdmap/multi_domain/base-run.sh 2     # ID 2번 실험만

# 여러 실험 선택 실행
./examples/hdmap/multi_domain/base-run.sh 0,5,10  # ID 0, 5, 10번 실험

# 실험 진행 상황 확인
ps aux | grep base-training                    # 실행 중인 실험들
nvidia-smi                                     # GPU 사용 현황
```

### Domain Transfer 전략

Multi-domain 실험에서는 다양한 전이학습 전략을 테스트할 수 있습니다:

#### 1. 단방향 전이 (Unidirectional Transfer)
```json
{
  "source_domain": "domain_A",
  "target_domains": ["domain_B", "domain_C", "domain_D"]
}
```

#### 2. 역방향 전이 (Reverse Transfer) 
```json
{
  "source_domain": "domain_B",
  "target_domains": ["domain_A", "domain_C", "domain_D"]
}
```

#### 3. 선택적 전이 (Selective Transfer)
```json
{
  "source_domain": "domain_A", 
  "target_domains": ["domain_C"]  // 특정 도메인만 평가
}
```

## 📈 결과 분석

### Multi-Domain 실험 결과 비교

각 실험 완료 후 다음 파일들을 확인:

```bash
# JSON 결과 파일 (source + targets 포함)
ls results/*/tensorboard_logs/result_*.json

# Source domain 성능 확인
grep -r "val_image_AUROC" results/*/tensorboard_logs/result_*.json

# Target domains별 전이 성능 비교
grep -r "target_results" results/*/tensorboard_logs/result_*.json | jq .

# 전이 효과 분석 (Transfer Ratio)
python examples/hdmap/analyze_experiment_results.py --results_dir results --all-models

# TensorBoard로 훈련 과정 시각화 (multi-domain 메트릭 포함)
tensorboard --logdir results/20250831_*/tensorboard_logs
```

### Domain Transfer 성능 벤치마크 예시

| Source → Targets | DRAEM | Dinomaly | PatchCore | DRAEM-SevNet |
|------------------|-------|----------|-----------|--------------|
| **A → B,C,D** | | | | |
| Source (A) AUROC | 0.82 | 0.88 | 0.85 | 0.84 |
| Target B AUROC | 0.75 | 0.82 | 0.80 | 0.77 |
| Target C AUROC | 0.73 | 0.85 | 0.82 | 0.76 |
| Target D AUROC | 0.71 | 0.81 | 0.79 | 0.74 |
| **평균 Transfer** | **0.73** | **0.83** | **0.80** | **0.76** |
| Transfer Ratio | 0.89 | 0.94 | 0.94 | 0.90 |

### Transfer Learning 분석 지표

Multi-domain 실험 결과에서 확인할 수 있는 주요 지표들:

#### 1. **Source Performance** 
- Source domain에서의 성능 (baseline)
- `val_image_AUROC` 메트릭

#### 2. **Target Performance**
- 각 target domain에서의 전이 성능  
- Target별 `test_image_AUROC` 메트릭

#### 3. **Transfer Ratio**
- 전이 효율성: `avg_target_auroc / source_auroc`
- 1.0에 가까울수록 완전한 전이

#### 4. **Domain Gap**
- Source와 각 target 간 성능 차이
- 작을수록 도메인 유사성 높음

## 🛠️ 문제 해결

### 일반적인 문제들

#### 1. Multi-Domain DataModule 오류
```bash
# MultiDomainHDMAPDataModule import 실패
pip install -e .  # anomalib 재설치
```

#### 2. Target Domain 평가 실패
```bash
# Target domains 설정 확인
python -c "
import json
with open('examples/hdmap/multi_domain/base-exp_condition1.json') as f:
    data = json.load(f)
print(data['experiment_conditions'][0]['config']['target_domains'])
"
```

#### 3. 메모리 부족 (Multi-domain은 더 많은 메모리 사용)
```bash
# Batch size 줄이기 (특히 Dinomaly)
"batch_size": 4  # 8에서 4로 변경
```

#### 4. Transfer 성능 저조
```bash
# Early stopping patience 증가
"early_stopping_patience": 15  # 10에서 15로 증가

# 학습률 조정
"learning_rate": 0.00005  # 더 보수적인 학습
```

## 🔄 Single Domain에서 Multi Domain으로 마이그레이션

### 기존 Single Domain 설정을 Multi Domain으로 변환

#### 1. 설정 변환 예시
```json
// Single Domain 설정
{
  "model_type": "draem",
  "source_domain": "domain_A",
  "val_split_ratio": 0.1  // 제거됨
}

// Multi Domain 설정으로 변환
{
  "model_type": "draem", 
  "source_domain": "domain_A",
  "target_domains": ["domain_B", "domain_C", "domain_D"]  // 추가됨
}
```

#### 2. 평가 메트릭 변화 이해
- Single: `test_image_AUROC` (1개 값)
- Multi: `val_image_AUROC` (source) + target별 `test_image_AUROC` (N개 값)

#### 3. 결과 해석 방법 변화
- Single: 절대 성능 중심
- Multi: 전이 성능 및 도메인 간 일반화 능력 중심

## 📚 추가 리소스

- **실험 조건 예시**: `base-exp_condition1.json` 참고 (16개 다양한 조건)
- **결과 분석 도구**: `examples/hdmap/analyze_experiment_results.py --all-models`
- **TensorBoard 시각화**: Multi-domain 메트릭 및 전이 학습 과정
- **Single Domain 비교**: `examples/hdmap/single_domain/README.md`

---

이 Multi-Domain 통합 시스템을 통해 **source domain 훈련 → multiple target domains 평가**로 모든 anomaly detection 모델의 도메인 전이 성능을 체계적으로 비교하고 분석할 수 있습니다.

🎯 **핵심 가치**: Single codebase로 모든 모델의 multi-domain transfer learning 실험을 효율적으로 관리하고, 도메인 간 일반화 성능을 정량적으로 평가할 수 있습니다.