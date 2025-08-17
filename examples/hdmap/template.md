# 멀티 도메인 머신러닝 실험 템플릿

이 문서는 HDMAP 데이터셋에서 다양한 Anomaly Detection 모델을 위한 표준화된 실험 템플릿을 제공합니다.

## 📁 파일 구조

각 모델은 **Python 스크립트**와 **실행 스크립트** 한 세트로 구성됩니다:

```
examples/hdmap/
├── multi_domain_hdmap_{MODEL_NAME}_training.py       # Python 실험 스크립트
├── multi_domain_hdmap_{MODEL_NAME}_training_run.sh   # Bash 병렬 실행 스크립트
├── experiment_utils.py                               # 공통 유틸리티 함수들
└── template.md                                       # 이 문서
```

## 🎯 템플릿 구조

### 1. Python 실험 스크립트 (`multi_domain_hdmap_{MODEL_NAME}_training.py`)

#### 1.1 파일 헤더 및 Docstring
```python
#!/usr/bin/env python3
"""HDMAP 다중 도메인 {MODEL_NAME} 모델 훈련 및 평가 스크립트.

이 스크립트는 HDMAP 데이터셋에서 {MODEL_NAME} 모델을 훈련하고 다중 도메인 평가를 수행합니다.

주요 기능:
- {MODEL_NAME} 모델을 사용한 이상 탐지
- 소스 도메인(domain_A)에서 훈련
- 타겟 도메인들(domain_B, C, D)에서 평가
- 실험 결과 시각화 및 저장
- 체계적인 실험 조건 관리

사용법:
    python multi_domain_hdmap_{MODEL_NAME}_training.py --experiment_name my_experiment --max_epochs 50
    python multi_domain_hdmap_{MODEL_NAME}_training.py --run_all_experiments
"""
```

#### 1.2 필수 Import 구조
```python
import argparse
import json
import logging
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# 모델별 특화 import
from anomalib.models.image.{MODEL_NAME} import {ModelClass}
from anomalib.data.datamodules.image.multi_domain_hdmap import MultiDomainHDMAPDataModule
from anomalib.engine import Engine
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# 공통 유틸리티 함수들 import
from experiment_utils import (
    setup_warnings_filter,
    setup_experiment_logging,
    cleanup_gpu_memory,
    create_multi_domain_datamodule,
    evaluate_source_domain,
    evaluate_target_domains,
    extract_training_info,
    create_experiment_visualization,
    organize_source_domain_results,
    save_experiment_results,
    analyze_multi_experiment_results
)

# 경고 메시지 비활성화
setup_warnings_filter()
logging.getLogger("anomalib.visualization.image.item_visualizer").setLevel(logging.ERROR)
```

#### 1.3 실험 조건 정의 (`EXPERIMENT_CONDITIONS`)
```python
# {MODEL_NAME} 실험 조건들 정의
EXPERIMENT_CONDITIONS = [
    {
        "name": "{MODEL_NAME}_baseline",
        "description": "{MODEL_NAME} 기본 설정",
        "config": {
            "max_epochs": 30,
            "early_stopping_patience": 5,
            "learning_rate": 1e-4,
            "batch_size": 32,
            "image_size": "224x224"
        }
        # 모델별 특화 파라미터 추가
    },
    # 추가 실험 조건들...
]
```

#### 1.4 핵심 함수들

##### A. 모델 훈련 함수
```python
def train_{MODEL_NAME}_model_multi_domain(
    datamodule: MultiDomainHDMAPDataModule, 
    config: Dict[str, Any],
    results_base_dir: str,
    experiment_name: str,
    logger: logging.Logger
) -> tuple[{ModelClass}, Engine, str]:
    """
    {MODEL_NAME} 모델 훈련 수행.
    
    Returns:
        tuple: (훈련된 모델, Engine 객체, 체크포인트 경로)
    """
    # 1. 모델 초기화
    model = {ModelClass}()
    
    # 2. Early stopping과 model checkpoint 설정
    early_stopping = EarlyStopping(
        monitor="val_loss",  # 또는 모델별 적절한 메트릭
        patience=config["early_stopping_patience"],
        mode="min",
        verbose=True
    )
    
    checkpoint_callback = ModelCheckpoint(
        filename=f"{MODEL_NAME}_multi_domain_{datamodule.source_domain}_" + "{epoch:02d}_{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True
    )
    
    # 3. TensorBoard 로거 설정
    tb_logger = TensorBoardLogger(
        save_dir=results_base_dir,
        name="tensorboard_logs",
        version=""  # 빈 버전으로 version_x 폴더 방지
    )
    
    # 4. Engine 생성 및 훈련
    engine = Engine(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[0] if torch.cuda.is_available() else 1,
        logger=tb_logger,
        max_epochs=config["max_epochs"],
        callbacks=[early_stopping, checkpoint_callback],
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
        log_every_n_steps=10,
        enable_model_summary=True,
        num_sanity_val_steps=0,
        default_root_dir=results_base_dir
    )
    
    # 5. 모델 훈련
    engine.fit(model=model, datamodule=datamodule)
    
    return model, engine, checkpoint_callback.best_model_path
```

##### B. 결과 분석 함수
```python
def analyze_{MODEL_NAME}_results(
    source_results: Dict[str, Any],
    target_results: Dict[str, Dict[str, Any]],
    training_info: Dict[str, Any],
    condition: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Any]:
    """모델별 특화 결과 분석 로직"""
    # 평균 타겟 AUROC 계산 등
    pass
```

##### C. 단일 실험 실행 함수
```python
def run_single_{MODEL_NAME}_experiment(
    condition: Dict[str, Any],
    source_domain: str = "domain_A",
    target_domains: List[str] = None,
    dataset_root: str = None,
    results_base_dir: str = "./results",
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """단일 {MODEL_NAME} 실험 조건에 대한 실험 수행."""
    
    try:
        # 1. GPU 메모리 정리
        cleanup_gpu_memory()
        
        # 2. 실험별 결과 디렉토리 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = f"{experiment_name}_{timestamp}"
        experiment_dir = Path(results_base_dir) / "MultiDomainHDMAP" / "{MODEL_NAME}" / experiment_folder
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. DataModule 생성
        datamodule = create_multi_domain_datamodule(...)
        
        # 4. 모델 훈련
        model, engine, best_checkpoint = train_{MODEL_NAME}_model_multi_domain(...)
        
        # 5. 훈련 정보 추출
        training_info = extract_training_info(engine)
        
        # 6. 성능 평가
        source_results = evaluate_source_domain(...)
        target_results = evaluate_target_domains(...)
        
        # 7. 시각화 생성
        viz_path = create_experiment_visualization(...)
        
        # 8. 결과 분석
        analysis = analyze_{MODEL_NAME}_results(...)
        
        # 9. 실험 결과 정리 및 JSON 저장
        experiment_result = {
            "condition": condition,
            "experiment_name": experiment_name, 
            "source_results": source_results,
            "target_results": target_results,
            "best_checkpoint": best_checkpoint,
            "training_info": training_info,
            "status": "success",
            "experiment_path": str(latest_version_path),
            "avg_target_auroc": analysis["avg_target_auroc"]
        }
        
        # JSON 결과 파일 저장 (각 실험의 tensorboard_logs 폴더에)
        result_filename = f"result_{condition['name']}_{timestamp}.json"
        result_path = latest_version_path / result_filename
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_result, f, indent=2, ensure_ascii=False)
        
        return experiment_result
        
    except Exception as e:
        return {
            "status": "failed",
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "experiment_name": experiment_name,
            "condition": condition,
            "error": str(e)
        }
```

#### 1.5 메인 함수
```python
def main():
    """메인 함수 - 실험 설정 및 실행."""
    parser = argparse.ArgumentParser(description="HDMAP 다중 도메인 {MODEL_NAME} 실험")
    
    # 공통 인자들
    parser.add_argument("--experiment_name", type=str, help="실험 이름")
    parser.add_argument("--max_epochs", type=int, default=30, help="최대 에폭 수")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="학습률")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--source_domain", type=str, default="domain_A", help="소스 도메인")
    parser.add_argument("--dataset_root", type=str, help="데이터셋 루트 경로")
    parser.add_argument("--results_dir", type=str, default="./results", help="결과 저장 디렉토리")
    parser.add_argument("--run_all_experiments", action="store_true", help="모든 실험 조건 실행")
    parser.add_argument("--log_level", type=str, default="INFO", help="로그 레벨")
    
    args = parser.parse_args()
    
    # 로그 설정 (공통 패턴)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # results_dir에서 timestamp 추출하여 통합 로그 파일 생성
    import re
    dir_parts = str(results_dir).split('/')
    run_timestamp = None
    for part in dir_parts:
        if re.match(r'\d{8}_\d{6}', part):
            run_timestamp = part
            break
    
    if not run_timestamp:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = results_dir / f"{MODEL_NAME}_experiment_{run_timestamp}.log"
    logger = setup_experiment_logging(str(log_file), f"{MODEL_NAME}_experiment")
    
    # 실험 실행 로직
    all_results = []
    
    if args.run_all_experiments:
        # 모든 실험 조건 실행
        for condition in EXPERIMENT_CONDITIONS:
            result = run_single_{MODEL_NAME}_experiment(...)
            all_results.append(result)
    else:
        # 단일 실험 실행
        result = run_single_{MODEL_NAME}_experiment(...)
        all_results.append(result)
    
    # 다중 실험 분석
    if len(all_results) > 1:
        analyze_multi_experiment_results(all_results, args.source_domain)


if __name__ == "__main__":
    main()
```

### 2. Bash 실행 스크립트 (`multi_domain_hdmap_{MODEL_NAME}_training_run.sh`)

#### 2.1 스크립트 헤더
```bash
#!/bin/bash

# {MODEL_NAME} 병렬 실험 실행 스크립트
# 멀티 GPU를 활용하여 실험 조건을 병렬로 실행
```

#### 2.2 실험 설정
```bash
# 설정 (사용 가능한 GPU와 실험 조건들)
AVAILABLE_GPUS=(3 4 5 6 7 8)  # 사용할 GPU 목록
EXPERIMENT_CONDITIONS=(
    "{MODEL_NAME}_baseline"
    "{MODEL_NAME}_variant1"
    "{MODEL_NAME}_variant2"
    # 추가 실험 조건들...
)
NUM_EXPERIMENTS=${#EXPERIMENT_CONDITIONS[@]}

# 로그 디렉토리 생성 (통합 timestamp 폴더)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="results/{MODEL_NAME}/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

SCRIPT_PATH="examples/hdmap/multi_domain_hdmap_{MODEL_NAME}_training.py"
```

#### 2.3 실험 실행 로직
```bash
echo "=================================="
echo "🚀 {MODEL_NAME} 병렬 실험 시작"
echo "=================================="
echo "📁 로그 디렉토리: ${LOG_DIR}"
echo "🖥️  사용 GPU: ${AVAILABLE_GPUS[*]}"
echo "🧪 실험 조건: ${NUM_EXPERIMENTS}개"
echo ""

# 실험 할당 및 실행
echo "📋 실험 할당:"
for i in $(seq 0 $((NUM_EXPERIMENTS-1))); do
    GPU_ID=${AVAILABLE_GPUS[$((i % ${#AVAILABLE_GPUS[@]}))]}
    EXP_NAME=${EXPERIMENT_CONDITIONS[$i]}
    echo "   GPU ${GPU_ID}: 실험 ${i} - ${EXP_NAME}"
done
echo ""

echo "🚀 병렬 실험 시작..."

# 백그라운드로 모든 실험 실행
SUCCESS_COUNT=0
FAILED_COUNT=0
PIDS=()

for i in $(seq 0 $((NUM_EXPERIMENTS-1))); do
    GPU_ID=${AVAILABLE_GPUS[$((i % ${#AVAILABLE_GPUS[@]}))]}
    EXP_NAME=${EXPERIMENT_CONDITIONS[$i]}
    
    echo "[$(date +%H:%M:%S)] 시작: GPU ${GPU_ID} - ${EXP_NAME}"
    
    # 백그라운드로 실험 실행
    cd /home/taewan.hwang/study/anomalib
    uv run "${SCRIPT_PATH}" \
        --gpu-id "${GPU_ID}" \
        --experiment-id "${i}" \
        --log-dir "${LOG_DIR}" \
        > "${LOG_DIR}/output_exp_${i}_gpu${GPU_ID}.log" 2>&1 &
    
    PID=$!
    PIDS[$i]=$PID
    
    sleep 2  # GPU 메모리 충돌 방지
done

# 모든 실험 완료 대기
for i in $(seq 0 $((NUM_EXPERIMENTS-1))); do
    PID=${PIDS[$i]}
    GPU_ID=${AVAILABLE_GPUS[$((i % ${#AVAILABLE_GPUS[@]}))]}
    EXP_NAME=${EXPERIMENT_CONDITIONS[$i]}
    
    # 프로세스 완료 대기
    wait $PID
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] ✅ 완료: GPU ${GPU_ID} - ${EXP_NAME}"
        ((SUCCESS_COUNT++))
    else
        echo "[$(date +%H:%M:%S)] ❌ 실패: GPU ${GPU_ID} - ${EXP_NAME} (종료 코드: ${EXIT_CODE})"
        ((FAILED_COUNT++))
    fi
done

echo ""
echo "=================================="
echo "🎉 모든 실험 완료!"
echo "   성공: ${SUCCESS_COUNT}/${NUM_EXPERIMENTS}"
echo "   실패: ${FAILED_COUNT}/${NUM_EXPERIMENTS}"
echo "   로그 디렉토리: ${LOG_DIR}"
echo "=================================="

# 실패한 실험이 있으면 경고
if [ $FAILED_COUNT -gt 0 ]; then
    echo "⚠️  ${FAILED_COUNT}개 실험이 실패했습니다."
    echo "   로그 파일을 확인하세요: ${LOG_DIR}/"
fi

echo ""
echo "📁 생성된 파일들:"
echo "   실험 로그: ${LOG_DIR}/{MODEL_NAME}_experiment_*.log"
echo "   출력 로그: ${LOG_DIR}/output_exp_*_gpu*.log"
echo "   실험별 폴더: ${LOG_DIR}/MultiDomainHDMAP/{MODEL_NAME}/*/"
echo "   체크포인트: ${LOG_DIR}/MultiDomainHDMAP/{MODEL_NAME}/*/tensorboard_logs/checkpoints/"
echo "   시각화 결과: ${LOG_DIR}/MultiDomainHDMAP/{MODEL_NAME}/*/tensorboard_logs/visualize/"
echo "   JSON 결과: ${LOG_DIR}/MultiDomainHDMAP/{MODEL_NAME}/*/tensorboard_logs/result_*.json"
```

## 🔧 구현 가이드

### 3.1 새 모델 추가 시 체크리스트

1. **파일 생성**
   - [ ] `multi_domain_hdmap_{MODEL_NAME}_training.py` 생성
   - [ ] `multi_domain_hdmap_{MODEL_NAME}_training_run.sh` 생성

2. **Python 스크립트 수정사항**
   - [ ] Docstring에서 `{MODEL_NAME}` 교체
   - [ ] 모델 import 경로 수정: `from anomalib.models.image.{MODEL_NAME} import {ModelClass}`
   - [ ] `EXPERIMENT_CONDITIONS` 모델별 특화 파라미터 정의
   - [ ] `train_{MODEL_NAME}_model_multi_domain` 함수 구현
   - [ ] `analyze_{MODEL_NAME}_results` 함수 구현 (모델별 특화 분석)
   - [ ] `run_single_{MODEL_NAME}_experiment` 함수 구현
   - [ ] Early stopping 메트릭 확인 (`val_loss` vs `val_image_AUROC` 등)

3. **Bash 스크립트 수정사항**
   - [ ] `AVAILABLE_GPUS` 배열 설정
   - [ ] `EXPERIMENT_CONDITIONS` 배열에 모델별 실험 조건 정의
   - [ ] `LOG_DIR` 경로를 `results/{MODEL_NAME}/${TIMESTAMP}` 형식으로 설정
   - [ ] `SCRIPT_PATH` 올바른 Python 스크립트 경로로 설정

### 3.2 공통 유틸리티 활용

모든 모델은 `experiment_utils.py`의 공통 함수들을 최대한 활용:

- `setup_warnings_filter()`: 경고 메시지 필터링
- `setup_experiment_logging()`: 로깅 설정
- `cleanup_gpu_memory()`: GPU 메모리 정리
- `create_multi_domain_datamodule()`: 데이터 모듈 생성
- `evaluate_source_domain()`: 소스 도메인 평가
- `evaluate_target_domains()`: 타겟 도메인들 평가
- `extract_training_info()`: 훈련 정보 추출
- `create_experiment_visualization()`: 시각화 생성
- `analyze_multi_experiment_results()`: 다중 실험 분석

### 3.3 결과 폴더 구조 (표준화)

```
results/{MODEL_NAME}/{TIMESTAMP}/
├── {MODEL_NAME}_experiment_{TIMESTAMP}.log      # 통합 실험 로그 (Python에서 생성)
├── output_exp_0_gpu*.log                        # 개별 실험 출력 (Bash에서 생성)
├── output_exp_1_gpu*.log
├── ...
└── MultiDomainHDMAP/
    └── {MODEL_NAME}/
        ├── {EXPERIMENT_NAME_1}_{TIMESTAMP}/
        │   └── tensorboard_logs/
        │       ├── result_{EXPERIMENT_NAME_1}_{TIMESTAMP}.json   # 실험 결과 JSON
        │       ├── checkpoints/
        │       │   └── {MODEL_NAME}_multi_domain_domain_A_*.ckpt
        │       └── visualize/
        │           ├── source_images/
        │           └── target_images/
        ├── {EXPERIMENT_NAME_2}_{TIMESTAMP}/
        └── ...
```

### 3.4 JSON 결과 파일 표준 구조

```json
{
  "condition": {
    "name": "...",
    "description": "...",
    "config": {...}
  },
  "experiment_name": "domain_A",
  "source_results": {
    "image_AUROC": 0.xxx,
    "image_F1Score": 0.xxx
  },
  "target_results": {
    "domain_B": {...},
    "domain_C": {...},
    "domain_D": {...}
  },
  "best_checkpoint": "/path/to/checkpoint.ckpt",
  "training_info": {
    "max_epochs_configured": 30,
    "last_trained_epoch": 15,
    "early_stopped": true,
    "completion_type": "early_stopping"
  },
  "status": "success",
  "experiment_path": "/path/to/tensorboard_logs",
  "avg_target_auroc": 0.xxx
}
```

## 📋 예시: 새 모델 추가

예를 들어, `PatchCore` 모델을 추가한다면:

1. **파일 생성**:
   - `multi_domain_hdmap_patchcore_training.py`
   - `multi_domain_hdmap_patchcore_training_run.sh`

2. **주요 변경사항**:
   ```python
   # Import 변경
   from anomalib.models.image.patchcore import PatchCore
   
   # 함수명 변경
   def train_patchcore_model_multi_domain(...):
   def analyze_patchcore_results(...):
   def run_single_patchcore_experiment(...):
   
   # 실험 조건 정의
   EXPERIMENT_CONDITIONS = [
       {
           "name": "patchcore_baseline",
           "description": "PatchCore 기본 설정",
           "config": {
               "backbone": "wide_resnet50_2",
               "pre_trained": True,
               "layers": ["layer2", "layer3"],
               "coreset_sampling_ratio": 0.1,
               # PatchCore 특화 파라미터들...
           }
       }
   ]
   ```

3. **Bash 스크립트**:
   ```bash
   LOG_DIR="results/patchcore/${TIMESTAMP}"
   SCRIPT_PATH="examples/hdmap/multi_domain_hdmap_patchcore_training.py"
   EXPERIMENT_CONDITIONS=(
       "patchcore_baseline"
       "patchcore_variant1"
       # ...
   )
   ```

이 템플릿을 따르면 **일관된 실험 환경**과 **표준화된 결과 구조**를 유지하면서 다양한 모델의 실험을 효율적으로 수행할 수 있습니다.
