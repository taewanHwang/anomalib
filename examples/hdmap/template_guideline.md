# 📋 MultiDomain HDMAP 모델별 학습 코드 구현 템플릿 가이드라인

## 📖 개요

이 문서는 HDMAP 데이터셋에서 다양한 Vision Anomaly Detection 모델들을 위한 **통일된 학습 코드 구현 템플릿**을 제공합니다. 

기존 DRAEM과 DRAEM-SevNet 구현을 기반으로 하여, 새로운 모델들도 동일한 실험 흐름과 평가 체계를 따를 수 있도록 설계되었습니다.

---

## 🏗️ 기본 코드 구조 템플릿

### 📁 파일 명명 규칙
```
multi_domain_hdmap_{모델명}-training.py
multi_domain_hdmap_{모델명}-exp_condition.json
```

**예시:**
- `multi_domain_hdmap_patchcore-training.py`
- `multi_domain_hdmap_padim-training.py`
- `multi_domain_hdmap_reverse_distillation-training.py`

### 📝 파일 헤더 템플릿
```python
#!/usr/bin/env python3
"""MultiDomain HDMAP {모델명} 도메인 전이 학습 예시.

{모델명} 모델과 MultiDomainHDMAPDataModule을 활용한 효율적인 도메인 전이 학습 실험 스크립트입니다.

{모델명} 특징:
- [모델별 핵심 특징 1]
- [모델별 핵심 특징 2]
- [모델별 핵심 특징 3]
- [모델별 특화 기능들]

실험 구조:
1. MultiDomainHDMAPDataModule 설정 (e.g. source: domain_A, targets: domain_B,C,D)
2. Source Domain에서 {모델명} 모델 훈련 (train 데이터)
3. Source Domain에서 성능 평가 (validation으로 사용될 test 데이터)
4. Target Domains에서 동시 성능 평가 (각 도메인별 test 데이터)
5. 도메인 전이 효과 종합 분석

주요 개선점 ({모델명} vs 기준 모델):
- [성능 개선점 1]
- [성능 개선점 2]

NOTE:
- 실험 조건들은 multi_domain_hdmap_{모델명}_exp_condition.json 파일에서 관리됩니다.
- 코드 유지보수성을 위해 실험 설정과 실행 로직을 분리했습니다.
"""
```

---

## 🔧 필수 Import 블록 템플릿

```python
import os
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import logging
import warnings
import argparse

# MultiDomain HDMAP import
from anomalib.data.datamodules.image.multi_domain_hdmap import MultiDomainHDMAPDataModule
from anomalib.models.image.{모델명} import {모델클래스명}  # 모델별 수정
from anomalib.engine import Engine
from pytorch_lightning.loggers import TensorBoardLogger

# Early Stopping import (모델이 학습을 요구하는 경우만)
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# 공통 유틸리티 함수들 import
from experiment_utils import (
    cleanup_gpu_memory,
    setup_warnings_filter,
    setup_experiment_logging,
    extract_training_info,
    organize_source_domain_results,
    evaluate_target_domains,
    save_experiment_results,
    create_experiment_visualization,
    create_multi_domain_datamodule,
    evaluate_source_domain,
    load_experiment_conditions,
    analyze_experiment_results,
    extract_target_domains_from_config
)
```

---

## 📊 모델 훈련 함수 템플릿

### 🎯 기본 템플릿 구조

```python
def train_{모델명}_model_multi_domain(
    datamodule: MultiDomainHDMAPDataModule, 
    config: Dict[str, Any],
    results_base_dir: str,
    logger: logging.Logger
) -> tuple[{모델클래스명}, Engine, str]:
    """
    {모델명} 모델 훈련 수행.
    
    Args:
        datamodule: 설정된 MultiDomainHDMAPDataModule
        config: 훈련 설정 딕셔너리
        results_base_dir: 결과 저장 기본 경로
        logger: 로거 객체
        
    Returns:
        tuple: (훈련된 모델, Engine 객체, 체크포인트 경로)
        
    Note:
        {모델명} 특징:
        - [모델별 핵심 특징들 나열]
    """
    
    print(f"\n🚀 {모델명} 모델 훈련 시작")
    logger.info("🚀 {모델명} 모델 훈련 시작")
    
    # 🎯 모델별 특화 설정 출력
    print(f"   🔧 Config 설정:")
    # [모델별 중요한 config 파라미터들 출력]
    
    logger.info("✅ {모델명} 모델 생성 완료")
    logger.info(f"🔧 Config 설정: [주요 설정들]")
    
    # 🎯 모델 생성 (모델별 특화)
    model = {모델클래스명}(
        # [모델별 필수/선택 파라미터들]
    )
    
    # 🎯 콜백 설정 (모델별 조건부)
    callbacks = []
    
    # [학습이 필요한 모델의 경우]
    if {모델이_학습을_요구하는가}:
        early_stopping = EarlyStopping(
            monitor="val_image_AUROC",
            patience=config["early_stopping_patience"],
            mode="max",
            verbose=True
        )
        
        checkpoint_callback = ModelCheckpoint(
            filename=f"{모델명}_multi_domain_{datamodule.source_domain}_" + "{epoch:02d}_{val_image_AUROC:.4f}",
            monitor="val_image_AUROC",
            mode="max",
            save_top_k=1,
            verbose=True
        )
        
        callbacks.extend([early_stopping, checkpoint_callback])
    
    # 🎯 TensorBoard 로거 설정
    tb_logger = TensorBoardLogger(
        save_dir=results_base_dir,
        name="tensorboard_logs",
        version=""
    )
    
    # 🎯 Engine 설정
    engine_kwargs = {
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": [0] if torch.cuda.is_available() else 1,
        "logger": tb_logger,
        "callbacks": callbacks,
        "enable_checkpointing": len(callbacks) > 0,
        "log_every_n_steps": 10,
        "enable_model_summary": True,
        "default_root_dir": results_base_dir
    }
    
    # [학습이 필요한 모델의 경우만]
    if {모델이_학습을_요구하는가}:
        engine_kwargs.update({
            "max_epochs": config["max_epochs"],
            "check_val_every_n_epoch": 1,
            "num_sanity_val_steps": 0
        })
    
    engine = Engine(**engine_kwargs)
    
    print(f"   🔧 Engine 설정 완료")
    print(f"   📁 결과 저장 경로: {results_base_dir}")
    logger.info(f"🔧 Engine 설정 완료")
    logger.info(f"📁 결과 저장 경로: {results_base_dir}")
    
    # 🎯 모델 훈련/피팅 (모델별 분기)
    if {모델이_학습을_요구하는가}:
        print(f"   🎯 모델 훈련 시작...")
        logger.info("🎯 모델 훈련 시작...")
        
        engine.fit(
            model=model,
            datamodule=datamodule
        )
        
        best_checkpoint = checkpoint_callback.best_model_path
        print(f"   🏆 Best Checkpoint: {best_checkpoint}")
        logger.info(f"🏆 Best Checkpoint: {best_checkpoint}")
    else:
        print(f"   🎯 모델 피팅 시작... (학습 불필요)")
        logger.info("🎯 모델 피팅 시작... (학습 불필요)")
        
        engine.fit(
            model=model,
            datamodule=datamodule
        )
        
        best_checkpoint = None  # 학습이 없는 모델은 체크포인트 없음
        print(f"   ✅ 모델 피팅 완료!")
        logger.info("✅ 모델 피팅 완료!")
    
    print(f"   ✅ 모델 훈련/피팅 완료!")
    logger.info("✅ 모델 훈련/피팅 완료!")
    
    return model, engine, best_checkpoint
```

---

## 🧪 실험 실행 함수 템플릿

```python
def run_single_{모델명}_experiment(
    condition: dict,
    log_dir: str = None
) -> dict:
    """단일 {모델명} 실험 조건에 대한 실험 수행."""
    
    # config에서 도메인 설정 가져오기
    config = condition["config"]
    source_domain = config["source_domain"]
    target_domains = extract_target_domains_from_config(config)
    
    # 실험 경로 설정 (기존 패턴 유지)
    from datetime import datetime
    if log_dir:
        base_timestamp_dir = log_dir
        timestamp_for_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = f"{condition['name']}_{timestamp_for_folder}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_timestamp_dir = f"results/{모델명}/{timestamp}"
        experiment_folder = f"{condition['name']}_{timestamp}"
    
    results_base_dir = f"{base_timestamp_dir}/MultiDomainHDMAP/{모델명}/{experiment_folder}"
    
    # 실험 이름 생성
    experiment_name = f"{source_domain}"
    
    print(f"\n{'='*80}")
    print(f"🔬 {모델명} 실험 조건: {condition['name']}")
    print(f"📝 설명: {condition['description']}")
    print(f"{'='*80}")
    
    try:
        # GPU 메모리 정리
        cleanup_gpu_memory()
        
        # DataModule 생성
        multi_datamodule = create_multi_domain_datamodule(
            datamodule_class=MultiDomainHDMAPDataModule,
            source_domain=source_domain,
            target_domains=target_domains,
            batch_size=config["batch_size"],
            image_size=config["image_size"]
        )
        
        # 모델 훈련
        trained_model, engine, best_checkpoint = train_{모델명}_model_multi_domain(
            datamodule=multi_datamodule,
            config=condition["config"],
            results_base_dir=results_base_dir,
            logger=logging.getLogger(__name__)
        )
        
        # [나머지 평가 및 저장 로직은 기존 패턴과 동일]
        # ...
        
    except Exception as e:
        # [에러 처리 로직]
        # ...
```

---

## 📋 모델별 특화 구현 가이드

### 🎯 Step 1: 모델 특성 파악

각 모델을 구현하기 전에 다음 사항들을 반드시 확인하세요:

#### A. 학습 요구사항
```python
# 확인해야 할 사항들:
학습이_필요한가 = True/False  # 예: DRAEM(True), PatchCore(False)
early_stopping_필요 = True/False
gradient_clipping_필요 = True/False
특별한_trainer_arguments = {...}
```

#### B. 모델 초기화 파라미터
```python
# 각 모델의 __init__ 파라미터들을 확인하고 config에서 제공해야 할 것들:
필수_파라미터 = [...]
선택_파라미터 = [...]
기본값_사용_파라미터 = [...]
```

#### C. 모델별 특화 설정
```python
# 예시:
# PatchCore: coreset_sampling_ratio, num_neighbors, backbone, layers
# Padim: backbone, layers, pre_trained
# ReverseDistillation: backbone, layers, lr, momentum
```

### 🎯 Step 2: JSON 설정 파일 생성

```json
{
  "condition_1": {
    "name": "{모델명}_baseline",
    "description": "{모델명} 기본 설정으로 실험",
    "config": {
      "source_domain": "domain_A",
      "target_domains": ["domain_B", "domain_C", "domain_D"],
      "batch_size": 32,
      "image_size": [256, 256],
      
      // 모델별 특화 파라미터들
      "{모델별_파라미터1}": "값1",
      "{모델별_파라미터2}": "값2",
      
      // 학습이 필요한 모델의 경우
      "max_epochs": 50,
      "early_stopping_patience": 10,
      "learning_rate": 0.001,
      "optimizer": "adamw",
      "weight_decay": 0.0001
    }
  }
}
```

### 🎯 Step 3: 모델별 수정 포인트

#### A. Import 구문 수정
```python
from anomalib.models.image.{모델명} import {모델클래스명}
```

#### B. 모델 생성 코드 수정
```python
# 예시 - PatchCore
model = Patchcore(
    backbone=config["backbone"],
    layers=config["layers"],
    coreset_sampling_ratio=config["coreset_sampling_ratio"],
    num_neighbors=config["num_neighbors"]
)

# 예시 - Padim
model = Padim(
    backbone=config["backbone"],
    layers=config["layers"],
    pre_trained=config["pre_trained"]
)
```

#### C. 학습/피팅 로직 분기
```python
if 모델이_학습을_요구하는가:
    # 학습 기반 모델 (DRAEM, ReverseDistillation 등)
    engine.fit(model=model, datamodule=datamodule)
    best_checkpoint = checkpoint_callback.best_model_path
else:
    # 피처 기반 모델 (PatchCore, Padim 등)
    engine.fit(model=model, datamodule=datamodule)
    best_checkpoint = None
```

---

## ⚠️ 주의사항 및 체크리스트

### 🔍 구현 전 체크리스트

- [ ] **모델 문서 확인**: Anomalib 공식 문서에서 모델 초기화 파라미터 확인
- [ ] **학습 요구사항 파악**: 모델이 학습을 필요로 하는지 확인
- [ ] **특화 설정 조사**: 모델별 고유한 설정이나 제약사항 파악
- [ ] **메모리 요구사항**: GPU 메모리 사용량이 큰 모델인지 확인

### 🛠️ 구현 중 체크리스트

- [ ] **Import 경로 정확성**: 모델 클래스 import가 올바른지 확인
- [ ] **Config 파라미터 매핑**: JSON 설정이 모델 초기화 파라미터와 일치하는지 확인
- [ ] **콜백 설정**: 학습이 필요없는 모델에서 EarlyStopping/ModelCheckpoint 제거
- [ ] **로깅 메시지**: 모델명이 로그 메시지에 정확히 반영되었는지 확인

### 🧪 테스트 체크리스트

- [ ] **단일 실험 실행**: 하나의 실험 조건으로 정상 동작 확인
- [ ] **GPU 메모리 정리**: 실험 완료 후 메모리 누수 없는지 확인
- [ ] **결과 파일 생성**: JSON 결과 파일과 시각화 파일이 정상 생성되는지 확인
- [ ] **로그 확인**: 콘솔 출력과 로그 파일에 오류가 없는지 확인

---

## 📚 모델별 특화 가이드

### 🎯 PatchCore 특화사항

```python
# 특징: 학습 불필요, 메모리 뱅크 기반
학습이_필요한가 = False
주요_파라미터 = ["backbone", "layers", "coreset_sampling_ratio", "num_neighbors"]
기본_backbone = "wide_resnet50_2"
기본_layers = ["layer2", "layer3"]
```

### 🎯 Padim 특화사항

```python
# 특징: 학습 불필요, 확률적 임베딩
학습이_필요한가 = False
주요_파라미터 = ["backbone", "layers", "pre_trained"]
기본_backbone = "resnet18"
기본_layers = ["layer1", "layer2", "layer3"]
```

### 🎯 ReverseDistillation 특화사항

```python
# 특징: 학습 필요, 지식 증류
학습이_필요한가 = True
주요_파라미터 = ["backbone", "layers", "anomaly_map_mode"]
특별한_설정 = "teacher-student 구조"
```

---

## 🖥️ 멀티 GPU 병렬 실험 실행 가이드

각 모델별로 **3개의 구성 요소**를 구현해야 합니다:

### 📋 필수 구현 파일들

```
1. multi_domain_hdmap_{모델명}-training.py    # 메인 훈련 스크립트
2. multi_domain_hdmap_{모델명}-exp_condition.json  # 실험 조건 설정
3. multi_domain_hdmap_{모델명}-run.sh          # 멀티 GPU 실행 스크립트
```

### 🎯 Step 1: JSON 실험 조건 파일 구조

```json
{
  "condition_1": {
    "name": "{모델명}_baseline",
    "description": "{모델명} 기본 설정으로 실험",
    "config": {
      "source_domain": "domain_A",
      "target_domains": ["domain_B", "domain_C", "domain_D"],
      "batch_size": 32,
      "image_size": [256, 256],
      
      // 모델별 핵심 파라미터들
      "{모델별_파라미터1}": "값1",
      "{모델별_파라미터2}": "값2",
      
      // 학습이 필요한 모델의 경우
      "max_epochs": 50,
      "early_stopping_patience": 10,
      "learning_rate": 0.001,
      "optimizer": "adamw",
      "weight_decay": 0.0001
    }
  },
  "condition_2": {
    "name": "{모델명}_optimized",
    "description": "{모델명} 최적화 설정으로 실험",
    "config": {
      // ... 다른 파라미터 조합
    }
  }
}
```

### 🎯 Step 2: 멀티 GPU 실행 스크립트 템플릿

**파일명**: `multi_domain_hdmap_{모델명}-run.sh`

```bash
#!/bin/bash
# nohup ./examples/hdmap/multi_domain_hdmap_{모델명}-run.sh > /dev/null 2>&1 &
# pkill -f "multi_domain_hdmap_{모델명}-run.sh"
# pkill -f "examples/hdmap/multi_domain_hdmap_{모델명}-training.py"

# {모델명} 병렬 실험 실행 스크립트
# 멀티 GPU를 활용하여 실험 조건을 병렬로 실행

AVAILABLE_GPUS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
EXPERIMENT_CONDITIONS=(
    "{모델명}_baseline"
    "{모델명}_optimized"
    "{모델명}_condition_3"
    # 필요한 만큼 추가...
)
NUM_EXPERIMENTS=${#EXPERIMENT_CONDITIONS[@]}

# 로그 디렉토리 생성 
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="results/{모델명}/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

SCRIPT_PATH="examples/hdmap/multi_domain_hdmap_{모델명}-training.py"

echo "=================================="
echo "🚀 {모델명} 병렬 실험 시작"
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

# 백그라운드로 모든 실험 시작
pids=()
for i in $(seq 0 $((NUM_EXPERIMENTS-1))); do
    GPU_ID=${AVAILABLE_GPUS[$((i % ${#AVAILABLE_GPUS[@]}))]}
    EXP_NAME=${EXPERIMENT_CONDITIONS[$i]}
    
    echo "🎯 GPU ${GPU_ID}에서 실험 ${i} (${EXP_NAME}) 시작..."
    
    # 각 실험을 백그라운드로 실행
    nohup python ${SCRIPT_PATH} \
        --gpu-id ${GPU_ID} \
        --experiment-id ${i} \
        --log-dir "${LOG_DIR}" \
        > "${LOG_DIR}/output_exp_${i}_gpu${GPU_ID}.log" 2>&1 &
    
    # PID 저장
    pids+=($!)
    
    # GPU간 시작 간격 (GPU 초기화 충돌 방지)
    sleep 5
done

echo ""
echo "✅ 모든 실험이 백그라운드에서 시작되었습니다!"
echo "📊 실시간 모니터링:"
echo "   watch -n 10 'nvidia-smi'"
echo ""
echo "📄 개별 로그 확인:"
for i in $(seq 0 $((NUM_EXPERIMENTS-1))); do
    GPU_ID=${AVAILABLE_GPUS[$((i % ${#AVAILABLE_GPUS[@]}))]}
    echo "   tail -f ${LOG_DIR}/output_exp_${i}_gpu${GPU_ID}.log"
done
echo ""

# 모든 백그라운드 작업 완료 대기
echo "⏳ 모든 실험 완료 대기 중..."
for pid in ${pids[*]}; do
    wait $pid
    echo "✅ 실험 완료: PID $pid"
done

echo ""
echo "🎉 모든 실험이 완료되었습니다!"
echo "📁 결과 위치: ${LOG_DIR}"
echo ""

# 최종 결과 요약
echo "📊 실험 결과 요약:"
for i in $(seq 0 $((NUM_EXPERIMENTS-1))); do
    EXP_NAME=${EXPERIMENT_CONDITIONS[$i]}
    RESULT_FILE="${LOG_DIR}/result_exp_$(printf "%02d" $i)_${EXP_NAME}_gpu*.json"
    if ls ${RESULT_FILE} 1> /dev/null 2>&1; then
        echo "   ✅ ${EXP_NAME}: 성공"
    else
        echo "   ❌ ${EXP_NAME}: 실패 또는 미완료"
    fi
done
```

### 🎯 Step 3: 자동 실험 러너 활용

기존 `auto_experiment_runner.sh`를 활용하여 GPU 모니터링 기반 자동 실행:

```bash
# {모델명} 실험을 자동으로 실행 (GPU 유휴시 시작)
nohup examples/hdmap/auto_experiment_runner.sh \
    -s examples/hdmap/multi_domain_hdmap_{모델명}-run.sh \
    10 > auto_experiment_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 🎯 Step 4: 실험 모니터링 명령어

```bash
# GPU 사용률 실시간 모니터링
watch -n 10 'nvidia-smi'

# 특정 실험 로그 확인
tail -f results/{모델명}/{timestamp}/output_exp_0_gpu0.log

# 실험 진행 상황 확인
ps aux | grep "multi_domain_hdmap_{모델명}-training.py"

# 실험 중단 (필요시)
pkill -f "multi_domain_hdmap_{모델명}-training.py"
```

### 🎯 Step 5: 결과 분석

각 실험 완료 후 자동 생성되는 파일들:

```
results/{모델명}/{timestamp}/
├── output_exp_0_gpu0.log           # 실험 0 로그
├── output_exp_1_gpu1.log           # 실험 1 로그
├── result_exp_00_{조건명}_gpu0.json  # 실험 0 결과
├── result_exp_01_{조건명}_gpu1.json  # 실험 1 결과
└── tensorboard_logs/               # TensorBoard 로그들
```

---

## 🔄 업데이트 가이드

새로운 모델을 추가할 때마다 이 템플릿을 참고하여:

1. **일관된 코드 구조** 유지
2. **동일한 실험 흐름** 보장  
3. **호환 가능한 결과 형식** 생성
4. **유지보수 용이성** 확보
5. **멀티 GPU 병렬 실행** 지원

이를 통해 모든 모델들이 **동일한 비교 기준**으로 평가될 수 있습니다.

---

## 📞 문의사항

구현 중 문제가 발생하거나 모델별 특화 요구사항이 발견되면, 이 템플릿 문서를 업데이트하여 향후 구현자들이 참고할 수 있도록 합니다.
