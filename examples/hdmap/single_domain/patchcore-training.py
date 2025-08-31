#!/usr/bin/env python3
"""SingleDomain HDMAP PatchCore 학습 예시.

PatchCore 모델과 HDMAPDataModule을 활용한 단일 도메인 이상 탐지 실험 스크립트입니다.

PatchCore 특징:
- Memory Bank 기반: 정상 샘플의 patch feature들을 메모리 뱅크에 저장
- 학습 불필요: Pretrained CNN backbone + 최근접 이웃 탐색으로 이상 탐지
- Coreset Subsampling: K-center-greedy 알고리즘으로 메모리 뱅크 최적화
- 1 Epoch 피팅: 정상 데이터에서 feature 추출 및 메모리 뱅크 구축만 수행
- Multi-layer Features: 여러 CNN layer에서 추출한 중간 레벨 feature 활용
- Nearest Neighbor Search: 테스트 시 메모리 뱅크와의 거리 기반 anomaly score 계산

실험 구조:
1. HDMAPDataModule 설정 (단일 도메인, validation을 train에서 분할)
2. 단일 도메인에서 PatchCore 모델 피팅 (train 데이터로 메모리 뱅크 구축)
3. 같은 도메인에서 성능 평가 (validation으로 사용할 train 분할 + test 데이터)
4. 메모리 사용량 및 성능 분석

주요 개선점 (PatchCore vs 학습 기반 모델):
- 훈련 시간 단축: 1 epoch 피팅으로 빠른 학습
- 메모리 효율성: Coreset subsampling으로 메모리 사용량 최적화
- 해석 가능성: Patch 단위 anomaly localization 제공
- 안정성: Pretrained backbone 활용으로 안정적인 성능

NOTE:
- 실험 조건들은 patchcore-exp_condition.json 파일에서 관리됩니다.
- PatchCore는 학습 없이 피팅만 수행하므로 early stopping, optimizer 설정이 불필요합니다.
"""

import os
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import logging
import warnings
import argparse

# HDMAP import
from anomalib.models.image.patchcore import Patchcore
from anomalib.engine import Engine
from pytorch_lightning.loggers import TensorBoardLogger

# 공통 유틸리티 함수들 import
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from experiment_utils import (
    cleanup_gpu_memory,
    setup_warnings_filter,
    setup_experiment_logging,
    extract_training_info,
    save_experiment_results,
    create_experiment_visualization,
    load_experiment_conditions,
    analyze_experiment_results,
    create_single_domain_datamodule
)

# 경고 메시지 비활성화
setup_warnings_filter()
logging.getLogger("anomalib.visualization.image.item_visualizer").setLevel(logging.ERROR)
logging.getLogger("anomalib.visualization").setLevel(logging.ERROR)
logging.getLogger("anomalib.callbacks").setLevel(logging.ERROR)


def train_patchcore_model_single_domain(
    datamodule, 
    config: Dict[str, Any],
    results_base_dir: str,
    logger: logging.Logger
) -> tuple[Patchcore, Engine, str]:
    """
    PatchCore 모델 피팅 수행.
    
    Args:
        datamodule: 설정된 HDMAPDataModule
        config: 피팅 설정 딕셔너리
        results_base_dir: 결과 저장 기본 경로
        logger: 로거 객체
        
    Returns:
        tuple: (피팅된 모델, Engine 객체, 체크포인트 경로)
        
    Note:
        PatchCore 특징:
        - 피처 기반 모델로 Early Stopping 및 체크포인트 불필요
        - 1 Epoch 피팅으로 메모리 뱅크 구축
        - Coreset subsampling으로 메모리 최적화
    """
    
    print(f"\n🚀 PatchCore 모델 피팅 시작")
    logger.info("🚀 PatchCore 모델 피팅 시작")
    
    # Config 설정 출력
    print(f"   🔧 Config 설정:")
    print(f"      Backbone: {config['backbone']}")
    print(f"      Layers: {config['layers']}")
    print(f"      Coreset Sampling Ratio: {config['coreset_sampling_ratio']}")
    print(f"      Num Neighbors: {config['num_neighbors']}")
    print(f"      Batch Size: {config['batch_size']}")
    
    logger.info("✅ PatchCore 모델 생성 완료")
    logger.info(f"🔧 Config 설정: backbone={config['backbone']}, coreset_ratio={config['coreset_sampling_ratio']}")
    
    # PatchCore 모델 생성
    model = Patchcore(
        backbone=config["backbone"],
        layers=config["layers"],
        pre_trained=config.get("pre_trained", True),
        coreset_sampling_ratio=config["coreset_sampling_ratio"],
        num_neighbors=config["num_neighbors"]
    )
    
    # TensorBoard 로거 설정
    tb_logger = TensorBoardLogger(
        save_dir=results_base_dir,
        name="tensorboard_logs",
        version=""
    )
    
    # Engine 설정 (피처 기반 모델 - 학습 없음)
    engine_kwargs = {
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": [0] if torch.cuda.is_available() else 1,
        "logger": tb_logger,
        "log_every_n_steps": 10,
        "enable_model_summary": True,
        "default_root_dir": results_base_dir
    }
    
    engine = Engine(**engine_kwargs)
    
    print(f"   🔧 Engine 설정 완료")
    print(f"   📁 결과 저장 경로: {results_base_dir}")
    logger.info(f"🔧 Engine 설정 완료")
    logger.info(f"📁 결과 저장 경로: {results_base_dir}")
    
    # 모델 피팅 (메모리 뱅크 구축)
    print(f"   🎯 모델 피팅 시작... (메모리 뱅크 구축)")
    logger.info("🎯 모델 피팅 시작... (메모리 뱅크 구축)")
    
    engine.fit(
        model=model,
        datamodule=datamodule
    )
    
    # PatchCore는 체크포인트가 없음
    best_checkpoint = None
    print(f"   ✅ 모델 피팅 완료! (메모리 뱅크 구축됨)")
    logger.info("✅ 모델 피팅 완료! (메모리 뱅크 구축됨)")
    
    return model, engine, best_checkpoint


def evaluate_single_domain(
    model: Patchcore, 
    engine: Engine, 
    datamodule, 
    logger: logging.Logger
) -> Dict[str, Any]:
    """단일 도메인에서 모델 성능 평가."""
    
    print(f"\n📊 {datamodule.domain} 도메인 성능 평가 시작")
    logger.info(f"📊 {datamodule.domain} 도메인 성능 평가 시작")
    
    # 테스트 수행
    test_results = engine.test(model=model, datamodule=datamodule)
    
    # 결과 정리
    if test_results and len(test_results) > 0:
        test_metrics = test_results[0]
        
        results = {
            "domain": datamodule.domain,
            "image_AUROC": test_metrics.get("test_image_AUROC", 0.0),
            "pixel_AUROC": test_metrics.get("test_pixel_AUROC", 0.0),
            "image_F1Score": test_metrics.get("test_image_F1Score", 0.0),
            "pixel_F1Score": test_metrics.get("test_pixel_F1Score", 0.0),
            "training_samples": len(datamodule.train_data),
            "test_samples": len(datamodule.test_data),
            "val_samples": len(datamodule.val_data) if datamodule.val_data else 0,
            "memory_bank_size": getattr(model, 'memory_bank', {}).get('size', 0) if hasattr(model, 'memory_bank') else 0
        }
        
        print(f"   ✅ {datamodule.domain} 평가 완료:")
        print(f"      Image AUROC: {results['image_AUROC']:.4f}")
        print(f"      Pixel AUROC: {results['pixel_AUROC']:.4f}")
        print(f"      Image F1: {results['image_F1Score']:.4f}")
        print(f"      Pixel F1: {results['pixel_F1Score']:.4f}")
        if results['memory_bank_size'] > 0:
            print(f"      Memory Bank Size: {results['memory_bank_size']}")
        
        logger.info(f"✅ {datamodule.domain} 평가 완료: Image AUROC={results['image_AUROC']:.4f}")
    else:
        results = {"domain": datamodule.domain, "error": "No test results available"}
        logger.error(f"❌ {datamodule.domain} 평가 실패")
    
    return results


def run_single_patchcore_experiment(
    condition: dict,
    log_dir: str = None
) -> dict:
    """단일 PatchCore 실험 조건에 대한 실험 수행."""
    
    # config에서 도메인 설정 가져오기
    config = condition["config"]
    domain = config["domain"]
    
    # 실험 경로 설정
    if log_dir:
        base_timestamp_dir = log_dir
        timestamp_for_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = f"{condition['name']}_{timestamp_for_folder}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_timestamp_dir = f"results/patchcore/{timestamp}"
        experiment_folder = f"{condition['name']}_{timestamp}"
    
    results_base_dir = f"{base_timestamp_dir}/MultiDomainHDMAP/PatchCore/{experiment_folder}"
    
    # 실험 이름 생성
    experiment_name = f"{domain}_single"
    
    print(f"\n{'='*80}")
    print(f"🔬 PatchCore Single Domain 실험 조건: {condition['name']}")
    print(f"📝 설명: {condition['description']}")
    print(f"🎯 도메인: {domain}")
    print(f"{'='*80}")
    
    try:
        # GPU 메모리 정리
        cleanup_gpu_memory()
        
        # 결과 디렉토리 생성
        os.makedirs(results_base_dir, exist_ok=True)
        
        # 실험 로깅 설정
        log_file_path = os.path.join(results_base_dir, f"{experiment_name}.log")
        logger = setup_experiment_logging(log_file_path, experiment_name)
        logger.info("🚀 PatchCore Single Domain 실험 시작")
        
        # DataModule 생성 (유틸리티 함수 사용)
        datamodule = create_single_domain_datamodule(
            domain=domain,
            batch_size=config["batch_size"],
            image_size="224x224",
            val_split_ratio=0.1,
            num_workers=4,
            seed=42
        )
        
        # 모델 피팅
        fitted_model, engine, best_checkpoint = train_patchcore_model_single_domain(
            datamodule=datamodule,
            config=config,
            results_base_dir=results_base_dir,
            logger=logger
        )
        
        # 성능 평가
        results = evaluate_single_domain(fitted_model, engine, datamodule, logger)
        
        # 훈련 정보 추출 (PatchCore는 훈련이 없으므로 최소 정보)
        training_info = {
            "epochs_run": 1,  # PatchCore는 1 epoch 피팅
            "training_time": "N/A (no training required)",
            "best_epoch": 1,
            "early_stopped": False,
            "checkpoint_path": None
        }
        
        # 실험 결과 정리 (Multi Domain 호환 형식)
        experiment_results = {
            "experiment_name": condition["name"],
            "description": condition["description"],
            "domain": domain,
            "config": config,
            "results": results,
            "training_info": training_info,
            "timestamp": datetime.now().isoformat(),
            "checkpoint_path": None,
            "status": "success",
            # Multi Domain 분석 호환을 위한 구조
            "condition": {
                "name": condition["name"],
                "description": condition["description"],
                "config": {
                    "source_domain": domain,  # Single domain을 source로 취급
                    **config
                }
            },
            "source_results": {
                "test_image_AUROC": results.get("image_AUROC", 0.0),
                "test_pixel_AUROC": results.get("pixel_AUROC", 0.0),
                "test_image_F1Score": results.get("image_F1Score", 0.0),
                "test_pixel_F1Score": results.get("pixel_F1Score", 0.0),
                "domain": domain
            },
            "target_results": {}  # Single domain이므로 target 없음
        }
        
        # 결과 저장 (Multi Domain 호환 형식)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # tensorboard_logs 디렉토리 생성
        tensorboard_logs_dir = Path(results_base_dir) / "tensorboard_logs"
        tensorboard_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Multi Domain 호환 결과 파일명
        result_filename = f"result_{timestamp}.json"
        
        results_file = save_experiment_results(
            result=experiment_results,
            result_filename=result_filename,
            log_dir=tensorboard_logs_dir,
            logger=logger,
            model_type="PatchCore"
        )
        print(f"📄 실험 결과 저장됨: {results_file}")
        
        # 시각화 생성
        try:
            create_experiment_visualization(
                experiment_results, 
                results_base_dir, 
                f"PatchCore_single_domain_{domain}",
                single_domain=True
            )
            print(f"📊 결과 시각화 생성 완료")
        except Exception as viz_error:
            print(f"⚠️ 시각화 생성 중 오류: {viz_error}")
            logger.warning(f"시각화 생성 중 오류: {viz_error}")
        
        # GPU 메모리 정리
        cleanup_gpu_memory()
        
        print(f"✅ 실험 완료: {condition['name']}")
        logger.info(f"✅ 실험 완료: {condition['name']}")
        
        return experiment_results
        
    except Exception as e:
        error_msg = f"❌ 실험 실패: {condition['name']} - {str(e)}"
        print(error_msg)
        if 'logger' in locals():
            logger.error(error_msg)
        
        cleanup_gpu_memory()
        
        return {
            "experiment_name": condition["name"],
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "status": "failed"
        }


def main():
    """메인 실행 함수."""
    parser = argparse.ArgumentParser(description="PatchCore Single Domain 실험")
    parser.add_argument("--gpu-id", type=int, default=0, help="사용할 GPU ID")
    parser.add_argument("--experiment-id", type=int, default=0, help="실험 조건 인덱스")
    parser.add_argument("--log-dir", type=str, default=None, help="로그 저장 디렉토리")
    
    args = parser.parse_args()
    
    # GPU 설정
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"🖥️ GPU {args.gpu_id} 사용")
    
    # 경고 필터 설정
    setup_warnings_filter()
    
    # 실험 조건 로드
    conditions = load_experiment_conditions("patchcore-exp_condition.json")
    
    if args.experiment_id >= len(conditions):
        print(f"❌ 잘못된 실험 ID: {args.experiment_id} (최대: {len(conditions)-1})")
        return
    
    condition = conditions[args.experiment_id]
    
    # 실험 실행
    result = run_single_patchcore_experiment(condition, args.log_dir)
    
    if "error" not in result:
        print(f"\n🎉 실험 성공!")
        if "results" in result and isinstance(result["results"], dict):
            print(f"   📊 최종 성과:")
            print(f"      Image AUROC: {result['results'].get('image_AUROC', 0):.4f}")
            print(f"      Pixel AUROC: {result['results'].get('pixel_AUROC', 0):.4f}")
            if result['results'].get('memory_bank_size', 0) > 0:
                print(f"      Memory Bank Size: {result['results']['memory_bank_size']}")
    else:
        print(f"\n💥 실험 실패: {result['error']}")


if __name__ == "__main__":
    main()