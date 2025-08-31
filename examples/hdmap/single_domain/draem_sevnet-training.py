#!/usr/bin/env python3
"""SingleDomain HDMAP DRAEM-SevNet 학습 예시.

DRAEM-SevNet 모델과 HDMAPDataModule을 활용한 단일 도메인 이상 탐지 학습 실험 스크립트입니다.

DRAEM-SevNet 특징:
- DRAEM Backbone Integration: 기존 DRAEM의 97.5M 파라미터 backbone 통합
- Wide ResNet Encoder: ImageNet pretrained encoder (기존 DRAEM과 동일)
- Reconstructive + Discriminative Sub-Networks: 기존 DRAEM 구조 완전 활용
- Spatial-Aware SeverityHead: 공간 정보 보존으로 성능 향상
  * GAP vs Spatial-Aware pooling 선택 가능
  * AdaptiveAvgPool2d로 부분 공간 정보 유지
  * Spatial Attention 메커니즘 선택적 적용
  * Multi-Scale Spatial Features 지원
- Multi-task Learning: Mask prediction + Severity prediction 동시 학습
- Score Combination: 다양한 조합 전략 (simple_average, weighted_average, maximum)
- Early Stopping: val_image_AUROC 기반 학습 효율성 향상

실험 구조:
1. HDMAPDataModule 설정 (단일 도메인, validation을 train에서 분할)
2. 단일 도메인에서 DRAEM-SevNet 모델 훈련 (train 데이터)
3. 같은 도메인에서 성능 평가 (validation으로 사용할 train 분할 + test 데이터)
4. 학습 곡선 및 성능 분석

주요 개선점 (DRAEM-SevNet vs DRAEM):
- 정보 효율성: Discriminative features 직접 활용으로 정보 손실 최소화
- 성능 향상: Mask + Severity 결합으로 detection 정확도 개선
- 안정적 성능: 최적화된 spatial-aware pooling으로 robust한 특징 추출

NOTE:
- 실험 조건들은 draem_sevnet-exp_condition.json 파일에서 관리됩니다.
- validation은 train 데이터에서 분할하므로 모두 정상 데이터로 구성됩니다.
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
from anomalib.models.image.draem_sevnet import DraemSevNet
from anomalib.engine import Engine
from pytorch_lightning.loggers import TensorBoardLogger

# Callbacks import (학습이 필요한 모델)
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
import torch.nn.functional as F

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



def train_draem_sevnet_model_single_domain(
    datamodule, 
    config: Dict[str, Any],
    results_base_dir: str,
    logger: logging.Logger
) -> tuple[DraemSevNet, Engine, str]:
    """
    DRAEM-SevNet 모델 훈련 수행.
    
    Args:
        datamodule: 설정된 HDMAPDataModule
        config: 훈련 설정 딕셔너리
        results_base_dir: 결과 저장 기본 경로
        logger: 로거 객체
        
    Returns:
        tuple: (훈련된 모델, Engine 객체, 체크포인트 경로)
        
    Note:
        DRAEM-SevNet 특징:
        - Multi-task Learning: Mask prediction + Severity prediction
        - Spatial-Aware SeverityHead: 공간 정보 보존 pooling
        - Early Stopping: val_image_AUROC 기반 조기 종료
    """
    
    print(f"\n🚀 DRAEM-SevNet 모델 훈련 시작")
    logger.info("🚀 DRAEM-SevNet 모델 훈련 시작")
    
    # Config 설정 출력
    print(f"   🔧 Config 설정:")
    print(f"      Max Epochs: {config['max_epochs']}")
    print(f"      Learning Rate: {config['learning_rate']}")
    print(f"      Batch Size: {config['batch_size']}")
    print(f"      Optimizer: {config.get('optimizer', 'adamw')}")
    print(f"      Early Stopping Patience: {config['early_stopping_patience']}")
    print(f"      Severity Head Mode: {config['severity_head_mode']}")
    print(f"      Score Combination: {config['score_combination']}")
    print(f"      Severity Max: {config['severity_max']}")
    print(f"      Spatial Size: {config['severity_head_spatial_size']}")
    
    logger.info("✅ DRAEM-SevNet 모델 생성 완료")
    logger.info(f"🔧 Config 설정: max_epochs={config['max_epochs']}, lr={config['learning_rate']}")
    logger.info(f"🎯 SevNet 설정: mode={config['severity_head_mode']}, spatial_size={config['severity_head_spatial_size']}")
    
    # DRAEM-SevNet 모델 생성
    model = DraemSevNet(
        severity_head_mode=config["severity_head_mode"],
        score_combination=config["score_combination"],
        severity_loss_type=config.get("severity_loss_type", "smooth_l1"),
        severity_weight=config.get("severity_weight", 1.0),
        severity_max=config.get("severity_max", 25.0),
        patch_width_range=config.get("patch_width_range", [32, 64]),
        patch_ratio_range=config.get("patch_ratio_range", [0.1, 0.5]),
        patch_count=config.get("patch_count", 1),
        severity_head_pooling_type=config.get("severity_head_pooling_type", "spatial_aware"),
        severity_head_spatial_size=config.get("severity_head_spatial_size", 8),
        severity_head_use_spatial_attention=config.get("severity_head_use_spatial_attention", False),
        optimizer=config.get("optimizer", "adamw"),
        learning_rate=config["learning_rate"]
    )
    
    # Early Stopping 콜백 설정 (validation loss 사용 - single domain에 적합)
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config["early_stopping_patience"],
        mode="min",  # loss는 작을수록 좋음
        verbose=True
    )
    
    checkpoint_callback = ModelCheckpoint(
        filename=f"draem_sevnet_single_domain_{datamodule.domain}_" + "{epoch:02d}_{val_loss:.4f}",
        monitor="val_loss",
        mode="min",  # loss는 작을수록 좋음
        save_top_k=1,
        verbose=True
    )
    
    callbacks = [early_stopping, checkpoint_callback]
    
    # TensorBoard 로거 설정
    tb_logger = TensorBoardLogger(
        save_dir=results_base_dir,
        name="tensorboard_logs",
        version=""
    )
    
    # Engine 설정 (학습 기반 모델)
    engine_kwargs = {
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": [0] if torch.cuda.is_available() else 1,
        "logger": tb_logger,
        "callbacks": callbacks,
        "enable_checkpointing": True,
        "log_every_n_steps": 10,
        "enable_model_summary": True,
        "default_root_dir": results_base_dir,
        "max_epochs": config["max_epochs"],
        "check_val_every_n_epoch": 1,
        "num_sanity_val_steps": 0
    }
    
    engine = Engine(**engine_kwargs)
    
    print(f"   🔧 Engine 설정 완료")
    print(f"   📁 결과 저장 경로: {results_base_dir}")
    logger.info(f"🔧 Engine 설정 완료")
    logger.info(f"📁 결과 저장 경로: {results_base_dir}")
    
    # 모델 훈련
    print(f"   🎯 모델 훈련 시작...")
    logger.info("🎯 모델 훈련 시작...")
    
    engine.fit(
        model=model,
        datamodule=datamodule
    )
    
    best_checkpoint = checkpoint_callback.best_model_path
    print(f"   🏆 Best Checkpoint: {best_checkpoint}")
    logger.info(f"🏆 Best Checkpoint: {best_checkpoint}")
    
    print(f"   ✅ 모델 훈련 완료!")
    logger.info("✅ 모델 훈련 완료!")
    
    return model, engine, best_checkpoint


def evaluate_single_domain(
    model: DraemSevNet, 
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
        
        # 디버깅: 결과 구조 확인
        print(f"   🔍 Test results keys: {list(test_metrics.keys())}")
        logger.info(f"🔍 Test results keys: {list(test_metrics.keys())}")
        
        results = {
            "domain": datamodule.domain,
            "image_AUROC": test_metrics.get("test_image_AUROC", test_metrics.get("image_AUROC", 0.0)),
            "pixel_AUROC": test_metrics.get("test_pixel_AUROC", test_metrics.get("pixel_AUROC", 0.0)),
            "image_F1Score": test_metrics.get("test_image_F1Score", test_metrics.get("image_F1Score", 0.0)),
            "pixel_F1Score": test_metrics.get("test_pixel_F1Score", test_metrics.get("pixel_F1Score", 0.0)),
            "training_samples": len(datamodule.train_data),
            "test_samples": len(datamodule.test_data),
            "val_samples": len(datamodule.val_data) if datamodule.val_data else 0
        }
        
        print(f"   ✅ {datamodule.domain} 평가 완료:")
        print(f"      Image AUROC: {results['image_AUROC']:.4f}")
        print(f"      Pixel AUROC: {results['pixel_AUROC']:.4f}")
        print(f"      Image F1: {results['image_F1Score']:.4f}")
        print(f"      Pixel F1: {results['pixel_F1Score']:.4f}")
        
        logger.info(f"✅ {datamodule.domain} 평가 완료: Image AUROC={results['image_AUROC']:.4f}")
    else:
        results = {"domain": datamodule.domain, "error": "No test results available"}
        logger.error(f"❌ {datamodule.domain} 평가 실패")
    
    return results


def run_single_draem_sevnet_experiment(
    condition: dict,
    log_dir: str = None
) -> dict:
    """단일 DRAEM-SevNet 실험 조건에 대한 실험 수행."""
    
    # config에서 도메인 설정 가져오기
    config = condition["config"]
    domain = "domain_A"  # Single Domain A 실험
    
    # 실험 경로 설정
    if log_dir:
        base_timestamp_dir = log_dir
        timestamp_for_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = f"{condition['name']}_{timestamp_for_folder}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_timestamp_dir = f"results/draem_sevnet/{timestamp}"
        experiment_folder = f"{condition['name']}_{timestamp}"
    
    results_base_dir = f"{base_timestamp_dir}/SingleDomainHDMAP/DRAEM_SevNet/{experiment_folder}"
    
    # 실험 이름 생성
    experiment_name = f"{domain}_single"
    
    print(f"\n{'='*80}")
    print(f"🔬 DRAEM-SevNet Single Domain 실험 조건: {condition['name']}")
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
        logger.info("🚀 DRAEM-SevNet Single Domain 실험 시작")
        
        # DataModule 생성 (유틸리티 함수 사용)
        datamodule = create_single_domain_datamodule(
            domain=domain,
            batch_size=config["batch_size"],
            image_size="224x224",
            val_split_ratio=0.1,
            num_workers=4,
            seed=42
        )
        
        # 모델 훈련
        trained_model, engine, best_checkpoint = train_draem_sevnet_model_single_domain(
            datamodule=datamodule,
            config=config,
            results_base_dir=results_base_dir,
            logger=logger
        )
        
        # 성능 평가
        results = evaluate_single_domain(trained_model, engine, datamodule, logger)
        
        # 훈련 정보 추출
        training_info = extract_training_info(engine)
        
        # 실험 결과 정리 (Single Domain 호환 형식)
        experiment_results = {
            "experiment_name": condition["name"],
            "description": condition["description"],
            "domain": domain,
            "config": config,
            "results": results,
            "training_info": training_info,
            "timestamp": datetime.now().isoformat(),
            "checkpoint_path": best_checkpoint,
            "status": "success",
            # Single Domain 분석을 위한 구조
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
        
        # 결과 저장 (Single Domain 호환 형식)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # tensorboard_logs 디렉토리 생성
        tensorboard_logs_dir = Path(results_base_dir) / "tensorboard_logs"
        tensorboard_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Single Domain 호환 결과 파일명
        result_filename = f"result_{timestamp}.json"
        
        results_file = save_experiment_results(
            result=experiment_results,
            result_filename=result_filename,
            log_dir=tensorboard_logs_dir,
            logger=logger,
            model_type="DRAEM_SevNet"
        )
        print(f"📄 실험 결과 저장됨: {results_file}")
        
        # 시각화 생성
        try:
            create_experiment_visualization(
                experiment_results, 
                results_base_dir, 
                f"DRAEM_SevNet_single_domain_{domain}",
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
    parser = argparse.ArgumentParser(description="DRAEM-SevNet Single Domain 실험")
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
    conditions = load_experiment_conditions("draem_sevnet-exp_condition.json")
    
    if args.experiment_id >= len(conditions):
        print(f"❌ 잘못된 실험 ID: {args.experiment_id} (최대: {len(conditions)-1})")
        return
    
    condition = conditions[args.experiment_id]
    
    # 실험 실행
    result = run_single_draem_sevnet_experiment(condition, args.log_dir)
    
    if "error" not in result:
        print(f"\n🎉 실험 성공!")
        if "results" in result and isinstance(result["results"], dict):
            print(f"   📊 최종 성과:")
            print(f"      Image AUROC: {result['results'].get('image_AUROC', 0):.4f}")
            print(f"      Pixel AUROC: {result['results'].get('pixel_AUROC', 0):.4f}")
    else:
        print(f"\n💥 실험 실패: {result['error']}")


if __name__ == "__main__":
    main()