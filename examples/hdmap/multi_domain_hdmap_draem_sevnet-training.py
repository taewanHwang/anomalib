#!/usr/bin/env python3
"""MultiDomain HDMAP DRAEM-SevNet 도메인 전이 학습 예시.

DRAEM-SevNet 모델과 MultiDomainHDMAPDataModule을 활용한 효율적인 도메인 전이 학습 실험 스크립트입니다.

DRAEM-SevNet 특징:
- DRAEM Backbone Integration: 기존 DRAEM의 97.5M 파라미터 backbone 통합
- Wide ResNet Encoder: ImageNet pretrained encoder (기존 DRAEM과 동일)
- Reconstructive + Discriminative Sub-Networks: 기존 DRAEM 구조 완전 활용
- SeverityHead: Discriminative encoder features 직접 활용
- Multi-task Learning: Mask prediction + Severity prediction 동시 학습
- Score Combination: (mask_score + severity_score) / 2로 최종 anomaly score 계산
- Early Stopping: val_image_AUROC 기반 학습 효율성 향상

실험 구조:
1. MultiDomainHDMAPDataModule 설정 (e.g. source: domain_A, targets: domain_B,C,D)
2. Source Domain에서 DRAEM-SevNet 모델 훈련 (train 데이터)
3. Source Domain에서 성능 평가 (validation으로 사용될 test 데이터)
4. Target Domains에서 동시 성능 평가 (각 도메인별 test 데이터)
5. 도메인 전이 효과 종합 분석

주요 개선점 (DRAEM-SevNet vs Custom DRAEM):
- 정보 효율성: Discriminative features 직접 활용으로 정보 손실 최소화
- 성능 향상: Mask + Severity 결합으로 detection 정확도 개선

NOTE:
- 실험 조건들은 multi_domain_hdmap_draem_sevnet_exp_condition.py 파일에서 관리됩니다.
- 코드 유지보수성을 위해 실험 설정과 실행 로직을 분리했습니다.
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

# MultiDomain HDMAP import
from anomalib.data.datamodules.image.multi_domain_hdmap import MultiDomainHDMAPDataModule
from anomalib.models.image.draem_sevnet import DraemSevNet
from anomalib.engine import Engine
from pytorch_lightning.loggers import TensorBoardLogger

# Early Stopping import
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


# JSON 파일에서 실험 조건 로드
EXPERIMENT_CONDITIONS = load_experiment_conditions("multi_domain_hdmap_draem_sevnet-exp_condition4.json")

# 경고 메시지 비활성화
setup_warnings_filter()
logging.getLogger("anomalib.visualization.image.item_visualizer").setLevel(logging.ERROR)
logging.getLogger("anomalib.visualization").setLevel(logging.ERROR)
logging.getLogger("anomalib.callbacks").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")

# ========================================================================================
# 모델 훈련 및 실험 함수들
# ========================================================================================

def train_draem_sevnet_model_multi_domain(
    datamodule: MultiDomainHDMAPDataModule, 
    config: Dict[str, Any],
    results_base_dir: str,
    logger: logging.Logger
) -> tuple[DraemSevNet, Engine, str]:
    """DRAEM-SevNet 모델 훈련 수행.
    
    Args:
        datamodule: 설정된 MultiDomainHDMAPDataModule
        config: 훈련 설정 딕셔너리
        results_base_dir: 결과 저장 기본 경로
        experiment_name: 실험 이름
        logger: 로거 객체
        
    Returns:
        tuple: (훈련된 모델, Engine 객체, 체크포인트 경로)
        
    Note:
        DRAEM-SevNet 특징:
        - DRAEM Backbone (97.5M): Wide ResNet encoder + Discriminative/Reconstructive subnetworks
        - SeverityHead: Discriminative encoder features 직접 활용
        - Multi-task Loss: L2+SSIM (recon) + FocalLoss (seg) + MSE/SmoothL1 (severity)
        - Score Combination: (mask_score + severity_score) / 2
        - Early Stopping: val_image_AUROC 기반 학습 효율성 개선
    """
    
    print(f"\n🚀 DRAEM-SevNet 모델 훈련 시작")
    logger.info("🚀 DRAEM-SevNet 모델 훈련 시작")
    
    print(f"   ✅ DRAEM-SevNet 모델 생성 완료")
    print(f"   🔧 Config 설정:")
    print(f"      • 옵티마이저: {config['optimizer'].upper()}")
    print(f"      • 학습률: {config['learning_rate']}")
    print(f"      • Severity Head Mode: {config['severity_head_mode']}")
    print(f"      • Score Combination: {config['score_combination']}")
    print(f"      • Severity Loss Type: {config['severity_loss_type']}")
    print(f"      • Severity Weight: {config['severity_weight']}")
    print(f"      • Patch Width Range: {config['patch_width_range']}")
    print(f"      • Patch Ratio Range: {config['patch_ratio_range']}")
    print(f"      • Patch Count: {config['patch_count']}")
    print(f"      • Severity Max: {config['severity_max']}")
    
    logger.info("✅ DRAEM-SevNet 모델 생성 완료")
    logger.info(f"🔧 Config 설정: optimizer={config['optimizer']}, lr={config['learning_rate']}, severity_max={config['severity_max']}")
    
    # DRAEM-SevNet 모델 생성
    model = DraemSevNet(
        # 🎯 DRAEM-SevNet 아키텍처 설정
        severity_head_mode=config["severity_head_mode"],
        score_combination=config["score_combination"],
        severity_loss_type=config["severity_loss_type"],
        
        # 🔧 Synthetic Fault Generation 설정 
        patch_width_range=config["patch_width_range"],
        patch_ratio_range=config["patch_ratio_range"],
        patch_count=config["patch_count"],
        severity_max=config["severity_max"],
        
        # 🔧 Loss 가중치 설정
        severity_weight=config["severity_weight"],
        
        # 🔧 옵티마이저 설정
        optimizer=config["optimizer"],
        learning_rate=config["learning_rate"],
    )
    
    # Early stopping과 model checkpoint 설정 (val_image_AUROC 기반)
    early_stopping = EarlyStopping(
        monitor="val_image_AUROC",
        patience=config["early_stopping_patience"],
        mode="max",  # AUROC는 높을수록 좋음
        verbose=True
    )
    
    print(f"   📊 Early Stopping: patience={config['early_stopping_patience']}, monitor=val_image_AUROC (max)")
    logger.info(f"📊 Early Stopping 설정: patience={config['early_stopping_patience']}, monitor=val_image_AUROC")
    
    # 체크포인트 경로 설정
    checkpoint_callback = ModelCheckpoint(
        filename=f"draem_sevnet_multi_domain_{datamodule.source_domain}_" + "{epoch:02d}_{val_image_AUROC:.4f}",
        monitor="val_image_AUROC",
        mode="max",  # AUROC는 높을수록 좋음
        save_top_k=1,
        verbose=True
    )
    
    print(f"   💾 Model Checkpoint: monitor=val_image_AUROC (max), save_top_k=1")
    logger.info(f"💾 Model Checkpoint 설정: monitor=val_image_AUROC")
    
    # TensorBoard 로거 설정 (DRAEM과 동일)
    tb_logger = TensorBoardLogger(
        save_dir=results_base_dir,
        name="tensorboard_logs",
        version=""  # 빈 버전으로 version_x 폴더 방지
    )
    
    # Engine 생성 및 훈련
    engine_kwargs = {
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": [0] if torch.cuda.is_available() else 1,
        "logger": tb_logger,
        "max_epochs": config["max_epochs"],
        "callbacks": [early_stopping, checkpoint_callback],
        "check_val_every_n_epoch": 1,
        "enable_checkpointing": True,
        "log_every_n_steps": 10,
        "enable_model_summary": True,
        "num_sanity_val_steps": 0,
        "default_root_dir": results_base_dir
    }
    
    engine = Engine(**engine_kwargs)
    
    print(f"   🔧 Engine 설정 완료 - max_epochs: {config['max_epochs']}")
    print(f"   📁 결과 저장 경로: {results_base_dir}")
    logger.info(f"🔧 Engine 설정 완료 - max_epochs: {config['max_epochs']}")
    logger.info(f"📁 결과 저장 경로: {results_base_dir}")
    
    # 모델 훈련
    print(f"   🎯 모델 훈련 시작...")
    logger.info("🎯 모델 훈련 시작...")
    
    engine.fit(
        model=model,
        datamodule=datamodule
    )
    
    print(f"   ✅ 모델 훈련 완료!")
    logger.info("✅ 모델 훈련 완료!")
    
    # 최고 성능 체크포인트 경로 확인
    best_checkpoint = checkpoint_callback.best_model_path
    print(f"   🏆 Best Checkpoint: {best_checkpoint}")
    logger.info(f"🏆 Best Checkpoint: {best_checkpoint}")
    
    return model, engine, best_checkpoint





def run_single_draem_sevnet_experiment(
    condition: dict,
    log_dir: str = None
) -> dict:
    """단일 DRAEM-SevNet 실험 조건에 대한 실험 수행."""
    
    # config에서 도메인 설정 가져오기
    config = condition["config"]
    source_domain = config["source_domain"]
    target_domains = extract_target_domains_from_config(config)
    
    # 각 실험마다 고유한 results 경로 생성
    from datetime import datetime
    # run 스크립트에서 전달받은 log_dir 사용 (DRAEM과 동일하게)
    if log_dir:
        # run 스크립트에서 호출된 경우: 기존 timestamp 폴더 재사용
        base_timestamp_dir = log_dir
        timestamp_for_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = f"{condition['name']}_{timestamp_for_folder}"
    else:
        # 직접 호출된 경우: 새로운 timestamp 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_timestamp_dir = f"results/draem_sevnet/{timestamp}"
        experiment_folder = f"{condition['name']}_{timestamp}"
    
    results_base_dir = f"{base_timestamp_dir}/MultiDomainHDMAP/draem_sevnet/{experiment_folder}"
    
    # 실험 이름 생성
    experiment_name = f"{source_domain}"
    
    print(f"\n{'='*80}")
    print(f"🔬 DRAEM-SevNet 실험 조건: {condition['name']}")
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
        trained_model, engine, best_checkpoint = train_draem_sevnet_model_multi_domain(
            datamodule=multi_datamodule,
            config=condition["config"],
            results_base_dir=results_base_dir,
            logger=logging.getLogger(__name__)
        )
        
        # Source Domain 성능 평가
        print(f"\n📊 Source Domain 성능 평가 - {condition['name']}")
        source_results = evaluate_source_domain(
            model=trained_model,
            engine=engine,
            datamodule=multi_datamodule,
            checkpoint_path=best_checkpoint
        )
        
        # 실험 결과 폴더 생성 및 이미지 복사 
        # 실제 Anomalib이 이미지를 저장한 경로 찾기
        try:
            # 1. TensorBoardLogger 경로 확인
            trainer_log_dir = None
            if hasattr(engine.trainer, 'logger') and hasattr(engine.trainer.logger, 'log_dir'):
                trainer_log_dir = Path(engine.trainer.logger.log_dir)
                print(f"   📂 Trainer log_dir: {trainer_log_dir}")
            
            # 2. 실제 Anomalib 이미지 경로 탐색 (중첩 경로 포함)
            anomalib_image_paths = []
            base_search_path = Path(results_base_dir)
            
            # DraemSevNet 이미지 경로 패턴 검색 (실제 생성되는 경로)
            patterns = [
                "**/DraemSevNet/MultiDomainHDMAPDataModule/*/images",  # v0, v1 등의 버전 폴더
                "**/DraemSevNet/latest/images"  # latest 링크가 있는 경우
            ]
            for pattern in patterns:
                found_paths = list(base_search_path.glob(pattern))
                anomalib_image_paths.extend(found_paths)
            
            print(f"   🔍 발견된 이미지 경로들: {[str(p) for p in anomalib_image_paths]}")
            
            # 가장 최신 이미지 경로 선택
            if anomalib_image_paths:
                # 경로 생성 시간 기준으로 최신 선택
                latest_image_path = max(anomalib_image_paths, key=lambda p: p.stat().st_mtime if p.exists() else 0)
                anomalib_results_path = latest_image_path.parent  # images 폴더의 부모
                print(f"   ✅ 실제 Anomalib 결과 경로: {anomalib_results_path}")
            else:
                anomalib_results_path = None
                
            # 시각화 폴더는 TensorBoardLogger 경로에 생성
            if trainer_log_dir:
                latest_version_path = trainer_log_dir
            else:
                latest_version_path = Path(results_base_dir)
                
        except Exception as e:
            print(f"   ⚠️ Warning: 실제 이미지 경로 찾기 실패: {e}")
            latest_version_path = Path(results_base_dir)
            anomalib_results_path = None
        
        # Target Domains 성능 평가
        print(f"\n🎯 Target Domains 성능 평가 - {condition['name']}")
        target_results = evaluate_target_domains(
            model=trained_model,
            engine=engine,
            datamodule=multi_datamodule,
            checkpoint_path=best_checkpoint,
            results_base_dir=str(anomalib_results_path) if anomalib_results_path else results_base_dir,  # 실제 Anomalib 이미지 경로
            save_samples=True,  # Target Domain 이미지 복사 활성화
            current_version_path=str(latest_version_path) if latest_version_path else None  # 시각화 폴더는 TensorBoard 경로
        )
        
        if latest_version_path:
            # DRAEM-SevNet 시각화 폴더 생성 (target_results 이후에 실행)
            sevnet_viz_path_str = create_experiment_visualization(
                experiment_name=condition['name'],
                model_type="DRAEM-SevNet",
                results_base_dir=str(latest_version_path),
                source_domain=source_domain,
                target_domains=multi_datamodule.target_domains,
                source_results=source_results,
                target_results=target_results
            )
            sevnet_viz_path = Path(sevnet_viz_path_str) if sevnet_viz_path_str else latest_version_path / "visualize"
            
            # Source Domain 이미지 복사
            if anomalib_results_path:
                source_success = organize_source_domain_results(
                    sevnet_viz_path=str(sevnet_viz_path),
                    results_base_dir=str(anomalib_results_path),  # 실제 Anomalib 이미지가 있는 경로
                    source_domain=source_domain,
                    specific_version_path=str(anomalib_results_path)  # 실제 이미지 경로 전달
                )
            else:
                print("   ⚠️ Anomalib 이미지 경로를 찾을 수 없어 Source Domain 이미지 복사를 건너뜁니다.")
                source_success = False
            
            if source_success:
                print(f"   ✅ Source Domain ({source_domain}) 이미지 복사 완료")
            else:
                print(f"   ⚠️ Source Domain ({source_domain}) 이미지 복사 실패")
        
        # 학습 과정 정보 추출
        training_info = extract_training_info(engine)
        
        # 결과 분석
        analysis = analyze_experiment_results(
            source_results=source_results,
            target_results=target_results,
            training_info=training_info,
            condition=condition,
            model_type="DRAEM-SevNet"
        )
        
        # JSON 저장을 위해 DRAEM과 호환되는 형식으로 결과 변환
        source_results_compat = {}
        if source_results and 'image_AUROC' in source_results:
            source_results_compat = {
                "test_image_AUROC": source_results['image_AUROC'],
                "test_image_F1Score": source_results.get('image_F1Score', 0.0)
            }
        
        target_results_compat = {}
        for domain, result in target_results.items():
            if 'image_AUROC' in result:
                target_results_compat[domain] = {
                    "test_image_AUROC": result['image_AUROC'],
                    "test_image_F1Score": result.get('image_F1Score', 0.0)
                }
        
        # 실험 결과 정리
        experiment_result = {
            "condition": condition,
            "experiment_name": experiment_name,
            "source_results": source_results_compat,
            "target_results": target_results_compat,
            "best_checkpoint": best_checkpoint,
            "training_info": training_info,
            "status": "success",
            "experiment_path": str(latest_version_path) if latest_version_path else None,
            "avg_target_auroc": analysis["avg_target_auroc"]
            }
        
        # DRAEM과 동일하게 각 실험의 tensorboard_logs 폴더에 JSON 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"result_{condition['name']}_{timestamp}.json"
        
        # latest_version_path가 이미 tensorboard_logs이므로 직접 저장
        result_path = latest_version_path / result_filename
        
        try:
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(experiment_result, f, indent=2, ensure_ascii=False)
            print(f"📄 실험 결과 JSON 저장: {result_path}")
        except Exception as e:
            print(f"⚠️  JSON 저장 실패: {e}")
        
        print(f"\n✅ 실험 완료: {condition['name']}")
        
        return experiment_result
        
    except Exception as e:
        print(f"❌ 실험 실패 - {condition['name']}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "condition": condition,
            "experiment_name": experiment_name,
            "status": "failed",
            "error": str(e),
            "experiment_path": None  # 실패 시에는 경로 없음
        }
    finally:
        # 메모리 정리
        cleanup_gpu_memory()

def main():
    """DRAEM-SevNet 실험 메인 함수."""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="DRAEM-SevNet 실험")
    parser.add_argument("--gpu-id", type=str, help="사용할 GPU ID")
    parser.add_argument("--experiment-id", type=int, help="실험 조건 ID (0부터 시작)")
    parser.add_argument("--log-dir", type=str, help="로그 저장 디렉토리")
    parser.add_argument("--get-experiment-count", action="store_true", help="실험 조건 개수만 반환")
    
    args = parser.parse_args()
    
    # 실험 조건 개수만 반환하는 경우
    if args.get_experiment_count:
        print(len(EXPERIMENT_CONDITIONS))
        return
    
    # 필수 인자 검증
    if not args.gpu_id or args.experiment_id is None or not args.log_dir:
        parser.error("--gpu-id, --experiment-id, --log-dir는 필수 인자입니다 (--get-experiment-count 제외)")
    
    # GPU 설정 및 실험 조건 검증
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    if args.experiment_id >= len(EXPERIMENT_CONDITIONS):
        print(f"❌ 잘못된 실험 ID: {args.experiment_id} (최대: {len(EXPERIMENT_CONDITIONS)-1})")
        return
    
    condition = EXPERIMENT_CONDITIONS[args.experiment_id]
    
    print("="*80)
    print(f"🚀 DRAEM-SevNet 실험 (GPU {args.gpu_id}): {condition['name']}")
    print("="*80)
    
    # 로그 설정
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"draem_sevnet_experiment_{timestamp}.log"
    logger = setup_experiment_logging(str(log_path), f"draem_sevnet_{condition['name']}")
    
    # GPU 메모리 정리
    cleanup_gpu_memory()
    
    try:
        # 실험 정보 로깅
        logger.info("="*80)
        logger.info(f"🚀 DRAEM-SevNet 실험 시작: {condition['name']}")
        logger.info(f"GPU ID: {args.gpu_id} | 실험 ID: {args.experiment_id}")
        logger.info(f"설명: {condition['description']}")
        logger.info("="*80)
        
        # 실험 수행
        result = run_single_draem_sevnet_experiment(
            condition=condition,
            log_dir=args.log_dir
        )
        
        # 결과 저장
        result_filename = f"result_exp_{args.experiment_id:02d}_{condition['name']}_gpu{args.gpu_id}_{timestamp}.json"
        save_experiment_results(result, result_filename, log_dir, logger)
        
        logger.info("✅ 실험 완료!")
        
    except Exception as e:
        logger.error(f"❌ 실험 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        cleanup_gpu_memory()
        logger.info("🧹 GPU 메모리 정리 완료")


if __name__ == "__main__":
    main()
