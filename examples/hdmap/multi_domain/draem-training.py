#!/usr/bin/env python3
"""HDMAP 다중 도메인 DRAEM 모델 훈련 및 평가 스크립트.

이 스크립트는 HDMAP 데이터셋에서 DRAEM 모델을 훈련하고 다중 도메인 평가를 수행합니다.

주요 기능:
- DRAEM 모델을 사용한 이상 탐지
- 소스 도메인(domain_A)에서 훈련
- 타겟 도메인들(domain_B, C, D)에서 평가
- 실험 결과 시각화 및 저장
- JSON 기반 실험 조건 관리

사용법:
    python multi_domain_hdmap_draem_training.py --experiment_name "DRAEM_quick_3epochs"
    python multi_domain_hdmap_draem_training.py --experiment_name "DRAEM_baseline_50epochs" --log_level DEBUG

실험 조건:
    모든 실험 조건은 multi_domain_hdmap_draem-exp_condition.json 파일에서 관리됩니다.
    병렬 실행을 위해서는 multi_domain_hdmap_draem-run.sh 스크립트를 사용하세요.

"""

import argparse
import logging
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from anomalib.models.image.draem import Draem
from anomalib.data.datamodules.image.multi_domain_hdmap import MultiDomainHDMAPDataModule
from anomalib.engine import Engine
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Experiment utilities import
# 공통 유틸리티 함수들 import - 상위 디렉토리에서 import
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
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
    analyze_multi_experiment_results,
    load_experiment_conditions,
    analyze_experiment_results,
    extract_target_domains_from_config,
    create_common_experiment_result
)

# 경고 메시지 비활성화 (DraemSevNet과 동일)
setup_warnings_filter()
logging.getLogger("anomalib.visualization.image.item_visualizer").setLevel(logging.ERROR)
logging.getLogger("anomalib.visualization").setLevel(logging.ERROR)
logging.getLogger("anomalib.callbacks").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")



# JSON 파일에서 실험 조건 로드
EXPERIMENT_CONDITIONS = load_experiment_conditions("multi_domain_hdmap_draem-exp_condition.json")


def train_draem_model_multi_domain(
    datamodule: MultiDomainHDMAPDataModule, 
    config: Dict[str, Any],
    results_base_dir: str,
    logger: logging.Logger
) -> tuple[Draem, Engine, str]:
    """DRAEM 모델 훈련 수행.
    
    Args:
        datamodule: 설정된 MultiDomainHDMAPDataModule
        config: 훈련 설정 딕셔너리
        results_base_dir: 결과 저장 기본 경로
        logger: 로거 객체
        
    Returns:
        tuple: (훈련된 모델, Engine 객체, 체크포인트 경로)
    """
    print(f"\n🚀 DRAEM 모델 훈련 시작")
    logger.info("🚀 DRAEM 모델 훈련 시작")
    
    # DRAEM 모델 초기화
    model = Draem()
    
    # Config에서 지원되는 파라미터들을 모델에 동적으로 설정
    if hasattr(model, '_config'):
        model._config = config
    else:
        # config를 모델에 저장하여 configure_optimizers에서 사용
        setattr(model, '_training_config', config)
    
    print(f"   ✅ DRAEM 모델 생성 완료")
    print(f"   🔧 Config 설정:")
    print(f"      • 옵티마이저: {config['optimizer'].upper()}")
    print(f"      • 학습률: {config['learning_rate']}")
    print(f"      • Weight Decay: {config['weight_decay']}")
    if 'scheduler' in config:
        print(f"      • 스케줄러: {config['scheduler']}")
    logger.info("✅ DRAEM 모델 생성 완료 (validation loss 포함)")
    logger.info(f"🔧 Config 설정: optimizer={config['optimizer']}, lr={config['learning_rate']}, weight_decay={config['weight_decay']}")
    
    # Early stopping과 model checkpoint 설정 (val_image_AUROC 기반)
    early_stopping = EarlyStopping(
        monitor="val_image_AUROC",
        patience=config["early_stopping_patience"],
        mode="max",  # AUROC는 높을수록 좋음
        verbose=True
    )
    
    # 체크포인트 경로 설정
    checkpoint_callback = ModelCheckpoint(
        filename=f"draem_multi_domain_{datamodule.source_domain}_" + "{epoch:02d}_{val_image_AUROC:.4f}",
        monitor="val_image_AUROC",
        mode="max",  # AUROC는 높을수록 좋음
        save_top_k=1,
        verbose=True
    )
    
    print(f"   📊 Early Stopping: patience={config['early_stopping_patience']}, monitor=val_image_AUROC (max)")
    print(f"   💾 Model Checkpoint: monitor=val_image_AUROC (max), save_top_k=1")
    logger.info(f"📊 Early Stopping 설정: patience={config['early_stopping_patience']}, monitor=val_image_AUROC")
    logger.info(f"💾 Model Checkpoint 설정: monitor=val_image_AUROC")
    
    # TensorBoard 로거 설정 (DraemSevNet과 동일)
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
    
    # gradient_clip_val 설정 (config에서 제공되면)
    if "gradient_clip_val" in config and config["gradient_clip_val"] is not None:
        engine_kwargs["gradient_clip_val"] = config["gradient_clip_val"]
        print(f"   🔧 Gradient Clipping 설정: {config['gradient_clip_val']}")
        logger.info(f"🔧 Gradient Clipping 설정: {config['gradient_clip_val']}")
    
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
    
    # 실제 생성된 디렉토리 경로 추출
    if hasattr(engine.trainer, 'default_root_dir'):
        actual_results_dir = engine.trainer.default_root_dir
        print(f"   📂 실제 결과 디렉토리: {actual_results_dir}")
        logger.info(f"📂 실제 결과 디렉토리: {actual_results_dir}")
    else:
        actual_results_dir = results_base_dir
    
    return model, engine, best_checkpoint





def run_single_draem_experiment(
    condition: Dict[str, Any],
    dataset_root: str = None,
    results_base_dir: str = "./results",
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """단일 DRAEM 실험 수행.
    
    Args:
        condition: 실험 조건 딕셔너리
        dataset_root: 데이터셋 루트 경로
        results_base_dir: 결과 저장 기본 경로
        logger: 로거 객체
        
    Returns:
        Dict[str, Any]: 실험 결과
    """
    # config에서 도메인 설정 가져오기
    config = condition["config"]
    source_domain = config["source_domain"]
    target_domains = extract_target_domains_from_config(config)
    
    experiment_name = condition["name"]
    
    print(f"\n{'='*80}")
    print(f"🧪 DRAEM 실험 시작: {experiment_name}")
    print(f"{'='*80}")
    
    if logger:
        logger.info(f"🧪 DRAEM 실험 시작: {experiment_name}")
        logger.info(f"실험 설정: {config}")
    
    try:
        # GPU 메모리 정리
        cleanup_gpu_memory()
        
        # 각 실험마다 고유한 results 경로 생성 (DraemSevNet과 동일한 구조)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = f"{condition['name']}_{timestamp}"
        experiment_dir = Path(results_base_dir) / "MultiDomainHDMAP" / "draem" / experiment_folder
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 실험 이름 생성 (DraemSevNet과 동일하게)
        experiment_name = f"{source_domain}"
        
        print(f"📁 실험 디렉토리: {experiment_dir}")
        if logger:
            logger.info(f"📁 실험 디렉토리: {experiment_dir}")
        
        # DataModule 생성
        datamodule = create_multi_domain_datamodule(
            datamodule_class=MultiDomainHDMAPDataModule,
            source_domain=source_domain,
            target_domains=target_domains,
            batch_size=config["batch_size"],
            image_size=config["image_size"],
            dataset_root=dataset_root
        )
        
        # 모델 훈련
        model, engine, best_checkpoint = train_draem_model_multi_domain(
            datamodule=datamodule,
            config=config,
            results_base_dir=str(experiment_dir),
            logger=logger or logging.getLogger(__name__)
        )
        
        # TensorBoardLogger 경로 확인
        try:
            if hasattr(engine.trainer, 'logger') and hasattr(engine.trainer.logger, 'log_dir'):
                latest_version_path = Path(engine.trainer.logger.log_dir)
                print(f"   📂 Trainer log_dir: {latest_version_path}")
            else:
                latest_version_path = Path(str(experiment_dir))
                print(f"   📂 기본 경로 사용: {latest_version_path}")
            
            # 이미지 경로는 평가 후에 생성되므로 일단 None으로 설정
            anomalib_results_path = None
                
        except Exception as e:
            print(f"   ⚠️ Warning: 로그 경로 설정 실패: {e}")
            latest_version_path = Path(str(experiment_dir))
            anomalib_results_path = None
        
        # 시각화 폴더 생성 (실제 결과 경로 사용)
        viz_path = create_experiment_visualization(
            experiment_name=experiment_name,
            model_type="DRAEM",
            results_base_dir=str(latest_version_path),  # TensorBoard 로그 경로 사용
            source_domain=source_domain,
            target_domains=target_domains
        )
        
        # Source Domain 평가
        print(f"\n📊 Source Domain 평가 시작")
        if logger:
            logger.info("📊 Source Domain 평가 시작")
        
        engine_for_eval = Engine(default_root_dir=str(latest_version_path))
        source_results = evaluate_source_domain(
            model=model,
            engine=engine_for_eval,
            datamodule=datamodule,
            checkpoint_path=best_checkpoint
        )
        
        # Source Domain 결과 정리
        organize_source_domain_results(
            sevnet_viz_path=viz_path,
            results_base_dir=str(latest_version_path),
            source_domain=source_domain
        )
        
        # Source Domain 평가 후 이미지 경로 탐색
        try:
            print(f"   🔍 Source Domain 평가 후 이미지 경로 탐색...")
            anomalib_image_paths = []
            
            # DRAEM 이미지 경로 패턴 검색 (tensorboard_logs 기준)
            patterns = [
                "**/Draem/latest/images"  # 실제 생성되는 이미지 경로
            ]
            for pattern in patterns:
                found_paths = list(latest_version_path.parent.glob(pattern))
                anomalib_image_paths.extend(found_paths)
            
            # 중복 제거
            anomalib_image_paths = list(set(anomalib_image_paths))
            print(f"   📂 발견된 이미지 경로들: {[str(p) for p in anomalib_image_paths]}")
            
            # 가장 최신 이미지 경로 선택
            if anomalib_image_paths:
                latest_image_path = max(anomalib_image_paths, key=lambda p: p.stat().st_mtime if p.exists() else 0)
                anomalib_results_path = latest_image_path.parent  # images 폴더의 부모
                print(f"   ✅ 실제 Anomalib 결과 경로: {anomalib_results_path}")
            else:
                print(f"   ⚠️ Warning: 평가 후에도 이미지 경로를 찾을 수 없습니다")
                anomalib_results_path = None
                
        except Exception as e:
            print(f"   ⚠️ Warning: 평가 후 이미지 경로 탐색 실패: {e}")
            anomalib_results_path = None
        
        # Target Domains 평가
        print(f"\n🎯 Target Domains 평가 시작")
        if logger:
            logger.info("🎯 Target Domains 평가 시작")
        
        target_results = evaluate_target_domains(
            model=model,
            engine=engine_for_eval,
            datamodule=datamodule,
            checkpoint_path=best_checkpoint,
            results_base_dir=str(anomalib_results_path) if anomalib_results_path else str(latest_version_path),
            save_samples=True,  # Target Domain 이미지 복사 활성화
            current_version_path=str(latest_version_path) if latest_version_path else None  # 시각화 폴더는 TensorBoard 경로
        )
        
        # 훈련 정보 추출
        training_info = extract_training_info(engine)
        
        # 결과 분석
        analysis = analyze_experiment_results(
            source_results=source_results,
            target_results=target_results,
            training_info=training_info,
            condition=condition,
            model_type="DRAEM"
        )
        
        # 실험 결과 정리 (공통 함수 활용)
        experiment_result = create_common_experiment_result(
            condition=condition,
            status="success",
            experiment_path=str(latest_version_path),
            source_results=source_results,
            target_results=target_results,
            training_info=training_info,
            best_checkpoint=best_checkpoint
        )
        
        # 결과 저장 (공통 함수 활용)
        result_filename = f"result_{condition['name']}_{timestamp}.json"
        save_experiment_results(
            result=experiment_result,
            result_filename=result_filename,
            log_dir=latest_version_path,
            logger=logger or logging.getLogger(__name__),
            model_type="DRAEM"
        )
        
        print(f"\n✅ 실험 완료: {experiment_name}")
        if logger:
            logger.info(f"✅ 실험 완료: {experiment_name}")
        
        return experiment_result
        
    except Exception as e:
        error_msg = f"실험 실패: {e}"
        print(f"\n❌ {error_msg}")
        if logger:
            logger.error(f"❌ {error_msg}")
        
        return create_common_experiment_result(
            condition=condition,
            status="failed",
            error=str(e)
        )


def main():
    """메인 함수 - 실험 설정 및 실행."""
    parser = argparse.ArgumentParser(description="HDMAP 다중 도메인 DRAEM 실험 스크립트")
    parser.add_argument("--experiment_name", type=str, required=True, help="실험 조건 이름 (JSON 파일에 정의된)")
    parser.add_argument("--results_dir", type=str, default="./results", help="결과 저장 디렉토리")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="로그 레벨")
    
    args = parser.parse_args()
    
    # 경고 필터링 설정
    setup_warnings_filter()
    
    # 로그 디렉토리 생성 (results_dir 내부에)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 로거 설정 (DraemSevNet과 동일하게 고정된 로그 파일명 사용)
    import re
    # results_dir에서 timestamp 추출 (예: results/draem/20250817_120156)
    dir_parts = str(results_dir).split('/')
    run_timestamp = None
    for part in dir_parts:
        if re.match(r'\d{8}_\d{6}', part):  # 20250817_120156 패턴
            run_timestamp = part
            break
    
    if not run_timestamp:
        # 직접 실행된 경우: 새로운 timestamp 생성
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = results_dir / f"draem_experiment_{run_timestamp}.log"
    logger = setup_experiment_logging(str(log_file), "draem_experiment")
    logger.setLevel(getattr(logging, args.log_level))
    
    print(f"\n🚀 HDMAP 다중 도메인 DRAEM 실험 시작")
    print(f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📝 로그 파일: {log_file}")
    
    logger.info("🚀 HDMAP 다중 도메인 DRAEM 실험 시작")
    logger.info(f"명령행 인수: {vars(args)}")
    
    all_results = []
    
    try:
        # JSON 파일에서 실험 조건 찾기
        condition = None
        for c in EXPERIMENT_CONDITIONS:
            if c["name"] == args.experiment_name:
                condition = c
                break
        
        if not condition:
            print(f"❌ 실험 조건을 찾을 수 없습니다: {args.experiment_name}")
            print(f"📋 사용 가능한 실험 조건들:")
            for c in EXPERIMENT_CONDITIONS:
                print(f"   - {c['name']}: {c['description']}")
            logger.error(f"실험 조건을 찾을 수 없음: {args.experiment_name}")
            return
        
        print(f"\n🎯 실험 실행: {condition['name']}")
        print(f"📄 설명: {condition['description']}")
        logger.info(f"🎯 실험 실행: {condition['name']}")
        
        result = run_single_draem_experiment(
            condition=condition,
            dataset_root=None,  # JSON 설정 사용
            results_base_dir=args.results_dir,
            logger=logger
        )
        
        all_results.append(result)
        
        print(f"\n🎉 실험 완료!")
        print(f"📁 결과 디렉토리: {args.results_dir}")
        print(f"📝 로그 파일: {log_file}")
        
        if result.get("status") == "success":
            print(f"✅ {condition['name']} 실험이 성공적으로 완료되었습니다!")
        else:
            print(f"❌ {condition['name']} 실험이 실패했습니다.")
        
        logger.info("🎉 실험 완료!")
        logger.info(f"📁 결과 디렉토리: {args.results_dir}")
        
    except Exception as e:
        error_msg = f"실험 도중 오류 발생: {e}"
        print(f"\n❌ {error_msg}")
        logger.error(f"❌ {error_msg}")
        raise


if __name__ == "__main__":
    main()
