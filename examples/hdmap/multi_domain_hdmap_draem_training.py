#!/usr/bin/env python3
"""HDMAP 다중 도메인 DRAEM 모델 훈련 및 평가 스크립트.

이 스크립트는 HDMAP 데이터셋에서 DRAEM 모델을 훈련하고 다중 도메인 평가를 수행합니다.

주요 기능:
- DRAEM 모델을 사용한 이상 탐지
- 소스 도메인(domain_A)에서 훈련
- 타겟 도메인들(domain_B, C, D)에서 평가
- 실험 결과 시각화 및 저장
- 체계적인 실험 조건 관리

사용법:
    python multi_domain_hdmap_draem_training.py --experiment_name my_experiment --max_epochs 50
    python multi_domain_hdmap_draem_training.py --run_all_experiments

"""

import argparse
import json
import logging
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from anomalib.models.image.draem import Draem
from anomalib.data.datamodules.image.multi_domain_hdmap import MultiDomainHDMAPDataModule
from anomalib.engine import Engine
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Experiment utilities import
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

# 경고 메시지 비활성화 (DraemSevNet과 동일)
setup_warnings_filter()
logging.getLogger("anomalib.visualization.image.item_visualizer").setLevel(logging.ERROR)
logging.getLogger("anomalib.visualization").setLevel(logging.ERROR)
logging.getLogger("anomalib.callbacks").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")



# 실험 조건 정의
EXPERIMENT_CONDITIONS = [
    {
        "name": "DRAEM_quick_3epochs",
        "description": "DRAEM 간단 테스트 (3 에포크)",
        "config": {
            "max_epochs": 3,
            "early_stopping_patience": 1,
            "learning_rate": 0.0001,
            "batch_size": 16,
            "image_size": "224x224"
        }
    },
    {
        "name": "DRAEM_baseline_50epochs",
        "description": "DRAEM 기본 설정 (50 에포크)",
        "config": {
            "max_epochs": 50,
            "early_stopping_patience": 10,
            "learning_rate": 0.0001,
            "batch_size": 16,
            "image_size": "224x224"
        }
    },
    {
        "name": "DRAEM_extended_100epochs",
        "description": "DRAEM 확장 훈련 (100 에포크)",
        "config": {
            "max_epochs": 100,
            "early_stopping_patience": 15,
            "learning_rate": 0.0001,
            "batch_size": 16,
            "image_size": "224x224"
        }
    },
    {
        "name": "DRAEM_lower_lr",
        "description": "DRAEM 낮은 학습률 (0.00005)",
        "config": {
            "max_epochs": 50,
            "early_stopping_patience": 10,
            "learning_rate": 0.00005,
            "batch_size": 16,
            "image_size": "224x224"
        }
    },
    {
        "name": "DRAEM_higher_lr",
        "description": "DRAEM 높은 학습률 (0.0002)",
        "config": {
            "max_epochs": 50,
            "early_stopping_patience": 10,
            "learning_rate": 0.0002,
            "batch_size": 16,
            "image_size": "224x224"
        }
    },
    {
        "name": "DRAEM_larger_batch",
        "description": "DRAEM 큰 배치 크기 (32)",
        "config": {
            "max_epochs": 50,
            "early_stopping_patience": 10,
            "learning_rate": 0.0001,
            "batch_size": 32,
            "image_size": "224x224"
        }
    },
    {
        "name": "DRAEM_smaller_batch",
        "description": "DRAEM 작은 배치 크기 (8)",
        "config": {
            "max_epochs": 50,
            "early_stopping_patience": 10,
            "learning_rate": 0.0001,
            "batch_size": 8,
            "image_size": "224x224"
        }
    },
    {
        "name": "DRAEM_longer_patience",
        "description": "DRAEM 긴 patience (20)",
        "config": {
            "max_epochs": 50,
            "early_stopping_patience": 20,
            "learning_rate": 0.0001,
            "batch_size": 16,
            "image_size": "224x224"
        }
    },
    {
        "name": "DRAEM_shorter_patience",
        "description": "DRAEM 짧은 patience (5)",
        "config": {
            "max_epochs": 50,
            "early_stopping_patience": 5,
            "learning_rate": 0.0001,
            "batch_size": 16,
            "image_size": "224x224"
        }
    },
    {
        "name": "DRAEM_quick_test",
        "description": "DRAEM 빠른 테스트 (5 에포크)",
        "config": {
            "max_epochs": 5,
            "early_stopping_patience": 3,
            "learning_rate": 0.0001,
            "batch_size": 8,
            "image_size": "224x224"
        }
    }
]


def train_draem_model_multi_domain(
    datamodule: MultiDomainHDMAPDataModule, 
    config: Dict[str, Any],
    results_base_dir: str,
    experiment_name: str,
    logger: logging.Logger
) -> tuple[Draem, Engine, str]:
    """DRAEM 모델 훈련 수행.
    
    Args:
        datamodule: 설정된 MultiDomainHDMAPDataModule
        config: 훈련 설정 딕셔너리
        results_base_dir: 결과 저장 기본 경로
        experiment_name: 실험 이름
        logger: 로거 객체
        
    Returns:
        tuple: (훈련된 모델, Engine 객체, 체크포인트 경로)
    """
    print(f"\n🚀 DRAEM 모델 훈련 시작")
    logger.info("🚀 DRAEM 모델 훈련 시작")
    
    # DRAEM 모델 초기화 (validation loss 포함)
    model = Draem()
    print(f"   ✅ DRAEM 모델 생성 완료 (validation loss 포함)")
    logger.info("✅ DRAEM 모델 생성 완료 (validation loss 포함)")
    
    # Early stopping과 model checkpoint 설정
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config["early_stopping_patience"],
        mode="min",
        verbose=True
    )
    
    # 체크포인트 경로 설정
    checkpoint_callback = ModelCheckpoint(
        filename=f"draem_multi_domain_{datamodule.source_domain}_" + "{epoch:02d}_{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True
    )
    
    print(f"   📊 Early Stopping: patience={config['early_stopping_patience']}, monitor=val_loss")
    print(f"   💾 Model Checkpoint: monitor=val_loss, save_top_k=1")
    logger.info(f"📊 Early Stopping 설정: patience={config['early_stopping_patience']}")
    logger.info(f"💾 Model Checkpoint 설정: monitor=val_loss")
    
    # TensorBoard 로거 설정 (DraemSevNet과 동일)
    tb_logger = TensorBoardLogger(
        save_dir=results_base_dir,
        name="tensorboard_logs",
        version=""  # 빈 버전으로 version_x 폴더 방지
    )
    
    # Engine 생성 및 훈련
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


def analyze_draem_results(
    source_results: Dict[str, Any],
    target_results: Dict[str, Dict[str, Any]],
    training_info: Dict[str, Any],
    condition: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Any]:
    """DRAEM 실험 결과 분석.
    
    Args:
        source_results: 소스 도메인 평가 결과
        target_results: 타겟 도메인 평가 결과
        training_info: 훈련 정보
        condition: 실험 조건
        logger: 로거 객체
        
    Returns:
        Dict[str, Any]: 분석된 결과 딕셔너리
    """
    print(f"\n📊 DRAEM 실험 결과 분석")
    logger.info("📊 DRAEM 실험 결과 분석 시작")
    
    # 타겟 도메인 평균 AUROC 계산
    target_aurocs = []
    for domain, result in target_results.items():
        if isinstance(result.get('image_AUROC'), (int, float)):
            target_aurocs.append(result['image_AUROC'])
    
    avg_target_auroc = sum(target_aurocs) / len(target_aurocs) if target_aurocs else 0.0
    
    # 소스 도메인 AUROC
    source_auroc = source_results.get('image_AUROC', 0.0) if source_results else 0.0
    
    # 도메인 전이 효과 계산
    transfer_ratio = avg_target_auroc / source_auroc if source_auroc > 0 else 0.0
    
    # 성능 평가
    if transfer_ratio > 0.9:
        transfer_grade = "우수"
    elif transfer_ratio > 0.8:
        transfer_grade = "양호"
    elif transfer_ratio > 0.7:
        transfer_grade = "보통"
    else:
        transfer_grade = "개선필요"
    
    # 결과 요약
    analysis = {
        "experiment_name": condition["name"],
        "source_auroc": source_auroc,
        "avg_target_auroc": avg_target_auroc,
        "transfer_ratio": transfer_ratio,
        "transfer_grade": transfer_grade,
        "target_domain_count": len(target_results),
        "training_epochs": training_info.get("last_trained_epoch", 0),
        "early_stopped": training_info.get("early_stopped", False),
        "best_val_auroc": training_info.get("best_val_auroc", 0.0)
    }
    
    # 도메인별 상세 성능
    domain_performances = {}
    for domain, result in target_results.items():
        domain_performances[domain] = {
            "auroc": result.get('image_AUROC', 0.0),
            "f1_score": result.get('image_F1Score', 0.0)
        }
    
    analysis["domain_performances"] = domain_performances
    
    # 로깅
    print(f"   📈 Source AUROC: {source_auroc:.4f}")
    print(f"   🎯 Target 평균 AUROC: {avg_target_auroc:.4f}")
    print(f"   🔄 전이 비율: {transfer_ratio:.3f} ({transfer_grade})")
    print(f"   📚 훈련 에포크: {analysis['training_epochs']}")
    
    logger.info(f"📈 Source AUROC: {source_auroc:.4f}")
    logger.info(f"🎯 Target 평균 AUROC: {avg_target_auroc:.4f}")
    logger.info(f"🔄 전이 비율: {transfer_ratio:.3f} ({transfer_grade})")
    logger.info(f"📚 훈련 에포크: {analysis['training_epochs']}")
    
    for domain, perf in domain_performances.items():
        print(f"   └─ {domain}: AUROC={perf['auroc']:.4f}")
        logger.info(f"   └─ {domain}: AUROC={perf['auroc']:.4f}")
    
    return analysis


def run_single_draem_experiment(
    condition: Dict[str, Any],
    source_domain: str = "domain_A",
    target_domains: List[str] = None,
    dataset_root: str = None,
    results_base_dir: str = "./results",
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """단일 DRAEM 실험 수행.
    
    Args:
        condition: 실험 조건 딕셔너리
        source_domain: 소스 도메인
        target_domains: 타겟 도메인 리스트
        dataset_root: 데이터셋 루트 경로
        results_base_dir: 결과 저장 기본 경로
        logger: 로거 객체
        
    Returns:
        Dict[str, Any]: 실험 결과
    """
    if target_domains is None:
        target_domains = ["domain_B", "domain_C", "domain_D"]
    
    experiment_name = condition["name"]
    config = condition["config"]
    
    print(f"\n{'='*80}")
    print(f"🧪 DRAEM 실험 시작: {experiment_name}")
    print(f"{'='*80}")
    
    if logger:
        logger.info(f"🧪 DRAEM 실험 시작: {experiment_name}")
        logger.info(f"실험 설정: {config}")
    
    try:
        # GPU 메모리 정리
        cleanup_gpu_memory()
        
        # 실험별 결과 디렉토리 생성 (DraemSevNet과 동일한 구조)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = f"{experiment_name}_{timestamp}"
        # DraemSevNet과 동일한 구조로 생성
        experiment_dir = Path(results_base_dir) / "MultiDomainHDMAP" / "draem" / experiment_folder
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
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
            experiment_name=experiment_name,
            logger=logger or logging.getLogger(__name__)
        )
        
        # 훈련 정보 추출
        training_info = extract_training_info(engine)
        
        # 실제 Anomalib 결과 경로 찾기 (DraemSevNet과 동일한 로직)
        try:
            # 1. TensorBoardLogger 경로 확인
            trainer_log_dir = None
            if hasattr(engine.trainer, 'logger') and hasattr(engine.trainer.logger, 'log_dir'):
                trainer_log_dir = Path(engine.trainer.logger.log_dir)
                print(f"   📂 Trainer log_dir: {trainer_log_dir}")
            
            # 2. 실제 Anomalib 이미지 경로 탐색
            anomalib_image_paths = []
            base_search_path = Path(str(experiment_dir))
            
            # DRAEM 이미지 경로 패턴 검색 (실제 구조에 맞춤)
            patterns = [
                "**/Draem/MultiDomainHDMAPDataModule/**/images",  # 원래 예상 패턴
                "**/Draem/latest/images",                        # 실제 생성된 패턴
                "**/Draem/**/images",                           # 더 넓은 패턴
                "**/images"                                     # 마지막 fallback
            ]
            for pattern in patterns:
                found_paths = list(base_search_path.glob(pattern))
                anomalib_image_paths.extend(found_paths)
            
            # 중복 제거
            anomalib_image_paths = list(set(anomalib_image_paths))
            print(f"   🔍 발견된 이미지 경로들: {[str(p) for p in anomalib_image_paths]}")
            
            # 가장 최신 이미지 경로 선택
            if anomalib_image_paths:
                latest_image_path = max(anomalib_image_paths, key=lambda p: p.stat().st_mtime if p.exists() else 0)
                anomalib_results_path = latest_image_path.parent  # images 폴더의 부모
                print(f"   ✅ 실제 Anomalib 결과 경로: {anomalib_results_path}")
                actual_results_dir = str(anomalib_results_path)
            else:
                print(f"   ⚠️ Warning: 이미지 경로를 찾을 수 없습니다")
                
            # 시각화 폴더는 TensorBoardLogger 경로에 생성
            if trainer_log_dir:
                latest_version_path = trainer_log_dir
            else:
                latest_version_path = Path(str(experiment_dir))
                
        except Exception as e:
            print(f"   ⚠️ Warning: 실제 이미지 경로 찾기 실패: {e}")
            latest_version_path = Path(str(experiment_dir))
        
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
        
        # Target Domains 평가
        print(f"\n🎯 Target Domains 평가 시작")
        if logger:
            logger.info("🎯 Target Domains 평가 시작")
        
        target_results = evaluate_target_domains(
            model=model,
            engine=engine_for_eval,
            datamodule=datamodule,
            checkpoint_path=best_checkpoint,
            results_base_dir=str(latest_version_path),
            current_version_path=str(latest_version_path)
        )
        
        # 결과 분석
        analysis = analyze_draem_results(
            source_results=source_results,
            target_results=target_results,
            training_info=training_info,
            condition=condition,
            logger=logger or logging.getLogger(__name__)
        )
        
        # 실험 결과 정리 (DraemSevNet과 동일한 구조)
        experiment_result = {
            "condition": condition,
            "experiment_name": experiment_name, 
            "source_results": source_results,
            "target_results": target_results,
            "best_checkpoint": best_checkpoint,
            "training_info": training_info,
            "status": "success",
            "experiment_path": str(latest_version_path),  # 실제 결과 경로 사용
            "avg_target_auroc": analysis["avg_target_auroc"]
        }
        
        # DraemSevNet처럼 각 실험의 tensorboard_logs 폴더에 JSON 결과 저장
        result_filename = f"result_{condition['name']}_{timestamp}.json"
        result_path = latest_version_path / result_filename
        
        try:
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(experiment_result, f, indent=2, ensure_ascii=False)
            print(f"📄 실험 결과 JSON 저장: {result_path}")
            if logger:
                logger.info(f"📄 실험 결과 JSON 저장: {result_path}")
        except Exception as e:
            print(f"⚠️  JSON 저장 실패: {e}")
            if logger:
                logger.warning(f"⚠️  JSON 저장 실패: {e}")
        
        print(f"\n✅ 실험 완료: {experiment_name}")
        if logger:
            logger.info(f"✅ 실험 완료: {experiment_name}")
        
        return experiment_result
        
    except Exception as e:
        error_msg = f"실험 실패: {e}"
        print(f"\n❌ {error_msg}")
        if logger:
            logger.error(f"❌ {error_msg}")
        
        return {
            "status": "failed",
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "experiment_name": experiment_name,
            "condition": condition,
            "error": str(e)
        }


def main():
    """메인 함수 - 실험 설정 및 실행."""
    parser = argparse.ArgumentParser(description="HDMAP 다중 도메인 DRAEM 실험 스크립트")
    parser.add_argument("--experiment_name", type=str, help="개별 실험 이름")
    parser.add_argument("--max_epochs", type=int, default=50, help="최대 에포크 수")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="학습률")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--source_domain", type=str, default="domain_A", help="소스 도메인")
    parser.add_argument("--dataset_root", type=str, help="데이터셋 루트 경로")
    parser.add_argument("--results_dir", type=str, default="./results", help="결과 저장 디렉토리")
    parser.add_argument("--run_all_experiments", action="store_true", help="모든 실험 조건 실행")
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
        if args.run_all_experiments:
            # 모든 실험 조건 실행
            print(f"\n📋 전체 실험 실행 - {len(EXPERIMENT_CONDITIONS)}개 조건")
            logger.info(f"📋 전체 실험 실행 - {len(EXPERIMENT_CONDITIONS)}개 조건")
            
            for i, condition in enumerate(EXPERIMENT_CONDITIONS, 1):
                print(f"\n[{i}/{len(EXPERIMENT_CONDITIONS)}] {condition['name']} 시작")
                logger.info(f"[{i}/{len(EXPERIMENT_CONDITIONS)}] {condition['name']} 시작")
                
                result = run_single_draem_experiment(
                    condition=condition,
                    source_domain=args.source_domain,
                    dataset_root=args.dataset_root,
                    results_base_dir=args.results_dir,
                    logger=logger
                )
                
                all_results.append(result)
                
                print(f"✅ [{i}/{len(EXPERIMENT_CONDITIONS)}] {condition['name']} 완료")
                logger.info(f"✅ [{i}/{len(EXPERIMENT_CONDITIONS)}] {condition['name']} 완료")
        else:
            # 단일 실험 실행
            if args.experiment_name:
                # 기존 조건에서 찾기
                condition = None
                for c in EXPERIMENT_CONDITIONS:
                    if c["name"] == args.experiment_name:
                        condition = c
                        break
            
            if not condition:
                # 사용자 정의 조건 생성
                condition = {
                    "name": args.experiment_name or f"custom_draem_{timestamp}",
                    "description": "사용자 정의 DRAEM 실험",
                    "config": {
                        "max_epochs": args.max_epochs,
                        "early_stopping_patience": args.early_stopping_patience,
                        "learning_rate": args.learning_rate,
                        "batch_size": args.batch_size,
                        "image_size": "224x224"
                    }
                }
            
            print(f"\n🎯 단일 실험 실행: {condition['name']}")
            logger.info(f"🎯 단일 실험 실행: {condition['name']}")
            
            result = run_single_draem_experiment(
                condition=condition,
                source_domain=args.source_domain,
                dataset_root=args.dataset_root,
                results_base_dir=args.results_dir,
                logger=logger
            )
            
            all_results.append(result)
        
        # 다중 실험 분석 (2개 이상인 경우)
        if len(all_results) > 1:
            analyze_multi_experiment_results(all_results, args.source_domain)
        
        print(f"\n🎉 모든 실험 완료!")
        print(f"📁 결과 디렉토리: {args.results_dir}")
        print(f"📝 로그 파일: {log_file}")
        
        logger.info("🎉 모든 실험 완료!")
        logger.info(f"📁 결과 디렉토리: {args.results_dir}")
        
    except Exception as e:
        error_msg = f"실험 도중 오류 발생: {e}"
        print(f"\n❌ {error_msg}")
        logger.error(f"❌ {error_msg}")
        raise


if __name__ == "__main__":
    main()
