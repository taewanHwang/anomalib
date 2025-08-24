#!/usr/bin/env python3
"""MultiDomain HDMAP EfficientAD 도메인 전이 학습 예시.

EfficientAD 모델과 MultiDomainHDMAPDataModule을 활용한 효율적인 도메인 전이 학습 실험 스크립트입니다.

EfficientAD 특징:
- Student-Teacher 구조: Pre-trained EfficientNet teacher + lightweight student
- Autoencoder 추가: Student-Autoencoder discrepancy로 전역 이상 탐지
- 밀리초 수준 추론: 매우 빠른 추론 속도
- ImageNet 학습: Teacher-Student 불일치를 위해 ImageNet 데이터 활용
- 이중 탐지: Teacher-Student discrepancy (지역) + Student-Autoencoder discrepancy (전역)

실험 구조:
1. MultiDomainHDMAPDataModule 설정 (e.g. source: domain_A, targets: domain_B,C,D)
2. Source Domain에서 EfficientAD 모델 훈련 (train 데이터)
3. Source Domain에서 성능 평가 (validation으로 사용될 test 데이터)
4. Target Domains에서 동시 성능 평가 (각 도메인별 test 데이터)
5. 도메인 전이 효과 종합 분석

주요 개선점 (EfficientAD vs PatchCore):
- 학습 기반 접근: Teacher-Student 지식 증류로 더 정교한 특징 학습
- 이중 탐지 메커니즘: 지역적 + 전역적 이상 탐지 능력
- 초고속 추론: 밀리초 수준의 매우 빠른 추론 속도
- 안정적 훈련: EfficientNet backbone으로 안정적 학습

NOTE:
- 실험 조건들은 multi_domain_hdmap_efficientad_exp_condition.json 파일에서 관리됩니다.
- EfficientAD는 학습이 필요하므로 early stopping, optimizer 설정이 중요합니다.
- ImageNet/Imagenette 데이터셋이 필수적으로 필요합니다.
- batch_size=1 권장 (논문 설정)
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
from anomalib.models.image.efficient_ad import EfficientAd
from anomalib.engine import Engine
from pytorch_lightning.loggers import TensorBoardLogger

# Early Stopping import (학습이 필요한 모델)
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
EXPERIMENT_CONDITIONS = load_experiment_conditions("multi_domain_hdmap_efficientad-exp_condition-test.json")

# 경고 메시지 비활성화
setup_warnings_filter()
logging.getLogger("anomalib.visualization.image.item_visualizer").setLevel(logging.ERROR)
logging.getLogger("anomalib.visualization").setLevel(logging.ERROR)
logging.getLogger("anomalib.callbacks").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")

# ========================================================================================
# 모델 훈련 및 실험 함수들
# ========================================================================================

def train_efficientad_model_multi_domain(
    datamodule: MultiDomainHDMAPDataModule,
    config: Dict[str, Any],
    results_base_dir: str,
    logger: logging.Logger
) -> tuple:
    """EfficientAD 모델을 Multi-Domain 설정으로 훈련합니다."""
    
    print(f"🤖 EfficientAD 모델 생성 중...")
    logger.info("🤖 EfficientAD 모델 생성 중...")
    
    # EfficientAD 모델 생성
    model = EfficientAd(
        imagenet_dir=config["imagenet_dir"],
        teacher_out_channels=config["teacher_out_channels"],
        model_size=config["model_size"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        padding=config["padding"],
        pad_maps=config["pad_maps"],
    )
    
    print(f"   ✅ EfficientAD 모델 생성 완료")
    print(f"   📊 특징: Student-Teacher + Autoencoder, Fast Inference")
    logger.info("📊 EfficientAD 특징: Student-Teacher + Autoencoder, Fast Inference")
    
    # Early Stopping 및 Checkpoint 설정 (EfficientAD는 train loss 기반)
    early_stopping = EarlyStopping(
        monitor="train_loss_epoch",
        patience=config["early_stopping_patience"],
        mode="min",
        verbose=True
    )
    
    checkpoint_callback = ModelCheckpoint(
        filename=f"efficientad_multi_domain_{datamodule.source_domain}_" + "{epoch:02d}_{train_loss_epoch:.4f}",
        monitor="train_loss_epoch",
        mode="min",
        save_top_k=1,
        verbose=True
    )
    
    print(f"   📊 Early Stopping: patience={config['early_stopping_patience']}, monitor=train_loss_epoch (min)")
    print(f"   💾 Model Checkpoint: monitor=train_loss_epoch (min), save_top_k=1")
    logger.info(f"📊 Early Stopping 설정: patience={config['early_stopping_patience']}, monitor=train_loss_epoch")
    logger.info(f"💾 Model Checkpoint 설정: monitor=train_loss_epoch")
    
    # TensorBoard 로거 설정
    tb_logger = TensorBoardLogger(
        save_dir=results_base_dir,
        name="tensorboard_logs",
        version=""  # 빈 버전으로 version_x 폴더 방지
    )
    
    # Engine 설정
    engine = Engine(
        max_epochs=config["max_epochs"],
        callbacks=[early_stopping, checkpoint_callback],
        logger=tb_logger,
        accelerator="auto",
        devices=1,
        enable_checkpointing=True,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        default_root_dir=results_base_dir,  # 모든 결과를 동일한 기본 디렉터리에 저장
    )
    
    print(f"⚙️ Engine 설정:")
    print(f"   📊 Max Epochs: {config['max_epochs']}")
    print(f"   🔧 Device: auto")
    print(f"   📝 Check Validation: 매 에폭")
    logger.info(f"⚙️ Engine 설정: max_epochs={config['max_epochs']}")
    
    # 훈련 실행
    print(f"🚀 EfficientAD 모델 훈련 시작...")
    logger.info("🚀 EfficientAD 모델 훈련 시작")
    
    engine.fit(
        model=model,
        datamodule=datamodule
    )
    
    # 최적 체크포인트 경로 가져오기
    best_checkpoint = None
    if checkpoint_callback.best_model_path:
        best_checkpoint = str(checkpoint_callback.best_model_path)
        print(f"   💾 최적 체크포인트: {best_checkpoint}")
        logger.info(f"💾 최적 체크포인트: {best_checkpoint}")
    else:
        print(f"   ⚠️ 체크포인트를 찾을 수 없습니다.")
        logger.warning("⚠️ 체크포인트를 찾을 수 없습니다.")
    
    print(f"   ✅ EfficientAD 훈련 완료! (val_image_AUROC 최적화)")
    logger.info("✅ EfficientAD 훈련 완료! (val_image_AUROC 최적화)")
    
    return model, engine, best_checkpoint


def run_single_efficientad_experiment(
    condition: dict,
    log_dir: str = None
) -> dict:
    """단일 EfficientAD 실험 조건에 대한 실험 수행."""
    
    # config에서 도메인 설정 가져오기
    config = condition["config"]
    source_domain = config["source_domain"]
    target_domains = extract_target_domains_from_config(config)
    
    # 각 실험마다 고유한 results 경로 생성 (DRAEM SevNet과 동일한 구조)
    from datetime import datetime
    if log_dir:
        # run 스크립트에서 호출된 경우: 기존 timestamp 폴더 재사용
        base_timestamp_dir = log_dir
        timestamp_for_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = f"{condition['name']}_{timestamp_for_folder}"
    else:
        # 직접 호출된 경우: 새로운 timestamp 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_timestamp_dir = f"results/efficientad/{timestamp}"
        experiment_folder = f"{condition['name']}_{timestamp}"
    
    results_base_dir = f"{base_timestamp_dir}/MultiDomainHDMAP/efficientad/{experiment_folder}"
    
    os.makedirs(results_base_dir, exist_ok=True)
    
    print(f"================================================================================")
    print(f"🚀 EfficientAD 실험 시작: {condition['name']}")
    print(f"================================================================================")
    print(f"\n🔬 실험 조건:")
    print(f"   📝 이름: {condition['name']}")
    print(f"   💬 설명: {condition['description']}")
    print(f"   🎯 Source Domain: {source_domain}")
    print(f"   🎯 Target Domains: {target_domains}")
    print(f"   📁 Results Dir: {results_base_dir}")
    print(f"================================================================================")
    
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
        fitted_model, engine, best_checkpoint = train_efficientad_model_multi_domain(
            datamodule=multi_datamodule,
            config=condition["config"],
            results_base_dir=results_base_dir,
            logger=logging.getLogger(__name__)
        )
        
        # Source Domain 성능 평가
        print(f"\n📊 Source Domain 성능 평가 - {condition['name']}")
        source_results = evaluate_source_domain(
            model=fitted_model,
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
            
            # 2. 실제 Anomalib 이미지 경로 탐색
            anomalib_image_paths = []
            base_search_path = Path(results_base_dir)
            
            # EfficientAD 이미지 경로 패턴 검색
            patterns = [
                "**/EfficientAd/MultiDomainHDMAPDataModule/*/images",  # v0, v1 등의 버전 폴더
                "**/EfficientAd/latest/images"  # latest 링크가 있는 경우
            ]
            for pattern in patterns:
                found_paths = list(base_search_path.glob(pattern))
                if found_paths:
                    anomalib_image_paths.extend(found_paths)
                    print(f"   🔍 발견된 이미지 경로: {found_paths}")
            
            # 3. 가장 최신 버전 찾기 (v 뒤의 숫자가 가장 큰 것)
            latest_version_path = None
            if anomalib_image_paths:
                # 버전 번호로 정렬 (v0, v1, v2, ... 순)
                version_paths = []
                for path in anomalib_image_paths:
                    # 경로에서 v0, v1 등의 버전 추출
                    for part in path.parts:
                        if part.startswith('v') and part[1:].isdigit():
                            version_num = int(part[1:])
                            version_paths.append((version_num, path))
                            break
                
                if version_paths:
                    # 가장 높은 버전 선택
                    version_paths.sort(reverse=True)  # 내림차순 정렬
                    latest_version_path = version_paths[0][1]
                    print(f"   📂 선택된 최신 이미지 경로: {latest_version_path}")
                else:
                    latest_version_path = anomalib_image_paths[0]
                    print(f"   📂 기본 이미지 경로 사용: {latest_version_path}")
            
            # 4. anomalib_results_path는 images 폴더의 상위 디렉터리
            anomalib_results_path = None
            if latest_version_path:
                # images의 상위 폴더가 실제 결과 디렉터리
                anomalib_results_path = latest_version_path.parent
                print(f"   📂 Anomalib 결과 경로: {anomalib_results_path}")
        
        except Exception as e:
            print(f"   ⚠️ 이미지 경로 탐색 중 오류: {e}")
            anomalib_results_path = None
            latest_version_path = None
        
        # Target Domains 성능 평가
        print(f"\n🎯 Target Domains 성능 평가 - {condition['name']}")
        target_results = evaluate_target_domains(
            model=fitted_model,
            engine=engine,
            datamodule=multi_datamodule,
            checkpoint_path=best_checkpoint,
            results_base_dir=str(anomalib_results_path) if anomalib_results_path else results_base_dir,  # 🎯 실제 Anomalib 이미지 경로
            save_samples=True,  # 🎯 Target Domain 이미지 복사 활성화
            current_version_path=f"{results_base_dir}/tensorboard_logs"  # 🎯 시각화 폴더는 TensorBoard 경로
        )
        
        # 시각화 폴더 생성
        if latest_version_path:
            # EfficientAD 시각화 폴더 생성 (target_results 이후에 실행)
            efficientad_viz_path_str = create_experiment_visualization(
                experiment_name=condition['name'],
                model_type="EfficientAD",
                source_domain=source_domain,
                target_domains=target_domains,
                results_base_dir=f"{results_base_dir}/tensorboard_logs"  # DRAEM SevNet처럼 tensorboard_logs 하위에 생성
            )
            efficientad_viz_path = Path(efficientad_viz_path_str)
            
            # Source Domain 이미지 복사
            if anomalib_results_path:
                source_success = organize_source_domain_results(
                    sevnet_viz_path=str(efficientad_viz_path),
                    results_base_dir=str(anomalib_results_path),  # 실제 Anomalib 이미지가 있는 경로
                    source_domain=source_domain
                )
                
                if source_success:
                    print(f"   ✅ Source Domain 이미지 복사 완료")
                else:
                    print(f"   ⚠️ Source Domain 이미지 복사 실패")
            else:
                print(f"   ⚠️ Anomalib 결과 경로를 찾을 수 없어 이미지 복사 생략")
        else:
            print(f"   ⚠️ 이미지 경로를 찾을 수 없어 시각화 생략")
        
        # 학습 과정 정보 추출
        training_info = extract_training_info(engine)
        
        # 결과 분석
        analysis = analyze_experiment_results(
            source_results=source_results,
            target_results=target_results,
            training_info=training_info,
            model_type="EfficientAD",
            condition=condition
        )
        
        print(f"\n📊 실험 결과 요약:")
        print(f"   🎯 Source Domain AUROC: {source_results.get('image_AUROC', 'N/A'):.4f}" if isinstance(source_results.get('image_AUROC'), (int, float)) else f"   🎯 Source Domain AUROC: {source_results.get('image_AUROC', 'N/A')}")
        
        target_aurocs = []
        for domain, result in target_results.items():
            auroc = result.get('image_AUROC', 0)  # image_AUROC로 변경
            if isinstance(auroc, (int, float)):
                print(f"   🎯 {domain} AUROC: {auroc:.4f}")
                target_aurocs.append(auroc)
            else:
                print(f"   🎯 {domain} AUROC: {auroc}")
        
        if target_aurocs:
            avg_auroc = sum(target_aurocs) / len(target_aurocs)
            print(f"   📊 Target Domains 평균 AUROC: {avg_auroc:.4f}")
        
        return {
            'condition': condition,
            'source_results': source_results,
            'target_results': target_results,
            'training_info': training_info,
            'analysis': analysis,
            'best_checkpoint': best_checkpoint,
            'status': 'success'
        }
        
    except Exception as e:
        error_msg = f"EfficientAD 실험 실패 - {condition['name']}: {str(e)}"
        print(f"❌ {error_msg}")
        logging.getLogger(__name__).error(error_msg)
        import traceback
        traceback.print_exc()
        
        return {
            'condition': condition,
            'status': 'failed',
            'error': error_msg,
            'source_results': {},
            'target_results': {},
            'analysis': {},
            'best_checkpoint': None
        }
    
    finally:
        # 메모리 정리
        cleanup_gpu_memory()

def main():
    """메인 함수 - PatchCore와 동일한 구조"""
    parser = argparse.ArgumentParser(description="EfficientAD MultiDomain HDMAP 실험 실행")
    parser.add_argument("--gpu-id", type=int, default=0, help="사용할 GPU ID")
    parser.add_argument("--experiment-id", type=int, default=0, help="실험 ID (병렬 실행용)")
    parser.add_argument("--results-dir", type=str, default="results/efficientad", help="결과 저장 디렉터리")
    parser.add_argument("--get-experiment-count", action="store_true", help="실험 조건 개수만 출력")
    
    args = parser.parse_args()
    
    if args.get_experiment_count:
        print(len(EXPERIMENT_CONDITIONS))
        return
    
    if not EXPERIMENT_CONDITIONS:
        print("❌ 실험 조건이 없습니다.")
        return
    
    # GPU 설정 및 실험 조건 검증
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    if args.experiment_id >= len(EXPERIMENT_CONDITIONS):
        print(f"❌ 잘못된 실험 ID: {args.experiment_id} (최대: {len(EXPERIMENT_CONDITIONS)-1})")
        return

    condition = EXPERIMENT_CONDITIONS[args.experiment_id]
    
    # 로그 디렉터리 설정
    log_dir = Path(args.results_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 로깅 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"efficientad_experiment_{timestamp}.log"
    logger = setup_experiment_logging(str(log_path), f"efficientad_{condition['name']}")
    
    # GPU 메모리 정리
    cleanup_gpu_memory()
    
    try:
        # 실험 실행
        result = run_single_efficientad_experiment(
            condition=condition,
            log_dir=str(log_dir)
        )
        
        # 결과 저장
        result_filename = f"result_exp_{args.experiment_id:02d}_{condition['name']}_gpu{args.gpu_id}_{timestamp}.json"
        save_experiment_results(result, result_filename, log_dir, logger)
        
        logger.info("✅ 실험 완료!")
        print("✅ 실험 완료!")
        
    except Exception as e:
        logger.error(f"❌ 실험 중 오류 발생: {e}")
        print(f"❌ 실험 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        cleanup_gpu_memory()
        logger.info("🧹 GPU 메모리 정리 완료")

if __name__ == "__main__":
    main()
