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
"""

import os
import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
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
    evaluate_source_domain
)

# 경고 메시지 비활성화
setup_warnings_filter()
logging.getLogger("anomalib.visualization.image.item_visualizer").setLevel(logging.ERROR)
logging.getLogger("anomalib.visualization").setLevel(logging.ERROR)
logging.getLogger("anomalib.callbacks").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")

# GPU 설정 - 명령행 인자 또는 환경변수로 설정
# GPU ID는 런타임에 설정됨

def train_draem_sevnet_model_multi_domain(
    datamodule: MultiDomainHDMAPDataModule, 
    experiment_name: str,
    results_base_dir: str,
    max_epochs: int = 15,
    severity_head_mode: str = "single_scale",
    score_combination: str = "simple_average",
    severity_loss_type: str = "mse",
    severity_weight: float = 0.5,
    patch_width_range: tuple = (32, 64),
    patch_ratio_range: tuple = (0.8, 1.2),
    patch_count: int = 1,
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-4,
    early_stopping: bool = True,
    patience: int = 3,
    min_delta: float = 0.005
) -> tuple[DraemSevNet, Engine]:
    """MultiDomain DataModule을 사용한 DRAEM-SevNet 모델 훈련.
    
    Args:
        datamodule: 멀티 도메인 데이터 모듈
        experiment_name: 실험 이름 (로그용)
        max_epochs: 최대 에포크 수 (기본값: 15)
        severity_head_mode: SeverityHead 모드 ("single_scale" 또는 "multi_scale")
        score_combination: Score 결합 방식 ("simple_average", "weighted_average", "maximum")
        severity_loss_type: Severity loss 타입 ("mse" 또는 "smoothl1")
        severity_weight: Severity loss 가중치 (기본값: 0.5)
        patch_width_range: 합성 고장 패치 크기 범위
        patch_ratio_range: 패치 종횡비 범위
        patch_count: 패치 개수
        optimizer_name: 옵티마이저 종류
        learning_rate: 학습률
        early_stopping: Early stopping 활성화 여부
        patience: Early stopping patience
        min_delta: Early stopping min_delta
        
    Returns:
        tuple: (훈련된 모델, Engine 객체)
        
    Note:
        DRAEM-SevNet 특징:
        - DRAEM Backbone (97.5M): Wide ResNet encoder + Discriminative/Reconstructive subnetworks
        - SeverityHead: Discriminative encoder features 직접 활용
        - Multi-task Loss: L2+SSIM (recon) + FocalLoss (seg) + MSE/SmoothL1 (severity)
        - Score Combination: (mask_score + severity_score) / 2
        - Early Stopping: val_image_AUROC 기반 학습 효율성 개선
    """
    
    print(f"\n🤖 DRAEM-SevNet 모델 훈련 시작 - {experiment_name}")
    print(f"   Results Base Dir: {results_base_dir}")
    print(f"   Source Domain: {datamodule.source_domain}")
    print(f"   Max Epochs: {max_epochs}")
    print(f"   Severity Head Mode: {severity_head_mode}")
    print(f"   Score Combination: {score_combination}")
    print(f"   Severity Loss Type: {severity_loss_type}")
    print(f"   Severity Weight: {severity_weight}")
    print(f"   Patch Width Range: {patch_width_range}")
    print(f"   Patch Ratio Range: {patch_ratio_range}")
    print(f"   Patch Count: {patch_count}")
    print(f"   Severity Max: 1.0 (default)")
    print(f"   Optimizer: {optimizer_name}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Early Stopping: {early_stopping} (patience={patience}, min_delta={min_delta})")
    
    # DRAEM-SevNet 모델 생성
    model = DraemSevNet(
        # 🎯 DRAEM-SevNet 아키텍처 설정
        severity_head_mode=severity_head_mode,  # single_scale 또는 multi_scale
        score_combination=score_combination,   # simple_average, weighted_average, maximum
        severity_loss_type=severity_loss_type, # mse 또는 smoothl1
        
        # 🔧 Synthetic Fault Generation 설정 
        patch_width_range=patch_width_range,
        patch_ratio_range=patch_ratio_range,
        patch_count=patch_count,
        
        # 🔧 Loss 가중치 설정
        severity_weight=severity_weight,  # DraemSevNetLoss의 severity 가중치
        
        # 🔧 옵티마이저 설정
        optimizer=optimizer_name,
        learning_rate=learning_rate,
    )
    
    # Callbacks 설정
    callbacks = []
    
    # Early Stopping 설정
    if early_stopping:
        early_stopping_callback = EarlyStopping(
            monitor="val_image_AUROC",  # Source domain validation AUROC
            patience=patience,
            min_delta=min_delta,
            mode="max",
            strict=True,
            verbose=True
        )
        callbacks.append(early_stopping_callback)
        print(f"✅ EarlyStopping: monitor=val_image_AUROC, patience={patience}, min_delta={min_delta}")
    
    # Model Checkpoint 설정
    checkpoint_callback = ModelCheckpoint(
        monitor="val_image_AUROC",
        mode="max",
        save_top_k=1,
        filename=f"draem_sevnet_{datamodule.source_domain}_" + "epoch={epoch:02d}_val_auroc={val_image_AUROC:.4f}",
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    print(f"✅ ModelCheckpoint: monitor=val_image_AUROC, mode=max")
    
    # TensorBoard 로거 설정 (표준 PyTorch Lightning Logger 사용)
    logger = TensorBoardLogger(
        save_dir=results_base_dir,
        name="tensorboard_logs",
        version=""  # 빈 버전으로 version_x 폴더 방지
    )
    
    # Engine 설정 (default_root_dir 직접 전달)
    engine = Engine(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[0] if torch.cuda.is_available() else 1,
        logger=logger,
        max_epochs=max_epochs,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
        log_every_n_steps=10,
        enable_model_summary=True,
        num_sanity_val_steps=0,
        default_root_dir=results_base_dir,  # Engine 생성 시 직접 전달
    )
    
    print(f"✅ Engine default_root_dir 설정: {results_base_dir}")
    
    # 모델 훈련
    print("🔥 훈련 시작...")
    engine.fit(model=model, datamodule=datamodule)
    
    print("✅ 모델 훈련 완료!")
    print(f"   체크포인트: {engine.trainer.checkpoint_callback.best_model_path}")
    
    return model, engine


def analyze_draem_sevnet_results(
    source_domain: str,
    source_results: Dict[str, Any],
    target_results: Dict[str, Dict[str, Any]]
):
    """DRAEM-SevNet 도메인 전이 학습 결과 분석 및 출력."""
    print(f"\n{'='*80}")
    print(f"📈 DRAEM-SevNet 도메인 전이 학습 결과 종합 분석")
    print(f"{'='*80}")
    
    # 성능 요약 테이블
    print(f"\n📊 성능 요약:")
    print(f"{'도메인':<12} {'Image AUROC':<12} {'Pixel AUROC':<12} {'유형':<10} {'설명'}")
    print("-" * 70)
    
    # Source domain 결과
    source_image_auroc = source_results.get('image_AUROC', None)
    source_pixel_auroc = source_results.get('pixel_AUROC', None)
    
    if source_image_auroc is not None:
        print(f"{source_domain:<12} {source_image_auroc:<12.3f} {source_pixel_auroc or 0:<12.3f} {'Source':<10} 베이스라인")
    else:
        print(f"{source_domain:<12} {'N/A':<12} {'N/A':<12} {'Source':<10} 베이스라인 (결과 없음)")
    
    # Target domains 결과
    target_performances = []
    for domain, results in target_results.items():
        target_image_auroc = results.get('image_AUROC', None)
        target_pixel_auroc = results.get('pixel_AUROC', None)
        
        if target_image_auroc is not None:
            print(f"{domain:<12} {target_image_auroc:<12.3f} {target_pixel_auroc or 0:<12.3f} {'Target':<10} 도메인 전이")
            target_performances.append((domain, target_image_auroc, target_pixel_auroc))
    
    # DRAEM-SevNet 특화 분석
    print(f"\n🔍 DRAEM-SevNet 특화 메트릭:")
    print("   ✅ SeverityHead: Discriminative encoder features 직접 활용")
    print("   ✅ Multi-task Learning: Mask + Severity 동시 최적화")
    print("   ✅ Score Combination: (mask_score + severity_score) / 2")
    print("   ✅ Early Stopping: val_image_AUROC 기반 학습 효율성 향상")


def run_single_draem_sevnet_experiment(
    multi_datamodule: MultiDomainHDMAPDataModule,
    condition: dict,
    source_domain: str,
    max_epochs: int,
    log_dir: str = None,
    gpu_id: int = 0,
    experiment_id: int = 0
) -> dict:
    """단일 DRAEM-SevNet 실험 조건에 대한 실험 수행."""
    # 각 실험마다 고유한 results 경로 생성
    import time
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
    
    # DRAEM과 동일한 구조: {base_timestamp_dir}/MultiDomainHDMAP/draem_sevnet/{experiment_name}/
    results_base_dir = f"{base_timestamp_dir}/MultiDomainHDMAP/draem_sevnet/{experiment_folder}"
    
    # 실험 이름 생성
    experiment_name = f"{source_domain}"
    
    print(f"\n{'='*80}")
    print(f"🔬 DRAEM-SevNet 실험 조건: {condition['name']}")
    print(f"📝 설명: {condition['description']}")
    print(f"{'='*80}")
    
    try:
        # 모델 훈련
        trained_model, engine = train_draem_sevnet_model_multi_domain(
            datamodule=multi_datamodule,
            experiment_name=experiment_name,
            results_base_dir=results_base_dir,
            max_epochs=max_epochs,
            severity_head_mode=condition["severity_head_mode"],
            score_combination=condition["score_combination"],
            severity_loss_type=condition["severity_loss_type"],
            severity_weight=condition["severity_weight"],
            patch_width_range=condition["patch_width_range"],
            patch_ratio_range=condition["patch_ratio_range"],
            patch_count=condition.get("patch_count", 1),
            optimizer_name=condition["optimizer"],
            learning_rate=condition["learning_rate"],
            early_stopping=condition.get("early_stopping", True),
            patience=condition.get("patience", 3),
            min_delta=condition.get("min_delta", 0.005)
        )
        
        best_checkpoint = engine.trainer.checkpoint_callback.best_model_path
        
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
            
            # DraemSevNet/MultiDomainHDMAPDataModule 패턴 검색
            for pattern in ["**/DraemSevNet/MultiDomainHDMAPDataModule/**/images", "**/images"]:
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
        
        # 실험 결과 정리
        experiment_result = {
            "condition": condition,
            "experiment_name": experiment_name,
            "source_results": source_results,
            "target_results": target_results,
            "best_checkpoint": best_checkpoint,
            "training_info": training_info,
            "status": "success",
            "experiment_path": str(latest_version_path) if latest_version_path else None  # 실험 결과 경로 추가
        }
        
        print(f"✅ 실험 완료 - {condition['name']}")
        print(f"   Source Domain AUROC: {source_results.get('image_AUROC', 'N/A'):.4f}")
        
        # Target Domain 평균 성능 계산
        if target_results:
            target_aurocs = [results.get('image_AUROC', 0) for results in target_results.values()]
            avg_target_auroc = sum(target_aurocs) / len(target_aurocs) if target_aurocs else 0
            print(f"   Target Domains Avg AUROC: {avg_target_auroc:.4f}")
            experiment_result["avg_target_auroc"] = avg_target_auroc
        
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
    """멀티 도메인 DRAEM-SevNet 실험 메인 함수."""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="DRAEM-SevNet 실험")
    parser.add_argument("--gpu-id", type=str, required=True, help="사용할 GPU ID")
    parser.add_argument("--experiment-id", type=int, required=True, help="실험 조건 ID (0부터 시작)")
    parser.add_argument("--log-dir", type=str, required=True, help="로그 저장 디렉토리")
    
    args = parser.parse_args()
    
    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    print("="*80)
    print(f"🚀 DRAEM-SevNet 개별 실험 (GPU {args.gpu_id})")
    print(f"실험 조건 ID: {args.experiment_id}")
    print("="*80)
    
    # 실험 설정
    SOURCE_DOMAIN = "domain_A"
    TARGET_DOMAINS = "auto"
    BATCH_SIZE = 16
    MAX_EPOCHS = 30  # Early stopping으로 효율적 학습
    
    # 📐 패치 형태 중심 Ablation Study (12개 조건)
    # 핵심 가설: HDMAP 구조의 선형적 특성상 가로형 패치가 이상감지에 최적일 것
    EXPERIMENT_CONDITIONS = [
        # === Group A: 극단적 Landscape 패치 (4개) ===
        # patch_ratio_range < 1.0 (landscape = 가로형)
        {
            "name": "ultra_landscape_tiny",
            "severity_head_mode": "single_scale",
            "score_combination": "simple_average",
            "severity_loss_type": "smooth_l1",
            "severity_weight": 1.0,
            "patch_width_range": (8, 16),      # 매우 작은 크기
            "patch_ratio_range": (0.25, 0.33), # 1:3~1:4 비율 (극단적 가로)
            "patch_count": 1,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "early_stopping": True,
            "patience": 3,
            "min_delta": 0.005,
            "description": "극도로 가늘고 긴 가로 패치 (Ultra landscape + tiny)"
        },
        {
            "name": "ultra_landscape_small",
            "severity_head_mode": "single_scale",
            "score_combination": "simple_average",
            "severity_loss_type": "smooth_l1",
            "severity_weight": 1.0,
            "patch_width_range": (16, 32),     # 작은 크기
            "patch_ratio_range": (0.25, 0.33), # 1:3~1:4 비율
            "patch_count": 1,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "early_stopping": True,
            "patience": 3,
            "min_delta": 0.005,
            "description": "가늘고 긴 가로 패치 (Ultra landscape + small)"
        },
        {
            "name": "super_landscape",
            "severity_head_mode": "single_scale",
            "score_combination": "simple_average",
            "severity_loss_type": "smooth_l1",
            "severity_weight": 1.0,
            "patch_width_range": (32, 64),     # 중간 크기
            "patch_ratio_range": (0.3, 0.4),   # 1:2.5~1:3.3 비율
            "patch_count": 1,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "early_stopping": True,
            "patience": 3,
            "min_delta": 0.005,
            "description": "매우 가로형 패치 (Super landscape)"
        },
        {
            "name": "landscape_optimal",
            "severity_head_mode": "single_scale",
            "score_combination": "simple_average",
            "severity_loss_type": "smooth_l1",
            "severity_weight": 1.0,
            "patch_width_range": (16, 32),     # 기존 최고 성능 크기
            "patch_ratio_range": (0.4, 0.67),  # 1:1.5~1:2.5 비율
            "patch_count": 1,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "early_stopping": True,
            "patience": 3,
            "min_delta": 0.005,
            "description": "검증된 최적 landscape 패치 (기존 최고 성능 재현)"
        },
        
        # === Group B: 극단적 Portrait 패치 (4개) ===
        # patch_ratio_range > 1.0 (portrait = 세로형)
        {
            "name": "ultra_portrait_tiny",
            "severity_head_mode": "single_scale",
            "score_combination": "simple_average",
            "severity_loss_type": "smooth_l1",
            "severity_weight": 1.0,
            "patch_width_range": (8, 16),      # 매우 작은 크기
            "patch_ratio_range": (3.0, 4.0),   # 3:1~4:1 비율 (극단적 세로)
            "patch_count": 1,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "early_stopping": True,
            "patience": 3,
            "min_delta": 0.005,
            "description": "극도로 가늘고 긴 세로 패치 (Ultra portrait + tiny)"
        },
        {
            "name": "ultra_portrait_small",
            "severity_head_mode": "single_scale",
            "score_combination": "simple_average",
            "severity_loss_type": "smooth_l1",
            "severity_weight": 1.0,
            "patch_width_range": (16, 32),     # 작은 크기
            "patch_ratio_range": (3.0, 4.0),   # 3:1~4:1 비율
            "patch_count": 1,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "early_stopping": True,
            "patience": 3,
            "min_delta": 0.005,
            "description": "가늘고 긴 세로 패치 (Ultra portrait + small)"
        },
        {
            "name": "super_portrait",
            "severity_head_mode": "single_scale",
            "score_combination": "simple_average",
            "severity_loss_type": "smooth_l1",
            "severity_weight": 1.0,
            "patch_width_range": (32, 64),     # 중간 크기
            "patch_ratio_range": (2.5, 3.5),   # 2.5:1~3.5:1 비율
            "patch_count": 1,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "early_stopping": True,
            "patience": 3,
            "min_delta": 0.005,
            "description": "매우 세로형 패치 (Super portrait)"
        },
        {
            "name": "portrait_moderate",
            "severity_head_mode": "single_scale",
            "score_combination": "simple_average",
            "severity_loss_type": "smooth_l1",
            "severity_weight": 1.0,
            "patch_width_range": (16, 32),     # 중간 크기
            "patch_ratio_range": (1.5, 2.5),   # 1.5:1~2.5:1 비율
            "patch_count": 1,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "early_stopping": True,
            "patience": 3,
            "min_delta": 0.005,
            "description": "중간 정도 세로형 패치 (Portrait moderate)"
        },
        
        # === Group C: 정사각형 & 크기 변화 (4개) ===
        {
            "name": "perfect_square_tiny",
            "severity_head_mode": "single_scale",
            "score_combination": "simple_average",
            "severity_loss_type": "smooth_l1",
            "severity_weight": 1.0,
            "patch_width_range": (8, 16),      # 작은 크기
            "patch_ratio_range": (0.95, 1.05), # 거의 정사각형
            "patch_count": 1,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "early_stopping": True,
            "patience": 3,
            "min_delta": 0.005,
            "description": "작은 정사각형 패치 (Perfect square tiny)"
        },
        {
            "name": "perfect_square_medium",
            "severity_head_mode": "single_scale",
            "score_combination": "simple_average",
            "severity_loss_type": "smooth_l1",
            "severity_weight": 1.0,
            "patch_width_range": (32, 48),     # 중간 크기
            "patch_ratio_range": (0.95, 1.05), # 거의 정사각형
            "patch_count": 1,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "early_stopping": True,
            "patience": 3,
            "min_delta": 0.005,
            "description": "중간 크기 정사각형 패치 (Perfect square medium)"
        },
        {
            "name": "perfect_square_large",
            "severity_head_mode": "single_scale",
            "score_combination": "simple_average",
            "severity_loss_type": "smooth_l1",
            "severity_weight": 1.0,
            "patch_width_range": (64, 96),     # 큰 크기
            "patch_ratio_range": (0.95, 1.05), # 거의 정사각형
            "patch_count": 1,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "early_stopping": True,
            "patience": 3,
            "min_delta": 0.005,
            "description": "큰 정사각형 패치 (Perfect square large)"
        },
        {
            "name": "giant_landscape",
            "severity_head_mode": "single_scale",
            "score_combination": "simple_average",
            "severity_loss_type": "smooth_l1",
            "severity_weight": 1.0,
            "patch_width_range": (64, 128),    # 매우 큰 크기
            "patch_ratio_range": (0.5, 0.75),  # 큰 가로형
            "patch_count": 1,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "early_stopping": True,
            "patience": 3,
            "min_delta": 0.005,
            "description": "거대한 landscape 패치 (Giant landscape)"
        }
    ]
    
    # 실험 조건 검증
    if args.experiment_id >= len(EXPERIMENT_CONDITIONS):
        print(f"❌ 잘못된 실험 ID: {args.experiment_id} (최대: {len(EXPERIMENT_CONDITIONS)-1})")
        return
    
    condition = EXPERIMENT_CONDITIONS[args.experiment_id]
    
    # 로그 디렉토리 생성
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 로그 파일명 생성 (DRAEM 스타일로 단순화)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"draem_sevnet_experiment_{timestamp}.log"
    log_path = log_dir / log_filename
    
    # 로깅 설정 (공통 함수 사용)
    logger = setup_experiment_logging(str(log_path), f"draem_sevnet_{condition['name']}")
    
    # GPU 메모리 정리
    cleanup_gpu_memory()
    
    try:
        # ========================================================================================
        # 실험 정보 로깅
        # ========================================================================================
        logger.info("="*80)
        logger.info(f"🚀 DRAEM-SevNet 개별 실험 시작")
        logger.info(f"GPU ID: {args.gpu_id}")
        logger.info(f"실험 조건 ID: {args.experiment_id}")
        logger.info(f"실험 이름: {condition['name']}")
        logger.info(f"설명: {condition['description']}")
        logger.info(f"로그 파일: {log_path}")
        logger.info("="*80)
        
        # ========================================================================================
        # 1단계: MultiDomainHDMAPDataModule 설정
        # ========================================================================================
        logger.info("📦 MultiDomainHDMAPDataModule 설정")
        
        multi_datamodule = create_multi_domain_datamodule(
            datamodule_class=MultiDomainHDMAPDataModule,
            source_domain=SOURCE_DOMAIN,
            target_domains=TARGET_DOMAINS,
            batch_size=BATCH_SIZE,
            image_size="224x224"
        )
        
        logger.info("📊 DRAEM-SevNet 구성 요약:")
        logger.info(f"   🔧 DRAEM Backbone: 97.5M 파라미터")
        logger.info(f"   🎯 SeverityHead: Discriminative encoder features 직접 활용")
        logger.info(f"   🔗 Score Combination: (mask_score + severity_score) / 2")
        logger.info(f"   ⏱️ Early Stopping: val_image_AUROC 기반")
        logger.info(f"   📐 이미지 크기: 224x224")
        logger.info(f"   🔥 배치 크기: {BATCH_SIZE}")
        
        # ======================================================================================== 
        # 2단계: 개별 실험 수행
        # ========================================================================================
        logger.info("🔬 개별 실험 수행 시작")
        
        result = run_single_draem_sevnet_experiment(
            multi_datamodule=multi_datamodule,
            condition=condition,
            source_domain=SOURCE_DOMAIN,
            max_epochs=MAX_EPOCHS,
            log_dir=args.log_dir,  # run 스크립트에서 전달받은 timestamp 폴더 사용
            gpu_id=args.gpu_id,
            experiment_id=args.experiment_id
        )
        
        # ========================================================================================
        # 3단계: 결과 저장 및 로깅 (공통 함수 사용)
        # ========================================================================================
        logger.info("📝 실험 결과 저장")
        
        result_filename = f"result_exp_{args.experiment_id:02d}_{condition['name']}_gpu{args.gpu_id}_{timestamp}.json"
        result_path = save_experiment_results(result, result_filename, log_dir, logger)

        
        # 메모리 정리
        cleanup_gpu_memory()
        logger.info("🧹 GPU 메모리 정리 완료")
        
        logger.info("="*80)
        logger.info("✅ 실험 완료!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ 실험 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 오류 발생 시에도 메모리 정리
        cleanup_gpu_memory()
        logger.error("🧹 오류 후 GPU 메모리 정리 완료")


if __name__ == "__main__":
    main()
