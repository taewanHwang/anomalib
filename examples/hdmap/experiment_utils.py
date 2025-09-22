#!/usr/bin/env python3
"""Anomaly Detection 실험을 위한 공통 유틸리티 함수들.

이 모듈은 DRAEM, PaDiM 등 다양한 Anomaly Detection 모델 실험에서 
재사용할 수 있는 공통 함수들을 제공합니다.

주요 기능:
- GPU 메모리 관리
- 실험 로깅 설정
- 결과 이미지 정리 및 시각화
- Target domain 평가
- 실험 결과 분석 및 저장
"""

import os
import gc
import json
import shutil
import warnings
import logging
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from lightning.pytorch.callbacks import EarlyStopping

# Anomalib imports
from anomalib.engine import Engine


def load_experiment_conditions(json_filename: str) -> List[Dict[str, Any]]:
    """
    JSON 파일에서 실험 조건을 로드합니다.
    
    Args:
        json_filename: 로드할 JSON 파일명 (확장자 포함)
        
    Returns:
        실험 조건 리스트
        
    Raises:
        FileNotFoundError: JSON 파일을 찾을 수 없는 경우
        json.JSONDecodeError: JSON 파싱 오류가 발생한 경우
    """
    # caller 스크립트가 있는 디렉토리 기준으로 JSON 파일 경로 생성
    import inspect
    caller_frame = inspect.stack()[1]
    caller_dir = os.path.dirname(os.path.abspath(caller_frame.filename))
    json_path = os.path.join(caller_dir, json_filename)
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"실험 조건 파일을 찾을 수 없습니다: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"JSON 파싱 오류: {e}")
    
    # JSON에서 로드한 데이터의 유효성 검사
    if 'experiment_conditions' not in data:
        raise ValueError("JSON 파일에 'experiment_conditions' 키가 없습니다.")
    
    experiment_conditions = data['experiment_conditions']
    
    # JSON에서는 tuple이 list로 저장되므로, 필요한 필드들을 다시 tuple로 변환
    for condition in experiment_conditions:
        if 'config' not in condition:
            continue
            
        config = condition['config']
        
        # range 타입의 필드들을 tuple로 변환
        range_fields = ['patch_width_range', 'patch_ratio_range']
        for field in range_fields:
            if field in config and isinstance(config[field], list):
                config[field] = tuple(config[field])
    
    return experiment_conditions

def create_experiment_visualization(
    experiment_name: str,
    model_type: str,
    results_base_dir: str,
    source_domain: str = None,
    target_domains: list = None,
    source_results: Dict[str, Any] = None,
    target_results: Dict[str, Dict[str, Any]] = None,
    single_domain: bool = False
) -> str:
    """실험 결과 시각화 폴더 구조 생성 및 실험 정보 저장.
    
    Args:
        experiment_name: 실험 이름
        model_type: 모델 타입 (예: "DRAEM-SevNet", "PaDiM")
        results_base_dir: 기본 결과 디렉토리 경로
        source_domain: 소스 도메인 이름 (single_domain=True일 때는 None 가능)
        target_domains: 타겟 도메인 리스트
        source_results: 소스 도메인 평가 결과
        target_results: 타겟 도메인들 평가 결과
        single_domain: 단일 도메인 실험 여부
        
    Returns:
        str: 생성된 visualize 디렉토리 경로
    """
    print(f"\n🎨 {model_type} Visualization 생성")
    
    # 기본 경로를 사용 (이미 실험별 고유 경로임)
    base_path = Path(results_base_dir)
    
    # 먼저 v* 패턴의 버전 폴더가 있는지 확인
    version_dirs = [d for d in base_path.glob("v*") if d.is_dir() and d.name.startswith('v') and d.name[1:].isdigit()]
    if version_dirs:
        # v0, v1 등의 패턴이 있으면 최신 버전 사용
        latest_version_path = max(version_dirs, key=lambda x: int(x.name[1:]))
    else:
        # 버전 폴더가 없으면 기본 경로 사용
        latest_version_path = base_path
    
    # visualize 폴더 생성
    viz_path = latest_version_path / "visualize"
    viz_path.mkdir(exist_ok=True)
    
    # 폴더 구조 생성 (single_domain 실험에서는 target_domains 폴더 생성하지 않음)
    if single_domain:
        folders_to_create = ["results"]
    else:
        folders_to_create = [
            "source_domain",
            "target_domains"
        ]
    
    for folder in folders_to_create:
        (viz_path / folder).mkdir(exist_ok=True)
    
    # 타겟 도메인별 하위 폴더 생성 (multi-domain 실험에만 적용)
    if not single_domain and target_domains:
        for domain in target_domains:
            (viz_path / "target_domains" / domain).mkdir(exist_ok=True)
    
    # 실험 정보를 JSON으로 저장
    experiment_info = {
        "experiment_name": experiment_name,
        "model_type": model_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results_path": str(latest_version_path),
        "source_domain": source_domain,
        "target_domains": target_domains or [],
        "results_summary": {
            "source_results": source_results or {},
            "target_results": target_results or {}
        }
    }
    
    # JSON 파일로 저장
    info_file = viz_path / "experiment_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_info, f, indent=2, ensure_ascii=False)
    
    # 실제 생성된 이미지 파일들을 visualize/results로 복사 (single domain의 경우)
    if single_domain:
        try:
            # anomalib 모델 결과 이미지가 저장되는 패턴 탐색
            image_patterns = [
                latest_version_path / "*" / "*" / source_domain / "latest" / "images",  # 일반적인 패턴
                latest_version_path / "*" / source_domain / "latest" / "images",        # 축약된 패턴
                latest_version_path / "images",                                          # 직접 images 폴더
            ]
            
            images_found = False
            for pattern_path in image_patterns:
                # glob 패턴으로 이미지 디렉토리 찾기
                for images_dir in Path(str(pattern_path).replace('*', '')).parent.glob('**/images'):
                    if images_dir.exists() and any(images_dir.iterdir()):
                        print(f"📁 이미지 발견: {images_dir}")
                        
                        # 이미지 파일들을 visualize/results로 복사
                        results_dir = viz_path / "results"
                        
                        # 서브 디렉토리별로 복사 (good, fault 등)
                        for subdir in images_dir.iterdir():
                            if subdir.is_dir():
                                target_subdir = results_dir / subdir.name
                                target_subdir.mkdir(exist_ok=True)
                                
                                # 모든 이미지 파일 복사
                                image_files = list(subdir.glob('*.png'))
                                for img_file in image_files:
                                    shutil.copy2(img_file, target_subdir / img_file.name)
                                
                                print(f"   📸 {len(image_files)}개 이미지를 {target_subdir}에 복사")
                        
                        images_found = True
                        break
                
                if images_found:
                    break
            
            if not images_found:
                print(f"⚠️ 이미지 파일을 찾을 수 없습니다: {latest_version_path}")
        
        except Exception as copy_error:
            print(f"⚠️ 이미지 복사 중 오류: {copy_error}")
    
    print(f"✅ {model_type} 폴더 구조 생성 완료: {viz_path}")
    
    return str(viz_path)


def cleanup_gpu_memory():
    """GPU 메모리 정리 (모든 모델에서 공통 사용 가능)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def setup_warnings_filter():
    """실험 시 불필요한 경고 메시지 필터링 (모든 모델에서 공통 사용 가능)."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # 시각화 관련 특정 경고 필터링 (더 포괄적)
    warnings.filterwarnings("ignore", message=".*Field.*gt_mask.*is None.*")
    warnings.filterwarnings("ignore", message=".*Skipping visualization.*")
    warnings.filterwarnings("ignore", message=".*gt_mask.*None.*")


def setup_experiment_logging(log_file_path: str, experiment_name: str) -> logging.Logger:
    """실험 로깅 설정 (모든 모델에서 공통 사용 가능).
    
    Args:
        log_file_path: 로그 파일 경로
        experiment_name: 실험 이름
        
    Returns:
        logging.Logger: 설정된 로거
    """
    # 로거 설정
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 파일 핸들러 추가
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 포맷터 설정
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger


def extract_training_info(engine: Engine) -> Dict[str, Any]:
    """PyTorch Lightning Engine에서 학습 정보 추출 (모든 모델에서 공통 사용 가능).
    
    Args:
        engine: Anomalib Engine 객체
        
    Returns:
        Dict[str, Any]: 학습 정보 딕셔너리
    """
    import torch
    
    trainer = engine.trainer
    
    # Early Stopping 콜백 찾기
    early_stopping_callback = None
    for callback in trainer.callbacks:
        if isinstance(callback, EarlyStopping):
            early_stopping_callback = callback
            break
    
    # 실제 학습 완료 에폭 계산
    if early_stopping_callback and hasattr(early_stopping_callback, 'stopped_epoch') and early_stopping_callback.stopped_epoch > 0:
        # Early stopping이 발생한 경우: stopped_epoch이 마지막 학습 에폭
        last_trained_epoch = early_stopping_callback.stopped_epoch + 1  # 0-based → 1-based
        early_stopped = True
    else:
        # 정상 완료 또는 early stopping 없음
        last_trained_epoch = trainer.current_epoch + 1  # 0-based → 1-based
        early_stopped = False
    
    # 기본 정보
    training_info = {
        "max_epochs_configured": trainer.max_epochs,
        "last_trained_epoch": last_trained_epoch,  # 실제 마지막으로 학습한 에폭 (1-based)
        "total_steps": trainer.global_step,
        "early_stopped": early_stopped,
        "early_stop_reason": None,
        "best_val_auroc": None,  # 최고 validation AUROC
        "f1_threshold_issue": "F1Score는 기본 threshold로 계산됨. AUROC 대비 낮을 수 있음."
    }
    
    # Early stopping 세부 정보 추가
    if early_stopping_callback:
        if training_info["early_stopped"]:
            training_info["early_stop_reason"] = f"No improvement for {early_stopping_callback.patience} epochs"
        
        # 최고 성능 기록
        if hasattr(early_stopping_callback, 'best_score') and early_stopping_callback.best_score is not None:
            best_score = early_stopping_callback.best_score
            if hasattr(best_score, 'cpu'):
                training_info["best_val_auroc"] = float(best_score.cpu())
            else:
                training_info["best_val_auroc"] = float(best_score)
    
    # 모델 체크포인트 정보에서도 최고 성능 추출
    checkpoint_callback = None
    for callback in trainer.callbacks:
        if hasattr(callback, 'best_model_score'):
            checkpoint_callback = callback
            break
    
    if checkpoint_callback and hasattr(checkpoint_callback, 'best_model_score') and checkpoint_callback.best_model_score is not None:
        best_score = checkpoint_callback.best_model_score
        if hasattr(best_score, 'cpu'):
            training_info["best_val_auroc"] = float(best_score.cpu())
        else:
            training_info["best_val_auroc"] = float(best_score)
    
    # 학습 완료 방식 결정
    if training_info["early_stopped"]:
        completion_type = "early_stopping"
        completion_description = f"에폭 {training_info['last_trained_epoch']}에서 early stopping으로 중단"
    elif training_info["last_trained_epoch"] >= training_info["max_epochs_configured"]:
        completion_type = "max_epochs_reached"
        completion_description = f"최대 에폭 {training_info['max_epochs_configured']} 완료"
    else:
        completion_type = "interrupted"
        completion_description = f"에폭 {training_info['last_trained_epoch']}에서 중단됨"
    
    training_info["completion_type"] = completion_type
    training_info["completion_description"] = completion_description
    
    return training_info

def save_experiment_results(
    result: Dict[str, Any], 
    result_filename: str, 
    log_dir: Path, 
    logger: logging.Logger,
    model_type: str = "Model"
) -> Path:
    """실험 결과를 JSON 파일로 저장 (모든 모델에서 공통 사용 가능).
    
    Args:
        result: 실험 결과 딕셔너리
        result_filename: 저장할 파일명
        log_dir: 로그 디렉토리 (실패 시 사용)
        logger: 로거 객체
        model_type: 모델 타입 (로깅용)
        
    Returns:
        Path: 실제 저장된 파일 경로
    """
    # 실험별 경로에 저장하거나, 실패 시 log_dir에 저장
    if result.get("experiment_path") and Path(result["experiment_path"]).exists():
        result_path = Path(result["experiment_path"]) / result_filename
        print(f"   📁 결과 파일을 실험 폴더에 저장: {result_path}")
    else:
        result_path = log_dir / result_filename
        print(f"   📁 결과 파일을 로그 폴더에 저장: {result_path}")
    
    # 결과 JSON 직렬화를 위한 처리
    serializable_result = json.loads(json.dumps(result, default=str))
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False)
    
    # 결과 요약 로깅
    if result["status"] == "success":
        logger.info("✅ 실험 성공!")
        
        # AUROC 정보 로깅 (multi-domain 또는 all-domains에 따라 다르게 처리)
        if 'source_results' in result:
            # Multi-domain 실험의 경우
            source_auroc = result['source_results'].get('image_AUROC', None)
            if isinstance(source_auroc, (int, float)):
                logger.info(f"   Source Domain AUROC: {source_auroc:.4f}")
            else:
                logger.info(f"   Source Domain AUROC: {source_auroc or 'N/A'}")
            
            # Target Domains Avg AUROC 안전한 포맷팅
            avg_target_auroc = result.get('avg_target_auroc', None)
            if isinstance(avg_target_auroc, (int, float)):
                logger.info(f"   Target Domains Avg AUROC: {avg_target_auroc:.4f}")
            else:
                logger.info(f"   Target Domains Avg AUROC: {avg_target_auroc or 'N/A'}")
                
        elif 'all_domains_results' in result:
            # All-domains 실험의 경우
            all_domains_auroc = result['all_domains_results'].get('test_image_AUROC', None)
            if isinstance(all_domains_auroc, (int, float)):
                logger.info(f"   All Domains AUROC: {all_domains_auroc:.4f}")
            else:
                logger.info(f"   All Domains AUROC: {all_domains_auroc or 'N/A'}")
        else:
            logger.info("   AUROC 정보 없음")
        
        logger.info(f"   체크포인트: {result.get('best_checkpoint', 'N/A')}")
        
        # 학습 과정 정보 로깅
        training_info = result.get('training_info', {})
        if training_info:
            logger.info("📊 학습 과정 정보:")
            logger.info(f"   설정된 최대 에포크: {training_info.get('max_epochs_configured', 'N/A')}")
            logger.info(f"   실제 학습 에포크: {training_info.get('last_trained_epoch', 'N/A')}")
            logger.info(f"   총 학습 스텝: {training_info.get('total_steps', 'N/A')}")
            logger.info(f"   Early Stopping 적용: {training_info.get('early_stopped', 'N/A')}")
            if training_info.get('early_stopped'):
                logger.info(f"   Early Stopping 사유: {training_info.get('early_stop_reason', 'N/A')}")
            # 최고 Validation AUROC 안전한 포맷팅
            best_val_auroc = training_info.get('best_val_auroc', None)
            if isinstance(best_val_auroc, (int, float)):
                logger.info(f"   최고 Validation AUROC: {best_val_auroc:.4f}")
            else:
                logger.info(f"   최고 Validation AUROC: {best_val_auroc or 'N/A'}")
            logger.info(f"   학습 완료 방식: {training_info.get('completion_description', 'N/A')}")
        
        # Target Domain별 상세 성능 로깅
        target_results = result.get('target_results', {})
        if target_results:
            logger.info("🎯 Target Domain별 성능:")
            for domain, domain_result in target_results.items():
                domain_auroc = domain_result.get('image_AUROC', 'N/A')
                if isinstance(domain_auroc, (int, float)):
                    logger.info(f"   {domain}: {domain_auroc:.4f}")
                else:
                    logger.info(f"   {domain}: {domain_auroc}")
    else:
        logger.info("❌ 실험 실패!")
        logger.info(f"   오류: {result.get('error', '알 수 없는 오류')}")
    
    logger.info(f"📁 결과 파일: {result_path}")
    
    # 실험별 경로에 저장된 경우 추가 정보 로깅
    if result.get("experiment_path"):
        logger.info(f"📂 실험 폴더: {result['experiment_path']}")
    
    return result_path


def create_single_domain_datamodule(
    domain: str,
    dataset_root: str,
    batch_size: int = 16,
    target_size: tuple[int, int] | None = None,
    resize_method: str = "resize",
    val_split_ratio: float = 0.2,
    val_split_mode: str = "FROM_TEST",
    num_workers: int = 4,
    seed: int = 42,
    verbose: bool = True,
):
    """Single Domain용 HDMAPDataModule 생성 및 설정.
    
    Args:
        domain: 단일 도메인 이름 (예: "domain_A")
        dataset_root: 데이터셋 루트 경로 (필수)
        batch_size: 배치 크기
        target_size: 타겟 이미지 크기 (height, width). None이면 리사이즈 안 함
        resize_method: 리사이즈 방법 ("resize", "black_padding", "noise_padding")
        val_split_ratio: validation 분할 비율
        val_split_mode: validation 분할 모드 ("FROM_TEST", "NONE", "FROM_TRAIN")
        num_workers: 워커 수
        seed: 랜덤 시드
        verbose: 상세 로그 출력 여부
        
    Returns:
        설정된 HDMAPDataModule
        
    Examples:
        # 256x256 리사이즈
        datamodule = create_single_domain_datamodule(
            domain="domain_A",
            dataset_root="/path/to/dataset",
            target_size=(256, 256),
            batch_size=8
        )
        
        # 224x224 블랙 패딩
        datamodule = create_single_domain_datamodule(
            domain="domain_B",
            dataset_root="/path/to/dataset",
            target_size=(224, 224),
            resize_method="black_padding"
        )
        
        # 원본 크기 유지
        datamodule = create_single_domain_datamodule(
            domain="domain_C",
            dataset_root="/path/to/dataset",
            target_size=None
        )
    """
    from anomalib.data.datamodules.image.hdmap import HDMAPDataModule
    from anomalib.data.utils import ValSplitMode
    from pathlib import Path
    
    # ValSplitMode 문자열을 enum으로 변환
    split_mode_map = {
        "FROM_TEST": ValSplitMode.FROM_TEST,
        "NONE": ValSplitMode.NONE,
        "FROM_TRAIN": ValSplitMode.FROM_TRAIN
    }
    val_split_mode_enum = split_mode_map.get(val_split_mode, ValSplitMode.FROM_TEST)
    
    if verbose:
        print(f"\n📦 HDMAPDataModule 생성 중...")
        print(f"   🎯 도메인: {domain}")
        if target_size:
            print(f"   📏 타겟 크기: {target_size[0]}x{target_size[1]}")
            print(f"   🔧 리사이즈 방법: {resize_method}")
        else:
            print(f"   📏 타겟 크기: 원본 크기 유지")
        print(f"   📊 배치 크기: {batch_size}")
        print(f"   🔄 Val 분할: {val_split_mode} (비율: {val_split_ratio})")
    
    # Path 객체로 변환 및 검증
    dataset_root = Path(dataset_root).resolve()
    
    if not dataset_root.exists():
        raise FileNotFoundError(f"데이터셋 경로가 존재하지 않습니다: {dataset_root}")
    
    if verbose:
        print(f"   📁 데이터셋 경로: {dataset_root}")
        print(f"   📁 도메인 경로: {dataset_root / domain}")
    
    # HDMAPDataModule 생성
    datamodule = HDMAPDataModule(
        root=str(dataset_root),
        domain=domain,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        val_split_mode=val_split_mode_enum,
        val_split_ratio=val_split_ratio,
        seed=seed,
        target_size=target_size,
        resize_method=resize_method
    )
    
    # 데이터 준비 및 설정
    if verbose:
        print(f"   ⚙️  DataModule 설정 중...")
    
    try:
        datamodule.prepare_data()
        datamodule.setup()
    except Exception as e:
        print(f"❌ DataModule 설정 실패: {e}")
        raise
    
    # 데이터 통계 출력
    if verbose:
        print(f"✅ {domain} 데이터 로드 완료")
        print(f"   훈련 샘플: {len(datamodule.train_data):,}개")
        
        val_count = len(datamodule.val_data) if hasattr(datamodule, 'val_data') and datamodule.val_data else 0
        print(f"   검증 샘플: {val_count:,}개")
        print(f"   테스트 샘플: {len(datamodule.test_data):,}개")
        
        # 첫 번째 배치로 데이터 형태 확인
        try:
            train_loader = datamodule.train_dataloader()
            sample_batch = next(iter(train_loader))
            print(f"   📊 이미지 형태: {sample_batch.image.shape}")
            print(f"   📊 레이블 형태: {sample_batch.gt_label.shape}")
            print(f"   📊 데이터 범위: [{sample_batch.image.min():.3f}, {sample_batch.image.max():.3f}]")
        except Exception as e:
            print(f"   ⚠️  배치 정보 확인 실패: {e}")
    
    return datamodule



# =============================================================================
# 상세 분석 함수들
# =============================================================================

def save_detailed_test_results(
    predictions: Dict[str, Any],
    ground_truth: Dict[str, Any], 
    image_paths: List[str],
    result_dir: Path,
    model_type: str = "unknown"
) -> None:
    """
    테스트 결과를 이미지별로 상세하게 CSV 파일로 저장합니다.
    
    Args:
        predictions: 모델 예측 결과 딕셔너리
        ground_truth: 실제 정답 딕셔너리
        image_paths: 이미지 경로 리스트
        result_dir: 결과 저장 디렉토리
        model_type: 모델 타입 (draem_sevnet, patchcore 등)
    """
    import pandas as pd
    
    # result_dir을 analysis_dir로 직접 사용 (중복 폴더 생성 방지)
    analysis_dir = Path(result_dir)
    
    # 결과 데이터 수집
    results_data = []
    
    for i, img_path in enumerate(image_paths):
        row = {
            "image_path": img_path,
            "ground_truth": ground_truth.get("labels", [0] * len(image_paths))[i] if isinstance(ground_truth.get("labels"), list) else ground_truth.get("label", [0])[i],
            "anomaly_score": predictions.get("pred_scores", [0] * len(image_paths))[i] if isinstance(predictions.get("pred_scores"), list) else 0,
        }
        
        
        # 예측 레이블 계산 (기본 threshold 0.5 사용)
        row["predicted_label"] = 1 if row["anomaly_score"] > 0.5 else 0
        
        results_data.append(row)
    
    # DataFrame 생성 및 저장
    df = pd.DataFrame(results_data)
    csv_path = analysis_dir / "test_results.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"📊 테스트 결과 CSV 저장: {csv_path}")


def plot_roc_curve(
    ground_truth: List[int],
    scores: List[float], 
    result_dir: Path,
    experiment_name: str = "Experiment"
) -> float:
    """
    ROC curve를 그리고 AUROC 값을 반환합니다.
    
    Args:
        ground_truth: 실제 정답 리스트 (0 또는 1)
        scores: 예측 점수 리스트
        result_dir: 결과 저장 디렉토리
        experiment_name: 실험 이름
        
    Returns:
        AUROC 값
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    import numpy as np
    
    # result_dir을 analysis_dir로 직접 사용 (중복 폴더 생성 방지)
    analysis_dir = Path(result_dir)
    
    # ROC curve 계산
    fpr, tpr, thresholds = roc_curve(ground_truth, scores)
    auroc = auc(fpr, tpr)
    
    # 최적 threshold 계산 (Youden's index)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # 플롯 생성
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.6)
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, 
                label=f'Optimal threshold = {optimal_threshold:.3f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {experiment_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # 저장
    roc_path = analysis_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📈 ROC Curve 저장: {roc_path}")
    return auroc


def save_metrics_report(
    ground_truth: List[int],
    predictions: List[int],
    scores: List[float],
    result_dir: Path,
    auroc: float,
    optimal_threshold: float = 0.5
) -> None:
    """
    성능 메트릭을 JSON 파일로 저장합니다.
    
    Args:
        ground_truth: 실제 정답 리스트
        predictions: 예측 결과 리스트  
        scores: 예측 점수 리스트
        result_dir: 결과 저장 디렉토리
        auroc: AUROC 값
        optimal_threshold: 최적 threshold
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    import json
    
    # result_dir을 analysis_dir로 직접 사용 (중복 폴더 생성 방지)
    analysis_dir = Path(result_dir)
    
    # 메트릭 계산
    precision = precision_score(ground_truth, predictions, zero_division=0)
    recall = recall_score(ground_truth, predictions, zero_division=0)
    f1 = f1_score(ground_truth, predictions, zero_division=0)
    cm = confusion_matrix(ground_truth, predictions).tolist()
    
    # 메트릭 보고서 생성
    metrics_report = {
        "auroc": float(auroc),
        "optimal_threshold": float(optimal_threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm,
        "total_samples": len(ground_truth),
        "positive_samples": sum(ground_truth),
        "negative_samples": len(ground_truth) - sum(ground_truth)
    }
    
    # JSON 파일로 저장
    metrics_path = analysis_dir / "metrics_report.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_report, f, indent=2, ensure_ascii=False)
    
    print(f"📋 메트릭 보고서 저장: {metrics_path}")


def plot_score_distributions(
    normal_scores: List[float],
    anomaly_scores: List[float], 
    result_dir: Path,
    experiment_name: str = "Experiment"
) -> None:
    """
    정상/이상 샘플의 점수 분포를 히스토그램으로 시각화합니다.
    
    Args:
        normal_scores: 정상 샘플 점수 리스트
        anomaly_scores: 이상 샘플 점수 리스트
        result_dir: 결과 저장 디렉토리
        experiment_name: 실험 이름
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # result_dir을 analysis_dir로 직접 사용 (중복 폴더 생성 방지)
    analysis_dir = Path(result_dir)
    
    # 히스토그램 생성
    plt.figure(figsize=(10, 6))
    
    # 정상 샘플 분포
    plt.hist(normal_scores, bins=50, alpha=0.6, label=f'Normal (n={len(normal_scores)})', 
             color='blue', density=True)
    
    # 이상 샘플 분포  
    plt.hist(anomaly_scores, bins=50, alpha=0.6, label=f'Anomaly (n={len(anomaly_scores)})', 
             color='red', density=True)
    
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title(f'Score Distributions - {experiment_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 통계 정보 텍스트 추가
    normal_mean, normal_std = np.mean(normal_scores), np.std(normal_scores)
    anomaly_mean, anomaly_std = np.mean(anomaly_scores), np.std(anomaly_scores)
    
    stats_text = f'Normal: μ={normal_mean:.3f}, σ={normal_std:.3f}\\nAnomaly: μ={anomaly_mean:.3f}, σ={anomaly_std:.3f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 저장
    dist_path = analysis_dir / "score_distributions.png"
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 점수 분포 저장: {dist_path}")

def save_experiment_summary(
    experiment_config: Dict[str, Any],
    results: Dict[str, float],
    result_dir: Path,
    training_time: Optional[str] = None
) -> None:
    """
    실험 설정과 결과를 YAML 파일로 요약 저장합니다.
    
    Args:
        experiment_config: 실험 설정 딕셔너리
        results: 실험 결과 딕셔너리
        result_dir: 결과 저장 디렉토리
        training_time: 학습 시간 (선택적)
    """
    import yaml
    from datetime import datetime
    
    # result_dir을 analysis_dir로 직접 사용 (중복 폴더 생성 방지)
    analysis_dir = Path(result_dir)
    
    # 요약 정보 생성
    summary = {
        'experiment_info': {
            'name': experiment_config.get('name', 'unknown'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_time': training_time or 'N/A'
        },
        'hyperparameters': experiment_config,
        'results': results,
        'analysis_files': [
            'test_results.csv',
            'roc_curve.png', 
            'metrics_report.json',
            'score_distributions.png',
            'extreme_samples/'
        ]
    }
    
    # YAML 파일로 저장
    summary_path = analysis_dir / "experiment_summary.yaml"
    with open(summary_path, 'w', encoding='utf-8') as f:
        yaml.dump(summary, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print(f"📄 실험 요약 저장: {summary_path}")


def analyze_test_data_distribution(datamodule, test_size: int) -> Tuple[int, int]:
    """테스트 데이터의 라벨 분포를 분석합니다.
    
    Args:
        datamodule: 테스트 데이터를 제공하는 데이터모듈
        test_size: 전체 테스트 데이터 크기
        
    Returns:
        Tuple[int, int]: (fault_count, good_count) 
    """
    import torch
    import numpy as np
    
    print(f"   🔍 테스트 데이터 라벨 분포 전체 확인 중 (총 {test_size}개)...")
    
    test_loader = datamodule.test_dataloader()
    fault_count = 0
    good_count = 0
    total_processed = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if hasattr(batch, 'gt_label'):
                labels = batch.gt_label.numpy()
                batch_fault_count = (labels == 1).sum()
                batch_good_count = (labels == 0).sum()
                
                fault_count += batch_fault_count
                good_count += batch_good_count
                total_processed += len(labels)
                
                # 진행률 표시 (100 배치마다)
                if batch_idx % 100 == 0 and batch_idx > 0:
                    print(f"      📊 진행률: {batch_idx+1} 배치, {total_processed}개 처리됨")
    
    print(f"   ✅ 테스트 데이터 분포 분석 완료 - 총 {total_processed}개 샘플")
    print(f"      🚨 최종 분포: Fault={fault_count}, Good={good_count}")
    
    # 분포 비율 계산 및 경고
    if total_processed > 0:
        fault_ratio = fault_count / total_processed * 100
        good_ratio = good_count / total_processed * 100
        print(f"      📈 비율: Fault={fault_ratio:.1f}%, Good={good_ratio:.1f}%")
        
        # 불균형 경고
        if fault_count == 0:
            print(f"      ⚠️  경고: Fault 이미지가 없습니다! AUROC 계산에 문제가 있을 수 있습니다.")
        elif good_count == 0:
            print(f"      ⚠️  경고: Good 이미지가 없습니다! AUROC 계산에 문제가 있을 수 있습니다.")
        elif abs(fault_count - good_count) > total_processed * 0.3:
            print(f"      ⚠️  경고: 라벨 분포가 불균형합니다 (30% 이상 차이)")
        else:
            print(f"      ✅ 테스트 데이터 라벨 분포 정상")
    
    return fault_count, good_count

def unified_model_evaluation(model, datamodule, experiment_dir, experiment_name, model_type, logger):
    """통합된 모델 평가 함수
    
    Args:
        model: Lightning 모델
        datamodule: 데이터 모듈
        experiment_dir: 실험 디렉터리 경로
        experiment_name: 실험 이름
        model_type: 모델 타입 (소문자)
        logger: 로거 객체
                
    Returns:
        dict: AUROC, threshold, precision, recall, f1 score, confusion matrix 등 평가 메트릭
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
    from PIL import Image
    from pathlib import Path
    
    print(f"   🚀 통합 모델 평가 시작...")
    
    # 모델을 evaluation 모드로 설정
    model.eval()
    
    # PyTorch 모델에 직접 접근
    torch_model = model.model
    torch_model.eval()
    
    # 모델을 GPU로 이동 (CUDA 사용 가능한 경우)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_model = torch_model.to(device)
    print(f"   🖥️ 모델을 {device}로 이동 완료")
    
    # 시각화 디렉터리 생성
    visualization_dir = Path(experiment_dir) / "visualizations"
    visualization_dir.mkdir(exist_ok=True)
    print(f"   🖼️ 시각화 저장 경로: {visualization_dir}")
    
    # 데이터 수집을 위한 리스트들
    all_image_paths = []
    all_ground_truth = []
    all_scores = []
    
    # 테스트 데이터로더 생성
    test_dataloader = datamodule.test_dataloader()
    print(f"   ✅ 테스트 데이터로더 생성 완료")
    total_batches = len(test_dataloader)
    
    print(f"   🔄 {total_batches}개 배치 처리 시작...")
    
    # 배치별로 예측 수행
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            print(f"   📝 처리 중: {batch_idx+1}/{total_batches} 배치 (진행률: {100*(batch_idx+1)/total_batches:.1f}%)")
            
            # 이미지 경로 추출 (필수)
            image_paths = batch.image_path
            if not isinstance(image_paths, list):
                image_paths = [image_paths]
            
            # 이미지 텐서 추출 및 디바이스 이동을 한 번에 처리
            image_tensor = batch.image.to(device)
            print(f"      🖼️  이미지 텐서 크기: {image_tensor.shape}, 경로 수: {len(image_paths)}, min: {image_tensor.min().item():.4f}, q1: {image_tensor.quantile(0.25).item():.4f}, q2: {image_tensor.quantile(0.5).item():.4f}, q3: {image_tensor.quantile(0.75).item():.4f}, max: {image_tensor.max().item():.4f}")
            
            # 모델로 직접 예측 수행
            model_output = torch_model(image_tensor)
            print(f"      ✅ 모델 출력 완료: {type(model_output)}")
                        
            # 모델별 출력에서 점수들 추출
            final_scores = extract_scores_from_model_output(
                model_output, image_tensor.shape[0], batch_idx, model_type
            )
            
            # 시각화 생성 (전체 배치)
            create_batch_visualizations(image_tensor, model_output, image_paths, visualization_dir, batch_idx)
            
            # Ground truth 추출 (이미지 경로에서)
            gt_labels = []
            for path in image_paths:
                if '/fault/' in path:
                    gt_labels.append(1)  # anomaly
                elif '/good/' in path:
                    gt_labels.append(0)  # normal
                else:
                    gt_labels.append(0)  # 기본값
            
            # 결과 수집
            all_image_paths.extend(image_paths)
            all_ground_truth.extend(gt_labels)
            all_scores.extend(final_scores.flatten() if hasattr(final_scores, 'flatten') else final_scores)
            
            print(f"      ✅ 배치 {batch_idx+1} 완료: {len(gt_labels)}개 샘플 추가")
    
    print(f"   ✅ 총 {len(all_image_paths)}개 샘플 처리 완료")
    
    # 길이 맞추기
    min_len = min(len(all_ground_truth), len(all_scores))
    all_ground_truth = all_ground_truth[:min_len]
    all_scores = all_scores[:min_len]
    
    print(f"   ✅ 통합 평가: {len(all_ground_truth)}개 샘플로 메트릭 계산")
    
    # NaN 값 확인 및 필터링
    import numpy as np
    scores_array = np.array(all_scores)
    gt_array = np.array(all_ground_truth)
    
    nan_mask = np.isnan(scores_array)
    if nan_mask.any():
        nan_count = nan_mask.sum()
        print(f"   ⚠️  NaN 점수 {nan_count}개 발견, 제거 후 계속")
        
        # NaN이 아닌 값들만 필터링
        valid_mask = ~nan_mask
        scores_array = scores_array[valid_mask]
        gt_array = gt_array[valid_mask]
        all_scores = scores_array.tolist()
        all_ground_truth = gt_array.tolist()
        
        print(f"   ✅ 유효한 샘플: {len(all_scores)}개")
    
    # AUROC 계산 및 ROC curve 생성
    try:
        auroc = roc_auc_score(all_ground_truth, all_scores)
    except ValueError as e:
        print(f"   ❌ 평가 실패: {e}")
        return None
    
    # 임계값 계산 (Youden's J statistic)
    fpr, tpr, thresholds = roc_curve(all_ground_truth, all_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"   📈 AUROC: {auroc:.4f}, 최적 임계값: {optimal_threshold:.4f}")
    
    # 예측 라벨 생성
    predictions = (np.array(all_scores) > optimal_threshold).astype(int)
    
    # Confusion Matrix 계산
    cm = confusion_matrix(all_ground_truth, predictions)
    
    # 메트릭 계산
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, len(all_ground_truth), 0)
    
    accuracy = (tp + tn) / len(all_ground_truth) if len(all_ground_truth) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 결과 출력
    print(f"   🧮 통합 Confusion Matrix:")
    print(f"       실제\\예측    Normal  Anomaly")
    print(f"       Normal     {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"       Anomaly    {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    print(f"   📈 통합 메트릭:")
    print(f"      AUROC: {auroc:.4f}")
    print(f"      Accuracy: {accuracy:.4f}")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall: {recall:.4f}")
    print(f"      F1-Score: {f1:.4f}")
    print(f"      Threshold: {optimal_threshold:.4f}")
    
    # 기본 메트릭 딕셔너리
    unified_metrics = {
        "auroc": float(auroc),
        "accuracy": float(accuracy), 
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "optimal_threshold": float(optimal_threshold),
        "total_samples": len(all_ground_truth),
        "positive_samples": int(np.sum(all_ground_truth)),
        "negative_samples": int(len(all_ground_truth) - np.sum(all_ground_truth))
    }
    
    # analysis 폴더 생성 및 결과 저장
    analysis_dir = Path(experiment_dir) / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    print(f"   💾 분석 결과 저장 중: {analysis_dir}")
    
    # 상세 테스트 결과 CSV 저장
    predictions_dict = {
        "pred_scores": all_scores,
    }
    ground_truth_dict = {
        "labels": all_ground_truth
    }
    save_detailed_test_results(
        predictions_dict, ground_truth_dict, all_image_paths, 
        analysis_dir, model_type
    )
    
    # ROC curve 생성
    plot_roc_curve(all_ground_truth, all_scores, analysis_dir, experiment_name)
    
    # 메트릭 보고서 저장
    save_metrics_report(all_ground_truth, predictions, all_scores, analysis_dir, auroc, optimal_threshold)
    
    # 점수 분포 히스토그램 생성
    normal_scores = [score for gt, score in zip(all_ground_truth, all_scores) if gt == 0]
    anomaly_scores = [score for gt, score in zip(all_ground_truth, all_scores) if gt == 1]
    plot_score_distributions(normal_scores, anomaly_scores, analysis_dir, experiment_name)
        
    # 실험 요약 저장
    save_experiment_summary({}, {"auroc": auroc}, analysis_dir)
    
    logger.info(f"통합 평가 완료: AUROC={auroc:.4f}, F1={f1:.4f}, 샘플수={len(all_image_paths)}")
    
    return unified_metrics


def extract_scores_from_model_output(model_output, batch_size, batch_idx, model_type):
    """
    모델별 출력에서 점수들을 추출합니다.
    
    Args:
        model_output: 모델 출력 객체
        batch_size: 배치 크기
        batch_idx: 배치 인덱스
        model_type: 모델 타입 (소문자)
        
    Returns:
        tuple: (anomaly_scores,)
    """
    import numpy as np
    
    model_type = model_type.lower()
    
    if model_type == "draem":
        # DRAEM: pred_score만 있음
        if hasattr(model_output, 'pred_score'):
            final_scores = model_output.pred_score.cpu().numpy()
            
            # NaN 값 확인 및 처리
            if np.isnan(final_scores).any():
                print(f"      ⚠️  DRAEM pred_score에 NaN 발견, 0.0으로 대체")
                final_scores = np.nan_to_num(final_scores, nan=0.0)
            
            # 없는 값은 0 으로 처리 (DRAEM에는 mask_score, severity_score, raw_severity_score, normalized_severity_score 없음)
            print(f"      📊 DRAEM 점수 추출: pred_score={final_scores[0]:.4f}")
        elif hasattr(model_output, 'anomaly_map'):
            # anomaly_map에서 점수 계산
            anomaly_map = model_output.anomaly_map.cpu().numpy()
            final_scores = [float(np.max(am)) if am.size > 0 else 0.0 for am in anomaly_map]
            print(f"      📊 DRAEM 점수 추출 (anomaly_map): max={final_scores[0]:.4f}")
        else:
            raise AttributeError("DRAEM 출력 속성 없음")

    elif model_type == "draem_cutpaste_clf":
        # draem_cutpaste_clf의 경우 pred_score와 anomaly_map이 모두 존재
        try:
            final_scores = model_output.pred_score.cpu().numpy()

            # NaN 값 확인 및 처리
            if np.isnan(final_scores).any():
                raise ValueError(f"      ❌ DRAEM CutPaste Clf pred_score에 NaN이 포함되어 있습니다. (배치 {batch_idx})")

            print(f"      📊 DRAEM CutPaste Clf 점수 추출: first pred_score={final_scores[0]:.4f}")
        except AttributeError:
            raise AttributeError("DRAEM CutPaste Clf 출력 속성 없음")

    elif model_type == "patchcore":
        # PatchCore: pred_score만 있음
        if hasattr(model_output, 'pred_score'):
            final_scores = model_output.pred_score.cpu().numpy()
            print(f"      📊 PatchCore 점수 추출: pred_score={final_scores[0]:.4f}")
        elif hasattr(model_output, 'anomaly_map'):
            # anomaly_map에서 점수 계산
            anomaly_map = model_output.anomaly_map.cpu().numpy()
            final_scores = [float(np.max(am)) if am.size > 0 else 0.0 for am in anomaly_map]
            print(f"      📊 PatchCore 점수 추출 (anomaly_map): max={final_scores[0]:.4f}")
        else:
            raise AttributeError("PatchCore 출력 속성 없음")
            
    elif model_type == "dinomaly":
        # Dinomaly: pred_score 또는 anomaly_map
        if hasattr(model_output, 'pred_score'):
            final_scores = model_output.pred_score.cpu().numpy()
            print(f"      📊 Dinomaly 점수 추출: pred_score={final_scores[0]:.4f}")
        elif hasattr(model_output, 'anomaly_map'):
            # anomaly_map에서 점수 계산
            anomaly_map = model_output.anomaly_map.cpu().numpy()
            final_scores = [float(np.max(am)) if am.size > 0 else 0.0 for am in anomaly_map]
            print(f"      📊 Dinomaly 점수 추출 (anomaly_map): max={final_scores[0]:.4f}")
        else:
            raise AttributeError("Dinomaly 출력 속성 없음")
            
    else:
        # 알 수 없는 모델 타입: 일반적인 속성으로 시도
        print(f"   ⚠️ 알 수 없는 모델 타입: {model_type}, 일반적인 속성으로 시도")
        if hasattr(model_output, 'pred_score'):
            final_scores = model_output.pred_score.cpu().numpy()
        elif hasattr(model_output, 'final_score'):
            final_scores = model_output.final_score.cpu().numpy()
        elif hasattr(model_output, 'anomaly_map'):
            anomaly_map = model_output.anomaly_map.cpu().numpy()
            final_scores = [float(np.max(am)) for am in anomaly_map]
        else:
            raise AttributeError(f"지원되지 않는 모델 출력 형식: {type(model_output)}")
            
        print(f"      📊 일반 모델 점수 추출: anomaly_score={final_scores[0]:.4f}")
        
    return final_scores


def create_anomaly_heatmap_with_colorbar(anomaly_map_array, target_size, cmap='viridis', show_colorbar=False):
    """Anomaly heatmap 생성 (colorbar 옵션)
    
    Args:
        anomaly_map_array: numpy array [H, W]
        target_size: (width, height) 목표 크기
        cmap: matplotlib colormap 이름 (기본값: 'viridis')
              - 'viridis': 파란색->초록색->노란색 (권장)
              - 'jet': 파란색->청록->노랑->빨강 (기존)
              - 'hot': 검정->빨강->노랑->흰색
              - 'plasma': 보라->분홍->노랑
              - 'inferno': 검정->보라->빨강->노랑
              - 'turbo': 파란색->청록->초록->노랑->빨강
              - 'coolwarm': 파란색->흰색->빨강
        show_colorbar: colorbar 표시 여부 (기본값: False)
        
    Returns:
        PIL.Image: heatmap 이미지 (colorbar 포함/제외)
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    import io
    from PIL import Image
    
    # 값 범위 계산
    vmin = anomaly_map_array.min()
    vmax = anomaly_map_array.max()
    
    # Figure 크기 및 레이아웃 설정
    if show_colorbar:
        # Colorbar 포함 레이아웃
        fig_width = 6
        fig_height = 4
        fig, (ax_img, ax_cbar) = plt.subplots(1, 2, figsize=(fig_width, fig_height), 
                                              gridspec_kw={'width_ratios': [4, 0.3]})
    else:
        # Colorbar 없는 레이아웃
        fig_width = 4
        fig_height = 4
        fig, ax_img = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    # Heatmap 생성
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax_img.imshow(anomaly_map_array, cmap=cmap, norm=norm, aspect='auto')
    ax_img.axis('off')
    
    # Colorbar 생성 (옵션에 따라)
    if show_colorbar:
        cbar = plt.colorbar(im, cax=ax_cbar)
        cbar.set_label('Anomaly Score', rotation=270, labelpad=15, fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    
    # PIL 이미지로 변환
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.05)
    buf.seek(0)
    heatmap_pil = Image.open(buf).convert('RGB')
    plt.close()
    
    return heatmap_pil


def create_batch_visualizations(image_tensor, model_output, image_paths, visualization_dir, batch_idx):
    """배치에 대한 시각화 생성 (원본 이미지 + anomaly map)
    
    Args:
        image_tensor: 입력 이미지 텐서 [B, C, H, W]
        model_output: 모델 출력 객체
        image_paths: 이미지 경로 리스트
        visualization_dir: 시각화 저장 디렉터리
        batch_idx: 배치 인덱스
    """
    import numpy as np
    from PIL import Image
    from pathlib import Path
    
    # anomalib 시각화 함수들 import
    from anomalib.visualization.image.functional import (
        overlay_images,
        create_image_grid,
        add_text_to_image
    )
    
    # 배치에서 anomaly map 추출
    anomaly_maps = None
    if hasattr(model_output, 'anomaly_map'):
        anomaly_maps = model_output.anomaly_map
    else:
        return
    
    # 배치 크기
    batch_size = image_tensor.shape[0]
    
    # 각 이미지에 대해 시각화 생성
    for i in range(batch_size):  # 전체 배치 시각화
        try:
            # 원본 이미지 추출 및 변환
            original_img_tensor = image_tensor[i]  # [C, H, W]
            
            # 텐서를 PIL 이미지로 변환 (정규화 해제)
            # 이미지가 [0, 1] 범위라고 가정
            if original_img_tensor.max() <= 1.0:
                original_img_array = (original_img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            else:
                original_img_array = original_img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            
            # 그레이스케일인 경우 RGB로 변환
            if original_img_array.shape[2] == 1:
                original_img_array = np.repeat(original_img_array, 3, axis=2)
            elif original_img_array.shape[2] > 3:
                original_img_array = original_img_array[:, :, :3]
                
            original_img_pil = Image.fromarray(original_img_array, mode='RGB')
            
            # Anomaly map 추출 및 변환
            anomaly_map_tensor = anomaly_maps[i]  # [H, W] 또는 [1, H, W]
            if len(anomaly_map_tensor.shape) == 3:
                anomaly_map_tensor = anomaly_map_tensor.squeeze(0)  # [H, W]
            
            # 원본 이미지와 anomaly map 크기 맞추기
            target_size = original_img_pil.size
            
            # anomaly map을 matplotlib으로 시각화
            # 다른 colormap 옵션: 'jet', 'hot', 'plasma', 'inferno', 'turbo', 'coolwarm'
            anomaly_map_vis = create_anomaly_heatmap_with_colorbar(
                anomaly_map_tensor.cpu().numpy(),
                target_size,
                cmap='hot',        # 원하는 colormap으로 변경 가능
                show_colorbar=False    # colorbar 표시: True/False
            )
            
            # 오버레이 생성 (원본 + anomaly map)
            overlay_img = overlay_images(
                base=original_img_pil,
                overlays=anomaly_map_vis,
                alpha=0.5
            )
            
            # 텍스트 추가
            original_with_text = add_text_to_image(
                original_img_pil.copy(), 
                "Original Image",
                font=None, size=10, color="white", background=(0, 0, 0, 128)
            )
            
            overlay_with_text = add_text_to_image(
                overlay_img.copy(), 
                "Original + Anomaly Map",
                font=None, size=10, color="white", background=(0, 0, 0, 128)
            )
            
            # 2개 이미지를 가로로 배치
            visualization_grid = create_image_grid(
                [original_with_text, overlay_with_text], 
                nrow=2
            )
            
            # 파일명 및 디렉터리 생성 (레이블별로 분류)
            if i < len(image_paths):
                image_path = Path(image_paths[i])
                image_filename = image_path.stem
                
                # 경로에서 레이블 추출 (fault 또는 good)
                label = None
                if '/fault/' in str(image_path):
                    label = 'fault'
                elif '/good/' in str(image_path):
                    label = 'good'
                else:
                    label = 'unknown'
                
                # 레이블별 디렉터리 생성
                label_dir = visualization_dir / label
                label_dir.mkdir(exist_ok=True)
                
                # 파일명은 원본 이미지 이름 사용
                save_filename = f"{image_filename}.png"
                save_path = label_dir / save_filename
            else:
                # 이미지 경로가 없는 경우 기본 형식 사용
                save_filename = f"batch_{batch_idx:03d}_sample_{i:02d}.png"
                save_path = visualization_dir / save_filename
            
            # 이미지 저장
            visualization_grid.save(save_path)
            
        except Exception as e:
            print(f"❌ 샘플 {i} 시각화 실패: {e}")
            import traceback
            traceback.print_exc()
