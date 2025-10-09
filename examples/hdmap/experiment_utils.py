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
        completion_description = f"Early stopped at epoch {training_info['last_trained_epoch']}"
    elif training_info["last_trained_epoch"] >= training_info["max_epochs_configured"]:
        completion_type = "max_epochs_reached"
        completion_description = f"Completed max epochs {training_info['max_epochs_configured']}"
    else:
        completion_type = "interrupted"
        completion_description = f"Interrupted at epoch {training_info['last_trained_epoch']}"
    
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
        logger.info(f"   Error: {result.get('error', 'Unknown error')}")
    
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
    
    # 안전한 데이터 접근을 위한 변수 준비
    labels = ground_truth.get("labels", [0] * len(image_paths))
    pred_scores = predictions.get("pred_scores", [0] * len(image_paths))
    
    # None 체크 및 기본값 설정
    if labels is None:
        labels = [0] * len(image_paths)
    if pred_scores is None:
        pred_scores = [0] * len(image_paths)
    
    # 길이 확인 및 조정
    if len(image_paths) == 0:
        print("⚠️ Warning: No image paths provided")
        return
        
    min_len = min(len(image_paths), len(labels), len(pred_scores))
    
    if min_len != len(image_paths):
        print(f"⚠️ Warning: Length mismatch - paths: {len(image_paths)}, labels: {len(labels)}, scores: {len(pred_scores)}, using min: {min_len}")
    
    for i in range(min_len):
        img_path = image_paths[i]
        row = {
            "image_path": img_path,
            "ground_truth": labels[i] if i < len(labels) else 0,
            "anomaly_score": pred_scores[i] if i < len(pred_scores) else 0,
        }
        
        
        # 예측 레이블 계산 (기본 threshold 0.5 사용)
        row["predicted_label"] = 1 if row["anomaly_score"] > 0.5 else 0
        
        results_data.append(row)
    
    # DataFrame 생성 및 저장
    df = pd.DataFrame(results_data)
    csv_path = analysis_dir / "test_results.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"📊 테스트 결과 CSV 저장: {csv_path}")


def find_optimal_threshold(ground_truth: List[int], scores: List[float]) -> float:
    """Youden's J statistic을 사용하여 optimal threshold 찾기"""
    from sklearn.metrics import roc_curve
    import numpy as np
    
    fpr, tpr, thresholds = roc_curve(ground_truth, scores)
    j_scores = tpr - fpr  # Youden's J statistic = Sensitivity + Specificity - 1
    optimal_idx = np.argmax(j_scores)
    return float(thresholds[optimal_idx])

def evaluate_with_fixed_threshold(
    scores: List[float], 
    labels: List[int], 
    threshold: float
) -> Dict[str, float]:
    """고정된 threshold로 성능 평가"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import numpy as np
    
    predictions = (np.array(scores) > threshold).astype(int)
    
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(labels, predictions)),
        'precision': float(precision_score(labels, predictions, zero_division=0)),
        'recall': float(recall_score(labels, predictions, zero_division=0)),
        'f1_score': float(f1_score(labels, predictions, zero_division=0))
    }

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
        "total_samples": int(len(ground_truth)),
        "positive_samples": int(sum(ground_truth)),
        "negative_samples": int(len(ground_truth) - sum(ground_truth))
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
    
    # 모델의 training flag를 명시적으로 False로 설정 (FastFlow 등에서 중요)
    torch_model.training = False
    
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
            
            # 모델로 직접 예측 수행 (inference mode에서 실행)
            with torch.no_grad():
                # FastFlow 등 모델의 경우 반드시 eval mode에서 호출해야 함
                model_output = torch_model(image_tensor)
            print(f"      ✅ 모델 출력 완료: {type(model_output)}")
            
            # FastFlow 모델의 경우 training mode인지 확인
            if model_type.lower() == "fastflow":
                print(f"      🔍 FastFlow 모델 상태: training={torch_model.training}")
                if hasattr(model_output, 'pred_score'):
                    print(f"      📊 FastFlow pred_score shape: {model_output.pred_score.shape}")
                    print(f"      📊 FastFlow pred_score 값: {model_output.pred_score.cpu().numpy()}")
                if hasattr(model_output, 'anomaly_map'):
                    print(f"      📊 FastFlow anomaly_map shape: {model_output.anomaly_map.shape}")
                    amap_stats = model_output.anomaly_map.cpu().numpy()
                    print(f"      📊 FastFlow anomaly_map 통계: min={amap_stats.min():.4f}, max={amap_stats.max():.4f}, mean={amap_stats.mean():.4f}")
                        
            # 모델별 출력에서 점수들 추출
            final_scores = extract_scores_from_model_output(
                model_output, image_tensor.shape[0], batch_idx, model_type
            )

            # 시각화 생성 (전체 배치)
            create_batch_visualizations(image_tensor, model_output, image_paths, visualization_dir, batch_idx, model=torch_model)

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
            
    elif model_type.lower() == "fastflow":
        # FastFlow 모델 처리
        print(f"      🌊 FastFlow 점수 추출")
        
        # pred_score가 있지만 모두 0인 경우 anomaly_map을 사용
        if hasattr(model_output, 'pred_score'):
            pred_scores = model_output.pred_score.cpu().numpy()
            print(f"      📊 FastFlow pred_score: min={np.min(pred_scores):.4f}, max={np.max(pred_scores):.4f}")
            
            # pred_score가 모두 0이면 anomaly_map으로 대체
            if np.max(pred_scores) == 0.0 and hasattr(model_output, 'anomaly_map'):
                print(f"      ⚠️ pred_score가 모두 0이므로 anomaly_map 사용")
                anomaly_map = model_output.anomaly_map.cpu().numpy()
                final_scores = np.array([float(np.mean(am)) if am.size > 0 else 0.0 for am in anomaly_map])
                print(f"      📊 FastFlow 점수 추출 (anomaly_map mean): min={np.min(final_scores):.4f}, max={np.max(final_scores):.4f}")
            else:
                final_scores = pred_scores
                
        elif hasattr(model_output, 'anomaly_map'):
            # anomaly_map에서 점수 계산 (fallback)
            anomaly_map = model_output.anomaly_map.cpu().numpy()
            final_scores = np.array([float(np.mean(am)) if am.size > 0 else 0.0 for am in anomaly_map])
            print(f"      📊 FastFlow 점수 추출 (anomaly_map only): min={np.min(final_scores):.4f}, max={np.max(final_scores):.4f}")
        else:
            raise AttributeError("FastFlow 출력 속성 없음")
            
    elif model_type == "draem_cutpaste":
        # DRAEM CutPaste: pred_score 사용
        if hasattr(model_output, 'pred_score'):
            final_scores = model_output.pred_score.cpu().numpy()
            
            # NaN 값 확인 및 처리
            if np.isnan(final_scores).any():
                print(f"      ⚠️  DRAEM CutPaste pred_score에 NaN 발견, 0.0으로 대체")
                final_scores = np.nan_to_num(final_scores, nan=0.0)
            
            print(f"      📊 DRAEM CutPaste 점수 추출: pred_score={final_scores[0]:.4f}")
        elif hasattr(model_output, 'anomaly_map'):
            # anomaly_map에서 점수 계산
            anomaly_map = model_output.anomaly_map.cpu().numpy()
            final_scores = [float(np.max(am)) if am.size > 0 else 0.0 for am in anomaly_map]
            print(f"      📊 DRAEM CutPaste 점수 추출 (anomaly_map): max={final_scores[0]:.4f}")
        else:
            raise AttributeError("DRAEM CutPaste 출력 속성 없음")
            
    else:
        # 알 수 없는 모델 타입: 일반적인 속성으로 시도
        print(f"   ⚠️ Unknown model type: {model_type}, trying generic attributes")
        if hasattr(model_output, 'pred_score'):
            final_scores = model_output.pred_score.cpu().numpy()
        elif hasattr(model_output, 'final_score'):
            final_scores = model_output.final_score.cpu().numpy()
        elif hasattr(model_output, 'anomaly_map'):
            anomaly_map = model_output.anomaly_map.cpu().numpy()
            final_scores = np.array([float(np.max(am)) if am.size > 0 else 0.0 for am in anomaly_map])
        else:
            raise AttributeError(f"지원되지 않는 모델 출력 형식: {type(model_output)}")
            
        print(f"      📊 일반 모델 점수 추출: anomaly_score={final_scores[0]:.4f}")
        
    return final_scores


def create_anomaly_heatmap_with_colorbar(
    anomaly_map_array,
    cmap='viridis',
    show_colorbar=False,
    fixed_range=True
):
    """Anomaly heatmap 생성 (colorbar 옵션)

    Args:
        anomaly_map_array: numpy array [H, W]
                          range: [0, 1] if fixed_range=True, 아니면 data-dependent
                          meaning: 픽셀별 이상 확률 (0=정상, 1=이상)
        cmap: matplotlib colormap 이름 (기본값: 'viridis')
              - 'viridis': 파란색->초록색->노란색 (권장)
              - 'jet': 파란색->청록->노랑->빨강 (기존)
              - 'hot': 검정->빨강->노랑->흰색
              - 'plasma': 보라->분홍->노랑
              - 'inferno': 검정->보라->빨강->노랑
              - 'turbo': 파란색->청록->초록->노랑->빨강
              - 'coolwarm': 파란색->흰색->빨강
        show_colorbar: colorbar 표시 여부 (기본값: False)
        fixed_range: colorbar range를 0~1로 고정할지 여부 (기본값: True)

    Returns:
        PIL.Image: heatmap 이미지 (colorbar 포함/제외)
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    import io
    from PIL import Image
    import numpy as np

    # 값 범위 설정
    if fixed_range:
        vmin, vmax = 0.0, 1.0
    else:
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


def create_batch_visualizations(image_tensor, model_output, image_paths, visualization_dir, batch_idx, model=None):
    """배치에 대한 시각화 생성

    - DRAEM 계열 (DRAEM, DRAEM CutPaste Clf): original, recon, disc_anomaly, residual, overlay_auto, overlay_fixed (6개)
    - 기타 모델: original, overlay_auto, overlay_fixed (3개)

    Args:
        image_tensor: 입력 이미지 텐서 [B, C, H, W] - range: [0, 1] normalized
        model_output: 모델 출력 객체 (InferenceBatch containing anomaly_map, pred_score, etc.)
        image_paths: 이미지 경로 리스트
        visualization_dir: 시각화 저장 디렉터리
        batch_idx: 배치 인덱스
        model: 모델 객체 (DRAEM 계열 모델인 경우 reconstruction 계산용)
    """
    import numpy as np
    from PIL import Image
    from pathlib import Path
    import torch

    # anomalib 시각화 함수들 import
    from anomalib.visualization.image.functional import (
        overlay_images,
        create_image_grid,
        add_text_to_image
    )

    # 배치에서 anomaly map 및 mask 추출
    anomaly_maps = None
    masks = None
    if hasattr(model_output, 'anomaly_map'):
        anomaly_maps = model_output.anomaly_map
    else:
        return

    # Mask 추출 (있는 경우) - pred_mask 또는 gt_mask
    if hasattr(model_output, 'pred_mask') and model_output.pred_mask is not None:
        masks = model_output.pred_mask
    elif hasattr(model_output, 'gt_mask') and model_output.gt_mask is not None:
        masks = model_output.gt_mask

    # DRAEM 계열 모델인지 확인 (DRAEM, DRAEM CutPaste Clf)
    is_draem = (model is not None and
                hasattr(model, 'reconstructive_subnetwork') and
                hasattr(model, 'discriminative_subnetwork'))

    # DRAEM 계열 모델인 경우 reconstruction과 discriminative anomaly 계산
    recon_batch = None
    disc_anomaly_batch = None
    if is_draem:
        with torch.no_grad():
            # 모델의 입력 채널 수 확인
            # DRAEM CutPaste Clf: 1채널, 원본 DRAEM: 3채널
            # encoder.block1[0]이 첫 번째 Conv2d 레이어
            model_input_channels = model.reconstructive_subnetwork.encoder.block1[0].in_channels

            # 모델 타입 확인
            model_name = "DRAEM CutPaste Clf" if hasattr(model, 'severity_head') else "DRAEM"
            print(f"   📊 Model Type: {model_name}")
            print(f"   🔧 Model Input Channels: {model_input_channels}")
            print(f"   📷 Image Tensor Shape: {image_tensor.shape}")

            # 모델 입력 채널 수에 맞게 이미지 준비
            if model_input_channels == 1:
                # 1채널 모델: 첫 번째 채널만 사용
                batch_input = image_tensor[:, :1, :, :]
                print(f"   ✂️  Using 1-channel mode: {batch_input.shape}")
            else:
                # 3채널 모델: 전체 사용
                batch_input = image_tensor
                print(f"   🎨 Using 3-channel mode: {batch_input.shape}")

            # Reconstruction 계산 (raw 값 직접 사용, sigmoid 제거)
            recon_batch = model.reconstructive_subnetwork(batch_input)

            # 🔍 DEBUG: reconstruction 값 범위 출력
            print(f"      🔍 Reconstruction stats:")
            print(f"         - min={recon_batch.min():.4f}, max={recon_batch.max():.4f}, mean={recon_batch.mean():.4f}, std={recon_batch.std():.4f}")
            print(f"      🔍 Input stats:")
            print(f"         - min={batch_input.min():.4f}, max={batch_input.max():.4f}, mean={batch_input.mean():.4f}, std={batch_input.std():.4f}")

            # Discriminative network 계산 (anomaly channel 추출)
            # Follow DRAEM convention: [original, reconstruction]
            joined_input = torch.cat([batch_input, recon_batch], dim=1)
            print(f"   🔗 Concat order: [original({batch_input.shape[1]}ch), recon({recon_batch.shape[1]}ch)] -> {joined_input.shape}")
            disc_output = model.discriminative_subnetwork(joined_input)

            # Softmax 적용하여 anomaly channel (channel 1) 추출
            # disc_output shape: [B, 2, H, W] -> [B, 1, H, W] (anomaly channel만)
            disc_anomaly_batch = torch.softmax(disc_output, dim=1)[:, 1:2, :, :]

            # 🔍 DEBUG: discriminative anomaly 값 범위 출력
            print(f"      🔍 Discriminative Anomaly stats:")
            print(f"         - min={disc_anomaly_batch.min():.4f}, max={disc_anomaly_batch.max():.4f}, mean={disc_anomaly_batch.mean():.4f}, std={disc_anomaly_batch.std():.4f}")

    # 배치 크기
    batch_size = image_tensor.shape[0]

    # 각 이미지에 대해 시각화 생성
    for i in range(batch_size):  # 전체 배치 시각화
        try:
            # 원본 이미지 추출 및 변환
            original_img_tensor = image_tensor[i]  # [C, H, W]

            # Min-max normalization으로 [0, 1] 범위로 변환
            original_np = original_img_tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
            original_min = original_np.min()
            original_max = original_np.max()
            if original_max > original_min:
                original_normalized = (original_np - original_min) / (original_max - original_min)
            else:
                original_normalized = np.zeros_like(original_np)
            original_img_array = (original_normalized * 255).astype(np.uint8)

            # print(f"      🔍 Original normalization: [{original_min:.4f}, {original_max:.4f}] -> [0, 1]")

            # 그레이스케일인 경우 RGB로 변환
            if original_img_array.shape[2] == 1:
                original_img_array = np.repeat(original_img_array, 3, axis=2)
            elif original_img_array.shape[2] > 3:
                original_img_array = original_img_array[:, :, :3]

            original_img_pil = Image.fromarray(original_img_array, mode='RGB')

            # Anomaly map 추출 및 변환
            anomaly_map_tensor = anomaly_maps[i]  # [H, W] 또는 [1, H, W]
            # range: [0, 1] (이미 softmax가 적용된 anomaly probability)
            # meaning: 픽셀별 이상 확률 (0=정상, 1=이상)
            if len(anomaly_map_tensor.shape) == 3:
                anomaly_map_tensor = anomaly_map_tensor.squeeze(0)  # [H, W]

            # Mask 추출 (boundary 표시용)
            mask_for_boundary = None
            if masks is not None:
                mask_tensor = masks[i]  # [H, W] 또는 [1, H, W]
                if len(mask_tensor.shape) == 3:
                    mask_tensor = mask_tensor.squeeze(0)  # [H, W]
                mask_for_boundary = mask_tensor.cpu().numpy()

            # Auto-range anomaly map 시각화 (colorbar=False)
            anomaly_map_vis_auto = create_anomaly_heatmap_with_colorbar(
                anomaly_map_tensor.cpu().numpy(),
                cmap='jet',
                show_colorbar=False,
                fixed_range=False  # Auto-range
            )

            # Fixed-range (0~1) anomaly map 시각화 (colorbar=False)
            anomaly_map_vis_fixed = create_anomaly_heatmap_with_colorbar(
                anomaly_map_tensor.cpu().numpy(),
                cmap='jet',
                show_colorbar=False,
                fixed_range=True  # 0~1 고정
            )

            # 오버레이 생성 (원본 + anomaly map, auto-range)
            overlay_img_auto = overlay_images(
                base=original_img_pil,
                overlays=anomaly_map_vis_auto,
                alpha=0.5
            )

            # 오버레이 생성 (원본 + anomaly map, fixed-range)
            overlay_img_fixed = overlay_images(
                base=original_img_pil,
                overlays=anomaly_map_vis_fixed,
                alpha=0.5
            )

            # DRAEM 계열 모델인 경우 5개 이미지 시각화
            if is_draem and recon_batch is not None:
                # Reconstruction 이미지 생성
                recon_tensor = recon_batch[i]  # [C, H, W] - C는 1 또는 3
                # range: unbounded (raw network output)
                # meaning: 재구성된 이미지 (학습 후 ~[0,1]에 수렴)

                # Min-max normalization으로 [0, 1] 범위로 변환
                recon_np = recon_tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
                recon_min = recon_np.min()
                recon_max = recon_np.max()
                if recon_max > recon_min:
                    recon_normalized = (recon_np - recon_min) / (recon_max - recon_min)
                else:
                    recon_normalized = np.zeros_like(recon_np)
                recon_array = (recon_normalized * 255).astype(np.uint8)

                print(f"      🔍 Recon normalization: [{recon_min:.4f}, {recon_max:.4f}] -> [0, 1]")

                # 채널 수에 따라 처리
                if recon_array.shape[2] == 1:
                    recon_array = np.repeat(recon_array, 3, axis=2)  # 1채널 → RGB
                elif recon_array.shape[2] == 3:
                    pass  # 이미 3채널, 그대로 사용
                else:
                    recon_array = recon_array[:, :, :3]  # 3채널 초과시 앞 3개만
                recon_img_pil = Image.fromarray(recon_array, mode='RGB')

                # Discriminative Anomaly 이미지 생성
                disc_anomaly_tensor = disc_anomaly_batch[i]  # [1, H, W] - softmax된 anomaly channel
                # range: [0, 1] (이미 softmax가 적용된 anomaly probability)
                # meaning: discriminative network의 픽셀별 이상 확률

                # Softmax 출력은 [0, 1] 범위
                disc_anomaly_array = (disc_anomaly_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                disc_anomaly_array = np.repeat(disc_anomaly_array, 3, axis=2)  # 1채널 → RGB
                disc_anomaly_img_pil = Image.fromarray(disc_anomaly_array, mode='RGB')

                # Residual 이미지 생성 (원본 - 재구성)
                # 원본과 재구성 이미지를 같은 정규화 범위로 맞춤
                original_for_residual = original_img_tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
                recon_for_residual = recon_tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
                
                # 채널 수 맞춤 (원본이 3채널, 재구성이 1채널인 경우)
                if original_for_residual.shape[2] == 3 and recon_for_residual.shape[2] == 1:
                    # 원본의 첫 번째 채널만 사용
                    original_for_residual = original_for_residual[:, :, :1]
                elif original_for_residual.shape[2] == 1 and recon_for_residual.shape[2] == 3:
                    # 재구성의 첫 번째 채널만 사용
                    recon_for_residual = recon_for_residual[:, :, :1]
                
                # Residual 계산 (차이의 절댓값)
                residual_np = np.abs(original_for_residual - recon_for_residual)
                
                # Min-max normalization
                residual_min = residual_np.min()
                residual_max = residual_np.max()
                if residual_max > residual_min:
                    residual_normalized = (residual_np - residual_min) / (residual_max - residual_min)
                else:
                    residual_normalized = np.zeros_like(residual_np)
                
                residual_array = (residual_normalized * 255).astype(np.uint8)
                
                # 1채널을 RGB로 변환
                if residual_array.shape[2] == 1:
                    residual_array = np.repeat(residual_array, 3, axis=2)
                    
                residual_img_pil = Image.fromarray(residual_array, mode='RGB')
                
                print(f"      🔍 Residual stats: min={residual_min:.4f}, max={residual_max:.4f}")

                # DRAEM용 오버레이 생성
                # 5번째: Auto-range, colorbar 없음
                anomaly_map_vis_draem_auto = create_anomaly_heatmap_with_colorbar(
                    anomaly_map_tensor.cpu().numpy(),
                    cmap='jet',
                    show_colorbar=False,
                    fixed_range=False  # Auto-range
                )
                overlay_img_draem_auto = overlay_images(
                    base=original_img_pil,
                    overlays=anomaly_map_vis_draem_auto,
                    alpha=0.5
                )

                # 6번째: Fixed 0-1, colorbar 있음
                anomaly_map_vis_draem_fixed = create_anomaly_heatmap_with_colorbar(
                    anomaly_map_tensor.cpu().numpy(),
                    cmap='jet',
                    show_colorbar=True,
                    fixed_range=True  # Fixed 0-1
                )
                overlay_img_draem_fixed = overlay_images(
                    base=original_img_pil,
                    overlays=anomaly_map_vis_draem_fixed,
                    alpha=0.5
                )

                # 텍스트 추가
                original_with_text = add_text_to_image(
                    original_img_pil.copy(),
                    "Original",
                    font=None, size=10, color="white", background=(0, 0, 0, 128)
                )

                recon_with_text = add_text_to_image(
                    recon_img_pil.copy(),
                    "Reconstruction",
                    font=None, size=10, color="white", background=(0, 0, 0, 128)
                )

                disc_anomaly_with_text = add_text_to_image(
                    disc_anomaly_img_pil.copy(),
                    "Disc Anomaly",
                    font=None, size=10, color="white", background=(0, 0, 0, 128)
                )

                residual_with_text = add_text_to_image(
                    residual_img_pil.copy(),
                    "Residual (|Orig-Recon|)",
                    font=None, size=10, color="white", background=(0, 0, 0, 128)
                )

                overlay_auto_with_text = add_text_to_image(
                    overlay_img_draem_auto.copy(),
                    "Original + Anomaly (Auto)",
                    font=None, size=10, color="white", background=(0, 0, 0, 128)
                )

                overlay_fixed_with_text = add_text_to_image(
                    overlay_img_draem_fixed.copy(),
                    "Original + Anomaly (Fixed 0-1, colorbar)",
                    font=None, size=10, color="white", background=(0, 0, 0, 128)
                )

                # 6개 이미지를 3열 그리드로 배치 (2행: 3개 + 3개)
                visualization_grid = create_image_grid(
                    [original_with_text, recon_with_text, disc_anomaly_with_text,
                     residual_with_text, overlay_auto_with_text, overlay_fixed_with_text],
                    nrow=3
                )
            else:
                # 기타 모델: 3개 이미지 시각화
                original_with_text = add_text_to_image(
                    original_img_pil.copy(),
                    "Original Image",
                    font=None, size=10, color="white", background=(0, 0, 0, 128)
                )

                overlay_auto_with_text = add_text_to_image(
                    overlay_img_auto.copy(),
                    "Original + Anomaly Map (Auto Range)",
                    font=None, size=10, color="white", background=(0, 0, 0, 128)
                )

                overlay_fixed_with_text = add_text_to_image(
                    overlay_img_fixed.copy(),
                    "Original + Anomaly Map (Fixed 0-1)",
                    font=None, size=10, color="white", background=(0, 0, 0, 128)
                )

                # 3개 이미지를 가로로 배치
                visualization_grid = create_image_grid(
                    [original_with_text, overlay_auto_with_text, overlay_fixed_with_text],
                    nrow=3
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


# ===========================
# Multi-Domain Specific Functions
# ===========================

def create_multi_domain_datamodule(
    source_domain: str,
    target_domains: Union[List[str], str],
    dataset_root: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (256, 256),
    resize_method: str = "resize",
    num_workers: int = 8,
    seed: int = 42,
    verbose: bool = True
):
    """Multi-Domain용 MultiDomainHDMAPDataModule 생성 및 설정.

    Args:
        source_domain: 소스 도메인 이름 (예: "domain_A")
        target_domains: 타겟 도메인 리스트 또는 "auto" (auto면 source 제외 모든 도메인)
        dataset_root: 데이터셋 루트 경로 (필수)
        batch_size: 배치 크기
        image_size: 이미지 크기 (height, width)
        num_workers: 워커 수
        seed: 랜덤 시드
        verbose: 상세 로그 출력 여부

    Returns:
        설정된 MultiDomainHDMAPDataModule

    Examples:
        # 수동으로 타겟 도메인 지정
        datamodule = create_multi_domain_datamodule(
            source_domain="domain_A",
            target_domains=["domain_B", "domain_C"],
            dataset_root="/path/to/dataset",
            batch_size=32
        )

        # 자동으로 타겟 도메인 설정 (source 제외한 모든 도메인)
        datamodule = create_multi_domain_datamodule(
            source_domain="domain_A",
            target_domains="auto",
            dataset_root="/path/to/dataset"
        )
    """
    from anomalib.data.datamodules.image.multi_domain_hdmap import MultiDomainHDMAPDataModule
    from pathlib import Path

    # Path 객체로 변환 및 검증
    dataset_root = Path(dataset_root).resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"데이터셋 경로가 존재하지 않습니다: {dataset_root}")

    # target_domains 처리
    if target_domains == "auto":
        all_domains = ["domain_A", "domain_B", "domain_C", "domain_D"]
        target_domains_list = [d for d in all_domains if d != source_domain]
    else:
        target_domains_list = target_domains

    if verbose:
        print(f"\n📦 MultiDomainHDMAPDataModule 생성 중...")
        print(f"   🌍 소스 도메인: {source_domain}")
        print(f"   🎯 타겟 도메인: {target_domains_list}")
        print(f"   📁 데이터셋 경로: {dataset_root}")
        print(f"   📏 이미지 크기: {image_size[0]}x{image_size[1]}")
        print(f"   📊 배치 크기: {batch_size}")
        print(f"   👷 워커 수: {num_workers}")
        print(f"   🎲 시드: {seed}")

    # MultiDomainHDMAPDataModule 생성
    datamodule = MultiDomainHDMAPDataModule(
        root=str(dataset_root),
        source_domain=source_domain,
        target_domains=target_domains_list,
        validation_strategy="source_test",  # 고정값: source test를 validation으로 사용
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        target_size=image_size,
        resize_method=resize_method,
        seed=seed
    )

    # Setup 호출
    datamodule.setup()

    if verbose:
        # 데이터셋 정보 출력
        print(f"\n✅ MultiDomainHDMAPDataModule 설정 완료:")
        print(f"   - Train 샘플 수: {len(datamodule.train_data)}")
        print(f"   - Validation 샘플 수 (source test): {len(datamodule.val_data)}")
        print(f"   - Test 도메인 수: {len(datamodule.test_data)}")
        for i, target_domain in enumerate(target_domains_list):
            print(f"     • {target_domain}: {len(datamodule.test_data[i])} 샘플")

    return datamodule


def evaluate_source_domain(
    model,
    datamodule,
    visualization_dir: Optional[Path] = None,
    model_type: str = "unknown",
    max_visualization_batches: int = 5,
    verbose: bool = True,
    analysis_dir: Optional[Path] = None
):
    """소스 도메인에서 모델 평가 (validation 역할).

    Source domain의 test 데이터를 사용하여 validation 수행.
    선택적으로 시각화도 생성.

    Args:
        model: 학습된 모델
        datamodule: MultiDomainHDMAPDataModule
        visualization_dir: 시각화 저장 디렉터리 (None이면 시각화 안함)
        model_type: 모델 타입 (점수 추출에 사용)
        max_visualization_batches: 시각화할 최대 배치 수 (-1이면 전체 시각화)
        verbose: 상세 출력 여부
        analysis_dir: 분석 결과 저장 디렉터리 (None이면 분석 안함)

    Returns:
        dict: 평가 결과
            - domain: 소스 도메인 이름
            - auroc: AUROC 점수
            - metrics: 기타 메트릭 (accuracy, precision, recall, f1)
            - num_samples: 평가 샘플 수
            - visualization_dir: 시각화 디렉터리 경로

    Example:
        >>> source_results = evaluate_source_domain(
        ...     model=trained_model,
        ...     datamodule=datamodule,
        ...     visualization_dir=Path("./results/viz/source"),
        ...     model_type="draem"
        ... )
    """
    import torch
    import numpy as np
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    from pathlib import Path

    if verbose:
        print(f"\n📊 소스 도메인 ({datamodule.source_domain}) 평가 시작...")

    # 시각화 디렉터리 생성
    if visualization_dir:
        visualization_dir = Path(visualization_dir)
        visualization_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"   📁 시각화 저장 경로: {visualization_dir}")

    # Validation dataloader (source domain test)
    val_loader = datamodule.val_dataloader()

    # 평가 모드로 전환하고 GPU로 이동
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    all_scores = []
    all_labels = []
    all_image_paths = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # GPU로 이동 (가능한 경우)
            if torch.cuda.is_available():
                image_tensor = batch.image.cuda()
                if hasattr(batch, 'mask'):
                    mask_tensor = batch.mask.cuda()
            else:
                image_tensor = batch.image

            # 모델 예측
            outputs = model(image_tensor)

            # 점수 추출 (모델별로 다름)
            scores = extract_scores_from_model_output(
                outputs,
                len(image_tensor),
                batch_idx,
                model_type
            )
            all_scores.extend(scores)

            # 이미지 경로 추출
            if hasattr(batch, 'image_path'):
                image_paths = batch.image_path
                if not isinstance(image_paths, list):
                    image_paths = [image_paths]
                all_image_paths.extend(image_paths)

                # Ground truth 추출 (이미지 경로에서)
                gt_labels = []
                for path in image_paths:
                    if '/fault/' in path:
                        gt_labels.append(1)  # anomaly
                    elif '/good/' in path:
                        gt_labels.append(0)  # normal
                    else:
                        gt_labels.append(0)  # 기본값
                all_labels.extend(gt_labels)

            # 시각화 생성 (max_visualization_batches=-1이면 전체, 아니면 지정된 배치 수만)
            should_visualize = (max_visualization_batches == -1 or batch_idx < max_visualization_batches)
            if visualization_dir and should_visualize:
                # create_batch_visualizations가 내부적으로 fault/good 폴더를 생성함
                create_batch_visualizations(
                    image_tensor,
                    outputs,
                    image_paths,
                    visualization_dir,  # batch 폴더 없이 직접 전달
                    batch_idx,
                    model=model.model if hasattr(model, 'model') else None
                )

            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"   처리 중: {batch_idx + 1}/{len(val_loader)} 배치")

    # 메트릭 계산
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # NaN 처리
    nan_mask = np.isnan(all_scores)
    if nan_mask.any():
        if verbose:
            print(f"   ⚠️ NaN 점수 {nan_mask.sum()}개 발견, 제거 후 계속")
        valid_mask = ~nan_mask
        all_scores = all_scores[valid_mask]
        all_labels = all_labels[valid_mask]

    # AUROC 계산
    try:
        auroc = roc_auc_score(all_labels, all_scores)
    except Exception as e:
        if verbose:
            print(f"   ⚠️ AUROC 계산 실패: {e}")
        auroc = 0.0

    # 이진 분류 메트릭 계산 (threshold = 0.5)
    predictions = (all_scores > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, predictions)
    precision = precision_score(all_labels, predictions, zero_division=0)
    recall = recall_score(all_labels, predictions, zero_division=0)
    f1 = f1_score(all_labels, predictions, zero_division=0)

    if verbose:
        print(f"\n✅ 소스 도메인 평가 완료:")
        print(f"   - Domain: {datamodule.source_domain}")
        print(f"   - AUROC: {auroc:.4f}")
        print(f"   - Accuracy: {accuracy:.4f}")
        print(f"   - Precision: {precision:.4f}")
        print(f"   - Recall: {recall:.4f}")
        print(f"   - F1-Score: {f1:.4f}")
        print(f"   - 샘플 수: {len(all_scores)}")

    # Analysis 기능 추가
    if analysis_dir:
        analysis_dir = Path(analysis_dir)
        source_analysis_dir = analysis_dir / f"source_{datamodule.source_domain}"
        source_analysis_dir.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f"   💾 소스 도메인 분석 결과 저장 중: {source_analysis_dir}")
        
        # 상세 테스트 결과 CSV 저장
        predictions_dict = {"pred_scores": all_scores}
        ground_truth_dict = {"labels": all_labels}
        
        # Debug: 길이 확인
        if verbose:
            print(f"   🔍 Debug - scores: {len(all_scores)}, labels: {len(all_labels)}, paths: {len(all_image_paths)}")
        
        # 길이가 맞지 않는 경우 처리
        min_length = min(len(all_scores), len(all_labels), len(all_image_paths))
        if len(all_scores) != len(all_labels) or len(all_scores) != len(all_image_paths):
            if verbose:
                print(f"   ⚠️ 길이 불일치 발견, 최소 길이 {min_length}로 조정")
            all_scores = all_scores[:min_length]
            all_labels = all_labels[:min_length]
            all_image_paths = all_image_paths[:min_length]
            predictions_dict = {"pred_scores": all_scores}
            ground_truth_dict = {"labels": all_labels}
        
        save_detailed_test_results(
            predictions_dict, ground_truth_dict, all_image_paths, 
            source_analysis_dir, f"{model_type}_source_{datamodule.source_domain}"
        )
        
        # ROC curve 생성
        plot_roc_curve(all_labels, all_scores, source_analysis_dir, 
                      f"{model_type.upper()} Source {datamodule.source_domain}")
        
        # Optimal threshold 계산 및 저장
        optimal_threshold = find_optimal_threshold(all_labels, all_scores)
        optimal_threshold_data = {
            "optimal_threshold": optimal_threshold,
            "method": "youden_j_statistic",
            "source_domain": datamodule.source_domain,
            "model_type": model_type
        }
        
        # Source threshold로 source domain 성능 평가 (참고용)
        source_metrics_with_optimal = evaluate_with_fixed_threshold(all_scores, all_labels, optimal_threshold)
        optimal_threshold_data["source_performance_with_optimal_threshold"] = source_metrics_with_optimal
        
        # Optimal threshold 저장
        threshold_path = source_analysis_dir / "optimal_threshold.json"
        with open(threshold_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(optimal_threshold_data, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print(f"   🎯 Optimal threshold: {optimal_threshold:.4f}")
            print(f"   💾 Optimal threshold 저장: {threshold_path}")
        
        # 메트릭 보고서 저장 (기존 0.5 threshold 기준)
        save_metrics_report(all_labels, predictions, all_scores, source_analysis_dir, auroc, 0.5)
        
        # 점수 분포 히스토그램 생성
        normal_scores = [score for gt, score in zip(all_labels, all_scores) if gt == 0]
        anomaly_scores = [score for gt, score in zip(all_labels, all_scores) if gt == 1]
        plot_score_distributions(normal_scores, anomaly_scores, source_analysis_dir, 
                               f"{model_type.upper()} Source {datamodule.source_domain}")
        
        # 실험 요약 저장
        source_config = {"domain": datamodule.source_domain, "model_type": model_type}
        source_results = {"auroc": auroc, "accuracy": accuracy, "precision": precision, 
                         "recall": recall, "f1_score": f1}
        save_experiment_summary(source_config, source_results, source_analysis_dir)

    result = {
        'domain': datamodule.source_domain,
        'auroc': float(auroc),
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        },
        'num_samples': len(all_scores),
        'visualization_dir': str(visualization_dir) if visualization_dir else None,
        'analysis_dir': str(source_analysis_dir) if analysis_dir else None
    }
    
    # Analysis가 활성화된 경우 optimal threshold 추가
    if analysis_dir:
        result['optimal_threshold'] = float(optimal_threshold)
        result['optimal_threshold_metrics'] = source_metrics_with_optimal
    
    return result


def evaluate_target_domains(
    model,
    datamodule,
    visualization_base_dir: Optional[Path] = None,
    model_type: str = "unknown",
    max_visualization_batches: int = 5,
    verbose: bool = True,
    analysis_base_dir: Optional[Path] = None,
    source_optimal_threshold: Optional[float] = None
):
    """타겟 도메인들에서 모델 평가 및 시각화.

    각 타겟 도메인에 대해 개별적으로 평가하고 결과를 수집.
    선택적으로 각 도메인별 시각화 생성.

    Args:
        model: 학습된 모델
        datamodule: MultiDomainHDMAPDataModule
        visualization_base_dir: 시각화 저장 기본 디렉터리 (None이면 시각화 안함)
        model_type: 모델 타입 (점수 추출에 사용)
        max_visualization_batches: 도메인별 시각화할 최대 배치 수 (-1이면 전체 시각화)
        verbose: 상세 출력 여부
        analysis_base_dir: 분석 결과 저장 기본 디렉터리 (None이면 분석 안함)

    Returns:
        dict: 타겟 도메인별 평가 결과
            각 도메인 키에 대해:
            - domain: 도메인 이름
            - auroc: AUROC 점수
            - metrics: 기타 메트릭 (accuracy, precision, recall, f1)
            - num_samples: 평가 샘플 수
            - visualization_dir: 시각화 디렉터리 경로

    Example:
        >>> target_results = evaluate_target_domains(
        ...     model=trained_model,
        ...     datamodule=datamodule,
        ...     visualization_base_dir=Path("./results/viz/targets"),
        ...     model_type="draem"
        ... )
        >>> for domain, result in target_results.items():
        ...     print(f"{domain}: AUROC={result['auroc']:.4f}")
    """
    import torch
    import numpy as np
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    from pathlib import Path

    if verbose:
        print(f"\n🎯 타겟 도메인 평가 시작...")
        print(f"   타겟 도메인 목록: {datamodule.target_domains}")

    # 시각화 기본 디렉터리 생성
    if visualization_base_dir:
        visualization_base_dir = Path(visualization_base_dir)
        visualization_base_dir.mkdir(parents=True, exist_ok=True)

    # Test dataloaders 가져오기
    test_dataloaders = datamodule.test_dataloader()
    if not isinstance(test_dataloaders, list):
        test_dataloaders = [test_dataloaders]

    # 평가 모드로 전환하고 GPU로 이동
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    target_results = {}

    # 각 타겟 도메인별로 평가
    for domain_idx, target_domain in enumerate(datamodule.target_domains):
        if verbose:
            print(f"\n   📊 {target_domain} 평가 중...")

        # 해당 도메인의 dataloader
        test_loader = test_dataloaders[domain_idx]

        # 도메인별 시각화 디렉터리
        if visualization_base_dir:
            domain_viz_dir = visualization_base_dir / target_domain
            domain_viz_dir.mkdir(exist_ok=True)
            if verbose:
                print(f"      📁 시각화 저장: {domain_viz_dir}")
        else:
            domain_viz_dir = None

        # 평가 데이터 수집
        all_scores = []
        all_labels = []
        all_image_paths = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # GPU로 이동 (가능한 경우)
                if torch.cuda.is_available():
                    image_tensor = batch.image.cuda()
                    if hasattr(batch, 'mask'):
                        mask_tensor = batch.mask.cuda()
                else:
                    image_tensor = batch.image

                # 모델 예측
                outputs = model(image_tensor)

                # 점수 추출
                scores = extract_scores_from_model_output(
                    outputs,
                    len(image_tensor),
                    batch_idx,
                    model_type
                )
                all_scores.extend(scores)

                # 이미지 경로 추출
                if hasattr(batch, 'image_path'):
                    image_paths = batch.image_path
                    if not isinstance(image_paths, list):
                        image_paths = [image_paths]
                    all_image_paths.extend(image_paths)

                    # Ground truth 추출 (이미지 경로에서)
                    gt_labels = []
                    for path in image_paths:
                        if '/fault/' in path:
                            gt_labels.append(1)  # anomaly
                        elif '/good/' in path:
                            gt_labels.append(0)  # normal
                        else:
                            gt_labels.append(0)  # 기본값
                    all_labels.extend(gt_labels)

                # 시각화 생성 (max_visualization_batches=-1이면 전체, 아니면 지정된 배치 수만)
                should_visualize = (max_visualization_batches == -1 or batch_idx < max_visualization_batches)
                if domain_viz_dir and should_visualize:
                    # create_batch_visualizations가 내부적으로 fault/good 폴더를 생성함
                    create_batch_visualizations(
                        image_tensor,
                        outputs,
                        image_paths,
                        domain_viz_dir,  # batch 폴더 없이 직접 전달
                        batch_idx,
                        model=model.model if hasattr(model, 'model') else None
                    )

                if verbose and (batch_idx + 1) % 10 == 0:
                    print(f"      처리 중: {batch_idx + 1}/{len(test_loader)} 배치")

        # 메트릭 계산
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        # NaN 처리
        nan_mask = np.isnan(all_scores)
        if nan_mask.any():
            if verbose:
                print(f"      ⚠️ NaN 점수 {nan_mask.sum()}개 발견, 제거 후 계속")
            valid_mask = ~nan_mask
            all_scores = all_scores[valid_mask]
            all_labels = all_labels[valid_mask]

        # AUROC 계산
        try:
            auroc = roc_auc_score(all_labels, all_scores)
        except Exception as e:
            if verbose:
                print(f"      ⚠️ AUROC 계산 실패: {e}")
            auroc = 0.0

        # 이진 분류 메트릭 계산 (threshold = 0.5)
        predictions = (all_scores > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, predictions)
        precision = precision_score(all_labels, predictions, zero_division=0)
        recall = recall_score(all_labels, predictions, zero_division=0)
        f1 = f1_score(all_labels, predictions, zero_division=0)

        if verbose:
            print(f"      ✅ {target_domain} 평가 완료:")
            print(f"         - AUROC: {auroc:.4f}")
            print(f"         - Accuracy: {accuracy:.4f}")
            print(f"         - F1-Score: {f1:.4f}")
            print(f"         - 샘플 수: {len(all_scores)}")

        # 결과 저장
        # Analysis 기능 추가 (도메인별)
        domain_analysis_dir = None
        if analysis_base_dir:
            analysis_base_dir = Path(analysis_base_dir)
            domain_analysis_dir = analysis_base_dir / f"target_{target_domain}"
            domain_analysis_dir.mkdir(parents=True, exist_ok=True)
            
            if verbose:
                print(f"   💾 타겟 도메인 분석 결과 저장 중: {domain_analysis_dir}")
            
            # 상세 테스트 결과 CSV 저장
            predictions_dict = {"pred_scores": all_scores}
            ground_truth_dict = {"labels": all_labels}
            
            # Debug: 길이 확인
            if verbose:
                print(f"   🔍 Debug [{target_domain}] - scores: {len(all_scores)}, labels: {len(all_labels)}, paths: {len(all_image_paths)}")
            
            # 길이가 맞지 않는 경우 처리
            min_length = min(len(all_scores), len(all_labels), len(all_image_paths))
            if len(all_scores) != len(all_labels) or len(all_scores) != len(all_image_paths):
                if verbose:
                    print(f"   ⚠️ [{target_domain}] 길이 불일치 발견, 최소 길이 {min_length}로 조정")
                all_scores = all_scores[:min_length]
                all_labels = all_labels[:min_length]
                all_image_paths = all_image_paths[:min_length]
                predictions_dict = {"pred_scores": all_scores}
                ground_truth_dict = {"labels": all_labels}
            
            save_detailed_test_results(
                predictions_dict, ground_truth_dict, all_image_paths, 
                domain_analysis_dir, f"{model_type}_target_{target_domain}"
            )
            
            # ROC curve 생성
            plot_roc_curve(all_labels, all_scores, domain_analysis_dir, 
                          f"{model_type.upper()} Target {target_domain}")
            
            # 메트릭 보고서 저장 (기존 0.5 threshold 기준)
            save_metrics_report(all_labels, predictions, all_scores, domain_analysis_dir, auroc, 0.5)
            
            # Source optimal threshold 기반 평가 (있는 경우)
            if source_optimal_threshold is not None:
                target_metrics_with_source_threshold = evaluate_with_fixed_threshold(
                    all_scores, all_labels, source_optimal_threshold
                )
                
                # Source threshold 기반 메트릭 저장
                source_threshold_data = {
                    "source_optimal_threshold": source_optimal_threshold,
                    "target_domain": target_domain,
                    "metrics_with_source_threshold": target_metrics_with_source_threshold,
                    "comparison": {
                        "default_threshold_0.5": {
                            "accuracy": float(accuracy),
                            "precision": float(precision),
                            "recall": float(recall),
                            "f1_score": float(f1)
                        },
                        "source_optimal_threshold": target_metrics_with_source_threshold
                    }
                }
                
                source_threshold_path = domain_analysis_dir / "metrics_with_source_threshold.json"
                with open(source_threshold_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(source_threshold_data, f, indent=2, ensure_ascii=False)
                
                if verbose:
                    print(f"   🎯 Source threshold ({source_optimal_threshold:.4f}) 적용 결과:")
                    print(f"      - Accuracy: {target_metrics_with_source_threshold['accuracy']:.4f}")
                    print(f"      - F1-Score: {target_metrics_with_source_threshold['f1_score']:.4f}")
                    print(f"   💾 Source threshold 분석 저장: {source_threshold_path}")
            
            # 점수 분포 히스토그램 생성
            normal_scores = [score for gt, score in zip(all_labels, all_scores) if gt == 0]
            anomaly_scores = [score for gt, score in zip(all_labels, all_scores) if gt == 1]
            plot_score_distributions(normal_scores, anomaly_scores, domain_analysis_dir, 
                                   f"{model_type.upper()} Target {target_domain}")
            
            # 실험 요약 저장
            target_config = {"domain": target_domain, "model_type": model_type}
            target_results_dict = {"auroc": auroc, "accuracy": accuracy, "precision": precision, 
                                 "recall": recall, "f1_score": f1}
            save_experiment_summary(target_config, target_results_dict, domain_analysis_dir)

        target_results[target_domain] = {
            'domain': target_domain,
            'auroc': float(auroc),
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            },
            'num_samples': len(all_scores),
            'visualization_dir': str(domain_viz_dir) if domain_viz_dir else None,
            'analysis_dir': str(domain_analysis_dir) if domain_analysis_dir else None
        }

    # 평균 성능 계산
    if verbose:
        auroc_values = [r['auroc'] for r in target_results.values()]
        avg_auroc = np.mean(auroc_values) if auroc_values else 0.0

        print(f"\n📈 타겟 도메인 평균 성능:")
        print(f"   - 평균 AUROC: {avg_auroc:.4f}")
        print(f"   - 도메인별 AUROC:")
        for domain, result in target_results.items():
            print(f"     • {domain}: {result['auroc']:.4f}")

    return target_results


def analyze_experiment_results(
    source_results: Dict,
    target_results: Dict,
    training_info: Optional[Dict] = None,
    experiment_config: Optional[Dict] = None,
    save_path: Optional[Union[str, Path]] = None,
    verbose: bool = True
) -> Dict:
    """Multi-domain 실험 결과 종합 분석.

    소스 및 타겟 도메인 결과를 종합적으로 분석하고 보고서 생성.

    Args:
        source_results: 소스 도메인 평가 결과
        target_results: 타겟 도메인별 평가 결과 딕셔너리
        training_info: 훈련 정보 (optional)
        experiment_config: 실험 설정 (optional)
        save_path: 분석 결과 저장 경로 (optional)
        verbose: 상세 출력 여부

    Returns:
        dict: 종합 분석 결과
    """
    import numpy as np
    from pathlib import Path
    import json
    from datetime import datetime

    if verbose:
        print("\n" + "="*80)
        print("📊 Multi-Domain 실험 결과 종합 분석")
        print("="*80)

    # 소스 도메인 성능
    source_performance = {
        'domain': source_results['domain'],
        'auroc': source_results['auroc'],
        'metrics': source_results.get('metrics', {})
    }

    # 타겟 도메인별 성능
    target_performance = {}
    for domain, result in target_results.items():
        target_performance[domain] = {
            'auroc': result['auroc'],
            'metrics': result.get('metrics', {}),
            'num_samples': result.get('num_samples', 0)
        }

    # 평균 메트릭 계산
    target_aurocs = [r['auroc'] for r in target_results.values()]
    target_accuracies = [r['metrics']['accuracy'] for r in target_results.values() if 'metrics' in r]
    target_f1_scores = [r['metrics']['f1_score'] for r in target_results.values() if 'metrics' in r]

    average_metrics = {
        'source_auroc': source_performance['auroc'],
        'target_avg_auroc': np.mean(target_aurocs) if target_aurocs else 0.0,
        'target_std_auroc': np.std(target_aurocs) if target_aurocs else 0.0,
        'target_avg_accuracy': np.mean(target_accuracies) if target_accuracies else 0.0,
        'target_avg_f1': np.mean(target_f1_scores) if target_f1_scores else 0.0
    }

    # 도메인 전이 성능 차이 (Source - Target Average)
    domain_transfer_gap = source_performance['auroc'] - average_metrics['target_avg_auroc']

    # 최고/최저 성능 타겟 도메인
    if target_aurocs:
        best_idx = np.argmax(target_aurocs)
        worst_idx = np.argmin(target_aurocs)
        target_domains_list = list(target_results.keys())

        best_target_domain = {
            'domain': target_domains_list[best_idx],
            'auroc': target_aurocs[best_idx]
        }

        worst_target_domain = {
            'domain': target_domains_list[worst_idx],
            'auroc': target_aurocs[worst_idx]
        }
    else:
        best_target_domain = worst_target_domain = None

    # 분석 결과 구성
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'source_performance': source_performance,
        'target_performance': target_performance,
        'average_metrics': average_metrics,
        'domain_transfer_gap': float(domain_transfer_gap),
        'best_target_domain': best_target_domain,
        'worst_target_domain': worst_target_domain,
        'training_info': training_info or {},
        'experiment_config': experiment_config or {}
    }

    # 결과 출력
    if verbose:
        print(f"\n📌 소스 도메인 ({source_performance['domain']}):")
        print(f"   - AUROC: {source_performance['auroc']:.4f}")
        if source_performance['metrics']:
            print(f"   - Accuracy: {source_performance['metrics'].get('accuracy', 0):.4f}")
            print(f"   - F1-Score: {source_performance['metrics'].get('f1_score', 0):.4f}")

        print(f"\n📌 타겟 도메인 평균 성능:")
        print(f"   - 평균 AUROC: {average_metrics['target_avg_auroc']:.4f} (±{average_metrics['target_std_auroc']:.4f})")
        print(f"   - 평균 Accuracy: {average_metrics['target_avg_accuracy']:.4f}")
        print(f"   - 평균 F1-Score: {average_metrics['target_avg_f1']:.4f}")

        print(f"\n📌 도메인별 AUROC:")
        for domain, perf in target_performance.items():
            print(f"   - {domain}: {perf['auroc']:.4f}")

        if best_target_domain:
            print(f"\n📌 최고 성능 타겟: {best_target_domain['domain']} (AUROC: {best_target_domain['auroc']:.4f})")
            print(f"📌 최저 성능 타겟: {worst_target_domain['domain']} (AUROC: {worst_target_domain['auroc']:.4f})")

        print(f"\n📌 도메인 전이 성능 차이: {domain_transfer_gap:+.4f}")
        print(f"   {'✅ 긍정적' if domain_transfer_gap < 0.1 else '⚠️ 주의 필요'}: Source-Target Gap")

        print("="*80)

    # 분석 결과 저장
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(analysis, f, indent=2)

        if verbose:
            print(f"\n💾 분석 결과 저장: {save_path}")

    return analysis
