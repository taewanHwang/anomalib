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
from typing import Dict, Any, List, Optional, Union
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# Anomalib imports
from anomalib.engine import Engine
from anomalib.data.datamodules.image.multi_domain_hdmap import MultiDomainHDMAPDataModule


def create_experiment_visualization(
    experiment_name: str,
    model_type: str,
    results_base_dir: str,
    source_domain: str,
    target_domains: list = None,
    source_results: Dict[str, Any] = None,
    target_results: Dict[str, Dict[str, Any]] = None
) -> str:
    """실험 결과 시각화 폴더 구조 생성 및 실험 정보 저장.
    
    Args:
        experiment_name: 실험 이름
        model_type: 모델 타입 (예: "DRAEM-SevNet", "PaDiM")
        results_base_dir: 기본 결과 디렉토리 경로
        source_domain: 소스 도메인 이름
        target_domains: 타겟 도메인 리스트
        source_results: 소스 도메인 평가 결과
        target_results: 타겟 도메인들 평가 결과
        
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
    
    # 폴더 구조 생성
    folders_to_create = [
        "source_domain",
        "target_domains"
    ]
    
    for folder in folders_to_create:
        (viz_path / folder).mkdir(exist_ok=True)
    
    # 타겟 도메인별 하위 폴더 생성
    if target_domains:
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
    
    print(f"✅ {model_type} 폴더 구조 생성 완료: {viz_path}")
    
    return str(viz_path)


def create_multi_domain_datamodule(
    datamodule_class,
    source_domain: str = "domain_A",
    target_domains: Union[str, List[str]] = "auto",
    batch_size: int = 16,
    image_size: str = "224x224",
    dataset_root: str = None,
    validation_strategy: str = "source_test",
    num_workers: int = 16
):
    """일반화된 MultiDomain DataModule 생성.
    
    Args:
        datamodule_class: DataModule 클래스 (예: MultiDomainHDMAPDataModule)
        source_domain: 소스 도메인 이름
        target_domains: 타겟 도메인 리스트 또는 "auto"
        batch_size: 배치 크기
        image_size: 이미지 크기
        dataset_root: 데이터셋 루트 경로
        validation_strategy: 검증 전략
        num_workers: 워커 수
        
    Returns:
        생성된 datamodule 인스턴스
    """
    print(f"\n📦 {datamodule_class.__name__} 생성 중...")
    print(f"   Source Domain: {source_domain}")
    print(f"   Target Domains: {target_domains}")
    
    # 기본 dataset_root 설정
    if dataset_root is None:
        dataset_root = f"./datasets/HDMAP/1000_8bit_resize_{image_size}"
    
    datamodule = datamodule_class(
        root=dataset_root,
        source_domain=source_domain,
        target_domains=target_domains,
        validation_strategy=validation_strategy,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
    )
    
    # 데이터 준비 및 설정
    datamodule.prepare_data()
    datamodule.setup()
    
    print(f"✅ {datamodule_class.__name__} 설정 완료")
    print(f"   실제 Target Domains: {datamodule.target_domains}")
    print(f"   훈련 데이터: {len(datamodule.train_data)} 샘플 (source: {datamodule.source_domain})")
    print(f"   검증 데이터: {len(datamodule.val_data)} 샘플 (source test)")
    
    total_test_samples = sum(len(test_data) for test_data in datamodule.test_data)
    print(f"   테스트 데이터: {total_test_samples} 샘플 (targets)")
    
    for i, target_domain in enumerate(datamodule.target_domains):
        print(f"     └─ {target_domain}: {len(datamodule.test_data[i])} 샘플")
    
    return datamodule


def evaluate_source_domain(
    model: Any, 
    engine: Any, 
    datamodule: Any,
    checkpoint_path: str = None
) -> Dict[str, Any]:
    """일반화된 Source Domain 성능 평가.
    
    Args:
        model: 평가할 모델
        engine: Lightning Engine
        datamodule: 데이터 모듈
        checkpoint_path: 체크포인트 경로 (선택사항)
        
    Returns:
        Dict[str, Any]: 평가 결과
    """
    print(f"\n📊 Source Domain 성능 평가 - {datamodule.source_domain}")
    print("   💡 평가 데이터: Source domain test (validation으로 사용된 데이터)")
    
    # Validation DataLoader를 수동으로 가져와서 engine.test()로 평가
    val_dataloader = datamodule.val_dataloader()
    
    # Engine의 경로 설정 확인 (fit() 후에만 접근 가능)
    try:
        if hasattr(engine, 'trainer') and engine.trainer is not None and hasattr(engine.trainer, 'default_root_dir'):
            print(f"   🔧 Source domain 평가 시 Engine default_root_dir: {engine.trainer.default_root_dir}")
    except Exception as e:
        print(f"   ⚠️ Warning: Engine 경로 확인 실패: {e}")
    
    if checkpoint_path:
        results = engine.test(
            model=model,
            dataloaders=val_dataloader,
            ckpt_path=checkpoint_path
        )
    else:
        results = engine.test(
            model=model,
            dataloaders=val_dataloader
        )
    
    if results and len(results) > 0:
        source_metrics = results[0]
        print(f"   ✅ Source Domain 평가 완료:")
        
        # 주요 메트릭 출력
        for key, value in source_metrics.items():
            if isinstance(value, (int, float)):
                if 'AUROC' in key or 'F1Score' in key:
                    print(f"     └─ {key}: {value:.4f}")
                else:
                    print(f"     └─ {key}: {value}")
        
        return source_metrics
    else:
        print(f"   ❌ Source Domain 평가 실패")
        return {}


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


def find_anomalib_image_paths(base_search_path: Path) -> Optional[Path]:
    """Anomalib이 생성한 이미지 경로를 자동으로 탐색 (모든 모델에서 공통 사용 가능).
    
    Args:
        base_search_path: 탐색할 기본 경로
        
    Returns:
        Optional[Path]: 발견된 이미지 경로의 부모 디렉토리 (images 폴더의 부모)
    """
    # 다양한 모델 구조에 대응하는 패턴들
    search_patterns = [
        "**/CustomDraem/MultiDomainHDMAPDataModule/**/images",  # DRAEM 계열
        "**/Padim/MultiDomainHDMAPDataModule/**/images",        # PaDiM 계열
        "**/*/MultiDomainHDMAPDataModule/**/images",            # 일반적인 패턴
        "**/images"                                              # 가장 일반적인 패턴
    ]
    
    anomalib_image_paths = []
    
    for pattern in search_patterns:
        found_paths = list(base_search_path.glob(pattern))
        anomalib_image_paths.extend(found_paths)
    
    if anomalib_image_paths:
        # 경로 생성 시간 기준으로 최신 선택
        latest_image_path = max(anomalib_image_paths, key=lambda p: p.stat().st_mtime if p.exists() else 0)
        anomalib_results_path = latest_image_path.parent  # images 폴더의 부모
        print(f"   ✅ 실제 Anomalib 결과 경로: {anomalib_results_path}")
        return anomalib_results_path
    
    return None


def organize_source_domain_results(
    sevnet_viz_path: str,
    results_base_dir: str,
    source_domain: str,
    specific_version_path: str = None
) -> bool:
    """Source Domain 평가 결과 이미지를 정리된 폴더로 복사 (모든 모델에서 공통 사용 가능).
    
    목적: engine.test()로 생성된 Source Domain 시각화 결과를 source_domain/ 폴더로 재배치하여
          나중에 분석할 때 용이하게 접근할 수 있도록 함
    
    Source Domain은 보통 fault/, good/ 폴더 구조로 구성됨:
    - fault/: 실제 anomaly가 있는 이미지들의 시각화 결과
    - good/: 정상 이미지들의 시각화 결과
    
    각 이미지는 다음 정보를 포함:
    - Original Image: 원본 이미지
    - Reconstructed: 재구성된 이미지 (reconstruction quality 확인)
    - Anomaly Map: Heat map 형태의 anomaly 점수 분포
    - Image + Pred Mask: Threshold 기반 binary mask (빨간색 영역만 표시)
    - Severity Score: SeverityHead의 심각도 예측값 (0.0~1.0)
      * DRAEM-SevNet은 mask + severity 결합으로 더 정교한 anomaly detection 제공
    
    Args:
        sevnet_viz_path: visualize 폴더 경로
        results_base_dir: 기본 결과 디렉토리 경로
        source_domain: 소스 도메인 이름
        specific_version_path: 특정 버전 경로 (선택적)
        
    Returns:
        bool: 성공 여부
    """
    try:
        # Source 폴더에서 이미지 파일 찾기
        if specific_version_path:
            source_path = Path(specific_version_path)
        else:
            source_path = Path(results_base_dir)
        
        # images 폴더 찾기
        images_folder = None
        for images_path in source_path.rglob("images"):
            if images_path.is_dir():
                images_folder = images_path
                break
        
        if not images_folder or not images_folder.exists():
            print(f"   ⚠️ Warning: {source_domain} images 폴더를 찾을 수 없습니다: {source_path}")
            return False
        
        # 타겟 경로 (visualize/source_domain/)
        sevnet_viz_path_obj = Path(sevnet_viz_path)
        target_source_path = sevnet_viz_path_obj / "source_domain"
        target_source_path.mkdir(parents=True, exist_ok=True)
        
        print(f"   📂 Source Domain 이미지 복사: {images_folder} → {target_source_path}")
        
        # fault/, good/ 폴더 복사
        copied_folders = []
        for subfolder in ["fault", "good"]:
            source_subfolder = images_folder / subfolder
            target_subfolder = target_source_path / subfolder
            
            if source_subfolder.exists():
                if target_subfolder.exists():
                    shutil.rmtree(target_subfolder)
                shutil.copytree(source_subfolder, target_subfolder)
                
                image_count = len(list(target_subfolder.glob("*.png")))
                copied_folders.append(f"{subfolder}({image_count})")
                print(f"     ✅ {subfolder}: {image_count} 이미지 복사 완료")
        
        if copied_folders:
            print(f"   ✅ Source Domain 복사 완료: {', '.join(copied_folders)}")
            return True
        else:
            print(f"   ⚠️ Warning: 복사할 이미지가 없습니다")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: Source Domain 이미지 복사 실패: {e}")
        return False


def copy_target_domain_results(
    domain: str,
    results_base_dir: str = None,
    specific_version_path: str = None
) -> bool:
    """Target Domain 평가 결과 전체 복사 및 보존 (모든 모델에서 공통 사용 가능).
    
    각 Target Domain 평가가 완료되면 images/ 폴더의 모든 결과를 
    visualize/target_domains/{domain}/ 폴더로 완전히 복사하여 보존합니다.
    
    목적: engine.test()로 생성된 시각화 결과를 도메인별로 재배치하여 
          나중에 분석할 때 용이하게 접근할 수 있도록 함
    
    Args:
        domain: 타겟 도메인 이름
        results_base_dir: 기본 결과 디렉토리 경로 (선택적)
        specific_version_path: 특정 버전 경로 (선택적)
        
    Returns:
        bool: 성공 여부
    """
    try:
        # 경로 결정 (specific_version_path 우선, 그 다음 results_base_dir)
        if specific_version_path:
            base_path = Path(specific_version_path)
        elif results_base_dir:
            base_path = Path(results_base_dir)
        else:
            print(f"         ❌ Error: 경로가 지정되지 않았습니다")
            return False
        
        # 타겟 경로 (visualize/target_domains/{domain}/)
        sevnet_viz_path = base_path / "visualize"
        target_domain_path = sevnet_viz_path / "target_domains" / domain
        target_domain_path.mkdir(parents=True, exist_ok=True)
        
        # Source에서 images 폴더 찾기
        all_images_paths = list(base_path.rglob("images"))
        
        # 만약 찾지 못했다면, 부모 경로에서도 탐색 (실제 Anomalib 결과 경로 포함)
        if not all_images_paths and base_path.name == "tensorboard_logs":
            parent_path = base_path.parent
            all_images_paths = list(parent_path.rglob("images"))
        
        images_folder = None
        for images_path in all_images_paths:
            if images_path.is_dir():
                images_folder = images_path
                break
        
        if not images_folder or not images_folder.exists():
            print(f"         ⚠️ Warning: {domain} images 폴더를 찾을 수 없습니다")
            return False
        
        # images 폴더 전체 복사
        copied_count = 0
        for item in images_folder.iterdir():
            if item.is_file() and item.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                target_file = target_domain_path / item.name
                shutil.copy2(item, target_file)
                copied_count += 1
            elif item.is_dir():
                target_subfolder = target_domain_path / item.name
                if target_subfolder.exists():
                    shutil.rmtree(target_subfolder)
                shutil.copytree(item, target_subfolder)
                subfolder_count = len(list(target_subfolder.rglob("*.png")))
                copied_count += subfolder_count
        
        if copied_count > 0:
            print(f"         ✅ {domain}: {copied_count} 이미지 복사 완료")
            return True
        else:
            print(f"         ⚠️ Warning: {domain}에 복사할 이미지가 없습니다")
            return False
            
    except Exception as e:
        print(f"         ❌ Error: {domain} 이미지 복사 실패: {e}")
        return False


def evaluate_target_domains(
    model: Any,
    engine: Engine,
    datamodule: Any,
    checkpoint_path: str,
    results_base_dir: str,
    save_samples: bool = True,
    current_version_path: str = None
) -> Dict[str, Dict[str, Any]]:
    """Target domains에 대한 성능 평가 수행 (모든 모델에서 공통 사용 가능).
    
    Args:
        model: 훈련된 모델
        engine: Anomalib Engine
        datamodule: Multi-domain 데이터모듈 (source용)
        checkpoint_path: 모델 체크포인트 경로
        results_base_dir: 결과 저장 기본 경로
        save_samples: 샘플 이미지 저장 여부
        current_version_path: 현재 버전 경로 (시각화 저장용)
        
    Returns:
        Dict[str, Dict[str, Any]]: 각 target domain별 평가 결과
    """
    from anomalib.data.datamodules.image.multi_domain_hdmap import MultiDomainHDMAPDataModule
    
    target_results = {}
    target_domains = ["domain_B", "domain_C", "domain_D"]
    
    for domain in target_domains:
        print(f"      🎯 Target Domain 평가: {domain}")
        
        try:
            # 개별 Target Domain용 DataModule 생성
            target_datamodule = MultiDomainHDMAPDataModule(
                root=datamodule.root,
                source_domain="domain_A",  # 원래 source domain 유지
                target_domains=[domain],   # 평가할 domain을 target으로 설정
                validation_strategy=getattr(datamodule, 'validation_strategy', "source_test"),
                train_batch_size=datamodule.train_batch_size,
                eval_batch_size=datamodule.eval_batch_size,
                num_workers=datamodule.num_workers
            )
            
            # Test 단계 설정
            target_datamodule.setup(stage="test")
            
            # 모델 평가
            print(f"         📊 {domain} DataModule 설정 완료, test 시작...")
            
            # Engine의 경로 설정 확인 (fit() 후에만 접근 가능)
            try:
                if hasattr(engine, 'trainer') and engine.trainer is not None:
                    engine.trainer.default_root_dir = results_base_dir
                    print(f"         🔧 Engine default_root_dir 재설정: {results_base_dir}")
            except Exception as e:
                print(f"         ⚠️ Warning: Engine 경로 재설정 실패: {e}")
            
            result = engine.test(
                model=model, 
                datamodule=target_datamodule,
                ckpt_path=checkpoint_path
            )
            
            print(f"         🔍 {domain} 평가 결과 타입: {type(result)}")
            print(f"         🔍 {domain} 평가 결과 내용: {result}")
            
            # 결과 저장
            if result:
                target_results[domain] = result[0] if isinstance(result, list) else result
                print(f"         ✅ {domain} 평가 완료 - AUROC: {target_results[domain].get('image_AUROC', 'N/A')}")
                if isinstance(target_results[domain].get('image_AUROC'), (int, float)):
                    print(f"         📊 {domain} 상세 성능: AUROC={target_results[domain].get('image_AUROC'):.4f}, F1={target_results[domain].get('image_F1Score', 'N/A')}")
            else:
                print(f"         ⚠️ Warning: {domain} 평가 결과가 None입니다")
                target_results[domain] = {"image_AUROC": 0.0, "image_F1Score": 0.0}
            
            # 이미지 복사 (선택적)
            if save_samples and current_version_path:
                copy_success = copy_target_domain_results(
                    domain=domain,
                    results_base_dir=results_base_dir,
                    specific_version_path=current_version_path
                )
                if not copy_success:
                    print(f"         ⚠️ Warning: {domain} 이미지 복사 실패")
                
        except Exception as e:
            print(f"         ❌ Error: {domain} 평가 실패: {e}")
            target_results[domain] = {"image_AUROC": 0.0, "image_F1Score": 0.0, "error": str(e)}
    
    return target_results


def save_experiment_results(
    result: Dict[str, Any], 
    result_filename: str, 
    log_dir: Path, 
    logger: logging.Logger
) -> Path:
    """실험 결과를 JSON 파일로 저장 (모든 모델에서 공통 사용 가능).
    
    Args:
        result: 실험 결과 딕셔너리
        result_filename: 저장할 파일명
        log_dir: 로그 디렉토리 (실패 시 사용)
        logger: 로거 객체
        
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
        logger.info(f"   Source Domain AUROC: {result['source_results'].get('image_AUROC', 'N/A'):.4f}")
        logger.info(f"   Target Domains Avg AUROC: {result.get('avg_target_auroc', 'N/A'):.4f}")
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
            logger.info(f"   최고 Validation AUROC: {training_info.get('best_val_auroc', 'N/A')}")
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


def analyze_multi_experiment_results(all_results: list, source_domain: str):
    """다중 실험 결과 분석 및 비교 (모든 모델에서 공통 사용 가능).
    
    Args:
        all_results: 모든 실험 결과 리스트
        source_domain: 소스 도메인 이름
    """
    print(f"\n{'='*80}")
    print(f"📈 다중 실험 결과 분석 및 비교")
    print(f"Source Domain: {source_domain}")
    print(f"{'='*80}")
    
    successful_results = [r for r in all_results if r["status"] == "success"]
    failed_results = [r for r in all_results if r["status"] == "failed"]
    
    print(f"\n📊 실험 요약:")
    print(f"   성공: {len(successful_results)}/{len(all_results)} 개")
    print(f"   실패: {len(failed_results)}/{len(all_results)} 개")
    
    if failed_results:
        print(f"\n❌ 실패한 실험들:")
        for result in failed_results:
            print(f"   - {result['condition']['name']}: {result['error']}")
    
    if successful_results:
        print(f"\n🏆 실험 결과 순위 (Target Domain 평균 AUROC 기준):")
        # Target Domain 평균 AUROC 기준으로 정렬
        sorted_results = sorted(successful_results, 
                              key=lambda x: x.get("avg_target_auroc", 0), 
                              reverse=True)
        
        print(f"{'순위':<4} {'실험 조건':<30} {'Source AUROC':<12} {'Target Avg':<12} {'도메인 전이':<10}")
        print("-" * 80)
        
        for idx, result in enumerate(sorted_results, 1):
            condition_name = result["condition"]["name"]
            source_auroc = result["source_results"].get("image_AUROC", 0)
            avg_target_auroc = result.get("avg_target_auroc", 0)
            
            # 도메인 전이 효과 계산
            if source_auroc > 0:
                transfer_effect = avg_target_auroc / source_auroc
                transfer_desc = "우수" if transfer_effect > 0.9 else "양호" if transfer_effect > 0.8 else "개선필요"
            else:
                transfer_desc = "N/A"
            
            print(f"{idx:<4} {condition_name:<30} {source_auroc:<12.3f} {avg_target_auroc:<12.3f} {transfer_desc:<10}")
        
        # 최고 성능 실험 세부 분석
        best_result = sorted_results[0]
        print(f"\n🥇 최고 성능 실험: {best_result['condition']['name']}")
        print(f"   📊 Target Domain별 세부 성능:")
        
        target_performances = []
        for domain, result in best_result["target_results"].items():
            domain_auroc = result.get("image_AUROC", 0)
            if isinstance(domain_auroc, (int, float)):
                print(f"   {domain:<12} {domain_auroc:<12.3f}")
                target_performances.append((domain, domain_auroc))
        
        # 도메인별 성능 분석
        if target_performances:
            best_domain = max(target_performances, key=lambda x: x[1])
            worst_domain = min(target_performances, key=lambda x: x[1])
            print(f"\n   🎯 최고 성능 도메인: {best_domain[0]} (AUROC: {best_domain[1]:.3f})")
            print(f"   ⚠️  최저 성능 도메인: {worst_domain[0]} (AUROC: {worst_domain[1]:.3f})")
