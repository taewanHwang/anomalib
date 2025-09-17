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
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# Anomalib imports
from anomalib.engine import Engine
from anomalib.data.datamodules.image.multi_domain_hdmap import MultiDomainHDMAPDataModule
from anomalib.data.datamodules.image.all_domains_hdmap import AllDomainsHDMAPDataModule


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
    
    return experiment_conditions


def load_all_domains_experiment_conditions(json_filename: str) -> List[Dict[str, Any]]:
    """
    AllDomains 실험을 위한 JSON 파일에서 실험 조건을 로드합니다.
    
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
    
    # AllDomains JSON은 직접 리스트 형태
    if isinstance(data, list):
        return data
    else:
        raise ValueError("AllDomains JSON 파일은 리스트 형태여야 합니다.")


def get_experiment_by_name(experiment_conditions: List[Dict[str, Any]], 
                          experiment_name: str) -> Dict[str, Any]:
    """
    실험 이름으로 특정 실험 조건을 찾습니다.
    
    Args:
        experiment_conditions: 전체 실험 조건 리스트
        experiment_name: 찾을 실험 이름
        
    Returns:
        해당 실험 조건 딕셔너리
        
    Raises:
        ValueError: 해당 이름의 실험을 찾을 수 없는 경우
    """
    for condition in experiment_conditions:
        if condition.get('name') == experiment_name:
            return condition
    
    available_names = [c.get('name', 'Unknown') for c in experiment_conditions]
    raise ValueError(f"실험 '{experiment_name}'을 찾을 수 없습니다. "
                    f"사용 가능한 실험: {available_names}")


def validate_experiment_config(config: Dict[str, Any]) -> bool:
    """
    실험 설정의 유효성을 검사합니다.
    
    Args:
        config: 검사할 실험 설정 딕셔너리
        
    Returns:
        설정이 유효하면 True, 그렇지 않으면 False
    """
    required_fields = [
        'max_epochs', 'learning_rate', 'batch_size', 'image_size',
        'source_domain', 'target_domains'
    ]
    
    for field in required_fields:
        if field not in config:
            print(f"필수 설정 '{field}'가 누락되었습니다.")
            return False
    
    # 값의 유효성 검사
    if config['max_epochs'] <= 0:
        print("max_epochs는 0보다 커야 합니다.")
        return False
        
    if config['learning_rate'] <= 0:
        print("learning_rate는 0보다 커야 합니다.")
        return False
        
    if config['batch_size'] <= 0:
        print("batch_size는 0보다 커야 합니다.")
        return False
    
    return True


def print_experiment_summary(experiment_conditions: List[Dict[str, Any]]) -> None:
    """
    실험 조건들의 요약 정보를 출력합니다.
    
    Args:
        experiment_conditions: 실험 조건 리스트
    """
    print(f"\n=== 실험 조건 요약 (총 {len(experiment_conditions)}개) ===")
    
    for i, condition in enumerate(experiment_conditions, 1):
        name = condition.get('name', 'Unknown')
        description = condition.get('description', 'No description')
        config = condition.get('config', {})
        
        epochs = config.get('max_epochs', 'Unknown')
        lr = config.get('learning_rate', 'Unknown')
        
        print(f"{i:2d}. {name}")
        print(f"    설명: {description}")
        print(f"    에포크: {epochs}, 학습률: {lr}")
        
        if 'patch_width_range' in config and 'patch_ratio_range' in config:
            width_range = config['patch_width_range']
            ratio_range = config['patch_ratio_range']
            print(f"    패치 크기: {width_range}, 비율: {ratio_range}")
        
        print()


def extract_target_domains_from_config(config: Dict[str, Any]) -> List[str]:
    """
    실험 설정에서 target domains를 추출합니다.
    
    Args:
        config: 실험 설정 딕셔너리
        
    Returns:
        List[str]: target domain 리스트
    """
    target_domains = config['target_domains']
    
    if target_domains == 'auto':
        # 기본 HDMAP 도메인 (source_domain 제외)
        source_domain = config['source_domain']
        all_domains = ['domain_A', 'domain_B', 'domain_C', 'domain_D']
        target_domains = [d for d in all_domains if d != source_domain]
    elif isinstance(target_domains, str):
        target_domains = [target_domains]
    elif not isinstance(target_domains, list):
        target_domains = ['domain_B', 'domain_C', 'domain_D']
    
    return target_domains


def analyze_experiment_results(
    source_results: Dict[str, Any],
    target_results: Dict[str, Dict[str, Any]],
    training_info: Dict[str, Any],
    condition: Dict[str, Any],
    model_type: str = "Model"
) -> Dict[str, Any]:
    """
    실험 결과를 분석합니다 (모든 모델에서 공통 사용 가능).
    
    Args:
        source_results: 소스 도메인 평가 결과
        target_results: 타겟 도메인 평가 결과
        training_info: 훈련 정보
        condition: 실험 조건
        model_type: 모델 타입 (출력용)
        
    Returns:
        Dict[str, Any]: 분석된 결과 딕셔너리
    """
    print(f"\n📊 {model_type} 실험 결과 분석")
    
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
    
    for domain, perf in domain_performances.items():
        print(f"   └─ {domain}: AUROC={perf['auroc']:.4f}")
    
    return analysis


def create_common_experiment_result(
    condition: Dict[str, Any],
    status: str = "success",
    experiment_path: str = None,
    source_results: Dict[str, Any] = None,
    target_results: Dict[str, Dict[str, Any]] = None,
    training_info: Dict[str, Any] = None,
    best_checkpoint: str = None,
    error: str = None
) -> Dict[str, Any]:
    """
    공통 실험 결과 딕셔너리를 생성합니다.
    
    Args:
        condition: 실험 조건
        status: 실험 상태 ("success" 또는 "failed")
        experiment_path: 실험 경로
        source_results: 소스 도메인 결과
        target_results: 타겟 도메인 결과들
        training_info: 훈련 정보
        best_checkpoint: 최고 체크포인트 경로
        error: 에러 메시지 (실패 시)
        
    Returns:
        Dict[str, Any]: 실험 결과 딕셔너리
    """
    result = {
        "condition": condition,
        "status": status,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_path": experiment_path,
        "source_results": source_results or {},
        "target_results": target_results or {},
        "training_info": training_info or {},
        "best_checkpoint": best_checkpoint,
    }
    
    if status == "failed":
        result["error"] = error
    else:
        # Target domain 평균 AUROC 계산
        if target_results:
            target_aurocs = []
            for domain, domain_result in target_results.items():
                auroc = domain_result.get('image_AUROC')
                if isinstance(auroc, (int, float)):
                    target_aurocs.append(auroc)
            
            if target_aurocs:
                result["avg_target_auroc"] = sum(target_aurocs) / len(target_aurocs)
            else:
                result["avg_target_auroc"] = 0.0
        else:
            result["avg_target_auroc"] = 0.0
    
    return result


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


def create_all_domains_datamodule(
    datamodule_class,
    batch_size: int,
    image_size: str,
    domains: list[str] = None,
    val_split_ratio: float = 0.2,
    dataset_root: str = None,
    num_workers: int = 8
) -> AllDomainsHDMAPDataModule:
    """AllDomainsHDMAPDataModule 생성 및 설정.
    
    Args:
        datamodule_class: AllDomainsHDMAPDataModule 클래스 (일관성을 위해 추가, 실제로는 사용하지 않음)
        batch_size: 배치 크기
        image_size: 이미지 크기 (예: "392x392")
        domains: 사용할 도메인 리스트. None이면 모든 도메인 사용
        val_split_ratio: 검증 데이터 분할 비율
        dataset_root: 데이터셋 루트 경로 (None이면 자동 생성)
        num_workers: 워커 수
        
    Returns:
        설정된 AllDomainsHDMAPDataModule
    """
    print(f"\n📦 AllDomainsHDMAPDataModule 생성 중...")
    
    # 이미지 크기 파싱
    try:
        width, height = map(int, image_size.split('x'))
        image_size_tuple = (width, height)
    except ValueError:
        # 기본값 사용
        image_size_tuple = (392, 392)
        print(f"   ⚠️ 이미지 크기 파싱 실패, 기본값 사용: {image_size_tuple}")
    
    # 도메인 정보 출력
    domains_info = f"전체 도메인 (A~D)" if not domains else f"{domains}"
    print(f"   🌍 도메인: {domains_info}")
    print(f"   📏 이미지 크기: {image_size_tuple}")
    print(f"   📊 배치 크기: {batch_size}")
    print(f"   🔄 Val 분할 비율: {val_split_ratio}")
    
    # 이미지 크기에 따른 데이터셋 루트 경로 설정
    if dataset_root is None:
        # 현재 작업 디렉토리를 기준으로 절대 경로 생성
        import os
        current_dir = os.getcwd()
        dataset_root = os.path.join(current_dir, "datasets", "HDMAP", f"1000_8bit_resize_{image_size}")
    
    # AllDomainsHDMAPDataModule 생성
    datamodule = AllDomainsHDMAPDataModule(
        root=dataset_root,
        domains=domains,  # None이면 모든 도메인 사용
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        val_split_ratio=val_split_ratio,  # train에서 validation 분할
        seed=42
    )
    
    # 데이터 설정
    print(f"   ⚙️  DataModule 설정 중...")
    datamodule.setup()
    
    # 데이터 통계 출력
    print(f"   ✅ DataModule 설정 완료!")
    print(f"      • Train 샘플: {len(datamodule.train_data):,}개 (모든 도메인 정상 데이터)")
    print(f"      • Val 샘플: {len(datamodule.val_data):,}개 (train에서 분할)")
    print(f"      • Test 샘플: {len(datamodule.test_data):,}개 (모든 도메인 정상+결함)")
    
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
    print("   🎯 재현성을 위해 훈련에 사용된 동일한 DataModule의 val_dataloader 사용")
    print(f"   📋 검증 데이터셋 크기: {len(datamodule.val_data)} 샘플")
    
    # 훈련에 사용된 동일한 DataModule의 validation DataLoader 사용
    # 이렇게 하면 완전히 동일한 데이터셋 인스턴스와 순서를 보장
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
        
        # test_image_AUROC -> image_AUROC 키 변환 (표준화)
        if 'test_image_AUROC' in source_metrics:
            source_metrics['image_AUROC'] = source_metrics['test_image_AUROC']
        if 'test_image_F1Score' in source_metrics:
            source_metrics['image_F1Score'] = source_metrics['test_image_F1Score']
        
        print(f"   ✅ Source Domain 평가 완료:")
        print(f"   📝 주요 메트릭 (Validation과 동일해야 함):")
        
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


def find_anomalib_image_paths(base_search_path: Path) -> Optional[Path]:
    """Anomalib이 생성한 이미지 경로를 자동으로 탐색 (모든 모델에서 공통 사용 가능).
    
    Args:
        base_search_path: 탐색할 기본 경로
        
    Returns:
        Optional[Path]: 발견된 이미지 경로의 부모 디렉토리 (images 폴더의 부모)
    """
    # 다양한 모델 구조에 대응하는 패턴들
    search_patterns = [
        "**/DraemSevNet/MultiDomainHDMAPDataModule/**/images",  # DRAEM 계열
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
    specific_version_path: str = None,
    visualization_base_path: str = None
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
        visualization_base_path: 시각화 저장 기본 경로 (선택적)
        
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
        
        # 시각화 경로 결정
        if visualization_base_path:
            viz_base_path = Path(visualization_base_path)
        else:
            viz_base_path = base_path / "visualize"
        
        # 타겟 경로 (visualize/target_domains/{domain}/)
        target_domain_path = viz_base_path / "target_domains" / domain
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
    target_domains: List[str] = None,
    datamodule_class = None,
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
        target_domains: 평가할 target domain 리스트 (None이면 datamodule에서 추출)
        datamodule_class: DataModule 클래스 (None이면 자동 감지)
        save_samples: 샘플 이미지 저장 여부
        current_version_path: 현재 버전 경로 (시각화 저장용)
        
    Returns:
        Dict[str, Dict[str, Any]]: 각 target domain별 평가 결과
    """
    # DataModule 클래스 자동 감지
    if datamodule_class is None:
        datamodule_class = type(datamodule)
    
    # Target domains 자동 추출
    if target_domains is None:
        if hasattr(datamodule, 'target_domains'):
            target_domains = datamodule.target_domains
        else:
            # 기본값으로 HDMAP 도메인 사용
            target_domains = ["domain_B", "domain_C", "domain_D"]
            print(f"   ⚠️ Warning: target_domains를 자동 감지할 수 없어 기본값 사용: {target_domains}")
    
    print(f"   🎯 평가할 Target Domains: {target_domains}")
    
    target_results = {}
    
    for domain in target_domains:
        print(f"      🎯 Target Domain 평가: {domain}")
        
        try:
            # 개별 Target Domain용 DataModule 생성 (동적 클래스 사용)
            target_datamodule = datamodule_class(
                root=datamodule.root,
                source_domain=getattr(datamodule, 'source_domain', "domain_A"),  # 원래 source domain 유지
                target_domains=[domain],   # 평가할 domain을 target으로 설정
                validation_strategy=getattr(datamodule, 'validation_strategy', "source_test"),
                train_batch_size=getattr(datamodule, 'train_batch_size', 16),
                eval_batch_size=getattr(datamodule, 'eval_batch_size', 16),
                num_workers=getattr(datamodule, 'num_workers', 16)
            )
            
            # Test 단계 설정
            target_datamodule.setup(stage="test")
            
            # 모델 평가
            print(f"         📊 {domain} DataModule 설정 완료, test 시작...")
            
            # Note: Engine의 default_root_dir은 훈련 후 변경 불가능하므로 재설정하지 않음
            # 결과는 각 실험의 tensorboard_logs 폴더에 정상적으로 저장됨
            
            result = engine.test(
                model=model, 
                datamodule=target_datamodule,
                ckpt_path=checkpoint_path
            )
            
            print(f"         🔍 {domain} 평가 결과 타입: {type(result)}")
            print(f"         🔍 {domain} 평가 결과 내용: {result}")
            
            # 결과 저장
            if result:
                domain_result = result[0] if isinstance(result, list) else result
                
                # test_image_AUROC -> image_AUROC 키 변환 (표준화)
                if 'test_image_AUROC' in domain_result:
                    domain_result['image_AUROC'] = domain_result['test_image_AUROC']
                if 'test_image_F1Score' in domain_result:
                    domain_result['image_F1Score'] = domain_result['test_image_F1Score']
                
                target_results[domain] = domain_result
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
        
        # 모델별 추가 점수들 (있는 경우에만)
        if predictions.get("mask_scores") and isinstance(predictions.get("mask_scores"), list) and len(predictions.get("mask_scores")) > i:
            row["mask_score"] = predictions["mask_scores"][i]
        else:
            row["mask_score"] = 0.0
            
        # DRAEM-SevNet 모델의 경우 raw_severity_score와 normalized_severity_score 구분
        if predictions.get("raw_severity_scores") and isinstance(predictions.get("raw_severity_scores"), list) and len(predictions.get("raw_severity_scores")) > i:
            row["raw_severity_score"] = predictions["raw_severity_scores"][i]
        else:
            row["raw_severity_score"] = 0.0
            
        if predictions.get("normalized_severity_scores") and isinstance(predictions.get("normalized_severity_scores"), list) and len(predictions.get("normalized_severity_scores")) > i:
            row["normalized_severity_score"] = predictions["normalized_severity_scores"][i]
        else:
            row["normalized_severity_score"] = 0.0
            
        # 기존 severity_score는 backward compatibility를 위해 유지 (normalized_severity_score와 동일)
        if predictions.get("severity_scores") and isinstance(predictions.get("severity_scores"), list) and len(predictions.get("severity_scores")) > i:
            row["severity_score"] = predictions["severity_scores"][i]
        else:
            # normalized_severity_score와 동일한 값 사용
            row["severity_score"] = row["normalized_severity_score"]
        
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


def save_extreme_samples(
    image_paths: List[str],
    ground_truth: List[int],
    scores: List[float],
    predictions: List[int],
    result_dir: Path,
    n_samples: int = 10
) -> None:
    """
    극값 샘플들(고신뢰도 맞춤/틀림, 저신뢰도)의 경로를 저장합니다.
    
    Args:
        image_paths: 이미지 경로 리스트
        ground_truth: 실제 정답 리스트
        scores: 예측 점수 리스트
        predictions: 예측 결과 리스트
        result_dir: 결과 저장 디렉토리  
        n_samples: 각 카테고리별 저장할 샘플 수
    """
    import numpy as np
    import pandas as pd
    
    # result_dir을 analysis_dir로 직접 사용 (중복 폴더 생성 방지)  
    analysis_dir = Path(result_dir)
    extreme_dir = analysis_dir / "extreme_samples"
    extreme_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터 정리
    data = pd.DataFrame({
        'image_path': image_paths,
        'ground_truth': ground_truth,
        'score': scores,
        'prediction': predictions
    })
    
    # 정확도 계산
    data['correct'] = (data['ground_truth'] == data['prediction'])
    data['confidence'] = np.abs(data['score'] - 0.5)  # 0.5에서 얼마나 먼지로 신뢰도 측정
    
    # 카테고리별 샘플 추출
    categories = {
        'high_confidence_correct': data[(data['correct'] == True)].nlargest(n_samples, 'confidence'),
        'high_confidence_wrong': data[(data['correct'] == False)].nlargest(n_samples, 'confidence'), 
        'low_confidence_samples': data.nsmallest(n_samples, 'confidence')
    }
    
    # 각 카테고리별로 CSV 저장
    for category, samples in categories.items():
        if len(samples) > 0:
            csv_path = extreme_dir / f"{category}.csv"
            samples.to_csv(csv_path, index=False)
            print(f"📸 {category} 샘플 저장: {csv_path}")


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


def analyze_dataset_statistics(datamodule, train_size: int, test_size: int, val_size: int = None) -> dict:
    """훈련, 테스트, 검증 데이터의 픽셀값 통계량을 분석합니다.
    
    Args:
        datamodule: 데이터를 제공하는 데이터모듈
        train_size: 훈련 데이터 크기
        test_size: 테스트 데이터 크기
        val_size: 검증 데이터 크기 (선택사항)
        
    Returns:
        dict: 각 데이터셋별 통계량 정보
    """
    import torch
    import numpy as np
    from typing import List
    
    print(f"   📊 데이터셋 픽셀값 통계 분석 시작...")
    
    statistics = {
        "train": {"values": [], "labels": [], "count": 0},
        "test": {"values": [], "labels": [], "count": 0}
    }
    
    if val_size is not None and val_size > 0:
        statistics["val"] = {"values": [], "labels": [], "count": 0}
    
    # 훈련 데이터 분석
    print(f"      🔍 훈련 데이터 분석 중 (총 {train_size}개)...")
    train_loader = datamodule.train_dataloader()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            # 이미지 데이터 수집
            images = batch.image.numpy()  # (B, C, H, W)
            labels = batch.gt_label.numpy() if hasattr(batch, 'gt_label') else np.zeros(images.shape[0])
            
            # 각 이미지별로 픽셀값과 라벨을 매핑
            for img, label in zip(images, labels):
                img_values = img.flatten()
                statistics["train"]["values"].extend(img_values.tolist())
                # 이미지의 모든 픽셀에 대해 동일한 라벨 적용
                statistics["train"]["labels"].extend([label] * len(img_values))
            
            statistics["train"]["count"] += len(labels)
            
            # 진행률 표시
            if batch_idx % 50 == 0 and batch_idx > 0:
                print(f"         📈 훈련 데이터: {batch_idx+1} 배치, {statistics['train']['count']}개 처리됨")
    
    # 테스트 데이터 분석
    print(f"      🔍 테스트 데이터 분석 중 (총 {test_size}개)...")
    test_loader = datamodule.test_dataloader()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # 이미지 데이터 수집
            images = batch.image.numpy()  # (B, C, H, W)
            labels = batch.gt_label.numpy() if hasattr(batch, 'gt_label') else np.zeros(images.shape[0])
            
            # 각 이미지별로 픽셀값과 라벨을 매핑
            for img, label in zip(images, labels):
                img_values = img.flatten()
                statistics["test"]["values"].extend(img_values.tolist())
                # 이미지의 모든 픽셀에 대해 동일한 라벨 적용
                statistics["test"]["labels"].extend([label] * len(img_values))
            
            statistics["test"]["count"] += len(labels)
            
            # 진행률 표시
            if batch_idx % 50 == 0 and batch_idx > 0:
                print(f"         📈 테스트 데이터: {batch_idx+1} 배치, {statistics['test']['count']}개 처리됨")
    
    # 검증 데이터 분석 (있는 경우)
    if "val" in statistics:
        print(f"      🔍 검증 데이터 분석 중 (총 {val_size}개)...")
        val_loader = datamodule.val_dataloader()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # 이미지 데이터 수집
                images = batch.image.numpy()  # (B, C, H, W)
                labels = batch.gt_label.numpy() if hasattr(batch, 'gt_label') else np.zeros(images.shape[0])
                
                # 각 이미지별로 픽셀값과 라벨을 매핑
                for img, label in zip(images, labels):
                    img_values = img.flatten()
                    statistics["val"]["values"].extend(img_values.tolist())
                    # 이미지의 모든 픽셀에 대해 동일한 라벨 적용
                    statistics["val"]["labels"].extend([label] * len(img_values))
                
                statistics["val"]["count"] += len(labels)
                
                # 진행률 표시
                if batch_idx % 50 == 0 and batch_idx > 0:
                    print(f"         📈 검증 데이터: {batch_idx+1} 배치, {statistics['val']['count']}개 처리됨")
    
    # 통계량 계산 및 출력
    print(f"   📊 픽셀값 통계량 계산 중...")
    
    results = {}
    for split_name, data in statistics.items():
        if len(data["values"]) == 0:
            continue
            
        values = np.array(data["values"])
        labels = np.array(data["labels"])
        
        # 전체 통계
        overall_stats = {
            "count": len(values),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75))
        }
        
        # 라벨별 통계 (라벨이 있는 경우)
        label_stats = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            label_values = values[mask] if np.any(mask) else np.array([])
            
            if len(label_values) > 0:
                label_name = "normal" if label == 0 else "fault"
                label_stats[label_name] = {
                    "count": len(label_values),
                    "min": float(np.min(label_values)),
                    "max": float(np.max(label_values)),
                    "mean": float(np.mean(label_values)),
                    "std": float(np.std(label_values)),
                    "median": float(np.median(label_values))
                }
        
        results[split_name] = {
            "overall": overall_stats,
            "by_label": label_stats
        }
    
    # 결과 출력
    print(f"   ✅ 데이터셋 통계 분석 완료!")
    print(f"   📋 === 픽셀값 통계 요약 ===")
    
    for split_name, split_stats in results.items():
        split_display = {"train": "훈련", "test": "테스트", "val": "검증"}
        print(f"   📊 {split_display.get(split_name, split_name).upper()} 데이터:")
        
        overall = split_stats["overall"]
        print(f"      🔢 전체 픽셀: {overall['count']:,}개")
        print(f"      📏 범위: [{overall['min']:.4f}, {overall['max']:.4f}]")
        print(f"      📊 평균±표준편차: {overall['mean']:.4f} ± {overall['std']:.4f}")
        print(f"      📍 중위수: {overall['median']:.4f}")
        print(f"      📈 Q1/Q3: {overall['q25']:.4f} / {overall['q75']:.4f}")
        
        # 라벨별 통계
        if split_stats["by_label"]:
            for label_name, label_stat in split_stats["by_label"].items():
                label_emoji = "✅" if label_name == "normal" else "🚨"
                print(f"      {label_emoji} {label_name.upper()} ({label_stat['count']:,}개 샘플):")
                print(f"         📏 범위: [{label_stat['min']:.4f}, {label_stat['max']:.4f}]")
                print(f"         📊 평균±표준편차: {label_stat['mean']:.4f} ± {label_stat['std']:.4f}")
        print()
    
    # 데이터셋 간 비교
    if len(results) > 1:
        print(f"   🔍 === 데이터셋 간 비교 ===")
        
        # 평균값 비교
        means = {name: stats["overall"]["mean"] for name, stats in results.items()}
        stds = {name: stats["overall"]["std"] for name, stats in results.items()}
        ranges = {name: (stats["overall"]["max"] - stats["overall"]["min"]) 
                 for name, stats in results.items()}
        
        print(f"   📊 평균값 비교:")
        for name, mean_val in means.items():
            split_name = {"train": "훈련", "test": "테스트", "val": "검증"}.get(name, name)
            print(f"      {split_name}: {mean_val:.4f}")
        
        print(f"   📏 표준편차 비교:")
        for name, std_val in stds.items():
            split_name = {"train": "훈련", "test": "테스트", "val": "검증"}.get(name, name)
            print(f"      {split_name}: {std_val:.4f}")
        
        print(f"   📈 값 범위 비교:")
        for name, range_val in ranges.items():
            split_name = {"train": "훈련", "test": "테스트", "val": "검증"}.get(name, name)
            print(f"      {split_name}: {range_val:.4f}")
        
        # 분포 일관성 확인
        train_mean = means.get("train", 0)
        test_mean = means.get("test", 0)
        
        if "train" in means and "test" in means:
            mean_diff = abs(train_mean - test_mean)
            if mean_diff > 0.1:  # 임계값 설정
                print(f"   ⚠️  경고: 훈련/테스트 데이터 평균값 차이가 큽니다 (차이: {mean_diff:.4f})")
            else:
                print(f"   ✅ 훈련/테스트 데이터 분포가 일관성 있습니다 (차이: {mean_diff:.4f})")
    
    return results


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
    
    # 데이터 수집을 위한 리스트들
    all_image_paths = []
    all_ground_truth = []
    all_scores = []
    all_mask_scores = []
    all_severity_scores = []
    all_raw_severity_scores = []
    all_normalized_severity_scores = []
    
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
            
            # 이미지 텐서 추출
            image_tensor = batch.image
            print(f"      🖼️  이미지 텐서 크기: {image_tensor.shape}, 경로 수: {len(image_paths)}")
            
            # 이미지 텐서를 모델과 같은 디바이스로 이동
            image_tensor = image_tensor.to(device)
            
            # 모델로 직접 예측 수행
            model_output = torch_model(image_tensor)
            print(f"      ✅ 모델 출력 완료: {type(model_output)}")
            
            # DRAEM 모델의 경우 NaN 디버깅
            if model_type.lower() == "draem" and hasattr(model_output, 'pred_score'):
                pred_score_tensor = model_output.pred_score
                print(f"      🔍 DRAEM 디버깅:")
                print(f"         pred_score shape: {pred_score_tensor.shape}")
                
                # pred_score가 유효한지 확인
                if torch.isnan(pred_score_tensor).any():
                    print(f"         ❌ pred_score에 NaN 발견! 개수: {torch.isnan(pred_score_tensor).sum().item()}")
                else:
                    print(f"         ✅ pred_score 정상, 범위: [{pred_score_tensor.min():.6f}, {pred_score_tensor.max():.6f}]")
                
                if hasattr(model_output, 'anomaly_map'):
                    anomaly_map_tensor = model_output.anomaly_map
                    print(f"         anomaly_map shape: {anomaly_map_tensor.shape}")
                    
                    if torch.isnan(anomaly_map_tensor).any():
                        print(f"         ❌ anomaly_map에 NaN 발견! 개수: {torch.isnan(anomaly_map_tensor).sum().item()}")
                    else:
                        print(f"         ✅ anomaly_map 정상, 범위: [{anomaly_map_tensor.min():.6f}, {anomaly_map_tensor.max():.6f}]")
            
            # 모델별 출력에서 점수들 추출
            final_scores, mask_scores, severity_scores, raw_severity_scores, normalized_severity_scores = extract_scores_from_model_output(
                model_output, image_tensor.shape[0], batch_idx, model_type
            )
            
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
            all_mask_scores.extend(mask_scores.flatten() if hasattr(mask_scores, 'flatten') else mask_scores)
            all_severity_scores.extend(severity_scores.flatten() if hasattr(severity_scores, 'flatten') else severity_scores)
            all_raw_severity_scores.extend(raw_severity_scores.flatten() if hasattr(raw_severity_scores, 'flatten') else raw_severity_scores)
            all_normalized_severity_scores.extend(normalized_severity_scores.flatten() if hasattr(normalized_severity_scores, 'flatten') else normalized_severity_scores)
            
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
        "mask_scores": all_mask_scores,
        "severity_scores": all_severity_scores,
        "raw_severity_scores": all_raw_severity_scores,
        "normalized_severity_scores": all_normalized_severity_scores
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
    
    # 극단적 신뢰도 샘플 저장
    save_extreme_samples(all_image_paths, all_ground_truth, all_scores, predictions, analysis_dir)
    
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
        tuple: (anomaly_scores, mask_scores, severity_scores, raw_severity_scores, normalized_severity_scores)
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
            
            mask_scores = [0.0] * batch_size  # DRAEM에는 mask_score 없음
            severity_scores = [0.0] * batch_size  # DRAEM에는 severity_score 없음
            raw_severity_scores = [0.0] * batch_size  # DRAEM에는 raw_severity_score 없음
            normalized_severity_scores = [0.0] * batch_size  # DRAEM에는 normalized_severity_score 없음
            print(f"      📊 DRAEM 점수 추출: pred_score={final_scores[0]:.4f}")
        elif hasattr(model_output, 'anomaly_map'):
            # anomaly_map에서 점수 계산
            anomaly_map = model_output.anomaly_map.cpu().numpy()
            final_scores = [float(np.max(am)) if am.size > 0 else 0.0 for am in anomaly_map]
            mask_scores = [0.0] * batch_size
            severity_scores = [0.0] * batch_size
            raw_severity_scores = [0.0] * batch_size
            normalized_severity_scores = [0.0] * batch_size
            print(f"      📊 DRAEM 점수 추출 (anomaly_map): max={final_scores[0]:.4f}")
        else:
            raise AttributeError("DRAEM 출력 속성 없음")
            
    elif model_type == "patchcore":
        # PatchCore: pred_score만 있음
        if hasattr(model_output, 'pred_score'):
            final_scores = model_output.pred_score.cpu().numpy()
            mask_scores = [0.0] * batch_size
            severity_scores = [0.0] * batch_size
            raw_severity_scores = [0.0] * batch_size
            normalized_severity_scores = [0.0] * batch_size
            print(f"      📊 PatchCore 점수 추출: pred_score={final_scores[0]:.4f}")
        elif hasattr(model_output, 'anomaly_map'):
            # anomaly_map에서 점수 계산
            anomaly_map = model_output.anomaly_map.cpu().numpy()
            final_scores = [float(np.max(am)) if am.size > 0 else 0.0 for am in anomaly_map]
            mask_scores = [0.0] * batch_size
            severity_scores = [0.0] * batch_size
            raw_severity_scores = [0.0] * batch_size
            normalized_severity_scores = [0.0] * batch_size
            print(f"      📊 PatchCore 점수 추출 (anomaly_map): max={final_scores[0]:.4f}")
        else:
            raise AttributeError("PatchCore 출력 속성 없음")
            
    elif model_type == "dinomaly":
        # Dinomaly: pred_score 또는 anomaly_map
        if hasattr(model_output, 'pred_score'):
            final_scores = model_output.pred_score.cpu().numpy()
            mask_scores = [0.0] * batch_size
            severity_scores = [0.0] * batch_size
            raw_severity_scores = [0.0] * batch_size
            normalized_severity_scores = [0.0] * batch_size
            print(f"      📊 Dinomaly 점수 추출: pred_score={final_scores[0]:.4f}")
        elif hasattr(model_output, 'anomaly_map'):
            # anomaly_map에서 점수 계산
            anomaly_map = model_output.anomaly_map.cpu().numpy()
            final_scores = [float(np.max(am)) if am.size > 0 else 0.0 for am in anomaly_map]
            mask_scores = [0.0] * batch_size
            severity_scores = [0.0] * batch_size
            raw_severity_scores = [0.0] * batch_size
            normalized_severity_scores = [0.0] * batch_size
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
            
        mask_scores = [0.0] * batch_size
        severity_scores = [0.0] * batch_size
        raw_severity_scores = [0.0] * batch_size
        normalized_severity_scores = [0.0] * batch_size
        print(f"      📊 일반 모델 점수 추출: anomaly_score={final_scores[0]:.4f}")
        
    return final_scores, mask_scores, severity_scores, raw_severity_scores, normalized_severity_scores
