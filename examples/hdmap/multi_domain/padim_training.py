#!/usr/bin/env python3
"""MultiDomain HDMAP PaDiM 도메인 전이 학습 예시.

MultiDomainHDMAPDataModule과 PaDiM 모델을 활용한 효율적인 도메인 전이 학습 실험 스크립트입니다.
PaDiM의 독특한 특성을 고려하여 Memory Bank 기반의 anomaly detection을 수행합니다.

PaDiM 모델 특징:
1. Train: 정상 이미지만으로 Memory Bank (평균/공분산) 구축, Loss 기반 학습 없음
2. Validation: 선택적 - threshold 설정용으로만 사용 (나중에 구현 예정)
3. Test: Memory Bank와의 Mahalanobis 거리로 anomaly score 계산

실험 구조:
1. MultiDomainHDMAPDataModule 설정 (source: domain_A, targets: domain_B,C,D)
2. Source Domain에서 PaDiM Memory Bank 구축 (정상 데이터만)
3. Source Domain에서 성능 평가 (test 데이터로 베이스라인 설정)
4. Target Domains에서 동시 성능 평가 (cross-domain anomaly detection)
5. 도메인 전이 효과 종합 분석

"""

import os
import torch
import gc
import logging
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# gt_mask 경고 메시지 비활성화
logging.getLogger("anomalib.visualization.image.item_visualizer").setLevel(logging.ERROR)

# MultiDomain HDMAP import
from anomalib.data.datamodules.image.multi_domain_hdmap import MultiDomainHDMAPDataModule
from anomalib.models import Padim
from anomalib.engine import Engine
from anomalib.loggers import AnomalibTensorBoardLogger


# GPU 설정 - 사용할 GPU 번호를 수정하세요
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def cleanup_gpu_memory():
    """GPU 메모리 정리 및 상태 출력."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU 메모리 예약량: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        print("GPU를 사용할 수 없습니다. CPU로 실행됩니다.")


def create_multi_domain_hdmap_datamodule(
    source_domain: str = "domain_A", 
    target_domains: str = "auto",
    batch_size: int = 32
) -> MultiDomainHDMAPDataModule:
    """MultiDomainHDMAPDataModule 생성.
    
    Args:
        source_domain: 훈련용 소스 도메인
        target_domains: 타겟 도메인들 ("auto"로 자동 선택)
        batch_size: 배치 크기
        
    Returns:
        MultiDomainHDMAPDataModule: 설정된 DataModule
    """
    print(f"\n🔧 MultiDomainHDMAPDataModule 설정")
    print(f"   📁 데이터 경로: datasets/HDMAP/1000_8bit_resize_256x256")
    print(f"   🎯 Source Domain: {source_domain}")
    print(f"   🎯 Target Domains: {target_domains}")
    print(f"   📦 Batch Size: {batch_size}")
    
    datamodule = MultiDomainHDMAPDataModule(
        root="datasets/HDMAP/1000_8bit_resize_256x256",
        source_domain=source_domain,
        target_domains=target_domains,
        validation_strategy="source_test",  # 소스 도메인 test를 validation으로 사용
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=16,
    )
    
    # 데이터 준비 및 설정
    datamodule.prepare_data()
    datamodule.setup()
    
    # DataModule 상태 출력
    print(f"   ✅ 훈련 데이터: {len(datamodule.train_data)} 샘플")
    print(f"   ✅ 검증 데이터: {len(datamodule.val_data)} 샘플")
    print(f"   ✅ 테스트 데이터: {len(datamodule.test_data)} 도메인")
    
    return datamodule


def train_padim_model_multi_domain(
    datamodule: MultiDomainHDMAPDataModule,
    max_epochs: int = 1,
    experiment_name: str = "multi_domain_padim"
) -> tuple[Padim, Engine]:
    """PaDiM 모델 훈련 (Memory Bank 구축).
    
    PaDiM 특징:
    - 정상 이미지만으로 Memory Bank (평균/공분산) 구축
    - 실제 Loss 기반 파라미터 업데이트 없음
    - 1 epoch만으로 충분 (통계값 계산)
    
    Args:
        datamodule: MultiDomainHDMAPDataModule
        max_epochs: 최대 에포크 수 (PaDiM은 1로 충분)
        experiment_name: 실험 이름
        
    Returns:
        tuple: (훈련된 PaDiM 모델, Engine)
    """
    print(f"\n🚀 PaDiM 모델 Memory Bank 구축 시작")
    print(f"   🎯 Source Domain: {datamodule.source_domain}")
    print(f"   📊 훈련 방식: Memory Bank 구축 (Loss 기반 학습 없음)")
    print(f"   ⏱️  Epochs: {max_epochs} (통계값 계산용)")
    
    # PaDiM 모델 생성
    model = Padim()
    
    # TensorBoard Logger 설정
    logger = AnomalibTensorBoardLogger(
        name="hdmap_experiments",
        version=experiment_name,
        save_dir="results/tensorboard"
    )
    
    # Engine 설정 (PaDiM에 최적화)
    engine = Engine(
        accelerator="gpu",
        devices=[0],
        logger=logger,
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,  # 매 에포크마다 validation
        enable_checkpointing=True,
        log_every_n_steps=10,
        enable_model_summary=True,
        num_sanity_val_steps=0,  # PaDiM 특성상 0으로 설정
        callbacks=[],  # ImageVisualizer 등 기본 callbacks 비활성화 (gt_mask 경고 방지)
    )
    
    print(f"   🔧 Engine 설정 완료")
    print(f"   📏 Metrics: AUROC (anomaly score 기반)")
    
    # 훈련 시작 (실제로는 Memory Bank 구축)
    print(f"\n⚡ Memory Bank 구축 시작...")
    engine.fit(model=model, datamodule=datamodule)
    
    print(f"✅ PaDiM Memory Bank 구축 완료!")
    print(f"   💾 Memory Bank: Source domain ({datamodule.source_domain}) 정상 분포 저장")
    
    return model, engine


def evaluate_source_domain(
    model: Padim,
    engine: Engine,
    datamodule: MultiDomainHDMAPDataModule,
    checkpoint_path: str = None
) -> Dict[str, Any]:
    """Source Domain 성능 평가.
    
    PaDiM Source 평가:
    - Source domain test set으로 베이스라인 성능 측정
    - Memory Bank와의 Mahalanobis 거리로 anomaly score 계산
    - 나중에 threshold 설정용으로도 활용 가능
    
    Args:
        model: 훈련된 PaDiM 모델
        engine: Engine
        datamodule: MultiDomainHDMAPDataModule
        checkpoint_path: 체크포인트 경로 (선택적)
        
    Returns:
        Dict: 평가 결과
    """
    print(f"\n📊 Source Domain 성능 평가 - {datamodule.source_domain}")
    print("   💡 평가 데이터: Source domain test (베이스라인 성능 측정)")
    print("   🧮 PaDiM: Memory Bank와의 Mahalanobis 거리로 anomaly score 계산")
    
    # Validation DataLoader를 수동으로 가져와서 engine.test()로 평가
    val_dataloader = datamodule.val_dataloader()
    
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
    
    # 결과가 리스트인 경우 첫 번째 요소 추출
    if isinstance(results, list) and len(results) > 0:
        result = results[0]
    else:
        result = results
    
    print(f"   📈 Source Domain 결과:")
    if isinstance(result, dict) and result:
        # PaDiM 메트릭 출력 (주로 AUROC)
        for key, value in result.items():
            if isinstance(value, (int, float)):
                print(f"      {key}: {value:.3f}")
    else:
        print("      ⚠️  평가 결과가 비어있습니다.")
    
    return result


def evaluate_target_domains(
    model: Padim,
    engine: Engine,
    datamodule: MultiDomainHDMAPDataModule,
    checkpoint_path: str = None,
    save_samples: bool = True
) -> Dict[str, Dict[str, Any]]:
    """Target Domains 성능 평가 및 결과 복사.
    
    PaDiM Target 평가:
    - Source Memory Bank를 사용해 Target domain anomaly detection
    - Cross-domain transfer 성능 측정
    - 각 도메인별 Mahalanobis 거리 기반 성능
    - 각 Target Domain별 전체 시각화 결과 복사 (선택적)
    
    Args:
        model: 훈련된 PaDiM 모델
        engine: Engine  
        datamodule: MultiDomainHDMAPDataModule
        checkpoint_path: 체크포인트 경로 (선택적)
        save_samples: Target Domain 전체 결과 복사 여부
        
    Returns:
        Dict: 각 타겟 도메인별 평가 결과
    """
    print(f"\n🎯 Target Domains 성능 평가")
    print(f"   📊 Source Memory Bank로 Target domain anomaly detection")
    print(f"   🔄 Cross-domain transfer 성능 측정")
    
    target_results = {}
    test_dataloaders = datamodule.test_dataloader()
    
    # 각 타겟 도메인 평가
    for i, (domain, dataloader) in enumerate(zip(datamodule.target_domains, test_dataloaders)):
        print(f"\n   🔍 평가 중: {domain}")
        
        if checkpoint_path:
            results = engine.test(
                model=model,
                dataloaders=dataloader,
                ckpt_path=checkpoint_path
            )
        else:
            results = engine.test(
                model=model,
                dataloaders=dataloader
            )
        
        # 결과가 리스트인 경우 첫 번째 요소 추출
        if isinstance(results, list) and len(results) > 0:
            result = results[0]
        else:
            result = results
            
        target_results[domain] = result
        
        # 각 도메인별 결과 출력
        print(f"      📈 {domain} 결과:")
        if isinstance(result, dict) and result:
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    print(f"         {key}: {value:.3f}")
        else:
            print(f"         ⚠️  결과가 비어있습니다.")
        
        # Target Domain 평가 결과 전체 복사 (평가 직후)
        if save_samples:
            copy_target_domain_results(domain=domain)
    
    return target_results


def copy_target_domain_results(
    domain: str,
    results_base_dir: str = "results/Padim/MultiDomainHDMAPDataModule"
) -> bool:
    """Target Domain 평가 결과 전체 복사 및 보존.
    
    각 Target Domain 평가가 완료되면 images/ 폴더의 모든 결과를 
    custom_visualize/target_domains/{domain}/ 폴더로 완전히 복사하여 보존합니다.
    
    목적: engine.test()로 생성된 시각화 결과를 도메인별로 재배치하여 
          나중에 분석할 때 용이하게 접근할 수 있도록 함
    
    Args:
        domain: 타겟 도메인 이름
        results_base_dir: 기본 결과 디렉토리 경로
        
    Returns:
        bool: 성공 여부
    """
    try:
        
        # 경로 설정
        base_path = Path(results_base_dir)
        if (base_path / "latest").exists():
            latest_version_path = base_path / "latest"
        else:
            version_dirs = [d for d in base_path.glob("v*") if d.is_dir()]
            if version_dirs:
                latest_version_path = max(version_dirs, key=lambda x: int(x.name[1:]))
            else:
                print(f"         ❌ 결과 폴더를 찾을 수 없습니다.")
                return False
        
        # 소스 경로 (현재 images/ 폴더 - 방금 평가한 domain의 결과)
        images_path = latest_version_path / "images"
        fault_path = images_path / "fault"
        good_path = images_path / "good"
        
        # 타겟 경로 (custom_visualize/target_domains/{domain}/)
        custom_viz_path = latest_version_path / "custom_visualize"
        target_domain_path = custom_viz_path / "target_domains" / domain
        target_fault_path = target_domain_path / "fault"
        target_good_path = target_domain_path / "good"
        
        # 타겟 폴더 생성
        target_fault_path.mkdir(parents=True, exist_ok=True)
        target_good_path.mkdir(parents=True, exist_ok=True)
        
        if not fault_path.exists() or not good_path.exists():
            print(f"         ⚠️  images/fault 또는 images/good 폴더가 존재하지 않습니다.")
            return False
        
        # 모든 파일 복사 (전체 결과 보존)
        fault_files = list(fault_path.glob("*.png"))
        good_files = list(good_path.glob("*.png"))
        
        # fault 폴더 전체 복사
        for src_file in fault_files:
            dest_file = target_fault_path / src_file.name
            shutil.copy2(src_file, dest_file)
        
        # good 폴더 전체 복사
        for src_file in good_files:
            dest_file = target_good_path / src_file.name
            shutil.copy2(src_file, dest_file)
    
        return True
        
    except Exception as e:
        print(f"         ❌ 샘플 저장 중 오류: {e}")
        return False


def create_custom_visualizations(
    experiment_name: str = "multi_domain_padim",
    results_base_dir: str = "results/Padim/MultiDomainHDMAPDataModule",
    source_domain: str = "domain_A",
    target_domains: list = None,
    source_results: Dict[str, Any] = None,
    target_results: Dict[str, Dict[str, Any]] = None
) -> str:
    """PaDiM Multi-Domain 실험을 위한 Custom Visualization 생성.
    
    1단계 구현: 기본 폴더 구조 생성 및 실험 정보 저장
    
    Args:
        experiment_name: 실험 이름
        results_base_dir: 기본 결과 디렉토리 경로
        source_domain: 소스 도메인 이름
        target_domains: 타겟 도메인 리스트
        source_results: 소스 도메인 평가 결과
        target_results: 타겟 도메인들 평가 결과
        
    Returns:
        str: 생성된 custom_visualize 디렉토리 경로
    """
    print(f"\n🎨 Custom Visualization 생성")
    
    # 최신 버전 폴더 찾기 (latest 심볼릭 링크 또는 최신 v* 폴더)
    base_path = Path(results_base_dir)
    if (base_path / "latest").exists() and (base_path / "latest").is_symlink():
        latest_version_path = base_path / "latest"
    else:
        # v*로 시작하는 폴더들 중 가장 최신 버전 찾기
        version_dirs = [d for d in base_path.glob("v*") if d.is_dir()]
        if version_dirs:
            latest_version_path = max(version_dirs, key=lambda x: int(x.name[1:]))
        else:
            raise FileNotFoundError(f"결과 폴더를 찾을 수 없습니다: {base_path}")
    
    print(f"   📁 결과 경로: {latest_version_path}")
    
    # Custom visualize 폴더 생성
    custom_viz_path = latest_version_path / "custom_visualize"
    custom_viz_path.mkdir(exist_ok=True)
    
    # 하위 폴더 구조 생성
    folders_to_create = [
        "source_domain",
        "target_domains"
    ]
    
    for folder in folders_to_create:
        (custom_viz_path / folder).mkdir(exist_ok=True)
    
    # 타겟 도메인별 하위 폴더 생성
    if target_domains:
        for domain in target_domains:
            (custom_viz_path / "target_domains" / domain).mkdir(exist_ok=True)
    
    # 실험 정보를 JSON으로 저장
    experiment_info = {
        "experiment_name": experiment_name,
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
    info_file = custom_viz_path / "experiment_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 폴더 구조 생성 완료: {custom_viz_path}")
    
    return str(custom_viz_path)


def organize_source_domain_results(
    custom_viz_path: str,
    results_base_dir: str = "results/Padim/MultiDomainHDMAPDataModule",
    source_domain: str = "domain_A"
) -> bool:
    """Source Domain 평가 결과 재배치 및 보존.
    
    목적: engine.test()로 생성된 Source Domain 시각화 결과를 source_domain/ 폴더로 재배치하여
          나중에 분석할 때 용이하게 접근할 수 있도록 함
    
    방식: 기존 images/ 폴더에서 모든 결과를 source_domain/ 폴더로 전체 복사
    
    📊 시각화 결과 해석:
    - Image: 원본 HDMAP 이미지
    - Image + Anomaly Map: PaDiM Memory Bank와의 Mahalanobis 거리 (파란색=정상, 빨강=이상)
    - Image + Pred Mask: F1AdaptiveThreshold 기반 binary mask (빨간색 영역만 표시)
      * Threshold는 validation 데이터에서 F1-Score 최대화하는 값으로 자동 계산
      * anomaly_score > threshold인 픽셀만 빨간색으로 표시
      * 일부 이상 샘플에서 빨간색 mask가 없을 수 있음 (모든 픽셀이 threshold 미만)
    
    Args:
        custom_viz_path: custom_visualize 폴더 경로
        results_base_dir: 기본 결과 디렉토리 경로
        source_domain: 소스 도메인 이름
        
    Returns:
        bool: 성공 여부
    """
    print(f"\n📁 Source Domain 결과 재배치")
    
    # 경로 설정
    custom_viz_path = Path(custom_viz_path)
    source_viz_path = custom_viz_path / "source_domain"
    
    # 최신 결과 폴더 찾기
    base_path = Path(results_base_dir)
    if (base_path / "latest").exists() and (base_path / "latest").is_symlink():
        latest_version_path = base_path / "latest"
    else:
        version_dirs = [d for d in base_path.glob("v*") if d.is_dir()]
        if version_dirs:
            latest_version_path = max(version_dirs, key=lambda x: int(x.name[1:]))
        else:
            print("   ❌ 결과 폴더를 찾을 수 없습니다.")
            return False
    
    # 기존 images 폴더 경로
    images_path = latest_version_path / "images"
    fault_path = images_path / "fault"
    good_path = images_path / "good"
    
    if not fault_path.exists() or not good_path.exists():
        print("   ❌ images/fault 또는 images/good 폴더가 존재하지 않습니다.")
        return False
    
    # 모든 파일 리스트 가져오기
    fault_files = list(fault_path.glob("*.png"))
    good_files = list(good_path.glob("*.png"))
    
    # Source domain 폴더에 전체 복사
    fault_dest = source_viz_path / "fault"
    good_dest = source_viz_path / "good"
    fault_dest.mkdir(exist_ok=True)
    good_dest.mkdir(exist_ok=True)
    
    # 이상 샘플 전체 복사 (Image | Anomaly Map | Pred Mask 3단 구성)
    for src_file in fault_files:
        dest_file = fault_dest / src_file.name
        shutil.copy2(src_file, dest_file)
    
    # 정상 샘플 전체 복사 (Image | Anomaly Map | Pred Mask 3단 구성)
    for src_file in good_files:
        dest_file = good_dest / src_file.name
        shutil.copy2(src_file, dest_file)
        
    return True




def analyze_padim_domain_transfer_results(
    source_domain: str,
    source_results: Dict[str, Any],
    target_results: Dict[str, Dict[str, Any]]
):
    """PaDiM 도메인 전이 결과 종합 분석.
    
    PaDiM 특화 분석:
    - Memory Bank 기반 cross-domain performance
    - Anomaly score 기반 AUROC 분석 
    - Domain gap 정량화
    
    Args:
        source_domain: 소스 도메인 이름
        source_results: 소스 도메인 평가 결과
        target_results: 타겟 도메인들 평가 결과
    """
    print(f"\n" + "="*80)
    print(f"📊 PaDiM 도메인 전이 학습 결과 종합 분석")
    print(f"🧮 Memory Bank 기반 Cross-Domain Anomaly Detection")
    print(f"="*80)
    
    print(f"\n📋 실험 요약:")
    print(f"   🎯 Source Domain: {source_domain} (Memory Bank 구축)")
    print(f"   🎯 Target Domains: {list(target_results.keys())} (Cross-domain 평가)")
    print(f"   🧮 알고리즘: PaDiM (Mahalanobis Distance)")
    print(f"   📏 메트릭: AUROC (Anomaly Score 기반)")
    
    # 결과 테이블 생성
    print(f"\n📈 도메인별 성능 비교 (AUROC):")
    print(f"{'Domain':<12} {'AUROC':<12} {'Type':<10} {'Note'}")
    print(f"{'-'*50}")
    
    # Source domain 결과 (주요 메트릭 추출)
    source_auroc = None
    
    for key, value in source_results.items():
        if 'AUROC' in key:
            source_auroc = value
            break
    
    if source_auroc is not None:
        print(f"{source_domain:<12} {source_auroc:<12.3f} {'Source':<10} Memory Bank 베이스라인")
    else:
        print(f"{source_domain:<12} {'N/A':<12} {'Source':<10} 베이스라인 (결과 없음)")
    
    # Target domains 결과
    target_performances = []
    for domain, results in target_results.items():
        target_auroc = None
        
        for key, value in results.items():
            if 'AUROC' in key:
                target_auroc = value
                break
        
        if target_auroc is not None:
            print(f"{domain:<12} {target_auroc:<12.3f} {'Target':<10} Cross-domain 전이")
            target_performances.append((domain, target_auroc))
        else:
            print(f"{domain:<12} {'N/A':<12} {'Target':<10} 전이 (결과 없음)")
    
def main():
    """메인 실험 함수."""
    print("="*80)
    print("🚀 MultiDomain HDMAP PaDiM 도메인 전이 학습 실험")
    print("MultiDomainHDMAPDataModule + PaDiM 모델 전용 Memory Bank 기반 실험")
    print("="*80)
    
    # 실험 설정
    SOURCE_DOMAIN = "domain_A"  # Memory Bank 구축용 소스 도메인
    TARGET_DOMAINS = "auto"  # 자동으로 나머지 도메인들 선택
    BATCH_SIZE = 32
    MAX_EPOCHS = 1  # PaDiM은 Memory Bank 구축만 필요
    
    # GPU 메모리 정리
    cleanup_gpu_memory()
    
    try:
        # 1단계: MultiDomainHDMAPDataModule 생성
        print(f"\n{'='*60}")
        print(f"1단계: MultiDomainHDMAPDataModule 설정")
        print(f"{'='*60}")
        
        datamodule = create_multi_domain_hdmap_datamodule(
            source_domain=SOURCE_DOMAIN,
            target_domains=TARGET_DOMAINS,
            batch_size=BATCH_SIZE
        )
        
        # 2단계: PaDiM Memory Bank 구축 (Source Domain)
        print(f"\n{'='*60}")
        print(f"2단계: PaDiM Memory Bank 구축 (Source Domain)")
        print(f"{'='*60}")
        
        model, engine = train_padim_model_multi_domain(
            datamodule=datamodule,
            max_epochs=MAX_EPOCHS,
            experiment_name=f"multi_domain_padim_{SOURCE_DOMAIN}",
        )
        
        cleanup_gpu_memory()
        
        # 3단계: Source Domain 성능 평가
        print(f"\n{'='*60}")
        print(f"3단계: Source Domain 성능 평가")
        print(f"{'='*60}")
        
        source_results = evaluate_source_domain(
            model=model,
            engine=engine,
            datamodule=datamodule
        )
        
        cleanup_gpu_memory()
        
        # 4단계: Target Domains 성능 평가
        print(f"\n{'='*60}")
        print(f"4단계: Target Domains 성능 평가")
        print(f"{'='*60}")
        
        target_results = evaluate_target_domains(
            model=model,
            engine=engine,
            datamodule=datamodule,
            save_samples=True  # Target Domain 전체 결과 복사 활성화
        )
        
        cleanup_gpu_memory()
        
        # 5단계: 결과 종합 분석
        print(f"\n{'='*60}")
        print(f"5단계: 결과 종합 분석")
        print(f"{'='*60}")
        
        analyze_padim_domain_transfer_results(
            source_domain=SOURCE_DOMAIN,
            source_results=source_results,
            target_results=target_results
        )
        
        # 6단계: Custom Visualization 생성
        
        custom_viz_path = create_custom_visualizations(
            experiment_name=f"multi_domain_padim_{SOURCE_DOMAIN}",
            source_domain=SOURCE_DOMAIN,
            target_domains=list(target_results.keys()),
            source_results=source_results,
            target_results=target_results
        )
        
        # 6-1단계: Source Domain 결과 재배치
        organize_source_domain_results(
            custom_viz_path=custom_viz_path,
            source_domain=SOURCE_DOMAIN
        )
        
        
        print(f"\n🎉 MultiDomain PaDiM 실험 완료!")
        print(f"   🎨 결과: {custom_viz_path}")
                
    except Exception as e:
        print(f"\n❌ 실험 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_gpu_memory()


if __name__ == "__main__":
    main()
