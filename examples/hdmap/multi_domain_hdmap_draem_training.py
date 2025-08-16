#!/usr/bin/env python3
"""MultiDomain HDMAP DRAEM 도메인 전이 학습 예시.

MultiDomainHDMAPDataModule과 DRAEM 모델을 활용한 효율적인 도메인 전이 학습 실험 스크립트입니다.
단일 DataModule로 여러 도메인을 동시에 관리하여 더 체계적이고 효율적인 실험을 수행합니다.

기존 방식 vs MultiDomain 방식:
- 기존: 각 도메인별로 개별 DataModule 생성 → 반복적인 코드, 메모리 비효율
- MultiDomain: 하나의 DataModule로 여러 도메인 통합 관리 → 깔끔한 코드, 효율적 관리

실험 구조:
1. MultiDomainHDMAPDataModule 설정 (source: domain_A, targets: domain_B,C,D)
2. Source Domain에서 DRAEM 모델 훈련 (train 데이터)
3. Source Domain에서 성능 평가 (validation으로 사용된 test 데이터)
4. Target Domains에서 동시 성능 평가 (각 도메인별 test 데이터)
5. 도메인 전이 효과 종합 분석

주요 개선점:
- 단일 DataModule로 모든 도메인 관리
- 자동 타겟 도메인 설정 (target_domains="auto")
- 효율적인 메모리 사용
- 일관된 실험 설정
- DRAEM 모델에 최적화된 기본 Evaluator 사용
"""

import os
import torch
import gc
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import logging

# MultiDomain HDMAP import
from anomalib.data.datamodules.image.multi_domain_hdmap import MultiDomainHDMAPDataModule
from anomalib.models import Draem
from anomalib.engine import Engine
from anomalib.loggers import AnomalibTensorBoardLogger

# gt_mask 경고 메시지 비활성화
logging.getLogger("anomalib.visualization.image.item_visualizer").setLevel(logging.ERROR)

# GPU 설정 - 사용할 GPU 번호를 수정하세요
os.environ["CUDA_VISIBLE_DEVICES"] = "11"


def cleanup_gpu_memory():
    """GPU 메모리 정리 및 상태 출력."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


def create_custom_visualizations(
    experiment_name: str = "multi_domain_draem",
    results_base_dir: str = "results/Draem/MultiDomainHDMAPDataModule",
    source_domain: str = "domain_A",
    target_domains: list = None,
    source_results: Dict[str, Any] = None,
    target_results: Dict[str, Dict[str, Any]] = None
) -> str:
    """Custom Visualization 폴더 구조 생성 및 실험 정보 저장.
    
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
        version_dirs = [d for d in base_path.glob("v*") if d.is_dir()]
        if version_dirs:
            latest_version_path = max(version_dirs, key=lambda x: int(x.name[1:]))
        else:
            print(f"   ❌ 결과 폴더를 찾을 수 없습니다: {base_path}")
            return ""
    
    # Custom visualize 폴더 생성
    custom_viz_path = latest_version_path / "custom_visualize"
    custom_viz_path.mkdir(exist_ok=True)
    
    # 실제 사용할 폴더만 생성
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
    results_base_dir: str = "results/Draem/MultiDomainHDMAPDataModule",
    source_domain: str = "domain_A"
) -> bool:
    """Source Domain 평가 결과 재배치 및 보존.
    
    목적: engine.test()로 생성된 Source Domain 시각화 결과를 source_domain/ 폴더로 재배치하여
          나중에 분석할 때 용이하게 접근할 수 있도록 함
    
    방식: 기존 images/ 폴더에서 모든 결과를 source_domain/ 폴더로 전체 복사
    
    📊 DRAEM 시각화 결과 해석:
    - Image: 원본 HDMAP 이미지
    - Image + Anomaly Map: DRAEM의 reconstruction error 기반 anomaly map
    - Image + Pred Mask: Threshold 기반 binary mask (빨간색 영역만 표시)
      * DRAEM은 reconstruction loss와 discriminator loss 기반으로 anomaly score 계산
      * threshold는 validation 데이터에서 최적화된 값 사용
    
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
    source_viz_path.mkdir(exist_ok=True)
    
    # 기존 images 폴더 경로
    base_path = Path(results_base_dir)
    if (base_path / "latest").exists():
        latest_version_path = base_path / "latest"
    else:
        version_dirs = [d for d in base_path.glob("v*") if d.is_dir()]
        latest_version_path = max(version_dirs, key=lambda x: int(x.name[1:]))
    
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


def copy_target_domain_results(
    domain: str,
    results_base_dir: str = "results/Draem/MultiDomainHDMAPDataModule"
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


def create_multi_domain_datamodule(
    source_domain: str = "domain_A",
    target_domains: str | List[str] = "auto",
    batch_size: int = 16,
    image_size: str = "224x224"
) -> MultiDomainHDMAPDataModule:
    """MultiDomain HDMAP DataModule 생성.
    
    Args:
        source_domain: 훈련용 소스 도메인 (예: "domain_A")
        target_domains: 타겟 도메인들 ("auto" 또는 명시적 리스트)
        batch_size: 배치 크기
        image_size: 이미지 크기 ("224x224" 또는 "256x256")
        
    Returns:
        MultiDomainHDMAPDataModule: 설정된 멀티 도메인 데이터 모듈
    
    Note:
        MultiDomainHDMAPDataModule의 주요 특징:
        - Source domain train 데이터로 모델 훈련
        - Source domain test 데이터로 validation (balanced data)
        - Target domains test 데이터로 도메인 전이 평가
        - target_domains="auto"는 source를 제외한 모든 도메인 자동 선택
    """
    print(f"\n📦 MultiDomainHDMAPDataModule 생성 중...")
    print(f"   Source Domain: {source_domain}")
    print(f"   Target Domains: {target_domains}")
    
    datamodule = MultiDomainHDMAPDataModule(
        root=f"./datasets/HDMAP/1000_8bit_resize_{image_size}",
        source_domain=source_domain,
        target_domains=target_domains,  # "auto" 또는 ["domain_B", "domain_C"]
        validation_strategy="source_test",  # 소스 도메인 test를 validation으로 사용
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=4,  # 시스템에 맞게 조정
    )
    
    # 데이터 준비 및 설정
    datamodule.prepare_data()
    datamodule.setup()
    
    print(f"✅ MultiDomainHDMAPDataModule 설정 완료")
    print(f"   실제 Target Domains: {datamodule.target_domains}")
    print(f"   훈련 데이터: {len(datamodule.train_data)} 샘플 (source: {datamodule.source_domain})")
    print(f"   검증 데이터: {len(datamodule.val_data)} 샘플 (source test)")
    
    total_test_samples = sum(len(test_data) for test_data in datamodule.test_data)
    print(f"   테스트 데이터: {total_test_samples} 샘플 (targets)")
    
    for i, target_domain in enumerate(datamodule.target_domains):
        print(f"     └─ {target_domain}: {len(datamodule.test_data[i])} 샘플")
    
    return datamodule


def train_draem_model_multi_domain(
    datamodule: MultiDomainHDMAPDataModule, 
    experiment_name: str,
    max_epochs: int = 20,
    optimizer_name: str = "adam",
    learning_rate: float = 1e-4,
) -> tuple[Draem, Engine]:
    """MultiDomain DataModule을 사용한 DRAEM 모델 훈련.
    
    Args:
        datamodule: 멀티 도메인 데이터 모듈
        experiment_name: 실험 이름 (로그용)
        max_epochs: 최대 에포크 수 (기본값: 20)
        optimizer_name: 옵티마이저 종류 ("adam", "adamw", "sgd") (기본값: "adam")
        learning_rate: 학습률 (기본값: 1e-4)
        
    Returns:
        tuple: (훈련된 모델, Engine 객체)
        
    Note:
        훈련 과정:
        1. Source domain train 데이터로 모델 훈련
        2. Source domain test 데이터로 validation (정상+이상 데이터 포함)
        3. 각 에포크마다 validation 성능으로 모델 개선 추적
        
        DRAEM 특징:
        - 기본 Evaluator 사용 (모델에 최적화된 메트릭)
        - Reconstructive + Discriminative 서브네트워크
        - Anomaly map 기반 픽셀 레벨 예측
    """
    print(f"\n🤖 DRAEM 모델 훈련 시작 - {experiment_name}")
    print(f"   Source Domain: {datamodule.source_domain}")
    print(f"   Validation Strategy: {datamodule.validation_strategy}")
    print(f"   Max Epochs: {max_epochs}")
    print(f"   Optimizer: {optimizer_name}")
    print(f"   Learning Rate: {learning_rate}")
    
    # DRAEM 모델 생성 (기본 Evaluator 사용)
    # DRAEM은 자체적으로 최적화된 evaluator를 가지고 있음
    model = Draem()
    
    # 🔧 Optimizer 설정 (configure_optimizers 오버라이드)
    def configure_optimizers_custom():
        """Custom optimizer configuration."""
        if optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(model.model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(model.model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(model.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 600], gamma=0.1)
        return [optimizer], [scheduler]
    
    # Optimizer 설정을 모델에 바인딩
    model.configure_optimizers = configure_optimizers_custom
    
    # TensorBoard 로거 설정
    logger = AnomalibTensorBoardLogger(
        save_dir="logs/hdmap_multi_domain",
        name=experiment_name
    )
    
    # Engine 설정
    engine = Engine(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[0] if torch.cuda.is_available() else 1,
        logger=logger,
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,  # 매 에포크마다 validation
        enable_checkpointing=True,
        log_every_n_steps=10,
        enable_model_summary=True,
        num_sanity_val_steps=0,  # DRAEM 특성상 0으로 설정
    )
    
    # 모델 훈련
    print("🔥 훈련 시작...")
    engine.fit(model=model, datamodule=datamodule)
    
    print("✅ 모델 훈련 완료!")
    print(f"   체크포인트: {engine.trainer.checkpoint_callback.best_model_path}")
    
    return model, engine


def evaluate_source_domain(
    model: Draem, 
    engine: Engine, 
    datamodule: MultiDomainHDMAPDataModule,
    checkpoint_path: str = None
) -> Dict[str, Any]:
    """Source Domain 성능 평가.
    
    Args:
        model: 평가할 모델
        engine: Engine 객체
        datamodule: 멀티 도메인 데이터 모듈
        checkpoint_path: 체크포인트 경로
        
    Returns:
        Dict: Source domain 평가 결과
        
    Note:
        Source domain 평가는 validation 데이터(source test)로 수행됩니다.
        val_split_mode=NONE 때문에 engine.validate()가 작동하지 않아서
        engine.test()를 사용하여 validation DataLoader로 평가합니다.
        
        DRAEM 평가 특징:
        - Image-level과 Pixel-level 메트릭 모두 제공
        - AUROC, F1-Score 등 다양한 메트릭
    """
    print(f"\n📊 Source Domain 성능 평가 - {datamodule.source_domain}")
    print("   💡 평가 데이터: Source domain test (validation으로 사용된 데이터)")
    
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
    
    print(f"✅ {datamodule.source_domain} 평가 완료")
    
    # 결과가 리스트인 경우 첫 번째 요소, 아니면 그대로 반환
    if isinstance(results, list) and len(results) > 0:
        result = results[0]
    elif isinstance(results, dict):
        result = results
    else:
        result = {}
    
    print(f"   📊 Source Domain 성능:")
    if isinstance(result, dict) and result:
        # DRAEM 메트릭 출력 (image-level과 pixel-level 모두)
        for key, value in result.items():
            if isinstance(value, (int, float)):
                print(f"      {key}: {value:.3f}")
    else:
        print("      ⚠️  평가 결과가 비어있습니다.")
    
    return result


def evaluate_target_domains(
    model: Draem, 
    engine: Engine, 
    datamodule: MultiDomainHDMAPDataModule,
    checkpoint_path: str = None,
    save_samples: bool = True
) -> Dict[str, Dict[str, Any]]:
    """Target Domains 성능 평가 및 결과 복사.
    
    Args:
        model: 평가할 모델
        engine: Engine 객체
        datamodule: 멀티 도메인 데이터 모듈
        checkpoint_path: 체크포인트 경로
        save_samples: Target Domain 전체 결과 복사 여부
        
    Returns:
        Dict: 각 target domain별 평가 결과
        
    Note:
        MultiDomainHDMAPDataModule의 test_dataloader()는 모든 target domain의 
        DataLoader 리스트를 반환합니다. 각 도메인별로 개별 평가를 수행합니다.
    """
    print(f"\n🎯 Target Domains 성능 평가")
    print(f"   Target Domains: {datamodule.target_domains}")
    print("   💡 각 도메인별 개별 평가 수행")
    
    target_results = {}
    test_dataloaders = datamodule.test_dataloader()
    
    for i, target_domain in enumerate(datamodule.target_domains):
        print(f"\n📋 {target_domain} 평가 중...")
        
        # 개별 target domain DataLoader로 평가
        target_dataloader = test_dataloaders[i]
        
        # 임시 single-domain 평가를 위한 설정
        if checkpoint_path:
            results = engine.test(
                model=model,
                dataloaders=target_dataloader,
                ckpt_path=checkpoint_path
            )
        else:
            results = engine.test(
                model=model,
                dataloaders=target_dataloader
            )
        
        target_results[target_domain] = results[0] if results else {}
        print(f"✅ {target_domain} 평가 완료")
        
        # Target Domain 평가 결과 전체 복사 (평가 직후)
        if save_samples:
            copy_target_domain_results(domain=target_domain)
    
    return target_results


def run_single_experiment(
    multi_datamodule: MultiDomainHDMAPDataModule,
    condition: dict,
    source_domain: str,
    max_epochs: int,
) -> dict:
    """단일 실험 조건에 대한 실험 수행.
    
    Args:
        multi_datamodule: 멀티 도메인 데이터 모듈
        condition: 실험 조건 딕셔너리
        source_domain: 소스 도메인
        max_epochs: 최대 에포크 수
        
    Returns:
        dict: 실험 결과 딕셔너리
    """
    experiment_name = f"multi_domain_draem_{source_domain}_{condition['name']}"
    
    print(f"\n{'='*80}")
    print(f"🔬 실험 조건: {condition['name']}")
    print(f"📝 설명: {condition['description']}")
    print(f"{'='*80}")
    
    try:
        # 모델 훈련
        trained_model, engine = train_draem_model_multi_domain(
            datamodule=multi_datamodule,
            experiment_name=experiment_name,
            max_epochs=max_epochs,
            optimizer_name=condition["optimizer"],
            learning_rate=condition["learning_rate"]
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
        
        # Target Domains 성능 평가
        print(f"\n🎯 Target Domains 성능 평가 - {condition['name']}")
        target_results = evaluate_target_domains(
            model=trained_model,
            engine=engine,
            datamodule=multi_datamodule,
            checkpoint_path=best_checkpoint,
            save_samples=False  # 다중 실험에서는 샘플 저장 비활성화
        )
        
        # 실험 결과 정리
        experiment_result = {
            "condition": condition,
            "experiment_name": experiment_name,
            "source_results": source_results,
            "target_results": target_results,
            "best_checkpoint": best_checkpoint,
            "status": "success"
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
            "error": str(e)
        }
    finally:
        # 메모리 정리
        cleanup_gpu_memory()


def analyze_multi_experiment_results(all_results: list, source_domain: str):
    """다중 실험 결과 분석 및 비교."""
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
        
        for i, result in enumerate(sorted_results, 1):
            condition = result["condition"]
            source_auroc = result["source_results"].get("image_AUROC", 0)
            target_auroc = result.get("avg_target_auroc", 0)
            
            print(f"   {i}. {condition['name']} ({condition['optimizer']})")
            print(f"      Source AUROC: {source_auroc:.4f}")
            print(f"      Target Avg AUROC: {target_auroc:.4f}")
            print(f"      Description: {condition['description']}")
            print()
        
        # 최고 성능 실험 하이라이트
        best_result = sorted_results[0]
        print(f"🥇 최고 성능 실험: {best_result['condition']['name']}")
        print(f"   Target Avg AUROC: {best_result.get('avg_target_auroc', 0):.4f}")
        print(f"   Checkpoint: {best_result['best_checkpoint']}")
        
        # Optimizer별 비교
        print(f"\n🚀 Optimizer별 평균 성능:")
        optimizer_groups = {}
        for result in successful_results:
            opt = result["condition"]["optimizer"]
            if opt not in optimizer_groups:
                optimizer_groups[opt] = []
            optimizer_groups[opt].append(result.get("avg_target_auroc", 0))
        
        for opt, aurocs in optimizer_groups.items():
            avg_auroc = sum(aurocs) / len(aurocs)
            print(f"   {opt.upper()}: {avg_auroc:.4f} (평균, {len(aurocs)}개 실험)")


def analyze_domain_transfer_results(
    source_domain: str,
    source_results: Dict[str, Any],
    target_results: Dict[str, Dict[str, Any]]
):
    """도메인 전이 학습 결과 분석 및 출력.
    
    Args:
        source_domain: 소스 도메인 이름
        source_results: 소스 도메인 평가 결과
        target_results: 타겟 도메인별 평가 결과
    """
    print(f"\n{'='*80}")
    print(f"📈 도메인 전이 학습 결과 종합 분석")
    print(f"{'='*80}")
    
    # 성능 요약 테이블
    print(f"\n📊 성능 요약:")
    print(f"{'도메인':<12} {'Image AUROC':<12} {'Pixel AUROC':<12} {'유형':<10} {'설명'}")
    print("-" * 70)
    
    # Source domain 결과 (주요 메트릭 추출)
    source_image_auroc = None
    source_pixel_auroc = None
    
    for key, value in source_results.items():
        if 'image_AUROC' == key:
            source_image_auroc = value
        elif 'pixel_AUROC' == key:
            source_pixel_auroc = value
    
    if source_image_auroc is not None:
        print(f"{source_domain:<12} {source_image_auroc:<12.3f} {source_pixel_auroc or 0:<12.3f} {'Source':<10} 베이스라인")
    else:
        print(f"{source_domain:<12} {'N/A':<12} {'N/A':<12} {'Source':<10} 베이스라인 (결과 없음)")
    
    # Target domains 결과
    target_performances = []
    for domain, results in target_results.items():
        target_image_auroc = None
        target_pixel_auroc = None
        
        for key, value in results.items():
            if 'image_AUROC' == key:
                target_image_auroc = value
            elif 'pixel_AUROC' == key:
                target_pixel_auroc = value
        
        if target_image_auroc is not None:
            print(f"{domain:<12} {target_image_auroc:<12.3f} {target_pixel_auroc or 0:<12.3f} {'Target':<10} 도메인 전이")
            target_performances.append((domain, target_image_auroc, target_pixel_auroc))


def main():
    """멀티 도메인 DRAEM 다중 실험 메인 함수."""
    print("="*80)
    print("🚀 MultiDomain HDMAP DRAEM 다중 실험")
    print("Optimizer 조합별 성능 비교 실험")
    print("="*80)
    
    # 실험 설정
    SOURCE_DOMAIN = "domain_A"  # 훈련용 소스 도메인
    TARGET_DOMAINS = "auto"  # 자동으로 나머지 도메인들 선택
    BATCH_SIZE = 16
    MAX_EPOCHS = 10  # Custom DRAEM과 동일한 10 epochs
    
    # 🧪 실험 조건 설정 - Custom DRAEM 결과 기반 최적화된 조건
    EXPERIMENT_CONDITIONS = [
        # 🥇 Baseline: AdamW 기본 조건 (Custom DRAEM 비교군)
        {
            "name": "draem_adamw_baseline",
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "description": "Original DRAEM baseline with AdamW"
        },
        
        # 🔬 Learning Rate 비교 실험
        {
            "name": "draem_adamw_lr_high",
            "optimizer": "adamw", 
            "learning_rate": 2e-4,
            "description": "Original DRAEM with higher learning rate"
        },
        {
            "name": "draem_adamw_lr_low",
            "optimizer": "adamw",
            "learning_rate": 5e-5,
            "description": "Original DRAEM with lower learning rate"
        },
        
        # 📊 Optimizer 비교 실험
        {
            "name": "draem_adam",
            "optimizer": "adam",
            "learning_rate": 1e-4,
            "description": "Original DRAEM with Adam optimizer"
        },
        {
            "name": "draem_sgd",
            "optimizer": "sgd",
            "learning_rate": 1e-3,  # SGD는 더 높은 학습률 필요
            "description": "Original DRAEM with SGD optimizer"
        },
        
        # 🎯 최적화된 조건 (Custom DRAEM에서 발견한 패턴 적용)
        {
            "name": "draem_adamw_optimized",
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "description": "Original DRAEM with optimized settings"
        },
    ]
    
    # GPU 메모리 정리
    cleanup_gpu_memory()
    
    try:
        # ========================================================================================
        # 1단계: MultiDomainHDMAPDataModule 설정
        # ========================================================================================
        print(f"\n{'='*60}")
        print(f"1단계: MultiDomainHDMAPDataModule 설정")
        print(f"{'='*60}")
        
        multi_datamodule = create_multi_domain_datamodule(
            source_domain=SOURCE_DOMAIN,
            target_domains=TARGET_DOMAINS,  # "auto" = 자동으로 나머지 도메인들
            batch_size=BATCH_SIZE,
            image_size="224x224"  # Custom DRAEM과 동일한 이미지 크기
        )
        
        # ======================================================================================== 
        # 2단계: 다중 실험 조건별 순차 수행
        # ========================================================================================
        print(f"\n{'='*60}")
        print(f"2단계: 다중 실험 조건별 순차 수행")
        print(f"📈 총 실험 조건: {len(EXPERIMENT_CONDITIONS)}개")
        print(f"🔬 실험 변수: Optimizer (AdamW/Adam/SGD), Learning Rate (5e-5/1e-4/2e-4/1e-3)")
        print(f"⏱️  예상 소요 시간: 약 {len(EXPERIMENT_CONDITIONS) * 25}분 (10 epochs × {len(EXPERIMENT_CONDITIONS)}개 조건, GPU 11)")
        print(f"{'='*60}")
        
        all_results = []
        
        for i, condition in enumerate(EXPERIMENT_CONDITIONS, 1):
            print(f"\n⏱️  진행상황: {i}/{len(EXPERIMENT_CONDITIONS)} - {condition['name']}")
            
            result = run_single_experiment(
                multi_datamodule=multi_datamodule,
                condition=condition,
                source_domain=SOURCE_DOMAIN,
                max_epochs=MAX_EPOCHS,
            )
            
            all_results.append(result)
            
            # 중간 결과 출력
            if result["status"] == "success":
                print(f"   📈 중간 결과: Source AUROC = {result['source_results'].get('image_AUROC', 0):.4f}, "
                      f"Target Avg AUROC = {result.get('avg_target_auroc', 0):.4f}")
            else:
                print(f"   ❌ 실험 실패: {result.get('error', '알 수 없는 오류')}")
        
        # ========================================================================================
        # 3단계: 전체 실험 결과 분석 및 비교
        # ========================================================================================
        analyze_multi_experiment_results(all_results, SOURCE_DOMAIN)
        
        # ========================================================================================
        # 4단계: 최고 성능 모델에 대한 상세 분석 (선택사항)
        # ========================================================================================
        successful_results = [r for r in all_results if r["status"] == "success"]
        if successful_results:
            # 최고 성능 모델 선택
            best_result = max(successful_results, key=lambda x: x.get("avg_target_auroc", 0))
            
            print(f"\n{'='*60}")
            print(f"4단계: 최고 성능 모델 상세 분석")
            print(f"선택된 모델: {best_result['condition']['name']}")
            print(f"{'='*60}")
            
            # 최고 성능 모델에 대해서만 상세 시각화 생성
            best_condition = best_result['condition']
            custom_viz_path = create_custom_visualizations(
                experiment_name=f"multi_domain_draem_v2024_12_{SOURCE_DOMAIN}_BEST_{best_condition['name']}",
                source_domain=SOURCE_DOMAIN,
                target_domains=list(best_result['target_results'].keys()),
                source_results=best_result['source_results'],
                target_results=best_result['target_results']
            )
            
            organize_source_domain_results(
                custom_viz_path=custom_viz_path,
                source_domain=SOURCE_DOMAIN
            )
            
            print(f"\n🎉 다중 실험 완료!")
            print(f"   🏆 최고 성능: {best_result['condition']['name']}")
            print(f"   📊 Target Avg AUROC: {best_result.get('avg_target_auroc', 0):.4f}")
            print(f"   🎨 상세 결과: {custom_viz_path}")
            print(f"   📁 최고 성능 체크포인트: {best_result['best_checkpoint']}")
            
        else:
            print(f"\n❌ 모든 실험이 실패했습니다.")
                
        # 메모리 정리
        cleanup_gpu_memory()
        
    except Exception as e:
        print(f"\n❌ 실험 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
        # 오류 발생 시에도 메모리 정리
        cleanup_gpu_memory()


if __name__ == "__main__":
    main()
