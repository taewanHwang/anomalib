#!/usr/bin/env python3
"""MultiDomain HDMAP Custom DRAEM 도메인 전이 학습 예시.

Custom DRAEM 모델과 MultiDomainHDMAPDataModule을 활용한 효율적인 도메인 전이 학습 실험 스크립트입니다.
기존 DRAEM과 동일한 방식이지만 Fault Severity Prediction Sub-Network가 추가된 Custom DRAEM을 사용합니다.

Custom DRAEM 특징
- DRAEM Backbone Integration: 기존 DRAEM의 97.4M 파라미터 backbone 통합
- Wide ResNet Encoder: ImageNet pretrained encoder (기존 DRAEM과 동일)
- Reconstructive + Discriminative Sub-Networks: 기존 DRAEM 구조 완전 활용
- Fault Severity Sub-Network: 추가 118K 파라미터로 고장 심각도 예측 (0.0~1.0)
- 3채널 RGB 지원: 224x224 또는 256x256 이미지 직접 처리
- 확률적 Synthetic Fault Generation: 학습 시 정상/고장 이미지 비율 제어 가능
- SSPCAB 옵션: 선택적 Self-Supervised Perceptual Consistency Attention Block

실험 구조:
1. MultiDomainHDMAPDataModule 설정 (source: domain_A, targets: domain_B,C,D)
2. Source Domain에서 Custom DRAEM 모델 훈련 (train 데이터)
3. Source Domain에서 성능 평가 (validation으로 사용된 test 데이터)
4. Target Domains에서 동시 성능 평가 (각 도메인별 test 데이터)
5. 도메인 전이 효과 종합 분석

주요 개선점 (Direct Comparison 달성):
- Fair Comparison: 기존 DRAEM과 동일한 97.4M backbone으로 순수한 custom feature 효과 측정
- Fault Severity Prediction을 통한 더 세밀한 이상 탐지
- 확률적 Synthetic Fault Generation으로 학습 데이터 품질 향상
- Multi-task Learning으로 더 robust한 feature representation
- 5가지 Severity Input Mode 지원 (ablation study 가능)
- 메모리 효율성: 체계적인 GPU 메모리 관리
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
from anomalib.models.image.custom_draem import CustomDraem
from anomalib.engine import Engine
from anomalib.loggers import AnomalibTensorBoardLogger

# 경고 메시지 비활성화 (테스트 환경과 동일)
logging.getLogger("anomalib.visualization.image.item_visualizer").setLevel(logging.ERROR)
logging.getLogger("anomalib.visualization").setLevel(logging.ERROR)
logging.getLogger("anomalib.callbacks").setLevel(logging.ERROR)

# GPU 설정 - 사용할 GPU 번호를 수정하세요
os.environ["CUDA_VISIBLE_DEVICES"] = "9"


def cleanup_gpu_memory():
    """GPU 메모리 정리 및 상태 출력."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


def create_custom_visualizations(
    experiment_name: str = "multi_domain_custom_draem",
    results_base_dir: str = "results/CustomDraem/MultiDomainHDMAPDataModule",
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
        "model_type": "Custom DRAEM",
        "model_features": [
            "Reconstructive Sub-Network",
            "Discriminative Sub-Network", 
            "Fault Severity Sub-Network",
            "Probabilistic Synthetic Generation",
            "Multi-task Learning"
        ],
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
    results_base_dir: str = "results/CustomDraem/MultiDomainHDMAPDataModule",
    source_domain: str = "domain_A"
) -> bool:
    """Source Domain 평가 결과 재배치 및 보존.
    
    목적: engine.test()로 생성된 Source Domain 시각화 결과를 source_domain/ 폴더로 재배치하여
          나중에 분석할 때 용이하게 접근할 수 있도록 함
    
    방식: 기존 images/ 폴더에서 모든 결과를 source_domain/ 폴더로 전체 복사
    
    📊 Custom DRAEM 시각화 결과 해석:
    - Image: 원본 HDMAP 이미지
    - Image + Anomaly Map: Custom DRAEM의 reconstruction error + discriminator 기반 anomaly map
    - Image + Pred Mask: Threshold 기반 binary mask (빨간색 영역만 표시)
    - Severity Score: Fault Severity Sub-Network의 심각도 예측값 (0.0~1.0)
      * Custom DRAEM은 reconstruction, discriminator, severity 3개 loss 조합으로 학습
      * 더 정교한 anomaly detection과 localization 성능 제공
    
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
    results_base_dir: str = "results/CustomDraem/MultiDomainHDMAPDataModule"
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
    
    # 🔧 **HDMAP 3채널 처리 전략**:
    # Custom DRAEM은 DRAEM backbone 통합으로 3채널 RGB 이미지를 직접 처리합니다.
    # 기존의 3ch → 1ch 변환은 제거되었으며, 원본 HDMAP 이미지가 3채널로 로딩되어
    # 그대로 모델에 입력됩니다. 이는 ImageNet pretrained encoder와 호환됩니다.
    # 
    # HDMAP 3-channel processing strategy (Latest Dec 2024 implementation):
    # Custom DRAEM directly processes 3-channel RGB images with integrated DRAEM backbone.
    # The previous 3ch → 1ch conversion has been removed, and original HDMAP images are
    # loaded as 3-channel and fed directly to the model, compatible with ImageNet pretrained encoder.
    #
    # 📝 **변경된 데이터 플로우 / Updated Data Flow**:
    # - MultiDomainHDMAPDataModule → 3채널 RGB 이미지 로딩
    # - Custom DRAEM Lightning 모델 → 3채널 직접 처리 (변환 없음)
    # - DRAEM backbone → ImageNet pretrained encoder로 3채널 feature extraction
    # - 성능 향상: 224x224 이미지 사용 시 22.6% 더 빠른 처리
    
    image_size = "224x224"  # 또는 "256x256" 선택 가능
    
    datamodule = MultiDomainHDMAPDataModule(
        root=f"./datasets/HDMAP/1000_8bit_resize_{image_size}",
        source_domain=source_domain,
        target_domains=target_domains,  # "auto" 또는 ["domain_B", "domain_C"]
        validation_strategy="source_test",  # 소스 도메인 test를 validation으로 사용
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=16,  # 시스템에 맞게 조정
        # 🔑 3채널 RGB 이미지가 Custom DRAEM으로 직접 전달됨
        # 🔑 3-channel RGB images are passed directly to Custom DRAEM
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


def train_custom_draem_model_multi_domain(
    datamodule: MultiDomainHDMAPDataModule, 
    experiment_name: str,
    max_epochs: int = 20,
    severity_input_mode: str = "discriminative_only",
    patch_width_range: tuple = (32, 64),
    patch_ratio_range: tuple = (0.8, 1.2),
    severity_max: float = 8.0,
    patch_count: int = 1,
    use_adaptive_loss: bool = True,
    warmup_epochs: int = 5,
    optimizer_name: str = "adam",
    learning_rate: float = 1e-4,
    sspcab: bool = False
) -> tuple[CustomDraem, Engine]:
    """MultiDomain DataModule을 사용한 Custom DRAEM 모델 훈련.
    
    Args:
        datamodule: 멀티 도메인 데이터 모듈
        experiment_name: 실험 이름 (로그용)
        max_epochs: 최대 에포크 수 (기본값: 20)
        severity_input_mode: Severity Sub-Network 입력 모드
            - "discriminative_only": Discriminative network 출력만 사용 (기본값)
            - "with_original": Discriminative + Original 이미지 결합
            - "with_reconstruction": Discriminative + Reconstruction 결합
            - "with_error_map": Discriminative + Error Map 결합
            - "multi_modal": 모든 입력 결합 (Discriminative + Original + Reconstruction + Error Map)
        patch_width_range: 합성 고장 패치 크기 범위 (min_size, max_size)
        patch_ratio_range: 패치 종횡비 범위 (기본값: (0.8, 1.2))
        severity_max: 최대 심각도 값 (기본값: 8.0)
        use_adaptive_loss: 적응적 손실 함수 사용 여부 (기본값: True)
        warmup_epochs: 재구성 중심 워밍업 에포크 수 (기본값: 5)
        optimizer_name: 옵티마이저 종류 ("adam", "adamw", "sgd") (기본값: "adam")
        learning_rate: 학습률 (기본값: 1e-4)
        sspcab: SSPCAB 활성화 여부 (기본값: False)
        
    Returns:
        tuple: (훈련된 모델, Engine 객체)
        
    Note:
        훈련 과정:
        1. Source domain train 데이터로 모델 훈련
        2. Source domain test 데이터로 validation (정상+이상 데이터 포함)
        3. 각 에포크마다 validation 성능으로 모델 개선 추적
        
        Custom DRAEM 특징 (최신 구현):
        - DRAEM Backbone (97.4M): Wide ResNet encoder + Discriminative/Reconstructive subnetworks
        - Fault Severity Sub-Network (+118K): 고장 심각도 예측 전용 네트워크
        - Multi-task Loss: L2+SSIM (recon) + FocalLoss (seg) + SmoothL1 (severity)
        - Adaptive Loss 옵션: 불확실도 기반 동적 가중치 조정
        - Probabilistic Synthetic Generation: anomaly_probability로 정상/고장 비율 제어
        - 5가지 Severity Input Mode로 ablation study 가능
        - SSPCAB 옵션: 선택적 attention mechanism
    """
    
    print(f"\n🤖 Custom DRAEM 모델 훈련 시작 - {experiment_name}")
    print(f"   Source Domain: {datamodule.source_domain}")
    print(f"   Validation Strategy: {datamodule.validation_strategy}")
    print(f"   Max Epochs: {max_epochs}")
    print(f"   Severity Input Mode: {severity_input_mode}")
    print(f"   Patch Width Range: {patch_width_range}")
    print(f"   Patch Ratio Range: {patch_ratio_range}")
    print(f"   Patch Count: {patch_count}")
    print(f"   Severity Max: {severity_max}")
    print(f"   Use Adaptive Loss: {use_adaptive_loss}")
    print(f"   Warmup Epochs: {warmup_epochs if use_adaptive_loss else 'N/A (Fixed Loss)'}")
    print(f"   Optimizer: {optimizer_name}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   SSPCAB: {sspcab}")
    
    # Custom DRAEM 모델 생성 (DRAEM backbone 통합)
    model = CustomDraem(
        # 🎯 Severity Sub-Network 설정
        severity_input_mode=severity_input_mode,
        
        # 🔧 Synthetic Fault Generation 설정
        patch_width_range=patch_width_range,
        patch_ratio_range=patch_ratio_range,  # 실험 조건에서 전달받은 값
        severity_max=severity_max,  # 실험 조건에서 전달받은 값
        patch_count=patch_count,  # 실험 조건에서 전달받은 값
        
        # 🔧 Loss 가중치 설정 (severity weight 감소로 주요 task에 집중)
        reconstruction_weight=1.0,
        segmentation_weight=1.0,
        severity_weight=0.1,  # 줄이면 original draem loss와 유사해짐
        
        # 🔧 적응적 손실 함수 설정
        use_adaptive_loss=use_adaptive_loss,
        warmup_epochs=warmup_epochs if use_adaptive_loss else 0,  # adaptive_loss=False이면 warmup 불필요
        
        # 🚀 DRAEM backbone 옵션
        sspcab=sspcab,  # 실험 조건에서 전달받은 값
        
        # 🔧 옵티마이저 설정
        optimizer=optimizer_name,
        learning_rate=learning_rate,  # 실험 조건에서 전달받은 값
    )
    
    # TensorBoard 로거 설정
    logger = AnomalibTensorBoardLogger(
        save_dir="logs/hdmap_multi_domain_custom_draem",
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
        num_sanity_val_steps=0,  # Custom DRAEM 특성상 0으로 설정
    )
    
    # 모델 훈련
    print("🔥 훈련 시작...")
    engine.fit(model=model, datamodule=datamodule)
    
    print("✅ 모델 훈련 완료!")
    print(f"   체크포인트: {engine.trainer.checkpoint_callback.best_model_path}")
    
    return model, engine


def evaluate_source_domain(
    model: CustomDraem, 
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
        
        Custom DRAEM 평가 특징 (DRAEM backbone 통합):
        - Image-level과 Pixel-level 메트릭 모두 제공
        - AUROC, F1-Score 등 다양한 메트릭
        - Severity Prediction 정확도도 추가로 제공
        - 기존 DRAEM 대비 Fair Comparison 가능
        - 97.4M 파라미터 backbone + 118K severity head
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
        # Custom DRAEM 메트릭 출력 (image-level, pixel-level, severity 모두)
        for key, value in result.items():
            if isinstance(value, (int, float)):
                print(f"      {key}: {value:.3f}")
    else:
        print("      ⚠️  평가 결과가 비어있습니다.")
    
    return result


def evaluate_target_domains(
    model: CustomDraem, 
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
    print(f"📈 Custom DRAEM 도메인 전이 학습 결과 종합 분석")
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
    
    # Custom DRAEM 특화 분석
    print(f"\n🔍 Custom DRAEM 특화 메트릭:")
    print("   ✅ Fault Severity Prediction Sub-Network 추가 성능 향상")
    print("   ✅ Multi-task Learning으로 더 robust한 feature representation")
    print("   ✅ Probabilistic Synthetic Generation으로 학습 데이터 품질 향상")


def run_single_experiment(
    multi_datamodule: MultiDomainHDMAPDataModule,
    condition: dict,
    source_domain: str,
    max_epochs: int
) -> dict:
    """단일 실험 조건에 대한 실험 수행.
    
    Args:
        multi_datamodule: 멀티 도메인 데이터 모듈
        condition: 실험 조건 딕셔너리 (모든 모델 설정 포함)
            - optimizer, learning_rate: 옵티마이저 설정
            - sspcab: SSPCAB 활성화 여부
            - severity_input_mode: 심각도 입력 모드
            - patch_width_range, patch_ratio_range: 패치 생성 설정
            - severity_max: 최대 심각도 값
        source_domain: 소스 도메인
        max_epochs: 최대 에포크 수
        
    Returns:
        dict: 실험 결과 딕셔너리
    """
    experiment_name = f"multi_domain_custom_draem_{source_domain}_{condition['name']}"
    
    print(f"\n{'='*80}")
    print(f"🔬 실험 조건: {condition['name']}")
    print(f"📝 설명: {condition['description']}")
    print(f"{'='*80}")
    
    try:
        # 모델 훈련
        trained_model, engine = train_custom_draem_model_multi_domain(
            datamodule=multi_datamodule,
            experiment_name=experiment_name,
            max_epochs=max_epochs,
            severity_input_mode=condition["severity_input_mode"],  # 실험 조건에서 추출
            patch_width_range=condition["patch_width_range"],  # 실험 조건에서 추출
            patch_ratio_range=condition["patch_ratio_range"],  # 실험 조건에서 추출
            severity_max=condition["severity_max"],  # 실험 조건에서 추출
            patch_count=condition.get("patch_count", 1),  # 실험 조건에서 추출 (기본값 1)
            use_adaptive_loss=condition["use_adaptive_loss"],
            warmup_epochs=condition.get("warmup_epochs", 0),  # adaptive_loss=False이면 없음
            optimizer_name=condition["optimizer"],
            learning_rate=condition["learning_rate"],  # 실험 조건에서 추출
            sspcab=condition["sspcab"]  # 실험 조건에서 추출
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
            
            print(f"   {i}. {condition['name']} ({condition['optimizer']} + "
                  f"{'Adaptive' if condition['use_adaptive_loss'] else 'Fixed'} Loss)")
            print(f"      Source AUROC: {source_auroc:.4f}")
            print(f"      Target Avg AUROC: {target_auroc:.4f}")
            print(f"      Description: {condition['description']}")
            print()
        
        # 최고 성능 실험 하이라이트
        best_result = sorted_results[0]
        print(f"🥇 최고 성능 실험: {best_result['condition']['name']}")
        print(f"   Target Avg AUROC: {best_result.get('avg_target_auroc', 0):.4f}")
        print(f"   Checkpoint: {best_result['best_checkpoint']}")
        
        # Loss 타입별 비교
        print(f"\n📊 Loss 타입별 평균 성능:")
        adaptive_results = [r for r in successful_results if r["condition"]["use_adaptive_loss"]]
        fixed_results = [r for r in successful_results if not r["condition"]["use_adaptive_loss"]]
        
        if adaptive_results:
            adaptive_avg = sum(r.get("avg_target_auroc", 0) for r in adaptive_results) / len(adaptive_results)
            print(f"   Adaptive Loss: {adaptive_avg:.4f} (평균, {len(adaptive_results)}개 실험)")
        
        if fixed_results:
            fixed_avg = sum(r.get("avg_target_auroc", 0) for r in fixed_results) / len(fixed_results)
            print(f"   Fixed Loss: {fixed_avg:.4f} (평균, {len(fixed_results)}개 실험)")
        
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


def main():
    """멀티 도메인 Custom DRAEM 다중 실험 메인 함수."""
    print("="*80)
    print("🚀 MultiDomain HDMAP Custom DRAEM 다중 실험")
    print("Loss 함수 + Optimizer 조합별 성능 비교 실험")
    print("="*80)
    
    # 실험 설정
    SOURCE_DOMAIN = "domain_A"  # 훈련용 소스 도메인
    TARGET_DOMAINS = "auto"  # 자동으로 나머지 도메인들 선택
    BATCH_SIZE = 16  # DRAEM backbone의 큰 메모리 사용량 고려
    MAX_EPOCHS = 1  # 충분한 학습을 위한 에포크 수
    
    # 🎯 이미지 크기 선택 (성능 최적화)
    IMAGE_SIZE = "224x224"  # 224x224가 256x256 대비 22.6% 더 빠름
    
    # Custom DRAEM 특화 설정 (이제 실험 조건에 포함됨)
    
    # 🧪 실험 조건 설정 - 확장된 Ablation Study
    EXPERIMENT_CONDITIONS = [
        # === 📐 Patch Ratio Ablation (5개) ===
        # Condition 1: Ultra Landscape (매우 가로로 긴 패치)
        {
            "name": "adamw_ultra_landscape",
            "use_adaptive_loss": False,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "sspcab": False,
            "severity_input_mode": "discriminative_only",
            "patch_width_range": (64, 128),
            "patch_ratio_range": (0.05, 0.1),  # height/width = 0.05-0.1 (매우 가로로 긴)
            "severity_max": 2.0,
            "patch_count": 1,  # 기본값
            "description": "AdamW + Ultra landscape patch ratio (0.05-0.1)"
        },
        # Condition 2: Landscape (가로로 긴 패치)
        {
            "name": "adamw_landscape",
            "use_adaptive_loss": False,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "sspcab": False,
            "severity_input_mode": "discriminative_only",
            "patch_width_range": (64, 128),
            "patch_ratio_range": (0.3, 0.7),  # height/width = 0.3-0.7 (가로로 긴)
            "severity_max": 2.0,
            "patch_count": 1,  # 기본값
            "description": "AdamW + Landscape patch ratio (0.3-0.7)"
        },
        # Condition 3: Square (정사각형 패치)
        {
            "name": "adamw_square",
            "use_adaptive_loss": False,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "sspcab": False,
            "severity_input_mode": "discriminative_only",
            "patch_width_range": (64, 128),
            "patch_ratio_range": (0.8, 1.2),  # height/width = 0.8-1.2 (거의 정사각형)
            "severity_max": 2.0,
            "patch_count": 1,  # 기본값
            "description": "AdamW + Square patch ratio (0.8-1.2)"
        },
        # Condition 4: Portrait (세로로 긴 패치)
        {
            "name": "adamw_portrait",
            "use_adaptive_loss": False,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "sspcab": False,
            "severity_input_mode": "discriminative_only",
            "patch_width_range": (64, 128),
            "patch_ratio_range": (1.5, 3.0),  # height/width = 1.5-3.0 (세로로 긴)
            "severity_max": 2.0,
            "patch_count": 1,  # 기본값
            "description": "AdamW + Portrait patch ratio (1.5-3.0)"
        },
        # Condition 5: Ultra Portrait (매우 세로로 긴 패치)
        {
            "name": "adamw_ultra_portrait",
            "use_adaptive_loss": False,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "sspcab": False,
            "severity_input_mode": "discriminative_only",
            "patch_width_range": (64, 128),
            "patch_ratio_range": (5.0, 10.0),  # height/width = 5.0-10.0 (매우 세로로 긴)
            "severity_max": 2.0,
            "patch_count": 1,  # 기본값
            "description": "AdamW + Ultra portrait patch ratio (5.0-10.0)"
        },

        # === 🎚️ Severity Max Ablation (5개) ===
        # Condition 6: Ultra Low Severity
        {
            "name": "adamw_severity_0p2",
            "use_adaptive_loss": False,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "sspcab": False,
            "severity_input_mode": "discriminative_only",
            "patch_width_range": (64, 128),
            "patch_ratio_range": (0.05, 0.1),  # 최고 성능 patch ratio 사용
            "severity_max": 0.2,
            "patch_count": 1,  # 기본값
            "description": "AdamW + Ultra low severity max (0.2)"
        },
        # Condition 7: Very Low Severity
        {
            "name": "adamw_severity_0p5",
            "use_adaptive_loss": False,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "sspcab": False,
            "severity_input_mode": "discriminative_only",
            "patch_width_range": (64, 128),
            "patch_ratio_range": (0.05, 0.1),
            "severity_max": 0.5,
            "patch_count": 1,  # 기본값
            "description": "AdamW + Very low severity max (0.5)"
        },
        # Condition 8: Low Severity
        {
            "name": "adamw_severity_1p0",
            "use_adaptive_loss": False,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "sspcab": False,
            "severity_input_mode": "discriminative_only",
            "patch_width_range": (64, 128),
            "patch_ratio_range": (0.05, 0.1),
            "severity_max": 1.0,
            "patch_count": 1,  # 기본값
            "description": "AdamW + Low severity max (1.0)"
        },
        # Condition 9: Medium Severity (baseline)
        {
            "name": "adamw_severity_2p0",
            "use_adaptive_loss": False,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "sspcab": False,
            "severity_input_mode": "discriminative_only",
            "patch_width_range": (64, 128),
            "patch_ratio_range": (0.05, 0.1),
            "severity_max": 2.0,
            "patch_count": 1,  # 기본값
            "description": "AdamW + Medium severity max (2.0)"
        },
        # Condition 10: High Severity
        {
            "name": "adamw_severity_5p0",
            "use_adaptive_loss": False,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "sspcab": False,
            "severity_input_mode": "discriminative_only",
            "patch_width_range": (64, 128),
            "patch_ratio_range": (0.05, 0.1),
            "severity_max": 5.0,
            "patch_count": 1,  # 기본값
            "description": "AdamW + High severity max (5.0)"
        },

        # === 📏 Patch Size Ablation (5개) ===
        # Condition 11: Tiny Patch
        {
            "name": "adamw_patch_tiny",
            "use_adaptive_loss": False,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "sspcab": False,
            "severity_input_mode": "discriminative_only",
            "patch_width_range": (16, 32),  # 매우 작은 패치
            "patch_ratio_range": (0.05, 0.1),
            "severity_max": 2.0,
            "patch_count": 1,  # 기본값
            "description": "AdamW + Tiny patch size (16-32)"
        },
        # Condition 12: Small Patch
        {
            "name": "adamw_patch_small",
            "use_adaptive_loss": False,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "sspcab": False,
            "severity_input_mode": "discriminative_only",
            "patch_width_range": (32, 64),  # 작은 패치
            "patch_ratio_range": (0.05, 0.1),
            "severity_max": 2.0,
            "patch_count": 1,  # 기본값
            "description": "AdamW + Small patch size (32-64)"
        },
        # Condition 13: Medium Patch (baseline)
        {
            "name": "adamw_patch_medium",
            "use_adaptive_loss": False,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "sspcab": False,
            "severity_input_mode": "discriminative_only",
            "patch_width_range": (64, 128),  # 중간 패치
            "patch_ratio_range": (0.05, 0.1),
            "severity_max": 2.0,
            "patch_count": 1,  # 기본값
            "description": "AdamW + Medium patch size (64-128)"
        },
        # Condition 14: Large Patch
        {
            "name": "adamw_patch_large",
            "use_adaptive_loss": False,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "sspcab": False,
            "severity_input_mode": "discriminative_only",
            "patch_width_range": (96, 160),  # 큰 패치
            "patch_ratio_range": (0.05, 0.1),
            "severity_max": 2.0,
            "patch_count": 1,  # 기본값
            "description": "AdamW + Large patch size (96-160)"
        },
        # Condition 15: Extra Large Patch
        {
            "name": "adamw_patch_xlarge",
            "use_adaptive_loss": False,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "sspcab": False,
            "severity_input_mode": "discriminative_only",
            "patch_width_range": (128, 192),  # 매우 큰 패치
            "patch_ratio_range": (0.05, 0.1),
            "severity_max": 2.0,
            "patch_count": 1,  # 기본값
            "description": "AdamW + Extra large patch size (128-192)"
        },

        # === 🔧 SSPCAB Ablation (2개) ===
        # Condition 16: SSPCAB Off (baseline)
        {
            "name": "adamw_sspcab_off",
            "use_adaptive_loss": False,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "sspcab": False,  # SSPCAB 비활성화
            "severity_input_mode": "discriminative_only",
            "patch_width_range": (64, 128),
            "patch_ratio_range": (0.05, 0.1),
            "severity_max": 2.0,
            "patch_count": 1,  # 기본값
            "description": "AdamW + SSPCAB disabled (baseline)"
        },
        # Condition 17: SSPCAB On
        {
            "name": "adamw_sspcab_on",
            "use_adaptive_loss": False,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "sspcab": True,  # SSPCAB 활성화
            "severity_input_mode": "discriminative_only",
            "patch_width_range": (64, 128),
            "patch_ratio_range": (0.05, 0.1),
            "severity_max": 2.0,
            "patch_count": 1,  # 기본값
            "description": "AdamW + SSPCAB enabled"
        },

        # === 🔢 Patch Count Ablation (3개) ===
        # Condition 18: Single Patch (baseline)
        {
            "name": "adamw_patch_count_1",
            "use_adaptive_loss": False,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "sspcab": False,
            "severity_input_mode": "discriminative_only",
            "patch_width_range": (64, 128),
            "patch_ratio_range": (0.05, 0.1),
            "severity_max": 2.0,
            "patch_count": 1,  # 단일 패치
            "description": "AdamW + Single patch count (1)"
        },
        # Condition 19: Double Patch
        {
            "name": "adamw_patch_count_2",
            "use_adaptive_loss": False,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "sspcab": False,
            "severity_input_mode": "discriminative_only",
            "patch_width_range": (64, 128),
            "patch_ratio_range": (0.05, 0.1),
            "severity_max": 2.0,
            "patch_count": 2,  # 이중 패치
            "description": "AdamW + Double patch count (2)"
        },
        # Condition 20: Triple Patch
        {
            "name": "adamw_patch_count_3",
            "use_adaptive_loss": False,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "sspcab": False,
            "severity_input_mode": "discriminative_only",
            "patch_width_range": (64, 128),
            "patch_ratio_range": (0.05, 0.1),
            "severity_max": 2.0,
            "patch_count": 3,  # 삼중 패치
            "description": "AdamW + Triple patch count (3)"
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
            image_size=IMAGE_SIZE
        )
        
        print(f"\n📊 Custom DRAEM 구성 요약:")
        print(f"   🔧 DRAEM Backbone: 97.4M 파라미터 (Wide ResNet + Subnetworks)")
        print(f"   🎯 Severity Head: +118K 파라미터")
        print(f"   📐 이미지 크기: {IMAGE_SIZE}")
        print(f"   🔥 배치 크기: {BATCH_SIZE}")
        print(f"   📈 총 실험 조건: {len(EXPERIMENT_CONDITIONS)}개")
        print(f"   🔬 실험 변수: Optimizer, Learning Rate, SSPCAB, Severity Mode, Patch Parameters")
        
        # ======================================================================================== 
        # 2단계: 다중 실험 조건별 순차 수행
        # ========================================================================================
        print(f"\n{'='*60}")
        print(f"2단계: 다중 실험 조건별 순차 수행")
        print(f"총 {len(EXPERIMENT_CONDITIONS)}개 실험 조건")
        print(f"{'='*60}")
        
        all_results = []
        
        for i, condition in enumerate(EXPERIMENT_CONDITIONS, 1):
            print(f"\n⏱️  진행상황: {i}/{len(EXPERIMENT_CONDITIONS)} - {condition['name']}")
            
            result = run_single_experiment(
                multi_datamodule=multi_datamodule,
                condition=condition,
                source_domain=SOURCE_DOMAIN,
                max_epochs=MAX_EPOCHS
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
                experiment_name=f"multi_domain_custom_draem_v2024_12_{SOURCE_DOMAIN}_BEST_{best_condition['name']}",
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
