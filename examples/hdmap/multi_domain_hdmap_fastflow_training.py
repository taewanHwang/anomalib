#!/usr/bin/env python3
"""MultiDomain HDMAP FastFlow 도메인 전이 학습 예시.

MultiDomainHDMAPDataModule과 FastFlow 모델을 활용한 효율적인 도메인 전이 학습 실험 스크립트입니다.
단일 DataModule로 여러 도메인을 동시에 관리하여 더 체계적이고 효율적인 실험을 수행합니다.

기존 방식 vs MultiDomain 방식:
- 기존: 각 도메인별로 개별 DataModule 생성 → 반복적인 코드, 메모리 비효율
- MultiDomain: 하나의 DataModule로 여러 도메인 통합 관리 → 깔끔한 코드, 효율적 관리

실험 구조:
1. MultiDomainHDMAPDataModule 설정 (source: domain_A, targets: domain_B,C,D)
2. Source Domain에서 FastFlow 모델 훈련 (train 데이터)
3. Source Domain에서 성능 평가 (validation으로 사용된 test 데이터)
4. Target Domains에서 동시 성능 평가 (각 도메인별 test 데이터)
5. 도메인 전이 효과 종합 분석

주요 개선점:
- 단일 DataModule로 모든 도메인 관리
- 자동 타겟 도메인 설정 (target_domains="auto")
- 효율적인 메모리 사용
- 일관된 실험 설정
- FastFlow 모델에 최적화된 AUROC 기반 평가
"""

import os
import torch
import gc
from typing import Dict, Any, List

# MultiDomain HDMAP import
from anomalib.data.datamodules.image.multi_domain_hdmap import MultiDomainHDMAPDataModule
from anomalib.models import Fastflow
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


def create_multi_domain_datamodule(
    source_domain: str = "domain_A",
    target_domains: str | List[str] = "auto",
    batch_size: int = 16
) -> MultiDomainHDMAPDataModule:
    """MultiDomain HDMAP DataModule 생성.
    
    Args:
        source_domain: 훈련용 소스 도메인 (예: "domain_A")
        target_domains: 타겟 도메인들 ("auto" 또는 명시적 리스트)
        batch_size: 배치 크기
        
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
        root="./datasets/HDMAP/1000_8bit_resize_pad_256x256",
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


def train_fastflow_model_multi_domain(
    datamodule: MultiDomainHDMAPDataModule, 
    experiment_name: str,
    max_epochs: int = 20,
) -> tuple[Fastflow, Engine]:
    """MultiDomain DataModule을 사용한 FastFlow 모델 훈련.
    
    Args:
        datamodule: 멀티 도메인 데이터 모듈
        experiment_name: 실험 이름 (로그용)
        max_epochs: 최대 에포크 수 (기본값: 20)
        learning_rate: 학습률 (기본값: 0.0001)
        
    Returns:
        tuple: (훈련된 모델, Engine 객체)
        
    Note:
        훈련 과정:
        1. Source domain train 데이터로 모델 훈련
        2. Source domain test 데이터로 validation (정상+이상 데이터 포함)
        3. 각 에포크마다 validation 성능으로 모델 개선 추적
    """
    print(f"\n🤖 FastFlow 모델 훈련 시작 - {experiment_name}")
    print(f"   Source Domain: {datamodule.source_domain}")
    print(f"   Validation Strategy: {datamodule.validation_strategy}")
    print(f"   Max Epochs: {max_epochs}")
    
    # FastFlow 모델 생성 (AUROC 전용 메트릭으로 설정)
    from anomalib.metrics import Evaluator, AUROC
    
    # AUROC만 사용 (FastFlow는 pred_score만 생성, pred_label 없음)
    # FastFlow는 anomaly score 기반 모델이므로 AUROC가 가장 적절한 메트릭
    auroc_metric = AUROC(fields=["pred_score", "gt_label"])
    
    evaluator = Evaluator(
        val_metrics=[auroc_metric],
        test_metrics=[auroc_metric]
    )
    
    model = Fastflow(evaluator=evaluator)
    
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
        num_sanity_val_steps=2,
    )
    
    # 모델 훈련
    print("🔥 훈련 시작...")
    engine.fit(model=model, datamodule=datamodule)
    
    print("✅ 모델 훈련 완료!")
    print(f"   체크포인트: {engine.trainer.checkpoint_callback.best_model_path}")
    
    return model, engine


def evaluate_source_domain(
    model: Fastflow, 
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
        이는 훈련 중에 사용된 validation 데이터와 동일합니다.
    """
    print(f"\n📊 Source Domain 성능 평가 - {datamodule.source_domain}")
    print("   💡 평가 데이터: Source domain test (validation으로 사용된 데이터)")
    
    # Source domain validation 데이터로 평가
    # 참고: MultiDomainHDMAPDataModule에서는 source test가 validation으로 사용됨
    if checkpoint_path:
        results = engine.validate(
            model=model,
            datamodule=datamodule,
            ckpt_path=checkpoint_path
        )
    else:
        results = engine.validate(
            model=model,
            datamodule=datamodule
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
    if isinstance(result, dict):
        # AUROC 결과 출력 (FastFlow 전용)
        auroc_value = result.get('AUROC', 'N/A')
        if auroc_value != 'N/A':
            print(f"      AUROC: {auroc_value:.3f}")
        else:
            print(f"      AUROC: 측정되지 않음")
    else:
        print(f"      평가 결과가 올바르지 않습니다: {type(result)}")
    
    return result


def evaluate_target_domains(
    model: Fastflow, 
    engine: Engine, 
    datamodule: MultiDomainHDMAPDataModule,
    checkpoint_path: str = None
) -> Dict[str, Dict[str, Any]]:
    """Target Domains 성능 평가.
    
    Args:
        model: 평가할 모델
        engine: Engine 객체
        datamodule: 멀티 도메인 데이터 모듈
        checkpoint_path: 체크포인트 경로
        
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
        # PyTorch Lightning Engine은 단일 DataLoader를 기대하므로
        # 각 target domain별로 개별 평가 수행
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
    print(f"📈 도메인 전이 학습 결과 종합 분석")
    print(f"{'='*80}")
    
    # 성능 요약 테이블
    print(f"\n📊 성능 요약:")
    print(f"{'도메인':<12} {'AUROC':<8} {'F1-Score':<10} {'유형':<15} {'설명'}")
    print("-" * 65)
    
    # Source domain 결과 (AUROC만 사용)
    source_auroc = source_results.get('AUROC', None)
    
    if source_auroc is not None:
        print(f"{source_domain:<12} {source_auroc:<8.3f} {'N/A':<10} {'Source':<15} 베이스라인 (훈련 도메인)")
    else:
        print(f"{source_domain:<12} {'N/A':<8} {'N/A':<10} {'Source':<15} 베이스라인 (결과 없음)")
    
    # Target domains 결과 (AUROC만 사용)
    target_performances = []
    for domain, results in target_results.items():
        # FastFlow test 결과에서 AUROC 추출
        target_auroc = results.get('AUROC', 'N/A')
        
        if target_auroc != 'N/A':
            print(f"{domain:<12} {target_auroc:<8.3f} {'N/A':<10} {'Target':<15} 도메인 전이")
            target_performances.append((domain, target_auroc, 'N/A'))
            

def main():
    """메인 실험 함수."""
    print("="*80)
    print("🚀 MultiDomain HDMAP FastFlow 도메인 전이 학습 실험")
    print("MultiDomainHDMAPDataModule + FastFlow 모델 전용 도메인 전이 학습")
    print("="*80)
    
    # 실험 설정 (개선된 파라미터)
    SOURCE_DOMAIN = "domain_D"  # 훈련용 소스 도메인
    TARGET_DOMAINS = "auto"  # 자동으로 나머지 도메인들 선택
    BATCH_SIZE = 16
    MAX_EPOCHS = 10
    
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
            batch_size=BATCH_SIZE
        )
        
        # ========================================================================================
        # 2단계: Source Domain에서 모델 훈련
        # ========================================================================================
        print(f"\n{'='*60}")
        print(f"2단계: Source Domain ({multi_datamodule.source_domain})에서 모델 훈련")
        print(f"{'='*60}")
        
        trained_model, engine = train_fastflow_model_multi_domain(
            datamodule=multi_datamodule,
            experiment_name=f"multi_domain_fastflow_{SOURCE_DOMAIN}",
            max_epochs=MAX_EPOCHS,
        )
        
        # 체크포인트 경로 저장
        best_checkpoint = engine.trainer.checkpoint_callback.best_model_path
        
        # ========================================================================================
        # 3단계: Source Domain 성능 평가 (베이스라인)
        # ========================================================================================
        print(f"\n{'='*60}")
        print(f"3단계: Source Domain 성능 평가 (베이스라인)")
        print(f"{'='*60}")
        
        source_results = evaluate_source_domain(
            model=trained_model,
            engine=engine,
            datamodule=multi_datamodule,
            checkpoint_path=best_checkpoint
        )
        
        # ========================================================================================
        # 4단계: Target Domains 성능 평가 (도메인 전이)
        # ========================================================================================
        print(f"\n{'='*60}")
        print(f"4단계: Target Domains 성능 평가 (도메인 전이)")
        print(f"{'='*60}")
        
        target_results = evaluate_target_domains(
            model=trained_model,
            engine=engine,
            datamodule=multi_datamodule,
            checkpoint_path=best_checkpoint
        )
        
        # ========================================================================================
        # 5단계: 결과 분석 및 인사이트
        # ========================================================================================
        analyze_domain_transfer_results(
            source_domain=multi_datamodule.source_domain,
            source_results=source_results,
            target_results=target_results
        )
                
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
