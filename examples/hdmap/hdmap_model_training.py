#!/usr/bin/env python3
"""HDMAP 도메인 전이 학습 예시.

HDMAP 데이터셋과 DRAEM 모델을 활용한 도메인 전이 학습 실험 스크립트입니다.
여러 도메인(domain_A, B, C, D) 간의 이상 탐지 성능을 비교하고 도메인 전이 효과를 분석합니다.

실험 구조:
1. Source Domain (domain_A)에서 DRAEM 모델 훈련
2. Source Domain에서 성능 평가 (베이스라인)
3. Target Domains (domain_B, C, D)에서 성능 평가 (도메인 전이)
4. 결과 비교 및 분석

주요 설정:
- Validation Split: FROM_TEST (기본값) - 테스트 데이터에서 일부를 분할하여 검증용으로 사용
- 도메인 전이: 한 도메인에서 학습한 모델을 다른 도메인에 적용하여 성능 비교
- 산업 응용: 실제 제조 환경에서의 다양한 운영 조건 변화에 대한 모델 적응성 평가
"""

import os

import torch
import gc
from typing import Dict, Any

# HDMAP import
from anomalib.data.datamodules.image.hdmap import HDMAPDataModule
from anomalib.models import Draem
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


def create_hdmap_datamodule(domain: str, batch_size: int = 16) -> HDMAPDataModule:
    """HDMAP DataModule 생성.
    
    Args:
        domain: 사용할 도메인 (domain_A, domain_B, domain_C, domain_D)
        batch_size: 배치 크기
        
    Returns:
        HDMAPDataModule: 설정된 데이터 모듈
    
    Note:
        기본 val_split_mode는 FROM_TEST로 설정되어 있어 테스트 데이터에서 일부를 분할하여
        검증용으로 사용합니다. 이는 HDMAP 데이터셋의 구조상 별도 검증 폴더가 없기 때문입니다.
    """
    print(f"\n📂 {domain} DataModule 생성 중...")
    
    datamodule = HDMAPDataModule(
        # root="./datasets/HDMAP/1000_8bit_resize_256x256",
        root="./datasets/HDMAP/1000_8bit_resize_pad_256x256",
        domain=domain,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=4,  # 시스템에 맞게 조정
        # val_split_mode=ValSplitMode.FROM_TEST (기본값)
        # 테스트 데이터에서 일부를 분할하여 검증용으로 사용
    )
    
    # 데이터 준비 및 설정
    datamodule.prepare_data()
    datamodule.setup()
    
    print(f"✅ {domain} 데이터 로드 완료")
    print(f"   훈련 샘플: {len(datamodule.train_data)}개")
    print(f"   검증 샘플: {len(datamodule.val_data) if datamodule.val_data else 0}개")
    print(f"   테스트 샘플: {len(datamodule.test_data)}개")
    
    return datamodule


def train_draem_model(
    datamodule: HDMAPDataModule, 
    experiment_name: str,
    max_epochs: int = 3
) -> tuple[Draem, Engine]:
    """DRAEM 모델 훈련.
    
    Args:
        datamodule: 훈련용 데이터 모듈
        experiment_name: 실험 이름 (로그용)
        max_epochs: 최대 에포크 수
        
    Returns:
        tuple: (훈련된 모델, Engine 객체)
    """
    print(f"\n🤖 DRAEM 모델 훈련 시작 - {experiment_name}")
    print(f"   도메인: {datamodule.domain}")
    print(f"   최대 에포크: {max_epochs}")
    
    # DRAEM 모델 생성
    model = Draem()
    
    # TensorBoard 로거 설정
    logger = AnomalibTensorBoardLogger(
        save_dir="logs/hdmap",
        name=experiment_name
    )
    
    # Engine 설정
    engine = Engine(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[0] if torch.cuda.is_available() else 1,
        logger=logger,
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
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


def evaluate_model(
    model: Draem, 
    engine: Engine, 
    datamodule: HDMAPDataModule,
    checkpoint_path: str = None
) -> Dict[str, Any]:
    """모델 평가.
    
    Args:
        model: 평가할 모델
        engine: Engine 객체
        datamodule: 평가용 데이터 모듈
        checkpoint_path: 체크포인트 경로 (None이면 현재 모델 사용)
        
    Returns:
        Dict: 평가 결과
    """
    print(f"\n📊 모델 평가 중 - {datamodule.domain}")
    
    # 평가 실행
    if checkpoint_path:
        results = engine.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=checkpoint_path
        )
    else:
        results = engine.test(
            model=model,
            datamodule=datamodule
        )
    
    print(f"✅ {datamodule.domain} 평가 완료")
    
    return results[0] if results else {}


def main():
    """메인 실험 함수."""
    print("="*80)
    print("🚀 HDMAP 도메인 전이 학습 실험 시작")
    print("도메인 전이: 한 도메인에서 학습한 모델을 다른 도메인에 적용")
    print("산업 응용: 다양한 운영 조건에서의 이상 탐지 모델 적응성 평가")
    print("="*80)
    
    # 실험 설정
    SOURCE_DOMAIN = "domain_A"  # 훈련용 소스 도메인
    TARGET_DOMAINS = ["domain_B", "domain_C", "domain_D"]  # 테스트용 타겟 도메인들
    BATCH_SIZE = 16
    MAX_EPOCHS = 3  # 빠른 실험을 위해 적게 설정
    
    # GPU 메모리 정리
    cleanup_gpu_memory()
    
    try:
        # ========================================================================================
        # 1단계: Source Domain에서 모델 훈련
        # ========================================================================================
        print(f"\n{'='*60}")
        print(f"1단계: Source Domain ({SOURCE_DOMAIN})에서 DRAEM 모델 훈련")
        print(f"{'='*60}")
        
        # Source domain 데이터 준비
        source_datamodule = create_hdmap_datamodule(
            domain=SOURCE_DOMAIN, 
            batch_size=BATCH_SIZE
        )
        
        # 모델 훈련
        trained_model, engine = train_draem_model(
            datamodule=source_datamodule,
            experiment_name=f"hdmap_tutorial_draem_{SOURCE_DOMAIN}",
            max_epochs=MAX_EPOCHS
        )
        
        # 체크포인트 경로 저장
        best_checkpoint = engine.trainer.checkpoint_callback.best_model_path
        
        # ========================================================================================
        # 2단계: Source Domain에서 성능 평가 (베이스라인)
        # ========================================================================================
        print(f"\n{'='*60}")
        print(f"2단계: Source Domain ({SOURCE_DOMAIN}) 성능 평가 (베이스라인)")
        print(f"{'='*60}")
        
        source_results = evaluate_model(
            model=trained_model,
            engine=engine,
            datamodule=source_datamodule,
            checkpoint_path=best_checkpoint
        )
        
        # ========================================================================================
        # 3단계: Target Domains에서 성능 평가 (도메인 전이)
        # ========================================================================================
        print(f"\n{'='*60}")
        print(f"3단계: Target Domains에서 성능 평가 (도메인 전이)")
        print(f"{'='*60}")
        
        target_results = {}
        
        for target_domain in TARGET_DOMAINS:
            print(f"\n📋 {target_domain} 평가 중...")
            
            # Target domain 데이터 준비
            target_datamodule = create_hdmap_datamodule(
                domain=target_domain,
                batch_size=BATCH_SIZE
            )
            
            # 평가 실행
            results = evaluate_model(
                model=trained_model,
                engine=engine,
                datamodule=target_datamodule,
                checkpoint_path=best_checkpoint
            )
            
            target_results[target_domain] = results
        
        # ========================================================================================
        # 4단계: 결과 비교 및 분석
        # ========================================================================================
        print(f"\n{'='*80}")
        print(f"4단계: 도메인 전이 학습 결과 분석")
        print(f"{'='*80}")
        
        print(f"\n📊 성능 요약:")
        print(f"{'도메인':<12} {'AUROC':<8} {'F1-Score':<10} {'설명'}")
        print("-" * 50)
        
        # Source domain 결과
        if 'test/AUROC' in source_results:
            auroc = source_results['test/AUROC']
            f1 = source_results.get('test/F1Score', 'N/A')
            print(f"{SOURCE_DOMAIN:<12} {auroc:<8.3f} {f1:<10} (Source - 베이스라인)")
        
        # Target domains 결과
        for domain, results in target_results.items():
            if 'test/AUROC' in results:
                auroc = results['test/AUROC']
                f1 = results.get('test/F1Score', 'N/A')
                print(f"{domain:<12} {auroc:<8.3f} {f1:<10} (Target - 전이학습)")
        
        # 도메인 전이 효과 분석
        print(f"\n🔍 도메인 전이 분석:")
        if 'test/AUROC' in source_results:
            source_auroc = source_results['test/AUROC']
            
            for domain, results in target_results.items():
                if 'test/AUROC' in results:
                    target_auroc = results['test/AUROC']
                    diff = target_auroc - source_auroc
                    percentage = (diff / source_auroc) * 100
                    
                    status = "🔥 성능 향상" if diff > 0 else "📉 성능 저하" if diff < -0.05 else "✅ 유사한 성능"
                    print(f"   {domain}: {diff:+.3f} ({percentage:+.1f}%) - {status}")
        
        print(f"\n🎯 실험 완료!")
        print(f"   로그 디렉토리: logs/hdmap/")
        print(f"   TensorBoard: tensorboard --logdir=logs/hdmap/")
        print(f"   체크포인트: {best_checkpoint}")
        
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
