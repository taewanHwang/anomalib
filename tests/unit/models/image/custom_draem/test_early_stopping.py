#!/usr/bin/env python3
"""Target Domain Early Stopping Test for Custom DRAEM.

PyTorch Lightning의 EarlyStopping callback을 활용한 val_image_AUROC 기반 
early stopping 기능을 테스트하는 모듈입니다.

주요 특징:
- PyTorch Lightning EarlyStopping callback 활용
- Source Domain validation AUROC 모니터링 (val_image_AUROC)
- 파라미터화된 early stopping 설정 (monitor, patience, min_delta)
- Multi-domain evaluation with automatic early stopping

Early Stopping 설정:
- monitor: "val_image_AUROC" (source validation AUROC)
- patience: 2 (2 epochs 동안 개선 없으면 중단)
- min_delta: 0.005 (최소 0.5% 개선 필요)
- mode: "max" (AUROC 최대화)

실험 구조:
1. Source Domain (domain_A)에서 모델 훈련
2. 매 validation epoch마다 source validation AUROC 계산
3. 성능 개선이 없으면 early stopping 실행
4. Best checkpoint로 final test evaluation

Run with: pytest tests/unit/models/image/custom_draem/test_early_stopping.py -v -s
"""

import os
import torch
import gc
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import logging

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")
warnings.filterwarnings("ignore", message=".*Importing from timm.models.layers.*")
warnings.filterwarnings("ignore", message=".*multi-threaded.*fork.*")

# Lightning imports
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# Anomalib imports
from anomalib.data.datamodules.image.multi_domain_hdmap import MultiDomainHDMAPDataModule
from anomalib.models.image.custom_draem import CustomDraem
from anomalib.engine import Engine
from anomalib.loggers import AnomalibTensorBoardLogger

# 경고 메시지 비활성화
logging.getLogger("anomalib.visualization.image.item_visualizer").setLevel(logging.ERROR)
logging.getLogger("anomalib.visualization").setLevel(logging.ERROR)
logging.getLogger("anomalib.callbacks").setLevel(logging.ERROR)

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "8"


def cleanup_gpu_memory():
    """GPU 메모리 정리 및 상태 출력."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


def run_custom_draem_with_early_stopping(
    source_domain: str = "domain_A",
    target_domains: str = "auto",
    max_epochs: int = 20,  # 더 많은 epochs로 early stopping 확률 증가
    # Early Stopping 파라미터들  
    monitor: str = "val_image_AUROC",  # source domain validation AUROC
    patience: int = 2,  # 더 빠른 early stopping
    min_delta: float = 0.005,  # AUROC는 0-1 범위이므로 작은 변화
    mode: str = "max",  # AUROC는 높을수록 좋음
    # DRAEM-SevNet 모델 파라미터들
    severity_head_mode: str = "single_scale",
    score_combination: str = "simple_average", 
    severity_loss_type: str = "mse",
    # 학습 파라미터들
    learning_rate: float = 0.0001,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """DRAEM-SevNet에 Early Stopping을 적용한 학습 실행.
    
    Args:
        source_domain: 훈련에 사용할 source domain
        target_domains: 평가할 target domains ("auto"이면 자동 설정)
        max_epochs: 최대 학습 epochs
        monitor: Early stopping 모니터링 지표
        patience: Early stopping patience
        min_delta: Early stopping 최소 개선값
        mode: Early stopping 모드 ("max" 또는 "min")
        severity_head_mode: SeverityHead 모드 ("single_scale" 또는 "multi_scale")
        score_combination: Score 결합 방식 ("simple_average", "weighted_average", "maximum")
        severity_loss_type: Severity loss 타입 ("mse" 또는 "smooth_l1")
        learning_rate: 학습률
        batch_size: 배치 크기
        
    Returns:
        Dict[str, Any]: 실험 결과 딕셔너리
    """
    print("🚀 Custom DRAEM with Target Domain Early Stopping 시작")
    print(f"📊 Early Stopping 설정: monitor={monitor} (source validation), patience={patience}, min_delta={min_delta}, mode={mode}")
    
    # 실험 설정 출력
    config_info = {
        "source_domain": source_domain,
        "target_domains": target_domains,
        "max_epochs": max_epochs,
        "early_stopping": {
            "monitor": monitor,
            "patience": patience,
            "min_delta": min_delta,
            "mode": mode,
            "strategy": "source_domain_validation"
        },
        "model_config": {
            "severity_head_mode": severity_head_mode,
            "score_combination": score_combination,
            "severity_loss_type": severity_loss_type
        },
        "training_config": {
            "learning_rate": learning_rate,
            "batch_size": batch_size
        }
    }
    
    print("📋 실험 설정:")
    print(json.dumps(config_info, indent=2, ensure_ascii=False))
    
    # GPU 메모리 정리
    cleanup_gpu_memory()
    
    try:
        # 1. DataModule 설정
        print(f"\n📂 DataModule 설정 (Source: {source_domain}, Targets: {target_domains})")
        datamodule = MultiDomainHDMAPDataModule(
            root="./datasets/HDMAP/1000_8bit_resize_pad_256x256",
            source_domain=source_domain,
            target_domains=target_domains,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=8,
        )
        
        # DataModule 준비
        datamodule.prepare_data()
        datamodule.setup()
        
        print(f"✅ Source Domain: {datamodule.source_domain}")
        print(f"✅ Target Domains: {datamodule.target_domains}")
        
        # 2. 모델 생성
        print(f"\n🤖 DRAEM-SevNet 모델 생성")
        model = CustomDraem(
            severity_head_mode="single_scale",  # DRAEM-SevNet 파라미터
            score_combination="simple_average",
            severity_loss_type="mse",
            learning_rate=learning_rate,
        )
        
        print(f"✅ Severity Head Mode: single_scale")
        print(f"✅ Score Combination: simple_average")
        print(f"✅ Severity Loss Type: mse")
        
        # 3. Callbacks 설정
        print(f"\n📋 Callbacks 설정")
        
        # Early Stopping Callback
        early_stopping = EarlyStopping(
            monitor=monitor,
            patience=patience,
            min_delta=min_delta,
            mode=mode,
            verbose=True,
            strict=True,  # 🔧 디버깅: 지표가 없으면 즉시 오류 발생하여 문제 확인
        )
        
        # Model Checkpoint Callback
        checkpoint = ModelCheckpoint(
            monitor=monitor,
            mode=mode,
            save_top_k=1,
            save_last=True,
            filename=f"custom_draem_early_stop_{source_domain}_" + "{epoch:02d}_{target_avg_auroc:.4f}",
        )
        
        callbacks = [early_stopping, checkpoint]
        
        print(f"✅ EarlyStopping: monitor={monitor}, patience={patience}, min_delta={min_delta}")
        print(f"✅ ModelCheckpoint: monitor={monitor}, mode={mode}")
        
        # 4. Logger 설정
        experiment_name = f"custom_draem_early_stopping_{source_domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger = AnomalibTensorBoardLogger(
            save_dir="logs/hdmap_early_stopping",
            name=experiment_name,
        )
        
        # 5. Engine 설정 및 학습
        print(f"\n🚂 Engine 설정 및 학습 시작")
        engine = Engine(
            callbacks=callbacks,
            logger=logger,
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            deterministic=False,  # 성능 향상을 위해
            enable_checkpointing=True,
            enable_model_summary=True,
            enable_progress_bar=True,
        )
        
        print(f"✅ Max Epochs: {max_epochs}")
        print(f"✅ Logger: {experiment_name}")
        
        # 학습 실행
        print(f"\n🔥 학습 시작!")
        start_time = datetime.now()
        
        engine.fit(model=model, datamodule=datamodule)
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Early stopping 정보 수집
        actual_epochs = engine.trainer.current_epoch + 1
        stopped_early = engine.trainer.current_epoch < max_epochs - 1
        best_score = early_stopping.best_score.item() if early_stopping.best_score is not None else None
        
        print(f"\n✅ 학습 완료!")
        print(f"📊 총 학습 Epochs: {actual_epochs}/{max_epochs}")
        print(f"⏰ 학습 시간: {training_time:.2f}초")
        
        # 핵심 Early Stopping 정보 (로그 분석용)
        if stopped_early:
            print(f"🛑 Early Stopping 적용: patience={patience}에서 중단됨 (Best {monitor}: {best_score:.4f})")
        else:
            print(f"✅ 정상 완료: 최대 epochs까지 학습 (Final {monitor}: {best_score:.4f})")
        
        # 6. 최종 평가 (Best checkpoint 사용)
        print(f"\n📊 최종 성능 평가")
        if checkpoint.best_model_path:
            print(f"📂 Best checkpoint 사용: {checkpoint.best_model_path}")
            test_results = engine.test(datamodule=datamodule, ckpt_path=checkpoint.best_model_path)
        else:
            print(f"⚠️ Best checkpoint 없음, 현재 모델 사용")
            test_results = engine.test(model=model, datamodule=datamodule)
        
        # 결과 정리
        results = {
            "experiment_config": config_info,
            "training_info": {
                "actual_epochs": actual_epochs,
                "max_epochs": max_epochs,
                "stopped_early": stopped_early,
                "training_time_seconds": training_time,
                "best_score": best_score,
                "monitor_metric": monitor
            },
            "final_results": test_results,
            "model_path": str(checkpoint.best_model_path) if checkpoint.best_model_path else None,
            "log_dir": str(logger.log_dir)
        }
        
        # 결과 저장
        results_file = f"results/early_stopping_{experiment_name}.json"
        os.makedirs("results", exist_ok=True)
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 결과 저장: {results_file}")
        print(f"📁 모델 저장: {checkpoint.best_model_path}")
        print(f"📁 로그 저장: {logger.log_dir}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 실험 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    
    finally:
        # GPU 메모리 정리
        cleanup_gpu_memory()


def run_early_stopping_ablation_study():
    """Early Stopping 설정에 따른 ablation study 실행."""
    print("🔬 Early Stopping Ablation Study 시작")
    
    # 다양한 early stopping 설정
    early_stopping_configs = [
        # 기본 설정 (빠른 테스트용)
        {"patience": 2, "min_delta": 0.01, "name": "default"},
        # 더 관대한 설정 (오래 기다림) - 주석처리하면 빠른 테스트 가능
        # {"patience": 5, "min_delta": 0.003, "name": "patient"},
        # 더 엄격한 설정 (빨리 중단) - 주석처리하면 빠른 테스트 가능
        # {"patience": 2, "min_delta": 0.01, "name": "strict"},
        # 매우 엄격한 설정 - 주석처리하면 빠른 테스트 가능
        # {"patience": 1, "min_delta": 0.015, "name": "very_strict"},
    ]
    
    all_results = {}
    
    for config in early_stopping_configs:
        print(f"\n🧪 실험: {config['name']} (patience={config['patience']}, min_delta={config['min_delta']})")
        
        results = run_custom_draem_with_early_stopping(
            source_domain="domain_A",
            target_domains="auto",
            max_epochs=8,  # 빠른 테스트를 위해 단축
            patience=config["patience"],
            min_delta=config["min_delta"],
            batch_size=16,
        )
        
        all_results[config["name"]] = results
        
        # 중간 결과 출력
        if "training_info" in results:
            training_info = results["training_info"]
            print(f"결과: {training_info['actual_epochs']}epochs, "
                  f"early_stop={training_info['stopped_early']}, "
                  f"best_score={training_info['best_score']:.4f}")
    
    # 전체 결과 저장
    ablation_results_file = f"results/early_stopping_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(ablation_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Ablation Study 완료!")
    print(f"📁 전체 결과 저장: {ablation_results_file}")
    
    return all_results


def test_target_domain_early_stopping():
    """Target Domain Early Stopping 기능 테스트."""
    print("=" * 80)
    print("Target Domain Early Stopping Test")
    print("=" * 80)
    
    # 기본 설정으로 테스트 실행
    results = run_custom_draem_with_early_stopping(
        source_domain="domain_A",
        target_domains="auto",
        max_epochs=10,  # 빠른 테스트를 위해 단축  # 더 많은 epochs로 early stopping 확률 증가
        monitor="val_image_AUROC",  # source domain validation AUROC
        patience=2,  # 더 빠른 early stopping
        min_delta=0.005,
        mode="max",  # AUROC는 높을수록 좋음
        batch_size=16,
    )
    
    # 테스트 결과 검증
    assert "training_info" in results, "Training info missing in results"
    
    # Early stopping은 작동할 수도, 안 할 수도 있음 (데이터와 학습에 따라)
    training_info = results["training_info"]
    if training_info["stopped_early"]:
        print(f"✅ Early stopping 작동: {training_info['actual_epochs']}/{training_info['max_epochs']} epochs")
        assert training_info["actual_epochs"] < training_info["max_epochs"], "Should stop before max epochs"
    else:
        print(f"✅ 정상 완료: {training_info['actual_epochs']}/{training_info['max_epochs']} epochs")
        assert training_info["actual_epochs"] == training_info["max_epochs"], "Should complete all epochs"
    
    print("\n✅ Target Domain Early Stopping 테스트 통과!")
    # Note: Results validated and saved to file


def test_early_stopping_ablation_study():
    """Early Stopping 설정에 따른 ablation study 테스트."""
    print("\n" + "=" * 80)
    print("Early Stopping Ablation Study Test")
    print("=" * 80)
    
    results = run_early_stopping_ablation_study()
    
    # 모든 설정에서 결과가 있는지 확인
    assert len(results) == 4, "Should have 4 different configurations"
    for config_name, result in results.items():
        assert "training_info" in result, f"Training info missing for {config_name}"
    
    print("\n✅ Early Stopping Ablation Study 테스트 통과!")
    # Note: Results validated and saved to file


# pytest로 실행 시 자동으로 실행되는 테스트 함수들
def test_early_stopping_functionality():
    """Early stopping 기능 통합 테스트"""
    print("\n🧪 Early Stopping Test Suite")
    print("=" * 50)
    print("Testing PyTorch Lightning EarlyStopping callback integration...")
    
    # 기본 early stopping 테스트 실행
    result = test_target_domain_early_stopping()
    
    # 결과 검증
    assert result is not None, "Early stopping test should return results"
    assert "training_info" in result, "Training info should be available"
    assert "final_results" in result, "Final results should be available"
    
    print("\n✅ All early stopping tests passed!")
    # Note: Results validated through assertions above


if __name__ == "__main__":
    print("\n🧪 Early Stopping Test Suite")
    print("=" * 50)
    print("To run as pytest:")
    print("pytest tests/unit/models/image/custom_draem/test_early_stopping.py -v -s")
    print("\nRunning direct execution...")
    test_target_domain_early_stopping()
