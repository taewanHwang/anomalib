#!/usr/bin/env python3
"""All Domains HDMAP Dinomaly 실험 스크립트.

Dinomaly 모델과 AllDomainsHDMAPDataModule을 활용한 all domains HDMAP 이상 탐지 실험 스크립트입니다.

Dinomaly 특징:
- Vision Transformer 기반: DINOv2 사전훈련 모델을 encoder로 활용
- Encoder-Decoder 구조: 특징 재구성을 통한 이상 탐지
- Multi-scale Feature 활용: DINOv2의 여러 중간 레이어에서 특징 추출
- Cosine Similarity 기반: 인코더-디코더 특징 간 유사도로 이상도 계산
- Reconstruction Loss: MSE + Cosine 손실을 통한 재구성 학습
- 고해상도 입력: 518x518 입력으로 세밀한 이상 탐지

실험 구조:
1. AllDomainsHDMAPDataModule 설정 (모든 도메인의 데이터 통합)
2. 통합된 데이터에서 Dinomaly 모델 훈련 (train 데이터)
3. 모든 도메인 통합 성능 평가 (test 데이터)

주요 개선점 (Dinomaly vs CNN 기반 모델):
- 전역 컨텍스트: ViT의 self-attention으로 전체 이미지 관계 파악
- 사전훈련 품질: DINOv2의 고품질 self-supervised 특징 활용
- 세밀한 로컬라이제이션: 고해상도 입력과 패치 기반 처리로 정밀한 이상 위치 파악
- 복잡한 패턴 탐지: transformer 구조로 복잡한 이상 패턴 모델링

NOTE:
- 실험 조건들은 dinomaly-exp_condition.json 파일에서 관리됩니다.
- DINOv2 모델은 518x518 입력 크기를 요구하므로 image_size 설정에 주의하세요.
"""

import os
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import logging
import warnings
import argparse

# All Domains HDMAP import
from anomalib.data.datamodules.image.all_domains_hdmap import AllDomainsHDMAPDataModule
from anomalib.models.image.dinomaly import Dinomaly
from anomalib.engine import Engine
from pytorch_lightning.loggers import TensorBoardLogger

# Early Stopping import (Dinomaly는 학습을 요구함)
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# 공통 유틸리티 함수들 import - 상위 디렉토리에서 import
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from experiment_utils import (
    cleanup_gpu_memory,
    setup_warnings_filter,
    setup_experiment_logging,
    extract_training_info,
    save_experiment_results,
    load_experiment_conditions
)


# JSON 파일에서 실험 조건 로드
EXPERIMENT_CONDITIONS = load_experiment_conditions("dinomaly-exp_condition.json")

# 경고 메시지 비활성화
setup_warnings_filter()
logging.getLogger("anomalib.visualization.image.item_visualizer").setLevel(logging.ERROR)
logging.getLogger("anomalib.visualization").setLevel(logging.ERROR)
logging.getLogger("anomalib.callbacks").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")

# ========================================================================================
# 모델 훈련 및 실험 함수들
# ========================================================================================

def train_dinomaly_model_all_domains(
    datamodule: AllDomainsHDMAPDataModule, 
    config: Dict[str, Any],
    results_base_dir: str,
    logger: logging.Logger
) -> tuple[Dinomaly, Engine, str]:
    """Dinomaly 모델 훈련 수행.
    
    Args:
        datamodule: 설정된 AllDomainsHDMAPDataModule
        config: 훈련 설정 딕셔너리
        results_base_dir: 결과 저장 기본 경로
        logger: 로거 객체
        
    Returns:
        tuple: (훈련된 모델, Engine 객체, 체크포인트 경로)
        
    Note:
        Dinomaly 특징:
        - Vision Transformer 기반: DINOv2를 encoder로 사용
        - Reconstruction Loss: MSE + Cosine 손실로 특징 재구성 학습
        - Multi-scale Features: 여러 ViT 레이어에서 특징 추출
        - High Resolution: 518x518 입력으로 세밀한 이상 탐지
    """
    
    print(f"\n🚀 Dinomaly 모델 훈련 시작")
    logger.info("🚀 Dinomaly 모델 훈련 시작")
    
    print(f"   🔧 Config 설정:")
    print(f"      • Encoder Name: {config['encoder_name']}")
    print(f"      • Target Layers: {config['target_layers']}")
    print(f"      • Bottleneck Dropout: {config['bottleneck_dropout']}")
    print(f"      • Decoder Depth: {config['decoder_depth']}")
    if 'max_steps' in config:
        print(f"      • Max Steps: {config['max_steps']}")
    elif 'max_epochs' in config:
        print(f"      • Max Epochs: {config['max_epochs']}")
    print(f"      • Input Size: {config['image_size']}")
    
    logger.info("✅ Dinomaly 모델 생성 완료 (DINOv2 기반 학습 모델)")
    logger.info(f"🔧 Config 설정: encoder_name={config['encoder_name']}, target_layers={config['target_layers']}")
    
    # Dinomaly 모델 생성 - None 값 처리
    model_params = {
        "encoder_name": config["encoder_name"],
        "bottleneck_dropout": config["bottleneck_dropout"],
        "decoder_depth": config["decoder_depth"],
        "remove_class_token": config["remove_class_token"]
    }
    
    # target_layers가 null이 아닌 경우만 전달 (None이면 기본값 사용)
    if config["target_layers"] is not None:
        model_params["target_layers"] = config["target_layers"]
    
    model = Dinomaly(**model_params)
    
    print(f"   ✅ Dinomaly 모델 생성 완료")
    print(f"   📊 특징: DINOv2 기반, Encoder-Decoder 구조, 재구성 기반 학습")
    logger.info("📊 Dinomaly 특징: DINOv2 기반, Encoder-Decoder 구조, 재구성 기반 학습")
    
    # 🎯 콜백 설정 (Dinomaly는 학습이 필요함)
    callbacks = []
    
    # val_image_AUROC가 pseudo 값(0.5)으로 고정되는 문제로 val_loss 기반으로 변경
    # Dinomaly는 reconstruction loss를 사용하므로 train_loss 기반이 더 안정적
    early_stopping = EarlyStopping(
        monitor="train_loss",  # train_loss는 더 안정적으로 감소함
        patience=config["early_stopping_patience"] * 2,  # patience를 늘려서 충분히 학습하도록
        mode="min",
        verbose=True,
        min_delta=0.001,
        stopping_threshold=0.01  # loss가 0.01 아래로 떨어지면 조기 종료하지 않음
    )
    
    checkpoint_callback = ModelCheckpoint(
        filename=f"dinomaly_all_domains_" + "{step:05d}_{train_loss:.4f}",
        monitor="train_loss",
        mode="min",
        save_top_k=3,  # 최고 성능 3개 체크포인트 저장
        verbose=True,
        save_last=True  # 마지막 체크포인트도 저장
    )
    
    callbacks.extend([early_stopping, checkpoint_callback])
    
    # TensorBoard 로거 설정
    tb_logger = TensorBoardLogger(
        save_dir=results_base_dir,
        name="tensorboard_logs",
        version=""  # 빈 버전으로 version_x 폴더 방지
    )
    
    # Engine 설정 (Dinomaly 특화 - 학습 필요)
    engine_kwargs = {
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": [0] if torch.cuda.is_available() else 1,
        "logger": tb_logger,
        "callbacks": callbacks,
        "enable_checkpointing": True,
        "log_every_n_steps": 10,
        "enable_model_summary": True,
        "num_sanity_val_steps": 0,
        "default_root_dir": results_base_dir
    }
    
    # Dinomaly는 max_steps 기반 학습을 권장 (소스코드 기본값)
    if 'max_steps' in config:
        engine_kwargs["max_steps"] = config["max_steps"]
        # Validation을 덜 자주 수행 (train_loss 기반 early stopping이므로)
        # validation은 주로 최종 성능 확인용으로만 사용
        engine_kwargs["val_check_interval"] = min(500, config["max_steps"] // 5)  # 5번만 validation 수행
        engine_kwargs["check_val_every_n_epoch"] = None  # epoch 기반 validation 비활성화
    elif 'max_epochs' in config:
        engine_kwargs["max_epochs"] = config["max_epochs"]
        engine_kwargs["check_val_every_n_epoch"] = 5  # 5 epoch마다 validation
    else:
        # 기본값으로 5000 steps 사용 (Dinomaly 소스코드 기본값)
        engine_kwargs["max_steps"] = 5000
        engine_kwargs["val_check_interval"] = 1000  # 5번만 validation 수행
    
    engine = Engine(**engine_kwargs)
    
    # 학습 설정 출력
    if 'max_steps' in config:
        print(f"   🔧 Engine 설정 완료 - max_steps: {config['max_steps']}")
    elif 'max_epochs' in config:
        print(f"   🔧 Engine 설정 완료 - max_epochs: {config['max_epochs']}")
    else:
        print(f"   🔧 Engine 설정 완료 - max_steps: 5000 (기본값)")
    print(f"   📁 결과 저장 경로: {results_base_dir}")
    logger.info(f"🔧 Engine 설정 완료 - Dinomaly 학습 기반 모델")
    logger.info(f"📁 결과 저장 경로: {results_base_dir}")
    
    # 모델 훈련 (학습 기반 모델)
    print(f"   🎯 Dinomaly 훈련 시작...")
    logger.info("🎯 Dinomaly 훈련 시작...")
    
    engine.fit(
        model=model,
        datamodule=datamodule
    )
    
    best_checkpoint = checkpoint_callback.best_model_path
    print(f"   🏆 Best Checkpoint: {best_checkpoint}")
    print(f"   ✅ Dinomaly 훈련 완료!")
    logger.info(f"🏆 Best Checkpoint: {best_checkpoint}")
    logger.info("✅ Dinomaly 훈련 완료!")
    
    return model, engine, best_checkpoint


def evaluate_all_domains_model(
    model: Dinomaly,
    engine: Engine,
    datamodule: AllDomainsHDMAPDataModule,
    checkpoint_path: str = None
) -> dict:
    """All domains 모델 성능 평가."""
    
    print(f"\n📊 All Domains 성능 평가")
    
    # 체크포인트에서 모델 로드 (지정된 경우)
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"   📂 체크포인트 로드: {checkpoint_path}")
        model = Dinomaly.load_from_checkpoint(checkpoint_path)
    
    # 테스트 수행
    print(f"   🧪 테스트 수행 중...")
    test_results = engine.test(model=model, datamodule=datamodule)
    
    if test_results and len(test_results) > 0:
        result_metrics = test_results[0]
        print(f"   ✅ All Domains 성능:")
        print(f"      • Image AUROC: {result_metrics.get('test_image_AUROC', 0.0):.4f}")
        print(f"      • Image F1-Score: {result_metrics.get('test_image_F1Score', 0.0):.4f}")
        return result_metrics
    else:
        print(f"   ⚠️ 테스트 결과 없음")
        return {}


def run_single_dinomaly_experiment(
    condition: dict,
    log_dir: str = None
) -> dict:
    """단일 Dinomaly 실험 조건에 대한 실험 수행."""
    
    # config에서 설정 가져오기
    config = condition["config"]
    
    # 각 실험마다 고유한 results 경로 생성
    from datetime import datetime
    # run 스크립트에서 전달받은 log_dir 사용
    if log_dir:
        # run 스크립트에서 호출된 경우: 기존 timestamp 폴더 재사용
        base_timestamp_dir = log_dir
        timestamp_for_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = f"{condition['name']}_{timestamp_for_folder}"
    else:
        # 직접 호출된 경우: 새로운 timestamp 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_timestamp_dir = f"results/dinomaly/{timestamp}"
        experiment_folder = f"{condition['name']}_{timestamp}"
    
    results_base_dir = f"{base_timestamp_dir}/AllDomainsHDMAP/dinomaly/{experiment_folder}"
    
    # 실험 이름 생성
    experiment_name = "all_domains"
    
    print(f"\n{'='*80}")
    print(f"🔬 Dinomaly 실험 조건: {condition['name']}")
    print(f"📝 설명: {condition['description']}")
    print(f"{'='*80}")
    
    try:
        # GPU 메모리 정리
        cleanup_gpu_memory()
        
        # DataModule 생성
        all_datamodule = AllDomainsHDMAPDataModule(
            train_batch_size=config["batch_size"],
            eval_batch_size=config["batch_size"]
        )
        all_datamodule.setup()
        
        print(f"   📊 AllDomainsHDMAPDataModule 설정 완료")
        print(f"      • Batch Size: {config['batch_size']}")
        print(f"      • Image Size: {config['image_size']}")
        
        # 모델 훈련
        trained_model, engine, best_checkpoint = train_dinomaly_model_all_domains(
            datamodule=all_datamodule,
            config=condition["config"],
            results_base_dir=results_base_dir,
            logger=logging.getLogger(__name__)
        )
        
        # All Domains 성능 평가
        print(f"\n📊 All Domains 성능 평가 - {condition['name']}")
        all_domains_results = evaluate_all_domains_model(
            model=trained_model,
            engine=engine,
            datamodule=all_datamodule,
            checkpoint_path=best_checkpoint
        )
        
        # 학습 과정 정보 추출
        training_info = extract_training_info(engine)
        
        # JSON 저장을 위해 호환되는 형식으로 결과 변환
        all_domains_results_compat = {}
        if all_domains_results and 'test_image_AUROC' in all_domains_results:
            all_domains_results_compat = {
                "test_image_AUROC": all_domains_results['test_image_AUROC'],
                "test_image_F1Score": all_domains_results.get('test_image_F1Score', 0.0)
            }
        
        # 실험 결과 정리
        experiment_result = {
            "condition": condition,
            "experiment_name": experiment_name,
            "all_domains_results": all_domains_results_compat,
            "best_checkpoint": best_checkpoint,
            "training_info": training_info,
            "status": "success",
            "experiment_path": results_base_dir
        }
        
        # 결과 저장 디렉토리 찾기
        results_save_dir = Path(results_base_dir)
        tensorboard_dir = results_save_dir / "tensorboard_logs"
        if not tensorboard_dir.exists():
            # TensorBoard 로그가 실제로 저장된 위치 찾기
            for path in results_save_dir.rglob("tensorboard_logs"):
                if path.is_dir():
                    tensorboard_dir = path
                    break
        
        # 각 실험의 tensorboard_logs 폴더에 JSON 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"result_{condition['name']}_{timestamp}.json"
        result_path = tensorboard_dir / result_filename
        
        try:
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(experiment_result, f, indent=2, ensure_ascii=False)
            print(f"📄 실험 결과 JSON 저장: {result_path}")
        except Exception as e:
            print(f"⚠️  JSON 저장 실패: {e}")
        
        print(f"\n✅ 실험 완료: {condition['name']}")
        
        return experiment_result
        
    except Exception as e:
        print(f"❌ 실험 실패 - {condition['name']}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "condition": condition,
            "experiment_name": experiment_name,
            "status": "failed",
            "error": str(e),
            "experiment_path": None  # 실패 시에는 경로 없음
        }
    finally:
        # 메모리 정리
        cleanup_gpu_memory()

def main():
    """Dinomaly 실험 메인 함수."""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="Dinomaly All Domains 실험")
    parser.add_argument("--gpu-id", type=str, help="사용할 GPU ID")
    parser.add_argument("--experiment-id", type=int, help="실험 조건 ID (0부터 시작)")
    parser.add_argument("--log-dir", type=str, help="로그 저장 디렉토리")
    parser.add_argument("--get-experiment-count", action="store_true", help="실험 조건 개수만 반환")
    
    args = parser.parse_args()
    
    # 실험 조건 개수만 반환하는 경우
    if args.get_experiment_count:
        print(len(EXPERIMENT_CONDITIONS))
        return
    
    # 필수 인자 검증
    if not args.gpu_id or args.experiment_id is None or not args.log_dir:
        parser.error("--gpu-id, --experiment-id, --log-dir는 필수 인자입니다 (--get-experiment-count 제외)")
    
    # GPU 설정 및 실험 조건 검증
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    if args.experiment_id >= len(EXPERIMENT_CONDITIONS):
        print(f"❌ 잘못된 실험 ID: {args.experiment_id} (최대: {len(EXPERIMENT_CONDITIONS)-1})")
        return
    
    condition = EXPERIMENT_CONDITIONS[args.experiment_id]
    
    print("="*80)
    print(f"🚀 Dinomaly All Domains 실험 (GPU {args.gpu_id}): {condition['name']}")
    print("="*80)
    
    # 로그 설정
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"dinomaly_all_domains_experiment_{timestamp}.log"
    logger = setup_experiment_logging(str(log_path), f"dinomaly_all_domains_{condition['name']}")
    
    # GPU 메모리 정리
    cleanup_gpu_memory()
    
    try:
        # 실험 정보 로깅
        logger.info("="*80)
        logger.info(f"🚀 Dinomaly All Domains 실험 시작: {condition['name']}")
        logger.info(f"GPU ID: {args.gpu_id} | 실험 ID: {args.experiment_id}")
        logger.info(f"설명: {condition['description']}")
        logger.info("="*80)
        
        # 실험 수행
        result = run_single_dinomaly_experiment(
            condition=condition,
            log_dir=args.log_dir
        )
        
        # 결과 저장
        result_filename = f"result_exp_{args.experiment_id:02d}_{condition['name']}_gpu{args.gpu_id}_{timestamp}.json"
        save_experiment_results(result, result_filename, log_dir, logger)
        
        logger.info("✅ 실험 완료!")
        
    except Exception as e:
        logger.error(f"❌ 실험 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        cleanup_gpu_memory()
        logger.info("🧹 GPU 메모리 정리 완료")


if __name__ == "__main__":
    main()