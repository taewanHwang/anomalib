#!/usr/bin/env python3
"""
Dinomaly Single Domain Training Script for HDMAP Dataset

이 스크립트는 HDMAP 데이터셋에서 Dinomaly 모델의 단일 도메인 실험을 수행합니다.
Domain A에서만 학습하고 평가하여 single domain 성능을 측정합니다.

사용법:
    python examples/hdmap/single_domain/dinomaly-training.py
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from anomalib import LearningType
from anomalib.data.datamodules.image.hdmap import HDMAPDataModule
from anomalib.engine import Engine
from anomalib.metrics import AUROC, Evaluator
from anomalib.models.image import Dinomaly

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DinomlaySingleDomainTrainer:
    """Dinomaly 단일 도메인 훈련을 위한 클래스"""
    
    def __init__(self, config: Dict[str, Any], experiment_name: str, session_timestamp: str):
        """
        Args:
            config: 실험 설정 딕셔너리
            experiment_name: 실험 이름
            session_timestamp: 전체 세션의 timestamp
        """
        self.config = config
        self.experiment_name = experiment_name
        self.session_timestamp = session_timestamp
        self.setup_paths()
        
    def setup_paths(self):
        """실험 경로 설정 (DRAEM과 동일한 구조)"""
        # 결과 저장 경로 (SingleDomainHDMAP 구조 사용, 하나의 timestamp 사용)
        self.results_dir = Path("results/dinomaly") / self.session_timestamp / "SingleDomainHDMAP" / "Dinomaly"
        self.experiment_dir = self.results_dir / f"{self.experiment_name}_{self.session_timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 로그 파일 경로
        self.log_file = self.experiment_dir / f"{self.config['source_domain']}_single.log"
        
    def setup_logging(self):
        """로깅 설정"""
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    def create_model(self) -> Dinomaly:
        """Dinomaly 모델 생성"""
        # 명시적으로 test_image_AUROC 메트릭 설정 (DRAEM-SevNet 교훈 적용)
        val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
        test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
        evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])
        
        model = Dinomaly(
            encoder_name=self.config["encoder_name"],
            target_layers=self.config["target_layers"],
            bottleneck_dropout=self.config["bottleneck_dropout"],
            decoder_depth=self.config["decoder_depth"],
            remove_class_token=self.config["remove_class_token"],
            evaluator=evaluator  # 명시적 evaluator 설정
        )
        
        return model
    
    def create_datamodule(self) -> HDMAPDataModule:
        """데이터 모듈 생성"""
        from anomalib.data.utils import ValSplitMode
        
        # 이미지 크기 문자열 (HDMAPDataModule에서는 문자열로 처리)
        image_size = self.config["image_size"]  # "392x392" 형태 그대로 사용
        
        # 데이터셋 경로 설정
        dataset_root = project_root / "datasets" / "HDMAP" / f"1000_8bit_resize_{image_size}"
        
        datamodule = HDMAPDataModule(
            root=str(dataset_root),
            domain=self.config["source_domain"],
            train_batch_size=self.config["batch_size"],
            eval_batch_size=self.config["batch_size"],
            num_workers=4,
            val_split_mode=ValSplitMode.FROM_TRAIN,  # train에서 validation 분할
            val_split_ratio=0.2,  # 20%를 validation으로 사용
            seed=42
        )
        
        return datamodule
    
    def create_callbacks(self):
        """콜백 설정"""
        callbacks = []
        
        # Early Stopping - val_image_AUROC 기반 (Dinomaly에서 사용 가능한 메트릭)
        early_stopping = EarlyStopping(
            monitor="val_image_AUROC",
            patience=self.config["early_stopping_patience"],
            mode="max",  # AUROC는 높을수록 좋음
            verbose=True
        )
        callbacks.append(early_stopping)
        
        # Model Checkpoint
        checkpoint = ModelCheckpoint(
            dirpath=self.experiment_dir / "tensorboard_logs" / "checkpoints",
            filename="dinomaly_single_domain_{source_domain}_epoch={{epoch:02d}}_val_loss={{val_loss:.4f}}".format(
                source_domain=self.config["source_domain"]
            ),
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True
        )
        callbacks.append(checkpoint)
        
        return callbacks
    
    def create_trainer(self, callbacks) -> Trainer:
        """Trainer 생성"""
        # TensorBoard 로거
        logger_tb = TensorBoardLogger(
            save_dir=self.experiment_dir / "tensorboard_logs",
            name="",
            version=""
        )
        
        trainer = Trainer(
            max_epochs=self.config["max_epochs"],
            callbacks=callbacks,
            logger=logger_tb,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            gradient_clip_val=self.config.get("gradient_clip_val", 0.0),
            enable_progress_bar=True,
            enable_model_summary=True,
            num_sanity_val_steps=2,
            # Dinomaly는 step-based 학습이므로 체크포인트 빈도 조정
            val_check_interval=0.25,  # 25% epoch마다 validation
            log_every_n_steps=50
        )
        
        return trainer
    
    def configure_optimizer(self, model: Dinomaly):
        """옵티마이저 설정"""
        optimizer_name = self.config.get("optimizer", "adamw").lower()
        learning_rate = self.config["learning_rate"]
        weight_decay = self.config.get("weight_decay", 0.0001)
        
        # 모델의 configure_optimizers를 override
        def custom_configure_optimizers():
            if optimizer_name == "adamw":
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
            elif optimizer_name == "adam":
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
            elif optimizer_name == "stable_adamw":
                # StableAdamW가 있다면 사용 (없으면 AdamW로 fallback)
                try:
                    from transformers import AdamW as StableAdamW
                    optimizer = StableAdamW(
                        model.parameters(),
                        lr=learning_rate,
                        weight_decay=weight_decay
                    )
                except ImportError:
                    logger.warning("StableAdamW를 찾을 수 없어 AdamW로 대체합니다.")
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=learning_rate,
                        weight_decay=weight_decay
                    )
            else:
                raise ValueError(f"지원하지 않는 옵티마이저: {optimizer_name}")
            
            # 스케줄러 설정
            scheduler_type = self.config.get("scheduler", None)
            if scheduler_type == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.config["max_epochs"],
                    eta_min=learning_rate * 0.01
                )
                return [optimizer], [scheduler]
            else:
                return optimizer
        
        # 모델의 configure_optimizers 메서드를 교체
        model.configure_optimizers = custom_configure_optimizers
    
    def save_results(self, trainer: Trainer, model: Dinomaly, datamodule: HDMAPDataModule):
        """결과 저장"""
        try:
            # 테스트 결과 가져오기
            test_results = trainer.logged_metrics
            
            # AUROC 값 추출 (여러 가능한 키에서 찾기)
            auroc_keys = ["test_image_AUROC", "image_AUROC", "test_AUROC"]
            image_auroc = 0.0
            for key in auroc_keys:
                if key in test_results:
                    image_auroc = float(test_results[key])
                    break
            
            # 결과 딕셔너리 생성
            results = {
                "experiment_name": self.experiment_name,
                "description": f"Domain {self.config['source_domain'].replace('domain_', '')} - Single domain Dinomaly training",
                "domain": self.config["source_domain"],
                "config": self.config,
                "results": {
                    "domain": self.config["source_domain"],
                    "image_AUROC": image_auroc,
                    "pixel_AUROC": test_results.get("test_pixel_AUROC", 0.0),
                    "image_F1Score": test_results.get("test_image_F1Score", 0.0),
                    "pixel_F1Score": test_results.get("test_pixel_F1Score", 0.0),
                    "training_samples": len(datamodule.train_dataloader().dataset) if datamodule.train_dataloader() else 0,
                    "test_samples": len(datamodule.test_dataloader().dataset) if datamodule.test_dataloader() else 0,
                    "val_samples": len(datamodule.val_dataloader().dataset) if datamodule.val_dataloader() else 0
                },
                "training_info": {
                    "max_epochs_configured": self.config["max_epochs"],
                    "last_trained_epoch": trainer.current_epoch,
                    "total_steps": trainer.global_step,
                    "early_stopped": trainer.early_stopping_callback.stopped_epoch > 0 if trainer.early_stopping_callback else False,
                    "early_stop_reason": f"No improvement for {self.config['early_stopping_patience']} epochs" if trainer.early_stopping_callback and trainer.early_stopping_callback.stopped_epoch > 0 else "Training completed",
                    "best_val_loss": float(trainer.early_stopping_callback.best_score) if trainer.early_stopping_callback else trainer.logged_metrics.get("val_loss", 0.0),
                    "f1_threshold_issue": "F1Score는 기본 threshold로 계산됨. AUROC 대비 낮을 수 있음.",
                    "completion_type": "early_stopping" if trainer.early_stopping_callback and trainer.early_stopping_callback.stopped_epoch > 0 else "max_epochs",
                    "completion_description": f"에폭 {trainer.current_epoch}에서 {'early stopping으로 중단' if trainer.early_stopping_callback and trainer.early_stopping_callback.stopped_epoch > 0 else '최대 에폭 완료'}"
                },
                "timestamp": datetime.now().isoformat(),
                "checkpoint_path": str(trainer.checkpoint_callback.best_model_path) if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path else "",
                "status": "success",
                "condition": {
                    "name": self.experiment_name,
                    "description": f"Domain {self.config['source_domain'].replace('domain_', '')} - Single domain Dinomaly training",
                    "config": self.config
                },
                # 단일 도메인이므로 source_results에 동일한 결과 저장
                "source_results": {
                    "test_image_AUROC": image_auroc,
                    "test_pixel_AUROC": test_results.get("test_pixel_AUROC", 0.0),
                    "test_image_F1Score": test_results.get("test_image_F1Score", 0.0),
                    "test_pixel_F1Score": test_results.get("test_pixel_F1Score", 0.0),
                    "domain": self.config["source_domain"]
                },
                "target_results": {}  # 단일 도메인이므로 비워둠
            }
            
            # JSON 파일 저장
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = self.experiment_dir / "tensorboard_logs" / f"result_{timestamp_str}.json"
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"결과 저장됨: {result_file}")
            logger.info(f"Final Image AUROC: {image_auroc:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"결과 저장 중 오류: {e}")
            return None
    
    def run_experiment(self):
        """실험 실행"""
        logger.info(f"실험 시작: {self.experiment_name}")
        logger.info(f"설정: {self.config}")
        
        try:
            # 로깅 설정
            self.setup_logging()
            
            # 모델 생성
            logger.info("Dinomaly 모델 생성 중...")
            model = self.create_model()
            
            # 옵티마이저 설정
            self.configure_optimizer(model)
            
            # 데이터 모듈 생성
            logger.info("데이터 모듈 생성 중...")
            datamodule = self.create_datamodule()
            
            # 콜백 생성
            callbacks = self.create_callbacks()
            
            # Trainer 생성
            logger.info("Trainer 생성 중...")
            trainer = self.create_trainer(callbacks)
            
            # 훈련 시작
            logger.info("훈련 시작...")
            trainer.fit(model, datamodule)
            
            # 테스트
            logger.info("테스트 시작...")
            trainer.test(model, datamodule)
            
            # 결과 저장
            logger.info("결과 저장 중...")
            results = self.save_results(trainer, model, datamodule)
            
            if results:
                logger.info(f"실험 완료: {self.experiment_name}")
                logger.info(f"Image AUROC: {results['results']['image_AUROC']:.4f}")
                return True
            else:
                logger.error("결과 저장 실패")
                return False
                
        except Exception as e:
            logger.error(f"실험 실행 중 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


def load_experiment_conditions(json_file: str):
    """실험 조건 JSON 파일 로드"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['experiment_conditions']


def main():
    """메인 함수"""
    # GPU 사용 가능 여부 확인
    if torch.cuda.is_available():
        logger.info(f"GPU 사용 가능: {torch.cuda.get_device_name()}")
    else:
        logger.info("CPU 사용")
    
    # 전체 세션용 timestamp 생성 (DRAEM과 동일한 구조)
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"세션 Timestamp: {session_timestamp}")
    
    # 실험 조건 로드
    script_dir = Path(__file__).parent
    json_file = script_dir / "dinomaly-exp_condition.json"
    
    if not json_file.exists():
        logger.error(f"실험 조건 파일을 찾을 수 없습니다: {json_file}")
        return
    
    conditions = load_experiment_conditions(json_file)
    logger.info(f"{len(conditions)}개의 실험 조건을 로드했습니다.")
    
    # 모든 실험 실행
    successful_experiments = 0
    total_experiments = len(conditions)
    
    for i, condition in enumerate(conditions, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"실험 {i}/{total_experiments}: {condition['name']}")
        logger.info(f"설명: {condition['description']}")
        logger.info(f"{'='*60}")
        
        trainer = DinomlaySingleDomainTrainer(condition['config'], condition['name'], session_timestamp)
        
        if trainer.run_experiment():
            successful_experiments += 1
            logger.info(f"실험 {condition['name']} 성공")
        else:
            logger.error(f"실험 {condition['name']} 실패")
    
    # 최종 결과
    logger.info(f"\n{'='*60}")
    logger.info(f"모든 실험 완료 - 세션: {session_timestamp}")
    logger.info(f"성공: {successful_experiments}/{total_experiments}")
    logger.info(f"실패: {total_experiments - successful_experiments}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()