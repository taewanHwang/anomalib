#!/usr/bin/env python3
"""
BaseAnomalyTrainer - 통합 Anomaly Detection 모델 훈련을 위한 베이스 클래스

이 모듈은 모든 anomaly detection 모델의 훈련을 통합 관리합니다.

지원 모델:
- DRAEM: Reconstruction + Anomaly Detection
- DRAEM CutPaste Clf: DRAEM with CutPaste augmentation + CNN classification
- Dinomaly: Vision Transformer 기반 anomaly detection with DINOv2
- PatchCore: Memory bank 기반 few-shot anomaly detection
"""

import torch
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# 공통 유틸리티 함수들 import
sys.path.append(str(Path(__file__).parent.parent))
from experiment_utils import (
    cleanup_gpu_memory,
    setup_experiment_logging,
    extract_training_info,
    save_experiment_results,
    create_experiment_visualization,
    create_single_domain_datamodule,
    analyze_test_data_distribution,
    unified_model_evaluation
)

# 모델별 imports
from anomalib.models.image.draem import Draem
from anomalib.models.image.draem_cutpaste_clf import DraemCutPasteClf
from anomalib.models.image import Dinomaly, Patchcore
from anomalib.engine import Engine
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from anomalib.metrics import AUROC, Evaluator



class BaseAnomalyTrainer:
    """통합 Anomaly Detection 모델 훈련을 위한 베이스 클래스"""
    
    def __init__(self, config: Dict[str, Any], experiment_name: str, session_timestamp: str, experiment_dir: str = None):
        """
        Args:
            config: 실험 설정 딕셔너리 (model_type 포함)
            experiment_name: 실험 이름
            session_timestamp: 전체 세션의 timestamp
            experiment_dir: 외부에서 지정한 실험 디렉터리 (선택적)
        """
        self.config = config
        self.experiment_name = experiment_name
        self.session_timestamp = session_timestamp
        self.model_type = config.get("model_type", "").lower()
        self.external_experiment_dir = experiment_dir
        self.setup_paths()
        
    def setup_paths(self):
        """실험 경로 설정"""
        if self.external_experiment_dir:
            # bash 스크립트에서 전달받은 디렉터리 사용
            self.experiment_dir = Path(self.external_experiment_dir)
            self.results_dir = self.experiment_dir.parent
        else:
            # 호환성을 위한 기본 방식 (단독 실행 시)
            self.results_dir = Path("results") / self.session_timestamp
            self.experiment_dir = self.results_dir / f"{self.experiment_name}_{self.session_timestamp}"
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
    def create_model(self):
        """Factory pattern으로 모델 생성"""
        if self.model_type == "draem":
            return self._create_draem_model()
        elif self.model_type == "draem_cutpaste_clf":
            return self._create_draem_cutpaste_clf_model()
        elif self.model_type == "dinomaly":
            return self._create_dinomaly_model()
        elif self.model_type == "patchcore":
            return self._create_patchcore_model()
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")
    
    def _create_draem_model(self):
        """DRAEM 모델 생성"""
        # 명시적으로 test_image_AUROC 메트릭 설정
        val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
        test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
        evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])
        
        # DRAEM 모델 생성
        model = Draem(evaluator=evaluator)
        
        # 학습 설정을 _training_config에 저장 (configure_optimizers에서 사용됨)
        model._training_config = {
            'learning_rate': self.config["learning_rate"],
            'optimizer': self.config["optimizer"],
            'weight_decay': self.config["weight_decay"],
        }
        
        return model

    def _create_draem_cutpaste_clf_model(self):
        """DRAEM CutPaste Classification 모델 생성"""
        # 명시적으로 test_image_AUROC 메트릭 설정
        val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
        test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
        evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])

        # 모델 파라미터 설정 - config에서만 값 할당
        model_params = {
            'evaluator': evaluator,
            'sspcab': self.config["sspcab"],
            'image_size': tuple(self.config["image_size"]),
            'severity_dropout': self.config["severity_dropout"],
            'cut_w_range': tuple(self.config["cut_w_range"]),
            'cut_h_range': tuple(self.config["cut_h_range"]),
            'a_fault_start': self.config["a_fault_start"],
            'a_fault_range_end': self.config["a_fault_range_end"],
            'augment_probability': self.config["augment_probability"],
            'norm': self.config["norm"],
            'clf_weight': self.config["clf_weight"],
        }

        # DraemCutPasteClf 모델 생성
        model = DraemCutPasteClf(**model_params)

        # 학습 설정을 _training_config에 저장 (configure_optimizers에서 사용됨)
        model._training_config = {
            'learning_rate': self.config["learning_rate"],
            'optimizer': self.config["optimizer"],
            'weight_decay': self.config["weight_decay"],
        }

        return model

    def _create_dinomaly_model(self):
        """Dinomaly 모델 생성"""
        # Dinomaly는 기본 evaluator를 사용하여 trainable 파라미터 설정 문제 회피
        return Dinomaly(
            encoder_name=self.config["encoder_name"],
            target_layers=self.config["target_layers"],
            bottleneck_dropout=self.config["bottleneck_dropout"],
            decoder_depth=self.config["decoder_depth"],
            remove_class_token=self.config["remove_class_token"],
            evaluator=True  # 기본 evaluator 사용
        )
    
    def _create_patchcore_model(self):
        """Patchcore 모델 생성"""
        return Patchcore(
            backbone=self.config["backbone"],
            layers=self.config["layers"],
            pre_trained=self.config["pre_trained"],
            coreset_sampling_ratio=self.config["coreset_sampling_ratio"],
            num_neighbors=self.config["num_neighbors"]
        )
    
    
    def create_datamodule(self):
        """모든 모델에 공통으로 사용할 데이터 모듈 생성"""
        # 도메인 설정 - 모델별로 다른 필드명 처리
        domain = self.config.get("source_domain") or self.config.get("domain")
        if not domain:
            raise ValueError("config에서 'source_domain' 또는 'domain' 필드를 찾을 수 없습니다")
        
        # dataset_root 필수 체크
        dataset_root = self.config.get("dataset_root")
        if not dataset_root:
            raise ValueError("config에서 'dataset_root' 필드를 찾을 수 없습니다")
        
        # target_size 설정 (list -> tuple 변환)
        target_size = self.config.get("target_size")
        if target_size and isinstance(target_size, list) and len(target_size) == 2:
            target_size = tuple(target_size)
        elif target_size:
            raise ValueError("target_size는 [height, width] 형태의 리스트여야 합니다")
            
        return create_single_domain_datamodule(
            domain=domain,
            dataset_root=dataset_root,
            batch_size=self.config["batch_size"],
            target_size=target_size,
            resize_method=self.config.get("resize_method", "resize"),
            val_split_ratio=self.config["val_split_ratio"],
            num_workers=self.config["num_workers"],
            seed=self.config["seed"]
        )
    
    def create_callbacks(self):
        """콜백 설정 - 모델별 적절한 early stopping 메트릭 사용"""
        callbacks = []
        
        # 모델별 EarlyStopping 설정
        if self.model_type == "patchcore":
            # PatchCore는 단일 epoch 훈련이므로 EarlyStopping과 ModelCheckpoint 모두 불필요
            # Engine에서 자동으로 ModelCheckpoint를 추가하므로 별도 추가하지 않음
            print("   ℹ️ PatchCore: EarlyStopping 및 ModelCheckpoint 비활성화 (단일 epoch 훈련)")
            return []  # 빈 콜백 리스트 반환
            
        else:
            # 모델별로 다른 EarlyStopping monitor 설정
            if self.model_type in ["draem", "draem_cutpaste_clf"]:
                # DRAEM: val_image_AUROC 기반 EarlyStopping (높을수록 좋음)
                monitor_metric = "val_image_AUROC"
                monitor_mode = "max"
                print(f"   ℹ️ {self.model_type.upper()}: EarlyStopping 활성화 (val_image_AUROC 모니터링)")
            else:
                # Dinomaly: val_loss 기반 EarlyStopping
                monitor_metric = "val_loss"
                monitor_mode = "min"
                print(f"   ℹ️ {self.model_type.upper()}: EarlyStopping 활성화 (val_loss 모니터링)")
            
            early_stopping = EarlyStopping(
                monitor=monitor_metric,
                patience=self.config["early_stopping_patience"],
                mode=monitor_mode,
                verbose=True
            )
            callbacks.append(early_stopping)
            
            # Model Checkpoint
            domain = self.config.get("source_domain") or self.config.get("domain")
            
            if self.model_type in ["draem", "draem_cutpaste_clf"]:
                checkpoint = ModelCheckpoint(
                    filename=f"{self.model_type}_single_domain_{domain}_" + "{epoch:02d}_{val_image_AUROC:.4f}",
                    monitor="val_image_AUROC",
                    mode="max",
                    save_top_k=1,
                    verbose=True
                )
            else:
                checkpoint = ModelCheckpoint(
                    filename=f"{self.model_type}_single_domain_{domain}_" + "{epoch:02d}_{val_loss:.4f}",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                    verbose=True
                )
            callbacks.append(checkpoint)
        
        return callbacks
    
    def configure_optimizer(self, model):
        """옵티마이저 설정 - 모든 모델 공통"""
        # PatchCore는 옵티마이저가 필요하지 않음
        if self.model_type == "patchcore":
            return
                
    def train_model(self, model, datamodule, logger) -> Tuple[Any, Engine, str]:
        """모델 훈련 수행"""
        print(f"\n🚀 {self.model_type.upper()} 모델 훈련 시작")
        logger.info(f"🚀 {self.model_type.upper()} 모델 훈련 시작")
        
        # Config 설정 출력
        print(f"   🔧 Config 설정:")
        print(f"      Model Type: {self.model_type}")
        if self.model_type != "patchcore":
            print(f"      Max Epochs: {self.config['max_epochs']}")
            print(f"      Learning Rate: {self.config['learning_rate']}")
            print(f"      Early Stopping Patience: {self.config['early_stopping_patience']}")
        print(f"      Batch Size: {self.config['batch_size']}")
        
        # 옵티마이저 설정
        self.configure_optimizer(model)
        
        # 콜백 설정
        callbacks = self.create_callbacks()
        
        # TensorBoard 로거 설정
        tb_logger = TensorBoardLogger(
            save_dir=str(self.experiment_dir),
            name="tensorboard_logs",
            version=""
        )
        
        # Engine 설정
        # PatchCore의 경우 max_epochs를 1로 강제 설정
        max_epochs = 1 if self.model_type == "patchcore" else self.config["max_epochs"]
        
        engine_kwargs = {
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": [0] if torch.cuda.is_available() else 1,
            "logger": tb_logger,
            "callbacks": callbacks,
            "enable_checkpointing": True,
            "log_every_n_steps": 10,
            "enable_model_summary": True,
            "default_root_dir": str(self.experiment_dir),
            "max_epochs": max_epochs,
            "check_val_every_n_epoch": 1,
            "num_sanity_val_steps": 0
        }
        
        if self.model_type == "patchcore":
            print(f"   ℹ️ PatchCore: max_epochs 강제 설정 (1 epoch)")
        else:
            print(f"   ℹ️ {self.model_type.upper()}: max_epochs = {max_epochs}")
        
        engine = Engine(**engine_kwargs)
        
        print(f"   🔧 Engine 설정 완료")
        print(f"   📁 결과 저장 경로: {self.experiment_dir}")
        logger.info(f"🔧 Engine 설정 완료")
        logger.info(f"📁 결과 저장 경로: {self.experiment_dir}")
        
        # 모델 훈련
        print(f"   🎯 모델 훈련 시작...")
        logger.info("🎯 모델 훈련 시작...")
        
        engine.fit(model=model, datamodule=datamodule)
        
        # 최고 체크포인트 찾기
        best_checkpoint = ""
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint) and hasattr(callback, 'best_model_path'):
                best_checkpoint = callback.best_model_path
                break
        
        print(f"   🏆 Best Checkpoint: {best_checkpoint}")
        logger.info(f"🏆 Best Checkpoint: {best_checkpoint}")
        
        print(f"   ✅ 모델 훈련 완료!")
        logger.info("✅ 모델 훈련 완료!")
        
        return model, engine, best_checkpoint
    
    def evaluate_model(self, model, datamodule, logger) -> Dict[str, Any]:
        """모델 성능 평가 - 통합된 단일 평가"""
        domain = self.config.get("source_domain") or self.config.get("domain")
        
        print(f"\n📊 {domain} 도메인 성능 평가 시작")
        logger.info(f"📊 {domain} 도메인 성능 평가 시작")
        
        try:
            # 통합된 성능 평가 수행
            evaluation_metrics = unified_model_evaluation(
                model, datamodule, self.experiment_dir, self.experiment_name, self.model_type, logger
            )
            
            if evaluation_metrics:
                # 결과 정리
                results = {
                    "domain": domain,
                    "image_AUROC": evaluation_metrics["auroc"],
                    "image_F1Score": evaluation_metrics["f1_score"],
                    "training_samples": len(datamodule.train_data),
                    "test_samples": len(datamodule.test_data),
                    "val_samples": len(datamodule.val_data) if datamodule.val_data else 0,
                    "evaluation_metrics": evaluation_metrics
                }
                
                # 최종 결과 출력
                print(f"   🎯 최종 평가 결과:")
                print(f"      📊 AUROC: {evaluation_metrics['auroc']:.4f}")
                print(f"      🎯 Accuracy: {evaluation_metrics['accuracy']:.4f}")
                print(f"      📈 F1-Score: {evaluation_metrics['f1_score']:.4f}")
                print(f"      🔍 Precision: {evaluation_metrics['precision']:.4f}")
                print(f"      📉 Recall: {evaluation_metrics['recall']:.4f}")
                print(f"      🔢 총 샘플: {evaluation_metrics['total_samples']}개")

                # TensorBoard에 test_image_AUROC 로깅
                if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'add_scalar'):
                    logger.experiment.add_scalar(
                        'test_image_AUROC',
                        evaluation_metrics['auroc'],
                        global_step=0
                    )
                    print(f"      📝 TensorBoard에 test_image_AUROC={evaluation_metrics['auroc']:.4f} 로깅 완료")
                
                logger.info(f"✅ {domain} 평가 완료: AUROC={evaluation_metrics['auroc']:.4f}")
                return results
            else:
                results = {"domain": domain, "error": "Evaluation failed"}
                logger.error(f"❌ {domain} 평가 실패")
                return results
                
        except Exception as e:
            print(f"   ❌ 평가 실패: {str(e)}")
            logger.error(f"평가 실패: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"domain": domain, "error": f"Evaluation failed: {str(e)}"}
        
    def save_results(self, results, training_info, best_checkpoint, logger):
        """실험 결과 저장"""
        domain = self.config.get("source_domain") or self.config.get("domain")
        
        experiment_results = {
            "experiment_name": self.experiment_name,
            "description": f"{domain} - {self.model_type.upper()} single domain training",
            "domain": domain,
            "config": self.config,
            "results": results,
            "training_info": training_info,
            "timestamp": datetime.now().isoformat(),
            "checkpoint_path": best_checkpoint,
            "status": "success",
            "condition": {
                "name": self.experiment_name,
                "description": f"{domain} - {self.model_type.upper()} single domain training",
                "config": {
                    "source_domain": domain,
                    **self.config
                }
            },
            "source_results": {
                "test_image_AUROC": results.get("image_AUROC", 0.0),
                "test_image_F1Score": results.get("image_F1Score", 0.0),
                "domain": domain
            },
            "target_results": {}  # Single domain이므로 비워둠
        }
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 실험 루트 디렉토리에 직접 저장 (multi-domain과 동일)
        result_filename = f"result_{timestamp}.json"
        
        results_file = save_experiment_results(
            result=experiment_results,
            result_filename=result_filename,
            log_dir=Path(self.experiment_dir),  # 실험 루트 디렉토리
            logger=logger,
            model_type=self.model_type.upper()
        )
        print(f"📄 실험 결과 저장됨: {results_file}")
        
        return experiment_results
    
    
    def run_experiment(self) -> dict:
        """전체 실험 실행"""
        domain = self.config.get("source_domain") or self.config.get("domain")
        
        print(f"🔬 {self.model_type.upper()} Single Domain 실험: {self.experiment_name}")
        
        try:
            # GPU 메모리 정리 
            cleanup_gpu_memory()
            
            # 로깅 설정
            log_file_path = self.experiment_dir / f"{domain}_single.log"
            logger = setup_experiment_logging(str(log_file_path), self.experiment_name)
            logger.info(f"🚀 {self.model_type.upper()} Single Domain 실험 시작")
            
            # 모델 생성
            model = self.create_model()
            
            # DataModule 생성
            datamodule = self.create_datamodule()
            
            # DataModule 데이터 개수 확인
            print(f"\n📊 DataModule 준비 및 설정 중:")
            datamodule.prepare_data()
            datamodule.setup()
            
            train_size = len(datamodule.train_data) if datamodule.train_data else 0
            test_size = len(datamodule.test_data) if datamodule.test_data else 0
            val_size = len(datamodule.val_data) if datamodule.val_data else 0
            
            print(f"   📈 훈련 데이터: {train_size:,}개 | 📊 테스트 데이터: {test_size:,}개 | 📋 검증 데이터: {val_size:,}개 | 🎯 총 데이터: {train_size + test_size + val_size:,}개")
            
            # 테스트 데이터 라벨 분포 확인 (전체 데이터)
            analyze_test_data_distribution(datamodule, test_size)
            
            # 모델 훈련
            trained_model, engine, best_checkpoint = self.train_model(model, datamodule, logger)
            
            # 성능 평가
            results = self.evaluate_model(trained_model, datamodule, logger)
            
            # 훈련 정보 추출
            training_info = extract_training_info(engine)
            
            # 결과 저장
            experiment_results = self.save_results(results, training_info, best_checkpoint, logger)
                        
            # GPU 메모리 정리
            cleanup_gpu_memory()
            
            print(f"✅ 실험 완료: {self.experiment_name}")
            logger.info(f"✅ 실험 완료: {self.experiment_name}")
            
            return experiment_results
            
        except Exception as e:
            error_msg = f"❌ 실험 실패: {self.experiment_name} - {str(e)}"
            print(error_msg)
            if 'logger' in locals():
                logger.error(error_msg)
            
            cleanup_gpu_memory()
            
            return {
                "experiment_name": self.experiment_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "failed"
            }
    