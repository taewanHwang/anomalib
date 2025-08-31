#!/usr/bin/env python3
"""Multi-Domain Anomaly Detection 통합 훈련 클래스

이 모듈은 multi-domain anomaly detection 실험을 위한 통합 훈련 클래스를 제공합니다.
Single domain과 달리 source domain에서 훈련하고 multiple target domains에서 평가합니다.

주요 특징:
- MultiDomainHDMAPDataModule 사용
- Source domain test가 validation 역할
- Target domains test가 평가 대상
- Source/Target 시각화 폴더 분리
"""

import os
import json
import logging
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from pytorch_lightning.loggers import TensorBoardLogger

# Anomalib imports
from anomalib.models.image.draem import Draem
from anomalib.models.image.dinomaly import Dinomaly
from anomalib.models.image.patchcore import Patchcore
from anomalib.models.image.draem_sevnet import DraemSevNet
from anomalib.data.datamodules.image.multi_domain_hdmap import MultiDomainHDMAPDataModule
from anomalib.engine import Engine
from anomalib.metrics import AUROC, Evaluator
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# Experiment utilities import
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from experiment_utils import (
    setup_warnings_filter,
    cleanup_gpu_memory,
    create_multi_domain_datamodule,
    evaluate_source_domain,
    evaluate_target_domains,
    extract_training_info,
    organize_source_domain_results,
    save_experiment_results,
    create_common_experiment_result,
    analyze_experiment_results
)


class MultiDomainAnomalyTrainer:
    """Multi-Domain Anomaly Detection 전용 훈련 클래스"""
    
    def __init__(self, config: Dict[str, Any], experiment_name: str, session_timestamp: str, experiment_dir: str):
        """
        Args:
            config: 실험 설정 딕셔너리 (model_type, source_domain, target_domains 포함)
            experiment_name: 실험 이름
            session_timestamp: 세션 timestamp
            experiment_dir: 실험 디렉터리 경로
        """
        self.config = config
        self.experiment_name = experiment_name
        self.session_timestamp = session_timestamp
        self.experiment_dir = Path(experiment_dir)
        
        # Multi-domain 설정 추출
        self.model_type = config["model_type"].lower()
        self.source_domain = config["source_domain"]
        self.target_domains = config["target_domains"]
        
        # 결과 저장 경로 설정 (single domain과 동일한 구조)
        self.results_dir = self.experiment_dir
        
        # 로거 설정
        self.logger = logging.getLogger(f"multi_domain_{self.model_type}")
        
    def create_model(self):
        """Factory pattern으로 모델 생성 (multi-domain 최적화)"""
        if self.model_type == "draem":
            return self._create_draem_model()
        elif self.model_type == "dinomaly":
            return self._create_dinomaly_model()
        elif self.model_type == "patchcore":
            return self._create_patchcore_model()
        elif self.model_type == "draem_sevnet":
            return self._create_draem_sevnet_model()
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")
    
    def _create_draem_model(self):
        """DRAEM 모델 생성 (multi-domain 평가용)"""
        # Multi-domain 평가를 위한 메트릭 설정
        val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
        test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
        evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])
        
        model = Draem(evaluator=evaluator)
        
        # Config를 모델에 저장 (optimizer 설정용)
        if hasattr(model, '_config'):
            model._config = self.config
        else:
            setattr(model, '_training_config', self.config)
            
        return model
    
    def _create_dinomaly_model(self):
        """Dinomaly 모델 생성 (multi-domain 평가용)"""
        # Config에서 Dinomaly 특화 설정 추출
        encoder_name = self.config["encoder_name"]
        target_layers = self.config["target_layers"]
        remove_class_token = self.config["remove_class_token"]
        
        # Multi-domain 평가를 위한 메트릭 설정
        val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
        test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
        evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])
        
        model = Dinomaly(
            encoder_name=encoder_name,
            target_layers=target_layers,
            remove_class_token=remove_class_token,
            evaluator=evaluator
        )
        
        # Config 저장
        setattr(model, '_training_config', self.config)
        return model
    
    def _create_patchcore_model(self):
        """PatchCore 모델 생성 (multi-domain 평가용)"""
        # Config에서 PatchCore 설정 추출
        backbone = self.config["backbone"]
        layers = self.config["layers"]
        coreset_sampling_ratio = self.config["coreset_sampling_ratio"]
        num_neighbors = self.config["num_neighbors"]
        
        # Multi-domain 평가를 위한 메트릭 설정  
        val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
        test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
        evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])
        
        model = Patchcore(
            backbone=backbone,
            layers=layers,
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
            evaluator=evaluator
        )
        
        # Config 저장
        setattr(model, '_training_config', self.config)
        return model
    
    def _create_draem_sevnet_model(self):
        """DRAEM-SevNet 모델 생성 (multi-domain 평가용)"""
        # Config에서 DRAEM-SevNet 설정 추출
        score_combination = self.config["score_combination"]
        severity_loss_type = self.config["severity_loss_type"]
        severity_head_pooling_type = self.config["severity_head_pooling_type"]
        severity_weight = self.config["severity_weight"]
        
        # Multi-domain 평가를 위한 메트릭 설정
        val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
        test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
        evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])
        
        model = DraemSevNet(
            score_combination=score_combination,
            severity_loss_type=severity_loss_type,
            severity_head_pooling_type=severity_head_pooling_type,
            severity_weight=severity_weight,
            evaluator=evaluator
        )
        
        # Config 저장
        setattr(model, '_training_config', self.config)
        return model
    
    def create_datamodule(self):
        """MultiDomainHDMAPDataModule 생성"""
        return create_multi_domain_datamodule(
            datamodule_class=MultiDomainHDMAPDataModule,
            source_domain=self.source_domain,
            target_domains=self.target_domains,
            batch_size=self.config["batch_size"],
            image_size=self.config["image_size"],
            num_workers=self.config["num_workers"],
            validation_strategy="source_test"  # Source test가 validation 역할
        )
    
    def train_model(self, model, datamodule) -> tuple:
        """모델 훈련 수행"""
        print(f"\n🚀 {self.model_type.upper()} Multi-Domain 모델 훈련 시작")
        self.logger.info(f"🚀 {self.model_type.upper()} Multi-Domain 모델 훈련 시작")
        
        # Early stopping 설정 (multi-domain은 val_image_AUROC 기반)
        early_stopping = EarlyStopping(
            monitor="val_image_AUROC",
            patience=self.config["early_stopping_patience"],
            mode="max",  # AUROC는 높을수록 좋음
            verbose=True
        )
        
        # Model checkpoint 설정
        checkpoint_callback = ModelCheckpoint(
            filename=f"{self.model_type}_multi_domain_{self.source_domain}_to_targets_" + "{epoch:02d}_{val_image_AUROC:.4f}",
            monitor="val_image_AUROC",
            mode="max",
            save_top_k=1,
            verbose=True
        )
        
        print(f"   📊 Early Stopping: patience={self.config['early_stopping_patience']}, monitor=val_image_AUROC (max)")
        print(f"   💾 Model Checkpoint: monitor=val_image_AUROC (max), save_top_k=1")
        
        # TensorBoard 로거 설정
        tb_logger = TensorBoardLogger(
            save_dir=str(self.results_dir),
            name="tensorboard_logs", 
            version=""
        )
        
        # Engine 설정 (PatchCore 특수 처리)
        max_epochs = self.config["max_epochs"]
        
        engine_kwargs = {
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": [0] if torch.cuda.is_available() else 1,
            "logger": tb_logger,
            "max_epochs": max_epochs,
            "callbacks": [early_stopping, checkpoint_callback],
            "check_val_every_n_epoch": 1,
            "enable_checkpointing": True,
            "log_every_n_steps": 10,
            "enable_model_summary": True,
            "num_sanity_val_steps": 0,
            "default_root_dir": str(self.results_dir)
        }
        
        # Gradient clipping 설정
        if "gradient_clip_val" in self.config and self.config["gradient_clip_val"] is not None:
            engine_kwargs["gradient_clip_val"] = self.config["gradient_clip_val"]
            print(f"   🔧 Gradient Clipping 설정: {self.config['gradient_clip_val']}")
        
        engine = Engine(**engine_kwargs)
        
        print(f"   🔧 Engine 설정 완료 - max_epochs: {max_epochs}")
        print(f"   📁 결과 저장 경로: {self.results_dir}")
        self.logger.info(f"🔧 Engine 설정 완료 - max_epochs: {max_epochs}")
        
        # 모델 훈련 시작
        print(f"   🎯 모델 훈련 시작...")
        self.logger.info("🎯 모델 훈련 시작...")
        
        if self.model_type == "patchcore":
            # PatchCore는 fit이 아닌 memory bank 구축
            engine.fit(model=model, datamodule=datamodule)
        else:
            # 일반 훈련 모델
            engine.fit(model=model, datamodule=datamodule)
        
        print(f"   ✅ 모델 훈련 완료!")
        self.logger.info("✅ 모델 훈련 완료!")
        
        # 최고 성능 체크포인트 확인
        best_checkpoint = checkpoint_callback.best_model_path
        print(f"   🏆 Best Checkpoint: {best_checkpoint}")
        self.logger.info(f"🏆 Best Checkpoint: {best_checkpoint}")
        
        return model, engine, best_checkpoint
    
    def run_multi_domain_evaluation(self, model, engine, datamodule, best_checkpoint):
        """Multi-domain 평가 수행 (source + targets)"""
        print(f"\n📊 Multi-Domain 평가 시작")
        self.logger.info("📊 Multi-Domain 평가 시작")
        
        # 1. Source Domain 평가 (validation 역할)
        print(f"📊 Source Domain ({self.source_domain}) 평가 시작")
        self.logger.info(f"📊 Source Domain ({self.source_domain}) 평가 시작")
        
        source_results = evaluate_source_domain(
            model=model,
            engine=engine,
            datamodule=datamodule,
            checkpoint_path=best_checkpoint
        )
        
        # 2. Target Domains 평가 (test 역할)
        print(f"\n🎯 Target Domains {self.target_domains} 평가 시작")
        self.logger.info(f"🎯 Target Domains {self.target_domains} 평가 시작")
        
        # TensorBoard 로그 경로 확인
        try:
            if hasattr(engine.trainer, 'logger') and hasattr(engine.trainer.logger, 'log_dir'):
                tensorboard_path = Path(engine.trainer.logger.log_dir)
            else:
                tensorboard_path = self.results_dir / "tensorboard_logs"
        except:
            tensorboard_path = self.results_dir / "tensorboard_logs"
        
        target_results = evaluate_target_domains(
            model=model,
            engine=engine,
            datamodule=datamodule,
            checkpoint_path=best_checkpoint,
            results_base_dir=str(tensorboard_path),
            target_domains=self.target_domains,
            save_samples=True,
            current_version_path=str(tensorboard_path)
        )
        
        # 3. Source/Target 시각화 결과 정리
        try:
            # Source domain 결과 정리
            organize_source_domain_results(
                sevnet_viz_path=str(tensorboard_path),
                results_base_dir=str(tensorboard_path),
                source_domain=self.source_domain
            )
            print(f"   ✅ Source domain 시각화 결과 정리 완료")
        except Exception as e:
            print(f"   ⚠️ Source domain 시각화 정리 중 오류: {e}")
        
        return source_results, target_results
    
    def run_experiment(self) -> Dict[str, Any]:
        """전체 multi-domain 실험 실행"""
        print(f"\n{'='*80}")
        print(f"🧪 Multi-Domain {self.model_type.upper()} 실험 시작: {self.experiment_name}")
        print(f"🌍 Source Domain: {self.source_domain}")
        print(f"🎯 Target Domains: {self.target_domains}")
        print(f"{'='*80}")
        
        self.logger.info(f"🧪 Multi-Domain {self.model_type.upper()} 실험 시작: {self.experiment_name}")
        self.logger.info(f"실험 설정: {self.config}")
        
        try:
            # GPU 메모리 정리
            cleanup_gpu_memory()
            
            # 실험 디렉터리 생성
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            
            # 모델 생성
            model = self.create_model()
            print(f"✅ {self.model_type.upper()} 모델 생성 완료")
            
            # DataModule 생성
            datamodule = self.create_datamodule()
            print(f"✅ Multi-Domain DataModule 생성 완료")
            print(f"   🌍 Source: {self.source_domain}")
            print(f"   🎯 Targets: {self.target_domains}")
            
            # 모델 훈련
            model, engine, best_checkpoint = self.train_model(model, datamodule)
            
            # Multi-domain 평가
            source_results, target_results = self.run_multi_domain_evaluation(
                model, engine, datamodule, best_checkpoint
            )
            
            # 훈련 정보 추출
            training_info = extract_training_info(engine)
            
            # 결과 분석
            analysis = analyze_experiment_results(
                source_results=source_results,
                target_results=target_results,
                training_info=training_info,
                condition={"name": self.experiment_name, "config": self.config},
                model_type=self.model_type.upper()
            )
            
            # 실험 결과 정리
            experiment_result = create_common_experiment_result(
                condition={"name": self.experiment_name, "config": self.config},
                status="success",
                experiment_path=str(self.experiment_dir),
                source_results=source_results,
                target_results=target_results,
                training_info=training_info,
                best_checkpoint=best_checkpoint
            )
            
            # 결과 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"result_{timestamp}.json"
            
            # TensorBoard 로그 경로 확인 후 저장
            try:
                if hasattr(engine.trainer, 'logger') and hasattr(engine.trainer.logger, 'log_dir'):
                    log_dir = Path(engine.trainer.logger.log_dir)
                else:
                    log_dir = self.experiment_dir / "tensorboard_logs"
            except:
                log_dir = self.experiment_dir / "tensorboard_logs"
            
            save_experiment_results(
                result=experiment_result,
                result_filename=result_filename,
                log_dir=log_dir,
                logger=self.logger,
                model_type=self.model_type.upper()
            )
            
            # GPU 메모리 정리
            cleanup_gpu_memory()
            
            print(f"\n✅ Multi-Domain 실험 완료: {self.experiment_name}")
            self.logger.info(f"✅ Multi-Domain 실험 완료: {self.experiment_name}")
            
            return experiment_result
            
        except Exception as e:
            error_msg = f"Multi-Domain 실험 실패: {e}"
            print(f"\n❌ {error_msg}")
            self.logger.error(f"❌ {error_msg}")
            
            # GPU 메모리 정리 (실패 시에도)
            cleanup_gpu_memory()
            
            return create_common_experiment_result(
                condition={"name": self.experiment_name, "config": self.config},
                status="failed",
                error=str(e)
            )