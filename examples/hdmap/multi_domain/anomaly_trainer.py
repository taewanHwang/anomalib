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
import logging
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from pytorch_lightning.loggers import TensorBoardLogger

# Anomalib imports
from anomalib.models.image.draem import Draem
from anomalib.models.image.dinomaly import Dinomaly
from anomalib.models.image.patchcore import Patchcore
from anomalib.engine import Engine
from anomalib.metrics import AUROC, Evaluator
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# Experiment utilities import
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from experiment_utils import (
    cleanup_gpu_memory,
    create_multi_domain_datamodule,
    evaluate_source_domain,
    evaluate_target_domains,
    extract_training_info,
    save_experiment_results,
    analyze_experiment_results
)


class MultiDomainAnomalyTrainer:
    """Multi-Domain Anomaly Detection 전용 훈련 클래스"""

    def __init__(self, config: Dict[str, Any], experiment_name: str, session_timestamp: str, experiment_dir: str = None):
        """
        Args:
            config: 실험 설정 딕셔너리 (model_type, source_domain, target_domains 포함)
            experiment_name: 실험 이름
            session_timestamp: 세션 timestamp
            experiment_dir: 실험 디렉터리 경로 (선택적)
        """
        self.config = config
        self.experiment_name = experiment_name
        self.session_timestamp = session_timestamp
        self.external_experiment_dir = experiment_dir

        # Multi-domain 설정 추출
        self.model_type = config["model_type"].lower()
        self.source_domain = config["source_domain"]
        self.target_domains = config["target_domains"]

        # 경로 설정
        self.setup_paths()

        # 로거 설정
        self.logger = logging.getLogger(f"multi_domain_{self.model_type}")

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
        """Factory pattern으로 모델 생성 (multi-domain 최적화)"""
        if self.model_type == "draem":
            return self._create_draem_model()
        elif self.model_type == "draem_cutpaste":
            return self._create_draem_cutpaste_model()
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
            'max_epochs': self.config["max_epochs"],
            'scheduler': self.config.get("scheduler", None),  # 스케줄러 설정 (선택사항)
        }

        return model
    
    def _create_draem_cutpaste_model(self):
        """DRAEM CutPaste 모델 생성"""
        # 명시적으로 test_image_AUROC 메트릭 설정
        val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
        test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
        evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])

        # 모델 파라미터 설정
        model_params = {
            'evaluator': evaluator,
            'enable_sspcab': self.config.get("sspcab", False),
            'cut_w_range': tuple(self.config["cut_w_range"]),
            'cut_h_range': tuple(self.config["cut_h_range"]),
            'a_fault_start': self.config["a_fault_start"],
            'a_fault_range_end': self.config["a_fault_range_end"],
            'augment_probability': self.config["augment_probability"],
        }

        # DraemCutPaste 모델 생성
        from anomalib.models.image.draem_cutpaste import DraemCutPaste
        model = DraemCutPaste(**model_params)

        # 학습 설정을 _training_config에 저장 (configure_optimizers에서 사용됨)
        model._training_config = {
            'learning_rate': self.config["learning_rate"],
            'optimizer': self.config["optimizer"],
            'weight_decay': self.config["weight_decay"],
            'max_epochs': self.config["max_epochs"],
            'scheduler': self.config.get("scheduler", None),  # 스케줄러 설정 (선택사항)
        }

        return model
    
    def _create_dinomaly_model(self):
        """Dinomaly 모델 생성"""
        # Dinomaly는 기본 evaluator를 사용하여 trainable 파라미터 설정 문제 회피
        model = Dinomaly(
            encoder_name=self.config["encoder_name"],
            target_layers=self.config["target_layers"],
            bottleneck_dropout=self.config["bottleneck_dropout"],
            decoder_depth=self.config["decoder_depth"],
            remove_class_token=self.config["remove_class_token"],
            evaluator=True  # 기본 evaluator 사용
        )
        # 학습 설정을 _training_config에 저장 (configure_optimizers에서 사용됨)
        model._training_config = {
            'learning_rate': self.config["learning_rate"],
            'weight_decay': self.config["weight_decay"],
        }
        return model
    
    def _create_patchcore_model(self):
        """Patchcore 모델 생성"""
        return Patchcore(
            backbone=self.config["backbone"],
            layers=self.config["layers"],
            pre_trained=self.config["pre_trained"],
            coreset_sampling_ratio=self.config["coreset_sampling_ratio"],
            num_neighbors=self.config["num_neighbors"]
        )
    
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
            'image_size': tuple(self.config["target_size"]),
            'severity_dropout': self.config["severity_dropout"],
            'severity_input_channels': self.config["severity_input_channels"],
            'cut_w_range': tuple(self.config["cut_w_range"]),
            'cut_h_range': tuple(self.config["cut_h_range"]),
            'a_fault_start': self.config["a_fault_start"],
            'a_fault_range_end': self.config["a_fault_range_end"],
            'augment_probability': self.config["augment_probability"],
            'clf_weight': self.config["clf_weight"],
        }

        # DraemCutPasteClf 모델 생성
        from anomalib.models.image.draem_cutpaste_clf import DraemCutPasteClf
        model = DraemCutPasteClf(**model_params)

        # 학습 설정을 _training_config에 저장 (configure_optimizers에서 사용됨)
        model._training_config = {
            'learning_rate': self.config["learning_rate"],
            'optimizer': self.config["optimizer"],
            'weight_decay': self.config["weight_decay"],
            'max_epochs': self.config["max_epochs"],
            'scheduler': self.config.get("scheduler", None),  # 스케줄러 설정 (선택사항)
        }

        return model

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
            if self.model_type in ["draem", "draem_cutpaste", "draem_cutpaste_clf"]:
                # DRAEM: val_loss 기반 EarlyStopping (낮을수록 좋음)
                monitor_metric = "val_loss"
                monitor_mode = "min"
                print(f"   ℹ️ {self.model_type.upper()}: EarlyStopping 활성화 (val_loss 모니터링)")
            else:
                # Dinomaly: val_loss 기반 EarlyStopping
                monitor_metric = "val_loss"
                monitor_mode = "min"
                print(f"   ℹ️ {self.model_type.upper()}: EarlyStopping 활성화 (val_loss 모니터링)")

            early_stopping = EarlyStopping(
                monitor=monitor_metric,
                patience=self.config["early_stopping_patience"],
                min_delta=self.config.get("early_stopping_min_delta", 0.001),
                mode=monitor_mode,
                verbose=True
            )
            callbacks.append(early_stopping)

            # Model Checkpoint
            if self.model_type in ["draem", "draem_cutpaste_clf"]:
                checkpoint = ModelCheckpoint(
                    filename=f"{self.model_type}_multi_domain_{self.source_domain}_to_targets_" + "{epoch:02d}_{val_loss:.4f}",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                    verbose=True
                )
            else:
                checkpoint = ModelCheckpoint(
                    filename=f"{self.model_type}_multi_domain_{self.source_domain}_to_targets_" + "{epoch:02d}_{val_loss:.4f}",
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

    def create_datamodule(self):
        """MultiDomainHDMAPDataModule 생성"""
        # dataset_root 상대 경로 처리
        dataset_root = self.config["dataset_root"]
        from pathlib import Path
        dataset_path = Path(dataset_root)
        if not dataset_path.is_absolute():
            # 프로젝트 루트 찾기 (anomalib 디렉토리 기준)
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent.parent  # 4단계 상위 = anomalib/
            dataset_root = str(project_root / dataset_root)
            print(f"   📁 상대 경로를 절대 경로로 변환: {dataset_root}")
        
        return create_multi_domain_datamodule(
            source_domain=self.source_domain,
            target_domains=self.target_domains,
            dataset_root=dataset_root,
            batch_size=self.config["batch_size"],
            image_size=self.config["target_size"],
            resize_method=self.config["resize_method"],
            num_workers=self.config["num_workers"],
            seed=self.config["seed"],
            verbose=True
        )
    
    def train_model(self, model, datamodule, logger) -> tuple:
        """모델 훈련 수행"""
        print(f"\n🚀 {self.model_type.upper()} Multi-Domain 모델 훈련 시작")
        logger.info(f"🚀 {self.model_type.upper()} Multi-Domain 모델 훈련 시작")

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
        self.tb_logger = TensorBoardLogger(
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
            "logger": self.tb_logger,
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

        # Best checkpoint 로드 (PatchCore 제외)
        if best_checkpoint and os.path.exists(best_checkpoint) and self.model_type != "patchcore":
            print(f"   📂 Best checkpoint 로드 중...")
            checkpoint = torch.load(best_checkpoint, map_location='cuda' if torch.cuda.is_available() else 'cpu')

            # state_dict 로드
            model.load_state_dict(checkpoint['state_dict'])

            print(f"   ✅ Best checkpoint 로드 완료!")
            logger.info(f"✅ Best checkpoint 로드 완료: {best_checkpoint}")
        elif self.model_type == "patchcore":
            print(f"   ℹ️ PatchCore: Best checkpoint 로드 건너뜀 (단일 epoch 모델)")
        else:
            print(f"   ⚠️ Best checkpoint 파일을 찾을 수 없음: {best_checkpoint}")
            logger.warning(f"Best checkpoint 파일을 찾을 수 없음: {best_checkpoint}")

        print(f"   ✅ 모델 훈련 완료!")
        logger.info("✅ 모델 훈련 완료!")

        return model, engine, best_checkpoint
    
    def run_multi_domain_evaluation(self, model, datamodule, logger):
        """Multi-domain 평가 수행 (source + targets)"""
        print(f"\n📊 Multi-Domain 평가 시작")
        logger.info("📊 Multi-Domain 평가 시작")

        # 시각화 기본 디렉터리 설정
        visualizations_dir = self.experiment_dir / "visualizations"

        # 1. Source Domain 평가 (validation 역할)
        print(f"📊 Source Domain ({self.source_domain}) 평가 시작")
        logger.info(f"📊 Source Domain ({self.source_domain}) 평가 시작")

        # 소스 도메인 시각화 경로: visualizations/source/domain_A/
        source_viz_dir = visualizations_dir / "source" / self.source_domain

        # Analysis 디렉터리 설정
        analysis_dir = visualizations_dir.parent / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        source_results = evaluate_source_domain(
            model=model,
            datamodule=datamodule,
            visualization_dir=source_viz_dir,
            model_type=self.model_type,
            max_visualization_batches=self.config.get("max_visualization_batches", 5),
            verbose=True,
            analysis_dir=analysis_dir
        )

        # 2. Target Domains 평가 (test 역할)
        print(f"\n🎯 Target Domains {self.target_domains} 평가 시작")
        logger.info(f"🎯 Target Domains {self.target_domains} 평가 시작")

        # 타겟 도메인 시각화 경로: visualizations/target/
        target_viz_base_dir = visualizations_dir / "target"

        # Source optimal threshold 추출 (analysis가 활성화된 경우)
        source_optimal_threshold = source_results.get('optimal_threshold', None)
        
        target_results = evaluate_target_domains(
            model=model,
            datamodule=datamodule,
            visualization_base_dir=target_viz_base_dir,
            model_type=self.model_type,
            max_visualization_batches=self.config.get("max_visualization_batches", 5),
            verbose=True,
            analysis_base_dir=analysis_dir,
            source_optimal_threshold=source_optimal_threshold
        )

        return source_results, target_results

    def save_results(self, source_results, target_results, training_info, best_checkpoint, logger):
        """실험 결과 저장"""
        experiment_results = {
            "experiment_name": self.experiment_name,
            "description": f"{self.source_domain} to {self.target_domains} - {self.model_type.upper()} multi domain training",
            "source_domain": self.source_domain,
            "target_domains": self.target_domains,
            "config": self.config,
            "source_results": source_results,
            "target_results": target_results,
            "training_info": training_info,
            "timestamp": datetime.now().isoformat(),
            "checkpoint_path": best_checkpoint,
            "status": "success",
            "condition": {
                "name": self.experiment_name,
                "description": f"{self.source_domain} to {self.target_domains} - {self.model_type.upper()} multi domain training",
                "config": {
                    "source_domain": self.source_domain,
                    "target_domains": self.target_domains,
                    **self.config
                }
            }
        }

        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 실험 루트 디렉토리에 직접 저장
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
            
            # 로깅 설정
            from experiment_utils import setup_experiment_logging
            log_file_path = self.experiment_dir / f"{self.source_domain}_multi_domain.log"
            logger = setup_experiment_logging(str(log_file_path), self.experiment_name)
            logger.info(f"🚀 Multi-Domain 실험 시작: {self.experiment_name}")

            # 모델 훈련
            model, engine, best_checkpoint = self.train_model(model, datamodule, logger)

            # Multi-domain 평가
            source_results, target_results = self.run_multi_domain_evaluation(
                model, datamodule, logger
            )
            
            # 훈련 정보 추출
            training_info = extract_training_info(engine)
            
            # 결과 분석 및 저장
            analysis_path = self.experiment_dir / "analysis_results.json"
            analyze_experiment_results(
                source_results=source_results,
                target_results=target_results,
                training_info=training_info,
                experiment_config=self.config,
                save_path=analysis_path,
                verbose=True
            )

            # 실험 결과 저장
            experiment_result = self.save_results(
                source_results, target_results, training_info, best_checkpoint, logger
            )
            
            # GPU 메모리 정리
            cleanup_gpu_memory()
            
            print(f"\n✅ Multi-Domain 실험 완료: {self.experiment_name}")
            logger.info(f"✅ Multi-Domain 실험 완료: {self.experiment_name}")
            
            return experiment_result
            
        except Exception as e:
            error_msg = f"Multi-Domain 실험 실패: {e}"
            print(f"\n❌ {error_msg}")
            if 'logger' in locals():
                logger.error(f"❌ {error_msg}")
            else:
                self.logger.error(f"❌ {error_msg}")
            
            # GPU 메모리 정리 (실패 시에도)
            cleanup_gpu_memory()
            
            return {
                "status": "failed",
                "error": str(e),
                "experiment_name": self.experiment_name,
                "config": self.config
            }