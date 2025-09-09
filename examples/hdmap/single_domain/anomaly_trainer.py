#!/usr/bin/env python3
"""
BaseAnomalyTrainer - 통합 Anomaly Detection 모델 훈련을 위한 베이스 클래스

이 모듈은 모든 anomaly detection 모델의 훈련을 통합 관리합니다.

지원 모델:
- DRAEM: Reconstruction + Anomaly Detection
- Dinomaly: Vision Transformer 기반 anomaly detection with DINOv2
- PatchCore: Memory bank 기반 few-shot anomaly detection  
- DRAEM-SevNet: Selective feature reconstruction
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

# 공통 유틸리티 함수들 import
from experiment_utils import (
    cleanup_gpu_memory,
    setup_experiment_logging,
    extract_training_info,
    save_experiment_results,
    create_experiment_visualization,
    create_single_domain_datamodule,
    save_detailed_test_results,
    plot_roc_curve,
    save_metrics_report,
    plot_score_distributions,
    save_extreme_samples,
    save_experiment_summary
)

# 모델별 imports
from anomalib.models.image.draem import Draem
from anomalib.models.image.draem_sevnet import DraemSevNet
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
        elif self.model_type == "dinomaly":
            return self._create_dinomaly_model()
        elif self.model_type == "patchcore":
            return self._create_patchcore_model()
        elif self.model_type == "draem_sevnet":
            return self._create_draem_sevnet_model()
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")
    
    def _create_draem_model(self):
        """DRAEM 모델 생성"""
        # 명시적으로 test_image_AUROC 메트릭 설정
        val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
        test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
        evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])
        
        return Draem(evaluator=evaluator)
    
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
    
    def _create_draem_sevnet_model(self):
        """DRAEM-SevNet 모델 생성"""
        # 명시적으로 test_image_AUROC 메트릭 설정
        val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
        test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
        evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])
        
        return DraemSevNet(
            severity_head_mode=self.config["severity_head_mode"],
            severity_head_hidden_dim=self.config["severity_head_hidden_dim"],
            score_combination=self.config["score_combination"],
            severity_weight_for_combination=self.config["severity_weight_for_combination"],
            severity_head_pooling_type=self.config["severity_head_pooling_type"],
            severity_head_spatial_size=self.config["severity_head_spatial_size"],
            severity_head_use_spatial_attention=self.config["severity_head_use_spatial_attention"],
            patch_ratio_range=self.config["patch_ratio_range"],
            patch_width_range=self.config["patch_width_range"],
            patch_count=self.config["patch_count"],
            anomaly_probability=self.config["anomaly_probability"],
            severity_weight=self.config["severity_weight"],
            severity_loss_type=self.config["severity_loss_type"],
            severity_max=self.config["severity_max"],
            evaluator=evaluator
        )
    
    def create_datamodule(self):
        """모든 모델에 공통으로 사용할 데이터 모듈 생성"""
        # 도메인 설정 - 모델별로 다른 필드명 처리
        domain = self.config.get("source_domain") or self.config.get("domain")
        if not domain:
            raise ValueError("config에서 'source_domain' 또는 'domain' 필드를 찾을 수 없습니다")
            
        return create_single_domain_datamodule(
            domain=domain,
            batch_size=self.config["batch_size"],
            image_size=self.config["image_size"],
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
            # DRAEM, DRAEM-SevNet, Dinomaly: val_loss 기반 EarlyStopping
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=self.config["early_stopping_patience"],
                mode="min",
                verbose=True
            )
            callbacks.append(early_stopping)
            
            # Model Checkpoint
            domain = self.config.get("source_domain") or self.config.get("domain")
            checkpoint = ModelCheckpoint(
                filename=f"{self.model_type}_single_domain_{domain}_" + "{epoch:02d}_{val_loss:.4f}",
                monitor="val_loss", 
                mode="min",
                save_top_k=1,
                verbose=True
            )
            callbacks.append(checkpoint)
            
            print(f"   ℹ️ {self.model_type.upper()}: EarlyStopping 활성화 (val_loss 모니터링)")
        
        return callbacks
    
    def configure_optimizer(self, model):
        """옵티마이저 설정 - 모든 모델 공통"""
        # PatchCore는 옵티마이저가 필요하지 않음
        if self.model_type == "patchcore":
            return
            
        # DRAEM과 DRAEM-SevNet은 이미 자체 configure_optimizers를 가지고 있음
        # Dinomaly만 기본 설정을 사용
        # 따라서 여기서는 별도 처리 불필요
    
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
    
    def evaluate_model(self, model, engine, datamodule, logger) -> Dict[str, Any]:
        """모델 성능 평가"""
        domain = self.config.get("source_domain") or self.config.get("domain")
        
        print(f"\n📊 {domain} 도메인 성능 평가 시작")
        logger.info(f"📊 {domain} 도메인 성능 평가 시작")
        
        # 🔧 FIX: Lightning Test용 새로운 DataModule 생성 (훈련된 DataModule 재사용 방지)
        print(f"   🆕 Lightning Test 전용 DataModule 생성 중...")
        test_datamodule = self.create_datamodule()
        test_datamodule.prepare_data()
        test_datamodule.setup()
        
        # Lightning Test용 DataModule 데이터 확인
        test_train_size = len(test_datamodule.train_data) if test_datamodule.train_data else 0
        test_test_size = len(test_datamodule.test_data) if test_datamodule.test_data else 0
        test_val_size = len(test_datamodule.val_data) if test_datamodule.val_data else 0
        
        print(f"   📊 Lightning Test DataModule: 훈련={test_train_size}, 테스트={test_test_size}, 검증={test_val_size}")
        logger.info(f"Lightning Test DataModule - 훈련: {test_train_size}, 테스트: {test_test_size}, 검증: {test_val_size}")
        
        # 테스트 수행 (새로운 DataModule 사용)
        test_results = engine.test(model=model, datamodule=test_datamodule)
        
        # Lightning Confusion Matrix 계산
        print(f"   🧮 Lightning Confusion Matrix 계산 중...")
        lightning_confusion_matrix = self._calculate_lightning_confusion_matrix(model, test_datamodule, logger)
        
        # 상세 분석 수행 (모든 모델 타입)
        print(f"   🔬 상세 분석 시작 - 이미지별 예측 점수 추출 ({self.model_type})")
        logger.info(f"🔬 상세 분석 시작 - 이미지별 예측 점수 추출 ({self.model_type})")
        try:
            custom_metrics = self._generate_detailed_analysis(model, test_datamodule, logger)
            print(f"   ✅ 상세 분석 완료")
            logger.info("✅ 상세 분석 완료")
        except Exception as e:
            print(f"   ⚠️ 상세 분석 실패: {str(e)}")
            logger.error(f"상세 분석 실패: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            custom_metrics = None
        
        # 결과 정리 - Custom Analysis 메트릭 우선 사용
        if custom_metrics and custom_metrics is not None:
            # Custom Analysis 메트릭을 메인 결과로 사용
            results = {
                "domain": domain,
                "image_AUROC": custom_metrics["custom_auroc"],
                "image_F1Score": custom_metrics["custom_f1_score"],
                "training_samples": len(test_datamodule.train_data),
                "test_samples": len(test_datamodule.test_data),
                "val_samples": len(test_datamodule.val_data) if test_datamodule.val_data else 0,
                "lightning_confusion_matrix": lightning_confusion_matrix,
                "custom_analysis_metrics": custom_metrics
            }
            
            print(f"   🎯 Using Custom Analysis Results:")
            print(f"      Custom AUROC: {custom_metrics['custom_auroc']:.4f}")
            print(f"      Custom F1: {custom_metrics['custom_f1_score']:.4f}")
            print(f"      Custom Accuracy: {custom_metrics['custom_accuracy']:.4f}")
            
        elif test_results and len(test_results) > 0:
            # Fallback to Lightning Test results if Custom Analysis fails
            test_metrics = test_results[0]
            image_auroc = test_metrics.get("test_image_AUROC", test_metrics.get("image_AUROC", 0.0))
            
            results = {
                "domain": domain,
                "image_AUROC": float(image_auroc),
                "image_F1Score": test_metrics.get("test_image_F1Score", test_metrics.get("image_F1Score", 0.0)),
                "training_samples": len(test_datamodule.train_data),
                "test_samples": len(test_datamodule.test_data),
                "val_samples": len(test_datamodule.val_data) if test_datamodule.val_data else 0,
                "lightning_confusion_matrix": lightning_confusion_matrix
            }
            
            print(f"   ⚠️ Fallback to Lightning Test Results:")
            print(f"      Lightning AUROC: {results['image_AUROC']:.4f}")
            print(f"      Lightning F1: {results['image_F1Score']:.4f}")
        else:
            results = {"domain": domain, "error": "No test results available"}
            logger.error(f"❌ {domain} 평가 실패")
        
        # Lightning vs Custom Analysis 비교 출력
        if lightning_confusion_matrix and custom_metrics:
            lightning_auroc = lightning_confusion_matrix.get('auroc', 0.0)
            custom_auroc = custom_metrics.get('custom_auroc', 0.0)
            print(f"      Lightning CM AUROC: {lightning_auroc:.4f}")
            print(f"      Custom Analysis AUROC: {custom_auroc:.4f}")
            print(f"      🔍 Lightning vs Custom 차이: {abs(lightning_auroc - custom_auroc):.4f}")
            
        logger.info(f"✅ {domain} 평가 완료: Custom AUROC={results.get('image_AUROC', 0.0):.4f}")
        
        return results
    
    def _calculate_lightning_confusion_matrix(self, model, test_datamodule, logger):
        """Lightning 결과의 confusion matrix 계산"""
        from sklearn.metrics import confusion_matrix, roc_auc_score
        import numpy as np
        
        print(f"   🔧 Lightning 예측 점수 수집 중...")
        
        # 모델을 evaluation 모드로 설정
        model.eval()
        
        # 모델을 적절한 디바이스로 이동 (한 번만 실행)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        print(f"   🖥️  Lightning CM: 모델을 {device} 디바이스로 이동")
        
        # 데이터 수집을 위한 리스트들
        all_predictions = []
        all_ground_truth = []
        all_scores = []
        
        test_dataloader = test_datamodule.test_dataloader()
        total_batches = len(test_dataloader)
        
        print(f"   📊 Lightning CM: {total_batches}개 배치 처리 중...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                if batch_idx % 10 == 0:  # 매 10번째 배치마다 진행률 출력
                    print(f"   📝 Lightning CM: {batch_idx+1}/{total_batches} 배치 처리 중...")
                
                # Ground truth 수집
                if hasattr(batch, 'gt_label'):
                    gt_labels = batch.gt_label.cpu().numpy()
                    all_ground_truth.extend(gt_labels)
                
                # 모델 예측
                try:
                    # 입력 데이터를 모델과 같은 디바이스로 이동
                    input_images = batch.image.to(device)
                    
                    # Lightning 모델로 직접 예측
                    outputs = model(input_images)
                    
                    # DraemSevNet의 경우 final_score 사용
                    if hasattr(outputs, 'final_score'):
                        scores = outputs.final_score.cpu().numpy()
                    elif hasattr(outputs, 'pred_score'):
                        scores = outputs.pred_score.cpu().numpy()
                    elif hasattr(outputs, 'anomaly_score'):
                        scores = outputs.anomaly_score.cpu().numpy()
                    else:
                        # InferenceBatch의 경우
                        scores = outputs.pred_score.cpu().numpy() if hasattr(outputs, 'pred_score') else np.zeros(len(gt_labels))
                    
                    all_scores.extend(scores)
                    
                except Exception as e:
                    print(f"   ⚠️ 배치 {batch_idx} 처리 오류: {e}")
                    logger.warning(f"Lightning CM 배치 {batch_idx} 처리 오류: {e}")
                    # 오류 시 더미 점수 추가
                    dummy_scores = np.zeros(len(gt_labels))
                    all_scores.extend(dummy_scores)
        
        if len(all_ground_truth) == 0 or len(all_scores) == 0:
            print(f"   ❌ Lightning CM: 데이터 수집 실패")
            return None
        
        # 길이 맞추기
        min_len = min(len(all_ground_truth), len(all_scores))
        all_ground_truth = all_ground_truth[:min_len]
        all_scores = all_scores[:min_len]
        
        print(f"   ✅ Lightning CM: {len(all_ground_truth)}개 샘플 수집 완료")
        
        # 점수 분포 분석 추가
        scores_array = np.array(all_scores)
        print(f"   🔍 Lightning 점수 분포:")
        print(f"      최소값: {scores_array.min():.4f}")
        print(f"      최대값: {scores_array.max():.4f}")
        print(f"      평균값: {scores_array.mean():.4f}")
        print(f"      중간값: {np.median(scores_array):.4f}")
        print(f"      표준편차: {scores_array.std():.4f}")
        
        # 라벨별 점수 분포
        gt_array = np.array(all_ground_truth)
        normal_scores = scores_array[gt_array == 0]
        anomaly_scores = scores_array[gt_array == 1]
        
        print(f"   📊 라벨별 점수 분포:")
        print(f"      Normal 평균: {normal_scores.mean():.4f} (min: {normal_scores.min():.4f}, max: {normal_scores.max():.4f})")
        print(f"      Anomaly 평균: {anomaly_scores.mean():.4f} (min: {anomaly_scores.min():.4f}, max: {anomaly_scores.max():.4f})")
        
        # AUROC 계산
        try:
            lightning_auroc = roc_auc_score(all_ground_truth, all_scores)
            print(f"   📊 Lightning 직접 계산 AUROC: {lightning_auroc:.4f}")
        except Exception as e:
            lightning_auroc = 0.0
            print(f"   ⚠️ Lightning AUROC 계산 오류: {e}")
        
        # Optimal threshold 찾기 (Youden's J statistic)
        from sklearn.metrics import roc_curve
        try:
            fpr, tpr, thresholds = roc_curve(all_ground_truth, all_scores)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            print(f"   🎯 Lightning 최적 임계값: {optimal_threshold:.4f}")
        except:
            optimal_threshold = np.median(all_scores)
            print(f"   🎯 Lightning 기본 임계값 (median): {optimal_threshold:.4f}")
        
        # 예측 라벨 생성
        predictions = (np.array(all_scores) > optimal_threshold).astype(int)
        
        # Confusion Matrix 계산
        cm = confusion_matrix(all_ground_truth, predictions)
        
        # 결과 출력
        print(f"   🧮 Lightning Confusion Matrix:")
        print(f"       실제\\예측    Normal  Anomaly")
        print(f"       Normal     {cm[0,0]:6d}  {cm[0,1]:6d}")
        print(f"       Anomaly    {cm[1,0]:6d}  {cm[1,1]:6d}")
        
        # 메트릭 계산
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"   📈 Lightning 메트릭:")
        print(f"      AUROC: {lightning_auroc:.4f}")
        print(f"      Accuracy: {accuracy:.4f}")
        print(f"      Precision: {precision:.4f}")
        print(f"      Recall: {recall:.4f}")
        print(f"      F1-Score: {f1:.4f}")
        
        lightning_cm_result = {
            "confusion_matrix": cm.tolist(),
            "auroc": float(lightning_auroc),
            "optimal_threshold": float(optimal_threshold),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "total_samples": len(all_ground_truth),
            "positive_samples": int(np.sum(all_ground_truth)),
            "negative_samples": int(len(all_ground_truth) - np.sum(all_ground_truth))
        }
        
        logger.info(f"Lightning CM - AUROC: {lightning_auroc:.4f}, CM: {cm.tolist()}")
        
        return lightning_cm_result
    
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
    
    def _generate_detailed_analysis(self, model, datamodule, logger):
        """이미지별 상세 예측 분석 수행 및 Custom Analysis 메트릭 반환"""
        print(f"   📊 새로운 테스트 데이터로더 생성 중...")
        
        # 모델을 evaluation 모드로 설정
        model.eval()
        
        # PyTorch 모델에 직접 접근
        if not hasattr(model, 'model'):
            raise AttributeError("모델에 'model' 속성이 없습니다.")
        
        torch_model = model.model
        torch_model.eval()
        
        # 모델을 GPU로 이동 (CUDA 사용 가능한 경우)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch_model = torch_model.to(device)
        print(f"   🖥️ 모델을 {device}로 이동 완료")
        
        # 데이터 수집을 위한 리스트들
        all_image_paths = []
        all_ground_truth = []
        all_scores = []
        all_mask_scores = []
        all_severity_scores = []
        
        # 테스트 데이터로더 생성 (이미 evaluate_model에서 새로운 DataModule 생성됨)
        test_dataloader = datamodule.test_dataloader()
        print(f"   ✅ 테스트 데이터로더 생성 완료")
        total_batches = len(test_dataloader)
        
        print(f"   🔄 {total_batches}개 배치 처리 시작...")
        
        # 배치별로 예측 수행
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                print(f"   📝 처리 중: {batch_idx+1}/{total_batches} 배치 (진행률: {100*(batch_idx+1)/total_batches:.1f}%)")
                
                try:
                    # 이미지 경로 추출
                    if hasattr(batch, 'image_path'):
                        image_paths = batch.image_path
                        if not isinstance(image_paths, list):
                            image_paths = [image_paths]
                    else:
                        # 경로가 없는 경우 배치 크기만큼 더미 경로 생성
                        batch_size = batch.image.shape[0]
                        image_paths = [f"batch_{batch_idx}_sample_{i}.jpg" for i in range(batch_size)]
                    
                    # 이미지 텐서 추출
                    image_tensor = batch.image
                    print(f"      🖼️  이미지 텐서 크기: {image_tensor.shape}, 경로 수: {len(image_paths)}")
                    
                    # 이미지 텐서를 모델과 같은 디바이스로 이동
                    image_tensor = image_tensor.to(device)
                    
                    # 모델로 직접 예측 수행
                    model_output = torch_model(image_tensor)
                    print(f"      ✅ 모델 출력 완료: {type(model_output)}")
                    
                    # 모델별 출력에서 점수들 추출
                    final_scores, mask_scores, severity_scores = self._extract_scores_from_model_output(
                        model_output, image_tensor.shape[0], batch_idx
                    )
                        
                except Exception as e:
                    print(f"   ❌ 배치 {batch_idx} 전체 처리 실패: {str(e)}")
                    import traceback
                    print(f"      🔍 상세 오류: {traceback.format_exc()}")
                    
                    # 기본값으로 건너뛰기
                    batch_size = image_tensor.shape[0] if 'image_tensor' in locals() else 16
                    final_scores = [0.5] * batch_size
                    mask_scores = [0.0] * batch_size
                    severity_scores = [0.0] * batch_size
                    image_paths = [f"batch_{batch_idx}_sample_{i}.jpg" for i in range(batch_size)]
                
                # Ground truth 추출 (이미지 경로에서)
                gt_labels = []
                for path in image_paths:
                    if isinstance(path, str):
                        if '/fault/' in path:
                            gt_labels.append(1)  # anomaly
                        elif '/good/' in path:
                            gt_labels.append(0)  # normal
                        else:
                            gt_labels.append(0)  # 기본값
                    else:
                        gt_labels.append(0)
                
                # 결과 수집
                all_image_paths.extend(image_paths)
                all_ground_truth.extend(gt_labels)
                all_scores.extend(final_scores.flatten() if hasattr(final_scores, 'flatten') else final_scores)
                all_mask_scores.extend(mask_scores.flatten() if hasattr(mask_scores, 'flatten') else mask_scores)
                all_severity_scores.extend(severity_scores.flatten() if hasattr(severity_scores, 'flatten') else severity_scores)
                
                print(f"      ✅ 배치 {batch_idx+1} 완료: {len(gt_labels)}개 샘플 추가")
        
        print(f"   ✅ 총 {len(all_image_paths)}개 샘플 처리 완료")
        
        # 예측 레이블 생성 (threshold 0.5)
        all_predictions = [1 if score > 0.5 else 0 for score in all_scores]
        
        # analysis 폴더 생성
        analysis_dir = self.experiment_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        print(f"   💾 분석 결과 저장 중: {analysis_dir}")
        
        # 상세 테스트 결과 CSV 저장
        predictions_dict = {
            "pred_scores": all_scores,
            "mask_scores": all_mask_scores,
            "severity_scores": all_severity_scores
        }
        ground_truth_dict = {
            "labels": all_ground_truth
        }
        save_detailed_test_results(
            predictions_dict, ground_truth_dict, all_image_paths, 
            analysis_dir, self.model_type
        )
        
        # AUROC 계산 및 ROC curve 생성
        from sklearn.metrics import roc_auc_score, roc_curve
        try:
            auroc = roc_auc_score(all_ground_truth, all_scores)
            plot_roc_curve(all_ground_truth, all_scores, analysis_dir, self.experiment_name)
            
            # 임계값 계산
            fpr, tpr, thresholds = roc_curve(all_ground_truth, all_scores)
            optimal_idx = (tpr - fpr).argmax()
            optimal_threshold = thresholds[optimal_idx]
            
            # 메트릭 보고서 저장
            save_metrics_report(all_ground_truth, all_predictions, all_scores, analysis_dir, auroc, optimal_threshold)
            
            # 점수 분포 히스토그램 생성
            normal_scores = [score for gt, score in zip(all_ground_truth, all_scores) if gt == 0]
            anomaly_scores = [score for gt, score in zip(all_ground_truth, all_scores) if gt == 1]
            plot_score_distributions(normal_scores, anomaly_scores, analysis_dir, self.experiment_name)
            
            # 극단적 신뢰도 샘플 저장
            save_extreme_samples(all_image_paths, all_ground_truth, all_scores, all_predictions, analysis_dir)
            
            # 실험 요약 저장
            save_experiment_summary(self.config, {"auroc": auroc}, analysis_dir)
            
            print(f"   📈 AUROC: {auroc:.4f}, 최적 임계값: {optimal_threshold:.4f}")
            logger.info(f"상세 분석 완료: AUROC={auroc:.4f}, 샘플수={len(all_image_paths)}")
            
            # Custom Analysis 메트릭 반환
            from sklearn.metrics import confusion_matrix
            predictions = (np.array(all_scores) > optimal_threshold).astype(int)
            cm = confusion_matrix(all_ground_truth, predictions)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, len(all_ground_truth), 0)
            
            accuracy = (tp + tn) / len(all_ground_truth) if len(all_ground_truth) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            custom_metrics = {
                "custom_auroc": float(auroc),
                "custom_f1_score": float(f1),
                "custom_accuracy": float(accuracy), 
                "custom_precision": float(precision),
                "custom_recall": float(recall),
                "custom_confusion_matrix": cm.tolist(),
                "custom_optimal_threshold": float(optimal_threshold),
                "custom_total_samples": len(all_ground_truth),
                "custom_positive_samples": int(np.sum(all_ground_truth)),
                "custom_negative_samples": int(len(all_ground_truth) - np.sum(all_ground_truth))
            }
            
            return custom_metrics
            
        except Exception as e:
            print(f"   ⚠️ 메트릭 계산 실패: {str(e)}")
            logger.error(f"메트릭 계산 실패: {str(e)}")
            return None
    
    def run_experiment(self) -> dict:
        """전체 실험 실행"""
        domain = self.config.get("source_domain") or self.config.get("domain")
        
        print(f"\n{'='*80}")
        print(f"🔬 {self.model_type.upper()} Single Domain 실험: {self.experiment_name}")
        print(f"🎯 도메인: {domain}")
        print(f"{'='*80}")
        
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
            print(f"\n📊 DataModule 데이터 개수 확인:")
            print(f"   🔧 DataModule 준비 및 설정 중...")
            datamodule.prepare_data()
            datamodule.setup()
            
            train_size = len(datamodule.train_data) if datamodule.train_data else 0
            test_size = len(datamodule.test_data) if datamodule.test_data else 0
            val_size = len(datamodule.val_data) if datamodule.val_data else 0
            
            print(f"   📈 훈련 데이터: {train_size:,}개")
            print(f"   📊 테스트 데이터: {test_size:,}개")
            print(f"   📋 검증 데이터: {val_size:,}개")
            print(f"   🎯 총 데이터: {train_size + test_size + val_size:,}개")
            
            # 테스트 데이터 라벨 분포 확인 (처음 몇 배치만 샘플링)
            test_loader = datamodule.test_dataloader()
            fault_count = 0
            good_count = 0
            sampled_images = 0
            max_sample = min(5 * datamodule.eval_batch_size, test_size)  # 처음 5배치 또는 전체
            
            print(f"   🔍 테스트 데이터 라벨 분포 확인 중 (샘플: {max_sample}개)...")
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    if hasattr(batch, 'gt_label'):
                        labels = batch.gt_label.numpy()
                        fault_count += (labels == 1).sum()
                        good_count += (labels == 0).sum()
                        sampled_images += len(labels)
                    
                    if batch_idx >= 4 or sampled_images >= max_sample:  # 처음 5배치만 확인
                        break
            
            print(f"   🚨 테스트 샘플 분포 ({sampled_images}개): Fault={fault_count}, Good={good_count}")
            if sampled_images > 0:
                fault_ratio = fault_count / sampled_images * 100
                good_ratio = good_count / sampled_images * 100
                print(f"   📊 테스트 샘플 비율: Fault={fault_ratio:.1f}%, Good={good_ratio:.1f}%")
                
                # 불균형 경고
                if fault_count == 0:
                    print(f"   ⚠️  경고: Fault 이미지가 없습니다! AUROC 계산에 문제가 있을 수 있습니다.")
                elif good_count == 0:
                    print(f"   ⚠️  경고: Good 이미지가 없습니다! AUROC 계산에 문제가 있을 수 있습니다.")
                elif abs(fault_count - good_count) > sampled_images * 0.3:
                    print(f"   ⚠️  경고: 라벨 분포가 불균형합니다 (30% 이상 차이)")
                else:
                    print(f"   ✅ 테스트 데이터 라벨 분포 정상")
            
            logger.info(f"DataModule - 훈련: {train_size}, 테스트: {test_size}, 검증: {val_size}")
            logger.info(f"테스트 샘플 분포 - Fault: {fault_count}, Good: {good_count}")
            
            # 모델 훈련
            trained_model, engine, best_checkpoint = self.train_model(model, datamodule, logger)
            
            # 성능 평가
            results = self.evaluate_model(trained_model, engine, datamodule, logger)
            
            # 훈련 정보 추출
            training_info = extract_training_info(engine)
            
            # 결과 저장
            experiment_results = self.save_results(results, training_info, best_checkpoint, logger)
            
            # 시각화 생성
            try:
                create_experiment_visualization(
                    experiment_name=self.experiment_name,
                    model_type=f"{self.model_type.upper()}_single_domain_{domain}",
                    results_base_dir=str(self.experiment_dir),
                    source_domain=domain,
                    source_results=experiment_results.get('results', {}),
                    single_domain=True
                )
                print(f"📊 결과 시각화 생성 완료")
            except Exception as viz_error:
                print(f"⚠️ 시각화 생성 중 오류: {viz_error}")
                logger.warning(f"시각화 생성 중 오류: {viz_error}")
            
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
    
    def _extract_scores_from_model_output(self, model_output, batch_size, batch_idx):
        """
        모델별 출력에서 점수들을 추출합니다.
        
        Args:
            model_output: 모델 출력 객체
            batch_size: 배치 크기
            batch_idx: 배치 인덱스
            
        Returns:
            tuple: (anomaly_scores, mask_scores, severity_scores)
        """
        model_type = self.model_type.lower()
        
        try:
            if model_type == "draem_sevnet":
                # DRAEM-SevNet: final_score, mask_score, severity_score 있음
                if hasattr(model_output, 'final_score'):
                    final_scores = model_output.final_score.cpu().numpy()
                    mask_scores = model_output.mask_score.cpu().numpy()
                    severity_scores = model_output.severity_score.cpu().numpy()
                    print(f"      📊 DRAEM-SevNet 점수 추출: final={final_scores[0]:.4f}, mask={mask_scores[0]:.4f}, severity={severity_scores[0]:.4f}")
                else:
                    raise AttributeError("DraemSevNetOutput 속성 없음")
                    
            elif model_type == "draem":
                # DRAEM: pred_score만 있음
                if hasattr(model_output, 'pred_score'):
                    final_scores = model_output.pred_score.cpu().numpy()
                    mask_scores = [0.0] * batch_size  # DRAEM에는 mask_score 없음
                    severity_scores = [0.0] * batch_size  # DRAEM에는 severity_score 없음
                    print(f"      📊 DRAEM 점수 추출: pred_score={final_scores[0]:.4f}")
                elif hasattr(model_output, 'anomaly_map'):
                    # anomaly_map에서 점수 계산
                    anomaly_map = model_output.anomaly_map.cpu().numpy()
                    final_scores = [float(np.max(am)) for am in anomaly_map]
                    mask_scores = [0.0] * batch_size
                    severity_scores = [0.0] * batch_size
                    print(f"      📊 DRAEM 점수 추출 (anomaly_map): max={final_scores[0]:.4f}")
                else:
                    raise AttributeError("DRAEM 출력 속성 없음")
                    
            elif model_type == "patchcore":
                # PatchCore: pred_score만 있음
                if hasattr(model_output, 'pred_score'):
                    final_scores = model_output.pred_score.cpu().numpy()
                    mask_scores = [0.0] * batch_size
                    severity_scores = [0.0] * batch_size
                    print(f"      📊 PatchCore 점수 추출: pred_score={final_scores[0]:.4f}")
                elif hasattr(model_output, 'anomaly_map'):
                    # anomaly_map에서 점수 계산
                    anomaly_map = model_output.anomaly_map.cpu().numpy()
                    final_scores = [float(np.max(am)) for am in anomaly_map]
                    mask_scores = [0.0] * batch_size
                    severity_scores = [0.0] * batch_size
                    print(f"      📊 PatchCore 점수 추출 (anomaly_map): max={final_scores[0]:.4f}")
                else:
                    raise AttributeError("PatchCore 출력 속성 없음")
                    
            elif model_type == "dinomaly":
                # Dinomaly: pred_score 또는 anomaly_map
                if hasattr(model_output, 'pred_score'):
                    final_scores = model_output.pred_score.cpu().numpy()
                    mask_scores = [0.0] * batch_size
                    severity_scores = [0.0] * batch_size
                    print(f"      📊 Dinomaly 점수 추출: pred_score={final_scores[0]:.4f}")
                elif hasattr(model_output, 'anomaly_map'):
                    # anomaly_map에서 점수 계산
                    anomaly_map = model_output.anomaly_map.cpu().numpy()
                    final_scores = [float(np.max(am)) for am in anomaly_map]
                    mask_scores = [0.0] * batch_size
                    severity_scores = [0.0] * batch_size
                    print(f"      📊 Dinomaly 점수 추출 (anomaly_map): max={final_scores[0]:.4f}")
                else:
                    raise AttributeError("Dinomaly 출력 속성 없음")
                    
            else:
                # 알 수 없는 모델 타입: 기본 처리
                print(f"   ⚠️ 알 수 없는 모델 타입: {model_type}, 일반적인 속성으로 시도")
                if hasattr(model_output, 'pred_score'):
                    final_scores = model_output.pred_score.cpu().numpy()
                elif hasattr(model_output, 'final_score'):
                    final_scores = model_output.final_score.cpu().numpy()
                elif hasattr(model_output, 'anomaly_map'):
                    anomaly_map = model_output.anomaly_map.cpu().numpy()
                    final_scores = [float(np.max(am)) for am in anomaly_map]
                else:
                    raise AttributeError(f"지원되지 않는 모델 출력 형식: {type(model_output)}")
                    
                mask_scores = [0.0] * batch_size
                severity_scores = [0.0] * batch_size
                print(f"      📊 일반 모델 점수 추출: anomaly_score={final_scores[0]:.4f}")
                
            return final_scores, mask_scores, severity_scores
            
        except Exception as e:
            # fallback: 더미 점수 사용
            print(f"   ⚠️ 배치 {batch_idx}: {model_type} 점수 추출 실패 - {str(e)}, 더미값 사용")
            final_scores = [0.5] * batch_size
            mask_scores = [0.0] * batch_size 
            severity_scores = [0.0] * batch_size
            return final_scores, mask_scores, severity_scores