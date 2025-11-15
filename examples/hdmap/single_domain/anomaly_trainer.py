#!/usr/bin/env python3
"""
BaseAnomalyTrainer - 통합 Anomaly Detection 모델 훈련을 위한 베이스 클래스

이 모듈은 모든 anomaly detection 모델의 훈련을 통합 관리합니다.

지원 모델:
- DRAEM: Reconstruction + Anomaly Detection
- DRAEM CutPaste: DRAEM with CutPaste augmentation (without severity head)
- DRAEM CutPaste Clf: DRAEM with CutPaste augmentation + CNN classification
- Dinomaly: Vision Transformer 기반 anomaly detection with DINOv2
- PatchCore: Memory bank 기반 few-shot anomaly detection
"""

import torch
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

# 공통 유틸리티 함수들 import
sys.path.append(str(Path(__file__).parent.parent))
from experiment_utils import (
    cleanup_gpu_memory,
    setup_experiment_logging,
    extract_training_info,
    save_experiment_results,
    create_single_domain_datamodule,
    analyze_test_data_distribution,
    unified_model_evaluation
)

# 모델별 imports
from anomalib.models.image.draem import Draem
from anomalib.models.image.draem_cutpaste import DraemCutPaste
from anomalib.models.image.draem_cutpaste_clf import DraemCutPasteClf
from anomalib.models.image import Dinomaly, Patchcore, EfficientAd, Fastflow, Padim, Supersimplenet, ReverseDistillation, CutPasteClassifier
from anomalib.models.image.reverse_distillation.anomaly_map import AnomalyMapGenerationMode
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
        elif self.model_type == "draem_cutpaste":
            return self._create_draem_cutpaste_model()
        elif self.model_type == "draem_cutpaste_clf":
            return self._create_draem_cutpaste_clf_model()
        elif self.model_type == "cutpaste_clf":
            return self._create_cutpaste_clf_model()
        elif self.model_type == "dinomaly":
            return self._create_dinomaly_model()
        elif self.model_type == "patchcore":
            return self._create_patchcore_model()
        elif self.model_type == "efficient_ad":
            return self._create_efficient_ad_model()
        elif self.model_type == "fastflow":
            return self._create_fastflow_model()
        elif self.model_type == "padim":
            return self._create_padim_model()
        elif self.model_type == "supersimplenet":
            return self._create_supersimplenet_model()
        elif self.model_type == "reverse_distillation":
            return self._create_reverse_distillation_model()
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
        """DRAEM CutPaste 모델 생성 (without severity head)"""
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
            'focal_alpha': self.config.get("focal_alpha", 0.9),
        }
        
        # DraemCutPaste 모델 생성
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
            'severity_input_channels': self.config["severity_input_channels"],
            'cut_w_range': tuple(self.config["cut_w_range"]),
            'cut_h_range': tuple(self.config["cut_h_range"]),
            'a_fault_start': self.config["a_fault_start"],
            'a_fault_range_end': self.config["a_fault_range_end"],
            'augment_probability': self.config["augment_probability"],
            'clf_weight': self.config["clf_weight"],
            'focal_alpha': self.config.get("focal_alpha", 0.9),
        }

        # DraemCutPasteClf 모델 생성
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
    
    def _create_efficient_ad_model(self):
        """EfficientAD 모델 생성"""
        # 절대 경로로 ImageNette와 pretrained weights 경로 설정
        from pathlib import Path
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent  # anomalib/ 디렉토리
        
        # HDMAP 데이터는 gt_mask가 없으므로 image-level 메트릭만 사용
        val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
        test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
        evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])
        
        return EfficientAd(
            teacher_out_channels=self.config.get("teacher_out_channels", 384),
            model_size=self.config.get("model_size", "small"),
            lr=self.config.get("learning_rate", 0.0001),
            weight_decay=self.config.get("weight_decay", 0.00001),
            padding=self.config.get("padding", False),
            pad_maps=self.config.get("pad_maps", True),
            imagenet_dir=str(project_root / "datasets" / "imagenette"),
            evaluator=evaluator  # 커스텀 evaluator 사용
        )
    
    def _create_fastflow_model(self):
        """FastFlow 모델 생성"""
        # FastFlow는 image-level 메트릭만 사용 (gt_mask 없음)
        val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
        test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
        evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])

        model = Fastflow(
            backbone=self.config.get("backbone", "resnet18"),
            flow_steps=self.config.get("flow_steps", 8),
            conv3x3_only=self.config.get("conv3x3_only", False),
            hidden_ratio=self.config.get("hidden_ratio", 1.0),
            evaluator=evaluator
        )

        # 학습 설정을 _training_config에 저장
        model._training_config = {
            'learning_rate': self.config.get("learning_rate", 0.001),
            'weight_decay': self.config.get("weight_decay", 0.00001),
            'max_epochs': self.config["max_epochs"],
        }

        return model

    def _create_padim_model(self):
        """PaDiM 모델 생성 (Training-free, Memory Bank)"""
        # PaDiM은 image-level 메트릭만 사용 (gt_mask 없음)
        val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
        test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
        evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])

        layers = self.config.get("layers", ["layer1", "layer2", "layer3"])
        print(f"   🔍 PaDiM 모델 생성 - 요청된 layers: {layers}")

        model = Padim(
            backbone=self.config.get("backbone", "resnet18"),
            layers=layers,
            pre_trained=self.config.get("pre_trained", True),
            n_features=self.config.get("n_features", None),  # None = use default (100 for resnet18)
            evaluator=evaluator
        )

        # 모델 생성 후 실제 사용되는 layers 확인
        if hasattr(model.model, 'layers'):
            print(f"   🔍 PaDiM 모델 생성 완료 - 실제 layers: {model.model.layers}")
        if hasattr(model.model, 'feature_extractor') and hasattr(model.model.feature_extractor, 'layers'):
            print(f"   🔍 PaDiM feature_extractor layers: {model.model.feature_extractor.layers}")

        return model

    def _create_supersimplenet_model(self):
        """SuperSimpleNet 모델 생성"""
        # SuperSimpleNet은 image-level 메트릭만 사용 (gt_mask 없음)
        val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
        test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
        evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])

        model = Supersimplenet(
            perlin_threshold=self.config.get("perlin_threshold", 0.2),
            backbone=self.config.get("backbone", "wide_resnet50_2.tv_in1k"),
            layers=self.config.get("layers", ["layer2", "layer3"]),
            supervised=self.config.get("supervised", False),
            evaluator=evaluator
        )

        # SuperSimpleNet은 내부에서 optimizer를 configure하므로 별도 설정 불필요
        # (AdamW with MultiStepLR scheduler)

        return model

    def _create_reverse_distillation_model(self):
        """Reverse Distillation 모델 생성"""
        # Reverse Distillation은 image-level 메트릭만 사용 (gt_mask 없음)
        val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
        test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
        evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])

        model = ReverseDistillation(
            backbone=self.config.get("backbone", "wide_resnet50_2"),
            layers=self.config.get("layers", ["layer1", "layer2", "layer3"]),
            anomaly_map_mode=AnomalyMapGenerationMode.ADD,  # 기본값
            pre_trained=self.config.get("pre_trained", True),
            evaluator=evaluator
        )

        # Reverse Distillation은 내부에서 optimizer를 configure하므로 별도 설정 불필요
        # (Adam with lr=0.005, betas=(0.5, 0.99))

        return model

    def _create_cutpaste_clf_model(self):
        """CutPaste + Simple CNN Classifier 모델 생성 (Baseline)"""
        # CutPaste Classifier는 image-level 메트릭만 사용 (gt_mask 없음)
        val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
        test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
        evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])

        model = CutPasteClassifier(
            image_size=tuple(self.config["image_size"]),
            cut_w_range=tuple(self.config["cut_w_range"]),
            cut_h_range=tuple(self.config["cut_h_range"]),
            a_fault_start=self.config["a_fault_start"],
            a_fault_range_end=self.config["a_fault_range_end"],
            augment_probability=self.config["augment_probability"],
            evaluator=evaluator,
            pre_processor=False  # HDMAPDataModule의 target_size로 리사이즈되므로 PreProcessor 비활성화
        )

        # 학습 설정을 _training_config에 저장
        model._training_config = {
            'learning_rate': self.config['learning_rate'],
            'optimizer': self.config['optimizer'],
            'weight_decay': self.config['weight_decay'],
            'max_epochs': self.config['max_epochs'],
            'scheduler': self.config.get('scheduler', None),
        }

        return model


    def create_datamodule(self):
        """모든 모델에 공통으로 사용할 데이터 모듈 생성"""
        # 도메인 설정 - 모델별로 다른 필드명 처리
        domain = self.config.get("source_domain") or self.config.get("domain")
        if not domain:
            raise ValueError("config에서 'source_domain' 또는 'domain' 필드를 찾을 수 없습니다")
        
        # dataset_root 필수 체크 및 상대 경로 처리
        dataset_root = self.config.get("dataset_root")
        if not dataset_root:
            raise ValueError("config에서 'dataset_root' 필드를 찾을 수 없습니다")
        
        # 상대 경로인 경우 프로젝트 루트 기준으로 변환
        from pathlib import Path
        dataset_path = Path(dataset_root)
        if not dataset_path.is_absolute():
            # 프로젝트 루트 찾기 (anomalib 디렉토리 기준)
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent.parent  # 4단계 상위 = anomalib/
            dataset_root = str(project_root / dataset_root)
            print(f"   📁 상대 경로를 절대 경로로 변환: {dataset_root}")
        
        # target_size 설정 (list -> tuple 변환)
        target_size = self.config.get("target_size")
        print(f"\n🔍 DEBUG create_datamodule:")
        print(f"   config.target_size = {self.config.get('target_size')}")
        print(f"   target_size variable = {target_size}")
        if target_size and isinstance(target_size, list) and len(target_size) == 2:
            target_size = tuple(target_size)
            print(f"   target_size after tuple conversion = {target_size}")
        elif target_size:
            raise ValueError("target_size는 [height, width] 형태의 리스트여야 합니다")

        val_split_mode = self.config.get("val_split_mode", "FROM_TEST")

        return create_single_domain_datamodule(
            domain=domain,
            dataset_root=dataset_root,
            batch_size=self.config["batch_size"],
            target_size=target_size,
            resize_method=self.config.get("resize_method", "resize"),
            val_split_ratio=self.config["val_split_ratio"],
            val_split_mode=val_split_mode,
            num_workers=self.config["num_workers"],
            seed=self.config["seed"]
        )
    
    def print_data_statistics(self, datamodule, logger):
        """데이터셋의 통계량 출력"""
        print(f"\n📊 데이터 분포 통계량:")
        logger.info("📊 데이터 분포 통계량:")

        import torch

        # Train 데이터 통계
        try:
            train_loader = datamodule.train_dataloader()
            train_batch = next(iter(train_loader))
            train_images = train_batch.image

            train_min = train_images.min().item()
            train_max = train_images.max().item()
            train_mean = train_images.mean().item()
            train_std = train_images.std().item()

            print(f"   📈 Train 데이터:")
            print(f"      - 범위: [{train_min:.6f}, {train_max:.6f}]")
            print(f"      - 평균: {train_mean:.6f}")
            print(f"      - 표준편차: {train_std:.6f}")
            logger.info(f"Train 데이터 - 범위: [{train_min:.6f}, {train_max:.6f}], 평균: {train_mean:.6f}, 표준편차: {train_std:.6f}")
        except Exception as e:
            print(f"   ⚠️ Train 데이터 통계 계산 실패: {e}")
            logger.warning(f"Train 데이터 통계 계산 실패: {e}")

        # Val 데이터 통계
        try:
            val_loader = datamodule.val_dataloader()
            val_batch = next(iter(val_loader))
            val_images = val_batch.image

            val_min = val_images.min().item()
            val_max = val_images.max().item()
            val_mean = val_images.mean().item()
            val_std = val_images.std().item()

            print(f"   📊 Val 데이터:")
            print(f"      - 범위: [{val_min:.6f}, {val_max:.6f}]")
            print(f"      - 평균: {val_mean:.6f}")
            print(f"      - 표준편차: {val_std:.6f}")
            logger.info(f"Val 데이터 - 범위: [{val_min:.6f}, {val_max:.6f}], 평균: {val_mean:.6f}, 표준편차: {val_std:.6f}")
        except Exception as e:
            print(f"   ⚠️ Val 데이터 통계 계산 실패: {e}")
            logger.warning(f"Val 데이터 통계 계산 실패: {e}")

        # Test 데이터 통계
        try:
            test_loader = datamodule.test_dataloader()
            test_batch = next(iter(test_loader))
            test_images = test_batch.image

            test_min = test_images.min().item()
            test_max = test_images.max().item()
            test_mean = test_images.mean().item()
            test_std = test_images.std().item()

            print(f"   🧪 Test 데이터:")
            print(f"      - 범위: [{test_min:.6f}, {test_max:.6f}]")
            print(f"      - 평균: {test_mean:.6f}")
            print(f"      - 표준편차: {test_std:.6f}")
            logger.info(f"Test 데이터 - 범위: [{test_min:.6f}, {test_max:.6f}], 평균: {test_mean:.6f}, 표준편차: {test_std:.6f}")
        except Exception as e:
            print(f"   ⚠️ Test 데이터 통계 계산 실패: {e}")
            logger.warning(f"Test 데이터 통계 계산 실패: {e}")

        print()

    def create_callbacks(self):
        """콜백 설정 - 모델별 적절한 early stopping 메트릭 사용"""
        callbacks = []

        # 모델별 EarlyStopping 설정
        if self.model_type in ["patchcore", "padim"]:
            # PatchCore와 PaDiM은 training-free 방식이므로 EarlyStopping과 ModelCheckpoint 불필요
            print(f"   ℹ️ {self.model_type.upper()}: EarlyStopping 및 ModelCheckpoint 비활성화 (Training-free 방식)")
            return []  # 빈 콜백 리스트 반환

        else:
            # 모델별로 다른 EarlyStopping monitor 설정
            if self.model_type in ["draem", "draem_cutpaste", "draem_cutpaste_clf"]:
                # DRAEM 계열: val_loss 기반 EarlyStopping (낮을수록 좋음)
                monitor_metric = "val_loss"
                monitor_mode = "min"
                print(f"   ℹ️ {self.model_type.upper()}: EarlyStopping 활성화 (val_loss 모니터링)")
            elif self.model_type in ["supersimplenet", "reverse_distillation"]:
                # SuperSimpleNet, Reverse Distillation: train_loss 기반 EarlyStopping
                monitor_metric = "train_loss"
                monitor_mode = "min"
                print(f"   ℹ️ {self.model_type.upper().replace('_', ' ')}: EarlyStopping 활성화 (train_loss 모니터링)")
            elif self.model_type in ["efficient_ad", "fastflow", "cutpaste_clf"]:
                # EfficientAD, FastFlow, CutPaste Classifier: val_image_AUROC 기반 EarlyStopping (높을수록 좋음)
                monitor_metric = "val_image_AUROC"
                monitor_mode = "max"
                print(f"   ℹ️ {self.model_type.upper().replace('_', ' ')}: EarlyStopping 활성화 (val_image_AUROC 모니터링)")
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

            if self.model_type in ["draem", "draem_cutpaste", "draem_cutpaste_clf"]:
                checkpoint = ModelCheckpoint(
                    filename=f"{self.model_type}_single_domain_{domain}_" + "{epoch:02d}_{val_loss:.4f}",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                    verbose=True
                )
            elif self.model_type in ["supersimplenet", "reverse_distillation"]:
                checkpoint = ModelCheckpoint(
                    filename=f"{self.model_type}_single_domain_{domain}_" + "{epoch:02d}_{train_loss:.4f}",
                    monitor="train_loss",
                    mode="min",
                    save_top_k=1,
                    verbose=True
                )
            elif self.model_type in ["efficient_ad", "fastflow", "cutpaste_clf"]:
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
        # Training-free 모델들은 옵티마이저가 필요하지 않음
        if self.model_type in ["patchcore", "padim", "efficient_ad"]:
            return
                
    def train_model(self, model, datamodule, logger) -> Tuple[Any, Engine, str]:
        """모델 훈련 수행"""
        print(f"\n🚀 {self.model_type.upper()} 모델 훈련 시작")
        logger.info(f"🚀 {self.model_type.upper()} 모델 훈련 시작")
        
        # Config 설정 출력
        print(f"   🔧 Config 설정:")
        print(f"      Model Type: {self.model_type}")

        # Training-free 모델들
        if self.model_type in ["patchcore", "padim"]:
            print(f"      Max Epochs: 1 (Training-free)")
        # 내부에서 optimizer를 configure하는 모델들 (learning_rate가 config에 없을 수 있음)
        elif self.model_type in ["supersimplenet", "reverse_distillation"]:
            print(f"      Max Epochs: {self.config['max_epochs']}")
            print(f"      Early Stopping Patience: {self.config['early_stopping_patience']}")
            if self.model_type == "supersimplenet":
                print(f"      Learning Rate: AdamW with MultiStepLR (internal)")
            elif self.model_type == "reverse_distillation":
                print(f"      Learning Rate: 0.005 (internal, Adam)")
        # 일반적인 모델들 (config에서 learning_rate 사용)
        else:
            print(f"      Max Epochs: {self.config['max_epochs']}")
            print(f"      Learning Rate: {self.config['learning_rate']}")
            print(f"      Early Stopping Patience: {self.config['early_stopping_patience']}")

        print(f"      Batch Size: {self.config['batch_size']}")
        
        # 옵티마이저 설정
        self.configure_optimizer(model)
        
        # 콜백 설정
        callbacks = self.create_callbacks()
        
        analysis_dir = Path(self.experiment_dir) / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(model, "validation_analysis_dir"):
            model.validation_analysis_dir = analysis_dir

        self.tb_logger = TensorBoardLogger(
            save_dir=str(self.experiment_dir),
            name="tensorboard_logs",
            version=""
        )
        
        # Engine 설정
        # Training-free 모델들은 1 epoch만 필요
        if self.model_type in ["patchcore", "padim"]:
            max_epochs = 1
        elif self.model_type == "efficient_ad":
            max_epochs = self.config["max_epochs"]
        else:
            max_epochs = self.config["max_epochs"]
        
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
        
        if self.model_type in ["patchcore", "padim"]:
            print(f"   ℹ️ {self.model_type.upper()}: max_epochs 강제 설정 (1 epoch - Training-free)")
        elif self.model_type == "efficient_ad":
            print(f"   ℹ️ EFFICIENT AD: max_epochs = {max_epochs} (특별한 훈련 방식)")
        else:
            print(f"   ℹ️ {self.model_type.upper().replace('_', ' ')}: max_epochs = {max_epochs}")
        
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

        # Best checkpoint 정보만 기록하고 로드하지 않음 (마지막 epoch 모델 사용)
        if best_checkpoint and os.path.exists(best_checkpoint) and self.model_type not in ["patchcore", "padim", "efficient_ad"]:
            print(f"   📂 Best checkpoint 저장됨: {best_checkpoint}")
            print(f"   ℹ️ 평가는 마지막 epoch 모델로 수행됩니다.")
            logger.info(f"Best checkpoint 저장됨 (사용하지 않음): {best_checkpoint}")
        elif self.model_type in ["patchcore", "padim", "efficient_ad"]:
            print(f"   ℹ️ {self.model_type.upper().replace('_', ' ')}: Best checkpoint 로드 건너뜀 (특별한 훈련 방식)")
        elif not best_checkpoint:
            print(f"   ⚠️ Best checkpoint 없음 - 마지막 epoch 모델 사용")
            logger.warning(f"Best checkpoint 없음 - 마지막 epoch 모델 사용")

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
                model, datamodule, self.experiment_dir, self.experiment_name, self.model_type, logger, visualization_interval=3
            )
            
            if evaluation_metrics:
                # TensorBoard에 test 메트릭 로깅
                if hasattr(self, 'tb_logger') and self.tb_logger is not None:
                    test_metrics = {
                        "test_auroc": evaluation_metrics["auroc"],
                        "test_f1_score": evaluation_metrics["f1_score"],
                        "test_accuracy": evaluation_metrics["accuracy"],
                        "test_precision": evaluation_metrics["precision"],
                        "test_recall": evaluation_metrics["recall"]
                    }
                    self.tb_logger.log_metrics(test_metrics, step=0)
                    print(f"   📊 TensorBoard에 test 메트릭 로깅 완료:")
                    print(f"      - test_auroc: {evaluation_metrics['auroc']:.4f}")
                    print(f"      - test_f1_score: {evaluation_metrics['f1_score']:.4f}")
                    logger.info(f"📊 TensorBoard에 test 메트릭 로깅 완료: AUROC={evaluation_metrics['auroc']:.4f}, F1={evaluation_metrics['f1_score']:.4f}")
                
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

            # 완전한 재현성 설정 (모델 초기화, augmentation, dataloader shuffle 제어)
            seed = self.config.get("seed", 42)
            print(f"🎲 완전한 재현성 설정 (Seed: {seed})")

            import random
            import numpy as np
            import lightning.pytorch as pl

            # PyTorch Lightning의 seed_everything으로 모든 랜덤 상태 제어
            pl.seed_everything(seed, workers=True)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            # GPU deterministic mode (재현성 보장, 약간의 성능 저하 있을 수 있음)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            print(f"   ✅ PyTorch, NumPy, Random 시드 설정 완료")
            print(f"   ✅ CUDA deterministic mode 활성화")

            # 로깅 설정
            log_file_path = self.experiment_dir / f"{domain}_single.log"
            logger = setup_experiment_logging(str(log_file_path), self.experiment_name)
            logger.info(f"🚀 {self.model_type.upper()} Single Domain 실험 시작 (Seed: {seed})")
            
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

            # 데이터 분포 통계량 출력
            self.print_data_statistics(datamodule, logger)

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
    
