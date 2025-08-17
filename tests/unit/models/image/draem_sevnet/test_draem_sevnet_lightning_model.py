"""Test suite for DRAEM-SevNet Lightning Model Update.

DRAEM-SevNet으로 업데이트된 Lightning Model의 모든 기능을 테스트합니다.

Run with: pytest tests/unit/models/image/custom_draem/test_lightning_model_update.py -v -s
Author: Taewan Hwang
"""

import torch
from unittest.mock import MagicMock
from anomalib.models.image.draem_sevnet import DraemSevNet
from anomalib.models.image.draem_sevnet.loss import DraemSevNetLoss

# 상세 출력을 위한 helper function
def verbose_print(message: str, level: str = "INFO"):
    """상세 출력을 위한 함수"""
    symbols = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}
    print(f"\n{symbols.get(level, 'ℹ️')} {message}")


class TestDraemSevNetLightningModel:
    """DRAEM-SevNet Lightning Model 테스트"""
    
    def test_model_initialization(self):
        """모델 초기화 테스트"""
        verbose_print("Testing DRAEM-SevNet Lightning model initialization...")
        
        # Default initialization
        model = DraemSevNet()
        verbose_print(f"Default - severity_head_mode: {model.severity_head_mode}")
        verbose_print(f"Default - score_combination: {model.score_combination}")
        verbose_print(f"Default - has model: {hasattr(model, 'model')}")
        verbose_print(f"Default - has loss: {hasattr(model, 'loss')}")
        verbose_print(f"Default - has augmenter: {hasattr(model, 'augmenter')}")
        
        assert model.severity_head_mode == "single_scale"
        assert model.score_combination == "simple_average"
        assert hasattr(model, 'model')
        assert hasattr(model, 'loss')
        assert hasattr(model, 'augmenter')
        
        # Custom initialization 
        custom_model = DraemSevNet(
            severity_head_mode="multi_scale",
            score_combination="weighted_average",
            severity_weight_for_combination=0.3,
            severity_weight=0.8,
            severity_loss_type="smooth_l1"
        )
        verbose_print(f"Custom - severity_head_mode: {custom_model.severity_head_mode}")
        verbose_print(f"Custom - score_combination: {custom_model.score_combination}")
        
        assert custom_model.severity_head_mode == "multi_scale"
        assert custom_model.score_combination == "weighted_average"
        
        verbose_print("Lightning model initialization test passed!", "SUCCESS")
        
    def test_training_step(self):
        """Training step 테스트"""
        model = DraemSevNet()
        model.train()
        
        # Create mock batch
        batch_size = 4
        batch = MagicMock()
        batch.image = torch.randn(batch_size, 3, 224, 224)
        batch.gt_label = torch.randint(0, 2, (batch_size,))
        
        # Training step
        output = model.training_step(batch)
        
        # Check output structure
        assert isinstance(output, dict)
        assert "loss" in output
        assert isinstance(output["loss"], torch.Tensor)
        assert output["loss"].dim() == 0  # Scalar loss
        
    def test_validation_step(self):
        """Validation step 테스트"""
        model = DraemSevNet()
        model.eval()
        
        # Initialize validation collections
        model._val_mask_scores = []
        model._val_severity_scores = []
        model._val_final_scores = []
        model._val_labels = []
        
        batch_size = 4
        batch = MagicMock()
        batch.image = torch.randn(batch_size, 3, 224, 224)
        batch.gt_label = torch.randint(0, 2, (batch_size,))
        
        # Validation step
        with torch.no_grad():
            output = model.validation_step(batch)
        
        # Check output and internal state
        assert isinstance(output, dict)
        assert "final_score" in output
        
        # Check that scores were collected
        assert len(model._val_mask_scores) == batch_size
        assert len(model._val_severity_scores) == batch_size
        assert len(model._val_final_scores) == batch_size
        assert len(model._val_labels) == batch_size
        
    def test_on_validation_epoch_end(self):
        """Validation epoch end 테스트"""
        model = DraemSevNet()
        
        # Simulate collected validation data
        batch_size = 10
        model._val_mask_scores = torch.rand(batch_size).tolist()
        model._val_severity_scores = torch.rand(batch_size).tolist()
        model._val_final_scores = torch.rand(batch_size).tolist()
        model._val_labels = torch.randint(0, 2, (batch_size,)).tolist()
        
        # Mock logging method
        logged_metrics = {}
        def mock_log(key, value, **kwargs):
            logged_metrics[key] = value
        model.log = mock_log
        
        # Call validation epoch end
        model.on_validation_epoch_end()
        
        # Check that all AUROC metrics were logged
        expected_keys = [
            "val_mask_AUROC", "val_severity_AUROC", 
            "val_combined_AUROC", "val_image_AUROC"
        ]
        for key in expected_keys:
            assert key in logged_metrics
            assert 0 <= logged_metrics[key] <= 1
        
        # Check that collections were reset
        assert len(model._val_mask_scores) == 0
        assert len(model._val_severity_scores) == 0
        assert len(model._val_final_scores) == 0
        assert len(model._val_labels) == 0
        
    def test_single_class_validation(self):
        """단일 클래스 validation 테스트"""
        model = DraemSevNet()
        
        # All labels are the same (single class)
        batch_size = 5
        model._val_mask_scores = torch.rand(batch_size).tolist()
        model._val_severity_scores = torch.rand(batch_size).tolist()
        model._val_final_scores = torch.rand(batch_size).tolist()
        model._val_labels = [1] * batch_size  # All same class
        
        logged_metrics = {}
        def mock_log(key, value, **kwargs):
            logged_metrics[key] = value
        model.log = mock_log
        
        model.on_validation_epoch_end()
        
        # All AUROCs should be 0.5 (random performance)
        expected_keys = [
            "val_mask_AUROC", "val_severity_AUROC", 
            "val_combined_AUROC", "val_image_AUROC"
        ]
        for key in expected_keys:
            assert logged_metrics[key] == 0.5
            
    def test_different_score_combinations(self):
        """다양한 score combination 방식 테스트"""
        combinations = ["simple_average", "weighted_average", "maximum"]
        
        for combination in combinations:
            model = DraemSevNet(
                score_combination=combination,
                severity_weight_for_combination=0.3
            )
            model.eval()
            
            batch = MagicMock()
            batch.image = torch.randn(2, 3, 224, 224)
            batch.gt_label = torch.randint(0, 2, (2,))
            
            # Should work without errors
            with torch.no_grad():
                output = model.validation_step(batch)
                assert "final_score" in output
                
    def test_severity_head_modes(self):
        """다양한 severity head 모드 테스트"""
        modes = ["single_scale", "multi_scale"]
        
        for mode in modes:
            model = DraemSevNet(severity_head_mode=mode)
            
            batch = MagicMock()
            batch.image = torch.randn(2, 3, 224, 224)
            batch.gt_label = torch.randint(0, 2, (2,))
            
            # Training mode
            model.train()
            train_output = model.training_step(batch)
            assert "loss" in train_output
            
            # Validation mode
            model.eval()
            with torch.no_grad():
                val_output = model.validation_step(batch)
                assert "final_score" in val_output
                
    def test_loss_function_integration(self):
        """Loss function 통합 테스트"""
        model = DraemSevNet(
            severity_weight=0.8,
            severity_loss_type="smooth_l1"
        )
        
        # Check loss type
        
        assert isinstance(model.loss, DraemSevNetLoss)
        assert model.loss.severity_weight == 0.8
        assert model.loss.severity_loss_type == "smooth_l1"
        
    def test_optimizer_configuration(self):
        """Optimizer 설정 테스트"""
        optimizers = ["adam", "adamw", "sgd"]
        
        for opt_name in optimizers:
            model = DraemSevNet(
                optimizer=opt_name,
                learning_rate=2e-4
            )
            
            optimizer = model.configure_optimizers()
            assert hasattr(optimizer, 'param_groups')
            assert optimizer.param_groups[0]['lr'] == 2e-4
            
    def test_synthetic_generator_integration(self):
        """Synthetic generator 통합 테스트"""
        model = DraemSevNet(
            patch_ratio_range=(1.5, 3.0),
            patch_count=2,
            anomaly_probability=0.8
        )
        
        # Check augmenter configuration
        assert model.augmenter.severity_max == 1.0  # Fixed for DRAEM-SevNet
        assert model.augmenter.patch_count == 2
        assert model.augmenter.probability == 0.8
        
    def test_backward_compatibility(self):
        """기존 코드와의 호환성 테스트"""
        model = DraemSevNet()
        
        # Should have val_image_AUROC for backward compatibility
        model._val_mask_scores = [0.1, 0.8, 0.3, 0.9]
        model._val_severity_scores = [0.2, 0.7, 0.4, 0.8]
        model._val_final_scores = [0.15, 0.75, 0.35, 0.85]
        model._val_labels = [0, 1, 0, 1]
        
        logged_metrics = {}
        def mock_log(key, value, **kwargs):
            logged_metrics[key] = value
        model.log = mock_log
        
        model.on_validation_epoch_end()
        
        # val_image_AUROC should exist for backward compatibility
        assert "val_image_AUROC" in logged_metrics
        assert "val_combined_AUROC" in logged_metrics
        # They should be the same (combined score)
        assert logged_metrics["val_image_AUROC"] == logged_metrics["val_combined_AUROC"]


# pytest로 실행 시 자동으로 실행되는 통합 테스트
def test_lightning_model_integration_summary():
    """전체 DRAEM-SevNet Lightning Model 테스트 요약"""
    verbose_print("🧪 DRAEM-SevNet Lightning Model Test Suite Integration Summary", "INFO")
    verbose_print("=" * 70)
    
    # 테스트 구성 요소 확인
    test_components = [
        "Lightning model initialization (default & custom parameters)",
        "Training step functionality (loss calculation, logging)",
        "Validation step functionality (score collection)",
        "Validation epoch end (multi-AUROC calculation)",
        "Single class validation edge case handling",
        "Different score combination methods",
        "Severity head mode configurations",
        "Loss function integration (DraemSevNetLoss)",
        "Optimizer configuration",
        "Synthetic generator integration",
        "Backward compatibility validation"
    ]
    
    verbose_print("Test components covered:")
    for i, component in enumerate(test_components, 1):
        verbose_print(f"  {i:2d}. {component}")
    
    verbose_print(f"\n🎯 Total {len(test_components)} test categories covered!", "SUCCESS")
    verbose_print("\nRun individual tests with: pytest tests/unit/models/image/custom_draem/test_lightning_model_update.py::TestDraemSevNetLightningModel::test_<method_name> -v -s")


if __name__ == "__main__":
    # 직접 실행 시에는 pytest 실행을 권장
    print("\n🧪 DRAEM-SevNet Lightning Model Test Suite")
    print("=" * 60)
    print("To run tests with verbose output:")
    print("pytest tests/unit/models/image/custom_draem/test_lightning_model_update.py -v -s")
    print("\nTo run specific test class:")
    print("pytest tests/unit/models/image/custom_draem/test_lightning_model_update.py::TestDraemSevNetLightningModel -v -s")
    print("\nTo run specific test method:")
    print("pytest tests/unit/models/image/custom_draem/test_lightning_model_update.py::TestDraemSevNetLightningModel::test_model_initialization -v -s")
    print("\n" + "=" * 60)
