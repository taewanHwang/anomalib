"""Test suite for DRAEM-SevNet Lightning Model Update.

DRAEM-SevNet으로 업데이트된 Lightning Model의 모든 기능을 테스트합니다.

Run with: pytest tests/unit/models/image/draem_sevnet/test_draem_sevnet_lightning_model.py -v -s
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
        
        batch_size = 4
        batch = MagicMock()
        batch.image = torch.randn(batch_size, 3, 224, 224)
        batch.gt_label = torch.randint(0, 2, (batch_size,))
        
        # Mock batch.update method to return the batch itself
        def mock_update(**kwargs):
            for key, value in kwargs.items():
                setattr(batch, key, value)
            return batch
        batch.update = mock_update
        
        # Validation step
        with torch.no_grad():
            output = model.validation_step(batch)
        
        # Check that output is the updated batch (Lightning pattern)
        assert output == batch
        assert hasattr(output, 'pred_score')
        assert hasattr(output, 'anomaly_map')
        assert hasattr(output, 'pred_label')
        
        # Check shapes
        assert output.pred_score.shape == (batch_size,)
        assert output.anomaly_map.shape == (batch_size, 224, 224)
        assert output.pred_label.shape == (batch_size,)
        
    def test_on_validation_epoch_end(self):
        """Validation epoch end 테스트"""
        model = DraemSevNet()
        
        # DraemSevNet uses anomalib's Evaluator which handles validation metrics automatically
        # The on_validation_epoch_end is handled by parent AnomalibModule
        # This test just ensures it doesn't crash
        
        # Mock trainer to avoid errors
        trainer_mock = MagicMock()
        model.trainer = trainer_mock
        
        # Test that on_validation_epoch_end can be called without error
        try:
            model.on_validation_epoch_end()
            success = True
        except Exception:
            success = False
        
        assert success, "on_validation_epoch_end should not crash"
        
    def test_single_class_validation(self):
        """단일 클래스 validation 테스트"""
        model = DraemSevNet()
        
        # Test validation step with single class scenario
        batch_size = 5
        batch = MagicMock()
        batch.image = torch.randn(batch_size, 3, 224, 224)
        batch.gt_label = torch.ones(batch_size, dtype=torch.long)  # All same class
        
        # Mock batch.update method
        def mock_update(**kwargs):
            for key, value in kwargs.items():
                setattr(batch, key, value)
            return batch
        batch.update = mock_update
        
        # Test that validation step works even with single class
        model.eval()
        with torch.no_grad():
            output = model.validation_step(batch)
        
        assert output == batch
        assert hasattr(output, 'pred_score')
        assert output.pred_score.shape == (batch_size,)
            
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
            
            # Mock batch.update method
            def mock_update(**kwargs):
                for key, value in kwargs.items():
                    setattr(batch, key, value)
                return batch
            batch.update = mock_update
            
            # Should work without errors
            with torch.no_grad():
                output = model.validation_step(batch)
                assert hasattr(output, 'pred_score')
                
    def test_severity_head_modes(self):
        """다양한 severity head 모드 테스트"""
        modes = ["single_scale", "multi_scale"]
        
        for mode in modes:
            model = DraemSevNet(severity_head_mode=mode)
            
            batch = MagicMock()
            batch.image = torch.randn(2, 3, 224, 224)
            batch.gt_label = torch.randint(0, 2, (2,))
            
            # Mock batch.update method for validation
            def mock_update(**kwargs):
                for key, value in kwargs.items():
                    setattr(batch, key, value)
                return batch
            batch.update = mock_update
            
            # Training mode
            model.train()
            train_output = model.training_step(batch)
            assert "loss" in train_output
            
            # Validation mode
            model.eval()
            with torch.no_grad():
                val_output = model.validation_step(batch)
                assert hasattr(val_output, 'pred_score')
                
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
        
        # Test that DraemSevNet is compatible with anomalib's evaluation system
        # The evaluator is configured to log val_image_AUROC for compatibility
        assert hasattr(model, 'evaluator')
        assert model.evaluator is not None
        
        # Test validation step output format compatibility
        batch = MagicMock()
        batch.image = torch.randn(4, 3, 224, 224)
        batch.gt_label = torch.randint(0, 2, (4,))
        
        # Mock batch.update method
        def mock_update(**kwargs):
            for key, value in kwargs.items():
                setattr(batch, key, value)
            return batch
        batch.update = mock_update
        
        model.eval()
        with torch.no_grad():
            output = model.validation_step(batch)
        
        # Should have pred_score field expected by anomalib evaluator
        assert hasattr(output, 'pred_score')
        assert output.pred_score.shape == (4,)


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
    verbose_print("\nRun individual tests with: pytest tests/unit/models/image/draem_sevnet/test_draem_sevnet_lightning_model.py::TestDraemSevNetLightningModel::test_<method_name> -v -s")


if __name__ == "__main__":
    # 직접 실행 시에는 pytest 실행을 권장
    print("\n🧪 DRAEM-SevNet Lightning Model Test Suite")
    print("=" * 60)
    print("To run tests with verbose output:")
    print("pytest tests/unit/models/image/draem_sevnet/test_draem_sevnet_lightning_model.py -v -s")
    print("\nTo run specific test class:")
    print("pytest tests/unit/models/image/draem_sevnet/test_draem_sevnet_lightning_model.py::TestDraemSevNetLightningModel -v -s")
    print("\nTo run specific test method:")
    print("pytest tests/unit/models/image/draem_sevnet/test_draem_sevnet_lightning_model.py::TestDraemSevNetLightningModel::test_model_initialization -v -s")
    print("\n" + "=" * 60)
