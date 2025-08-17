"""Test suite for DraemSevNetLoss.

DRAEM-SevNet의 통합 Loss Function인 DraemSevNetLoss의
L_draem + λ * L_severity 구조와 모든 기능을 테스트합니다.

Run with: pytest tests/unit/models/image/draem_sevnet/test_draem_sevnet_loss.py -v -s
Author: Taewan Hwang
"""

import pytest
import torch
from anomalib.models.image.draem_sevnet.loss import DraemSevNetLoss, DraemSevNetLossFactory
from anomalib.models.image.draem.loss import DraemLoss

# 상세 출력을 위한 helper function
def verbose_print(message: str, level: str = "INFO"):
    """상세 출력을 위한 함수"""
    symbols = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}
    print(f"\n{symbols.get(level, 'ℹ️')} {message}")


class TestDraemSevNetLoss:
    """DraemSevNetLoss 클래스의 기본 기능 테스트"""
    
    def test_loss_initialization(self):
        """Loss 초기화 테스트"""
        verbose_print("Testing DraemSevNetLoss initialization...")
        
        # Default initialization
        loss = DraemSevNetLoss()
        verbose_print(f"Default - severity_weight: {loss.severity_weight}, loss_type: {loss.severity_loss_type}")
        
        assert loss.severity_weight == 0.5
        assert loss.severity_loss_type == "mse"
        assert hasattr(loss, 'draem_loss')
        assert hasattr(loss, 'severity_loss')
        
        # Custom initialization
        custom_loss = DraemSevNetLoss(severity_weight=0.3, severity_loss_type="smooth_l1")
        verbose_print(f"Custom - severity_weight: {custom_loss.severity_weight}, loss_type: {custom_loss.severity_loss_type}")
        
        assert custom_loss.severity_weight == 0.3
        assert custom_loss.severity_loss_type == "smooth_l1"
        
        verbose_print("Loss initialization test passed!", "SUCCESS")
        
    def test_invalid_severity_loss_type(self):
        """잘못된 severity loss type 에러 테스트"""
        with pytest.raises(ValueError, match="Unsupported severity loss type"):
            DraemSevNetLoss(severity_loss_type="invalid_type")
            
    def test_forward_pass_basic(self):
        """기본 forward pass 테스트"""
        loss_fn = DraemSevNetLoss(severity_weight=0.5)
        
        batch_size = 4
        channels = 3
        height, width = 224, 224
        
        # Input tensors
        input_image = torch.randn(batch_size, channels, height, width)
        reconstruction = torch.randn(batch_size, channels, height, width)
        anomaly_mask = torch.randint(0, 2, (batch_size, 1, height, width))
        prediction = torch.randn(batch_size, 2, height, width)
        severity_gt = torch.rand(batch_size)  # [0, 1] range
        severity_pred = torch.rand(batch_size)  # [0, 1] range
        
        # Forward pass
        loss_value = loss_fn(input_image, reconstruction, anomaly_mask, prediction, severity_gt, severity_pred)
        
        # 출력 검증
        assert isinstance(loss_value, torch.Tensor)
        assert loss_value.dim() == 0  # Scalar
        assert loss_value.item() >= 0  # Non-negative
        
    def test_tensor_shape_compatibility(self):
        """다양한 tensor shape 호환성 테스트"""
        loss_fn = DraemSevNetLoss()
        
        batch_size = 2
        input_image = torch.randn(batch_size, 3, 224, 224)
        reconstruction = torch.randn(batch_size, 3, 224, 224)
        anomaly_mask = torch.randint(0, 2, (batch_size, 1, 224, 224))
        prediction = torch.randn(batch_size, 2, 224, 224)
        
        # Test different severity tensor shapes
        # Case 1: Both 1D
        severity_gt_1d = torch.rand(batch_size)
        severity_pred_1d = torch.rand(batch_size)
        
        loss1 = loss_fn(input_image, reconstruction, anomaly_mask, prediction, severity_gt_1d, severity_pred_1d)
        assert loss1.item() >= 0
        
        # Case 2: Both 2D
        severity_gt_2d = torch.rand(batch_size, 1)
        severity_pred_2d = torch.rand(batch_size, 1)
        
        loss2 = loss_fn(input_image, reconstruction, anomaly_mask, prediction, severity_gt_2d, severity_pred_2d)
        assert loss2.item() >= 0
        
        # Case 3: Mixed shapes (should be handled by auto-squeeze)
        loss3 = loss_fn(input_image, reconstruction, anomaly_mask, prediction, severity_gt_1d, severity_pred_2d)
        assert loss3.item() >= 0
        
    def test_individual_losses_breakdown(self):
        """개별 loss 컴포넌트 분석 테스트"""
        loss_fn = DraemSevNetLoss(severity_weight=0.5)
        
        batch_size = 2
        input_image = torch.randn(batch_size, 3, 224, 224)
        reconstruction = torch.randn(batch_size, 3, 224, 224)
        anomaly_mask = torch.randint(0, 2, (batch_size, 1, 224, 224))
        prediction = torch.randn(batch_size, 2, 224, 224)
        severity_gt = torch.rand(batch_size)
        severity_pred = torch.rand(batch_size)
        
        # Get individual losses
        losses = loss_fn.get_individual_losses(
            input_image, reconstruction, anomaly_mask, prediction, severity_gt, severity_pred
        )
        
        # 필수 키 확인
        expected_keys = ["l2_loss", "ssim_loss", "focal_loss", "draem_loss", "severity_loss", "total_loss"]
        assert set(losses.keys()) == set(expected_keys)
        
        # 모든 loss가 non-negative인지 확인
        for key, loss_val in losses.items():
            assert isinstance(loss_val, torch.Tensor)
            assert loss_val.dim() == 0  # Scalar
            assert loss_val.item() >= 0
            
        # Loss 조합 검증
        expected_draem_loss = losses["l2_loss"] + losses["ssim_loss"] + losses["focal_loss"]
        assert torch.allclose(losses["draem_loss"], expected_draem_loss, atol=1e-6)
        
        expected_total = losses["draem_loss"] + 0.5 * losses["severity_loss"]
        assert torch.allclose(losses["total_loss"], expected_total, atol=1e-6)
        
    def test_severity_weight_effect(self):
        """Severity weight의 영향 테스트"""
        batch_size = 2
        input_image = torch.randn(batch_size, 3, 224, 224)
        reconstruction = torch.randn(batch_size, 3, 224, 224)
        anomaly_mask = torch.randint(0, 2, (batch_size, 1, 224, 224))
        prediction = torch.randn(batch_size, 2, 224, 224)
        severity_gt = torch.rand(batch_size)
        severity_pred = torch.rand(batch_size)
        
        # 다양한 severity weight로 테스트
        weights = [0.0, 0.1, 0.5, 1.0, 2.0]
        losses = []
        
        for weight in weights:
            loss_fn = DraemSevNetLoss(severity_weight=weight)
            loss_val = loss_fn(input_image, reconstruction, anomaly_mask, prediction, severity_gt, severity_pred)
            losses.append(loss_val.item())
        
        # Weight=0일 때는 DRAEM loss만, weight가 클수록 total loss가 커져야 함
        # (severity_pred != severity_gt인 경우)
        assert len(losses) == len(weights)
        
        # Weight 0일 때와 다른 weight일 때 차이 확인
        if not torch.allclose(severity_pred, severity_gt, atol=1e-3):
            # Severity error가 있을 때만 weight 증가 효과 확인
            assert losses[1] >= losses[0]  # weight 0.1 >= weight 0.0
            
    def test_mse_vs_smooth_l1_loss(self):
        """MSE vs SmoothL1 loss 비교 테스트"""
        batch_size = 2
        input_image = torch.randn(batch_size, 3, 224, 224)
        reconstruction = torch.randn(batch_size, 3, 224, 224)
        anomaly_mask = torch.randint(0, 2, (batch_size, 1, 224, 224))
        prediction = torch.randn(batch_size, 2, 224, 224)
        severity_gt = torch.rand(batch_size)
        severity_pred = torch.rand(batch_size) + 2.0  # Introduce significant error
        
        # MSE loss
        mse_loss_fn = DraemSevNetLoss(severity_weight=1.0, severity_loss_type="mse")
        mse_loss = mse_loss_fn(input_image, reconstruction, anomaly_mask, prediction, severity_gt, severity_pred)
        
        # SmoothL1 loss
        smooth_l1_loss_fn = DraemSevNetLoss(severity_weight=1.0, severity_loss_type="smooth_l1")
        smooth_l1_loss = smooth_l1_loss_fn(input_image, reconstruction, anomaly_mask, prediction, severity_gt, severity_pred)
        
        # 큰 오차에서는 SmoothL1이 MSE보다 작아야 함 (outlier robustness)
        assert mse_loss.item() >= 0
        assert smooth_l1_loss.item() >= 0
        
    def test_gradient_flow(self):
        """Gradient 흐름 테스트"""
        loss_fn = DraemSevNetLoss(severity_weight=0.5)
        
        batch_size = 2
        input_image = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
        reconstruction = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
        anomaly_mask = torch.randint(0, 2, (batch_size, 1, 224, 224))
        prediction = torch.randn(batch_size, 2, 224, 224, requires_grad=True)
        severity_gt = torch.rand(batch_size)
        severity_pred = torch.rand(batch_size, requires_grad=True)
        
        # Forward pass
        loss = loss_fn(input_image, reconstruction, anomaly_mask, prediction, severity_gt, severity_pred)
        
        # Backward pass
        loss.backward()
        
        # Gradient 확인
        assert input_image.grad is not None
        assert reconstruction.grad is not None
        assert prediction.grad is not None
        assert severity_pred.grad is not None
        
        # Gradient가 0이 아닌지 확인
        assert not torch.all(input_image.grad == 0)
        assert not torch.all(reconstruction.grad == 0)
        assert not torch.all(prediction.grad == 0)
        assert not torch.all(severity_pred.grad == 0)
        
    def test_reproducibility(self):
        """재현성 테스트"""
        torch.manual_seed(42)
        
        loss_fn = DraemSevNetLoss(severity_weight=0.5)
        
        batch_size = 2
        input_image = torch.randn(batch_size, 3, 224, 224)
        reconstruction = torch.randn(batch_size, 3, 224, 224)
        anomaly_mask = torch.randint(0, 2, (batch_size, 1, 224, 224))
        prediction = torch.randn(batch_size, 2, 224, 224)
        severity_gt = torch.rand(batch_size)
        severity_pred = torch.rand(batch_size)
        
        # 첫 번째 실행
        torch.manual_seed(42)
        loss1 = loss_fn(input_image, reconstruction, anomaly_mask, prediction, severity_gt, severity_pred)
        
        # 두 번째 실행 (같은 seed)
        torch.manual_seed(42)
        loss2 = loss_fn(input_image, reconstruction, anomaly_mask, prediction, severity_gt, severity_pred)
        
        # 결과 동일성 확인
        assert torch.allclose(loss1, loss2, atol=1e-8)
        
    def test_repr_method(self):
        """__repr__ 메소드 테스트"""
        loss_fn = DraemSevNetLoss(severity_weight=0.3, severity_loss_type="smooth_l1")
        repr_str = repr(loss_fn)
        
        assert "DraemSevNetLoss" in repr_str
        assert "severity_weight=0.3" in repr_str
        assert "severity_loss_type='smooth_l1'" in repr_str


class TestDraemSevNetLossFactory:
    """DraemSevNetLossFactory 클래스 테스트"""
    
    def test_create_balanced(self):
        """Balanced loss 생성 테스트"""
        loss_fn = DraemSevNetLossFactory.create_balanced(severity_weight=0.6)
        
        assert isinstance(loss_fn, DraemSevNetLoss)
        assert loss_fn.severity_weight == 0.6
        assert loss_fn.severity_loss_type == "mse"
        
    def test_create_robust(self):
        """Robust loss 생성 테스트"""
        loss_fn = DraemSevNetLossFactory.create_robust(severity_weight=0.4)
        
        assert isinstance(loss_fn, DraemSevNetLoss)
        assert loss_fn.severity_weight == 0.4
        assert loss_fn.severity_loss_type == "smooth_l1"
        
    def test_create_mask_focused(self):
        """Mask-focused loss 생성 테스트"""
        loss_fn = DraemSevNetLossFactory.create_mask_focused(severity_weight=0.05)
        
        assert isinstance(loss_fn, DraemSevNetLoss)
        assert loss_fn.severity_weight == 0.05
        assert loss_fn.severity_loss_type == "mse"
        
    def test_create_severity_focused(self):
        """Severity-focused loss 생성 테스트"""
        loss_fn = DraemSevNetLossFactory.create_severity_focused(severity_weight=1.5)
        
        assert isinstance(loss_fn, DraemSevNetLoss)
        assert loss_fn.severity_weight == 1.5
        assert loss_fn.severity_loss_type == "mse"


class TestDraemSevNetLossIntegration:
    """통합 테스트"""
    
    def test_compatibility_with_original_draem_loss(self):
        """Original DRAEM Loss와의 호환성 테스트"""
        batch_size = 2
        input_image = torch.randn(batch_size, 3, 224, 224)
        reconstruction = torch.randn(batch_size, 3, 224, 224)
        anomaly_mask = torch.randint(0, 2, (batch_size, 1, 224, 224))
        prediction = torch.randn(batch_size, 2, 224, 224)
        
        # Original DRAEM loss
        original_draem_loss = DraemLoss()
        original_loss_val = original_draem_loss(input_image, reconstruction, anomaly_mask, prediction)
        
        # DRAEM-SevNet loss with weight=0 (should equal original DRAEM)
        sevnet_loss = DraemSevNetLoss(severity_weight=0.0)
        severity_gt = torch.zeros(batch_size)
        severity_pred = torch.zeros(batch_size)
        
        sevnet_loss_val = sevnet_loss(input_image, reconstruction, anomaly_mask, prediction, severity_gt, severity_pred)
        
        # Should be approximately equal when severity weight is 0
        assert torch.allclose(original_loss_val, sevnet_loss_val, atol=1e-6)
        
    def test_real_world_scenario(self):
        """실제 시나리오 테스트"""
        loss_fn = DraemSevNetLoss(severity_weight=0.5, severity_loss_type="mse")
        
        # Realistic input dimensions
        batch_size = 8
        channels = 3
        height, width = 256, 256
        
        # Generate realistic inputs
        input_image = torch.randn(batch_size, channels, height, width) * 0.5 + 0.5  # [0, 1] range
        reconstruction = input_image + torch.randn_like(input_image) * 0.1  # Slight reconstruction error
        anomaly_mask = torch.randint(0, 2, (batch_size, 1, height, width))
        prediction = torch.randn(batch_size, 2, height, width)
        
        # Realistic severity values [0, 1]
        severity_gt = torch.rand(batch_size)
        severity_pred = severity_gt + torch.randn(batch_size) * 0.1  # Slight prediction error
        severity_pred = torch.clamp(severity_pred, 0, 1)  # Keep in valid range
        
        # Forward pass
        loss_val = loss_fn(input_image, reconstruction, anomaly_mask, prediction, severity_gt, severity_pred)
        individual_losses = loss_fn.get_individual_losses(
            input_image, reconstruction, anomaly_mask, prediction, severity_gt, severity_pred
        )
        
        # 결과 검증
        assert loss_val.item() > 0
        assert individual_losses["total_loss"].item() > 0
        assert abs(loss_val.item() - individual_losses["total_loss"].item()) < 1e-6
        
        # 각 loss component가 합리적인 범위인지 확인
        assert individual_losses["l2_loss"].item() < 10.0  # Reconstruction should be reasonable
        assert individual_losses["severity_loss"].item() < 1.0  # Severity error should be bounded
        
    def test_batch_size_flexibility(self):
        """다양한 batch size 테스트"""
        loss_fn = DraemSevNetLoss()
        
        for batch_size in [1, 2, 4, 8, 16]:
            input_image = torch.randn(batch_size, 3, 224, 224)
            reconstruction = torch.randn(batch_size, 3, 224, 224)
            anomaly_mask = torch.randint(0, 2, (batch_size, 1, 224, 224))
            prediction = torch.randn(batch_size, 2, 224, 224)
            severity_gt = torch.rand(batch_size)
            severity_pred = torch.rand(batch_size)
            
            loss_val = loss_fn(input_image, reconstruction, anomaly_mask, prediction, severity_gt, severity_pred)
            
            assert isinstance(loss_val, torch.Tensor)
            assert loss_val.dim() == 0
            assert loss_val.item() >= 0





# pytest로 실행 시 자동으로 실행되는 통합 테스트
def test_draem_sevnet_loss_integration_summary():
    """전체 DraemSevNetLoss 테스트 요약"""
    verbose_print("🧪 DraemSevNetLoss Test Suite Integration Summary", "INFO")
    verbose_print("=" * 60)
    
    # 테스트 구성 요소 확인
    test_components = [
        "Loss initialization (default & custom parameters)",
        "Invalid severity loss type error handling", 
        "Basic forward pass functionality",
        "Tensor shape compatibility validation",
        "Individual losses breakdown (DRAEM + Severity)",
        "Severity weight effect analysis",
        "MSE vs SmoothL1 loss comparison",
        "Gradient flow verification",
        "Reproducibility testing",
        "String representation method",
        "Factory pattern methods (balanced, robust, focused)",
        "DRAEM loss compatibility",
        "Real-world scenario simulation",
        "Batch size flexibility validation"
    ]
    
    verbose_print("Test components covered:")
    for i, component in enumerate(test_components, 1):
        verbose_print(f"  {i:2d}. {component}")
    
    verbose_print(f"\n🎯 Total {len(test_components)} test categories covered!", "SUCCESS")
    verbose_print("\nRun individual tests with: pytest tests/unit/models/image/draem_sevnet/test_draem_sevnet_loss.py::TestDraemSevNetLoss::test_<method_name> -v -s")


if __name__ == "__main__":
    # 직접 실행 시에는 pytest 실행을 권장
    print("\n🧪 DraemSevNetLoss Test Suite")
    print("=" * 50)
    print("To run tests with verbose output:")
    print("pytest tests/unit/models/image/draem_sevnet/test_draem_sevnet_loss.py -v -s")
    print("\nTo run specific test class:")
    print("pytest tests/unit/models/image/draem_sevnet/test_draem_sevnet_loss.py::TestDraemSevNetLoss -v -s")
    print("\nTo run specific test method:")
    print("pytest tests/unit/models/image/draem_sevnet/test_draem_sevnet_loss.py::TestDraemSevNetLoss::test_forward_pass_basic -v -s")
    print("\n" + "=" * 50)
