"""Test suite for DRAEM-SevNet DraemSevNetModel.

DRAEM-SevNet 아키텍처로 완전 재작성된 DraemSevNetModel의
모든 기능과 통합성을 테스트합니다.

Run with: pytest tests/unit/models/image/draem_sevnet/test_draem_sevnet_torch_model.py -v -s
Author: Taewan Hwang
"""

import pytest
import torch
from anomalib.models.image.draem_sevnet.torch_model import DraemSevNetModel, DraemSevNetOutput

# 상세 출력을 위한 helper function
def verbose_print(message: str, level: str = "INFO"):
    """상세 출력을 위한 함수"""
    symbols = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}
    print(f"\n{symbols.get(level, 'ℹ️')} {message}")


class TestDraemSevNetModelRewrite:
    """재작성된 DraemSevNetModel 기본 기능 테스트"""
    
    def test_model_initialization(self):
        """모델 초기화 테스트"""
        verbose_print("Testing DRAEM-SevNet model initialization...")
        
        # Default initialization
        model = DraemSevNetModel()
        verbose_print(f"Default - severity_head_mode: {model.severity_head_mode}")
        verbose_print(f"Default - score_combination: {model.score_combination}")
        verbose_print(f"Default - severity_weight: {model.severity_weight_for_combination}")
        
        assert model.severity_head_mode == "single_scale"
        assert model.score_combination == "simple_average"
        assert model.severity_weight_for_combination == 0.5
        assert hasattr(model, 'reconstructive_subnetwork')
        assert hasattr(model, 'discriminative_subnetwork')
        assert hasattr(model, 'severity_head')
        
        # Custom initialization
        custom_model = DraemSevNetModel(
            severity_head_mode="multi_scale",
            score_combination="weighted_average",
            severity_weight_for_combination=0.3
        )
        verbose_print(f"Custom - severity_head_mode: {custom_model.severity_head_mode}")
        verbose_print(f"Custom - score_combination: {custom_model.score_combination}")
        verbose_print(f"Custom - severity_weight: {custom_model.severity_weight_for_combination}")
        
        assert custom_model.severity_head_mode == "multi_scale"
        assert custom_model.score_combination == "weighted_average"
        assert custom_model.severity_weight_for_combination == 0.3
        
        verbose_print("Model initialization test passed!", "SUCCESS")
        
    def test_invalid_parameters(self):
        """잘못된 파라미터 에러 테스트"""
        # Invalid severity_head_mode
        with pytest.raises(ValueError, match="Unsupported severity_head_mode"):
            DraemSevNetModel(severity_head_mode="invalid_mode")
            
        # Invalid score_combination (tested in forward pass)
        model = DraemSevNetModel(score_combination="invalid_combination")
        model.eval()
        input_tensor = torch.randn(2, 3, 224, 224)
        
        with pytest.raises(ValueError, match="Unsupported score_combination"):
            model(input_tensor)
            
    def test_training_mode_forward(self):
        """Training 모드 forward pass 테스트"""
        for mode in ["single_scale", "multi_scale"]:
            model = DraemSevNetModel(severity_head_mode=mode)
            model.train()
            
            batch_size = 4
            input_tensor = torch.randn(batch_size, 3, 224, 224)
            
            output = model(input_tensor)
            
            # Training mode returns tuple
            assert isinstance(output, tuple)
            assert len(output) == 3
            
            reconstruction, mask_logits, severity_score = output
            
            # Shape verification
            assert reconstruction.shape == (batch_size, 3, 224, 224)
            assert mask_logits.shape == (batch_size, 2, 224, 224)
            assert severity_score.shape == (batch_size,)
            
            # Value validity verification (training mode allows real values)
            assert torch.all(torch.isfinite(severity_score))
            
    def test_inference_mode_forward(self):
        """Inference 모드 forward pass 테스트"""
        for mode in ["single_scale", "multi_scale"]:
            model = DraemSevNetModel(severity_head_mode=mode)
            model.eval()
            
            batch_size = 4
            input_tensor = torch.randn(batch_size, 3, 224, 224)
            
            output = model(input_tensor)
            
            # Inference mode returns DraemSevNetOutput
            assert isinstance(output, DraemSevNetOutput)
            
            # Shape verification
            assert output.reconstruction.shape == (batch_size, 3, 224, 224)
            assert output.mask_logits.shape == (batch_size, 2, 224, 224)
            assert output.normalized_severity_score.shape == (batch_size,)
            assert output.raw_severity_score.shape == (batch_size,)
            assert output.mask_score.shape == (batch_size,)
            assert output.final_score.shape == (batch_size,)
            assert output.anomaly_map.shape == (batch_size, 224, 224)
            
            # Value range verification
            assert torch.all((output.normalized_severity_score >= 0) & (output.normalized_severity_score <= 1))
            assert torch.all(output.raw_severity_score >= 0)  # Inference mode clamps to [0, ∞]
            assert torch.all((output.mask_score >= 0) & (output.mask_score <= 1))
            assert torch.all((output.final_score >= 0) & (output.final_score <= 1))
            assert torch.all((output.anomaly_map >= 0) & (output.anomaly_map <= 1))
            
    def test_score_combination_methods(self):
        """Score combination 방법들 테스트"""
        combinations = [
            ("simple_average", 0.5),
            ("weighted_average", 0.3),
            ("weighted_average", 0.7),
            ("maximum", 0.5)
        ]
        
        for combination, weight in combinations:
            model = DraemSevNetModel(
                score_combination=combination,
                severity_weight_for_combination=weight
            )
            model.eval()
            
            batch_size = 2
            input_tensor = torch.randn(batch_size, 3, 224, 224)
            
            output = model(input_tensor)
            
            # Verify final_score is calculated correctly
            mask_score = output.mask_score
            severity_score = output.normalized_severity_score  # Use normalized for final_score
            final_score = output.final_score
            
            if combination == "simple_average":
                expected = (mask_score + severity_score) / 2.0
                assert torch.allclose(final_score, expected, atol=1e-6)
            elif combination == "weighted_average":
                expected = (1 - weight) * mask_score + weight * severity_score
                assert torch.allclose(final_score, expected, atol=1e-6)
            elif combination == "maximum":
                expected = torch.maximum(mask_score, severity_score)
                assert torch.allclose(final_score, expected, atol=1e-6)
                
    def test_mask_score_calculation_reliability(self):
        """Mask score 계산 신뢰성 테스트"""
        model = DraemSevNetModel()
        model.eval()
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        # Multiple runs with same input
        torch.manual_seed(42)
        output1 = model(input_tensor)
        
        torch.manual_seed(42)
        output2 = model(input_tensor)
        
        # Should be identical (deterministic)
        assert torch.allclose(output1.mask_score, output2.mask_score, atol=1e-8)
        assert torch.allclose(output1.final_score, output2.final_score, atol=1e-8)
        
        # Manual verification of mask score calculation
        mask_logits = output1.mask_logits
        expected_mask_score = torch.amax(
            torch.softmax(mask_logits, dim=1)[:, 1, ...], 
            dim=(-2, -1)
        )
        assert torch.allclose(output1.mask_score, expected_mask_score, atol=1e-6)
        
    def test_different_input_sizes(self):
        """다양한 입력 크기 테스트"""
        model = DraemSevNetModel()
        
        input_sizes = [(224, 224), (256, 256), (512, 512)]
        
        for height, width in input_sizes:
            # Training mode
            model.train()
            input_tensor = torch.randn(2, 3, height, width)
            
            reconstruction, mask_logits, severity_score = model(input_tensor)
            
            assert reconstruction.shape == (2, 3, height, width)
            assert mask_logits.shape == (2, 2, height, width)
            assert severity_score.shape == (2,)
            
            # Inference mode
            model.eval()
            output = model(input_tensor)
            
            assert output.reconstruction.shape == (2, 3, height, width)
            assert output.anomaly_map.shape == (2, height, width)
            
    def test_gradient_flow(self):
        """Gradient 흐름 테스트"""
        model = DraemSevNetModel()
        model.train()
        
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
        
        reconstruction, mask_logits, severity_score = model(input_tensor)
        
        # Combined loss for testing
        loss = reconstruction.sum() + mask_logits.sum() + severity_score.sum()
        loss.backward()
        
        # Check gradients
        assert input_tensor.grad is not None
        assert not torch.all(input_tensor.grad == 0)
        
        # Check model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                
    def test_model_parameter_count(self):
        """모델 파라미터 수 확인"""
        single_scale_model = DraemSevNetModel(severity_head_mode="single_scale")
        multi_scale_model = DraemSevNetModel(severity_head_mode="multi_scale")
        
        single_params = sum(p.numel() for p in single_scale_model.parameters())
        multi_params = sum(p.numel() for p in multi_scale_model.parameters())
        
        # Multi-scale should have more parameters (larger SeverityHead)
        assert multi_params > single_params
        
        # Should be reasonable model sizes (not too large)
        assert single_params > 1_000_000  # At least 1M parameters (DRAEM backbone)
        assert single_params < 150_000_000  # Less than 150M parameters
        
        print(f"Single-scale model parameters: {single_params:,}")
        print(f"Multi-scale model parameters: {multi_params:,}")
        
    def test_sspcab_option(self):
        """SSPCAB 옵션 테스트"""
        model_without_sspcab = DraemSevNetModel(sspcab=False)
        model_with_sspcab = DraemSevNetModel(sspcab=True)
        
        # Both should work
        input_tensor = torch.randn(2, 3, 224, 224)
        
        model_without_sspcab.eval()
        output1 = model_without_sspcab(input_tensor)
        
        model_with_sspcab.eval()
        output2 = model_with_sspcab(input_tensor)
        
        # Should produce outputs with same shapes
        assert output1.final_score.shape == output2.final_score.shape
        assert output1.normalized_severity_score.shape == output2.normalized_severity_score.shape


class TestDraemSevNetOutput:
    """DraemSevNetOutput 데이터클래스 테스트"""
    
    def test_dataclass_structure(self):
        """데이터클래스 구조 테스트"""
        batch_size = 2
        height, width = 224, 224
        
        output = DraemSevNetOutput(
            reconstruction=torch.randn(batch_size, 3, height, width),
            mask_logits=torch.randn(batch_size, 2, height, width),
            raw_severity_score=torch.rand(batch_size),
            normalized_severity_score=torch.rand(batch_size),
            mask_score=torch.rand(batch_size),
            final_score=torch.rand(batch_size),
            anomaly_map=torch.rand(batch_size, height, width)
        )
        
        # Verify all attributes exist and have correct shapes
        assert hasattr(output, 'reconstruction')
        assert hasattr(output, 'mask_logits')
        assert hasattr(output, 'raw_severity_score')
        assert hasattr(output, 'normalized_severity_score')
        assert hasattr(output, 'mask_score')
        assert hasattr(output, 'final_score')
        assert hasattr(output, 'anomaly_map')
        
        assert output.reconstruction.shape == (batch_size, 3, height, width)
        assert output.mask_logits.shape == (batch_size, 2, height, width)
        assert output.raw_severity_score.shape == (batch_size,)
        assert output.normalized_severity_score.shape == (batch_size,)
        assert output.mask_score.shape == (batch_size,)
        assert output.final_score.shape == (batch_size,)
        assert output.anomaly_map.shape == (batch_size, height, width)


class TestDraemSevNetModelIntegration:
    """통합 테스트"""
    
    def test_end_to_end_workflow(self):
        """End-to-end 워크플로우 테스트"""
        model = DraemSevNetModel(severity_head_mode="multi_scale")
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 256, 256)
        
        # Training workflow
        model.train()
        
        # Forward pass
        reconstruction, mask_logits, severity_score = model(input_tensor)
        
        # Simulate loss calculation (would use DraemSevNetLoss in practice)
        dummy_gt_mask = torch.randint(0, 2, (batch_size, 1, 256, 256))
        dummy_gt_severity = torch.rand(batch_size)
        
        # Basic loss terms
        recon_loss = torch.nn.functional.mse_loss(reconstruction, input_tensor)
        severity_loss = torch.nn.functional.mse_loss(severity_score, dummy_gt_severity)
        
        total_loss = recon_loss + severity_loss
        
        # Backward pass
        total_loss.backward()
        
        # Inference workflow
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            
        # Verify inference outputs
        assert isinstance(output, DraemSevNetOutput)
        assert torch.all((output.final_score >= 0) & (output.final_score <= 1))
        
    def test_comparison_with_different_modes(self):
        """다른 모드들 간 비교 테스트"""
        input_tensor = torch.randn(4, 3, 224, 224)
        
        single_model = DraemSevNetModel(severity_head_mode="single_scale")
        multi_model = DraemSevNetModel(severity_head_mode="multi_scale")
        
        single_model.eval()
        multi_model.eval()
        
        with torch.no_grad():
            single_output = single_model(input_tensor)
            multi_output = multi_model(input_tensor)
        
        # Both should produce valid outputs
        assert torch.all((single_output.final_score >= 0) & (single_output.final_score <= 1))
        assert torch.all((multi_output.final_score >= 0) & (multi_output.final_score <= 1))
        
        # Outputs might be different due to different feature usage
        # (This is expected behavior, not an error)
        
    def test_batch_size_flexibility(self):
        """다양한 배치 크기 유연성 테스트"""
        model = DraemSevNetModel()
        
        for batch_size in [1, 2, 4, 8, 16]:
            input_tensor = torch.randn(batch_size, 3, 224, 224)
            
            # Training mode
            model.train()
            reconstruction, mask_logits, severity_score = model(input_tensor)
            
            assert reconstruction.shape[0] == batch_size
            assert mask_logits.shape[0] == batch_size
            assert severity_score.shape[0] == batch_size
            
            # Inference mode
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                
            assert output.final_score.shape[0] == batch_size





# pytest로 실행 시 자동으로 실행되는 통합 테스트
def test_draem_sevnet_model_integration_summary():
    """전체 DRAEM-SevNet DraemSevNetModel 테스트 요약"""
    verbose_print("🧪 DRAEM-SevNet DraemSevNetModel Test Suite Integration Summary", "INFO")
    verbose_print("=" * 70)
    
    # 테스트 구성 요소 확인
    test_components = [
        "Model initialization (default & custom parameters)",
        "Invalid parameters error handling", 
        "Training mode forward pass functionality",
        "Inference mode forward pass functionality",
        "Score combination methods (simple_average, weighted_average, maximum)",
        "Mask score calculation reliability",
        "Different input sizes support",
        "Gradient flow verification",
        "Model parameter count validation",
        "SSPCAB option configuration",
        "DraemSevNetOutput dataclass structure",
        "End-to-end workflow integration",
        "Multi-mode comparison testing",
        "Batch size flexibility validation"
    ]
    
    verbose_print("Test components covered:")
    for i, component in enumerate(test_components, 1):
        verbose_print(f"  {i:2d}. {component}")
    
    verbose_print(f"\n🎯 Total {len(test_components)} test categories covered!", "SUCCESS")
    verbose_print("\nRun individual tests with: pytest tests/unit/models/image/draem_sevnet/test_draem_sevnet_torch_model.py::TestDraemSevNetModelRewrite::test_<method_name> -v -s")


if __name__ == "__main__":
    # 직접 실행 시에는 pytest 실행을 권장
    print("\n🧪 DRAEM-SevNet DraemSevNetModel Test Suite")
    print("=" * 60)
    print("To run tests with verbose output:")
    print("pytest tests/unit/models/image/draem_sevnet/test_draem_sevnet_torch_model.py -v -s")
    print("\nTo run specific test class:")
    print("pytest tests/unit/models/image/draem_sevnet/test_draem_sevnet_torch_model.py::TestDraemSevNetModelRewrite -v -s")
    print("\nTo run specific test method:")
    print("pytest tests/unit/models/image/draem_sevnet/test_draem_sevnet_torch_model.py::TestDraemSevNetModelRewrite::test_model_initialization -v -s")
    print("\n" + "=" * 60)
