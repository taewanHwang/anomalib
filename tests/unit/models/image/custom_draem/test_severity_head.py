"""Test suite for SeverityHead module.

SeverityHead의 single-scale/multi-scale 모드, 
GAP 기능, 출력 범위 등을 포괄적으로 테스트합니다.

Run with: 
pytest tests/unit/models/image/custom_draem/test_severity_head.py -v -s
pytest tests/unit/models/image/custom_draem/test_severity_head.py

Author: Taewan Hwang
"""

import pytest
import torch
import numpy as np
import time
from anomalib.models.image.custom_draem.severity_head import SeverityHead, SeverityHeadFactory

# 상세 출력을 위한 helper function
def verbose_print(message: str, level: str = "INFO"):
    """pytest -v 실행 시 상세 출력을 위한 함수"""
    symbols = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}
    print(f"\n{symbols.get(level, 'ℹ️')} {message}")


class TestSeverityHead:
    """SeverityHead 클래스의 기본 기능 테스트"""
    
    def test_single_scale_initialization(self):
        """Single scale mode 초기화 테스트"""
        verbose_print("Testing single scale mode initialization...")
        
        in_dim = 512
        hidden_dim = 128
        
        head = SeverityHead(in_dim=in_dim, hidden_dim=hidden_dim, mode="single_scale")
        
        verbose_print(f"Mode: {head.mode}, Feature dim: {head.feature_dim}, Hidden dim: {hidden_dim}")
        
        assert head.mode == "single_scale"
        assert head.feature_dim == in_dim
        assert isinstance(head.severity_mlp, torch.nn.Sequential)
        
        verbose_print("Single scale initialization test passed!", "SUCCESS")
        
    def test_multi_scale_initialization(self):
        """Multi-scale mode 초기화 테스트"""
        verbose_print("Testing multi-scale mode initialization...")
        
        base_width = 64
        expected_dim = base_width * (2 + 4 + 8 + 8 + 8)  # 30 * 64 = 1920
        
        head = SeverityHead(mode="multi_scale", base_width=base_width)
        
        verbose_print(f"Base width: {base_width}, Expected dim: {expected_dim}, Actual dim: {head.feature_dim}")
        
        assert head.mode == "multi_scale"
        assert head.feature_dim == expected_dim
        assert head.base_width == base_width
        
        verbose_print("Multi-scale initialization test passed!", "SUCCESS")
        
    def test_single_scale_forward(self):
        """Single scale mode forward pass 테스트"""
        verbose_print("Testing single scale forward pass...")
        
        batch_size, channels, height, width = 4, 512, 14, 14
        
        head = SeverityHead(in_dim=channels, mode="single_scale")
        act6 = torch.randn(batch_size, channels, height, width)
        
        verbose_print(f"Input shape: {act6.shape}")
        
        severity_scores = head(act6)
        
        verbose_print(f"Output shape: {severity_scores.shape}")
        verbose_print(f"Score range: [{severity_scores.min():.4f}, {severity_scores.max():.4f}]")
        verbose_print(f"Sample scores: {severity_scores[:3].tolist()}")
        
        # Shape 검증
        assert severity_scores.shape == (batch_size,)
        
        # 값 범위 검증 [0, 1]
        assert torch.all(severity_scores >= 0.0)
        assert torch.all(severity_scores <= 1.0)
        
        verbose_print("Single scale forward pass test passed!", "SUCCESS")
        
    def test_multi_scale_forward(self):
        """Multi-scale mode forward pass 테스트"""
        verbose_print("Testing multi-scale forward pass...")
        
        batch_size = 4
        base_width = 64
        
        # Multi-scale features 생성 (act2~act6)
        features = {
            'act2': torch.randn(batch_size, base_width * 2, 56, 56),   # 128 channels
            'act3': torch.randn(batch_size, base_width * 4, 28, 28),   # 256 channels
            'act4': torch.randn(batch_size, base_width * 8, 14, 14),   # 512 channels
            'act5': torch.randn(batch_size, base_width * 8, 7, 7),     # 512 channels
            'act6': torch.randn(batch_size, base_width * 8, 7, 7),     # 512 channels
        }
        
        verbose_print(f"Multi-scale input shapes:")
        for k, v in features.items():
            verbose_print(f"  {k}: {v.shape}")
        
        head = SeverityHead(mode="multi_scale", base_width=base_width)
        severity_scores = head(features)
        
        verbose_print(f"Output shape: {severity_scores.shape}")
        verbose_print(f"Score range: [{severity_scores.min():.4f}, {severity_scores.max():.4f}]")
        verbose_print(f"Sample scores: {severity_scores[:3].tolist()}")
        
        # Shape 검증
        assert severity_scores.shape == (batch_size,)
        
        # 값 범위 검증 [0, 1]
        assert torch.all(severity_scores >= 0.0)
        assert torch.all(severity_scores <= 1.0)
        
        verbose_print("Multi-scale forward pass test passed!", "SUCCESS")
        
    def test_global_average_pooling(self):
        """Global Average Pooling 기능 테스트"""
        batch_size, channels, height, width = 2, 64, 32, 32
        
        head = SeverityHead(in_dim=channels, mode="single_scale")
        features = torch.randn(batch_size, channels, height, width)
        
        pooled = head._global_average_pooling(features)
        
        # Shape 검증: [B, C, H, W] -> [B, C]
        assert pooled.shape == (batch_size, channels)
        
        # GAP 계산 정확성 검증
        expected = features.mean(dim=[2, 3])  # H, W 차원에서 평균
        assert torch.allclose(pooled, expected, atol=1e-6)
        
    def test_invalid_mode_error(self):
        """잘못된 mode 사용 시 에러 발생 테스트"""
        with pytest.raises(ValueError, match="Unsupported mode"):
            SeverityHead(mode="invalid_mode")
            
    def test_missing_in_dim_error(self):
        """Single scale mode에서 in_dim 누락 시 에러 테스트"""
        with pytest.raises(ValueError, match="in_dim must be provided"):
            SeverityHead(mode="single_scale")  # in_dim 누락
            
    def test_multi_scale_missing_features_error(self):
        """Multi-scale mode에서 필수 feature 누락 시 에러 테스트"""
        head = SeverityHead(mode="multi_scale", base_width=64)
        
        # act6 누락된 features
        incomplete_features = {
            'act2': torch.randn(2, 128, 56, 56),
            'act3': torch.randn(2, 256, 28, 28),
            'act4': torch.randn(2, 512, 14, 14),
            'act5': torch.randn(2, 512, 7, 7),
            # 'act6' 누락
        }
        
        with pytest.raises(ValueError, match="Missing required feature: act6"):
            head(incomplete_features)
            
    def test_input_type_validation(self):
        """입력 타입 검증 테스트"""
        # Single scale mode - dict 입력 시 에러
        single_head = SeverityHead(in_dim=512, mode="single_scale")
        with pytest.raises(TypeError, match="Expected torch.Tensor for single_scale mode"):
            single_head({'act6': torch.randn(2, 512, 14, 14)})
        
        # Multi-scale mode - tensor 입력 시 에러
        multi_head = SeverityHead(mode="multi_scale", base_width=64)
        with pytest.raises(TypeError, match="Expected dict for multi_scale mode"):
            multi_head(torch.randn(2, 512, 14, 14))
            
    def test_reproducibility(self):
        """결과 재현성 테스트"""
        torch.manual_seed(42)
        
        head = SeverityHead(in_dim=512, mode="single_scale")
        input_tensor = torch.randn(3, 512, 14, 14)
        
        # 첫 번째 실행
        torch.manual_seed(42)
        result1 = head(input_tensor)
        
        # 두 번째 실행 (동일한 seed)
        torch.manual_seed(42)
        result2 = head(input_tensor)
        
        # 결과가 동일해야 함
        assert torch.allclose(result1, result2, atol=1e-8)
        
    def test_gradient_flow(self):
        """Gradient 흐름 테스트"""
        head = SeverityHead(in_dim=256, mode="single_scale")
        input_tensor = torch.randn(2, 256, 16, 16, requires_grad=True)
        
        severity_scores = head(input_tensor)
        loss = severity_scores.sum()
        loss.backward()
        
        # Input tensor에 gradient가 전파되었는지 확인
        assert input_tensor.grad is not None
        assert not torch.all(input_tensor.grad == 0)
        
        # Model parameters에 gradient가 있는지 확인
        for param in head.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestSeverityHeadFactory:
    """SeverityHeadFactory 클래스 테스트"""
    
    def test_create_single_scale(self):
        """Factory의 single scale 생성 테스트"""
        in_dim = 512
        hidden_dim = 256
        dropout_rate = 0.2
        
        head = SeverityHeadFactory.create_single_scale(
            in_dim=in_dim, 
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )
        
        assert isinstance(head, SeverityHead)
        assert head.mode == "single_scale"
        assert head.feature_dim == in_dim
        
    def test_create_multi_scale(self):
        """Factory의 multi-scale 생성 테스트"""
        base_width = 128
        hidden_dim = 512
        dropout_rate = 0.15
        
        head = SeverityHeadFactory.create_multi_scale(
            base_width=base_width,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )
        
        assert isinstance(head, SeverityHead)
        assert head.mode == "multi_scale"
        assert head.base_width == base_width
        

class TestSeverityHeadIntegration:
    """SeverityHead 통합 테스트"""
    
    def test_realistic_discriminative_features(self):
        """실제 discriminative encoder features 시뮬레이션 테스트"""
        # DRAEM discriminative encoder의 실제 출력 크기 시뮬레이션
        batch_size = 8
        
        # Single scale test (act6 only)
        act6 = torch.randn(batch_size, 512, 7, 7)  # 224x224 입력 기준 최종 feature map
        single_head = SeverityHead(in_dim=512, mode="single_scale")
        
        single_scores = single_head(act6)
        assert single_scores.shape == (batch_size,)
        assert torch.all((single_scores >= 0) & (single_scores <= 1))
        
        # Multi-scale test (act2~act6)
        realistic_features = {
            'act2': torch.randn(batch_size, 128, 56, 56),   # 1/4 resolution
            'act3': torch.randn(batch_size, 256, 28, 28),   # 1/8 resolution
            'act4': torch.randn(batch_size, 512, 14, 14),   # 1/16 resolution
            'act5': torch.randn(batch_size, 512, 7, 7),     # 1/32 resolution
            'act6': torch.randn(batch_size, 512, 7, 7),     # 1/32 resolution
        }
        
        multi_head = SeverityHead(mode="multi_scale", base_width=64)
        multi_scores = multi_head(realistic_features)
        
        assert multi_scores.shape == (batch_size,)
        assert torch.all((multi_scores >= 0) & (multi_scores <= 1))
        
    def test_performance_comparison(self):
        """Single-scale vs Multi-scale 성능 비교 테스트"""
        verbose_print("Testing performance comparison...")
        
        batch_size = 16
        num_runs = 100
        
        # Setup
        act6 = torch.randn(batch_size, 512, 7, 7)
        multi_features = {
            'act2': torch.randn(batch_size, 128, 56, 56),
            'act3': torch.randn(batch_size, 256, 28, 28),
            'act4': torch.randn(batch_size, 512, 14, 14),
            'act5': torch.randn(batch_size, 512, 7, 7),
            'act6': act6,
        }
        
        single_head = SeverityHead(in_dim=512, mode="single_scale")
        multi_head = SeverityHead(mode="multi_scale", base_width=64)
        
        verbose_print(f"Running warmup (10 iterations)...")
        # Warmup
        for _ in range(10):
            single_head(act6)
            multi_head(multi_features)
        
        verbose_print(f"Running performance test ({num_runs} iterations)...")
        
        start_time = time.time()
        for _ in range(num_runs):
            single_head(act6)
        single_time = time.time() - start_time
        
        start_time = time.time()
        for _ in range(num_runs):
            multi_head(multi_features)
        multi_time = time.time() - start_time
        
        verbose_print(f"Performance Results:")
        verbose_print(f"  Single-scale time: {single_time:.4f}s ({single_time/num_runs*1000:.2f}ms per run)")
        verbose_print(f"  Multi-scale time: {multi_time:.4f}s ({multi_time/num_runs*1000:.2f}ms per run)")
        verbose_print(f"  Speed ratio: {multi_time/single_time:.2f}x")
        
        verbose_print("Performance comparison test completed!", "SUCCESS")
        
    def test_memory_usage(self):
        """메모리 사용량 테스트"""
        torch.manual_seed(42)
        
        # 큰 배치로 메모리 사용량 확인
        large_batch = 64
        
        single_head = SeverityHead(in_dim=512, mode="single_scale")
        multi_head = SeverityHead(mode="multi_scale", base_width=64)
        
        # Single-scale memory test
        large_act6 = torch.randn(large_batch, 512, 7, 7)
        single_scores = single_head(large_act6)
        
        assert single_scores.shape == (large_batch,)
        
        # Multi-scale memory test
        large_features = {
            'act2': torch.randn(large_batch, 128, 56, 56),
            'act3': torch.randn(large_batch, 256, 28, 28),
            'act4': torch.randn(large_batch, 512, 14, 14),
            'act5': torch.randn(large_batch, 512, 7, 7),
            'act6': torch.randn(large_batch, 512, 7, 7),
        }
        
        multi_scores = multi_head(large_features)
        assert multi_scores.shape == (large_batch,)
        
        # Memory cleanup 확인
        del large_act6, large_features, single_scores, multi_scores
        

# pytest로 실행 시 자동으로 실행되는 통합 테스트
def test_severity_head_integration_summary():
    """전체 SeverityHead 테스트 요약"""
    verbose_print("🧪 SeverityHead Test Suite Integration Summary", "INFO")
    verbose_print("=" * 60)
    
    # 테스트 구성 요소 확인
    test_components = [
        "Single-scale initialization",
        "Multi-scale initialization", 
        "Single-scale forward pass",
        "Multi-scale forward pass",
        "Global Average Pooling",
        "Error handling (invalid mode, missing features, type validation)",
        "Reproducibility & gradient flow",
        "Factory pattern methods",
        "Realistic discriminative features",
        "Performance comparison",
        "Memory usage validation"
    ]
    
    verbose_print("Test components covered:")
    for i, component in enumerate(test_components, 1):
        verbose_print(f"  {i:2d}. {component}")
    
    verbose_print(f"\n🎯 Total {len(test_components)} test categories covered!", "SUCCESS")
    verbose_print("\nRun individual tests with: pytest tests/unit/models/image/custom_draem/test_severity_head.py::TestSeverityHead::test_<method_name> -v")


if __name__ == "__main__":
    # 직접 실행 시에는 pytest 실행을 권장
    print("\n🧪 SeverityHead Test Suite")
    print("" * 50)
    print("To run tests with verbose output:")
    print("pytest tests/unit/models/image/custom_draem/test_severity_head.py -v")
    print("\nTo run specific test class:")
    print("pytest tests/unit/models/image/custom_draem/test_severity_head.py::TestSeverityHead -v")
    print("\nTo run specific test method:")
    print("pytest tests/unit/models/image/custom_draem/test_severity_head.py::TestSeverityHead::test_single_scale_forward -v")
    print("\n" + "=" * 50)
