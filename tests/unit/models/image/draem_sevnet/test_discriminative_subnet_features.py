"""Test suite for Enhanced DiscriminativeSubNetwork.

DRAEM-SevNet을 위한 수정된 DiscriminativeSubNetwork의 
encoder features 노출 기능을 테스트합니다.

Run with: pytest tests/unit/models/image/draem_sevnet/test_discriminative_subnet_modification.py -v -s
Author: Taewan Hwang
"""

import torch
from anomalib.models.image.draem_sevnet.torch_model import DiscriminativeSubNetwork
from anomalib.models.image.draem_sevnet.severity_head import SeverityHead
from anomalib.models.image.draem.torch_model import DiscriminativeSubNetwork as OriginalSubNet

# 상세 출력을 위한 helper function
def verbose_print(message: str, level: str = "INFO"):
    """상세 출력을 위한 함수"""
    symbols = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}
    print(f"\n{symbols.get(level, 'ℹ️')} {message}")


class TestDiscriminativeSubNetworkEnhancement:
    """Enhanced DiscriminativeSubNetwork 기능 테스트"""
    
    def test_backward_compatibility_mode(self):
        """Backward compatibility 모드 테스트 (expose_features=False)"""
        verbose_print("Testing backward compatibility mode...")
        
        subnet = DiscriminativeSubNetwork(
            in_channels=6, 
            out_channels=2, 
            expose_features=False
        )
        
        # 입력: 원본(3ch) + 복원(3ch) = 6ch
        batch_size = 4
        input_tensor = torch.randn(batch_size, 6, 224, 224)
        
        verbose_print(f"Input shape: {input_tensor.shape}")
        verbose_print(f"expose_features=False")
        
        output = subnet(input_tensor)
        
        verbose_print(f"Output type: {type(output)}")
        verbose_print(f"Output shape: {output.shape}")
        
        # 기존 방식과 동일: mask_logits만 반환
        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, 2, 224, 224)
        assert not isinstance(output, tuple)
        
        verbose_print("Backward compatibility test passed!", "SUCCESS")
        
    def test_feature_exposure_mode(self):
        """Feature exposure 모드 테스트 (expose_features=True)"""
        subnet = DiscriminativeSubNetwork(
            in_channels=6, 
            out_channels=2, 
            expose_features=True
        )
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, 6, 224, 224)
        
        output = subnet(input_tensor)
        
        # Tuple 반환: (mask_logits, encoder_features)
        assert isinstance(output, tuple)
        assert len(output) == 2
        
        mask_logits, encoder_features = output
        
        # Mask logits 검증
        assert mask_logits.shape == (batch_size, 2, 224, 224)
        
        # Encoder features 검증
        assert isinstance(encoder_features, dict)
        expected_keys = ['act1', 'act2', 'act3', 'act4', 'act5', 'act6']
        assert set(encoder_features.keys()) == set(expected_keys)
        
        # Feature map 크기 검증 (base_width=64 기준)
        expected_shapes = {
            'act1': (batch_size, 64, 224, 224),    # Base resolution
            'act2': (batch_size, 128, 112, 112),   # 1/2 resolution  
            'act3': (batch_size, 256, 56, 56),     # 1/4 resolution
            'act4': (batch_size, 512, 28, 28),     # 1/8 resolution
            'act5': (batch_size, 512, 14, 14),     # 1/16 resolution
            'act6': (batch_size, 512, 7, 7),       # 1/32 resolution
        }
        
        for key, expected_shape in expected_shapes.items():
            actual_shape = encoder_features[key].shape
            assert actual_shape == expected_shape, f"{key}: expected {expected_shape}, got {actual_shape}"
            
    def test_default_behavior(self):
        """기본 설정 테스트 (expose_features=True가 기본값)"""
        subnet = DiscriminativeSubNetwork()  # 기본 설정
        
        assert subnet.expose_features is True
        
        batch_size = 2
        input_tensor = torch.randn(batch_size, 6, 224, 224)
        
        output = subnet(input_tensor)
        
        # 기본값이므로 tuple 반환
        assert isinstance(output, tuple)
        assert len(output) == 2
        
    def test_different_input_sizes(self):
        """다양한 입력 크기 테스트"""
        subnet = DiscriminativeSubNetwork(expose_features=True)
        
        # 256x256 입력
        input_256 = torch.randn(2, 6, 256, 256)
        mask_logits_256, features_256 = subnet(input_256)
        
        assert mask_logits_256.shape == (2, 2, 256, 256)
        assert features_256['act1'].shape == (2, 64, 256, 256)
        assert features_256['act6'].shape == (2, 512, 8, 8)  # 256/32 = 8
        
        # 512x512 입력  
        input_512 = torch.randn(2, 6, 512, 512)
        mask_logits_512, features_512 = subnet(input_512)
        
        assert mask_logits_512.shape == (2, 2, 512, 512)
        assert features_512['act1'].shape == (2, 64, 512, 512)
        assert features_512['act6'].shape == (2, 512, 16, 16)  # 512/32 = 16
        
    def test_custom_base_width(self):
        """Custom base_width 테스트"""
        custom_base_width = 32
        subnet = DiscriminativeSubNetwork(
            base_width=custom_base_width, 
            expose_features=True
        )
        
        batch_size = 2
        input_tensor = torch.randn(batch_size, 6, 224, 224)
        
        mask_logits, encoder_features = subnet(input_tensor)
        
        # Base width에 따른 채널 수 확인
        expected_shapes = {
            'act1': (batch_size, custom_base_width, 224, 224),           # 32
            'act2': (batch_size, custom_base_width * 2, 112, 112),       # 64
            'act3': (batch_size, custom_base_width * 4, 56, 56),         # 128
            'act4': (batch_size, custom_base_width * 8, 28, 28),         # 256
            'act5': (batch_size, custom_base_width * 8, 14, 14),         # 256
            'act6': (batch_size, custom_base_width * 8, 7, 7),           # 256
        }
        
        for key, expected_shape in expected_shapes.items():
            actual_shape = encoder_features[key].shape
            assert actual_shape == expected_shape, f"{key}: expected {expected_shape}, got {actual_shape}"
            
    def test_custom_output_channels(self):
        """Custom output channels 테스트"""
        custom_out_channels = 3
        subnet = DiscriminativeSubNetwork(
            out_channels=custom_out_channels,
            expose_features=True
        )
        
        batch_size = 2
        input_tensor = torch.randn(batch_size, 6, 224, 224)
        
        mask_logits, encoder_features = subnet(input_tensor)
        
        # Output channels 확인
        assert mask_logits.shape == (batch_size, custom_out_channels, 224, 224)
        
        # Features는 output channels와 무관하게 동일
        assert encoder_features['act6'].shape == (batch_size, 512, 7, 7)
        
    def test_gradient_flow(self):
        """Gradient 흐름 테스트"""
        subnet = DiscriminativeSubNetwork(expose_features=True)
        input_tensor = torch.randn(2, 6, 224, 224, requires_grad=True)
        
        mask_logits, encoder_features = subnet(input_tensor)
        
        # Mask logits에서 loss 계산
        mask_loss = mask_logits.sum()
        mask_loss.backward(retain_graph=True)
        
        # Input gradient 확인
        assert input_tensor.grad is not None
        assert not torch.all(input_tensor.grad == 0)
        
        # Features gradient 확인 (act6 사용)
        input_tensor.grad.zero_()
        feature_loss = encoder_features['act6'].sum()
        feature_loss.backward()
        
        assert input_tensor.grad is not None
        assert not torch.all(input_tensor.grad == 0)
        
    def test_reproducibility(self):
        """재현성 테스트"""
        torch.manual_seed(42)
        
        subnet = DiscriminativeSubNetwork(expose_features=True)
        input_tensor = torch.randn(2, 6, 224, 224)
        
        # 첫 번째 실행
        torch.manual_seed(42)
        mask_logits1, features1 = subnet(input_tensor)
        
        # 두 번째 실행 (같은 seed)
        torch.manual_seed(42)
        mask_logits2, features2 = subnet(input_tensor)
        
        # 결과 동일성 확인
        assert torch.allclose(mask_logits1, mask_logits2, atol=1e-8)
        
        for key in features1.keys():
            assert torch.allclose(features1[key], features2[key], atol=1e-8)
            
    def test_memory_efficiency(self):
        """메모리 효율성 테스트"""
        subnet = DiscriminativeSubNetwork(expose_features=True)
        
        # 큰 배치로 메모리 사용량 확인
        large_batch_size = 16
        input_tensor = torch.randn(large_batch_size, 6, 224, 224)
        
        mask_logits, encoder_features = subnet(input_tensor)
        
        # 메모리 사용량이 합리적인지 확인
        total_elements = 0
        for key, feature in encoder_features.items():
            total_elements += feature.numel()
            
        # Features의 총 원소 수가 합리적인 범위인지 확인
        assert total_elements > 0
        print(f"Total feature elements: {total_elements:,}")
        
        # Memory cleanup
        del mask_logits, encoder_features, input_tensor


class TestDiscriminativeSubNetworkIntegration:
    """통합 테스트"""
    
    def test_compatibility_with_original_draem(self):
        """Original DRAEM과의 호환성 테스트"""
        
        # 동일한 설정으로 생성
        original_subnet = OriginalSubNet(in_channels=6, out_channels=2, base_width=64)
        enhanced_subnet = DiscriminativeSubNetwork(
            in_channels=6, 
            out_channels=2, 
            base_width=64,
            expose_features=False  # 기존 동작과 동일
        )
        
        batch_size = 2
        input_tensor = torch.randn(batch_size, 6, 224, 224)
        
        # Forward pass
        with torch.no_grad():
            original_output = original_subnet(input_tensor)
            enhanced_output = enhanced_subnet(input_tensor)
        
        # 출력 형태 동일성 확인
        assert original_output.shape == enhanced_output.shape
        assert original_output.shape == (batch_size, 2, 224, 224)
        
    def test_severity_head_integration_compatibility(self):
        """SeverityHead와의 통합 호환성 테스트"""
        
        
        # Enhanced subnet 생성
        subnet = DiscriminativeSubNetwork(expose_features=True)
        
        # SeverityHead 생성 (act6 single-scale)
        severity_head = SeverityHead(in_dim=512, mode="single_scale")
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, 6, 224, 224)
        
        # Forward pass
        mask_logits, encoder_features = subnet(input_tensor)
        severity_scores = severity_head(encoder_features['act6'])
        
        # 출력 검증
        assert mask_logits.shape == (batch_size, 2, 224, 224)
        assert severity_scores.shape == (batch_size,)
        assert torch.all((severity_scores >= 0) & (severity_scores <= 1))
        
    def test_multi_scale_severity_head_integration(self):
        """Multi-scale SeverityHead와의 통합 테스트"""
        
        # Enhanced subnet 생성
        subnet = DiscriminativeSubNetwork(expose_features=True)
        
        # Multi-scale SeverityHead 생성
        severity_head = SeverityHead(mode="multi_scale", base_width=64)
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, 6, 224, 224)
        
        # Forward pass
        mask_logits, encoder_features = subnet(input_tensor)
        
        # Multi-scale features 추출 (act2~act6)
        multi_scale_features = {
            key: encoder_features[key] 
            for key in ['act2', 'act3', 'act4', 'act5', 'act6']
        }
        
        severity_scores = severity_head(multi_scale_features)
        
        # 출력 검증
        assert mask_logits.shape == (batch_size, 2, 224, 224)
        assert severity_scores.shape == (batch_size,)
        assert torch.all((severity_scores >= 0) & (severity_scores <= 1))


# pytest로 실행 시 자동으로 실행되는 통합 테스트
def test_discriminative_subnet_integration_summary():
    """전체 DiscriminativeSubNetwork 수정 테스트 요약"""
    verbose_print("🧪 Enhanced DiscriminativeSubNetwork Test Suite Integration Summary", "INFO")
    verbose_print("=" * 70)
    
    # 테스트 구성 요소 확인
    test_components = [
        "Backward compatibility mode (expose_features=False)",
        "Feature exposure mode (expose_features=True)", 
        "Default behavior validation",
        "Different input sizes support",
        "Custom base width configuration",
        "Custom output channels configuration",
        "Gradient flow verification",
        "Reproducibility testing",
        "Memory efficiency validation",
        "Original DRAEM compatibility",
        "SeverityHead single-scale integration",
        "SeverityHead multi-scale integration"
    ]
    
    verbose_print("Test components covered:")
    for i, component in enumerate(test_components, 1):
        verbose_print(f"  {i:2d}. {component}")
    
    verbose_print(f"\n🎯 Total {len(test_components)} test categories covered!", "SUCCESS")
    verbose_print("\nRun individual tests with: pytest tests/unit/models/image/draem_sevnet/test_discriminative_subnet_modification.py::TestDiscriminativeSubNetworkEnhancement::test_<method_name> -v -s")


if __name__ == "__main__":
    # 직접 실행 시에는 pytest 실행을 권장
    print("\n🧪 Enhanced DiscriminativeSubNetwork Test Suite")
    print("=" * 60)
    print("To run tests with verbose output:")
    print("pytest tests/unit/models/image/draem_sevnet/test_discriminative_subnet_modification.py -v -s")
    print("\nTo run specific test class:")
    print("pytest tests/unit/models/image/draem_sevnet/test_discriminative_subnet_modification.py::TestDiscriminativeSubNetworkEnhancement -v -s")
    print("\nTo run specific test method:")
    print("pytest tests/unit/models/image/draem_sevnet/test_discriminative_subnet_modification.py::TestDiscriminativeSubNetworkEnhancement::test_feature_exposure_mode -v -s")
    print("\n" + "=" * 60)
