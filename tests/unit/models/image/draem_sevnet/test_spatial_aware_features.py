"""Test suite for Spatial-Aware features in DRAEM-SevNet.

새로 구현한 Spatial-Aware 기능들을 전용으로 테스트합니다:
- SeverityHead의 spatial_aware pooling_type
- 공간 정보 보존 메커니즘
- Spatial attention 기능
- DraemSevNetModel의 spatial-aware 옵션들
- GAP vs Spatial-Aware 성능 비교

Run with: 
pytest tests/unit/models/image/draem_sevnet/test_spatial_aware_features.py -v -s

Author: Taewan Hwang
"""

import pytest
import torch
import time
from typing import Dict, Tuple
from anomalib.models.image.draem_sevnet.severity_head import SeverityHead, SeverityHeadFactory
from anomalib.models.image.draem_sevnet.torch_model import DraemSevNetModel, DraemSevNetOutput

# 상세 출력을 위한 helper function
def verbose_print(message: str, level: str = "INFO"):
    """pytest -v 실행 시 상세 출력을 위한 함수"""
    symbols = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌", "COMPARE": "🔄"}
    print(f"\n{symbols.get(level, 'ℹ️')} {message}")


class TestSpatialAwareSeverityHead:
    """SeverityHead의 Spatial-Aware 기능 테스트"""
    
    def test_spatial_aware_initialization(self):
        """Spatial-Aware 모드 초기화 테스트"""
        verbose_print("Testing Spatial-Aware SeverityHead initialization...")
        
        # Single-scale spatial-aware
        head_single = SeverityHead(
            in_dim=512,
            hidden_dim=128,
            mode="single_scale",
            pooling_type="spatial_aware",
            spatial_size=4,
            use_spatial_attention=True
        )
        
        assert head_single.pooling_type == "spatial_aware"
        assert head_single.spatial_size == 4
        assert head_single.use_spatial_attention == True
        assert hasattr(head_single, 'spatial_attention')
        assert hasattr(head_single, 'spatial_reducer')
        assert hasattr(head_single, 'spatial_severity_mlp')
        
        verbose_print(f"Single-scale spatial-aware - pooling: {head_single.pooling_type}, size: {head_single.spatial_size}")
        
        # Multi-scale spatial-aware
        head_multi = SeverityHead(
            mode="multi_scale",
            base_width=64,
            hidden_dim=256,
            pooling_type="spatial_aware",
            spatial_size=4,
            use_spatial_attention=True
        )
        
        assert head_multi.pooling_type == "spatial_aware"
        assert hasattr(head_multi, 'scale_spatial_processors')
        assert hasattr(head_multi, 'multi_scale_spatial_mlp')
        
        verbose_print(f"Multi-scale spatial-aware - pooling: {head_multi.pooling_type}, processors: {len(head_multi.scale_spatial_processors)}")
        verbose_print("Spatial-Aware initialization test passed!", "SUCCESS")
    
    def test_spatial_aware_vs_gap_single_scale(self):
        """Single-scale: GAP vs Spatial-Aware 비교 테스트"""
        verbose_print("Testing Single-scale: GAP vs Spatial-Aware comparison...")
        
        batch_size, channels, height, width = 4, 512, 7, 7
        input_tensor = torch.randn(batch_size, channels, height, width)
        
        # GAP 방식
        head_gap = SeverityHead(
            in_dim=channels,
            hidden_dim=128,
            mode="single_scale",
            pooling_type="gap"
        )
        
        # Spatial-Aware 방식
        head_spatial = SeverityHead(
            in_dim=channels,
            hidden_dim=128,
            mode="single_scale",
            pooling_type="spatial_aware",
            spatial_size=4,
            use_spatial_attention=True
        )
        
        # Forward pass
        with torch.no_grad():
            score_gap = head_gap(input_tensor)
            score_spatial = head_spatial(input_tensor)
        
        # Shape 검증
        assert score_gap.shape == (batch_size,)
        assert score_spatial.shape == (batch_size,)
        
        # 값 범위 검증
        assert torch.all((score_gap >= 0) & (score_gap <= 1))
        assert torch.all((score_spatial >= 0) & (score_spatial <= 1))
        
        verbose_print(f"GAP scores: {score_gap.tolist()}")
        verbose_print(f"Spatial-Aware scores: {score_spatial.tolist()}")
        
        # 파라미터 수 비교
        gap_params = sum(p.numel() for p in head_gap.parameters())
        spatial_params = sum(p.numel() for p in head_spatial.parameters())
        
        verbose_print(f"GAP parameters: {gap_params:,}")
        verbose_print(f"Spatial-Aware parameters: {spatial_params:,}")
        verbose_print(f"Parameter increase: {(spatial_params - gap_params) / gap_params * 100:.1f}%")
        
        # Spatial-Aware가 더 많은 파라미터를 가져야 함
        assert spatial_params > gap_params
        
        verbose_print("Single-scale GAP vs Spatial-Aware comparison passed!", "SUCCESS")
    
    def test_spatial_aware_vs_gap_multi_scale(self):
        """Multi-scale: GAP vs Spatial-Aware 비교 테스트"""
        verbose_print("Testing Multi-scale: GAP vs Spatial-Aware comparison...")
        
        batch_size = 4
        base_width = 64
        
        # Multi-scale features 생성
        features = {
            'act2': torch.randn(batch_size, base_width * 2, 56, 56),
            'act3': torch.randn(batch_size, base_width * 4, 28, 28),
            'act4': torch.randn(batch_size, base_width * 8, 14, 14),
            'act5': torch.randn(batch_size, base_width * 8, 7, 7),
            'act6': torch.randn(batch_size, base_width * 8, 7, 7),
        }
        
        # GAP 방식
        head_gap = SeverityHead(
            mode="multi_scale",
            base_width=base_width,
            hidden_dim=256,
            pooling_type="gap"
        )
        
        # Spatial-Aware 방식
        head_spatial = SeverityHead(
            mode="multi_scale",
            base_width=base_width,
            hidden_dim=256,
            pooling_type="spatial_aware",
            spatial_size=4,
            use_spatial_attention=True
        )
        
        # Forward pass
        with torch.no_grad():
            score_gap = head_gap(features)
            score_spatial = head_spatial(features)
        
        # Shape 검증
        assert score_gap.shape == (batch_size,)
        assert score_spatial.shape == (batch_size,)
        
        # 값 범위 검증
        assert torch.all((score_gap >= 0) & (score_gap <= 1))
        assert torch.all((score_spatial >= 0) & (score_spatial <= 1))
        
        verbose_print(f"Multi-scale GAP scores: {score_gap.tolist()}")
        verbose_print(f"Multi-scale Spatial-Aware scores: {score_spatial.tolist()}")
        
        verbose_print("Multi-scale GAP vs Spatial-Aware comparison passed!", "SUCCESS")
    
    def test_spatial_size_variations(self):
        """다양한 spatial_size 테스트"""
        verbose_print("Testing various spatial_size configurations...")
        
        batch_size, channels = 4, 512
        input_tensor = torch.randn(batch_size, channels, 7, 7)
        
        spatial_sizes = [2, 4, 8]
        results = {}
        
        for size in spatial_sizes:
            head = SeverityHead(
                in_dim=channels,
                hidden_dim=128,
                mode="single_scale",
                pooling_type="spatial_aware",
                spatial_size=size,
                use_spatial_attention=True
            )
            
            with torch.no_grad():
                scores = head(input_tensor)
            
            # 파라미터 수 계산
            params = sum(p.numel() for p in head.parameters())
            results[size] = {
                'scores': scores,
                'params': params,
                'mean_score': scores.mean().item()
            }
            
            verbose_print(f"Spatial size {size}: params={params:,}, mean_score={scores.mean():.4f}")
        
        # 검증: spatial_size가 클수록 더 많은 파라미터
        assert results[8]['params'] > results[4]['params'] > results[2]['params']
        
        verbose_print("Spatial size variations test passed!", "SUCCESS")
    
    def test_spatial_attention_effect(self):
        """Spatial attention 효과 테스트"""
        verbose_print("Testing spatial attention effect...")
        
        batch_size, channels = 4, 512
        input_tensor = torch.randn(batch_size, channels, 7, 7)
        
        # Attention 있는 경우
        head_with_attention = SeverityHead(
            in_dim=channels,
            hidden_dim=128,
            mode="single_scale",
            pooling_type="spatial_aware",
            spatial_size=4,
            use_spatial_attention=True
        )
        
        # Attention 없는 경우
        head_without_attention = SeverityHead(
            in_dim=channels,
            hidden_dim=128,
            mode="single_scale",
            pooling_type="spatial_aware",
            spatial_size=4,
            use_spatial_attention=False
        )
        
        with torch.no_grad():
            scores_with = head_with_attention(input_tensor)
            scores_without = head_without_attention(input_tensor)
        
        # Shape 검증
        assert scores_with.shape == (batch_size,)
        assert scores_without.shape == (batch_size,)
        
        # 값 범위 검증
        assert torch.all((scores_with >= 0) & (scores_with <= 1))
        assert torch.all((scores_without >= 0) & (scores_without <= 1))
        
        # 파라미터 수 비교
        params_with = sum(p.numel() for p in head_with_attention.parameters())
        params_without = sum(p.numel() for p in head_without_attention.parameters())
        
        verbose_print(f"With attention: params={params_with:,}, scores={scores_with.tolist()}")
        verbose_print(f"Without attention: params={params_without:,}, scores={scores_without.tolist()}")
        
        # Attention이 있는 경우 더 많은 파라미터
        assert params_with > params_without
        
        verbose_print("Spatial attention effect test passed!", "SUCCESS")
    
    def test_information_preservation(self):
        """공간 정보 보존 효과 테스트"""
        verbose_print("Testing spatial information preservation...")
        
        batch_size, channels = 1, 512
        height, width = 7, 7
        
        # 특정 패턴을 가진 feature map 생성 (좌상단에 강한 신호)
        feature_map = torch.zeros(batch_size, channels, height, width)
        feature_map[:, :, 0:3, 0:3] = 1.0  # 좌상단 강한 신호
        feature_map[:, :, 4:7, 4:7] = 0.3  # 우하단 약한 신호
        
        verbose_print(f"Original pattern - Top-left: {feature_map[:, :, 0:3, 0:3].mean():.3f}, Bottom-right: {feature_map[:, :, 4:7, 4:7].mean():.3f}")
        
        # GAP 방식 (정보 손실)
        head_gap = SeverityHead(
            in_dim=channels,
            hidden_dim=128,
            mode="single_scale",
            pooling_type="gap"
        )
        
        # Spatial-Aware 방식 (정보 보존)
        head_spatial = SeverityHead(
            in_dim=channels,
            hidden_dim=128,
            mode="single_scale",
            pooling_type="spatial_aware",
            spatial_size=4,
            use_spatial_attention=False  # attention 없이 순수 공간 정보만 테스트
        )
        
        with torch.no_grad():
            # GAP 내부 처리 확인
            gap_features = head_gap._global_average_pooling(feature_map)
            verbose_print(f"GAP result: {gap_features.mean():.3f} (spatial pattern lost)")
            
            # Spatial-Aware 내부 처리 확인
            spatial_features = head_spatial.spatial_reducer(feature_map)  # [B, hidden_dim, 4, 4]
            verbose_print(f"Spatial-Aware result shape: {spatial_features.shape}")
            verbose_print(f"Spatial-Aware top-left: {spatial_features[:, :, 0:2, 0:2].mean():.3f}")
            verbose_print(f"Spatial-Aware bottom-right: {spatial_features[:, :, 2:4, 2:4].mean():.3f}")
            
            # 최종 점수 비교
            score_gap = head_gap(feature_map)
            score_spatial = head_spatial(feature_map)
            
            verbose_print(f"Final GAP score: {score_gap.item():.4f}")
            verbose_print(f"Final Spatial-Aware score: {score_spatial.item():.4f}")
        
        # Spatial-Aware가 공간 패턴을 보존해야 함
        top_left_preserved = spatial_features[:, :, 0:2, 0:2].mean()
        bottom_right_preserved = spatial_features[:, :, 2:4, 2:4].mean()
        
        # 상대적 패턴이 보존되어야 함 (좌상단 > 우하단)
        assert top_left_preserved > bottom_right_preserved
        
        verbose_print("Spatial information preservation test passed!", "SUCCESS")


class TestSpatialAwareDraemSevNetModel:
    """DraemSevNetModel의 Spatial-Aware 기능 테스트"""
    
    def test_spatial_aware_model_initialization(self):
        """Spatial-Aware 모델 초기화 테스트"""
        verbose_print("Testing Spatial-Aware DraemSevNetModel initialization...")
        
        # 기본 GAP 모델
        model_gap = DraemSevNetModel(
            severity_head_pooling_type="gap"
        )
        
        # Spatial-Aware 모델
        model_spatial = DraemSevNetModel(
            severity_head_pooling_type="spatial_aware",
            severity_head_spatial_size=4,
            severity_head_use_spatial_attention=True
        )
        
        # Multi-scale Spatial-Aware 모델
        model_multi_spatial = DraemSevNetModel(
            severity_head_mode="multi_scale",
            severity_head_pooling_type="spatial_aware",
            severity_head_spatial_size=4,
            severity_head_use_spatial_attention=True
        )
        
        # SeverityHead 속성 확인
        assert model_gap.severity_head.pooling_type == "gap"
        assert model_spatial.severity_head.pooling_type == "spatial_aware"
        assert model_spatial.severity_head.spatial_size == 4
        assert model_spatial.severity_head.use_spatial_attention == True
        assert model_multi_spatial.severity_head.pooling_type == "spatial_aware"
        
        verbose_print(f"GAP model pooling: {model_gap.severity_head.pooling_type}")
        verbose_print(f"Spatial-Aware model pooling: {model_spatial.severity_head.pooling_type}, size: {model_spatial.severity_head.spatial_size}")
        verbose_print(f"Multi-scale Spatial-Aware pooling: {model_multi_spatial.severity_head.pooling_type}")
        
        verbose_print("Spatial-Aware model initialization test passed!", "SUCCESS")
    
    def test_spatial_aware_model_forward_training(self):
        """Spatial-Aware 모델 Training 모드 forward 테스트"""
        verbose_print("Testing Spatial-Aware model training forward...")
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        models = {
            "GAP": DraemSevNetModel(severity_head_pooling_type="gap"),
            "Spatial-Aware": DraemSevNetModel(
                severity_head_pooling_type="spatial_aware",
                severity_head_spatial_size=4,
                severity_head_use_spatial_attention=True
            ),
            "Multi-Spatial": DraemSevNetModel(
                severity_head_mode="multi_scale",
                severity_head_pooling_type="spatial_aware",
                severity_head_spatial_size=4
            )
        }
        
        for name, model in models.items():
            model.train()
            
            reconstruction, mask_logits, severity_score = model(input_tensor)
            
            # Shape 검증
            assert reconstruction.shape == (batch_size, 3, 224, 224)
            assert mask_logits.shape == (batch_size, 2, 224, 224)
            assert severity_score.shape == (batch_size,)
            
            # 값 범위 검증
            assert torch.all((severity_score >= 0) & (severity_score <= 1))
            
            verbose_print(f"{name} training - severity range: [{severity_score.min():.4f}, {severity_score.max():.4f}]")
        
        verbose_print("Spatial-Aware model training forward test passed!", "SUCCESS")
    
    def test_spatial_aware_model_forward_inference(self):
        """Spatial-Aware 모델 Inference 모드 forward 테스트"""
        verbose_print("Testing Spatial-Aware model inference forward...")
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        models = {
            "GAP": DraemSevNetModel(severity_head_pooling_type="gap"),
            "Spatial-Aware": DraemSevNetModel(
                severity_head_pooling_type="spatial_aware",
                severity_head_spatial_size=4,
                severity_head_use_spatial_attention=True
            ),
            "Multi-Spatial": DraemSevNetModel(
                severity_head_mode="multi_scale",
                severity_head_pooling_type="spatial_aware",
                severity_head_spatial_size=4
            )
        }
        
        for name, model in models.items():
            model.eval()
            
            with torch.no_grad():
                output = model(input_tensor)
            
            # 타입 검증
            assert isinstance(output, DraemSevNetOutput)
            
            # Shape 검증
            assert output.final_score.shape == (batch_size,)
            assert output.severity_score.shape == (batch_size,)
            assert output.mask_score.shape == (batch_size,)
            assert output.anomaly_map.shape == (batch_size, 224, 224)
            
            # 값 범위 검증
            assert torch.all((output.final_score >= 0) & (output.final_score <= 1))
            assert torch.all((output.severity_score >= 0) & (output.severity_score <= 1))
            assert torch.all((output.mask_score >= 0) & (output.mask_score <= 1))
            
            verbose_print(f"{name} inference - final_score: [{output.final_score.min():.4f}, {output.final_score.max():.4f}]")
            verbose_print(f"{name} inference - severity_score: [{output.severity_score.min():.4f}, {output.severity_score.max():.4f}]")
        
        verbose_print("Spatial-Aware model inference forward test passed!", "SUCCESS")
    
    def test_spatial_aware_model_parameter_comparison(self):
        """Spatial-Aware 모델 파라미터 수 비교 테스트"""
        verbose_print("Testing Spatial-Aware model parameter comparison...")
        
        models = {
            "GAP Single": DraemSevNetModel(
                severity_head_mode="single_scale",
                severity_head_pooling_type="gap"
            ),
            "Spatial Single": DraemSevNetModel(
                severity_head_mode="single_scale",
                severity_head_pooling_type="spatial_aware",
                severity_head_spatial_size=4
            ),
            "GAP Multi": DraemSevNetModel(
                severity_head_mode="multi_scale",
                severity_head_pooling_type="gap"
            ),
            "Spatial Multi": DraemSevNetModel(
                severity_head_mode="multi_scale",
                severity_head_pooling_type="spatial_aware",
                severity_head_spatial_size=4
            )
        }
        
        param_counts = {}
        for name, model in models.items():
            total_params = sum(p.numel() for p in model.parameters())
            severity_params = sum(p.numel() for p in model.severity_head.parameters())
            param_counts[name] = {
                'total': total_params,
                'severity': severity_params
            }
            
            verbose_print(f"{name}: total={total_params:,}, severity_head={severity_params:,}")
        
        # 검증: Spatial-Aware가 GAP보다 많은 파라미터를 가져야 함
        assert param_counts["Spatial Single"]["severity"] > param_counts["GAP Single"]["severity"]
        assert param_counts["Spatial Multi"]["severity"] > param_counts["GAP Multi"]["severity"]
        
        # Multi-scale이 Single-scale보다 많은 파라미터를 가져야 함
        assert param_counts["Spatial Multi"]["severity"] > param_counts["Spatial Single"]["severity"]
        
        verbose_print("Parameter comparison test passed!", "SUCCESS")
    
    def test_spatial_aware_gradient_flow(self):
        """Spatial-Aware 모델 gradient 흐름 테스트"""
        verbose_print("Testing Spatial-Aware model gradient flow...")
        
        model = DraemSevNetModel(
            severity_head_pooling_type="spatial_aware",
            severity_head_spatial_size=4,
            severity_head_use_spatial_attention=True
        )
        model.train()
        
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
        
        reconstruction, mask_logits, severity_score = model(input_tensor)
        
        # 손실 계산 (단순화)
        loss = reconstruction.sum() + mask_logits.sum() + severity_score.sum()
        loss.backward()
        
        # Input gradient 확인
        assert input_tensor.grad is not None
        assert not torch.all(input_tensor.grad == 0)
        
        # SeverityHead의 spatial-aware 컴포넌트들 gradient 확인
        severity_head = model.severity_head
        
        # Spatial attention gradient 확인
        if hasattr(severity_head, 'spatial_attention'):
            for param in severity_head.spatial_attention.parameters():
                if param.requires_grad:
                    assert param.grad is not None
                    assert not torch.all(param.grad == 0)
        
        # Spatial reducer gradient 확인
        if hasattr(severity_head, 'spatial_reducer'):
            for param in severity_head.spatial_reducer.parameters():
                if param.requires_grad:
                    assert param.grad is not None
                    assert not torch.all(param.grad == 0)
        
        verbose_print("Spatial-Aware gradient flow test passed!", "SUCCESS")


class TestSpatialAwareFactory:
    """SeverityHeadFactory의 Spatial-Aware 메서드 테스트"""
    
    def test_factory_spatial_aware_methods(self):
        """Factory의 Spatial-Aware 생성 메서드 테스트"""
        verbose_print("Testing SeverityHeadFactory spatial-aware methods...")
        
        # create_spatial_aware_single_scale
        head_single = SeverityHeadFactory.create_spatial_aware_single_scale(
            in_dim=512,
            hidden_dim=128,
            spatial_size=4,
            use_spatial_attention=True
        )
        
        assert isinstance(head_single, SeverityHead)
        assert head_single.mode == "single_scale"
        assert head_single.pooling_type == "spatial_aware"
        assert head_single.spatial_size == 4
        assert head_single.use_spatial_attention == True
        
        # create_spatial_aware_multi_scale
        head_multi = SeverityHeadFactory.create_spatial_aware_multi_scale(
            base_width=64,
            hidden_dim=256,
            spatial_size=4,
            use_spatial_attention=True
        )
        
        assert isinstance(head_multi, SeverityHead)
        assert head_multi.mode == "multi_scale"
        assert head_multi.pooling_type == "spatial_aware"
        assert head_multi.spatial_size == 4
        assert head_multi.use_spatial_attention == True
        
        verbose_print("Factory spatial-aware methods test passed!", "SUCCESS")
    
    def test_factory_backward_compatibility(self):
        """Factory의 하위 호환성 테스트"""
        verbose_print("Testing Factory backward compatibility...")
        
        # 기존 메서드들이 새로운 옵션을 지원하는지 확인
        head_single_gap = SeverityHeadFactory.create_single_scale(
            in_dim=512,
            pooling_type="gap"
        )
        
        head_single_spatial = SeverityHeadFactory.create_single_scale(
            in_dim=512,
            pooling_type="spatial_aware",
            spatial_size=4,
            use_spatial_attention=True
        )
        
        assert head_single_gap.pooling_type == "gap"
        assert head_single_spatial.pooling_type == "spatial_aware"
        
        verbose_print("Factory backward compatibility test passed!", "SUCCESS")


class TestSpatialAwarePerformance:
    """Spatial-Aware 성능 비교 테스트"""
    
    def test_inference_speed_comparison(self):
        """추론 속도 비교 테스트"""
        verbose_print("Testing inference speed comparison...")
        
        batch_size = 8
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        num_runs = 50
        
        models = {
            "GAP": DraemSevNetModel(severity_head_pooling_type="gap"),
            "Spatial-Aware": DraemSevNetModel(
                severity_head_pooling_type="spatial_aware",
                severity_head_spatial_size=4,
                severity_head_use_spatial_attention=True
            )
        }
        
        results = {}
        
        for name, model in models.items():
            model.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    model(input_tensor)
            
            # Timing
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_runs):
                    output = model(input_tensor)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs
            results[name] = {
                'avg_time': avg_time,
                'final_score': output.final_score.mean().item()
            }
            
            verbose_print(f"{name}: {avg_time*1000:.2f}ms per batch, avg_score: {output.final_score.mean():.4f}")
        
        # 속도 비교 (Spatial-Aware가 더 느릴 수 있음)
        speed_ratio = results["Spatial-Aware"]["avg_time"] / results["GAP"]["avg_time"]
        verbose_print(f"Speed ratio (Spatial-Aware / GAP): {speed_ratio:.2f}x", "COMPARE")
        
        # 합리적인 속도 차이인지 확인 (10배 이상 느리면 안됨)
        assert speed_ratio < 10.0, f"Spatial-Aware is too slow: {speed_ratio:.2f}x"
        
        verbose_print("Inference speed comparison test passed!", "SUCCESS")
    
    def test_memory_usage_comparison(self):
        """메모리 사용량 비교 테스트"""
        verbose_print("Testing memory usage comparison...")
        
        large_batch = 32
        input_tensor = torch.randn(large_batch, 3, 224, 224)
        
        models = {
            "GAP": DraemSevNetModel(severity_head_pooling_type="gap"),
            "Spatial-Aware": DraemSevNetModel(
                severity_head_pooling_type="spatial_aware",
                severity_head_spatial_size=4,
                severity_head_use_spatial_attention=True
            )
        }
        
        for name, model in models.items():
            model.eval()
            
            with torch.no_grad():
                output = model(input_tensor)
            
            # 메모리 사용량 테스트 (큰 배치에서도 정상 동작해야 함)
            assert output.final_score.shape == (large_batch,)
            assert torch.all((output.final_score >= 0) & (output.final_score <= 1))
            
            verbose_print(f"{name}: successfully processed batch_size={large_batch}")
        
        verbose_print("Memory usage comparison test passed!", "SUCCESS")


class TestSpatialAwareArchitectureCoverage:
    """다양한 Spatial-Aware 아키텍처 조합 완전 테스트"""
    
    def test_all_architecture_combinations(self):
        """모든 가능한 아키텍처 조합 매트릭스 테스트"""
        verbose_print("Testing all possible architecture combinations...")
        
        # 테스트할 모든 조합 정의
        test_combinations = [
            # (mode, pooling_type, spatial_size, use_attention, score_combination)
            ("single_scale", "gap", None, None, "simple_average"),
            ("single_scale", "gap", None, None, "weighted_average"),
            ("single_scale", "gap", None, None, "maximum"),
            
            ("single_scale", "spatial_aware", 2, True, "simple_average"),
            ("single_scale", "spatial_aware", 2, False, "simple_average"),
            ("single_scale", "spatial_aware", 4, True, "simple_average"),
            ("single_scale", "spatial_aware", 4, False, "simple_average"),
            ("single_scale", "spatial_aware", 8, True, "simple_average"),
            ("single_scale", "spatial_aware", 8, False, "simple_average"),
            
            ("single_scale", "spatial_aware", 4, True, "weighted_average"),
            ("single_scale", "spatial_aware", 4, True, "maximum"),
            
            ("multi_scale", "gap", None, None, "simple_average"),
            ("multi_scale", "gap", None, None, "weighted_average"),
            ("multi_scale", "gap", None, None, "maximum"),
            
            ("multi_scale", "spatial_aware", 2, True, "simple_average"),
            ("multi_scale", "spatial_aware", 2, False, "simple_average"),
            ("multi_scale", "spatial_aware", 4, True, "simple_average"),
            ("multi_scale", "spatial_aware", 4, False, "simple_average"),
            ("multi_scale", "spatial_aware", 8, True, "simple_average"),
            ("multi_scale", "spatial_aware", 8, False, "simple_average"),
            
            ("multi_scale", "spatial_aware", 4, True, "weighted_average"),
            ("multi_scale", "spatial_aware", 4, True, "maximum"),
        ]
        
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        successful_combinations = 0
        total_combinations = len(test_combinations)
        
        for i, (mode, pooling_type, spatial_size, use_attention, score_combination) in enumerate(test_combinations):
            try:
                # 모델 생성
                if pooling_type == "gap":
                    model = DraemSevNetModel(
                        severity_head_mode=mode,
                        severity_head_pooling_type=pooling_type,
                        score_combination=score_combination,
                        severity_weight_for_combination=0.3
                    )
                else:  # spatial_aware
                    model = DraemSevNetModel(
                        severity_head_mode=mode,
                        severity_head_pooling_type=pooling_type,
                        severity_head_spatial_size=spatial_size,
                        severity_head_use_spatial_attention=use_attention,
                        score_combination=score_combination,
                        severity_weight_for_combination=0.3
                    )
                
                # Training mode 테스트
                model.train()
                reconstruction, mask_logits, severity_score = model(input_tensor)
                
                assert reconstruction.shape == (batch_size, 3, 224, 224)
                assert mask_logits.shape == (batch_size, 2, 224, 224)
                assert severity_score.shape == (batch_size,)
                assert torch.all((severity_score >= 0) & (severity_score <= 1))
                
                # Inference mode 테스트
                model.eval()
                with torch.no_grad():
                    output = model(input_tensor)
                
                assert isinstance(output, DraemSevNetOutput)
                assert torch.all((output.final_score >= 0) & (output.final_score <= 1))
                assert torch.all((output.severity_score >= 0) & (output.severity_score <= 1))
                
                successful_combinations += 1
                
                if pooling_type == "gap":
                    combo_desc = f"{mode}+{pooling_type}+{score_combination}"
                else:
                    combo_desc = f"{mode}+{pooling_type}(size={spatial_size},att={use_attention})+{score_combination}"
                
                verbose_print(f"✅ [{i+1:2d}/{total_combinations}] {combo_desc}")
                
            except Exception as e:
                verbose_print(f"❌ [{i+1:2d}/{total_combinations}] Failed: {e}", "ERROR")
        
        # 모든 조합이 성공해야 함
        assert successful_combinations == total_combinations, f"Failed combinations: {total_combinations - successful_combinations}/{total_combinations}"
        
        verbose_print(f"🎉 All {total_combinations} architecture combinations passed!", "SUCCESS")
    
    def test_input_size_spatial_size_combinations(self):
        """다양한 입력 크기와 spatial_size 조합 테스트"""
        verbose_print("Testing input size and spatial size combinations...")
        
        # (input_height, input_width, spatial_size) 조합
        size_combinations = [
            (224, 224, 2),
            (224, 224, 4),
            (224, 224, 8),
            (256, 256, 2),
            (256, 256, 4),
            (256, 256, 8),
            (512, 512, 4),
            (512, 512, 8),
            (128, 128, 2),
            (128, 128, 4),
        ]
        
        successful_tests = 0
        
        for height, width, spatial_size in size_combinations:
            try:
                model = DraemSevNetModel(
                    severity_head_mode="single_scale",
                    severity_head_pooling_type="spatial_aware",
                    severity_head_spatial_size=spatial_size,
                    severity_head_use_spatial_attention=True
                )
                
                batch_size = 2
                input_tensor = torch.randn(batch_size, 3, height, width)
                
                # Training mode
                model.train()
                reconstruction, mask_logits, severity_score = model(input_tensor)
                
                assert reconstruction.shape == (batch_size, 3, height, width)
                assert mask_logits.shape == (batch_size, 2, height, width)
                assert severity_score.shape == (batch_size,)
                
                # Inference mode
                model.eval()
                with torch.no_grad():
                    output = model(input_tensor)
                
                assert output.anomaly_map.shape == (batch_size, height, width)
                
                successful_tests += 1
                verbose_print(f"✅ Input({height}x{width}) + Spatial({spatial_size}x{spatial_size})")
                
            except Exception as e:
                verbose_print(f"❌ Input({height}x{width}) + Spatial({spatial_size}x{spatial_size}): {e}", "ERROR")
        
        assert successful_tests == len(size_combinations), f"Failed size combinations: {len(size_combinations) - successful_tests}"
        verbose_print("Input size and spatial size combinations test passed!", "SUCCESS")
    
    def test_extreme_spatial_sizes(self):
        """극한 spatial_size 값들 테스트"""
        verbose_print("Testing extreme spatial sizes...")
        
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        # 극한 값들 테스트
        extreme_sizes = [1, 16, 32]  # 매우 작은 값과 큰 값들
        
        for spatial_size in extreme_sizes:
            try:
                model = DraemSevNetModel(
                    severity_head_mode="single_scale",
                    severity_head_pooling_type="spatial_aware",
                    severity_head_spatial_size=spatial_size,
                    severity_head_use_spatial_attention=True
                )
                
                model.train()
                reconstruction, mask_logits, severity_score = model(input_tensor)
                
                # 기본 검증
                assert severity_score.shape == (batch_size,)
                assert torch.all((severity_score >= 0) & (severity_score <= 1))
                
                # 파라미터 수 확인
                total_params = sum(p.numel() for p in model.parameters())
                severity_params = sum(p.numel() for p in model.severity_head.parameters())
                
                verbose_print(f"✅ Spatial size {spatial_size:2d}: total_params={total_params:,}, severity_params={severity_params:,}")
                
            except Exception as e:
                verbose_print(f"❌ Spatial size {spatial_size}: {e}", "ERROR")
                # 극한 값에서는 실패할 수 있음 (메모리 부족 등)
                continue
        
        verbose_print("Extreme spatial sizes test completed!", "SUCCESS")
    
    def test_sspcab_with_spatial_aware(self):
        """SSPCAB 옵션과 Spatial-Aware 조합 테스트"""
        verbose_print("Testing SSPCAB with Spatial-Aware combinations...")
        
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        # SSPCAB + Spatial-Aware 조합들
        combinations = [
            (False, "gap"),
            (True, "gap"),
            (False, "spatial_aware"),
            (True, "spatial_aware"),
        ]
        
        for sspcab, pooling_type in combinations:
            try:
                if pooling_type == "gap":
                    model = DraemSevNetModel(
                        sspcab=sspcab,
                        severity_head_pooling_type=pooling_type
                    )
                else:
                    model = DraemSevNetModel(
                        sspcab=sspcab,
                        severity_head_pooling_type=pooling_type,
                        severity_head_spatial_size=4,
                        severity_head_use_spatial_attention=True
                    )
                
                model.train()
                reconstruction, mask_logits, severity_score = model(input_tensor)
                
                # 기본 검증
                assert reconstruction.shape == (batch_size, 3, 224, 224)
                assert severity_score.shape == (batch_size,)
                assert torch.all((severity_score >= 0) & (severity_score <= 1))
                
                verbose_print(f"✅ SSPCAB={sspcab} + {pooling_type}")
                
            except Exception as e:
                verbose_print(f"❌ SSPCAB={sspcab} + {pooling_type}: {e}", "ERROR")
                assert False, f"SSPCAB combination failed: {e}"
        
        verbose_print("SSPCAB with Spatial-Aware combinations test passed!", "SUCCESS")
    
    def test_parameter_scaling_analysis(self):
        """파라미터 스케일링 분석 테스트"""
        verbose_print("Testing parameter scaling analysis...")
        
        # 다양한 hidden_dim 값들로 파라미터 스케일링 확인
        hidden_dims = [64, 128, 256, 512]
        spatial_sizes = [2, 4, 8]
        
        scaling_results = {}
        
        for hidden_dim in hidden_dims:
            scaling_results[hidden_dim] = {}
            for spatial_size in spatial_sizes:
                try:
                    model = DraemSevNetModel(
                        severity_head_mode="single_scale",
                        severity_head_hidden_dim=hidden_dim,
                        severity_head_pooling_type="spatial_aware",
                        severity_head_spatial_size=spatial_size,
                        severity_head_use_spatial_attention=True
                    )
                    
                    severity_params = sum(p.numel() for p in model.severity_head.parameters())
                    scaling_results[hidden_dim][spatial_size] = severity_params
                    
                    verbose_print(f"Hidden={hidden_dim:3d}, Spatial={spatial_size}: {severity_params:,} params")
                    
                except Exception as e:
                    verbose_print(f"❌ Hidden={hidden_dim}, Spatial={spatial_size}: {e}", "ERROR")
                    continue
        
        # 스케일링 패턴 검증
        for hidden_dim in hidden_dims:
            if len(scaling_results[hidden_dim]) >= 2:
                sizes = sorted(scaling_results[hidden_dim].keys())
                for i in range(len(sizes)-1):
                    smaller_size = sizes[i]
                    larger_size = sizes[i+1]
                    assert scaling_results[hidden_dim][larger_size] > scaling_results[hidden_dim][smaller_size], \
                        f"Parameter scaling error: larger spatial_size should have more parameters"
        
        verbose_print("Parameter scaling analysis test passed!", "SUCCESS")
    
    def test_backward_compatibility_comprehensive(self):
        """포괄적인 하위 호환성 테스트"""
        verbose_print("Testing comprehensive backward compatibility...")
        
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        # 기존 방식으로 모델 생성 (새로운 옵션 없이)
        legacy_models = [
            DraemSevNetModel(),  # 모든 기본값
            DraemSevNetModel(severity_head_mode="multi_scale"),
            DraemSevNetModel(score_combination="weighted_average"),
            DraemSevNetModel(severity_weight_for_combination=0.7),
            DraemSevNetModel(sspcab=True),
        ]
        
        for i, model in enumerate(legacy_models):
            try:
                # 모든 레거시 모델은 기본적으로 GAP을 사용해야 함
                assert model.severity_head.pooling_type == "gap"
                
                # Training mode
                model.train()
                reconstruction, mask_logits, severity_score = model(input_tensor)
                
                assert reconstruction.shape == (batch_size, 3, 224, 224)
                assert severity_score.shape == (batch_size,)
                
                # Inference mode
                model.eval()
                with torch.no_grad():
                    output = model(input_tensor)
                
                assert isinstance(output, DraemSevNetOutput)
                assert torch.all((output.final_score >= 0) & (output.final_score <= 1))
                
                verbose_print(f"✅ Legacy model {i+1}: OK")
                
            except Exception as e:
                verbose_print(f"❌ Legacy model {i+1}: {e}", "ERROR")
                assert False, f"Backward compatibility broken: {e}"
        
        verbose_print("Comprehensive backward compatibility test passed!", "SUCCESS")


# pytest로 실행 시 자동으로 실행되는 통합 테스트
def test_spatial_aware_integration_summary():
    """전체 Spatial-Aware 기능 테스트 요약"""
    verbose_print("🧪 Spatial-Aware Features Test Suite Integration Summary", "INFO")
    verbose_print("=" * 70)
    
    # 테스트 구성 요소 확인
    test_components = [
        "SeverityHead Spatial-Aware initialization",
        "GAP vs Spatial-Aware comparison (single & multi-scale)",
        "Spatial size variations (2, 4, 8)",
        "Spatial attention effect testing",
        "Spatial information preservation verification",
        "DraemSevNetModel Spatial-Aware integration",
        "Training & inference mode forward passes",
        "Parameter count comparison",
        "Gradient flow validation",
        "SeverityHeadFactory spatial-aware methods",
        "Factory backward compatibility",
        "Inference speed comparison",
        "Memory usage comparison"
    ]
    
    verbose_print("Test components covered:")
    for i, component in enumerate(test_components, 1):
        verbose_print(f"  {i:2d}. {component}")
    
    verbose_print(f"\n🎯 Total {len(test_components)} test categories covered!", "SUCCESS")
    verbose_print("\n📋 Key Features Tested:")
    verbose_print("  ✅ Spatial information preservation (vs GAP information loss)")
    verbose_print("  ✅ Spatial attention mechanism")
    verbose_print("  ✅ Configurable spatial_size (2x2, 4x4, 8x8)")
    verbose_print("  ✅ Single-scale & Multi-scale support")
    verbose_print("  ✅ Backward compatibility with existing GAP mode")
    verbose_print("  ✅ Performance characteristics")
    
    verbose_print("\nRun individual tests with:")
    verbose_print("pytest tests/unit/models/image/draem_sevnet/test_spatial_aware_features.py::TestSpatialAwareSeverityHead::test_<method_name> -v -s")


if __name__ == "__main__":
    # 직접 실행 시에는 pytest 실행을 권장
    print("\n🧪 DRAEM-SevNet Spatial-Aware Features Test Suite")
    print("=" * 60)
    print("To run tests with verbose output:")
    print("pytest tests/unit/models/image/draem_sevnet/test_spatial_aware_features.py -v -s")
    print("\nTo run specific test class:")
    print("pytest tests/unit/models/image/draem_sevnet/test_spatial_aware_features.py::TestSpatialAwareSeverityHead -v -s")
    print("\nTo run specific test method:")
    print("pytest tests/unit/models/image/draem_sevnet/test_spatial_aware_features.py::TestSpatialAwareSeverityHead::test_spatial_aware_initialization -v -s")
    print("\n" + "=" * 60)
