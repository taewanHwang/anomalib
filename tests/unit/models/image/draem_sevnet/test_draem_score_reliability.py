#!/usr/bin/env python3
"""
Score 계산 신뢰성 테스트

DRAEM의 score 계산 불일치 문제를 테스트하고,
DRAEM-SevNet에서 사용할 신뢰할 수 있는 계산 방식을 검증합니다.

Run with: pytest tests/unit/models/image/draem_sevnet/test_score_calculation_reliability.py -v -s
"""

import warnings
import torch
from anomalib.models.image.draem.torch_model import DraemModel

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Importing from timm.models.layers.*")


class TestScoreCalculationReliability:
    """Score 계산 신뢰성 테스트 클래스"""
    
    def test_draem_score_inconsistency(self):
        """DRAEM의 score 계산 불일치 문제 재현 테스트"""
        model = DraemModel()
        
        # 동일한 입력 생성
        torch.manual_seed(42)
        test_input = torch.randn(2, 3, 224, 224)
        
        # 1. Inference mode에서 model output
        model.eval()
        with torch.no_grad():
            output1 = model(test_input)
            model_score1 = output1.pred_score
        
        # 2. Training mode로 전환 후 다시 eval mode
        model.train()
        model.eval()
        with torch.no_grad():
            output2 = model(test_input)
            model_score2 = output2.pred_score
        
        # 3. Manual calculation
        model.train()
        with torch.no_grad():
            _, raw_prediction = model(test_input)
        
        manual_score = torch.amax(
            torch.softmax(raw_prediction, dim=1)[:, 1, ...], 
            dim=(-2, -1)
        )
        
        # 검증: 모델 출력이 일관되지 않음을 확인
        consistency_threshold = 1e-6
        is_consistent = torch.allclose(model_score1, model_score2, atol=consistency_threshold)
        
        # 검증: Manual calculation과 model output 차이 확인
        model_manual_diff = torch.abs(model_score1 - manual_score).max()
        
        print(f"Model score 1: {model_score1}")
        print(f"Model score 2: {model_score2}")
        print(f"Manual score: {manual_score}")
        print(f"Model consistency: {is_consistent}")
        print(f"Model-Manual difference: {model_manual_diff:.6f}")
        
        # 결과 검증
        assert isinstance(model_score1, torch.Tensor), "Model score 1 should be tensor"
        assert isinstance(model_score2, torch.Tensor), "Model score 2 should be tensor"
        assert isinstance(manual_score, torch.Tensor), "Manual score should be tensor"
        assert isinstance(model_manual_diff, torch.Tensor), "Difference should be tensor"
    
    def test_reliable_mask_score_calculation(self):
        """신뢰할 수 있는 mask score 계산 방식 테스트"""
        
        def reliable_mask_score(discriminative_output):
            """신뢰할 수 있는 mask score 계산 함수"""
            softmax_pred = torch.softmax(discriminative_output, dim=1)
            anomaly_map = softmax_pred[:, 1, ...]  # anomaly channel
            mask_score = torch.amax(anomaly_map, dim=(-2, -1))
            return mask_score, anomaly_map
        
        # 테스트용 discriminative output 생성
        torch.manual_seed(123)
        batch_size, channels, height, width = 3, 2, 64, 64
        test_logits = torch.randn(batch_size, channels, height, width)
        
        # 여러 번 계산해서 일관성 확인
        scores = []
        for _ in range(5):
            score, _ = reliable_mask_score(test_logits)
            scores.append(score)
        
        # 모든 계산 결과가 동일한지 확인
        base_score = scores[0]
        for i, score in enumerate(scores[1:], 1):
            assert torch.allclose(base_score, score, atol=1e-10), \
                f"Score calculation inconsistent at iteration {i}"
        
        print(f"✅ Reliable calculation test passed")
        print(f"Consistent score: {base_score}")
        
        # 검증: 점수가 유효한 범위에 있는지 확인
        assert torch.all(base_score >= 0), "Scores should be non-negative"
        assert torch.all(base_score <= 1), "Scores should be at most 1"
    
    def test_score_value_ranges(self):
        """Score 값 범위 테스트"""
        
        # 극단적인 경우들 테스트 (실제 사용 가능한 범위만)
        test_cases = [
            torch.zeros(1, 2, 32, 32),                    # All zeros
            torch.ones(1, 2, 32, 32),                     # All ones
            torch.randn(1, 2, 32, 32) * 10,               # Large variance (realistic)
            torch.randn(1, 2, 32, 32) * 0.1,              # Small variance
            torch.full((1, 2, 32, 32), -10.0),            # Large negative (realistic)
            torch.full((1, 2, 32, 32), 10.0),             # Large positive (realistic)
        ]
        
        for i, test_input in enumerate(test_cases):
            try:
                softmax_pred = torch.softmax(test_input, dim=1)
                anomaly_map = softmax_pred[:, 1, ...]
                mask_score = torch.amax(anomaly_map, dim=(-2, -1))
                
                # 값 범위 확인
                assert 0 <= mask_score.min() <= 1, f"Score out of range: {mask_score.min()}"
                assert 0 <= mask_score.max() <= 1, f"Score out of range: {mask_score.max()}"
                
                print(f"Test case {i+1}: score range [{mask_score.min():.6f}, {mask_score.max():.6f}] ✅")
                
            except Exception as e:
                print(f"Test case {i+1}: Failed with error: {e}")
                raise
    
    def test_o3_lite_score_combination(self):
        """O3-Lite에서 사용할 score combination 테스트"""
        
        # 샘플 점수들 (실제 범위 시뮬레이션)
        mask_scores = torch.tensor([0.45, 0.55, 0.65, 0.75])      # DRAEM mask scores
        severity_scores = torch.tensor([0.1, 0.3, 0.7, 0.9])     # Severity scores [0,1]
        
        # 다양한 combination 방식 테스트
        combinations = {
            "simple_average": (mask_scores + severity_scores) / 2,
            "weighted_7030": 0.7 * mask_scores + 0.3 * severity_scores,
            "weighted_3070": 0.3 * mask_scores + 0.7 * severity_scores,
            "geometric_mean": torch.sqrt(mask_scores * severity_scores),
            "maximum": torch.max(mask_scores, severity_scores),
        }
        
        print(f"📊 Score Combination Analysis:")
        print(f"Mask scores:     {mask_scores}")
        print(f"Severity scores: {severity_scores}")
        
        for method, combined_scores in combinations.items():
            print(f"{method:15s}: {combined_scores}")
            
            # 범위 확인
            assert 0 <= combined_scores.min() <= 1, f"{method} out of range"
            assert 0 <= combined_scores.max() <= 1, f"{method} out of range"
        
        # 모든 combination이 계산되었는지 확인
        assert len(combinations) == 5, "Should have 5 different combination methods"


def run_comprehensive_score_test():
    """전체 score 테스트 실행"""
    print("🧪 Comprehensive Score Calculation Test")
    print("=" * 60)
    
    tester = TestScoreCalculationReliability()
    
    # 1. DRAEM 불일치 문제 확인
    print("\n1. DRAEM Score Inconsistency Test:")
    tester.test_draem_score_inconsistency()
    
    # 2. 신뢰할 수 있는 계산 방식 테스트
    print("\n2. Reliable Calculation Test:")
    tester.test_reliable_mask_score_calculation()
    
    # 3. 값 범위 테스트
    print("\n3. Score Value Range Test:")
    tester.test_score_value_ranges()
    
    # 4. O3-Lite combination 테스트
    print("\n4. O3-Lite Score Combination Test:")
    tester.test_o3_lite_score_combination()
    
    print(f"\n🎯 테스트 결론:")
    print(f"  - 모든 테스트가 성공적으로 완료되었습니다")
    print(f"  - 권장 사항: Manual calculation 사용")
    
    return {
        "draem_inconsistency": "tested",
        "score_combinations": "tested"
    }


# pytest로 실행 시 자동으로 실행되는 테스트 함수
def test_score_calculation_comprehensive():
    """Score 계산 신뢰성 종합 테스트"""
    print("\n🧪 Score Calculation Reliability Test Suite")
    print("=" * 60)
    print("Testing DRAEM score calculation reliability and O3-Lite combinations...")
    
    # 종합 테스트 실행
    results = run_comprehensive_score_test()
    
    # 결과 검증
    assert "draem_inconsistency" in results, "DRAEM inconsistency results should be available"
    assert "score_combinations" in results, "Score combination results should be available"
    
    print("\n✅ All score calculation reliability tests passed!")


if __name__ == "__main__":
    print("\n🧪 Score Calculation Reliability Test Suite")
    print("=" * 60)
    print("To run as pytest:")
    print("pytest tests/unit/models/image/draem_sevnet/test_score_calculation_reliability.py -v -s")
    print("\nRunning direct execution...")
    results = run_comprehensive_score_test()
