#!/usr/bin/env python3
"""
모델 파라미터 개수 분석 스크립트
외부 구현(ReconstructiveSubNetwork_256, DiscriminativeSubNetwork_256)과
현재 구현(ReconstructiveSubNetwork, DiscriminativeSubNetwork) 비교
"""

import sys
import torch
import torch.nn as nn

# 외부 모델 import를 위한 경로 추가
sys.path.append('./DRAME_CutPaste')

def count_parameters(model, model_name):
    """모델의 파라미터 개수를 계산하고 출력"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 {model_name}:")
    print(f"   - 총 파라미터: {total_params:,}")
    print(f"   - 학습 가능 파라미터: {trainable_params:,}")
    return total_params, trainable_params

def analyze_layer_details(model, model_name):
    """레이어별 상세 파라미터 분석"""
    print(f"\n🔍 {model_name} 레이어별 상세 분석:")
    print("-" * 60)
    total = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total += param_count
        print(f"   {name:<40} | {param_count:>10,} | {list(param.shape)}")
    print("-" * 60)
    print(f"   {'TOTAL':<40} | {total:>10,}")
    return total

def main():
    print("🔧 모델 파라미터 분석 시작")
    print("=" * 80)

    # ========================================
    # 1. 외부 구현 모델 (256x256 전용)
    # ========================================
    print("\n🎯 외부 구현 모델 (DRAME_CutPaste/utils_model_for_HDmap_v2.py)")
    try:
        from utils.utils_model_for_HDmap_v2 import ReconstructiveSubNetwork_256, DiscriminativeSubNetwork_256

        # 외부 모델 생성 (1채널 입력, 256x256)
        ext_model_rec = ReconstructiveSubNetwork_256(in_channels=1, out_channels=1)
        ext_model_disc = DiscriminativeSubNetwork_256(in_channels=2, out_channels=2)

        ext_rec_total, ext_rec_trainable = count_parameters(ext_model_rec, "External ReconstructiveSubNetwork_256")
        ext_disc_total, ext_disc_trainable = count_parameters(ext_model_disc, "External DiscriminativeSubNetwork_256")

        ext_total = ext_rec_total + ext_disc_total
        print(f"\n🎯 외부 구현 전체 파라미터: {ext_total:,}")

        # 상세 분석
        analyze_layer_details(ext_model_rec, "External ReconstructiveSubNetwork_256")
        analyze_layer_details(ext_model_disc, "External DiscriminativeSubNetwork_256")

    except ImportError as e:
        print(f"❌ 외부 모델 import 실패: {e}")
        ext_rec_total = ext_disc_total = ext_total = 0

    # ========================================
    # 2. 현재 구현 모델 (표준 DRAEM)
    # ========================================
    print("\n🎯 현재 구현 모델 (src/anomalib/models/image/draem/torch_model.py)")
    try:
        from anomalib.models.image.draem.torch_model import (
            ReconstructiveSubNetwork,
            DiscriminativeSubNetwork,
        )

        # 현재 모델 생성 (1채널 입력, 기본 설정)
        curr_model_rec = ReconstructiveSubNetwork(in_channels=1, out_channels=1, base_width=128, sspcab=False)
        curr_model_disc = DiscriminativeSubNetwork(in_channels=2, out_channels=2, base_width=64)

        curr_rec_total, curr_rec_trainable = count_parameters(curr_model_rec, "Current ReconstructiveSubNetwork")
        curr_disc_total, curr_disc_trainable = count_parameters(curr_model_disc, "Current DiscriminativeSubNetwork")

        curr_total = curr_rec_total + curr_disc_total
        print(f"\n🎯 현재 구현 전체 파라미터: {curr_total:,}")

        # 상세 분석
        analyze_layer_details(curr_model_rec, "Current ReconstructiveSubNetwork")
        analyze_layer_details(curr_model_disc, "Current DiscriminativeSubNetwork")

    except ImportError as e:
        print(f"❌ 현재 모델 import 실패: {e}")
        curr_rec_total = curr_disc_total = curr_total = 0

    # ========================================
    # 3. 비교 분석
    # ========================================
    print("\n📊 모델 비교 분석")
    print("=" * 80)

    if ext_total > 0 and curr_total > 0:
        print(f"외부 구현 총 파라미터:  {ext_total:,}")
        print(f"현재 구현 총 파라미터:  {curr_total:,}")
        print(f"차이:                 {abs(ext_total - curr_total):,}")
        print(f"비율:                 {ext_total/curr_total:.2f}x (외부/현재)")

        print(f"\n📈 Reconstructive Network 비교:")
        print(f"   외부: {ext_rec_total:,}")
        print(f"   현재: {curr_rec_total:,}")
        print(f"   차이: {abs(ext_rec_total - curr_rec_total):,}")

        print(f"\n📈 Discriminative Network 비교:")
        print(f"   외부: {ext_disc_total:,}")
        print(f"   현재: {curr_disc_total:,}")
        print(f"   차이: {abs(ext_disc_total - curr_disc_total):,}")

        # 성능 차이 예상 분석
        print(f"\n🎯 성능 차이 예상 분석:")
        if ext_total > curr_total:
            ratio = ext_total / curr_total
            print(f"   - 외부 모델이 {ratio:.1f}배 더 많은 파라미터를 가짐")
            print(f"   - 더 복잡한 구조로 인해 표현력이 높을 가능성")
            print(f"   - 256x256에 특화된 아키텍처의 장점")
        elif curr_total > ext_total:
            ratio = curr_total / ext_total
            print(f"   - 현재 모델이 {ratio:.1f}배 더 많은 파라미터를 가짐")
            print(f"   - 하지만 범용적 구조로 인한 최적화 부족 가능성")
        else:
            print(f"   - 파라미터 개수는 유사하지만 구조적 차이 존재")

    # ========================================
    # 4. 테스트 입력으로 출력 형태 확인
    # ========================================
    print(f"\n🔧 테스트 입력 (1, 1, 256, 256) 및 (1, 2, 256, 256)으로 출력 형태 확인")

    test_input_1ch = torch.randn(1, 1, 256, 256)
    test_input_2ch = torch.randn(1, 2, 256, 256)

    try:
        if 'ext_model_rec' in locals():
            ext_rec_out = ext_model_rec(test_input_1ch)
            print(f"외부 Reconstructive 출력: {ext_rec_out.shape}")

        if 'ext_model_disc' in locals():
            ext_disc_out = ext_model_disc(test_input_2ch)
            print(f"외부 Discriminative 출력: {ext_disc_out.shape}")

        if 'curr_model_rec' in locals():
            curr_rec_out = curr_model_rec(test_input_1ch)
            print(f"현재 Reconstructive 출력: {curr_rec_out.shape}")

        if 'curr_model_disc' in locals():
            curr_disc_out = curr_model_disc(test_input_2ch)
            print(f"현재 Discriminative 출력: {curr_disc_out.shape}")

    except Exception as e:
        print(f"❌ 테스트 실행 중 오류: {e}")

if __name__ == "__main__":
    main()