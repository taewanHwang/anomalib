#!/usr/bin/env python3
"""
모델 파라미터 수 계산 스크립트

실행 방법:
    python examples/hdmap/paper/count_model_parameters.py
"""

import sys
from pathlib import Path
import torch

# 프로젝트 루트 경로
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

from anomalib.models.image.draem_cutpaste_clf.torch_model import DraemCutPasteModel
from anomalib.models.image.cutpaste_clf.torch_model import CutPasteClf
from anomalib.models.image.draem.torch_model import DraemModel


def count_parameters(model, model_name="Model"):
    """모델의 파라미터 수 계산

    Args:
        model: PyTorch 모델
        model_name: 모델 이름

    Returns:
        dict: 파라미터 통계
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 서브 모듈별 파라미터 계산
    submodule_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        submodule_params[name] = params

    return {
        "model_name": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "submodule_params": submodule_params,
        "total_params_M": total_params / 1e6,
        "trainable_params_M": trainable_params / 1e6
    }


def print_model_info(stats):
    """모델 정보 출력"""
    print(f"\n{'='*80}")
    print(f"📊 {stats['model_name']}")
    print(f"{'='*80}")
    print(f"Total Parameters: {stats['total_params']:,} ({stats['total_params_M']:.2f}M)")
    print(f"Trainable Parameters: {stats['trainable_params']:,} ({stats['trainable_params_M']:.2f}M)")
    print(f"\nSubmodule Breakdown:")
    for name, params in stats['submodule_params'].items():
        print(f"  - {name}: {params:,} ({params/1e6:.2f}M)")


def main():
    print("="*80)
    print("🔍 Model Parameter Counting")
    print("="*80)

    # =========================================================================
    # 1. DRAEM CutPaste Clf (Proposed Method)
    # =========================================================================
    print("\n\n" + "="*80)
    print("1️⃣  DRAEM CutPaste Clf (Proposed)")
    print("="*80)

    draem_cutpaste_clf = DraemCutPasteModel(
        sspcab=False,
        image_size=(128, 128),
        severity_dropout=0.3,
        severity_input_channels="original+mask",
        detach_mask=True,
        cut_w_range=(2, 127),
        cut_h_range=(4, 8),
        a_fault_start=0.0,
        a_fault_range_end=0.3,
        augment_probability=0.5
    )

    draem_cutpaste_clf_stats = count_parameters(
        draem_cutpaste_clf,
        "DRAEM CutPaste Clf"
    )
    print_model_info(draem_cutpaste_clf_stats)

    # =========================================================================
    # 2. CutPaste Clf (Baseline)
    # =========================================================================
    print("\n\n" + "="*80)
    print("2️⃣  CutPaste Clf (Baseline)")
    print("="*80)

    cutpaste_clf = CutPasteClf(image_size=(128, 128))

    cutpaste_clf_stats = count_parameters(
        cutpaste_clf,
        "CutPaste Clf"
    )
    print_model_info(cutpaste_clf_stats)

    # =========================================================================
    # 3. DRAEM (Original)
    # =========================================================================
    print("\n\n" + "="*80)
    print("3️⃣  DRAEM (Original)")
    print("="*80)

    draem = DraemModel(sspcab=False)

    draem_stats = count_parameters(
        draem,
        "DRAEM"
    )
    print_model_info(draem_stats)

    # =========================================================================
    # Summary Comparison
    # =========================================================================
    print("\n\n" + "="*80)
    print("📊 Summary Comparison")
    print("="*80)

    print(f"\n{'Model':<30} {'Total Params':<20} {'Params (M)':<15}")
    print("-"*65)
    print(f"{'DRAEM CutPaste Clf':<30} {draem_cutpaste_clf_stats['total_params']:>15,} {draem_cutpaste_clf_stats['total_params_M']:>12.2f}")
    print(f"{'CutPaste Clf':<30} {cutpaste_clf_stats['total_params']:>15,} {cutpaste_clf_stats['total_params_M']:>12.2f}")
    print(f"{'DRAEM':<30} {draem_stats['total_params']:>15,} {draem_stats['total_params_M']:>12.2f}")

    # =========================================================================
    # Save to file
    # =========================================================================
    output_dir = project_root / "examples" / "hdmap" / "paper"
    output_file = output_dir / "model_parameter_counts.txt"

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Model Parameter Counts\n")
        f.write("="*80 + "\n\n")

        # DRAEM CutPaste Clf
        f.write("1. DRAEM CutPaste Clf (Proposed)\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Parameters: {draem_cutpaste_clf_stats['total_params']:,} ({draem_cutpaste_clf_stats['total_params_M']:.2f}M)\n")
        f.write(f"Trainable Parameters: {draem_cutpaste_clf_stats['trainable_params']:,} ({draem_cutpaste_clf_stats['trainable_params_M']:.2f}M)\n")
        f.write("\nSubmodule Breakdown:\n")
        for name, params in draem_cutpaste_clf_stats['submodule_params'].items():
            f.write(f"  - {name}: {params:,} ({params/1e6:.2f}M)\n")
        f.write("\n\n")

        # CutPaste Clf
        f.write("2. CutPaste Clf (Baseline)\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Parameters: {cutpaste_clf_stats['total_params']:,} ({cutpaste_clf_stats['total_params_M']:.2f}M)\n")
        f.write(f"Trainable Parameters: {cutpaste_clf_stats['trainable_params']:,} ({cutpaste_clf_stats['trainable_params_M']:.2f}M)\n")
        f.write("\nSubmodule Breakdown:\n")
        for name, params in cutpaste_clf_stats['submodule_params'].items():
            f.write(f"  - {name}: {params:,} ({params/1e6:.2f}M)\n")
        f.write("\n\n")

        # DRAEM
        f.write("3. DRAEM (Original)\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Parameters: {draem_stats['total_params']:,} ({draem_stats['total_params_M']:.2f}M)\n")
        f.write(f"Trainable Parameters: {draem_stats['trainable_params']:,} ({draem_stats['trainable_params_M']:.2f}M)\n")
        f.write("\nSubmodule Breakdown:\n")
        for name, params in draem_stats['submodule_params'].items():
            f.write(f"  - {name}: {params:,} ({params/1e6:.2f}M)\n")
        f.write("\n\n")

        # Summary
        f.write("="*80 + "\n")
        f.write("Summary Comparison\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Model':<30} {'Total Params':<20} {'Params (M)':<15}\n")
        f.write("-"*65 + "\n")
        f.write(f"{'DRAEM CutPaste Clf':<30} {draem_cutpaste_clf_stats['total_params']:>15,} {draem_cutpaste_clf_stats['total_params_M']:>12.2f}\n")
        f.write(f"{'CutPaste Clf':<30} {cutpaste_clf_stats['total_params']:>15,} {cutpaste_clf_stats['total_params_M']:>12.2f}\n")
        f.write(f"{'DRAEM':<30} {draem_stats['total_params']:>15,} {draem_stats['total_params_M']:>12.2f}\n")

    print(f"\n✅ Results saved to: {output_file}")

    return {
        "draem_cutpaste_clf": draem_cutpaste_clf_stats,
        "cutpaste_clf": cutpaste_clf_stats,
        "draem": draem_stats
    }


if __name__ == "__main__":
    main()
