#!/usr/bin/env python3
"""HDMAP 데이터 채널 변환 테스트 스크립트.

이 스크립트는 HDMAP 데이터셋의 이미지 로딩 과정에서 발생하는 채널 변환을 테스트합니다.
특히 1채널 grayscale 이미지가 3채널 RGB로 어떻게 변환되는지 확인합니다.

테스트 항목:
1. 원본 이미지가 1채널인지 확인
2. MultiDomainHDMAPDataModule을 통한 로딩 시 3채널로 변환되는지 확인  
3. 채널 변환 방식이 repeat (동일한 값) 방식인지 확인
4. read_image 함수의 동작 확인
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Anomalib 모듈 import
sys.path.append(str(Path(__file__).parent.parent.parent))

from anomalib.data.datamodules.image.multi_domain_hdmap import MultiDomainHDMAPDataModule
from anomalib.data.utils.image import read_image

def test_original_image_channels():
    """원본 HDMAP 이미지의 채널 수 확인."""
    print("="*80)
    print("🔍 1. 원본 HDMAP 이미지 채널 수 테스트")
    print("="*80)
    
    # HDMAP 데이터셋 경로 (실제 경로로 수정 필요)
    hdmap_root = Path("./datasets/HDMAP/1000_8bit_resize_224x224")
    
    if not hdmap_root.exists():
        print(f"❌ HDMAP 데이터셋 경로를 찾을 수 없습니다: {hdmap_root}")
        print("   경로를 확인하고 다시 실행해주세요.")
        return False
    
    # 각 도메인에서 샘플 이미지 찾기
    domains = ["domain_A", "domain_B", "domain_C", "domain_D"]
    sample_images = []
    
    for domain in domains:
        domain_path = hdmap_root / domain / "train" / "good"
        if domain_path.exists():
            image_files = list(domain_path.glob("*.png"))
            if image_files:
                sample_images.append((domain, image_files[0]))
                break
    
    if not sample_images:
        print("❌ 테스트할 이미지를 찾을 수 없습니다.")
        return False
    
    # 원본 이미지 분석
    for domain, image_path in sample_images:
        print(f"\n📁 도메인: {domain}")
        print(f"📄 이미지 경로: {image_path}")
        
        # PIL로 원본 이미지 로드
        pil_image = Image.open(image_path)
        print(f"   • PIL 이미지 모드: {pil_image.mode}")
        print(f"   • PIL 이미지 크기: {pil_image.size}")
        
        # NumPy 배열로 변환
        np_image = np.array(pil_image)
        print(f"   • NumPy 배열 shape: {np_image.shape}")
        print(f"   • NumPy 배열 dtype: {np_image.dtype}")
        print(f"   • 값 범위: [{np_image.min()}, {np_image.max()}]")
        
        # 채널 수 확인
        if len(np_image.shape) == 2:
            print("   ✅ 원본 이미지는 1채널 (grayscale) 입니다.")
            original_channels = 1
        elif len(np_image.shape) == 3:
            print(f"   ⚠️  원본 이미지는 {np_image.shape[2]}채널입니다.")
            original_channels = np_image.shape[2]
        else:
            print(f"   ❌ 예상치 못한 이미지 shape: {np_image.shape}")
            return False
            
        return original_channels, image_path
    
    return False

def test_read_image_function(image_path):
    """read_image 함수의 채널 변환 동작 테스트."""
    print("\n" + "="*80)
    print("🔍 2. read_image 함수 채널 변환 테스트")
    print("="*80)
    
    print(f"📄 테스트 이미지: {image_path}")
    
    # 1. PIL로 원본 로드
    print("\n🔸 PIL 원본 로드:")
    pil_original = Image.open(image_path)
    print(f"   • 원본 모드: {pil_original.mode}")
    
    # 2. PIL convert("RGB") 테스트
    print("\n🔸 PIL convert('RGB') 테스트:")
    pil_rgb = pil_original.convert("RGB")
    print(f"   • 변환 후 모드: {pil_rgb.mode}")
    
    np_rgb = np.array(pil_rgb)
    print(f"   • NumPy shape: {np_rgb.shape}")
    print(f"   • 값 범위: [{np_rgb.min()}, {np_rgb.max()}]")
    
    # 3. read_image 함수 테스트
    print("\n🔸 read_image 함수 테스트:")
    
    # NumPy 배열로 로드
    image_np = read_image(image_path, as_tensor=False)
    print(f"   • NumPy 결과 shape: {image_np.shape}")
    print(f"   • NumPy 결과 dtype: {image_np.dtype}")
    print(f"   • NumPy 값 범위: [{image_np.min():.3f}, {image_np.max():.3f}]")
    
    # Tensor로 로드
    image_tensor = read_image(image_path, as_tensor=True)
    print(f"   • Tensor 결과 shape: {image_tensor.shape}")
    print(f"   • Tensor 결과 dtype: {image_tensor.dtype}")
    print(f"   • Tensor 값 범위: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
    
    return image_np, image_tensor

def test_channel_conversion_method(image_np):
    """채널 변환 방식 테스트 (repeat vs 다른 방식)."""
    print("\n" + "="*80)
    print("🔍 3. 채널 변환 방식 테스트 (repeat 방식인지 확인)")
    print("="*80)
    
    if len(image_np.shape) != 3 or image_np.shape[2] != 3:
        print("❌ 3채널 이미지가 아닙니다.")
        return False
    
    # 각 채널 분리
    r_channel = image_np[:, :, 0]
    g_channel = image_np[:, :, 1] 
    b_channel = image_np[:, :, 2]
    
    print(f"📊 채널별 통계:")
    print(f"   • R 채널 - 평균: {r_channel.mean():.6f}, 표준편차: {r_channel.std():.6f}")
    print(f"   • G 채널 - 평균: {g_channel.mean():.6f}, 표준편차: {g_channel.std():.6f}")
    print(f"   • B 채널 - 평균: {b_channel.mean():.6f}, 표준편차: {b_channel.std():.6f}")
    
    # 채널 간 동일성 확인
    r_g_equal = np.allclose(r_channel, g_channel, rtol=1e-10, atol=1e-10)
    g_b_equal = np.allclose(g_channel, b_channel, rtol=1e-10, atol=1e-10)
    r_b_equal = np.allclose(r_channel, b_channel, rtol=1e-10, atol=1e-10)
    
    print(f"\n🔍 채널 간 동일성 검사:")
    print(f"   • R == G: {r_g_equal}")
    print(f"   • G == B: {g_b_equal}")
    print(f"   • R == B: {r_b_equal}")
    
    if r_g_equal and g_b_equal and r_b_equal:
        print("   ✅ 모든 채널이 동일합니다 → repeat 방식으로 변환됨")
        conversion_method = "repeat"
    else:
        print("   ⚠️  채널들이 서로 다릅니다 → 다른 변환 방식 사용")
        conversion_method = "different"
    
    # 샘플 픽셀 값 확인
    print(f"\n🔍 샘플 픽셀 값 확인 (좌상단 5x5):")
    print("R 채널:")
    print(r_channel[:5, :5])
    print("G 채널:")
    print(g_channel[:5, :5])
    print("B 채널:")
    print(b_channel[:5, :5])
    
    return conversion_method

def test_datamodule_loading():
    """MultiDomainHDMAPDataModule을 통한 데이터 로딩 테스트."""
    print("\n" + "="*80)
    print("🔍 4. MultiDomainHDMAPDataModule 데이터 로딩 테스트")
    print("="*80)
    
    try:
        # DataModule 생성
        datamodule = MultiDomainHDMAPDataModule(
            root="./datasets/HDMAP/1000_8bit_resize_224x224",
            source_domain="domain_A",
            target_domains=["domain_B"],
            train_batch_size=2,
            eval_batch_size=2
        )
        
        print("✅ MultiDomainHDMAPDataModule 생성 완료")
        
        # Setup 호출
        datamodule.setup()
        print("✅ DataModule setup 완료")
        
        # Train DataLoader 테스트
        train_loader = datamodule.train_dataloader()
        print(f"✅ Train DataLoader 생성 완료 - 배치 수: {len(train_loader)}")
        
        # 첫 번째 배치 로드
        batch = next(iter(train_loader))
        print(f"\n📊 Train 배치 정보:")
        print(f"   • 이미지 shape: {batch.image.shape}")
        print(f"   • 이미지 dtype: {batch.image.dtype}")
        print(f"   • 이미지 값 범위: [{batch.image.min():.3f}, {batch.image.max():.3f}]")
        print(f"   • 라벨 shape: {batch.gt_label.shape}")
        print(f"   • 라벨 값: {batch.gt_label}")
        
        # 채널 확인
        if len(batch.image.shape) == 4:  # [B, C, H, W]
            channels = batch.image.shape[1]
            print(f"   • 채널 수: {channels}")
            
            if channels == 3:
                print("   ✅ 3채널로 정상 로딩됨")
                
                # 첫 번째 이미지의 채널별 동일성 확인
                first_image = batch.image[0]  # [C, H, W]
                r_channel = first_image[0]
                g_channel = first_image[1]
                b_channel = first_image[2]
                
                r_g_equal = torch.allclose(r_channel, g_channel, rtol=1e-6, atol=1e-6)
                g_b_equal = torch.allclose(g_channel, b_channel, rtol=1e-6, atol=1e-6)
                
                print(f"   • 채널 간 동일성: R==G: {r_g_equal}, G==B: {g_b_equal}")
                
                if r_g_equal and g_b_equal:
                    print("   ✅ DataModule에서도 repeat 방식으로 변환 확인")
                else:
                    print("   ⚠️  DataModule에서 다른 변환 방식 사용")
            else:
                print(f"   ⚠️  예상과 다른 채널 수: {channels}")
        
        # Validation DataLoader 테스트
        val_loader = datamodule.val_dataloader()
        val_batch = next(iter(val_loader))
        print(f"\n📊 Validation 배치 정보:")
        print(f"   • 이미지 shape: {val_batch.image.shape}")
        print(f"   • 라벨 값: {val_batch.gt_label}")
        
        return True
        
    except Exception as e:
        print(f"❌ DataModule 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_visualization(image_np, output_dir="./test_results"):
    """채널 변환 결과 시각화."""
    print("\n" + "="*80)
    print("🎨 5. 채널 변환 결과 시각화")
    print("="*80)
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if len(image_np.shape) != 3 or image_np.shape[2] != 3:
            print("❌ 시각화할 수 없는 이미지 형태입니다.")
            return
        
        # 채널별 분리
        r_channel = image_np[:, :, 0]
        g_channel = image_np[:, :, 1]
        b_channel = image_np[:, :, 2]
        
        # 시각화 생성
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('HDMAP 이미지 채널 변환 결과', fontsize=16)
        
        # 원본 RGB 이미지
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title('RGB 이미지 (변환 후)')
        axes[0, 0].axis('off')
        
        # R 채널
        axes[0, 1].imshow(r_channel, cmap='Reds')
        axes[0, 1].set_title('R 채널')
        axes[0, 1].axis('off')
        
        # G 채널
        axes[1, 0].imshow(g_channel, cmap='Greens')
        axes[1, 0].set_title('G 채널')
        axes[1, 0].axis('off')
        
        # B 채널
        axes[1, 1].imshow(b_channel, cmap='Blues')
        axes[1, 1].set_title('B 채널')
        axes[1, 1].axis('off')
        
        # 저장
        save_path = output_path / "hdmap_channel_conversion_test.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 시각화 결과 저장: {save_path}")
        
        # 채널 차이 히스토그램
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('채널별 히스토그램', fontsize=16)
        
        axes[0].hist(r_channel.flatten(), bins=50, alpha=0.7, color='red', label='R')
        axes[0].set_title('R 채널 히스토그램')
        axes[0].set_xlabel('픽셀 값')
        axes[0].set_ylabel('빈도')
        
        axes[1].hist(g_channel.flatten(), bins=50, alpha=0.7, color='green', label='G')
        axes[1].set_title('G 채널 히스토그램')
        axes[1].set_xlabel('픽셀 값')
        axes[1].set_ylabel('빈도')
        
        axes[2].hist(b_channel.flatten(), bins=50, alpha=0.7, color='blue', label='B')
        axes[2].set_title('B 채널 히스토그램')
        axes[2].set_xlabel('픽셀 값')
        axes[2].set_ylabel('빈도')
        
        hist_save_path = output_path / "hdmap_channel_histograms.png"
        plt.tight_layout()
        plt.savefig(hist_save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 히스토그램 저장: {hist_save_path}")
        
    except Exception as e:
        print(f"❌ 시각화 생성 실패: {e}")

def main():
    """메인 테스트 함수."""
    print("🚀 HDMAP 데이터 채널 변환 테스트 시작")
    print("="*80)
    
    # 1. 원본 이미지 채널 수 확인
    result = test_original_image_channels()
    if not result:
        print("❌ 원본 이미지 테스트 실패")
        return
    
    original_channels, sample_image_path = result
    
    # 2. read_image 함수 테스트
    image_np, image_tensor = test_read_image_function(sample_image_path)
    
    # 3. 채널 변환 방식 확인
    conversion_method = test_channel_conversion_method(image_np)
    
    # 4. DataModule 로딩 테스트
    datamodule_success = test_datamodule_loading()
    
    # 5. 시각화
    create_visualization(image_np)
    
    # 결과 요약
    print("\n" + "="*80)
    print("📋 테스트 결과 요약")
    print("="*80)
    print(f"1. 원본 이미지 채널 수: {original_channels}채널")
    print(f"2. read_image 함수 결과: 3채널 (RGB)")
    print(f"3. 채널 변환 방식: {conversion_method}")
    print(f"4. DataModule 로딩: {'성공' if datamodule_success else '실패'}")
    
    if original_channels == 1 and conversion_method == "repeat":
        print("\n✅ 결론: HDMAP 데이터는 1채널 → 3채널로 repeat 방식으로 변환됩니다.")
        print("   PIL의 convert('RGB')가 grayscale을 RGB로 변환할 때 각 채널에 동일한 값을 복사합니다.")
    else:
        print(f"\n⚠️  예상과 다른 결과입니다. 추가 조사가 필요합니다.")
    
    print("\n🎯 핵심 발견:")
    print("   • anomalib.data.utils.image.read_image() 함수의 Line 319:")
    print("     image = Image.open(path).convert('RGB')")
    print("   • 이 부분에서 1채널 grayscale → 3채널 RGB 변환이 발생합니다.")
    print("   • PIL의 convert('RGB')는 grayscale 값을 R, G, B 채널에 동일하게 복사합니다.")

if __name__ == "__main__":
    main()
