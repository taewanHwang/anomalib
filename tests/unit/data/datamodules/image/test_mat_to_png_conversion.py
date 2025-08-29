#!/usr/bin/env python3
"""MAT to PNG 변환 검증 테스트 스크립트.

prepare_hdmap_dataset.py의 변환 로직을 검증하여 
원본 mat 파일의 float 값이 PNG로 변환 시 어떻게 처리되는지 확인합니다.
"""

import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def test_mat_to_png_conversion():
    """MAT → PNG 변환 검증 테스트."""
    
    print("="*80)
    print("MAT → PNG 변환 검증 테스트")
    print("="*80)
    
    # 테스트할 mat 파일 경로 (실제 존재하는 파일로 수정 필요)
    test_mat_files = [
        'datasets/raw/KRISS_share_nipa2023/Normal/Normal2_LSSm0.3_HSS0/Class1/1/HDMap_train_test/1_TSA_DIF_train.mat',
        'datasets/raw/KRISS_share_nipa2023/Planet_fault_ring/1.42_LSSm0.3_HSS0/Class1/1/HDMap_train_test/1_TSA_DIF_test.mat'
    ]
    
    # 테스트 결과 저장 폴더 생성
    output_dir = Path("test_conversion_results")
    output_dir.mkdir(exist_ok=True)
    
    for mat_file in test_mat_files:
        if not os.path.exists(mat_file):
            print(f"⚠️ 파일 없음: {mat_file}")
            continue
            
        print(f"\n📂 테스트 파일: {mat_file}")
        
        # MAT 파일 로드
        try:
            mat_data = scipy.io.loadmat(mat_file)
            image_data = mat_data['Xdata']
            print(f"   Shape: {image_data.shape}")
            print(f"   Data type: {image_data.dtype}")
        except Exception as e:
            print(f"❌ MAT 파일 로드 실패: {e}")
            continue
        
        # 첫 번째 이미지 선택 (테스트용)
        if len(image_data.shape) == 4:
            test_img = image_data[:, :, 0, 0]  # 첫 번째 이미지
        else:
            test_img = image_data[:, :, 0] if len(image_data.shape) == 3 else image_data
        
        print(f"\n🔍 원본 이미지 분석:")
        print(f"   Shape: {test_img.shape}")
        print(f"   Data type: {test_img.dtype}")
        print(f"   Min value: {test_img.min():.6f}")
        print(f"   Max value: {test_img.max():.6f}")
        print(f"   Mean value: {test_img.mean():.6f}")
        print(f"   Std value: {test_img.std():.6f}")
        
        # 원본 값의 히스토그램 출력 (대표 값들)
        unique_values = np.unique(test_img.flatten())
        print(f"   Unique values count: {len(unique_values)}")
        if len(unique_values) <= 20:
            print(f"   All unique values: {unique_values}")
        else:
            print(f"   First 10 values: {unique_values[:10]}")
            print(f"   Last 10 values: {unique_values[-10:]}")
        
        # 파일명 기반으로 저장명 생성
        file_stem = Path(mat_file).stem
        
        # 1. 원본 데이터를 numpy 배열로 저장 (검증용)
        np.save(output_dir / f"{file_stem}_original.npy", test_img)
        
        # 2. 8bit PNG 변환 테스트
        print(f"\n🔄 8bit PNG 변환:")
        img_8bit_normalized = ((test_img - test_img.min()) / (test_img.max() - test_img.min()) * 255).astype(np.uint8)
        print(f"   Normalized shape: {img_8bit_normalized.shape}")
        print(f"   Normalized dtype: {img_8bit_normalized.dtype}")
        print(f"   Normalized min: {img_8bit_normalized.min()}")
        print(f"   Normalized max: {img_8bit_normalized.max()}")
        
        # PNG로 저장
        img_8bit_pil = Image.fromarray(img_8bit_normalized)
        png_8bit_path = output_dir / f"{file_stem}_8bit.png"
        img_8bit_pil.save(png_8bit_path)
        print(f"   8bit PNG 저장: {png_8bit_path}")
        
        # 3. 16bit PNG 변환 테스트  
        print(f"\n🔄 16bit PNG 변환:")
        img_16bit_normalized = ((test_img - test_img.min()) / (test_img.max() - test_img.min()) * 65535).astype(np.uint16)
        print(f"   Normalized shape: {img_16bit_normalized.shape}")
        print(f"   Normalized dtype: {img_16bit_normalized.dtype}")
        print(f"   Normalized min: {img_16bit_normalized.min()}")
        print(f"   Normalized max: {img_16bit_normalized.max()}")
        
        # 16bit PNG로 저장
        img_16bit_pil = Image.fromarray(img_16bit_normalized)
        png_16bit_path = output_dir / f"{file_stem}_16bit.png"
        img_16bit_pil.save(png_16bit_path, format='PNG')
        print(f"   16bit PNG 저장: {png_16bit_path}")
        
        # 4. PNG 파일을 다시 로드해서 값 확인 (역변환 테스트)
        print(f"\n🔄 PNG → Array 역변환 테스트:")
        
        # 8bit PNG 로드
        loaded_8bit = np.array(Image.open(png_8bit_path))
        print(f"   8bit 로드 후 - Shape: {loaded_8bit.shape}, dtype: {loaded_8bit.dtype}")
        print(f"   8bit 로드 후 - Min: {loaded_8bit.min()}, Max: {loaded_8bit.max()}")
        
        # 16bit PNG 로드
        loaded_16bit = np.array(Image.open(png_16bit_path))
        print(f"   16bit 로드 후 - Shape: {loaded_16bit.shape}, dtype: {loaded_16bit.dtype}")
        print(f"   16bit 로드 후 - Min: {loaded_16bit.min()}, Max: {loaded_16bit.max()}")
        
        # 5. 원본 값으로 역변환 테스트
        print(f"\n🔄 원본 값 복원 테스트:")
        
        # 8bit에서 원본 복원
        original_min, original_max = test_img.min(), test_img.max()
        restored_from_8bit = (loaded_8bit / 255.0) * (original_max - original_min) + original_min
        print(f"   8bit 복원 - Min: {restored_from_8bit.min():.6f}, Max: {restored_from_8bit.max():.6f}")
        print(f"   8bit 복원 오차 - Mean: {np.mean(np.abs(test_img - restored_from_8bit)):.6f}")
        print(f"   8bit 복원 오차 - Max: {np.max(np.abs(test_img - restored_from_8bit)):.6f}")
        
        # 16bit에서 원본 복원  
        restored_from_16bit = (loaded_16bit / 65535.0) * (original_max - original_min) + original_min
        print(f"   16bit 복원 - Min: {restored_from_16bit.min():.6f}, Max: {restored_from_16bit.max():.6f}")
        print(f"   16bit 복원 오차 - Mean: {np.mean(np.abs(test_img - restored_from_16bit)):.6f}")
        print(f"   16bit 복원 오차 - Max: {np.max(np.abs(test_img - restored_from_16bit)):.6f}")
        
        # 6. 시각화 생성
        print(f"\n📊 시각화 생성:")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'MAT → PNG 변환 검증: {file_stem}', fontsize=16)
        
        # 원본 이미지
        im1 = axes[0, 0].imshow(test_img, cmap='viridis')
        axes[0, 0].set_title(f'Original MAT\n(min: {test_img.min():.3f}, max: {test_img.max():.3f})')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 8bit 정규화
        im2 = axes[0, 1].imshow(img_8bit_normalized, cmap='viridis')
        axes[0, 1].set_title(f'8bit Normalized\n(min: {img_8bit_normalized.min()}, max: {img_8bit_normalized.max()})')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 16bit 정규화
        im3 = axes[0, 2].imshow(img_16bit_normalized, cmap='viridis')
        axes[0, 2].set_title(f'16bit Normalized\n(min: {img_16bit_normalized.min()}, max: {img_16bit_normalized.max()})')
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # 8bit 복원
        im4 = axes[1, 0].imshow(restored_from_8bit, cmap='viridis')
        axes[1, 0].set_title(f'8bit Restored\n(error: {np.max(np.abs(test_img - restored_from_8bit)):.6f})')
        axes[1, 0].axis('off')
        plt.colorbar(im4, ax=axes[1, 0])
        
        # 16bit 복원
        im5 = axes[1, 1].imshow(restored_from_16bit, cmap='viridis')
        axes[1, 1].set_title(f'16bit Restored\n(error: {np.max(np.abs(test_img - restored_from_16bit)):.6f})')
        axes[1, 1].axis('off')
        plt.colorbar(im5, ax=axes[1, 1])
        
        # 오차 히트맵
        error_8bit = np.abs(test_img - restored_from_8bit)
        im6 = axes[1, 2].imshow(error_8bit, cmap='Reds')
        axes[1, 2].set_title(f'8bit Error Map\n(max error: {np.max(error_8bit):.6f})')
        axes[1, 2].axis('off')
        plt.colorbar(im6, ax=axes[1, 2])
        
        plt.tight_layout()
        plot_path = output_dir / f"{file_stem}_conversion_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   시각화 저장: {plot_path}")
        
        # 7. 요약 리포트 생성
        print(f"\n📋 요약 리포트:")
        print(f"   원본 데이터 범위: [{test_img.min():.6f}, {test_img.max():.6f}]")
        print(f"   8bit 변환 손실: 최대 {np.max(np.abs(test_img - restored_from_8bit)):.6f}")
        print(f"   16bit 변환 손실: 최대 {np.max(np.abs(test_img - restored_from_16bit)):.6f}")
        
        if np.max(np.abs(test_img - restored_from_8bit)) < 1e-3:
            print(f"   ✅ 8bit 변환: 정밀도 손실 미미")
        else:
            print(f"   ⚠️  8bit 변환: 정밀도 손실 주의 필요")
            
        if np.max(np.abs(test_img - restored_from_16bit)) < 1e-6:
            print(f"   ✅ 16bit 변환: 정밀도 손실 거의 없음")
        else:
            print(f"   ⚠️  16bit 변환: 정밀도 손실 있음")
    
    # 최종 결론
    print(f"\n" + "="*80)
    print("🎯 결론:")
    print("="*80)
    print("1. MAT → PNG 변환 시 정규화(scaling) 수행:")
    print("   - 원본 float 값을 [min, max] → [0, 255] (8bit) 또는 [0, 65535] (16bit)로 변환")
    print("   - 원본 값의 상대적 비율은 유지됨")
    print("")
    print("2. 정밀도 손실:")
    print("   - 8bit: 256 단계로 양자화 → 일부 정밀도 손실 가능")
    print("   - 16bit: 65536 단계로 양자화 → 정밀도 손실 거의 없음")
    print("")
    print("3. 역변환 가능:")
    print("   - PNG에서 원본 float 범위로 역변환 가능")
    print("   - 단, 양자화로 인한 손실은 복구 불가")
    print("")
    print(f"📁 테스트 결과 저장 위치: {output_dir.absolute()}")


def analyze_specific_sample():
    """특정 샘플의 값 분포를 자세히 분석."""
    
    print(f"\n" + "="*80)
    print("📊 값 분포 세부 분석")
    print("="*80)
    
    # 작은 테스트 배열 생성
    test_array = np.array([
        [0.1, 0.5, 1.0],
        [1.5, 2.0, 2.5],  
        [3.0, 3.5, 4.0]
    ], dtype=np.float64)
    
    print(f"테스트 배열:")
    print(test_array)
    print(f"원본 범위: [{test_array.min():.3f}, {test_array.max():.3f}]")
    
    # 8bit 변환
    normalized_8bit = ((test_array - test_array.min()) / (test_array.max() - test_array.min()) * 255).astype(np.uint8)
    print(f"\n8bit 변환 결과:")
    print(normalized_8bit)
    
    # 16bit 변환
    normalized_16bit = ((test_array - test_array.min()) / (test_array.max() - test_array.min()) * 65535).astype(np.uint16)
    print(f"\n16bit 변환 결과:")
    print(normalized_16bit)
    
    # 역변환
    restored_8bit = (normalized_8bit / 255.0) * (test_array.max() - test_array.min()) + test_array.min()
    restored_16bit = (normalized_16bit / 65535.0) * (test_array.max() - test_array.min()) + test_array.min()
    
    print(f"\n8bit 역변환 결과:")
    print(restored_8bit)
    print(f"8bit 복원 오차:")
    print(np.abs(test_array - restored_8bit))
    
    print(f"\n16bit 역변환 결과:")
    print(restored_16bit)
    print(f"16bit 복원 오차:")
    print(np.abs(test_array - restored_16bit))


if __name__ == "__main__":
    test_mat_to_png_conversion()
    analyze_specific_sample()