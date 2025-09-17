#!/usr/bin/env python3
"""MAT 파일을 TIFF로 변환하여 32비트 데이터 손실 없이 저장하는 테스트"""

import os
import tempfile
from pathlib import Path

import numpy as np
import scipy.io
import tifffile


def main():
    """메인 테스트 실행 함수"""
    converter = TestMatToTiffConversion()
    
    # 실제 MAT 파일 경로 확인
    sample_path = "datasets/raw/KRISS_share_nipa2023/Normal/Normal2_LSSm0.3_HSS0/Class1/1/HDMap_train_test/3_TSA_DIF_train.mat"
    if not os.path.exists(sample_path):
        print(f"❌ MAT 파일을 찾을 수 없습니다: {sample_path}")
        return
    
    # 출력 디렉토리 설정
    output_dir = Path(__file__).parent / "test_output"
    output_dir.mkdir(exist_ok=True)
    
    print("🧪 MAT to TIFF 변환 테스트 시작...")
    
    try:
        # 테스트 실행
        converter.test_mat_file_loading_and_basic_info(sample_path)
        converter.test_tiff_conversion_and_data_integrity(sample_path, str(output_dir))
        converter.test_value_range_comparison(sample_path, str(output_dir))
        converter.test_file_size_comparison(sample_path, str(output_dir))
        
        print("\n✅ 모든 테스트가 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


class TestMatToTiffConversion:
    """MAT 파일을 TIFF 형식으로 변환하여 데이터 무결성을 테스트"""
    
    def test_mat_file_loading_and_basic_info(self, sample_mat_file_path):
        """MAT 파일 로딩 및 기본 정보 확인"""
        mat_data = scipy.io.loadmat(sample_mat_file_path)
        image_data = mat_data['Xdata']
        
        print(f"\n=== MAT 파일 기본 정보 ===")
        print(f"파일 경로: {sample_mat_file_path}")
        print(f"데이터 형태: {image_data.shape}")
        print(f"데이터 타입: {image_data.dtype}")
        print(f"최솟값: {image_data.min()}")
        print(f"최댓값: {image_data.max()}")
        print(f"평균값: {image_data.mean():.6f}")
        print(f"표준편차: {image_data.std():.6f}")
        
        # 기본 검증
        assert image_data is not None
        assert len(image_data.shape) == 4  # (height, width, channel, samples)
        assert image_data.dtype in [np.float32, np.float64]  # 32비트 부동소수점
    
    def test_tiff_conversion_and_data_integrity(self, sample_mat_file_path, output_dir):
        """TIFF 변환 후 데이터 무결성 확인"""
        # 1. MAT 파일 로드
        mat_data = scipy.io.loadmat(sample_mat_file_path)
        image_data = mat_data['Xdata']
        
        # 2. 첫 10개 이미지를 TIFF로 저장
        num_test_images = min(10, image_data.shape[3])
        original_images = []
        
        print(f"\n=== TIFF 변환 테스트 ===")
        print(f"테스트할 이미지 수: {num_test_images}")
        print(f"저장 경로: {output_dir}")
        
        for i in range(num_test_images):
            # 원본 이미지 추출 (첫 번째 채널만)
            original_img = image_data[:, :, 0, i]
            original_images.append(original_img.copy())
            
            # TIFF 파일로 저장 (32비트 부동소수점)
            tiff_path = os.path.join(output_dir, f"test_image_{i:03d}.tiff")
            tifffile.imwrite(tiff_path, original_img.astype(np.float32))
            
            print(f"이미지 {i}: 저장 완료 - {tiff_path}")
        
        # 3. TIFF 파일 읽기 및 데이터 무결성 확인
        print(f"\n=== 데이터 무결성 검증 ===")
        
        for i in range(num_test_images):
            tiff_path = os.path.join(output_dir, f"test_image_{i:03d}.tiff")
            
            # TIFF 파일 읽기
            loaded_img = tifffile.imread(tiff_path)
            original_img = original_images[i]
            
            # 데이터 타입 및 형태 확인
            assert loaded_img.shape == original_img.shape, f"이미지 {i}: 형태 불일치"
            assert loaded_img.dtype == np.float32, f"이미지 {i}: 데이터 타입 불일치"
            
            # 완벽한 데이터 일치 확인 (32비트 정밀도 내에서)
            np.testing.assert_array_almost_equal(
                loaded_img, original_img.astype(np.float32), 
                decimal=6, err_msg=f"이미지 {i}: 데이터 불일치"
            )
            
            # 통계값 비교
            print(f"이미지 {i}:")
            print(f"  원본   - min: {original_img.min():.6f}, max: {original_img.max():.6f}, mean: {original_img.mean():.6f}")
            print(f"  TIFF   - min: {loaded_img.min():.6f}, max: {loaded_img.max():.6f}, mean: {loaded_img.mean():.6f}")
            print(f"  차이   - max_diff: {np.abs(loaded_img - original_img.astype(np.float32)).max():.10f}")
        
        print("✅ 모든 이미지에서 데이터 무결성 확인 완료")
    
    def test_value_range_comparison(self, sample_mat_file_path, output_dir):
        """원본 MAT과 16비트 PNG, 32비트 TIFF의 값 범위 비교"""
        from PIL import Image
        
        # MAT 파일 로드
        mat_data = scipy.io.loadmat(sample_mat_file_path)
        image_data = mat_data['Xdata']
        
        # 테스트용 첫 번째 이미지
        original_img = image_data[:, :, 0, 0]
        
        print(f"\n=== 값 범위 비교 분석 ===")
        print(f"원본 32비트 데이터:")
        print(f"  최솟값: {original_img.min():.6f}")
        print(f"  최댓값: {original_img.max():.6f}")
        print(f"  범위: {original_img.max() - original_img.min():.6f}")
        print(f"  데이터 타입: {original_img.dtype}")
        
        # 1. 기존 방식: 16비트 PNG 저장 (정규화 + 양자화)
        img_normalized = (original_img - original_img.min()) / (original_img.max() - original_img.min())
        img_16bit = (img_normalized * 65535).astype(np.uint16)
        png_path = os.path.join(output_dir, "test_16bit.png")
        Image.fromarray(img_16bit).save(png_path)
        
        # 16비트 PNG 읽기
        loaded_png = np.array(Image.open(png_path))
        png_restored = (loaded_png.astype(np.float64) / 65535.0) * (original_img.max() - original_img.min()) + original_img.min()
        
        print(f"\n16비트 PNG 변환 후:")
        print(f"  양자화된 값 범위: 0 ~ 65535")
        print(f"  복원된 최솟값: {png_restored.min():.6f}")
        print(f"  복원된 최댓값: {png_restored.max():.6f}")
        print(f"  원본과의 최대 오차: {np.abs(png_restored - original_img).max():.6f}")
        
        # 2. 새로운 방식: 32비트 TIFF 저장 (무손실)
        tiff_path = os.path.join(output_dir, "test_32bit.tiff")
        tifffile.imwrite(tiff_path, original_img.astype(np.float32))
        
        # 32비트 TIFF 읽기
        loaded_tiff = tifffile.imread(tiff_path)
        
        print(f"\n32비트 TIFF 변환 후:")
        print(f"  저장된 최솟값: {loaded_tiff.min():.6f}")
        print(f"  저장된 최댓값: {loaded_tiff.max():.6f}")
        print(f"  원본과의 최대 오차: {np.abs(loaded_tiff - original_img.astype(np.float32)).max():.10f}")
        
        # 3. 손실 비교
        png_loss = np.abs(png_restored - original_img).mean()
        tiff_loss = np.abs(loaded_tiff - original_img.astype(np.float32)).mean()
        
        print(f"\n=== 데이터 손실 비교 ===")
        print(f"16비트 PNG 평균 오차: {png_loss:.10f}")
        print(f"32비트 TIFF 평균 오차: {tiff_loss:.15f}")
        print(f"손실 비율 (TIFF/PNG): {tiff_loss/png_loss:.2e}")
        
        # 검증: TIFF가 훨씬 정확해야 함
        assert tiff_loss < png_loss * 0.001, "TIFF 저장이 PNG보다 훨씬 정확해야 함"
        assert np.abs(loaded_tiff - original_img.astype(np.float32)).max() < 1e-6, "TIFF는 거의 무손실이어야 함"
    
    def test_file_size_comparison(self, sample_mat_file_path, output_dir):
        """PNG vs TIFF 파일 크기 비교"""
        from PIL import Image
        
        # MAT 파일 로드
        mat_data = scipy.io.loadmat(sample_mat_file_path)
        image_data = mat_data['Xdata']
        
        # 테스트용 첫 번째 이미지
        original_img = image_data[:, :, 0, 0]
        
        # 16비트 PNG 저장
        img_normalized = (original_img - original_img.min()) / (original_img.max() - original_img.min())
        img_16bit = (img_normalized * 65535).astype(np.uint16)
        png_path = os.path.join(output_dir, "test_16bit.png")
        Image.fromarray(img_16bit).save(png_path)
        
        # 32비트 TIFF 저장
        tiff_path = os.path.join(output_dir, "test_32bit.tiff")
        tifffile.imwrite(tiff_path, original_img.astype(np.float32))
        
        # 파일 크기 비교
        png_size = os.path.getsize(png_path)
        tiff_size = os.path.getsize(tiff_path)
        
        print(f"\n=== 파일 크기 비교 ===")
        print(f"이미지 크기: {original_img.shape}")
        print(f"16비트 PNG: {png_size:,} bytes ({png_size/1024:.1f} KB)")
        print(f"32비트 TIFF: {tiff_size:,} bytes ({tiff_size/1024:.1f} KB)")
        print(f"크기 비율 (TIFF/PNG): {tiff_size/png_size:.2f}x")
        
        # 일반적으로 TIFF가 더 클 것으로 예상되지만, 압축에 따라 달라질 수 있음
        assert png_size > 0 and tiff_size > 0, "파일이 정상적으로 저장되어야 함"


if __name__ == "__main__":
    main()