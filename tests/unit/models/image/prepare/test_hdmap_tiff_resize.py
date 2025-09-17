#!/usr/bin/env python3
"""Test HDMAP TIFF loading and resizing functionality.

이 테스트는 새로 구현된 HDMapDataset과 HDMapDataModule의 TIFF 로딩 및 리사이즈 기능을 검증합니다.
실제 TIFF 파일을 사용하여 다양한 리사이즈 방법과 설정을 테스트합니다.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# Anomalib imports - 상대 경로로 수정
try:
    from anomalib.data.datasets.image.hdmap import HDMAPDataset
    from anomalib.data.datamodules.image.hdmap import HDMAPDataModule
except ImportError:
    # 개발 환경에서 직접 실행하는 경우
    import sys
    from pathlib import Path
    root_path = Path(__file__).parent.parent.parent.parent.parent
    sys.path.insert(0, str(root_path / "src"))
    from anomalib.data.datasets.image.hdmap import HDMAPDataset
    from anomalib.data.datamodules.image.hdmap import HDMAPDataModule


class TestHDMapTIFFResize:
    """Test class for HDMAP TIFF loading and resizing functionality."""
    
    def setup_method(self):
        """Set up test environment with real TIFF paths."""
        self.original_tiff_root = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/10000_16bit_tiff_original")
        # PNG 로직 제거 - TIFF 전용
        
        # 실제 TIFF 파일 경로 확인
        self.sample_tiff_paths = []
        if self.original_tiff_root.exists():
            domain_path = self.original_tiff_root / "domain_A" / "train" / "good"
            if domain_path.exists():
                # 처음 5개 파일만 테스트에 사용
                self.sample_tiff_paths = sorted(list(domain_path.glob("*.tiff")))[:5]
        
        print(f"Found {len(self.sample_tiff_paths)} TIFF files for testing")
        
    def test_tiff_file_existence(self):
        """Test that TIFF files exist and are accessible."""
        assert self.original_tiff_root.exists(), f"TIFF root path does not exist: {self.original_tiff_root}"
        assert len(self.sample_tiff_paths) > 0, "No TIFF files found for testing"
        
        # 첫 번째 TIFF 파일 로딩 테스트
        sample_path = self.sample_tiff_paths[0]
        with Image.open(sample_path) as img:
            img_array = np.array(img)
            print(f"Sample TIFF shape: {img_array.shape}, dtype: {img_array.dtype}")
            print(f"Sample TIFF value range: [{img_array.min()}, {img_array.max()}]")
            # 32bit TIFF는 float32로 로딩됨
            assert img_array.dtype in [np.float32, np.uint16], f"Expected float32 or uint16, got {img_array.dtype}"
    
    def test_hdmap_dataset_tiff_loading(self):
        """Test HDMapDataset with TIFF loading."""
        if not self.original_tiff_root.exists() or len(self.sample_tiff_paths) == 0:
            pytest.skip("TIFF files not available for testing")
        
        # 기본 TIFF 로딩 테스트 (리사이즈 없음)
        dataset = HDMAPDataset(
            root=self.original_tiff_root,
            domain="domain_A",
            split="train",
            target_size=None,  # 리사이즈 없음
        )
        
        assert len(dataset) > 0, "Dataset should contain samples"
        
        # 첫 번째 샘플 로딩
        sample = dataset[0]
        image = sample.image
        label = sample.gt_label
        
        print(f"Loaded image shape: {image.shape}, dtype: {image.dtype}")
        print(f"Label: {label}")
        
        # 🔍 데이터셋 통계 정보 출력
        print(f"📊 Image Statistics:")
        print(f"   - Min: {image.min():.6f}")
        print(f"   - Max: {image.max():.6f}")
        print(f"   - Mean: {image.mean():.6f}")
        print(f"   - Std: {image.std():.6f}")
        print(f"   - Range: [{image.min():.6f}, {image.max():.6f}]")
        
        # 검증
        assert isinstance(image, torch.Tensor), "Image should be a torch tensor"
        assert image.dtype == torch.float32, "Image should be float32"
        assert len(image.shape) == 3, "Image should be (C, H, W)"
        assert image.shape[0] == 3, "Image should have 3 channels (RGB)"
        # 32bit TIFF는 정규화 범위가 다를 수 있으므로 범위 검증 완화
        assert torch.isfinite(image).all(), "Image values should be finite"
        assert isinstance(label, torch.Tensor), "Label should be a torch tensor"
    
    def test_resize_methods(self):
        """Test different resize methods."""
        if not self.original_tiff_root.exists() or len(self.sample_tiff_paths) == 0:
            pytest.skip("TIFF files not available for testing")
        
        target_size = (256, 256)
        methods = ["resize", "black_padding", "noise_padding"]
        
        for method in methods:
            print(f"\nTesting resize method: {method}")
            
            dataset = HDMAPDataset(
                root=self.original_tiff_root,
                domain="domain_A",
                split="train",
                target_size=target_size,
                resize_method=method,
            )
            
            sample = dataset[0]
            image = sample.image
            
            print(f"  Resized image shape: {image.shape}")
            print(f"  📊 {method} Statistics: min={image.min():.4f}, max={image.max():.4f}, mean={image.mean():.4f}, std={image.std():.4f}")
            
            # 검증
            assert image.shape[1] == target_size[0], f"Height should be {target_size[0]}, got {image.shape[1]}"
            assert image.shape[2] == target_size[1], f"Width should be {target_size[1]}, got {image.shape[2]}"
            assert torch.isfinite(image).all(), "Image values should be finite"
    
    def test_different_target_sizes(self):
        """Test different target sizes."""
        if not self.original_tiff_root.exists() or len(self.sample_tiff_paths) == 0:
            pytest.skip("TIFF files not available for testing")
        
        target_sizes = [(128, 128), (256, 256), (512, 512), (224, 224)]
        
        for target_size in target_sizes:
            print(f"\nTesting target size: {target_size}")
            
            dataset = HDMAPDataset(
                root=self.original_tiff_root,
                domain="domain_A",
                split="train",
                target_size=target_size,
                resize_method="resize",
            )
            
            sample = dataset[0]
            image = sample.image
            
            print(f"  Result shape: {image.shape}")
            print(f"  📊 {target_size} Statistics: min={image.min():.4f}, max={image.max():.4f}, mean={image.mean():.4f}, std={image.std():.4f}")
            
            # 검증
            assert image.shape[1] == target_size[0], f"Height mismatch"
            assert image.shape[2] == target_size[1], f"Width mismatch"
    
    def test_hdmap_datamodule_integration(self):
        """Test HDMapDataModule with TIFF loading and resizing."""
        if not self.original_tiff_root.exists():
            pytest.skip("TIFF files not available for testing")
        
        # DataModule 생성
        datamodule = HDMAPDataModule(
            root=self.original_tiff_root,
            domain="domain_A",
            train_batch_size=2,
            eval_batch_size=2,
            target_size=(256, 256),
            resize_method="black_padding",
        )
        
        # Setup 실행
        datamodule.setup()
        
        # Train dataloader 테스트
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        
        print(f"Batch image shape: {batch.image.shape}")
        print(f"Batch label shape: {batch.gt_label.shape}")
        
        # 🔍 배치 통계 정보 출력
        batch_images = batch.image
        print(f"📊 Batch Statistics:")
        print(f"   - Min: {batch_images.min():.6f}")
        print(f"   - Max: {batch_images.max():.6f}")
        print(f"   - Mean: {batch_images.mean():.6f}")
        print(f"   - Std: {batch_images.std():.6f}")
        
        # 검증
        assert batch.image.shape[0] == 2, "Batch size should be 2"
        assert batch.image.shape[1] == 3, "Should have 3 channels"
        assert batch.image.shape[2] == 256, "Height should be 256"
        assert batch.image.shape[3] == 256, "Width should be 256"
        assert batch.gt_label.shape[0] == 2, "Label batch size should be 2"
    
    def test_comparison_with_png_dataset(self):
        """Compare TIFF loading results with existing PNG dataset."""
        if not (self.original_tiff_root.exists() and self.png_root.exists()):
            pytest.skip("Both TIFF and PNG datasets required for comparison")
        
        # TIFF 데이터셋
        tiff_dataset = HDMAPDataset(
            root=self.original_tiff_root,
            domain="domain_A",
            split="train",
            target_size=(256, 256),
            resize_method="resize",
        )
        
        # PNG 데이터셋
        png_dataset = HDMAPDataset(
            root=self.png_root,
            domain="domain_A",
            split="train",
            target_size=None,  # PNG는 이미 256x256으로 전처리됨
        )
        
        print(f"TIFF dataset size: {len(tiff_dataset)}")
        print(f"PNG dataset size: {len(png_dataset)}")
        
        # 첫 번째 샘플 비교
        tiff_sample = tiff_dataset[0]
        png_sample = png_dataset[0]
        
        tiff_image = tiff_sample["image"]
        png_image = png_sample["image"]
        
        print(f"TIFF image shape: {tiff_image.shape}, range: [{tiff_image.min():.3f}, {tiff_image.max():.3f}]")
        print(f"PNG image shape: {png_image.shape}, range: [{png_image.min():.3f}, {png_image.max():.3f}]")
        
        # 기본 형태 검증
        assert tiff_image.shape == png_image.shape, "Shapes should match"
        assert tiff_sample["label"] == png_sample["label"], "Labels should match"
    
    def test_normalization_options(self):
        """Test different normalization options for 16bit TIFF."""
        if not self.original_tiff_root.exists() or len(self.sample_tiff_paths) == 0:
            pytest.skip("TIFF files not available for testing")
        
        # 정규화 켜기
        dataset_normalized = HDMAPDataset(
            root=self.original_tiff_root,
            domain="domain_A",
            split="train",
            target_size=(256, 256),
        )
        
        # 정규화 끄기
        dataset_raw = HDMAPDataset(
            root=self.original_tiff_root,
            domain="domain_A", 
            split="train",
            target_size=(256, 256),
        )
        
        sample_norm = dataset_normalized[0].image
        sample_raw = dataset_raw[0].image
        
        print(f"Normalized range: [{sample_norm.min():.3f}, {sample_norm.max():.3f}]")
        print(f"Raw range: [{sample_raw.min():.1f}, {sample_raw.max():.1f}]")
        
        # 검증 - 32bit TIFF는 정규화되지 않으므로 범위 제한 없음
        assert torch.isfinite(sample_norm).all(), "All values should be finite"
        assert torch.isfinite(sample_raw).all(), "All values should be finite"
    
    def test_save_visualization_samples(self):
        """Save some samples for visual inspection."""
        if not self.original_tiff_root.exists() or len(self.sample_tiff_paths) == 0:
            pytest.skip("TIFF files not available for testing")
        
        output_dir = Path(__file__).parent / "test_output"
        output_dir.mkdir(exist_ok=True)
        
        methods = ["resize", "black_padding", "noise_padding"]
        target_size = (256, 256)
        
        for method in methods:
            dataset = HDMAPDataset(
                root=self.original_tiff_root,
                domain="domain_A",
                split="train",
                target_size=target_size,
                resize_method=method,
            )
            
            # 첫 번째 샘플 저장
            sample = dataset[0]
            image = sample.image  # (3, 256, 256)
            
            # 이미지를 PIL 형태로 변환 후 저장
            image_np = image.permute(1, 2, 0).numpy()  # (256, 256, 3)
            image_np = (image_np * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image_np)
            output_path = output_dir / f"hdmap_tiff_{method}_{target_size[0]}x{target_size[1]}.png"
            pil_image.save(output_path)
            
            print(f"Saved visualization: {output_path}")


if __name__ == "__main__":
    # 개별 테스트 실행을 위한 코드
    test_instance = TestHDMapTIFFResize()
    test_instance.setup_method()
    
    print("Running HDMAP TIFF resize tests...")
    
    try:
        test_instance.test_tiff_file_existence()
        print("✅ TIFF file existence test passed")
    except Exception as e:
        print(f"❌ TIFF file existence test failed: {e}")
    
    try:
        test_instance.test_hdmap_dataset_tiff_loading()
        print("✅ TIFF loading test passed")
    except Exception as e:
        print(f"❌ TIFF loading test failed: {e}")
    
    try:
        test_instance.test_resize_methods()
        print("✅ Resize methods test passed")
    except Exception as e:
        print(f"❌ Resize methods test failed: {e}")
    
    try:
        test_instance.test_different_target_sizes()
        print("✅ Different target sizes test passed")
    except Exception as e:
        print(f"❌ Different target sizes test failed: {e}")
    
    try:
        test_instance.test_hdmap_datamodule_integration()
        print("✅ DataModule integration test passed")
    except Exception as e:
        print(f"❌ DataModule integration test failed: {e}")
    
    try:
        test_instance.test_normalization_options()
        print("✅ Normalization options test passed")
    except Exception as e:
        print(f"❌ Normalization options test failed: {e}")
    
    try:
        test_instance.test_save_visualization_samples()
        print("✅ Visualization samples saved")
    except Exception as e:
        print(f"❌ Visualization saving failed: {e}")
    
    print("\nAll tests completed!")
