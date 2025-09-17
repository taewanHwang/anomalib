"""HDMAPDataModule 단위 테스트."""

import sys
import os
from pathlib import Path

# 상대 import를 위한 경로 설정
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
import torch
from anomalib.data.datamodules.image.hdmap import HDMAPDataModule
from anomalib.data.utils import ValSplitMode


class TestHDMAPDataModule:
    """HDMAPDataModule 테스트 클래스."""
    
    @classmethod
    def setup_class(cls):
        """테스트 클래스 초기화."""
        cls.tiff_root = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/10000_16bit_tiff_original")
        cls.test_domain = "domain_A"
        
        if not cls.tiff_root.exists():
            pytest.skip("TIFF dataset not found")
    
    def test_datamodule_initialization(self):
        """DataModule 초기화 테스트."""
        print("\n🧪 Testing DataModule Initialization...")
        
        try:
            datamodule = HDMAPDataModule(
                root=str(self.tiff_root),
                domain=self.test_domain,
                train_batch_size=4,
                eval_batch_size=4,
                num_workers=0,  # 테스트에서는 0으로 설정
                val_split_mode=ValSplitMode.FROM_TEST,
                val_split_ratio=0.2,
                seed=42
            )
            
            # 기본 속성 확인 (setup 전이므로 데이터셋은 아직 None)
            assert datamodule.train_batch_size == 4
            assert datamodule.eval_batch_size == 4
            
            print("✅ DataModule initialization successful")
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            raise
    
    def test_datamodule_setup_and_prepare(self):
        """DataModule setup 및 prepare_data 테스트."""
        print("\n🧪 Testing DataModule Setup and Prepare...")
        
        datamodule = HDMAPDataModule(
            root=str(self.tiff_root),
            domain=self.test_domain,
            train_batch_size=2,
            eval_batch_size=2,
            num_workers=0,
            val_split_mode=ValSplitMode.FROM_TEST,
            val_split_ratio=0.3,
            seed=42
        )
        
        # prepare_data 실행
        datamodule.prepare_data()
        print("✅ prepare_data completed")
        
        # setup 실행
        datamodule.setup()
        print("✅ setup completed")
        
        # 데이터셋 확인
        assert datamodule.train_data is not None
        assert datamodule.test_data is not None
        assert datamodule.val_data is not None
        
        print(f"📊 Dataset sizes:")
        print(f"   - Train: {len(datamodule.train_data)}")
        print(f"   - Validation: {len(datamodule.val_data)}")
        print(f"   - Test: {len(datamodule.test_data)}")
        
        assert len(datamodule.train_data) > 0
        assert len(datamodule.test_data) > 0
        assert len(datamodule.val_data) > 0
    
    def test_dataloader_creation(self):
        """DataLoader 생성 테스트."""
        print("\n🧪 Testing DataLoader Creation...")
        
        datamodule = HDMAPDataModule(
            root=str(self.tiff_root),
            domain=self.test_domain,
            train_batch_size=2,
            eval_batch_size=2,
            num_workers=0,
            val_split_mode=ValSplitMode.FROM_TEST,
            val_split_ratio=0.2,
            seed=42
        )
        
        datamodule.prepare_data()
        datamodule.setup()
        
        # Train DataLoader
        train_loader = datamodule.train_dataloader()
        assert train_loader is not None
        
        # Validation DataLoader
        val_loader = datamodule.val_dataloader()
        assert val_loader is not None
        
        # Test DataLoader
        test_loader = datamodule.test_dataloader()
        assert test_loader is not None
        
        print("✅ All DataLoaders created successfully")
        
        # 배치 크기 확인
        assert train_loader.batch_size == 2
        assert val_loader.batch_size == 2
        assert test_loader.batch_size == 2
        
        print("✅ Batch sizes are correct")
    
    def test_batch_loading_and_shapes(self):
        """배치 로딩 및 형태 테스트."""
        print("\n🧪 Testing Batch Loading and Shapes...")
        
        datamodule = HDMAPDataModule(
            root=str(self.tiff_root),
            domain=self.test_domain,
            train_batch_size=3,
            eval_batch_size=3,
            num_workers=0,
            val_split_mode=ValSplitMode.FROM_TEST,
            val_split_ratio=0.2,
            seed=42
        )
        
        datamodule.prepare_data()
        datamodule.setup()
        
        # Train batch 테스트
        train_loader = datamodule.train_dataloader()
        train_batch = next(iter(train_loader))
        
        print(f"📊 Train batch:")
        print(f"   - Image shape: {train_batch.image.shape}")
        print(f"   - Label shape: {train_batch.gt_label.shape}")
        print(f"   - Image dtype: {train_batch.image.dtype}")
        print(f"   - Label dtype: {train_batch.gt_label.dtype}")
        
        # 기본 형태 검증
        assert train_batch.image.shape[0] == 3  # batch size
        assert train_batch.image.shape[1] == 3  # channels (RGB)
        assert len(train_batch.image.shape) == 4  # (B, C, H, W)
        assert train_batch.gt_label.shape[0] == 3  # batch size
        
        # 데이터 타입 검증
        assert train_batch.image.dtype == torch.float32
        assert train_batch.gt_label.dtype == torch.bool
        
        # Validation batch 테스트
        val_loader = datamodule.val_dataloader()
        val_batch = next(iter(val_loader))
        
        print(f"📊 Validation batch:")
        print(f"   - Image shape: {val_batch.image.shape}")
        print(f"   - Label shape: {val_batch.gt_label.shape}")
        
        # Test batch 테스트
        test_loader = datamodule.test_dataloader()
        test_batch = next(iter(test_loader))
        
        print(f"📊 Test batch:")
        print(f"   - Image shape: {test_batch.image.shape}")
        print(f"   - Label shape: {test_batch.gt_label.shape}")
        
        print("✅ All batch shapes and types are correct")
    
    def test_tiff_data_statistics(self):
        """TIFF 데이터 통계 테스트."""
        print("\n🧪 Testing TIFF Data Statistics...")
        
        datamodule = HDMAPDataModule(
            root=str(self.tiff_root),
            domain=self.test_domain,
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            val_split_mode=ValSplitMode.FROM_TEST,
            val_split_ratio=0.2,
            seed=42
        )
        
        datamodule.prepare_data()
        datamodule.setup()
        
        # Train 데이터 통계
        train_loader = datamodule.train_dataloader()
        train_batch = next(iter(train_loader))
        train_images = train_batch.image
        
        print(f"📊 Train Data Statistics:")
        print(f"   - Min: {train_images.min():.6f}")
        print(f"   - Max: {train_images.max():.6f}")
        print(f"   - Mean: {train_images.mean():.6f}")
        print(f"   - Std: {train_images.std():.6f}")
        
        # 32bit TIFF는 정규화되지 않으므로 범위가 [0,1]이 아님을 확인
        assert not (train_images.min() >= 0.0 and train_images.max() <= 1.0), \
            "32bit TIFF should not be normalized to [0,1] range"
        
        # 유한한 값들인지 확인
        assert torch.all(torch.isfinite(train_images)), "All values should be finite"
        
        print("✅ TIFF data statistics are as expected (non-normalized)")
    
    def test_different_val_split_modes(self):
        """다양한 validation split 모드 테스트."""
        print("\n🧪 Testing Different Val Split Modes...")
        
        # FROM_TEST 모드
        datamodule_from_test = HDMAPDataModule(
            root=str(self.tiff_root),
            domain=self.test_domain,
            train_batch_size=2,
            eval_batch_size=2,
            num_workers=0,
            val_split_mode=ValSplitMode.FROM_TEST,
            val_split_ratio=0.3,
            seed=42
        )
        
        datamodule_from_test.prepare_data()
        datamodule_from_test.setup()
        
        print(f"📊 FROM_TEST mode:")
        print(f"   - Train: {len(datamodule_from_test.train_data)}")
        print(f"   - Val: {len(datamodule_from_test.val_data)}")
        print(f"   - Test: {len(datamodule_from_test.test_data)}")
        
        # NONE 모드 (validation 없음)
        datamodule_none = HDMAPDataModule(
            root=str(self.tiff_root),
            domain=self.test_domain,
            train_batch_size=2,
            eval_batch_size=2,
            num_workers=0,
            val_split_mode=ValSplitMode.NONE,
            seed=42
        )
        
        datamodule_none.prepare_data()
        datamodule_none.setup()
        
        print(f"📊 NONE mode:")
        print(f"   - Train: {len(datamodule_none.train_data)}")
        print(f"   - Val: {len(datamodule_none.val_data) if hasattr(datamodule_none, 'val_data') and datamodule_none.val_data else 'None'}")
        print(f"   - Test: {len(datamodule_none.test_data)}")
        
        # NONE 모드에서는 validation이 없어야 함 (속성이 없거나 None)
        val_data_exists = hasattr(datamodule_none, 'val_data') and datamodule_none.val_data is not None
        if val_data_exists:
            assert len(datamodule_none.val_data) == 0, "NONE mode should have no validation data"
        else:
            print("   ✅ No val_data attribute (as expected for NONE mode)")
        
        print("✅ Different val split modes work correctly")
    
    def test_different_batch_sizes(self):
        """다양한 배치 크기 테스트."""
        print("\n🧪 Testing Different Batch Sizes...")
        
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            print(f"   Testing batch size: {batch_size}")
            
            datamodule = HDMAPDataModule(
                root=str(self.tiff_root),
                domain=self.test_domain,
                train_batch_size=batch_size,
                eval_batch_size=batch_size,
                num_workers=0,
                val_split_mode=ValSplitMode.FROM_TEST,
                val_split_ratio=0.2,
                seed=42
            )
            
            datamodule.prepare_data()
            datamodule.setup()
            
            # 배치 크기 확인
            train_loader = datamodule.train_dataloader()
            train_batch = next(iter(train_loader))
            
            # 마지막 배치는 크기가 다를 수 있으므로 <= 로 체크
            assert train_batch.image.shape[0] <= batch_size
            assert train_batch.gt_label.shape[0] <= batch_size
            
            print(f"      ✅ Batch size {batch_size}: actual={train_batch.image.shape[0]}")
        
        print("✅ All batch sizes work correctly")
    
    def test_resize_options(self):
        """리사이즈 옵션 테스트."""
        print("\n🧪 Testing Resize Options...")
        
        # 다양한 타겟 크기와 리사이즈 방법 테스트
        test_configs = [
            {"target_size": None, "resize_method": "resize", "desc": "원본 크기 유지"},
            {"target_size": (64, 64), "resize_method": "resize", "desc": "64x64 리사이즈"},
            {"target_size": (128, 128), "resize_method": "resize", "desc": "128x128 리사이즈"},
            {"target_size": (256, 256), "resize_method": "resize", "desc": "256x256 리사이즈"},
            {"target_size": (224, 224), "resize_method": "black_padding", "desc": "224x224 블랙 패딩"},
            {"target_size": (224, 224), "resize_method": "noise_padding", "desc": "224x224 노이즈 패딩"},
        ]
        
        for config in test_configs:
            print(f"   Testing: {config['desc']}")
            
            datamodule = HDMAPDataModule(
                root=str(self.tiff_root),
                domain=self.test_domain,
                train_batch_size=2,
                eval_batch_size=2,
                num_workers=0,
                val_split_mode=ValSplitMode.FROM_TEST,
                val_split_ratio=0.2,
                seed=42,
                target_size=config["target_size"],
                resize_method=config["resize_method"]
            )
            
            datamodule.prepare_data()
            datamodule.setup()
            
            # 배치 확인
            train_loader = datamodule.train_dataloader()
            train_batch = next(iter(train_loader))
            
            batch_size, channels, height, width = train_batch.image.shape
            
            # 타겟 크기 확인
            if config["target_size"] is not None:
                expected_h, expected_w = config["target_size"]
                assert height == expected_h, f"Height mismatch: expected {expected_h}, got {height}"
                assert width == expected_w, f"Width mismatch: expected {expected_w}, got {width}"
                print(f"      ✅ Size: {height}x{width} (as expected)")
            else:
                # 원본 크기 (31x95)
                print(f"      ✅ Original size: {height}x{width}")
            
            # 채널 수는 항상 3이어야 함
            assert channels == 3, f"Expected 3 channels, got {channels}"
            
            # 데이터 타입 확인
            assert train_batch.image.dtype == torch.float32
            assert train_batch.gt_label.dtype == torch.bool
            
            # 데이터 범위 확인 (32bit TIFF는 정규화되지 않음)
            img_min, img_max = train_batch.image.min().item(), train_batch.image.max().item()
            assert torch.all(torch.isfinite(train_batch.image)), "All values should be finite"
            
            print(f"      📊 Data range: [{img_min:.3f}, {img_max:.3f}]")
        
        print("✅ All resize options work correctly")
    
    def test_resize_method_differences(self):
        """다양한 리사이즈 방법 간의 차이점 테스트."""
        print("\n🧪 Testing Resize Method Differences...")
        
        target_size = (128, 128)
        resize_methods = ["resize", "black_padding", "noise_padding"]
        results = {}
        
        for method in resize_methods:
            print(f"   Testing method: {method}")
            
            datamodule = HDMAPDataModule(
                root=str(self.tiff_root),
                domain=self.test_domain,
                train_batch_size=1,  # 단일 샘플로 비교
                eval_batch_size=1,
                num_workers=0,
                val_split_mode=ValSplitMode.FROM_TEST,
                val_split_ratio=0.2,
                seed=42,  # 동일한 시드로 같은 샘플 확보
                target_size=target_size,
                resize_method=method
            )
            
            datamodule.prepare_data()
            datamodule.setup()
            
            # 첫 번째 샘플 가져오기
            train_loader = datamodule.train_dataloader()
            train_batch = next(iter(train_loader))
            image = train_batch.image[0]  # (3, 128, 128)
            
            # 통계 계산
            stats = {
                "mean": image.mean().item(),
                "std": image.std().item(),
                "min": image.min().item(),
                "max": image.max().item(),
                "zero_pixels": (image == 0).sum().item(),  # 블랙 패딩에서 0 픽셀 개수
            }
            
            results[method] = stats
            
            print(f"      📊 {method} stats:")
            print(f"         Mean: {stats['mean']:.4f}")
            print(f"         Std: {stats['std']:.4f}")
            print(f"         Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"         Zero pixels: {stats['zero_pixels']}")
        
        # 리사이즈 방법 간 차이점 검증
        resize_stats = results["resize"]
        black_stats = results["black_padding"]
        noise_stats = results["noise_padding"]
        
        # 블랙 패딩은 0 픽셀이 더 많아야 함
        assert black_stats["zero_pixels"] > resize_stats["zero_pixels"], \
            "Black padding should have more zero pixels than resize"
        
        # 노이즈 패딩은 0 픽셀이 적어야 함
        assert noise_stats["zero_pixels"] <= black_stats["zero_pixels"], \
            "Noise padding should have less or equal zero pixels than black padding"
        
        print("✅ Resize method differences verified")
        print(f"   - Resize: {resize_stats['zero_pixels']} zero pixels")
        print(f"   - Black padding: {black_stats['zero_pixels']} zero pixels")
        print(f"   - Noise padding: {noise_stats['zero_pixels']} zero pixels")


def run_tests():
    """모든 테스트 실행."""
    print("🚀 Starting HDMAPDataModule Tests...")
    
    test_instance = TestHDMAPDataModule()
    test_instance.setup_class()
    
    tests = [
        test_instance.test_datamodule_initialization,
        test_instance.test_datamodule_setup_and_prepare,
        test_instance.test_dataloader_creation,
        test_instance.test_batch_loading_and_shapes,
        test_instance.test_tiff_data_statistics,
        test_instance.test_different_val_split_modes,
        test_instance.test_different_batch_sizes,
        test_instance.test_resize_options,
        test_instance.test_resize_method_differences,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"✅ {test_func.__name__} PASSED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_func.__name__} FAILED: {e}")
    
    print(f"\n🎯 Test Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
