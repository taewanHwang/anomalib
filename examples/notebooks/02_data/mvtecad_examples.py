#!/usr/bin/env python3
"""MVTec AD DataModule Usage Examples / MVTec AD 데이터모듈 사용 예시들.

이 스크립트는 MVTec AD 데이터모듈의 다양한 사용법을 실제로 실행해볼 수 있는 예시들을 제공합니다.
This script provides practical examples of various usage patterns for the MVTec AD datamodule.

실행 방법 / How to run:
    python examples/notebooks/02_data/mvtecad_examples.py

주의사항 / Notes:
    - 처음 실행 시 MVTec AD 데이터셋이 자동으로 다운로드됩니다 (~4.9GB)
    - GPU가 있으면 더 빠르게 실행됩니다
    - The MVTec AD dataset will be automatically downloaded on first run (~4.9GB)
    - Runs faster with GPU if available
"""

import logging
from torchvision.transforms import v2

# anomalib imports
from anomalib.data import MVTecAD
from anomalib.data.utils import ValSplitMode

# 로깅 설정 / Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def save_sample_image(tensor_img, save_path):
    """샘플 이미지를 저장하는 함수 / Save a sample image to disk."""
    import torchvision.utils as vutils
    import os

    img_to_save = tensor_img.clone().detach()
    if img_to_save.min() < 0 or img_to_save.max() > 1:
        img_to_save = (img_to_save - img_to_save.min()) / (img_to_save.max() - img_to_save.min() + 1e-8)
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    vutils.save_image(img_to_save, save_path)
    print(f"✅ 샘플 이미지 1개 저장 완료: {save_path} / Saved one sample image: {save_path}")


def example_1_basic_usage():
    """예시 1: 기본 사용법 - 가장 간단한 형태 / Example 1: Basic usage - simplest form."""
    print("\n" + "="*60)
    print("예시 1: 기본 사용법 / Example 1: Basic Usage")
    print("="*60)
    
    try:
        # MVTec AD 데이터모듈 생성 / Create MVTec AD datamodule
        datamodule = MVTecAD(
            root="./datasets/MVTecAD",
            category="bottle",  # 15개 카테고리 중 하나 선택 / Choose one of 15 categories
            train_batch_size=16,
            eval_batch_size=8,
        )
        
        # 데이터 준비 / Prepare data
        print("데이터 준비 중... / Preparing data...")
        datamodule.prepare_data()  # 다운로드 (필요한 경우) / Download if needed
        datamodule.setup()  # 데이터 설정 / Setup data
        
        # 데이터로더 생성 및 정보 출력 / Create dataloaders and print info
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()
        
        print(f"✅ 훈련 배치 수 / Training batches: {len(train_loader)}")
        print(f"✅ 검증 배치 수 / Validation batches: {len(val_loader)}")
        print(f"✅ 테스트 배치 수 / Test batches: {len(test_loader)}")
        
        # 첫 번째 배치 확인 / Check first batch
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        test_batch = next(iter(test_loader))
        
        # ImageBatch 객체의 속성들 확인 / Check ImageBatch object attributes
        print(f"✅ 배치 타입 / Batch type: {type(train_batch)}")
        print(f"✅ 사용 가능한 속성들 / Available attributes: {dir(train_batch)}")
        
        print("\n" + "="*60)        
        # 주요 속성들 확인 / Check main attributes
        if hasattr(train_batch, 'image'):
            print(f"✅ 이미지 형태 / Image shape: {train_batch.image.shape}")
            print(f"✅ 이미지 최솟값 / Image min value: {train_batch.image.min().item():.3f}")
            print(f"✅ 이미지 최댓값 / Image max value: {train_batch.image.max().item():.3f}")
            print(f"✅ 이미지 평균값 / Image mean value: {train_batch.image.mean().item():.3f}")
            # 샘플 이미지 저장 / Save one sample image
            save_sample_image(train_batch.image[0], "./example1_sample_image.png")
        if hasattr(train_batch, 'gt_label'):
            print(f"✅ 라벨 형태 / Label shape: {train_batch.gt_label.shape}")
            print(f"✅ 라벨 값들 / Label values: {train_batch.gt_label}")
        if hasattr(train_batch, 'gt_mask'):
            print(f"✅ 마스크 형태 / Mask shape: {train_batch.gt_mask.shape}")
        if hasattr(train_batch, 'image_path'):
            print(f"✅ 이미지 경로 개수 / Number of image paths: {len(train_batch.image_path)}")
            print(f"✅ 첫 번째 이미지 경로 / First image path: {train_batch.image_path[0]}")
            
        print("\n" + "="*60)        
        if hasattr(val_batch, 'image'):
            print(f"✅ 이미지 형태 / Image shape: {val_batch.image.shape}")
            print(f"✅ 이미지 최솟값 / Image min value: {val_batch.image.min().item():.3f}")
            print(f"✅ 이미지 최댓값 / Image max value: {val_batch.image.max().item():.3f}")
            print(f"✅ 이미지 평균값 / Image mean value: {val_batch.image.mean().item():.3f}")

        print("\n" + "="*60)        
        if hasattr(test_batch, 'image'):
            print(f"✅ 이미지 형태 / Image shape: {test_batch.image.shape}")
            print(f"✅ 이미지 최솟값 / Image min value: {test_batch.image.min().item():.3f}")
            print(f"✅ 이미지 최댓값 / Image max value: {test_batch.image.max().item():.3f}")
            print(f"✅ 이미지 평균값 / Image mean value: {test_batch.image.mean().item():.3f}")
                
    except Exception as e:
        print(f"❌ 예시 1 실행 중 오류 / Error in Example 1: {e}")
        return False
    
    return True


def example_2_custom_settings():
    """예시 2: 다른 카테고리와 배치 크기 조정 / Example 2: Different category and batch size."""
    print("\n" + "="*60)
    print("예시 2: 설정 커스터마이징 / Example 2: Custom Settings")
    print("="*60)
    
    try:
        # 설정을 조정한 데이터모듈 / Datamodule with custom settings
        datamodule = MVTecAD(
            category="cable",  # 케이블 결함 탐지 / Cable defect detection
            train_batch_size=16,
            eval_batch_size=8,
        )
        
        print("케이블 카테고리로 데이터 설정 중... / Setting up cable category data...")
        datamodule.prepare_data()
        datamodule.setup()
        
        train_loader = datamodule.train_dataloader()
        
        # 배치 크기 확인 / Check batch size
        train_batch = next(iter(train_loader))
        batch_size = train_batch.image.shape[0]
        print(f"✅ 실제 배치 크기 / Actual batch size: {batch_size}")
        print(f"✅ 카테고리 / Category: {datamodule.category}")
        print(f"✅ 워커 수 / Number of workers: {datamodule.num_workers}")
        print(f"✅ 이미지 형태 / Image shape: {train_batch.image.shape}")
        print(f"✅ 이미지 최솟값 / Image min value: {train_batch.image.min().item():.3f}")
        print(f"✅ 이미지 최댓값 / Image max value: {train_batch.image.max().item():.3f}")
        print(f"✅ 이미지 평균값 / Image mean value: {train_batch.image.mean().item():.3f}")

        # 샘플 이미지 1개 저장 / Save one sample image
        save_sample_image(train_batch.image[0], "./example2_sample_image.png")
        
    except Exception as e:
        print(f"❌ 예시 2 실행 중 오류 / Error in Example 2: {e}")
        return False
    
    return True

def example_3_data_augmentation():
    """예시 3: 데이터 증강 적용 / Example 3: Data augmentation application."""
    print("\n" + "="*60)
    print("예시 3: 데이터 증강 / Example 3: Data Augmentation")
    print("="*60)
    
    try:
        # torchvision v2를 사용한 데이터 증강 정의 / Define data augmentation using torchvision v2
        # 
        # ⚠️ 이상 탐지(Anomaly Detection)에서 데이터 증강 주의사항 / Cautions for Data Augmentation in Anomaly Detection
        # 
        # A. 학습 방식에 따른 차이점 / Differences by Training Method:
        #    • 비지도 AD (정상만 학습, MVTec 표준) / Unsupervised AD (normal only, MVTec standard):
        #      - train에는 정상만 → 결함이 "사라질" 위험이 적음 / Only normal in train → low risk of defect loss
        #      - 그러나 지나친 RandomResizedCrop/강한 회전은 "정상 패턴"을 왜곡시켜 분포학습 방해 가능
        #        / However, excessive RandomResizedCrop/strong rotation can distort normal patterns
        #    • (세미)지도 AD/Localization 학습 (마스크·BoundingBox 사용) / (Semi-)supervised AD/Localization:
        #      - 결함이 포함된 샘플 학습 시, 크롭으로 결함이 프레임 밖으로 나가지 않게 주의
        #        / When training with defect samples, ensure crops don't push defects out of frame
        #      - 해결방법: 마스크/박스 보존형 크롭 또는 약한 Affine 변환만 사용
        #        / Solution: mask/box-preserving crop or only weak affine transformations
        # 
        # B. 검증/테스트는 기하 변환 금지 / Prohibit geometric transforms for val/test:
        #    • Pixel-level metric (PRO/IoU 등) 사용 시 val/test에서 회전/크롭 금지
        #      / When using pixel-level metrics, avoid rotation/crop in val/test
        #    • 마스크 정합이 깨져 평가가 왜곡됨 / Mask alignment breaks, distorting evaluation
        #    • (Resize/Normalize는 PreProcessor에서 "결정적"으로만 적용)
        #      / (Resize/Normalize should only be applied "deterministically" in PreProcessor)
        # 
        # C. ratio·scale의 보수적 설정 / Conservative ratio·scale settings:
        #    • ratio: 1.0에 가깝게 (예: 0.75~1.33). 너무 극단적인 비율은 결함을 잘라낼 위험↑
        #      / ratio: close to 1.0 (e.g., 0.75~1.33). Extreme ratios risk cutting defects
        #    • scale: 1.0에 가깝게 (예: 0.85~1.0). 과도한 줌-인은 결함 일부만 남기거나 사라지게 함
        #      / scale: close to 1.0 (e.g., 0.85~1.0). Excessive zoom-in leaves partial defects or removes them
        
        print("torchvision v2로 데이터 증강 설정 중... / Setting up data augmentation with torchvision v2...")
        train_aug = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),  # 50% 확률로 수평 뒤집기, 결함이 대칭성에 관계없는 경우에 유용 / 50% horizontal flip
            v2.RandomVerticalFlip(p=0.2),    # 20% 확률로 수직 뒤집기, 결함이 대칭성에 관계없는 경우에 유용 / 20% vertical flip  
            v2.RandomRotation(degrees=10),   # ±10도 회전, 카메라 설치각이나 제품 위치가 조금씩 달라지는 상황 대응 / ±10 degree rotation
            v2.RandomResizedCrop(
                size=(224, 224),           # 출력 이미지 크기 (height, width) / Output image size (height, width)
                scale=(0.85, 1.0),         # 원본 이미지에서 남길 상대적 크기 범위 85%~100% 사이 랜덤 / Range of the proportion of the original image to crop (min, max)
                ratio=(0.9, 1.1),          # 크롭 영역의 종횡비 범위 0.9~1.1 (보수적 설정) / Aspect ratio range of the crop (conservative setting)
                interpolation=2,           # 리사이즈 보간 방식 (2=bilinear, 기본값) / Interpolation method for resizing (2=bilinear, default)
                antialias=True             # 리사이즈 시 앤티앨리어싱 적용 (기본값 True) / Apply antialiasing when resizing (default True)
            ),  # 크롭 후 리사이즈, 보수적 설정으로 결함 손실 최소화 / Crop and resize, conservative settings to minimize defect loss
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),  # 색상 조정 / Color adjustment
        ])
        
        # Example1과 동일한 설정 + 데이터 증강 추가 / Same settings as Example1 + data augmentation
        print("MVTecAD 데이터모듈 설정 중... / Setting up MVTecAD datamodule...")
        datamodule = MVTecAD(
            root="./datasets/MVTecAD",
            category="bottle",  # Example1과 동일한 카테고리 / Same category as Example1
            train_batch_size=16,
            eval_batch_size=8,
            train_augmentations=train_aug,    # 훈련에만 증강 적용 / Apply augmentation only to training
            val_augmentations=None,           # 검증에는 증강 적용 안함 / No augmentation for validation  
            test_augmentations=None,          # 테스트에는 증강 적용 안함 / No augmentation for test
        )
        
        datamodule.prepare_data()
        datamodule.setup()
        
        # 데이터로더 생성 / Create dataloaders
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()
        
        print(f"✅ 훈련 배치 수 / Training batches: {len(train_loader)}")
        print(f"✅ 검증 배치 수 / Validation batches: {len(val_loader)}")
        print(f"✅ 테스트 배치 수 / Test batches: {len(test_loader)}")
        
        
        # 배치 데이터 가져오기 / Get batch data
        print("배치 데이터 가져오는 중... / Getting batch data...")
        train_batch = next(iter(train_loader))  # 증강 적용됨 / With augmentation
        val_batch = next(iter(val_loader))      # 증강 적용 안됨 / No augmentation
        test_batch = next(iter(test_loader))    # 증강 적용 안됨 / No augmentation
        
        print("\n🎯 데이터 증강 효과 비교 / Data Augmentation Effect Comparison")
        print("="*60) 
        
        # 이미지 저장 및 비교 / Save images and compare
            
        if hasattr(train_batch, 'image'):
            save_sample_image(train_batch.image[0], "./example3_train_augmented.png")
            print("✅ 증강된 훈련 이미지 저장 완료 / Augmented training image saved")
            
        if hasattr(val_batch, 'image'):
            save_sample_image(val_batch.image[0], "./example3_val_original.png")
            print("✅ 검증 이미지 저장 완료 / Validation image saved")

        if hasattr(test_batch, 'image'):
            save_sample_image(test_batch.image[0], "./example3_test_original.png")
            print("✅ 테스트 이미지 저장 완료 / Test image saved")
        
        # 증강 설정 정보 출력 / Print augmentation settings info
        print(f"\n📊 증강 설정 정보 / Augmentation Settings Info:")
        print(f"✅ 훈련 데이터 증강 / Training augmentations: 적용됨 (v2.Compose) / Applied (v2.Compose)")
        print(f"✅ 검증 데이터 증강 / Validation augmentations: {datamodule.val_augmentations}")
        print(f"✅ 테스트 데이터 증강 / Test augmentations: {datamodule.test_augmentations}")
        
        print("\n" + "="*60)        
        print("🎯 비교 포인트 / Comparison Points:")
        print("1. example1_sample_image.png: 기본 이미지 (증강 없음) / Basic image (no augmentation)")
        print("2. example3_train_augmented.png: 훈련용 증강 이미지 / Training augmented image")
        print("   - 수평/수직 뒤집기, 회전, 보수적 크롭, 색상 조정이 적용됨")
        print("   - Horizontal/vertical flip, rotation, conservative crop, color jitter applied")
        print("3. example3_val_original.png: 검증용 원본 이미지 / Validation original image")
        print("4. example3_test_original.png: 테스트용 원본 이미지 / Test original image")
        print("✨ 증강은 훈련 데이터에만 적용되고 검증/테스트는 원본 그대로! / Augmentation only for training, val/test remain original!")
        print("이상 탐지에서는 결함 손실을 방지하기 위해 보수적 설정 사용! / Conservative settings to prevent defect loss in AD!")
        
        
                
    except Exception as e:
        print(f"❌ 예시 3 실행 중 오류 / Error in Example 3: {e}")
        return False
    
    return True


def example_4_validation_split_modes():
    """예시 4: 검증 세트 분할 모드 비교 / Example 4: Comparison of validation split modes."""
    print("\n" + "="*60)
    print("예시 4: 검증 세트 분할 모드 비교 / Example 4: Validation Split Mode Comparison")
    print("="*60)
    
    # 🎯 검증 세트 분할 모드별 특성 설명 / Characteristics of validation split modes
    print("\n📚 검증 세트 분할 모드 이해 / Understanding Validation Split Modes:")
    print("1. FROM_TEST: 테스트 데이터에서 일부를 검증용으로 분리 / Split some test data for validation")
    print("   - 장점: 실제 이상 데이터 포함 / Pros: Contains real anomaly data")
    print("   - 단점: 테스트 세트 크기 감소 / Cons: Reduces test set size")
    print("2. SAME_AS_TEST: 테스트 데이터를 검증용으로도 사용 / Use test data as validation")
    print("   - 주의: 데이터 누수 위험! / Caution: Risk of data leakage!")
    
    try:
        # 모드 1: FROM_TEST - 테스트에서 검증 분리 / Mode 1: FROM_TEST - Split validation from test
        print("\n" + "="*50)
        print("🔄 모드 1: FROM_TEST (테스트에서 검증 분리)")
        print("="*50)
        datamodule_from_test = MVTecAD(
            root="./datasets/MVTecAD",
            category="bottle",  # Example1과 동일한 카테고리로 비교 / Same category as Example1 for comparison
            train_batch_size=16,
            eval_batch_size=8,
            val_split_mode=ValSplitMode.FROM_TEST,
            val_split_ratio=0.3,  # 테스트 데이터의 30%를 검증용으로 / 30% of test data for validation
            seed=42  # 재현 가능한 분할 / Reproducible split
        )
        
        datamodule_from_test.prepare_data()
        datamodule_from_test.setup()
        
        train_loader_1 = datamodule_from_test.train_dataloader()
        val_loader_1 = datamodule_from_test.val_dataloader()
        test_loader_1 = datamodule_from_test.test_dataloader()
        
        print(f"✅ 훈련 배치 수 / Training batches: {len(train_loader_1)}")
        print(f"✅ 검증 배치 수 / Validation batches: {len(val_loader_1)}")
        print(f"✅ 테스트 배치 수 / Test batches: {len(test_loader_1)}")
        
        # 검증 데이터 특성 확인 / Check validation data characteristics
        val_batch_1 = next(iter(val_loader_1))
        if hasattr(val_batch_1, 'gt_label'):
            normal_count_1 = (~val_batch_1.gt_label).sum().item()  # False (정상) 개수
            anomaly_count_1 = val_batch_1.gt_label.sum().item()    # True (이상) 개수
            print(f"🔍 검증 데이터 구성 - 정상/이상: {normal_count_1}/{anomaly_count_1}")
        
        # 모드 2: SAME_AS_TEST - 테스트와 동일 / Mode 2: SAME_AS_TEST - Same as test
        print("\n" + "="*50)
        print("🔄 모드 2: SAME_AS_TEST (테스트와 동일)")
        print("="*50)
        datamodule_same_as_test = MVTecAD(
            root="./datasets/MVTecAD",
            category="bottle",
            train_batch_size=16,
            eval_batch_size=8,
            val_split_mode=ValSplitMode.SAME_AS_TEST,  # 테스트 데이터를 검증으로도 사용 / Use test data as validation
            seed=42
        )
        
        datamodule_same_as_test.prepare_data()
        datamodule_same_as_test.setup()
        
        val_loader_2 = datamodule_same_as_test.val_dataloader()
        test_loader_2 = datamodule_same_as_test.test_dataloader()
        
        print(f"✅ 검증 배치 수 / Validation batches: {len(val_loader_2)}")
        print(f"✅ 테스트 배치 수 / Test batches: {len(test_loader_2)}")
        print("⚠️ 주의: 검증과 테스트가 동일함 - 데이터 누수 위험! / Caution: Val and test are identical - data leakage risk!")
        

        
        # 비교 요약 / Comparison summary
        print("\n" + "="*60)
        print("📊 분할 모드별 비교 요약 / Split Mode Comparison Summary")
        print("="*60)
        print(f"1. FROM_TEST    - 검증: {len(val_loader_1):2d}배치, 테스트: {len(test_loader_1):2d}배치")
        print(f"2. SAME_AS_TEST - 검증: {len(val_loader_2):2d}배치, 테스트: {len(test_loader_2):2d}배치 (동일)")
        
        print("\n🎯 사용 권장사항 / Usage Recommendations:")
        print("• 연구/실험용: FROM_TEST (가장 현실적) / Research: FROM_TEST (most realistic)")
        print("• 빠른 프로토타입: SAME_AS_TEST (주의 필요) / Quick prototype: SAME_AS_TEST (caution needed)")
        
    except Exception as e:
        print(f"❌ 예시 4 실행 중 오류 / Error in Example 4: {e}")
        return False
    
    return True

if __name__ == "__main__":
    """모든 예시 실행 / Run all examples."""
    print("MVTec AD DataModule 사용 예시들 / MVTec AD DataModule Usage Examples")
    print("=" * 80)
    
    # 예시들 실행 / Run examples
    example_1_basic_usage()
    example_2_custom_settings()
    example_3_data_augmentation()
    example_4_validation_split_modes()
