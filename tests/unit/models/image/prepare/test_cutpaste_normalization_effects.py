#!/usr/bin/env python3
"""
CutPaste Augmentation 정규화 타입별 효과 테스트

이 테스트는 서로 다른 정규화 방식(zscore, minmax, original)에서 
CutPaste augmentation이 어떻게 다르게 동작하는지 시각화하여 확인합니다.

실행 방법:
cd /mnt/ex-disk/taewan.hwang/study/anomalib
python tests/unit/models/image/prepare/test_cutpaste_normalization_effects.py
"""

import os
import sys
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parents[5]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "DRAME_CutPaste"))

# 필요한 모듈 import
from DRAME_CutPaste.utils.utils_data_loader_v2 import HDmapdataset_v2

# =============================================================================
# 🔧 테스트 설정 (사용자가 직접 수정 가능)
# =============================================================================
A_FAULT_RANGE_START = 0    # CutPaste 패치 증폭 시작값
A_FAULT_RANGE_END = 0.1    # CutPaste 패치 증폭 끝값
NUM_SAMPLES = 20             # 각 정규화 타입별 생성할 샘플 수
DOMAIN = 'A'                 # 테스트할 도메인 (A, B, C, D)
RANDOM_SEED = 42             # 재현 가능한 결과를 위한 시드값
CUTPASTE_NORM = True

class CutPasteNormalizationTester:
    def __init__(self):
        self.project_root = Path(__file__).parents[5]
        self.data_root = self.project_root / "datasets" / "HDMAP"
        self.output_dir = self.project_root / "tests" / "unit" / "models" / "image" / "prepare" / "outputs"
        self.output_dir.mkdir(exist_ok=True)
        
        # 정규화 타입별 데이터 경로
        self.normalization_types = {
            'zscore': '1000_tiff_zscore',
            'minmax': '1000_tiff_minmax', 
            'original': '1000_tiff_original'
        }
        
    
    def load_sample_image(self, normalization_type, domain=DOMAIN):
        """각 정규화 타입별로 랜덤 샘플 이미지 로드"""
        data_path = self.data_root / self.normalization_types[normalization_type]
        train_good_path = data_path / f"domain_{domain}" / "train" / "good"
        
        # TIFF 파일 목록 가져오기
        tiff_files = list(train_good_path.glob("*.tiff"))
        if not tiff_files:
            tiff_files = list(train_good_path.glob("*.tif"))
            
        if not tiff_files:
            raise FileNotFoundError(f"No TIFF files found in {train_good_path}")
            
        # 랜덤 파일 선택
        random_file = random.choice(tiff_files)
        
        # 이미지 로드
        img = Image.open(random_file)
        img_array = np.array(img).astype(np.float32)
        
        # (H, W) -> (1, H, W) 차원 추가
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, str(random_file)
    
    def create_dummy_dataset(self, img_array):
        """단일 이미지로 HDmapdataset_v2 생성"""
        # 이미지를 (N, C, H, W) 형태로 변환
        data = np.expand_dims(img_array, axis=0)  # (1, 1, H, W)
        
        # HDmapdataset_v2 생성
        dataset = HDmapdataset_v2(
            data=data,
            a_fault_start=A_FAULT_RANGE_START,
            a_fault_range_end=A_FAULT_RANGE_END,
            norm=CUTPASTE_NORM,  # 정규화 활성화
            resize=(31, 95),
            method='resize',
            enable_augment=True,
            category=None  # 훈련 모드
        )
        
        return dataset
    
    def extract_augmentation_results(self, dataset, num_samples=NUM_SAMPLES):
        """여러 번 augmentation을 수행하여 결과 수집"""
        results = []
        
        for i in range(num_samples):
            sample = dataset[0]  # 첫 번째 (유일한) 이미지 사용
            
            original = sample['image'].numpy()  # (1, H, W)
            augmented = sample['augmented_image'].numpy()  # (1, H, W)
            mask = sample['anomaly_mask'].numpy()  # (1, H, W)
            has_anomaly = sample['has_anomaly'].item()
            fault_scale = sample['fault_scale']
            
            # 패치 영역 추출 (마스크가 1인 부분)
            if has_anomaly and np.sum(mask) > 0:
                patch_pixels = augmented[mask > 0]
            else:
                patch_pixels = np.array([])
            
            results.append({
                'original': original,
                'augmented': augmented,
                'mask': mask,
                'has_anomaly': has_anomaly,
                'fault_scale': fault_scale,
                'patch_pixels': patch_pixels
            })
            
        return results
    
    def plot_histograms(self, results_dict, save_path):
        """정규화 타입별 히스토그램 비교 플롯"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('CutPaste Augmentation: Pixel Value Histograms by Normalization Type', 
                     fontsize=16, fontweight='bold')
        
        colors = {'original': 'blue', 'augmented': 'red', 'patch': 'green'}
        
        for row, (norm_type, results) in enumerate(results_dict.items()):
            # 모든 샘플의 데이터 수집
            all_original = []
            all_augmented = []
            all_patches = []
            
            for result in results:
                all_original.extend(result['original'].flatten())
                all_augmented.extend(result['augmented'].flatten())
                if len(result['patch_pixels']) > 0:
                    all_patches.extend(result['patch_pixels'])
            
            # 원본 이미지 히스토그램
            axes[row, 0].hist(all_original, bins=50, alpha=0.7, color=colors['original'], density=True)
            axes[row, 0].set_title(f'{norm_type.upper()}: Original Image')
            axes[row, 0].set_xlabel('Pixel Value')
            axes[row, 0].set_ylabel('Density')
            axes[row, 0].grid(True, alpha=0.3)
            
            # 통계 정보 추가
            orig_stats = f'Mean: {np.mean(all_original):.4f}\nStd: {np.std(all_original):.4f}\nRange: [{np.min(all_original):.4f}, {np.max(all_original):.4f}]'
            axes[row, 0].text(0.02, 0.98, orig_stats, transform=axes[row, 0].transAxes, 
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                             fontsize=8)
            
            # Augmented 이미지 히스토그램
            axes[row, 1].hist(all_augmented, bins=50, alpha=0.7, color=colors['augmented'], density=True)
            axes[row, 1].set_title(f'{norm_type.upper()}: Augmented Image')
            axes[row, 1].set_xlabel('Pixel Value')
            axes[row, 1].set_ylabel('Density')
            axes[row, 1].grid(True, alpha=0.3)
            
            # 통계 정보 추가
            aug_stats = f'Mean: {np.mean(all_augmented):.4f}\nStd: {np.std(all_augmented):.4f}\nRange: [{np.min(all_augmented):.4f}, {np.max(all_augmented):.4f}]'
            axes[row, 1].text(0.02, 0.98, aug_stats, transform=axes[row, 1].transAxes, 
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
                             fontsize=8)
            
            # 패치 히스토그램
            if all_patches:
                axes[row, 2].hist(all_patches, bins=30, alpha=0.7, color=colors['patch'], density=True)
                patch_stats = f'Mean: {np.mean(all_patches):.4f}\nStd: {np.std(all_patches):.4f}\nRange: [{np.min(all_patches):.4f}, {np.max(all_patches):.4f}]'
                axes[row, 2].text(0.02, 0.98, patch_stats, transform=axes[row, 2].transAxes, 
                                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                                 fontsize=8)
            else:
                axes[row, 2].text(0.5, 0.5, 'No Anomaly Patches\n(All Normal Samples)', 
                                 transform=axes[row, 2].transAxes, ha='center', va='center',
                                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
            axes[row, 2].set_title(f'{norm_type.upper()}: Patch Only')
            axes[row, 2].set_xlabel('Pixel Value')
            axes[row, 2].set_ylabel('Density')
            axes[row, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 히스토그램 저장됨: {save_path}")
    
    def plot_image_comparisons(self, results_dict, save_path):
        """정규화 타입별 이미지 비교 플롯"""
        fig, axes = plt.subplots(3, 6, figsize=(24, 12))
        fig.suptitle('CutPaste Augmentation: Image Comparisons by Normalization Type', 
                     fontsize=16, fontweight='bold')
        
        for row, (norm_type, results) in enumerate(results_dict.items()):
            # anomaly가 있는 샘플 찾기
            anomaly_sample = None
            normal_sample = None
            
            for result in results:
                if result['has_anomaly'] and anomaly_sample is None:
                    anomaly_sample = result
                elif not result['has_anomaly'] and normal_sample is None:
                    normal_sample = result
                    
                if anomaly_sample is not None and normal_sample is not None:
                    break
            
            # Normal 샘플 시각화
            if normal_sample is not None:
                # 원본
                axes[row, 0].imshow(normal_sample['original'][0], cmap='viridis')
                axes[row, 0].set_title(f'{norm_type.upper()}: Normal Original')
                axes[row, 0].axis('off')
                
                # Augmented (Normal과 동일해야 함)
                axes[row, 1].imshow(normal_sample['augmented'][0], cmap='viridis')
                axes[row, 1].set_title(f'{norm_type.upper()}: Normal Augmented')
                axes[row, 1].axis('off')
                
                # 차이 (거의 0이어야 함)
                diff_normal = normal_sample['augmented'][0] - normal_sample['original'][0]
                im_diff_normal = axes[row, 2].imshow(diff_normal, cmap='RdBu', vmin=-0.1, vmax=0.1)
                axes[row, 2].set_title(f'{norm_type.upper()}: Normal Diff')
                axes[row, 2].axis('off')
                plt.colorbar(im_diff_normal, ax=axes[row, 2], shrink=0.8)
            
            # Anomaly 샘플 시각화
            if anomaly_sample is not None:
                # 원본
                axes[row, 3].imshow(anomaly_sample['original'][0], cmap='viridis')
                axes[row, 3].set_title(f'{norm_type.upper()}: Anomaly Original')
                axes[row, 3].axis('off')
                
                # Augmented
                axes[row, 4].imshow(anomaly_sample['augmented'][0], cmap='viridis')
                axes[row, 4].set_title(f'{norm_type.upper()}: Anomaly Augmented')
                axes[row, 4].axis('off')
                
                # 마스크 오버레이
                mask_overlay = anomaly_sample['original'][0].copy()
                mask_overlay[anomaly_sample['mask'][0] > 0] = np.max(mask_overlay) * 1.2  # 마스크 영역 강조
                axes[row, 5].imshow(mask_overlay, cmap='viridis')
                axes[row, 5].set_title(f'{norm_type.upper()}: Mask Overlay')
                axes[row, 5].axis('off')
                
                # 패치 정보 텍스트 추가
                fault_scale = anomaly_sample['fault_scale']
                if hasattr(fault_scale, '__iter__') and len(fault_scale) > 0:
                    fault_val = fault_scale[0]
                else:
                    fault_val = fault_scale
                    
                patch_info = f'Fault Scale: {fault_val:.2f}\nPatch Pixels: {len(anomaly_sample["patch_pixels"])}'
                axes[row, 5].text(0.02, 0.98, patch_info, transform=axes[row, 5].transAxes, 
                                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                                 fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"🖼️  이미지 비교 저장됨: {save_path}")
    
    def run_test(self):
        """전체 테스트 실행"""
        print("=" * 80)
        print("🧪 CutPaste Augmentation 정규화 타입별 효과 테스트 시작")
        print("=" * 80)
        
        results_dict = {}
        
        # 각 정규화 타입별로 테스트 수행
        for norm_type in self.normalization_types.keys():
            print(f"\n🔄 {norm_type.upper()} 정규화 타입 테스트 중...")
            
            # 샘플 이미지 로드
            img_array, filename = self.load_sample_image(norm_type)
            print(f"   - 샘플 파일: {filename}")
            print(f"   - 이미지 크기: {img_array.shape}")
            print(f"   - 값 범위: [{img_array.min():.4f}, {img_array.max():.4f}]")
            print(f"   - 평균: {img_array.mean():.4f}, 표준편차: {img_array.std():.4f}")
            
            # 데이터셋 생성
            dataset = self.create_dummy_dataset(img_array)
            
            # Augmentation 결과 수집
            results = self.extract_augmentation_results(dataset)
            results_dict[norm_type] = results
            
            # 통계 출력
            anomaly_count = sum(1 for r in results if r['has_anomaly'])
            print(f"   - 생성된 샘플: {len(results)}개")
            print(f"   - Anomaly 샘플: {anomaly_count}개")
            print(f"   - Normal 샘플: {len(results) - anomaly_count}개")
        
        # 시각화 생성
        print(f"\n📊 시각화 생성 중...")
        
        # 히스토그램 비교
        hist_save_path = self.output_dir / "cutpaste_normalization_histograms.png"
        self.plot_histograms(results_dict, hist_save_path)
        
        # 이미지 비교
        img_save_path = self.output_dir / "cutpaste_normalization_images.png"
        self.plot_image_comparisons(results_dict, img_save_path)
        
        # 요약 리포트 생성
        self.generate_summary_report(results_dict)
        
        print(f"\n✅ 테스트 완료!")
        print(f"📁 결과 저장 위치: {self.output_dir}")
        print("=" * 80)
    
    def generate_summary_report(self, results_dict):
        """요약 리포트 생성"""
        report_path = self.output_dir / "cutpaste_normalization_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("CutPaste Augmentation 정규화 타입별 효과 분석 리포트\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"테스트 설정:\n")
            f.write(f"- a_fault_range_start: {A_FAULT_RANGE_START}\n")
            f.write(f"- a_fault_range_end: {A_FAULT_RANGE_END}\n")
            f.write(f"- 샘플 수: {NUM_SAMPLES}개 (각 정규화 타입별)\n")
            f.write(f"- 도메인: {DOMAIN}\n\n")
            
            for norm_type, results in results_dict.items():
                f.write(f"{norm_type.upper()} 정규화 타입 분석:\n")
                f.write("-" * 40 + "\n")
                
                # 전체 통계
                all_original = np.concatenate([r['original'].flatten() for r in results])
                all_augmented = np.concatenate([r['augmented'].flatten() for r in results])
                all_patches = np.concatenate([r['patch_pixels'] for r in results if len(r['patch_pixels']) > 0])
                
                f.write(f"원본 이미지 통계:\n")
                f.write(f"  - 평균: {np.mean(all_original):.6f}\n")
                f.write(f"  - 표준편차: {np.std(all_original):.6f}\n")
                f.write(f"  - 범위: [{np.min(all_original):.6f}, {np.max(all_original):.6f}]\n\n")
                
                f.write(f"Augmented 이미지 통계:\n")
                f.write(f"  - 평균: {np.mean(all_augmented):.6f}\n")
                f.write(f"  - 표준편차: {np.std(all_augmented):.6f}\n")
                f.write(f"  - 범위: [{np.min(all_augmented):.6f}, {np.max(all_augmented):.6f}]\n\n")
                
                if len(all_patches) > 0:
                    f.write(f"패치 영역 통계:\n")
                    f.write(f"  - 평균: {np.mean(all_patches):.6f}\n")
                    f.write(f"  - 표준편차: {np.std(all_patches):.6f}\n")
                    f.write(f"  - 범위: [{np.min(all_patches):.6f}, {np.max(all_patches):.6f}]\n")
                    f.write(f"  - 원본 대비 패치 평균 비율: {np.mean(all_patches) / np.mean(all_original):.2f}x\n")
                else:
                    f.write(f"패치 영역: 없음 (모든 샘플이 Normal)\n")
                
                # Anomaly 비율
                anomaly_count = sum(1 for r in results if r['has_anomaly'])
                f.write(f"Anomaly 생성 비율: {anomaly_count}/{len(results)} ({anomaly_count/len(results)*100:.1f}%)\n\n")
        
        print(f"📄 요약 리포트 저장됨: {report_path}")


if __name__ == "__main__":
    # 랜덤 시드 설정 (재현 가능한 결과를 위해)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # 테스트 설정 출력
    print("🔧 테스트 설정:")
    print(f"   - A_FAULT_RANGE_START: {A_FAULT_RANGE_START}")
    print(f"   - A_FAULT_RANGE_END: {A_FAULT_RANGE_END}")
    print(f"   - NUM_SAMPLES: {NUM_SAMPLES}")
    print(f"   - DOMAIN: {DOMAIN}")
    print(f"   - RANDOM_SEED: {RANDOM_SEED}")
    print()
    
    # 테스트 실행
    tester = CutPasteNormalizationTester()
    tester.run_test()
