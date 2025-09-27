#!/usr/bin/env python3
"""
CutPaste Synthetic Generator v2 분석 테스트

synthetic_generator_v2.py의 CutPasteSyntheticGenerator를 사용하여
다양한 정규화 조건에서 augmentation 결과를 분석합니다.

실행 방법:
cd /mnt/ex-disk/taewan.hwang/study/anomalib
python tests/unit/models/image/draem_cutpaste_clf/test_synthetic_generator_analysis.py
"""

import os
import sys
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from typing import Dict, List, Tuple

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parents[5]
sys.path.insert(0, str(project_root))

# 필요한 모듈 import
from src.anomalib.models.image.draem_cutpaste_clf.synthetic_generator_v2 import CutPasteSyntheticGenerator

# =============================================================================
# 🔧 테스트 설정 (사용자가 직접 수정 가능)
# =============================================================================
class TestConfig:
    """테스트 설정"""
    # CutPaste 파라미터
    CUT_W_RANGE = (10, 80)        # 패치 너비 범위
    CUT_H_RANGE = (1, 2)          # 패치 높이 범위
    A_FAULT_START = 1.0           # 증폭 시작값
    A_FAULT_RANGE_END = 2.0      # 증폭 끝값
    PROBABILITY = 0.5             # Anomaly 생성 확률

    # 테스트 파라미터
    NUM_SAMPLES = 30              # 각 정규화 타입별 생성할 샘플 수
    DOMAIN = 'A'                  # 테스트할 도메인
    RANDOM_SEED = 42              # 재현 가능한 결과를 위한 시드값
    IMAGE_SIZE = (31, 95)         # 리사이즈 크기 (H, W)

    # 출력 설정
    OUTPUT_DIR = project_root / "tests" / "unit" / "models" / "image" / "draem_cutpaste_clf" / "outputs_synthetic_v2"
    SAVE_INDIVIDUAL_SAMPLES = True  # 개별 샘플 저장 여부


class SyntheticGeneratorAnalyzer:
    """CutPasteSyntheticGenerator 분석기"""

    def __init__(self, config: TestConfig):
        self.config = config
        self.project_root = project_root
        self.data_root = self.project_root / "datasets" / "HDMAP"
        self.output_dir = config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 정규화 타입별 데이터 경로
        self.normalization_types = {
            'zscore': '10000_tiff_zscore',
            'minmax': '10000_tiff_minmax',
            'original': '10000_tiff_original'
        }

        # Generator 초기화
        self.generator = CutPasteSyntheticGenerator(
            cut_w_range=config.CUT_W_RANGE,
            cut_h_range=config.CUT_H_RANGE,
            a_fault_start=config.A_FAULT_START,
            a_fault_range_end=config.A_FAULT_RANGE_END,
            probability=config.PROBABILITY,
            validation_enabled=True
        )

        print(f"✅ Generator 초기화 완료")
        print(f"   설정: {self.generator.get_config_info()}")

    def load_sample_image(self, normalization_type: str, domain: str = None) -> Tuple[torch.Tensor, str]:
        """샘플 이미지 로드 및 전처리"""
        domain = domain or self.config.DOMAIN
        data_path = self.data_root / self.normalization_types[normalization_type]
        train_good_path = data_path / f"domain_{domain}" / "train" / "good"

        # TIFF 파일 목록
        tiff_files = list(train_good_path.glob("*.tiff")) + list(train_good_path.glob("*.tif"))
        if not tiff_files:
            raise FileNotFoundError(f"No TIFF files found in {train_good_path}")

        # 랜덤 파일 선택
        random_file = random.choice(tiff_files)

        # 이미지 로드 및 리사이즈
        img = Image.open(random_file)
        if self.config.IMAGE_SIZE:
            img = img.resize((self.config.IMAGE_SIZE[1], self.config.IMAGE_SIZE[0]))  # (W, H) for PIL

        # numpy array로 변환 후 tensor로
        img_array = np.array(img).astype(np.float32)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, H, W)

        return img_tensor, str(random_file)

    def generate_samples(self, img_tensor: torch.Tensor, num_samples: int) -> List[Dict]:
        """여러 augmentation 샘플 생성"""
        results = []

        for i in range(num_samples):
            # Generator 호출 (patch_info 포함)
            synthetic, mask, severity_label, patch_info = self.generator(
                img_tensor,
                return_patch_info=True
            )

            # 결과 저장
            result = {
                'idx': i,
                'original': img_tensor.clone(),
                'synthetic': synthetic,
                'mask': mask,
                'severity_label': severity_label.item(),
                'patch_info': patch_info,
                'has_anomaly': patch_info['has_anomaly']
            }

            # 패치 픽셀 추출
            if result['has_anomaly']:
                mask_bool = mask[0, 0] > 0
                result['patch_pixels'] = synthetic[0, 0][mask_bool].cpu().numpy()
                result['original_patch_pixels'] = img_tensor[0][mask_bool].cpu().numpy()
            else:
                result['patch_pixels'] = np.array([])
                result['original_patch_pixels'] = np.array([])

            results.append(result)

        return results

    def analyze_statistics(self, results: List[Dict]) -> Dict:
        """통계 분석"""
        stats = {
            'total_samples': len(results),
            'anomaly_count': sum(1 for r in results if r['has_anomaly']),
            'normal_count': sum(1 for r in results if not r['has_anomaly'])
        }

        # 원본 이미지 통계
        all_original = np.concatenate([r['original'].numpy().flatten() for r in results])
        stats['original'] = {
            'mean': float(np.mean(all_original)),
            'std': float(np.std(all_original)),
            'min': float(np.min(all_original)),
            'max': float(np.max(all_original))
        }

        # Synthetic 이미지 통계
        all_synthetic = np.concatenate([r['synthetic'].numpy().flatten() for r in results])
        stats['synthetic'] = {
            'mean': float(np.mean(all_synthetic)),
            'std': float(np.std(all_synthetic)),
            'min': float(np.min(all_synthetic)),
            'max': float(np.max(all_synthetic))
        }

        # 패치 영역 통계 (anomaly가 있는 경우만)
        all_patches = []
        all_original_patches = []
        severity_values = []

        for r in results:
            if r['has_anomaly'] and len(r['patch_pixels']) > 0:
                all_patches.extend(r['patch_pixels'])
                all_original_patches.extend(r['original_patch_pixels'])
                severity_values.append(r['severity_label'])

        if all_patches:
            stats['patches'] = {
                'mean': float(np.mean(all_patches)),
                'std': float(np.std(all_patches)),
                'min': float(np.min(all_patches)),
                'max': float(np.max(all_patches)),
                'count': len(all_patches)
            }
            stats['original_patches'] = {
                'mean': float(np.mean(all_original_patches)),
                'std': float(np.std(all_original_patches)),
                'min': float(np.min(all_original_patches)),
                'max': float(np.max(all_original_patches))
            }
            stats['severity'] = {
                'mean': float(np.mean(severity_values)),
                'std': float(np.std(severity_values)),
                'min': float(np.min(severity_values)),
                'max': float(np.max(severity_values))
            }

            # 증폭 비율 계산
            stats['amplification_ratio'] = stats['patches']['mean'] / stats['original_patches']['mean'] if stats['original_patches']['mean'] != 0 else 0

        stats['anomaly_ratio'] = stats['anomaly_count'] / stats['total_samples']

        return stats

    def plot_analysis(self, all_results: Dict[str, List[Dict]], save_path: Path):
        """종합 분석 플롯 생성"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.25)

        fig.suptitle('CutPaste Synthetic Generator v2 Analysis', fontsize=16, fontweight='bold')

        norm_types = list(all_results.keys())
        colors = {'zscore': 'blue', 'minmax': 'green', 'original': 'red'}

        # 1. 히스토그램 비교 (각 정규화 타입별)
        for idx, norm_type in enumerate(norm_types):
            results = all_results[norm_type]
            color = colors[norm_type]

            # Original vs Synthetic 히스토그램
            ax = fig.add_subplot(gs[idx, 0])

            all_original = np.concatenate([r['original'].numpy().flatten() for r in results])
            all_synthetic = np.concatenate([r['synthetic'].numpy().flatten() for r in results])

            ax.hist(all_original, bins=50, alpha=0.5, label='Original', color='gray', density=True)
            ax.hist(all_synthetic, bins=50, alpha=0.5, label='Synthetic', color=color, density=True)
            ax.set_title(f'{norm_type.upper()}: Original vs Synthetic')
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 패치 영역 히스토그램
            ax = fig.add_subplot(gs[idx, 1])

            patch_pixels = []
            original_patch_pixels = []
            for r in results:
                if r['has_anomaly'] and len(r['patch_pixels']) > 0:
                    patch_pixels.extend(r['patch_pixels'])
                    original_patch_pixels.extend(r['original_patch_pixels'])

            if patch_pixels:
                ax.hist(original_patch_pixels, bins=30, alpha=0.5, label='Original Patch', color='gray', density=True)
                ax.hist(patch_pixels, bins=30, alpha=0.5, label='Augmented Patch', color=color, density=True)
                ax.set_title(f'{norm_type.upper()}: Patch Comparison')
                ax.set_xlabel('Pixel Value')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No Anomaly Patches', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{norm_type.upper()}: Patch Comparison')

            # Severity 분포
            ax = fig.add_subplot(gs[idx, 2])

            severity_values = [r['severity_label'] for r in results if r['has_anomaly']]
            if severity_values:
                ax.hist(severity_values, bins=20, color=color, alpha=0.7)
                ax.set_title(f'{norm_type.upper()}: Severity Distribution')
                ax.set_xlabel('Severity (Amplitude)')
                ax.set_ylabel('Count')
                ax.grid(True, alpha=0.3)

                # 통계 정보 추가
                stats_text = f"Mean: {np.mean(severity_values):.2f}\nStd: {np.std(severity_values):.2f}"
                ax.text(0.7, 0.9, stats_text, transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 4. 샘플 이미지 비교 (마지막 행)
        for idx, norm_type in enumerate(norm_types):
            results = all_results[norm_type]

            # Anomaly 샘플 찾기
            anomaly_sample = None
            for r in results:
                if r['has_anomaly']:
                    anomaly_sample = r
                    break

            if anomaly_sample:
                ax = fig.add_subplot(gs[3, idx])

                # Original과 Synthetic을 나란히 표시
                original_img = anomaly_sample['original'][0].numpy().squeeze()  # Remove channel dimension
                synthetic_img = anomaly_sample['synthetic'][0, 0].numpy()
                mask = anomaly_sample['mask'][0, 0].numpy()

                # 3개 이미지를 하나로 합치기 (2D arrays)
                separator = np.ones((original_img.shape[0], 2)) * np.max(original_img)
                combined = np.hstack([
                    original_img,
                    separator,
                    synthetic_img,
                    separator,
                    mask * np.max(original_img)
                ])

                ax.imshow(combined, cmap='viridis')
                ax.set_title(f'{norm_type.upper()}: Original | Synthetic | Mask')
                ax.axis('off')

                # 패치 정보 추가
                patch_info = anomaly_sample['patch_info']
                info_text = f"Size: {patch_info['cut_w']}×{patch_info['cut_h']}\n"
                info_text += f"Amplitude: {patch_info['a_fault']:.2f}\n"
                info_text += f"Coverage: {patch_info['coverage_percentage']:.1f}%"
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 분석 플롯 저장됨: {save_path}")

    def save_individual_samples(self, results: List[Dict], norm_type: str, output_dir: Path):
        """개별 샘플 저장"""
        sample_dir = output_dir / f"samples_{norm_type}"
        sample_dir.mkdir(exist_ok=True)

        # Anomaly 샘플만 저장 (최대 5개)
        anomaly_samples = [r for r in results if r['has_anomaly']][:5]

        for i, sample in enumerate(anomaly_samples):
            try:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                # Original
                axes[0].imshow(sample['original'][0].numpy().squeeze(), cmap='viridis')
                axes[0].set_title('Original')
                axes[0].axis('off')

                # Synthetic
                axes[1].imshow(sample['synthetic'][0, 0].numpy(), cmap='viridis')
                axes[1].set_title('Synthetic')
                axes[1].axis('off')

                # Mask
                axes[2].imshow(sample['mask'][0, 0].numpy(), cmap='hot')
                axes[2].set_title('Mask')
                axes[2].axis('off')

                # 패치 정보 추가
                patch_info = sample['patch_info']
                info_text = f"Patch: {patch_info['cut_w']}×{patch_info['cut_h']}\n"
                info_text += f"From: ({patch_info['from_location_w']}, {patch_info['from_location_h']})\n"
                info_text += f"To: ({patch_info['to_location_w']}, {patch_info['to_location_h']})\n"
                info_text += f"Amplitude: {patch_info['a_fault']:.2f}"

                fig.text(0.02, 0.5, info_text, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                plt.tight_layout()
                save_path = sample_dir / f"sample_{i:02d}.png"
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"      Warning: Could not save sample {i}: {e}")
                plt.close()
                continue

        if len(anomaly_samples) > 0:
            print(f"   💾 샘플 저장됨: {sample_dir}")

    def generate_report(self, all_stats: Dict[str, Dict], save_path: Path):
        """분석 리포트 생성"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CutPaste Synthetic Generator v2 Analysis Report\n")
            f.write("=" * 80 + "\n\n")

            # Generator 설정
            f.write("Generator Configuration:\n")
            f.write("-" * 40 + "\n")
            config_info = self.generator.get_config_info()
            for key, value in config_info.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            # 테스트 설정
            f.write("Test Configuration:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  NUM_SAMPLES: {self.config.NUM_SAMPLES}\n")
            f.write(f"  DOMAIN: {self.config.DOMAIN}\n")
            f.write(f"  IMAGE_SIZE: {self.config.IMAGE_SIZE}\n")
            f.write(f"  RANDOM_SEED: {self.config.RANDOM_SEED}\n\n")

            # 각 정규화 타입별 통계
            for norm_type, stats in all_stats.items():
                f.write(f"\n{norm_type.upper()} Normalization Analysis:\n")
                f.write("=" * 60 + "\n")

                f.write(f"Sample Distribution:\n")
                f.write(f"  Total: {stats['total_samples']}\n")
                f.write(f"  Anomaly: {stats['anomaly_count']} ({stats['anomaly_ratio']*100:.1f}%)\n")
                f.write(f"  Normal: {stats['normal_count']} ({(1-stats['anomaly_ratio'])*100:.1f}%)\n\n")

                f.write(f"Original Image Statistics:\n")
                f.write(f"  Mean: {stats['original']['mean']:.6f}\n")
                f.write(f"  Std: {stats['original']['std']:.6f}\n")
                f.write(f"  Range: [{stats['original']['min']:.6f}, {stats['original']['max']:.6f}]\n\n")

                f.write(f"Synthetic Image Statistics:\n")
                f.write(f"  Mean: {stats['synthetic']['mean']:.6f}\n")
                f.write(f"  Std: {stats['synthetic']['std']:.6f}\n")
                f.write(f"  Range: [{stats['synthetic']['min']:.6f}, {stats['synthetic']['max']:.6f}]\n\n")

                if 'patches' in stats:
                    f.write(f"Patch Region Statistics:\n")
                    f.write(f"  Original Patch Mean: {stats['original_patches']['mean']:.6f}\n")
                    f.write(f"  Augmented Patch Mean: {stats['patches']['mean']:.6f}\n")
                    f.write(f"  Amplification Ratio: {stats['amplification_ratio']:.3f}x\n")
                    f.write(f"  Patch Std: {stats['patches']['std']:.6f}\n")
                    f.write(f"  Patch Range: [{stats['patches']['min']:.6f}, {stats['patches']['max']:.6f}]\n\n")

                    f.write(f"Severity Statistics:\n")
                    f.write(f"  Mean: {stats['severity']['mean']:.3f}\n")
                    f.write(f"  Std: {stats['severity']['std']:.3f}\n")
                    f.write(f"  Range: [{stats['severity']['min']:.3f}, {stats['severity']['max']:.3f}]\n")
                else:
                    f.write(f"Patch Region Statistics: No anomaly patches generated\n")

                f.write("\n" + "-" * 60 + "\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("End of Report\n")
            f.write("=" * 80 + "\n")

        print(f"📄 리포트 저장됨: {save_path}")

    def run_analysis(self):
        """전체 분석 실행"""
        print("\n" + "=" * 80)
        print("🚀 CutPaste Synthetic Generator v2 Analysis Started")
        print("=" * 80 + "\n")

        all_results = {}
        all_stats = {}

        # 각 정규화 타입별로 분석
        for norm_type in self.normalization_types.keys():
            print(f"\n📊 Analyzing {norm_type.upper()} normalization...")

            try:
                # 이미지 로드
                img_tensor, filename = self.load_sample_image(norm_type)
                print(f"   ✓ Image loaded: {Path(filename).name}")
                print(f"   ✓ Shape: {img_tensor.shape}")
                print(f"   ✓ Range: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")

                # 샘플 생성
                results = self.generate_samples(img_tensor, self.config.NUM_SAMPLES)
                all_results[norm_type] = results

                # 통계 분석
                stats = self.analyze_statistics(results)
                all_stats[norm_type] = stats

                print(f"   ✓ Generated {len(results)} samples")
                print(f"   ✓ Anomaly ratio: {stats['anomaly_ratio']*100:.1f}%")

                # 개별 샘플 저장
                if self.config.SAVE_INDIVIDUAL_SAMPLES:
                    self.save_individual_samples(results, norm_type, self.output_dir)

            except Exception as e:
                print(f"   ❌ Error processing {norm_type}: {e}")
                continue

        # 종합 분석 플롯
        print(f"\n📊 Generating analysis plots...")
        plot_path = self.output_dir / "synthetic_v2_analysis.png"
        self.plot_analysis(all_results, plot_path)

        # 리포트 생성
        report_path = self.output_dir / "synthetic_v2_report.txt"
        self.generate_report(all_stats, report_path)

        print("\n" + "=" * 80)
        print("✅ Analysis Complete!")
        print(f"📁 Results saved to: {self.output_dir}")
        print("=" * 80)

        return all_results, all_stats


def main():
    """메인 실행 함수"""
    # 설정 초기화
    config = TestConfig()

    # 랜덤 시드 설정
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    # 분석기 생성 및 실행
    analyzer = SyntheticGeneratorAnalyzer(config)
    results, stats = analyzer.run_analysis()

    # 선택적: 결과를 반환하여 추가 분석 가능
    return results, stats


if __name__ == "__main__":
    main()