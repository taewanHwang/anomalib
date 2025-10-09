#!/usr/bin/env python3
"""
HDMAP 데이터셋 통계 분석 및 시각화 스크립트

이 스크립트는 TIFF/PNG 형식의 HDMAP 데이터셋의 통계를 분석하고 시각화합니다.

사용법:
    python tests/unit/models/image/prepare/analyze_dataset_statistics.py

실행 전 설정:
    DATASET_PATH 변수를 분석할 데이터셋 경로로 수정하세요.
    예: "datasets/HDMAP/1000_tiff_original"
"""

import numpy as np
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

# =============================================================================
# 🔧 사용자 설정 (분석할 데이터셋 경로)
# =============================================================================
DATASET_PATH = "datasets/HDMAP/1000_tiff_minmax"  # 여기를 수정하세요


class DatasetStatisticsAnalyzer:
    """데이터셋 통계 분석 및 시각화 클래스"""

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path("tests/unit/models/image/prepare/outputs/dataset_statistics")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 통계 저장용
        self.statistics = defaultdict(lambda: defaultdict(list))
        self.folder_info = []

    def load_image(self, file_path):
        """이미지 파일 로드 (TIFF 또는 PNG)"""
        file_path = Path(file_path)

        if file_path.suffix.lower() in ['.tiff', '.tif']:
            # TIFF 파일
            img = tifffile.imread(str(file_path))
        elif file_path.suffix.lower() == '.png':
            # PNG 파일 (16-bit)
            import cv2
            img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            # PNG는 [0, 65535] 범위의 uint16이므로 [0, 1]로 정규화
            img = img.astype(np.float32) / 65535.0
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_path.suffix}")

        return img

    def analyze_folder(self, folder_path):
        """단일 폴더의 모든 이미지 분석"""
        folder_path = Path(folder_path)

        print(f"\n📁 분석 중: {folder_path.relative_to(self.dataset_path.parent)}")

        # 이미지 파일 찾기
        image_files = sorted(list(folder_path.glob("*.tiff")) +
                           list(folder_path.glob("*.tif")) +
                           list(folder_path.glob("*.png")))

        if not image_files:
            print(f"   ⚠️ 이미지 파일을 찾을 수 없습니다.")
            return None

        print(f"   📊 파일 수: {len(image_files)}")

        # 폴더별 통계
        folder_stats = {
            'path': str(folder_path.relative_to(self.dataset_path.parent)),
            'num_files': len(image_files),
            'mins': [],
            'maxs': [],
            'means': [],
            'stds': [],
            'shapes': [],
            'dtypes': []
        }

        # 각 파일 분석
        for idx, img_file in enumerate(image_files):
            try:
                img = self.load_image(img_file)

                # 통계 계산
                img_min = float(np.min(img))
                img_max = float(np.max(img))
                img_mean = float(np.mean(img))
                img_std = float(np.std(img))

                folder_stats['mins'].append(img_min)
                folder_stats['maxs'].append(img_max)
                folder_stats['means'].append(img_mean)
                folder_stats['stds'].append(img_std)
                folder_stats['shapes'].append(img.shape)
                folder_stats['dtypes'].append(str(img.dtype))

                # 진행률 출력 (매 100개마다)
                if (idx + 1) % 100 == 0:
                    print(f"   진행: {idx + 1}/{len(image_files)}")

            except Exception as e:
                print(f"   ❌ 파일 로드 실패: {img_file.name} - {e}")
                continue

        # 폴더 전체 통계 출력
        if folder_stats['mins']:
            print(f"   ✅ 분석 완료:")
            print(f"      - Shape: {folder_stats['shapes'][0]}")
            print(f"      - DType: {folder_stats['dtypes'][0]}")
            print(f"      - Min: {min(folder_stats['mins']):.6f} ~ {max(folder_stats['mins']):.6f}")
            print(f"      - Max: {min(folder_stats['maxs']):.6f} ~ {max(folder_stats['maxs']):.6f}")
            print(f"      - Mean: {min(folder_stats['means']):.6f} ~ {max(folder_stats['means']):.6f}")
            print(f"      - Std: {min(folder_stats['stds']):.6f} ~ {max(folder_stats['stds']):.6f}")
            print(f"      - Total Elements: {np.prod(folder_stats['shapes'][0]) * len(image_files):,}")

        return folder_stats

    def analyze_dataset(self):
        """전체 데이터셋 분석"""
        print("=" * 80)
        print("🔬 HDMAP 데이터셋 통계 분석 시작")
        print("=" * 80)
        print(f"📁 데이터셋 경로: {self.dataset_path}")

        # 도메인 폴더 찾기
        domain_dirs = sorted([d for d in self.dataset_path.iterdir()
                             if d.is_dir() and d.name.startswith('domain_')])

        if not domain_dirs:
            print(f"❌ {self.dataset_path}에서 도메인 폴더를 찾을 수 없습니다.")
            return

        print(f"📊 발견된 도메인: {len(domain_dirs)}개")
        print(f"   {[d.name for d in domain_dirs]}")

        # 각 도메인 분석
        for domain_dir in domain_dirs:
            # 하위 폴더 찾기 (train/good, test/good, test/fault)
            subfolders = [
                domain_dir / "train" / "good",
                domain_dir / "test" / "good",
                domain_dir / "test" / "fault"
            ]

            for subfolder in subfolders:
                if subfolder.exists():
                    stats = self.analyze_folder(subfolder)
                    if stats:
                        self.folder_info.append(stats)

        if not self.folder_info:
            print("❌ 분석할 수 있는 데이터가 없습니다.")
            return

        # 시각화 생성
        self.create_visualizations()

        # 요약 리포트 생성
        self.generate_summary_report()

        print(f"\n✅ 분석 완료!")
        print(f"📁 결과 저장 위치: {self.output_dir}")
        print("=" * 80)

    def create_visualizations(self):
        """통계 시각화 생성"""
        print(f"\n📊 시각화 생성 중...")

        # 각 폴더별로 시각화
        for folder_stats in self.folder_info:
            self.plot_folder_statistics(folder_stats)

        # 전체 요약 시각화
        self.plot_overall_summary()

    def plot_folder_statistics(self, folder_stats):
        """개별 폴더의 통계 시각화"""
        folder_name = Path(folder_stats['path']).name
        parent_name = Path(folder_stats['path']).parent.name
        grandparent_name = Path(folder_stats['path']).parent.parent.name

        # 파일명 생성 (domain_A_train_good.png)
        safe_name = f"{grandparent_name}_{parent_name}_{folder_name}".replace('/', '_')

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Statistics: {folder_stats["path"]}', fontsize=16, fontweight='bold')

        x = np.arange(len(folder_stats['mins']))

        # Min 값
        axes[0, 0].plot(x, folder_stats['mins'], 'b-', linewidth=0.5, alpha=0.7)
        axes[0, 0].set_title('Minimum Values')
        axes[0, 0].set_xlabel('File Index')
        axes[0, 0].set_ylabel('Min Value')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([min(folder_stats['mins']) - 0.01, max(folder_stats['mins']) + 0.01])

        # Max 값
        axes[0, 1].plot(x, folder_stats['maxs'], 'r-', linewidth=0.5, alpha=0.7)
        axes[0, 1].set_title('Maximum Values')
        axes[0, 1].set_xlabel('File Index')
        axes[0, 1].set_ylabel('Max Value')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([min(folder_stats['maxs']) - 0.01, max(folder_stats['maxs']) + 0.01])

        # Mean 값
        axes[1, 0].plot(x, folder_stats['means'], 'g-', linewidth=0.5, alpha=0.7)
        axes[1, 0].set_title('Mean Values')
        axes[1, 0].set_xlabel('File Index')
        axes[1, 0].set_ylabel('Mean Value')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([min(folder_stats['means']) - 0.01, max(folder_stats['means']) + 0.01])

        # Std 값
        axes[1, 1].plot(x, folder_stats['stds'], 'm-', linewidth=0.5, alpha=0.7)
        axes[1, 1].set_title('Standard Deviation Values')
        axes[1, 1].set_xlabel('File Index')
        axes[1, 1].set_ylabel('Std Value')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([min(folder_stats['stds']) - 0.01, max(folder_stats['stds']) + 0.01])

        plt.tight_layout()

        save_path = self.output_dir / f"{safe_name}_statistics.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   📊 저장됨: {save_path.name}")

    def plot_overall_summary(self):
        """전체 데이터셋 요약 시각화"""
        print(f"\n📈 전체 요약 시각화 생성 중...")

        # 폴더별 평균 통계 계산
        folder_names = []
        avg_mins = []
        avg_maxs = []
        avg_means = []
        avg_stds = []

        for folder_stats in self.folder_info:
            folder_names.append(folder_stats['path'].split('/')[-3:])  # domain/train/good
            avg_mins.append(np.mean(folder_stats['mins']))
            avg_maxs.append(np.mean(folder_stats['maxs']))
            avg_means.append(np.mean(folder_stats['means']))
            avg_stds.append(np.mean(folder_stats['stds']))

        # 폴더명 단축
        short_names = ['/'.join(name) for name in folder_names]

        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Overall Dataset Statistics Summary', fontsize=16, fontweight='bold')

        x = np.arange(len(short_names))

        # 평균 Min
        axes[0, 0].bar(x, avg_mins, color='blue', alpha=0.7)
        axes[0, 0].set_title('Average Minimum Values')
        axes[0, 0].set_xlabel('Folder')
        axes[0, 0].set_ylabel('Avg Min')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # 평균 Max
        axes[0, 1].bar(x, avg_maxs, color='red', alpha=0.7)
        axes[0, 1].set_title('Average Maximum Values')
        axes[0, 1].set_xlabel('Folder')
        axes[0, 1].set_ylabel('Avg Max')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # 평균 Mean
        axes[1, 0].bar(x, avg_means, color='green', alpha=0.7)
        axes[1, 0].set_title('Average Mean Values')
        axes[1, 0].set_xlabel('Folder')
        axes[1, 0].set_ylabel('Avg Mean')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # 평균 Std
        axes[1, 1].bar(x, avg_stds, color='magenta', alpha=0.7)
        axes[1, 1].set_title('Average Standard Deviation Values')
        axes[1, 1].set_xlabel('Folder')
        axes[1, 1].set_ylabel('Avg Std')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        save_path = self.output_dir / "overall_summary.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   📊 저장됨: {save_path.name}")

    def generate_summary_report(self):
        """요약 리포트 생성"""
        report_path = self.output_dir / "summary_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("HDMAP 데이터셋 통계 분석 리포트\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"데이터셋 경로: {self.dataset_path}\n")
            f.write(f"분석된 폴더 수: {len(self.folder_info)}\n\n")

            for folder_stats in self.folder_info:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"폴더: {folder_stats['path']}\n")
                f.write(f"{'=' * 80}\n")

                f.write(f"파일 수: {folder_stats['num_files']}\n")

                if folder_stats['shapes']:
                    f.write(f"Shape: {folder_stats['shapes'][0]}\n")
                    f.write(f"DType: {folder_stats['dtypes'][0]}\n")
                    f.write(f"Total Elements: {np.prod(folder_stats['shapes'][0]) * folder_stats['num_files']:,}\n\n")

                if folder_stats['mins']:
                    f.write(f"통계 (전체 파일):\n")
                    f.write(f"  Min 범위: [{min(folder_stats['mins']):.6f}, {max(folder_stats['mins']):.6f}]\n")
                    f.write(f"  Max 범위: [{min(folder_stats['maxs']):.6f}, {max(folder_stats['maxs']):.6f}]\n")
                    f.write(f"  Mean 범위: [{min(folder_stats['means']):.6f}, {max(folder_stats['means']):.6f}]\n")
                    f.write(f"  Std 범위: [{min(folder_stats['stds']):.6f}, {max(folder_stats['stds']):.6f}]\n\n")

                    f.write(f"통계 (평균값):\n")
                    f.write(f"  평균 Min: {np.mean(folder_stats['mins']):.6f}\n")
                    f.write(f"  평균 Max: {np.mean(folder_stats['maxs']):.6f}\n")
                    f.write(f"  평균 Mean: {np.mean(folder_stats['means']):.6f}\n")
                    f.write(f"  평균 Std: {np.mean(folder_stats['stds']):.6f}\n")

        print(f"\n📄 요약 리포트 저장됨: {report_path}")


def main():
    """메인 실행 함수"""
    # 데이터셋 경로 확인
    dataset_path = Path(DATASET_PATH)

    if not dataset_path.exists():
        print(f"❌ 데이터셋 경로를 찾을 수 없습니다: {dataset_path}")
        print(f"   현재 작업 디렉토리: {Path.cwd()}")
        print(f"\n💡 DATASET_PATH 변수를 올바른 경로로 수정하세요.")
        sys.exit(1)

    # 분석 실행
    analyzer = DatasetStatisticsAnalyzer(dataset_path)
    analyzer.analyze_dataset()


if __name__ == "__main__":
    main()
