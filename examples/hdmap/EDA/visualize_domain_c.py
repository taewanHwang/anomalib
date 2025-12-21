"""
Domain C 이미지 시각화 분석 스크립트
- 정상(good) vs 고장(fault) 이미지 비교
- TIFF 원본 데이터 사용
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random

# 경로 설정
TIFF_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_tiff_minmax/domain_C")
OUTPUT_DIR = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/examples/hdmap/EDA/results/domain_c_visual")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_tiff_image(path: Path) -> np.ndarray:
    """TIFF 이미지 로드 (float32 유지)"""
    img = Image.open(path)
    return np.array(img, dtype=np.float32)

def get_image_stats(img: np.ndarray) -> dict:
    """이미지 통계 계산"""
    return {
        'min': float(np.min(img)),
        'max': float(np.max(img)),
        'mean': float(np.mean(img)),
        'std': float(np.std(img)),
        'shape': img.shape
    }

def visualize_samples(n_samples: int = 6):
    """정상/고장 샘플 시각화"""

    # 파일 경로 수집
    train_good_dir = TIFF_ROOT / "train" / "good"
    test_good_dir = TIFF_ROOT / "test" / "good"
    test_fault_dir = TIFF_ROOT / "test" / "fault"

    train_good_files = sorted(list(train_good_dir.glob("*.tiff")))
    test_good_files = sorted(list(test_good_dir.glob("*.tiff")))
    test_fault_files = sorted(list(test_fault_dir.glob("*.tiff")))

    print(f"Train Good: {len(train_good_files)} files")
    print(f"Test Good: {len(test_good_files)} files")
    print(f"Test Fault: {len(test_fault_files)} files")

    # 랜덤 샘플 선택
    random.seed(42)
    train_samples = random.sample(train_good_files, min(n_samples, len(train_good_files)))
    test_good_samples = random.sample(test_good_files, min(n_samples, len(test_good_files)))
    test_fault_samples = random.sample(test_fault_files, min(n_samples, len(test_fault_files)))

    # Figure 1: Train Good vs Test Good vs Test Fault 비교
    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 3, 10))
    fig.suptitle('Domain C: Train Good vs Test Good vs Test Fault', fontsize=16, fontweight='bold')

    row_labels = ['Train Good', 'Test Good', 'Test Fault']
    sample_lists = [train_samples, test_good_samples, test_fault_samples]

    for row_idx, (label, samples) in enumerate(zip(row_labels, sample_lists)):
        for col_idx, img_path in enumerate(samples):
            img = load_tiff_image(img_path)
            stats = get_image_stats(img)

            ax = axes[row_idx, col_idx]
            im = ax.imshow(img, cmap='viridis')
            ax.set_title(f'{img_path.stem}\nμ={stats["mean"]:.2f}, σ={stats["std"]:.2f}', fontsize=8)
            ax.axis('off')

            if col_idx == 0:
                ax.set_ylabel(label, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'domain_c_comparison_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'domain_c_comparison_grid.png'}")

    # Figure 2: 더 자세한 정상 vs 고장 비교 (동일 vmin/vmax 사용)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Domain C: Detailed Good vs Fault Comparison (Same Color Scale)', fontsize=14, fontweight='bold')

    # 전체 데이터의 min/max 계산
    all_good = [load_tiff_image(f) for f in test_good_samples[:4]]
    all_fault = [load_tiff_image(f) for f in test_fault_samples[:4]]

    global_min = min(np.min(img) for img in all_good + all_fault)
    global_max = max(np.max(img) for img in all_good + all_fault)

    for col_idx in range(4):
        # Good
        ax_good = axes[0, col_idx]
        img_good = all_good[col_idx]
        im = ax_good.imshow(img_good, cmap='viridis', vmin=global_min, vmax=global_max)
        ax_good.set_title(f'Good: {test_good_samples[col_idx].stem}', fontsize=10)
        ax_good.axis('off')

        # Fault
        ax_fault = axes[1, col_idx]
        img_fault = all_fault[col_idx]
        im = ax_fault.imshow(img_fault, cmap='viridis', vmin=global_min, vmax=global_max)
        ax_fault.set_title(f'Fault: {test_fault_samples[col_idx].stem}', fontsize=10)
        ax_fault.axis('off')

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Pixel Intensity')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'domain_c_good_vs_fault_detailed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'domain_c_good_vs_fault_detailed.png'}")

    # Figure 3: Difference Map (Fault와 가장 유사한 Good 찾아서 차이 시각화)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Domain C: Fault Images with Mean Good Subtracted', fontsize=14, fontweight='bold')

    # Mean good image 계산
    mean_good = np.mean([load_tiff_image(f) for f in train_good_files[:100]], axis=0)

    for col_idx in range(4):
        fault_img = load_tiff_image(test_fault_samples[col_idx])
        diff_img = fault_img - mean_good

        # Fault 이미지
        axes[0, col_idx].imshow(fault_img, cmap='viridis')
        axes[0, col_idx].set_title(f'Fault: {test_fault_samples[col_idx].stem}', fontsize=10)
        axes[0, col_idx].axis('off')

        # Mean Good
        axes[1, col_idx].imshow(mean_good, cmap='viridis')
        axes[1, col_idx].set_title('Mean Good (100 samples)', fontsize=10)
        axes[1, col_idx].axis('off')

        # Difference
        im = axes[2, col_idx].imshow(diff_img, cmap='RdBu_r', vmin=-np.max(np.abs(diff_img)), vmax=np.max(np.abs(diff_img)))
        axes[2, col_idx].set_title(f'Difference (max abs: {np.max(np.abs(diff_img)):.2f})', fontsize=10)
        axes[2, col_idx].axis('off')

    plt.colorbar(im, ax=axes[2, :], orientation='horizontal', fraction=0.05, pad=0.1, label='Difference')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'domain_c_difference_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'domain_c_difference_map.png'}")

def analyze_intensity_distribution():
    """정상/고장 픽셀 강도 분포 분석"""

    train_good_dir = TIFF_ROOT / "train" / "good"
    test_good_dir = TIFF_ROOT / "test" / "good"
    test_fault_dir = TIFF_ROOT / "test" / "fault"

    train_good_files = list(train_good_dir.glob("*.tiff"))
    test_good_files = list(test_good_dir.glob("*.tiff"))
    test_fault_files = list(test_fault_dir.glob("*.tiff"))

    # 샘플링 (속도를 위해)
    n_sample = 200
    random.seed(42)

    train_good_sample = random.sample(train_good_files, min(n_sample, len(train_good_files)))
    test_good_sample = random.sample(test_good_files, min(n_sample, len(test_good_files)))
    test_fault_sample = random.sample(test_fault_files, min(n_sample, len(test_fault_files)))

    # 통계 수집
    def collect_stats(files, label):
        means, stds, mins, maxs = [], [], [], []
        for f in files:
            img = load_tiff_image(f)
            means.append(np.mean(img))
            stds.append(np.std(img))
            mins.append(np.min(img))
            maxs.append(np.max(img))
        return {
            'label': label,
            'means': means,
            'stds': stds,
            'mins': mins,
            'maxs': maxs
        }

    print("Collecting statistics...")
    train_good_stats = collect_stats(train_good_sample, 'Train Good')
    test_good_stats = collect_stats(test_good_sample, 'Test Good')
    test_fault_stats = collect_stats(test_fault_sample, 'Test Fault')

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Domain C: Intensity Distribution Analysis', fontsize=14, fontweight='bold')

    colors = {'Train Good': 'green', 'Test Good': 'blue', 'Test Fault': 'red'}

    for stats in [train_good_stats, test_good_stats, test_fault_stats]:
        label = stats['label']
        color = colors[label]
        alpha = 0.5 if label != 'Test Fault' else 0.7

        axes[0, 0].hist(stats['means'], bins=50, alpha=alpha, label=label, color=color)
        axes[0, 1].hist(stats['stds'], bins=50, alpha=alpha, label=label, color=color)
        axes[1, 0].hist(stats['mins'], bins=50, alpha=alpha, label=label, color=color)
        axes[1, 1].hist(stats['maxs'], bins=50, alpha=alpha, label=label, color=color)

    axes[0, 0].set_xlabel('Mean Intensity')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Mean Intensity Distribution')
    axes[0, 0].legend()

    axes[0, 1].set_xlabel('Std Intensity')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Std Intensity Distribution')
    axes[0, 1].legend()

    axes[1, 0].set_xlabel('Min Intensity')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Min Intensity Distribution')
    axes[1, 0].legend()

    axes[1, 1].set_xlabel('Max Intensity')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Max Intensity Distribution')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'domain_c_intensity_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'domain_c_intensity_distribution.png'}")

    # 통계 출력
    print("\n=== Domain C Intensity Statistics ===")
    for stats in [train_good_stats, test_good_stats, test_fault_stats]:
        print(f"\n{stats['label']}:")
        print(f"  Mean: {np.mean(stats['means']):.4f} ± {np.std(stats['means']):.4f}")
        print(f"  Std:  {np.mean(stats['stds']):.4f} ± {np.std(stats['stds']):.4f}")
        print(f"  Min:  {np.mean(stats['mins']):.4f} ± {np.std(stats['mins']):.4f}")
        print(f"  Max:  {np.mean(stats['maxs']):.4f} ± {np.std(stats['maxs']):.4f}")

def compare_with_other_domains():
    """다른 도메인과 Domain C 비교"""

    domains = ['domain_A', 'domain_B', 'domain_C', 'domain_D']
    base_path = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_tiff_minmax")

    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    fig.suptitle('All Domains Comparison: Good (left 3) vs Fault (right 3)', fontsize=14, fontweight='bold')

    random.seed(42)

    for row_idx, domain in enumerate(domains):
        good_dir = base_path / domain / "test" / "good"
        fault_dir = base_path / domain / "test" / "fault"

        good_files = list(good_dir.glob("*.tiff"))
        fault_files = list(fault_dir.glob("*.tiff"))

        good_samples = random.sample(good_files, min(3, len(good_files)))
        fault_samples = random.sample(fault_files, min(3, len(fault_files)))

        # Good samples
        for col_idx, img_path in enumerate(good_samples):
            img = load_tiff_image(img_path)
            ax = axes[row_idx, col_idx]
            ax.imshow(img, cmap='viridis')
            ax.axis('off')
            if col_idx == 0:
                ax.set_ylabel(domain, fontsize=12, fontweight='bold')
            if row_idx == 0:
                ax.set_title('Good', fontsize=10)

        # Fault samples
        for col_idx, img_path in enumerate(fault_samples):
            img = load_tiff_image(img_path)
            ax = axes[row_idx, col_idx + 3]
            ax.imshow(img, cmap='viridis')
            ax.axis('off')
            if row_idx == 0:
                ax.set_title('Fault', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'all_domains_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'all_domains_comparison.png'}")

if __name__ == "__main__":
    print("=" * 60)
    print("Domain C Visual Analysis")
    print("=" * 60)

    visualize_samples(n_samples=6)
    analyze_intensity_distribution()
    compare_with_other_domains()

    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)
