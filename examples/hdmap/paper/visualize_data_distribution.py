#!/usr/bin/env python3
"""
HDMAP 데이터셋의 Scaling 전/후 픽셀 값 분포 시각화

이 스크립트는 다음을 생성합니다:
1. Figure 1: Unscaled range - 원본 TIFF 이미지의 픽셀 값 분포
2. Figure 2: Scaled range - MinMax scaling 후의 픽셀 값 분포

실행 방법:
    python examples/hdmap/paper/visualize_data_distribution.py

출력:
    - examples/hdmap/paper/figure1_unscaled_distribution.png
    - examples/hdmap/paper/figure2_scaled_distribution.png
    - examples/hdmap/paper/distribution_statistics.txt
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import scipy.io
from tqdm import tqdm

# 프로젝트 루트 경로
# __file__ = .../anomalib/examples/hdmap/paper/visualize_data_distribution.py
# parents[0] = .../anomalib/examples/hdmap/paper
# parents[1] = .../anomalib/examples/hdmap
# parents[2] = .../anomalib/examples
# parents[3] = .../anomalib (프로젝트 루트)
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

# Matplotlib 폰트 설정 - Computer Modern (LaTeX 기본 폰트)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['cmr10', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern for math text
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['font.size'] = 12

# =============================================================================
# 설정
# =============================================================================
# 데이터셋 선택 (1000 또는 100000)
N_SAMPLES = 100000  # 1000 또는 100000

# 샘플링 설정 (빠른 테스트를 위해 일부 샘플만 사용)
MAX_SAMPLES_PER_CATEGORY = None  # None이면 전체 사용, 숫자 지정시 해당 개수만 샘플링 (테스트: 100)

# 데이터셋 경로
DATASET_ROOT = project_root / "datasets" / "HDMAP"
SCALED_DATASET = DATASET_ROOT / f"{N_SAMPLES}_tiff_minmax"

# 원본 mat 파일 경로 (unscaled 통계 계산용)
RAW_DATA_PATH = project_root / "datasets" / "raw" / "KRISS_share_nipa2023"

# 출력 경로
OUTPUT_DIR = project_root / "examples" / "hdmap" / "paper"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# DOMAIN_CONFIG (prepare_hdmap_dataset.py와 동일)
# IEEE/Nature journal style colors
DOMAIN_CONFIG = {
    'A': {
        'user_min': 0.0,
        'user_max': 0.324670,
        'color': '#1f77b4'  # Blue (IEEE style)
    },
    'B': {
        'user_min': 0.0,
        'user_max': 1.324418,
        'color': '#d62728'  # Red (IEEE style)
    },
    'C': {
        'user_min': 0.0,
        'user_max': 0.087341,
        'color': '#2ca02c'  # Green (IEEE style)
    },
    'D': {
        'user_min': 0.0,
        'user_max': 0.418999,
        'color': '#ff7f0e'  # Orange (IEEE style)
    },
}

# 카테고리 정의
CATEGORIES = [
    ('train', 'good', 'Train (Normal)'),
    ('test', 'good', 'Test (Normal)'),
    ('test', 'fault', 'Test (Fault)')
]

# =============================================================================
# 데이터 로드 및 통계 계산
# =============================================================================
def generate_mat_paths():
    """도메인 구성 정보로부터 mat 파일 경로 생성"""
    paths = {}

    for domain, config in DOMAIN_CONFIG.items():
        sensor_path = config.get('sensor', DOMAIN_CONFIG[domain].get('sensor'))
        data_type = config.get('data_type', DOMAIN_CONFIG[domain].get('data_type'))

        # DOMAIN_CONFIG에서 sensor, data_type 정보 가져오기
        # prepare_hdmap_dataset.py에서 사용한 것과 동일한 구조
        if domain == 'A':
            sensor_path = 'Class1/1'
            data_type = '3_TSA_DIF'
        elif domain == 'B':
            sensor_path = 'Class1/1'
            data_type = '1_TSA_DIF'
        elif domain == 'C':
            sensor_path = 'Class3/1'
            data_type = '3_TSA_DIF'
        elif domain == 'D':
            sensor_path = 'Class3/1'
            data_type = '1_TSA_DIF'

        # 기본 경로 구성
        normal_base = RAW_DATA_PATH / "Normal" / "Normal2_LSSm0.3_HSS0" / sensor_path / "HDMap_train_test"
        fault_base = RAW_DATA_PATH / "Planet_fault_ring" / "1.42_LSSm0.3_HSS0" / sensor_path / "HDMap_train_test"

        paths[domain] = {
            'train_good': normal_base / f"{data_type}_train.mat",
            'test_good': normal_base / f"{data_type}_test.mat",
            'test_fault': fault_base / f"{data_type}_test.mat"
        }

    return paths

def load_mat_statistics(domain, split, label, max_samples=None):
    """
    원본 mat 파일에서 직접 통계 계산 (unscaled)

    Args:
        domain: 'A', 'B', 'C', 'D'
        split: 'train' 또는 'test'
        label: 'good' 또는 'fault'
        max_samples: 최대 샘플 수 (None이면 전체)

    Returns:
        dict: {'min', 'mean', 'max', 'std', 'median', 'q25', 'q75', 'all_values'}
    """
    mat_paths = generate_mat_paths()
    category_key = f"{split}_{label}"
    mat_path = mat_paths[domain][category_key]

    if not mat_path.exists():
        print(f"  ⚠️ Warning: {mat_path} does not exist")
        return None

    try:
        # mat 파일 로드
        print(f"  Loading {mat_path.name}...")
        mat_data = scipy.io.loadmat(str(mat_path))
        image_data = mat_data['Xdata']

        # 샘플 수 결정
        actual_samples = image_data.shape[3] if len(image_data.shape) > 3 else 1
        num_samples = min(max_samples, actual_samples) if max_samples else actual_samples

        # 모든 픽셀 값 수집
        all_pixels = []
        for i in tqdm(range(num_samples), desc=f"  Processing {domain}-{split}-{label}", leave=False):
            img = image_data[:, :, 0, i]
            all_pixels.append(img.flatten())

        all_pixels = np.concatenate(all_pixels)

        # 통계 계산
        stats = {
            'min': float(np.min(all_pixels)),
            'mean': float(np.mean(all_pixels)),
            'max': float(np.max(all_pixels)),
            'std': float(np.std(all_pixels)),
            'median': float(np.median(all_pixels)),
            'q25': float(np.percentile(all_pixels, 25)),
            'q75': float(np.percentile(all_pixels, 75)),
            'all_values': all_pixels,  # Violin plot용
            'n_images': num_samples,
            'n_pixels': len(all_pixels)
        }

        return stats

    except Exception as e:
        print(f"  ❌ Error loading {mat_path}: {e}")
        return None

def load_tiff_statistics(domain, split, label, dataset_path, max_images=None):
    """
    특정 도메인/카테고리의 TIFF 이미지들을 로드하고 통계 계산

    Args:
        domain: 'A', 'B', 'C', 'D'
        split: 'train' 또는 'test'
        label: 'good' 또는 'fault'
        dataset_path: 데이터셋 루트 경로
        max_images: 최대 로드 이미지 수 (None이면 전체)

    Returns:
        dict: {'min', 'mean', 'max', 'std', 'median', 'q25', 'q75', 'all_values'}
    """
    data_path = dataset_path / f"domain_{domain}" / split / label

    if not data_path.exists():
        print(f"  ⚠️ Warning: {data_path} does not exist")
        return None

    # TIFF 파일 목록
    tiff_files = sorted(data_path.glob("*.tiff")) + sorted(data_path.glob("*.tif"))

    if not tiff_files:
        print(f"  ⚠️ Warning: No TIFF files found in {data_path}")
        return None

    # 최대 이미지 수 제한
    if max_images is not None:
        tiff_files = tiff_files[:max_images]

    # 모든 픽셀 값 수집
    all_pixels = []
    for tiff_file in tqdm(tiff_files, desc=f"  Loading {domain}-{split}-{label}", leave=False):
        try:
            img = tifffile.imread(tiff_file)
            all_pixels.append(img.flatten())
        except Exception as e:
            print(f"    Error loading {tiff_file}: {e}")
            continue

    if not all_pixels:
        return None

    # 통계 계산
    all_pixels = np.concatenate(all_pixels)

    stats = {
        'min': float(np.min(all_pixels)),
        'mean': float(np.mean(all_pixels)),
        'max': float(np.max(all_pixels)),
        'std': float(np.std(all_pixels)),
        'median': float(np.median(all_pixels)),
        'q25': float(np.percentile(all_pixels, 25)),
        'q75': float(np.percentile(all_pixels, 75)),
        'all_values': all_pixels,  # Violin plot용
        'n_images': len(tiff_files),
        'n_pixels': len(all_pixels)
    }

    return stats

def collect_unscaled_statistics(domains=['A', 'B', 'C', 'D'], max_samples=None):
    """
    원본 mat 파일에서 unscaled 통계 수집

    Returns:
        dict: {domain: {category_key: stats_dict}}
    """
    print(f"📊 Unscaled 통계 수집 중 (원본 mat 파일)")

    all_stats = {}

    for domain in domains:
        print(f"\n🔹 Domain {domain}")
        domain_stats = {}

        for split, label, name in CATEGORIES:
            category_key = f"{split}_{label}"
            stats = load_mat_statistics(domain, split, label, max_samples)

            if stats:
                domain_stats[category_key] = stats
                print(f"  ✅ {name}: {stats['n_images']} images, "
                      f"range=[{stats['min']:.6f}, {stats['max']:.6f}], "
                      f"mean={stats['mean']:.6f}")
            else:
                domain_stats[category_key] = None

        all_stats[domain] = domain_stats

    return all_stats

def collect_scaled_statistics(dataset_path, domains=['A', 'B', 'C', 'D'], max_images=None):
    """
    TIFF 파일에서 scaled 통계 수집

    Returns:
        dict: {domain: {category_key: stats_dict}}
    """
    print(f"📊 Scaled 통계 수집 중: {dataset_path.name}")

    all_stats = {}

    for domain in domains:
        print(f"\n🔹 Domain {domain}")
        domain_stats = {}

        for split, label, name in CATEGORIES:
            category_key = f"{split}_{label}"
            stats = load_tiff_statistics(domain, split, label, dataset_path, max_images)

            if stats:
                domain_stats[category_key] = stats
                print(f"  ✅ {name}: {stats['n_images']} images, "
                      f"range=[{stats['min']:.6f}, {stats['max']:.6f}], "
                      f"mean={stats['mean']:.6f}")
            else:
                domain_stats[category_key] = None

        all_stats[domain] = domain_stats

    return all_stats

# =============================================================================
# 시각화 함수
# =============================================================================
def plot_distribution_bars(stats_data, title, xlabel, output_path, include_violin=False):
    """
    Horizontal bar chart로 분포 시각화 (참고 이미지와 동일한 스타일)

    Args:
        stats_data: {domain: {category: stats}} 구조
        title: 그래프 제목
        xlabel: x축 레이블
        output_path: 저장 경로
        include_violin: Violin plot 포함 여부
    """
    # 12개 카테고리 생성 (역순으로 배치 - 그림에서 A가 위에 오도록)
    categories = []
    colors = []
    mins = []
    maxs = []
    means = []

    domains = ['D', 'C', 'B', 'A']  # 역순

    for domain in domains:
        for split, label, name in reversed(CATEGORIES):  # 역순
            category_key = f"{split}_{label}"
            category_label = f"{domain}-{name}"

            stats = stats_data.get(domain, {}).get(category_key)

            if stats:
                categories.append(category_label)
                colors.append(DOMAIN_CONFIG[domain]['color'])
                mins.append(stats['min'])
                maxs.append(stats['max'])
                means.append(stats['mean'])
            else:
                # 데이터가 없으면 0으로 채움
                categories.append(category_label)
                colors.append(DOMAIN_CONFIG[domain]['color'])
                mins.append(0)
                maxs.append(0)
                means.append(0)

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(14, 10))

    y_pos = np.arange(len(categories))

    # Horizontal bar (min ~ max 범위)
    for i, (cat, color, min_val, max_val, mean_val) in enumerate(zip(categories, colors, mins, maxs, means)):
        # Bar 그리기 (min에서 max까지)
        width = max_val - min_val
        ax.barh(i, width, left=min_val, height=0.6, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Mean marker (원)
        ax.plot(mean_val, i, 'o', color='black', markersize=8, markeredgecolor='black', markeredgewidth=1.5, zorder=3)

    # Y축 설정
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=12)
    ax.set_ylim(-0.5, len(categories) - 0.5)

    # X축 설정
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # 제목
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # 범례 (도메인별)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=DOMAIN_CONFIG[d]['color'], edgecolor='black', label=f'Domain {d}')
                      for d in ['A', 'B', 'C', 'D']]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_path}")

def plot_violin_distribution(stats_data, title, xlabel, output_path):
    """
    Violin plot으로 상세 분포 시각화 (선택적)

    Args:
        stats_data: {domain: {category: stats}} 구조
        title: 그래프 제목
        xlabel: x축 레이블
        output_path: 저장 경로
    """
    # 카테고리 및 데이터 준비
    categories = []
    all_data = []
    colors = []
    original_stats = []  # 원본 min/max 저장

    domains = ['D', 'C', 'B', 'A']  # 역순

    # Set random seed for reproducibility
    np.random.seed(42)

    for domain in domains:
        for split, label, name in reversed(CATEGORIES):
            category_key = f"{split}_{label}"
            category_label = f"{domain}-{name}"

            stats = stats_data.get(domain, {}).get(category_key)

            if stats and 'all_values' in stats and len(stats['all_values']) > 0:
                # 샘플링 (너무 많은 데이터 포인트 방지) - deterministic sampling
                n_samples = min(50000, len(stats['all_values']))  # Increased from 10000 to 50000
                sampled_values = np.random.choice(stats['all_values'],
                                                  n_samples,
                                                  replace=False)
                categories.append(category_label)
                all_data.append(sampled_values)
                colors.append(DOMAIN_CONFIG[domain]['color'])

                # 원본 통계 저장 (샘플링과 관계없이 전체 데이터의 min/max)
                original_stats.append({
                    'min': stats['min'],
                    'max': stats['max']
                })

                # Debug: Print statistics for train data
                if 'train' in split and 'good' in label:
                    print(f"  Debug {category_label}: sampled_min={np.min(sampled_values):.4f}, "
                          f"sampled_max={np.max(sampled_values):.4f}, "
                          f"original_min={stats['min']:.4f}, original_max={stats['max']:.4f}")

    # 데이터가 없으면 그래프 생성 건너뛰기
    if not all_data:
        print(f"  ⚠️ Warning: No data available for violin plot. Skipping {output_path.name}")
        return

    # Violin plot 생성
    fig, ax = plt.subplots(figsize=(14, 10))

    positions = np.arange(len(categories))
    parts = ax.violinplot(all_data, positions=positions, vert=False, widths=0.7,
                          showmeans=True, showmedians=False, showextrema=False)

    # Violin 색상 설정
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.5)

    # Mean 라인을 검은색으로 변경
    if 'cmeans' in parts:
        parts['cmeans'].set_edgecolor('black')
        parts['cmeans'].set_linewidth(1.5)
    if 'cbars' in parts:
        parts['cbars'].set_edgecolor('black')
        parts['cbars'].set_linewidth(1.5)
    if 'cmaxes' in parts:
        parts['cmaxes'].set_edgecolor('black')
        parts['cmaxes'].set_linewidth(1.5)
    if 'cmins' in parts:
        parts['cmins'].set_edgecolor('black')
        parts['cmins'].set_linewidth(1.5)

    # Min/Max 표시 추가 (원본 통계 사용)
    for i, (data, pos) in enumerate(zip(all_data, positions)):
        # 원본 데이터셋의 실제 통계값 사용 (샘플링된 데이터가 아닌)
        min_val = original_stats[i]['min']
        max_val = original_stats[i]['max']

        # Whiskers (min-max) - 얇은 수평선
        ax.plot([min_val, max_val], [pos, pos], color='black', linewidth=1.0, alpha=0.8, zorder=3)

        # Min/Max markers - 작은 세로선
        ax.plot([min_val, min_val], [pos-0.05, pos+0.05], color='black', linewidth=1.5, alpha=0.8, zorder=3)
        ax.plot([max_val, max_val], [pos-0.05, pos+0.05], color='black', linewidth=1.5, alpha=0.8, zorder=3)

    # Y축 설정 (폰트 150%)
    ax.set_yticks(positions)
    ax.set_yticklabels(categories, fontsize=18)  # 12 * 1.5 = 18
    ax.set_ylim(-0.5, len(categories) - 0.5)
    ax.tick_params(axis='x', labelsize=18)  # X축 눈금 폰트도 150%

    # X축 설정 (폰트 150%)
    ax.set_xlabel(xlabel, fontsize=21, fontweight='bold')  # 14 * 1.5 = 21
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5, color='gray')

    # 제목 (폰트 150%)
    ax.set_title(title, fontsize=24, fontweight='bold', pad=20)  # 16 * 1.5 = 24

    # 범례 (폰트 150%)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=DOMAIN_CONFIG[d]['color'], edgecolor='black', label=f'Domain {d}')
                      for d in ['A', 'B', 'C', 'D']]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=18, framealpha=0.9)  # 12 * 1.5 = 18

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_path}")

# =============================================================================
# 메인 실행
# =============================================================================
def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("📊 HDMAP 데이터셋 분포 시각화")
    print("=" * 80)
    print(f"데이터셋: {N_SAMPLES}개 샘플")
    print(f"Unscaled: 원본 mat 파일에서 로드")
    print(f"Scaled: {SCALED_DATASET}")
    print("=" * 80)

    # 1. Unscaled 통계 수집 (원본 mat 파일에서)
    print("\n" + "=" * 80)
    print("📈 Figure 1: Unscaled Distribution (from mat files)")
    print("=" * 80)
    unscaled_stats = collect_unscaled_statistics(max_samples=MAX_SAMPLES_PER_CATEGORY)

    # 2. Scaled 통계 수집 (TIFF 파일에서)
    print("\n" + "=" * 80)
    print("📈 Figure 2: Scaled Distribution (from TIFF files)")
    print("=" * 80)
    scaled_stats = collect_scaled_statistics(SCALED_DATASET, max_images=MAX_SAMPLES_PER_CATEGORY)

    # 3. Figure 1 생성 (Unscaled)
    print("\n" + "=" * 80)
    print("🎨 Generating Figure 1: Unscaled Range")
    print("=" * 80)
    plot_distribution_bars(
        unscaled_stats,
        title="Unscaled range",
        xlabel="Value",
        output_path=OUTPUT_DIR / "figure1_unscaled_distribution.png"
    )

    # 4. Figure 2 생성 (Scaled)
    print("\n" + "=" * 80)
    print("🎨 Generating Figure 2: Scaled Range")
    print("=" * 80)
    plot_distribution_bars(
        scaled_stats,
        title="Scaled range",
        xlabel="Scaled Value",
        output_path=OUTPUT_DIR / "figure2_scaled_distribution.png"
    )

    # 5. (선택적) Violin plot 생성
    print("\n" + "=" * 80)
    print("🎨 Generating Violin Plots (Optional)")
    print("=" * 80)

    print("Generating unscaled violin plot...")
    plot_violin_distribution(
        unscaled_stats,
        title="Unscaled Data Distribution",
        xlabel="Value",
        output_path=OUTPUT_DIR / "figure1_unscaled_violin.png"
    )

    print("Generating scaled violin plot...")
    plot_violin_distribution(
        scaled_stats,
        title="Scaled Data Distribution",
        xlabel="Scaled Value",
        output_path=OUTPUT_DIR / "figure2_scaled_violin.png"
    )

    # 6. 통계 리포트 저장
    print("\n" + "=" * 80)
    print("📄 Saving Statistics Report")
    print("=" * 80)

    report_path = OUTPUT_DIR / "distribution_statistics.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("HDMAP 데이터셋 분포 통계 리포트\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"데이터셋: {N_SAMPLES}개 샘플\n")
        f.write(f"Unscaled: 원본 mat 파일에서 로드\n")
        f.write(f"Scaled: {SCALED_DATASET}\n\n")

        f.write("=" * 80 + "\n")
        f.write("Unscaled Statistics\n")
        f.write("=" * 80 + "\n\n")

        for domain in ['A', 'B', 'C', 'D']:
            f.write(f"\n도메인 {domain} (user_min={DOMAIN_CONFIG[domain]['user_min']}, "
                   f"user_max={DOMAIN_CONFIG[domain]['user_max']})\n")
            f.write("-" * 80 + "\n")

            for split, label, name in CATEGORIES:
                category_key = f"{split}_{label}"
                stats = unscaled_stats.get(domain, {}).get(category_key)

                if stats:
                    f.write(f"  {name}:\n")
                    f.write(f"    이미지 수: {stats['n_images']}\n")
                    f.write(f"    픽셀 수: {stats['n_pixels']:,}\n")
                    f.write(f"    Min: {stats['min']:.6f}\n")
                    f.write(f"    Max: {stats['max']:.6f}\n")
                    f.write(f"    Mean: {stats['mean']:.6f}\n")
                    f.write(f"    Std: {stats['std']:.6f}\n")
                    f.write(f"    Median: {stats['median']:.6f}\n")
                    f.write(f"    Q25: {stats['q25']:.6f}\n")
                    f.write(f"    Q75: {stats['q75']:.6f}\n\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Scaled Statistics\n")
        f.write("=" * 80 + "\n\n")

        for domain in ['A', 'B', 'C', 'D']:
            f.write(f"\n도메인 {domain}\n")
            f.write("-" * 80 + "\n")

            for split, label, name in CATEGORIES:
                category_key = f"{split}_{label}"
                stats = scaled_stats.get(domain, {}).get(category_key)

                if stats:
                    f.write(f"  {name}:\n")
                    f.write(f"    이미지 수: {stats['n_images']}\n")
                    f.write(f"    픽셀 수: {stats['n_pixels']:,}\n")
                    f.write(f"    Min: {stats['min']:.6f}\n")
                    f.write(f"    Max: {stats['max']:.6f}\n")
                    f.write(f"    Mean: {stats['mean']:.6f}\n")
                    f.write(f"    Std: {stats['std']:.6f}\n")
                    f.write(f"    Median: {stats['median']:.6f}\n")
                    f.write(f"    Q25: {stats['q25']:.6f}\n")
                    f.write(f"    Q75: {stats['q75']:.6f}\n\n")

    print(f"✅ Statistics report saved: {report_path}")

    # 완료 메시지
    print("\n" + "=" * 80)
    print("✅ 모든 시각화 완료!")
    print("=" * 80)
    print(f"\n생성된 파일:")
    print(f"  - {OUTPUT_DIR / 'figure1_unscaled_distribution.png'}")
    print(f"  - {OUTPUT_DIR / 'figure2_scaled_distribution.png'}")
    print(f"  - {OUTPUT_DIR / 'figure1_unscaled_violin.png'} (선택적)")
    print(f"  - {OUTPUT_DIR / 'figure2_scaled_violin.png'} (선택적)")
    print(f"  - {OUTPUT_DIR / 'distribution_statistics.txt'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
