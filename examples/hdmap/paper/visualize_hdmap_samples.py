"""
HDMAP 샘플 시각화 스크립트 (논문용)
- 정상/고장 샘플 각 1개 시각화
- 원본 크기(95x31) 유지
- X축: Planet gear tooth number, Y축: Ring gear tooth number
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ============================================================
# 하드코딩 설정 (필요시 수정)
# ============================================================
DOMAIN = "domain_B"  # domain_A, domain_B, domain_C, domain_D
NORMAL_SAMPLE_IDX = 100  # 정상 샘플 번호 (0-499: cold, 500-999: warm)
FAULT_SAMPLE_IDX = 1400  # 고장 샘플 번호 (1000-1499: cold, 1500-1999: warm)

# 데이터 경로
DATA_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_tiff_minmax")

# 출력 경로
OUTPUT_DIR = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/examples/hdmap/paper/samples")
# ============================================================


def load_hdmap_image(domain: str, sample_idx: int) -> np.ndarray:
    """HDMAP 이미지 로드 (0~1 범위를 0~255로 변환)"""
    # 샘플 번호에 따라 폴더 결정
    # 0-999: good, 1000-1999: fault
    if sample_idx < 1000:
        folder = "good"
        file_idx = sample_idx
    else:
        folder = "fault"
        file_idx = sample_idx - 1000

    image_path = DATA_ROOT / domain / "test" / folder / f"{file_idx:06d}.tiff"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path)
    img_array = np.array(img, dtype=np.float32)

    # 0~1 범위를 0~255로 변환 (clipping으로 1 초과값 처리)
    img_array = np.clip(img_array, 0, 1) * 255
    return img_array.astype(np.uint8)


def visualize_samples(
    domain: str,
    normal_idx: int,
    fault_idx: int,
    output_path: Path = None,
    show: bool = True
):
    """정상/고장 샘플 시각화"""

    # Times New Roman 폰트 설정 (없으면 serif로 대체)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']

    # 이미지 로드
    normal_img = load_hdmap_image(domain, normal_idx)
    fault_img = load_hdmap_image(domain, fault_idx)

    print(f"Normal sample shape: {normal_img.shape}")
    print(f"Fault sample shape: {fault_img.shape}")

    # 이미지 크기 (95 x 31)
    H, W = normal_img.shape[:2]

    # Figure 생성 (2행 1열, 더 작은 사이즈)
    fig, axes = plt.subplots(2, 1, figsize=(3, 3))

    # X축: 위치(0-based)와 라벨(1-based) 분리
    x_tick_pos = [0, 19, 39, 59, 79, 94]  # 0-based 위치
    x_tick_labels = [1, 20, 40, 60, 80, 95]  # 1-based 라벨
    # Y축: 위치(0-based)와 라벨(1-based) 분리
    y_tick_pos = [0, 14, 30]  # 0-based 위치
    y_tick_labels = [1, 15, 31]  # 1-based 라벨

    # 정상 샘플
    ax1 = axes[0]
    im1 = ax1.imshow(normal_img, cmap='gray', aspect='equal')
    ax1.set_xticks(x_tick_pos)
    ax1.set_xticklabels(x_tick_labels, fontsize=7)
    ax1.set_yticks(y_tick_pos)
    ax1.set_yticklabels(y_tick_labels, fontsize=7)
    ax1.tick_params(axis='both', length=2, color='gray', labelcolor='black')

    # 고장 샘플
    ax2 = axes[1]
    im2 = ax2.imshow(fault_img, cmap='gray', aspect='equal')
    ax2.set_xticks(x_tick_pos)
    ax2.set_xticklabels(x_tick_labels, fontsize=7)
    ax2.set_yticks(y_tick_pos)
    ax2.set_yticklabels(y_tick_labels, fontsize=7)
    ax2.tick_params(axis='both', length=2, color='gray', labelcolor='black')

    plt.subplots_adjust(hspace=0.02)  # 두 subplot 사이 간격 최소화

    # 저장
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    print(f"Domain: {DOMAIN}")
    print(f"Normal sample: {NORMAL_SAMPLE_IDX}")
    print(f"Fault sample: {FAULT_SAMPLE_IDX}")
    print("-" * 50)

    output_path = OUTPUT_DIR / f"hdmap_samples_{DOMAIN}.png"

    visualize_samples(
        domain=DOMAIN,
        normal_idx=NORMAL_SAMPLE_IDX,
        fault_idx=FAULT_SAMPLE_IDX,
        output_path=output_path,
        show=False
    )


if __name__ == "__main__":
    main()
