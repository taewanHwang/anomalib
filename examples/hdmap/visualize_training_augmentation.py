#!/usr/bin/env python3
"""
실제 학습에 사용되는 CutPaste Augmentation 시각화 스크립트

exp_51-54_draem_cp.json의 exp-52 조건으로 실제 학습 데이터를 생성하고 저장합니다.

실행 방법:
cd /mnt/ex-disk/taewan.hwang/study/anomalib
python examples/hdmap/visualize_training_augmentation.py
"""

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

# Anomalib 모듈 import
from anomalib.models.image.draem_cutpaste_clf.synthetic_generator_v2 import CutPasteSyntheticGenerator

# =============================================================================
# 🔧 실험 설정 (exp-52.C 조건)
# =============================================================================
DOMAIN = 'C'  # A, B, C, D 중 선택 가능
DATASET_ROOT = "datasets/HDMAP/100000_tiff_minmax"
TARGET_SIZE = [128, 128]
CUT_W_RANGE = [2, 127]
CUT_H_RANGE = [4, 8]
A_FAULT_START = 0.0
A_FAULT_RANGE_END = 0.3
AUGMENT_PROBABILITY = 0.5
SEED = 52
NUM_SAMPLES = 20  # 생성할 샘플 수

# 출력 디렉토리
OUTPUT_DIR = project_root / "examples" / "hdmap" / "single_domain" / "visualizations" / "training_augmentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_sample_images(domain, num_samples=20):
    """데이터셋에서 샘플 이미지 로드"""
    from PIL import Image

    data_path = project_root / DATASET_ROOT / f"domain_{domain}" / "train" / "good"

    # TIFF 파일 목록 가져오기
    tiff_files = list(data_path.glob("*.tiff"))
    if not tiff_files:
        tiff_files = list(data_path.glob("*.tif"))

    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files found in {data_path}")

    # 랜덤 샘플링
    random.shuffle(tiff_files)
    selected_files = tiff_files[:num_samples]

    images = []
    filenames = []
    for file_path in selected_files:
        img = Image.open(file_path)
        img_array = np.array(img).astype(np.float32)

        # Resize to target size
        img_pil = Image.fromarray(img_array)
        img_resized = img_pil.resize((TARGET_SIZE[1], TARGET_SIZE[0]), Image.BILINEAR)
        img_array_resized = np.array(img_resized).astype(np.float32)

        # (H, W) -> (1, H, W) 변환
        img_tensor = torch.from_numpy(img_array_resized).unsqueeze(0)

        # (1, H, W) -> (3, H, W) - RGB 형태로 변환 (CutPaste generator가 3채널 입력 기대)
        img_tensor_rgb = img_tensor.repeat(3, 1, 1)

        images.append(img_tensor_rgb)
        filenames.append(file_path.stem)  # 확장자 제외한 파일명

    print(f"✅ 로드된 이미지: {len(images)}개")
    print(f"   - 첫 번째 이미지 shape: {images[0].shape}")
    print(f"   - 값 범위: [{images[0].min():.4f}, {images[0].max():.4f}]")

    return images, filenames

def generate_augmented_samples(images, filenames):
    """CutPaste Augmentation 적용"""

    # CutPasteSyntheticGenerator 초기화
    generator = CutPasteSyntheticGenerator(
        cut_w_range=tuple(CUT_W_RANGE),
        cut_h_range=tuple(CUT_H_RANGE),
        a_fault_start=A_FAULT_START,
        a_fault_range_end=A_FAULT_RANGE_END,
        probability=AUGMENT_PROBABILITY,
        validation_enabled=False  # Training mode
    )

    print(f"🔧 CutPaste Generator 설정:")
    print(f"   - cut_w_range: {CUT_W_RANGE}")
    print(f"   - cut_h_range: {CUT_H_RANGE}")
    print(f"   - a_fault_range: [{A_FAULT_START}, {A_FAULT_RANGE_END}]")
    print(f"   - probability: {AUGMENT_PROBABILITY}")

    samples = []
    for i, (img_tensor, filename) in enumerate(zip(images, filenames)):
        # 배치 형태로 변환: (3, H, W) -> (1, 3, H, W)
        img_batch = img_tensor.unsqueeze(0)

        # CutPaste augmentation 적용
        augmented_batch, anomaly_mask, anomaly_label = generator(img_batch)

        # 결과 저장
        samples.append({
            'image': img_tensor.cpu().numpy(),  # (3, H, W)
            'augmented': augmented_batch[0].cpu().numpy(),  # (3, H, W)
            'mask': anomaly_mask[0].cpu().numpy(),  # (1, H, W)
            'has_anomaly': anomaly_label[0].item(),
            'filename': filename  # 원본 파일명 추가
        })

    print(f"✅ Augmentation 완료: {len(samples)}개")

    # 통계 출력
    anomaly_count = sum(1 for s in samples if s['has_anomaly'])
    normal_count = len(samples) - anomaly_count
    print(f"   - Anomaly 샘플: {anomaly_count}개")
    print(f"   - Normal 샘플: {normal_count}개")

    return samples

def save_individual_images(samples, output_dir):
    """각 샘플을 개별 이미지 파일로 저장"""
    from PIL import Image

    # Anomaly와 Normal 분리
    anomaly_samples = [s for s in samples if s['has_anomaly']]
    normal_samples = [s for s in samples if not s['has_anomaly']]

    # Anomaly 이미지 저장
    anomaly_dir = output_dir / "anomaly_samples"
    anomaly_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(anomaly_samples):
        filename = sample['filename']

        # Original
        orig_img = sample['image'][0]  # 첫 번째 채널
        orig_img_normalized = ((orig_img - orig_img.min()) / (orig_img.max() - orig_img.min()) * 255).astype(np.uint8)
        Image.fromarray(orig_img_normalized).save(anomaly_dir / f"anomaly_{i+1:02d}_{filename}_original.png")

        # Augmented
        aug_img = sample['augmented'][0]  # 첫 번째 채널
        aug_img_normalized = ((aug_img - aug_img.min()) / (aug_img.max() - aug_img.min()) * 255).astype(np.uint8)
        Image.fromarray(aug_img_normalized).save(anomaly_dir / f"anomaly_{i+1:02d}_{filename}_augmented.png")

        # Mask
        mask_img = sample['mask'][0]
        mask_img_normalized = (mask_img * 255).astype(np.uint8)
        Image.fromarray(mask_img_normalized).save(anomaly_dir / f"anomaly_{i+1:02d}_{filename}_mask.png")

    print(f"✅ Anomaly 샘플 저장: {len(anomaly_samples)}개 → {anomaly_dir}")

    # Normal 이미지 저장
    normal_dir = output_dir / "normal_samples"
    normal_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(normal_samples):
        filename = sample['filename']

        # Original
        orig_img = sample['image'][0]
        orig_img_normalized = ((orig_img - orig_img.min()) / (orig_img.max() - orig_img.min()) * 255).astype(np.uint8)
        Image.fromarray(orig_img_normalized).save(normal_dir / f"normal_{i+1:02d}_{filename}_original.png")

        # Augmented (should be same as original)
        aug_img = sample['augmented'][0]
        aug_img_normalized = ((aug_img - aug_img.min()) / (aug_img.max() - aug_img.min()) * 255).astype(np.uint8)
        Image.fromarray(aug_img_normalized).save(normal_dir / f"normal_{i+1:02d}_{filename}_augmented.png")

    print(f"✅ Normal 샘플 저장: {len(normal_samples)}개 → {normal_dir}")

def plot_statistics(samples, save_path):
    """픽셀 값 분포 히스토그램"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Training Augmentation Statistics - Domain {DOMAIN}', fontsize=16, fontweight='bold')

    # 데이터 수집 (첫 번째 채널만 사용)
    all_original = []
    all_augmented = []
    anomaly_augmented = []
    normal_augmented = []

    for sample in samples:
        all_original.extend(sample['image'][0].flatten())
        all_augmented.extend(sample['augmented'][0].flatten())

        if sample['has_anomaly']:
            anomaly_augmented.extend(sample['augmented'][0].flatten())
        else:
            normal_augmented.extend(sample['augmented'][0].flatten())

    # Original 히스토그램
    axes[0, 0].hist(all_original, bins=50, alpha=0.7, color='blue', density=True)
    axes[0, 0].set_title('Original Images')
    axes[0, 0].set_xlabel('Pixel Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].grid(True, alpha=0.3)
    stats_text = f'Mean: {np.mean(all_original):.4f}\nStd: {np.std(all_original):.4f}'
    axes[0, 0].text(0.02, 0.98, stats_text, transform=axes[0, 0].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Augmented 전체 히스토그램
    axes[0, 1].hist(all_augmented, bins=50, alpha=0.7, color='red', density=True)
    axes[0, 1].set_title('Augmented Images (All)')
    axes[0, 1].set_xlabel('Pixel Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].grid(True, alpha=0.3)
    stats_text = f'Mean: {np.mean(all_augmented):.4f}\nStd: {np.std(all_augmented):.4f}'
    axes[0, 1].text(0.02, 0.98, stats_text, transform=axes[0, 1].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    # Anomaly augmented 히스토그램
    if anomaly_augmented:
        axes[1, 0].hist(anomaly_augmented, bins=50, alpha=0.7, color='orange', density=True)
        axes[1, 0].set_title('Augmented Images (Anomaly Only)')
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].grid(True, alpha=0.3)
        stats_text = f'Mean: {np.mean(anomaly_augmented):.4f}\nStd: {np.std(anomaly_augmented):.4f}\nCount: {len([s for s in samples if s["has_anomaly"]])}'
        axes[1, 0].text(0.02, 0.98, stats_text, transform=axes[1, 0].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Normal augmented 히스토그램
    if normal_augmented:
        axes[1, 1].hist(normal_augmented, bins=50, alpha=0.7, color='green', density=True)
        axes[1, 1].set_title('Augmented Images (Normal Only)')
        axes[1, 1].set_xlabel('Pixel Value')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].grid(True, alpha=0.3)
        stats_text = f'Mean: {np.mean(normal_augmented):.4f}\nStd: {np.std(normal_augmented):.4f}\nCount: {len([s for s in samples if not s["has_anomaly"]])}'
        axes[1, 1].text(0.02, 0.98, stats_text, transform=axes[1, 1].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 통계 그래프 저장: {save_path}")

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("🧪 실제 학습 데이터 Augmentation 시각화 (exp-52.C 조건)")
    print("=" * 80)
    print(f"🔧 설정:")
    print(f"   - DOMAIN: {DOMAIN}")
    print(f"   - DATASET_ROOT: {DATASET_ROOT}")
    print(f"   - TARGET_SIZE: {TARGET_SIZE}")
    print(f"   - CUT_W_RANGE: {CUT_W_RANGE}")
    print(f"   - CUT_H_RANGE: {CUT_H_RANGE}")
    print(f"   - A_FAULT_START: {A_FAULT_START}")
    print(f"   - A_FAULT_RANGE_END: {A_FAULT_RANGE_END}")
    print(f"   - AUGMENT_PROBABILITY: {AUGMENT_PROBABILITY}")
    print(f"   - SEED: {SEED}")
    print()

    # 시드 설정
    set_seed(SEED)

    # 샘플 이미지 로드
    print("📁 샘플 이미지 로드 중...")
    images, filenames = load_sample_images(DOMAIN, NUM_SAMPLES)

    # Augmentation 적용
    print("\n🎨 CutPaste Augmentation 적용 중...")
    samples = generate_augmented_samples(images, filenames)

    # 개별 이미지 저장
    print("\n💾 개별 이미지 저장 중...")
    save_individual_images(samples, OUTPUT_DIR / f"domain_{DOMAIN}")

    # 통계 시각화
    print("\n📊 통계 시각화 생성 중...")
    plot_statistics(samples, OUTPUT_DIR / f"domain_{DOMAIN}" / f"augmentation_statistics.png")

    # 요약 리포트 저장
    report_path = OUTPUT_DIR / f"domain_{DOMAIN}" / "augmentation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("실제 학습 데이터 Augmentation 리포트\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"실험 조건: exp-52.{DOMAIN}\n")
        f.write(f"도메인: {DOMAIN}\n")
        f.write(f"데이터셋: {DATASET_ROOT}\n")
        f.write(f"이미지 크기: {TARGET_SIZE}\n")
        f.write(f"패치 너비 범위: {CUT_W_RANGE}\n")
        f.write(f"패치 높이 범위: {CUT_H_RANGE}\n")
        f.write(f"Fault 강도 범위: [{A_FAULT_START}, {A_FAULT_RANGE_END}]\n")
        f.write(f"Augmentation 확률: {AUGMENT_PROBABILITY}\n")
        f.write(f"시드: {SEED}\n\n")

        f.write(f"생성된 샘플 통계:\n")
        f.write(f"- 전체 샘플: {len(samples)}개\n")
        anomaly_count = sum(1 for s in samples if s['has_anomaly'])
        f.write(f"- Anomaly 샘플: {anomaly_count}개\n")
        f.write(f"- Normal 샘플: {len(samples) - anomaly_count}개\n\n")

        # 픽셀 값 통계 (첫 번째 채널만)
        all_original = np.concatenate([s['image'][0].flatten() for s in samples])
        all_augmented = np.concatenate([s['augmented'][0].flatten() for s in samples])

        f.write(f"픽셀 값 통계:\n")
        f.write(f"원본 이미지:\n")
        f.write(f"  - 평균: {np.mean(all_original):.6f}\n")
        f.write(f"  - 표준편차: {np.std(all_original):.6f}\n")
        f.write(f"  - 범위: [{np.min(all_original):.6f}, {np.max(all_original):.6f}]\n\n")

        f.write(f"Augmented 이미지:\n")
        f.write(f"  - 평균: {np.mean(all_augmented):.6f}\n")
        f.write(f"  - 표준편차: {np.std(all_augmented):.6f}\n")
        f.write(f"  - 범위: [{np.min(all_augmented):.6f}, {np.max(all_augmented):.6f}]\n")

    print(f"📄 리포트 저장: {report_path}")
    print()
    print("✅ 완료!")
    print(f"📁 결과 저장 위치: {OUTPUT_DIR / f'domain_{DOMAIN}'}")
    print(f"   - anomaly_samples/: Anomaly 이미지 (original, augmented, mask)")
    print(f"   - normal_samples/: Normal 이미지 (original, augmented)")
    print(f"   - augmentation_statistics.png: 통계 그래프")
    print(f"   - augmentation_report.txt: 상세 리포트")
    print("=" * 80)

if __name__ == "__main__":
    main()
