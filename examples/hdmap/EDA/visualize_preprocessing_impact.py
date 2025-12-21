"""
Domain C: Dinomaly 전처리가 이미지에 미치는 영향 시각화
- 원본 31x95 → 448x448 resize → 392x392 crop
- Aspect ratio 변화와 defect 신호에 미치는 영향 분석
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# 경로 설정
TIFF_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_tiff_minmax/domain_C")
OUTPUT_DIR = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/examples/hdmap/EDA/results/preprocessing_impact")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_tiff_image(path: Path) -> np.ndarray:
    """TIFF 이미지 로드"""
    img = Image.open(path)
    return np.array(img, dtype=np.float32)

def dinomaly_preprocess(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dinomaly 전처리 과정 시뮬레이션
    Returns: (original, resized_448, cropped_392)
    """
    # numpy to tensor (C, H, W)
    if len(img.shape) == 2:
        img_tensor = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
    else:
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0).float()  # (1, 3, H, W)

    # Step 1: Resize to 448x448
    resized_448 = F.interpolate(img_tensor, size=(448, 448), mode='bilinear', align_corners=False)

    # Step 2: Center crop to 392x392
    crop_transform = transforms.CenterCrop(392)
    cropped_392 = crop_transform(resized_448)

    # Convert back to numpy for visualization
    return (
        img,  # Original
        resized_448.squeeze(0)[0].numpy(),  # Resized (take first channel)
        cropped_392.squeeze(0)[0].numpy()   # Cropped
    )

def calculate_patch_representation(img: np.ndarray, patch_size: int = 14) -> np.ndarray:
    """패치별 평균값으로 패치 표현 계산"""
    h, w = img.shape
    patches_h = h // patch_size
    patches_w = w // patch_size

    # Trim to fit patches
    img_trimmed = img[:patches_h * patch_size, :patches_w * patch_size]

    # Reshape to patches
    img_reshaped = img_trimmed.reshape(patches_h, patch_size, patches_w, patch_size)

    # Mean over each patch
    patch_means = img_reshaped.mean(axis=(1, 3))

    return patch_means

def visualize_preprocessing(n_samples: int = 4):
    """전처리 과정 시각화"""

    # 정상/고장 샘플 로드
    good_files = sorted(list((TIFF_ROOT / "test" / "good").glob("*.tiff")))[:n_samples]
    fault_files = sorted(list((TIFF_ROOT / "test" / "fault").glob("*.tiff")))[:n_samples]

    fig, axes = plt.subplots(n_samples * 2, 5, figsize=(20, n_samples * 4))
    fig.suptitle('Domain C: Preprocessing Impact on Good vs Fault Images', fontsize=14, fontweight='bold')

    col_titles = ['Original\n(31x95)', 'Resized\n(448x448)', 'Cropped\n(392x392)',
                  'Patch Rep\n(28x28)', 'Diff from\nMean Good']

    # 먼저 good 이미지들의 평균 패치 표현 계산
    good_patch_reps = []
    for f in good_files[:20]:  # Use more samples for mean
        orig = load_tiff_image(f)
        _, _, cropped = dinomaly_preprocess(orig)
        patch_rep = calculate_patch_representation(cropped)
        good_patch_reps.append(patch_rep)
    mean_good_patch_rep = np.mean(good_patch_reps, axis=0)

    for idx, (sample_type, files) in enumerate([('Good', good_files), ('Fault', fault_files)]):
        for sample_idx, img_path in enumerate(files):
            row = idx * n_samples + sample_idx

            # Load and preprocess
            orig = load_tiff_image(img_path)
            orig_np, resized, cropped = dinomaly_preprocess(orig)

            # Calculate patch representation
            patch_rep = calculate_patch_representation(cropped)

            # Calculate difference from mean good
            diff_from_mean = patch_rep - mean_good_patch_rep

            # Plot
            axes[row, 0].imshow(orig_np, cmap='viridis', aspect='auto')
            axes[row, 0].set_title(f'{sample_type}: {img_path.stem}', fontsize=9)
            axes[row, 0].axis('off')

            axes[row, 1].imshow(resized, cmap='viridis')
            axes[row, 1].axis('off')

            axes[row, 2].imshow(cropped, cmap='viridis')
            axes[row, 2].axis('off')

            axes[row, 3].imshow(patch_rep, cmap='viridis')
            axes[row, 3].set_title(f'min={patch_rep.min():.3f}\nmax={patch_rep.max():.3f}', fontsize=8)
            axes[row, 3].axis('off')

            # Difference map
            vmax = max(abs(diff_from_mean.min()), abs(diff_from_mean.max()))
            im = axes[row, 4].imshow(diff_from_mean, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[row, 4].set_title(f'max diff: {np.max(np.abs(diff_from_mean)):.3f}', fontsize=8)
            axes[row, 4].axis('off')

    # Column titles
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title + '\n' + axes[0, col_idx].get_title(), fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'preprocessing_impact_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'preprocessing_impact_comparison.png'}")

def analyze_defect_signal_preservation():
    """결함 신호가 전처리 후에도 보존되는지 분석"""

    # 모든 샘플 로드
    good_files = list((TIFF_ROOT / "test" / "good").glob("*.tiff"))[:100]
    fault_files = list((TIFF_ROOT / "test" / "fault").glob("*.tiff"))[:100]

    print(f"Analyzing {len(good_files)} good and {len(fault_files)} fault samples...")

    # 원본 이미지 통계
    orig_good_max = []
    orig_fault_max = []

    # 전처리 후 통계
    proc_good_max = []
    proc_fault_max = []

    # 패치 표현 통계
    patch_good_max = []
    patch_fault_max = []

    for f in good_files:
        orig = load_tiff_image(f)
        _, _, cropped = dinomaly_preprocess(orig)
        patch_rep = calculate_patch_representation(cropped)

        orig_good_max.append(np.max(orig))
        proc_good_max.append(np.max(cropped))
        patch_good_max.append(np.max(patch_rep))

    for f in fault_files:
        orig = load_tiff_image(f)
        _, _, cropped = dinomaly_preprocess(orig)
        patch_rep = calculate_patch_representation(cropped)

        orig_fault_max.append(np.max(orig))
        proc_fault_max.append(np.max(cropped))
        patch_fault_max.append(np.max(patch_rep))

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Domain C: Defect Signal (Max Intensity) Preservation Through Preprocessing', fontsize=12)

    # Original
    axes[0].hist(orig_good_max, bins=30, alpha=0.6, label='Good', color='blue')
    axes[0].hist(orig_fault_max, bins=30, alpha=0.6, label='Fault', color='red')
    axes[0].set_xlabel('Max Intensity')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Original (31x95)\nSeparation: Good={np.mean(orig_good_max):.3f}±{np.std(orig_good_max):.3f}\nFault={np.mean(orig_fault_max):.3f}±{np.std(orig_fault_max):.3f}')
    axes[0].legend()

    # After preprocessing
    axes[1].hist(proc_good_max, bins=30, alpha=0.6, label='Good', color='blue')
    axes[1].hist(proc_fault_max, bins=30, alpha=0.6, label='Fault', color='red')
    axes[1].set_xlabel('Max Intensity')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'After Preprocessing (392x392)\nSeparation: Good={np.mean(proc_good_max):.3f}±{np.std(proc_good_max):.3f}\nFault={np.mean(proc_fault_max):.3f}±{np.std(proc_fault_max):.3f}')
    axes[1].legend()

    # After patch representation
    axes[2].hist(patch_good_max, bins=30, alpha=0.6, label='Good', color='blue')
    axes[2].hist(patch_fault_max, bins=30, alpha=0.6, label='Fault', color='red')
    axes[2].set_xlabel('Max Patch Mean')
    axes[2].set_ylabel('Count')
    axes[2].set_title(f'Patch Representation (28x28)\nSeparation: Good={np.mean(patch_good_max):.3f}±{np.std(patch_good_max):.3f}\nFault={np.mean(patch_fault_max):.3f}±{np.std(patch_fault_max):.3f}')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'defect_signal_preservation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'defect_signal_preservation.png'}")

    # Cohen's d 계산
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (np.mean(group2) - np.mean(group1)) / pooled_std

    print("\n=== Cohen's d (Fault - Good) ===")
    print(f"Original:        {cohens_d(orig_good_max, orig_fault_max):.3f}")
    print(f"After Preproc:   {cohens_d(proc_good_max, proc_fault_max):.3f}")
    print(f"Patch Rep:       {cohens_d(patch_good_max, patch_fault_max):.3f}")

def visualize_aspect_ratio_impact():
    """Aspect ratio 변화가 결함 패턴에 미치는 영향"""

    # 대표적인 fault 이미지 선택
    fault_files = list((TIFF_ROOT / "test" / "fault").glob("*.tiff"))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Domain C: Aspect Ratio Distortion Impact on Horizontal Line Defect', fontsize=14, fontweight='bold')

    for idx, f in enumerate(fault_files[:4]):
        orig = load_tiff_image(f)
        h, w = orig.shape

        # 원본 aspect ratio 유지하며 스케일업
        scale_factor = 392 / max(h, w)
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)

        # 원본 aspect ratio 유지 버전
        img_tensor = torch.from_numpy(orig).unsqueeze(0).unsqueeze(0).float()
        preserved_aspect = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
        preserved_aspect = preserved_aspect.squeeze().numpy()

        # Dinomaly 방식 (square resize)
        _, _, cropped_square = dinomaly_preprocess(orig)

        # 시각화
        axes[0, idx].imshow(orig, cmap='viridis', aspect='auto')
        axes[0, idx].set_title(f'{f.stem}\nOriginal: {h}x{w}')
        axes[0, idx].axis('off')

        axes[1, idx].imshow(cropped_square, cmap='viridis')
        axes[1, idx].set_title(f'Square Resize: 392x392\n(Aspect: {w/h:.2f} → 1.0)')
        axes[1, idx].axis('off')

    axes[0, 0].set_ylabel('Original\n(preserving aspect)', fontsize=12)
    axes[1, 0].set_ylabel('Dinomaly Preproc\n(square)', fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'aspect_ratio_distortion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'aspect_ratio_distortion.png'}")

if __name__ == "__main__":
    print("=" * 60)
    print("Domain C: Preprocessing Impact Analysis")
    print("=" * 60)

    visualize_preprocessing(n_samples=4)
    analyze_defect_signal_preservation()
    visualize_aspect_ratio_impact()

    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)
