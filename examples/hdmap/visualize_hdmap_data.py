#!/usr/bin/env python3
"""HDMAP 데이터셋 시각화 스크립트.

이 스크립트는 prepare_hdmap_dataset.py로 생성된 TIFF 데이터를 
도메인별로 시각화하여 PNG 이미지로 저장합니다.

실행 명령어:
python /mnt/ex-disk/taewan.hwang/study/anomalib/examples/hdmap/visualize_hdmap_data.py
"""

import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile

# =============================================================================
# 🚀 사용자 설정
# =============================================================================
# 기본 데이터 경로
BASE_PATH = "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png_minmax"

# 출력 폴더
OUTPUT_DIR = "/mnt/ex-disk/taewan.hwang/study/anomalib/examples/hdmap/visualize_hdmap_data"

# 도메인 목록
DOMAINS = ['A', 'B', 'C', 'D']

# 데이터 타입 (good/fault)
DATA_TYPES = ['good', 'fault']

# 각 폴더에서 선택할 이미지 수
N_SAMPLES = 18

# 이미지 출력 설정
FIGSIZE = (20, 16)  # 전체 figure 크기 (3x6 그리드용)
COLORMAP = 'gray'  # 컬러맵

# 그리드 설정
GRID_ROWS = 6  # 세로 6개
GRID_COLS = 3  # 가로 3개


def setup_output_directory():
    """출력 디렉토리 생성"""
    output_path = Path(OUTPUT_DIR)
    if output_path.exists():
        print(f"📁 기존 출력 폴더 사용: {OUTPUT_DIR}")
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 새 출력 폴더 생성: {OUTPUT_DIR}")
    return output_path


def get_random_image_files(folder_path, n_samples):
    """폴더에서 랜덤하게 이미지 파일 선택 (TIFF 또는 PNG)"""
    folder = Path(folder_path)

    if not folder.exists():
        print(f"  ⚠️ 경고: 폴더가 존재하지 않습니다 - {folder_path}")
        return []

    # TIFF와 PNG 파일 모두 가져오기
    image_files = list(folder.glob("*.tiff")) + list(folder.glob("*.png"))

    if len(image_files) == 0:
        print(f"  ⚠️ 경고: 이미지 파일이 없습니다 - {folder_path}")
        return []

    # 랜덤 샘플링
    n_available = len(image_files)
    n_select = min(n_samples, n_available)

    selected_files = random.sample(image_files, n_select)
    print(f"  📂 {folder.name}: {n_select}/{n_available}개 파일 선택")

    return selected_files


def load_image(file_path):
    """이미지 로드 (TIFF 또는 PNG, 확장자로 자동 판별)"""
    try:
        file_ext = Path(file_path).suffix.lower()

        if file_ext in ['.tiff', '.tif']:
            image = tifffile.imread(file_path)
        elif file_ext == '.png':
            # PNG는 16비트 uint16으로 로드 후 [0, 1]로 정규화
            image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if image is not None:
                image = image.astype(np.float32) / 65535.0
        else:
            raise ValueError(f"지원하지 않는 이미지 포맷: {file_ext}")
        return image
    except Exception as e:
        print(f"  ❌ 이미지 로드 실패: {file_path} - {e}")
        return None


def visualize_domain_data(domain, output_path):
    """특정 도메인의 데이터 시각화"""
    print(f"\n🔍 도메인 {domain} 시각화 중...")
    
    # 각 데이터 타입별로 파일 수집
    all_images = {}
    all_filenames = {}
    
    for data_type in DATA_TYPES:
        folder_path = Path(BASE_PATH) / f"domain_{domain}" / "test" / data_type
        selected_files = get_random_image_files(folder_path, N_SAMPLES)

        images = []
        filenames = []

        for file_path in selected_files:
            image = load_image(file_path)
            if image is not None:
                images.append(image)
                filenames.append(file_path.name)

        all_images[data_type] = images
        all_filenames[data_type] = filenames
    
    # 시각화
    if all_images['good'] or all_images['fault']:
        create_visualization(domain, all_images, all_filenames, output_path)
    else:
        print(f"  ❌ 도메인 {domain}: 시각화할 이미지가 없습니다.")


def create_visualization(domain, images_dict, filenames_dict, output_path):
    """도메인별 시각화 생성 및 저장 (정상/고장 각각 3x4 그리드)"""
    # 전체 이미지 수 계산
    n_good = len(images_dict['good'])
    n_fault = len(images_dict['fault'])
    
    if n_good == 0 and n_fault == 0:
        print(f"  ❌ 도메인 {domain}: 시각화할 이미지가 없습니다.")
        return
    
    # Figure 생성 (6행 x 6열: 좌측 3열=정상, 우측 3열=고장)
    fig, axes = plt.subplots(GRID_ROWS, 2*GRID_COLS, figsize=FIGSIZE)
    fig.suptitle(f'Domain {domain} - Random Samples (Good: {n_good}, Fault: {n_fault})', 
                 fontsize=16, fontweight='bold')
    
    # 열 제목 추가
    fig.text(0.25, 0.95, 'Good Samples', ha='center', va='center', fontsize=14, fontweight='bold')
    fig.text(0.75, 0.95, 'Fault Samples', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 모든 축 비활성화
    for i in range(GRID_ROWS):
        for j in range(2*GRID_COLS):
            axes[i, j].axis('off')
    
    # 정상 이미지 배치 (좌측 3열)
    if n_good > 0:
        for i in range(min(n_good, GRID_ROWS * GRID_COLS)):
            row = i // GRID_COLS
            col = i % GRID_COLS
            
            image = images_dict['good'][i]
            filename = filenames_dict['good'][i]
            
            # 스케일 0~1로 고정
            im = axes[row, col].imshow(image, cmap=COLORMAP, vmin=0, vmax=1)
            axes[row, col].set_title(f'{filename}', fontsize=8)
            axes[row, col].axis('off')
    
    # 고장 이미지 배치 (우측 3열)
    if n_fault > 0:
        for i in range(min(n_fault, GRID_ROWS * GRID_COLS)):
            row = i // GRID_COLS
            col = i % GRID_COLS + GRID_COLS  # 오른쪽 3열로 이동
            
            image = images_dict['fault'][i]
            filename = filenames_dict['fault'][i]
            
            # 스케일 0~1로 고정
            im = axes[row, col].imshow(image, cmap=COLORMAP, vmin=0, vmax=1)
            axes[row, col].set_title(f'{filename}', fontsize=8)
            axes[row, col].axis('off')
    
    # 레이아웃 조정
    plt.tight_layout(rect=[0, 0, 1.0, 0.92])
    
    # 저장
    output_file = output_path / f'domain_{domain}_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ 저장 완료: {output_file}")


def generate_summary_statistics(output_path):
    """전체 도메인 통계 요약 생성"""
    print(f"\n📊 데이터셋 통계 요약 생성 중...")
    
    summary_data = {}
    
    for domain in DOMAINS:
        domain_stats = {'good': 0, 'fault': 0}
        
        for data_type in DATA_TYPES:
            folder_path = Path(BASE_PATH) / f"domain_{domain}" / "test" / data_type
            if folder_path.exists():
                image_files = list(folder_path.glob("*.tiff")) + list(folder_path.glob("*.png"))
                domain_stats[data_type] = len(image_files)
        
        summary_data[domain] = domain_stats
    
    # 요약 텍스트 파일 생성
    summary_file = output_path / 'dataset_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("HDMAP 데이터셋 시각화 요약\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"기본 경로: {BASE_PATH}\n")
        f.write(f"출력 경로: {OUTPUT_DIR}\n")
        f.write(f"샘플링 수: {N_SAMPLES}개씩\n\n")
        
        f.write("도메인별 파일 수:\n")
        f.write("-" * 30 + "\n")
        
        total_good = 0
        total_fault = 0
        
        for domain in DOMAINS:
            stats = summary_data[domain]
            f.write(f"Domain {domain}: Good={stats['good']:,}, Fault={stats['fault']:,}\n")
            total_good += stats['good']
            total_fault += stats['fault']
        
        f.write("-" * 30 + "\n")
        f.write(f"전체 합계: Good={total_good:,}, Fault={total_fault:,}\n")
    
    print(f"  ✅ 요약 파일 저장: {summary_file}")
    
    # 콘솔에도 출력
    print(f"\n📈 데이터셋 통계:")
    for domain in DOMAINS:
        stats = summary_data[domain]
        print(f"  Domain {domain}: Good={stats['good']:,}, Fault={stats['fault']:,}")


def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("🎨 HDMAP 데이터셋 시각화 도구")
    print("=" * 80)
    print(f"📂 데이터 경로: {BASE_PATH}")
    print(f"💾 출력 경로: {OUTPUT_DIR}")
    print(f"🎲 샘플링: 각 폴더당 {N_SAMPLES}개 (3x6 그리드)")
    print(f"🏷️  도메인: {', '.join(DOMAINS)}")
    
    # 기본 경로 확인
    base_path = Path(BASE_PATH)
    if not base_path.exists():
        print(f"\n❌ 오류: 기본 데이터 경로가 존재하지 않습니다 - {BASE_PATH}")
        print("   prepare_hdmap_dataset.py를 먼저 실행하여 데이터를 생성하세요.")
        return
    
    # 출력 디렉토리 설정
    output_path = setup_output_directory()
    
    # 랜덤 시드 설정 (재현 가능한 결과)
    random.seed(42)
    
    # 각 도메인별 시각화
    for domain in DOMAINS:
        visualize_domain_data(domain, output_path)
    
    # 요약 통계 생성
    generate_summary_statistics(output_path)
    
    print(f"\n✅ 모든 시각화 완료!")
    print(f"📁 결과 확인: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
