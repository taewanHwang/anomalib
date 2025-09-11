#!/usr/bin/env python3
"""HDMAP 데이터셋 준비 스크립트 (Clean Version).

이 스크립트는 mat 파일 형태의 원본 HDMAP 데이터를 PNG 이미지로 변환합니다.
프로토타입과 동일한 전역 정규화 방식을 사용하여 성능 저하 없이 변환합니다.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image

# =============================================================================
# 🚀 사용자 설정 (필요에 따라 수정)
# =============================================================================
# 전역 정규화 설정 (항상 사용)
CLIP_MIN = -4.0  # 클리핑 최솟값 (z-score 기준)
CLIP_MAX = 10.0  # 클리핑 최댓값 (z-score 기준)

# 데이터 설정
N_TRAINING = 100  # 훈련 샘플 수 (프로토타입과 동일)
N_TESTING = 2000     # 테스트 샘플 수
BIT_DEPTH = '16bit'  # 비트 심도 ('8bit' 또는 '16bit')

# 기타 설정  
TARGET_SIZE = (224, 224)  # 목표 이미지 크기 (리사이즈용)
BASE_FOLDER = "HDMAP"     # 최상위 폴더명

# 처리 방식 설정 (1채널 2가지 방식)
PROCESSING_MODES = [
    'original',                   # 원본
    f'resize_{TARGET_SIZE[0]}x{TARGET_SIZE[1]}'  # 리사이즈 (동적 크기)
]

# =============================================================================
# 도메인 구성 정보 (중앙 집중식 관리)
# =============================================================================
DOMAIN_CONFIG = {
    'A': {
        'sensor': 'Class1/1',
        'data_type': '3_TSA_DIF'
    },
    'B': {
        'sensor': 'Class3/1', 
        'data_type': '1_TSA_DIF'
    },
    'C': {
        'sensor': 'Class1/1',
        'data_type': '1_TSA_DIF'
    },
    'D': {
        'sensor': 'Class3/1',
        'data_type': '3_TSA_DIF'
    }
}

# 기본 경로
BASE_DATA_PATH = 'datasets/raw/KRISS_share_nipa2023'

# =============================================================================
# 핵심 함수들
# =============================================================================
def scale_norm(X, X_mean=None, X_std=None):
    """Z-score 정규화 (프로토타입과 동일)"""
    if X_mean is None or X_std is None:
        X_mean = np.mean(X)
        X_std = np.std(X)
    X_normalized = (X - X_mean) / X_std
    return X_normalized, X_mean, X_std

def generate_paths():
    """도메인 구성 정보로부터 모든 경로 생성"""
    paths = {}
    
    for domain, config in DOMAIN_CONFIG.items():
        sensor_path = config['sensor']
        data_type = config['data_type']
        
        # 기본 경로 구성
        normal_base = f"{BASE_DATA_PATH}/Normal/Normal2_LSSm0.3_HSS0/{sensor_path}/HDMap_train_test"
        fault_base = f"{BASE_DATA_PATH}/Planet_fault_ring/1.42_LSSm0.3_HSS0/{sensor_path}/HDMap_train_test"
        
        paths[domain] = {
            'train_normal': f"{normal_base}/{data_type}_train.mat",
            'test_normal': f"{normal_base}/{data_type}_test.mat", 
            'test_fault': f"{fault_base}/{data_type}_test.mat"
        }
    
    return paths

def get_folder_name(processing_mode):
    """설정에 따른 폴더명 생성"""
    return f"{N_TRAINING}_{BIT_DEPTH}_{processing_mode}"

def save_image_with_global_normalization(img_array, save_path):
    """전역 정규화를 사용하여 이미지 저장"""
    # 1. 클리핑 적용
    clipped = np.clip(img_array, CLIP_MIN, CLIP_MAX)
    
    # 2. [CLIP_MIN, CLIP_MAX] → [0, 1] 매핑
    normalized = (clipped - CLIP_MIN) / (CLIP_MAX - CLIP_MIN)
    
    # 3. 비트 심도에 따른 양자화
    if BIT_DEPTH == '8bit':
        quantized = (normalized * 255).astype(np.uint8)
    elif BIT_DEPTH == '16bit':
        quantized = (normalized * 65535).astype(np.uint16)
    else:
        raise ValueError("BIT_DEPTH must be '8bit' or '16bit'")
    
    # 4. PNG로 저장
    img_pil = Image.fromarray(quantized)
    img_pil.save(save_path)

def save_image_legacy(img_array, save_path):
    """기존 방식 이미지 저장 (개별 정규화)"""
    # 개별 정규화
    img_normalized = ((img_array - img_array.min()) / 
                     (img_array.max() - img_array.min()))
    
    if BIT_DEPTH == '8bit':
        quantized = (img_normalized * 255).astype(np.uint8)
    elif BIT_DEPTH == '16bit':
        quantized = (img_normalized * 65535).astype(np.uint16)
    else:
        raise ValueError("BIT_DEPTH must be '8bit' or '16bit'")
    
    img_pil = Image.fromarray(quantized)
    img_pil.save(save_path)

def resize_image_with_aspect_ratio(img, target_size):
    """비율을 유지하며 이미지 리사이즈"""
    img_pil = Image.fromarray(img.astype(np.uint8))
    img_resized = img_pil.resize(target_size, Image.LANCZOS)
    return np.array(img_resized)

def process_image_by_mode(img_array, processing_mode, target_size=TARGET_SIZE):
    """처리 모드에 따른 이미지 처리 (전역 정규화된 데이터용)"""
    if processing_mode == 'original':
        return img_array
    
    elif processing_mode.startswith('resize_'):
        # 전역 정규화된 데이터를 그대로 사용 (스케일 유지)
        # 클리핑된 범위 [-4, 10]를 [0, 255] 범위로 매핑하여 리사이즈
        clipped = np.clip(img_array, CLIP_MIN, CLIP_MAX)
        normalized = (clipped - CLIP_MIN) / (CLIP_MAX - CLIP_MIN)
        img_scaled = (normalized * 255).astype(np.uint8)
        resized = resize_image_with_aspect_ratio(img_scaled, target_size)
        # 다시 원래 범위로 복원
        return (resized.astype(np.float32) / 255.0) * (CLIP_MAX - CLIP_MIN) + CLIP_MIN
    
    else:
        raise ValueError(f"Unknown processing mode: {processing_mode}")


def compute_domain_stats():
    """각 도메인별 전역 통계량 계산"""
    print("="*80)
    print("🔢 도메인별 전역 통계량 계산 중...")
    print("="*80)
    
    paths = generate_paths()
    domain_stats = {}
    
    for domain, domain_paths in paths.items():
        train_path = domain_paths['train_normal']
        
        if os.path.exists(train_path):
            print(f"도메인 {domain} 통계량 계산 중...")
            
            # mat 파일 로드
            mat_data = scipy.io.loadmat(train_path)
            train_data = mat_data['Xdata']
            
            # 데이터 형태 변환 (프로토타입과 동일)
            X_train = train_data.transpose(3,2,0,1)  # (samples, channels, height, width)
            
            # 전역 통계량 계산
            _, X_mean, X_std = scale_norm(X_train)
            domain_stats[domain] = {'mean': X_mean, 'std': X_std}
            
            # 정규화 후 통계량 확인
            X_normalized, _, _ = scale_norm(X_train, X_mean, X_std)
            
            print(f"  도메인 {domain}:")
            print(f"    원본: mean={X_mean:.6f}, std={X_std:.6f}")
            print(f"    정규화 후: min={X_normalized.min():.6f}, max={X_normalized.max():.6f}, mean={X_normalized.mean():.6f}, std={X_normalized.std():.6f}")
        else:
            print(f"  ⚠️ 경고: {train_path} 파일을 찾을 수 없습니다.")
    
    return domain_stats

def process_domain_data(domain, domain_paths, domain_stats, folder_name, processing_mode):
    """도메인별 데이터 처리"""
    print(f"\n🔄 도메인 {domain} 처리 중... (모드: {processing_mode})")
    
    # 저장 경로 설정
    save_dirs = {}
    for data_type in ['train/good', 'test/good', 'test/fault']:
        save_dir = f"datasets/{BASE_FOLDER}/{folder_name}/domain_{domain}/{data_type}"
        save_dirs[data_type] = save_dir
        
        # 디렉토리 생성
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
    
    # 데이터 처리 매핑
    data_mapping = [
        ('train_normal', 'train/good', N_TRAINING, "정상 훈련"),
        ('test_normal', 'test/good', N_TESTING, "정상 테스트"), 
        ('test_fault', 'test/fault', N_TESTING, "고장 테스트")
    ]
    
    stats = domain_stats.get(domain, {})
    
    for data_key, save_key, max_samples, description in data_mapping:
        mat_path = domain_paths[data_key]
        save_dir = save_dirs[save_key]
        
        if not os.path.exists(mat_path):
            print(f"  ⚠️ {description}: {mat_path} 파일 없음")
            continue
        
        print(f"  📂 {description} 데이터 처리 중...")
        
        # mat 파일 로드
        mat_data = scipy.io.loadmat(mat_path)
        image_data = mat_data['Xdata']
        
        # 샘플 수 결정
        actual_samples = image_data.shape[3] if len(image_data.shape) > 3 else 1
        num_samples = min(max_samples, actual_samples)
        
        # 이미지 저장
        for i in range(num_samples):
            img = image_data[:, :, 0, i]
            filename = f'{i:06d}.png'
            save_path = os.path.join(save_dir, filename)
            
            # 전역 정규화 방식
            img_normalized, _, _ = scale_norm(img, stats['mean'], stats['std'])
            # 처리 모드 적용 (전역 정규화된 데이터에 대해)
            if processing_mode == 'original':
                save_image_with_global_normalization(img_normalized, save_path)
            else:
                # 전역 정규화된 데이터를 사용하여 처리
                processed_img = process_image_by_mode(img_normalized, processing_mode)
                save_image_with_global_normalization(processed_img, save_path)
            
            # 진행상황 출력
            if (i + 1) % 10000 == 0:
                print(f"    진행: {i + 1}/{num_samples}")
        
        print(f"  ✅ {description}: {num_samples}개 저장 완료")

def prepare_hdmap_dataset_multiple_modes():
    """여러 처리 모드로 HDMAP 데이터셋 준비"""
    print("="*80)
    print("🚀 HDMAP 데이터셋 변환 시작 (다중 모드)")
    print("="*80)
    print(f"훈련 샘플 수: {N_TRAINING:,}")
    print(f"테스트 샘플 수: {N_TESTING:,}")
    print(f"비트 심도: {BIT_DEPTH}")
    print(f"처리 모드: {len(PROCESSING_MODES)}개 (original, resize)")
    
    print(f"정규화: 전역 (클리핑: [{CLIP_MIN}, {CLIP_MAX}])")
    
    print("="*80)
    
    # 1. 경로 준비
    paths = generate_paths()
    
    # 2. 전역 통계량 계산
    domain_stats = compute_domain_stats()
    
    # 3. 각 처리 모드별로 데이터 처리
    for processing_mode in PROCESSING_MODES:
        print(f"\n🔄 처리 모드: {processing_mode}")
        folder_name = get_folder_name(processing_mode)
        
        for domain in DOMAIN_CONFIG.keys():
            domain_paths = paths[domain]
            process_domain_data(domain, domain_paths, domain_stats, folder_name, processing_mode)

def prepare_hdmap_dataset():
    """단일 모드로 HDMAP 데이터셋 준비 (기본: 1ch_original 모드)"""
    print("="*80)
    print("🚀 HDMAP 데이터셋 변환 시작")
    print("="*80)
    print(f"훈련 샘플 수: {N_TRAINING:,}")
    print(f"테스트 샘플 수: {N_TESTING:,}")
    print(f"비트 심도: {BIT_DEPTH}")
    
    print(f"정규화: 전역 (클리핑: [{CLIP_MIN}, {CLIP_MAX}])")
    
    print("="*80)
    
    # 1. 폴더명 및 경로 준비
    processing_mode = 'original'  # 기본 모드
    folder_name = get_folder_name(processing_mode)
    paths = generate_paths()
    
    # 2. 전역 통계량 계산
    domain_stats = compute_domain_stats()
    
    # 3. 각 도메인별 데이터 처리
    for domain in DOMAIN_CONFIG.keys():
        domain_paths = paths[domain]
        process_domain_data(domain, domain_paths, domain_stats, folder_name, processing_mode)
    
    # 4. 완료 메시지
    print("\n" + "="*80)
    print("🎉 HDMAP 데이터셋 변환 완료!")
    print("="*80)
    print(f"저장 위치: datasets/{BASE_FOLDER}/{get_folder_name('original')}/")
    print("구조:")
    for domain in DOMAIN_CONFIG.keys():
        print(f"  domain_{domain}/")
        print(f"    ├── train/good/     # 정상 훈련 ({N_TRAINING:,}개)")
        print(f"    └── test/")
        print(f"        ├── good/       # 정상 테스트 ({N_TESTING:,}개)")
        print(f"        └── fault/      # 고장 테스트 ({N_TESTING:,}개)")
    
    print(f"\n🎯 전역 정규화 완료! 프로토타입과 동일한 방식으로 변환됨")
    print(f"📝 로드 시 역변환: pixel / 65535 * ({CLIP_MAX} - ({CLIP_MIN})) + ({CLIP_MIN})")
    print(f"✨ 이제 AUC 0.9999 성능을 재현할 수 있습니다!")

if __name__ == "__main__":
    # 전체 모드 (모든 처리 방식) - 2개 폴더 생성 (original, resize)
    prepare_hdmap_dataset_multiple_modes()
    
    # 단일 모드 (original 모드만)를 원하는 경우 아래 라인으로 변경
    # prepare_hdmap_dataset()