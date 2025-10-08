#!/usr/bin/env python3
"""HDMAP 데이터셋 준비 스크립트 (Clean Version).

이 스크립트는 mat 파일 형태의 원본 HDMAP 데이터를 PNG 이미지로 변환합니다.
프로토타입과 동일한 전역 정규화 방식을 사용하여 성능 저하 없이 변환합니다.

실행 명령어:
nohup python /mnt/ex-disk/taewan.hwang/study/anomalib/examples/hdmap/prepare_hdmap_dataset.py > hdmap_dataset.log 2>&1 &

프로세스 종료:
1. 실행 중인 프로세스 확인: ps aux | grep prepare_hdmap_dataset.py
2. 프로세스 종료: kill -9 <PID>
또는
pkill -f prepare_hdmap_dataset.py
"""

import os
import shutil

import cv2
import numpy as np
import scipy.io
import tifffile

# =============================================================================
# 🚀 사용자 설정 (필요에 따라 수정)
# =============================================================================
# 데이터 설정
N_TRAINING = 1000  # 훈련 샘플 수
N_TESTING = 2000   # 테스트 샘플 수
SAVE_FORMATS = ['png']  # 저장 형식 (TIFF, PNG)
BASE_FOLDER = "HDMAP"    # 최상위 폴더명
RANDOM_SEED = 42  # 랜덤 시드 (재현성 보장)

# 정규화 방식 설정
NORMALIZATION_MODES = [
    'original',    # 원본 데이터 (스케일링 없음)
    'zscore',      # 기존 domain_stats 기반 z-score 정규화
    'minmax',      # 사용자 제공 min-max 스케일링
]

# =============================================================================
# 도메인 구성 정보 (중앙 집중식 관리)
# =============================================================================
DOMAIN_CONFIG = {
    'A': {
        'sensor': 'Class1/1',
        'data_type': '3_TSA_DIF',
        'user_min': 0.0,
        'user_max': 0.32
    },
    'B': {
        'sensor': 'Class1/1',
        'data_type': '1_TSA_DIF',
        'user_min': 0.0,
        'user_max': 1.2
    },
    'C': {
        'sensor': 'Class3/1',
        'data_type': '3_TSA_DIF',
        'user_min': 0.0,
        'user_max': 0.09
    },
    'D': {
        'sensor': 'Class3/1', 
        'data_type': '1_TSA_DIF',
        'user_min': 0.0,
        'user_max': 0.41
    },
}

# 기본 경로
BASE_DATA_PATH = 'datasets/raw/KRISS_share_nipa2023'

# =============================================================================
# 핵심 함수들
# =============================================================================
def normalize_zscore(X, X_mean=None, X_std=None):
    """Z-score 정규화 (프로토타입과 동일)"""
    if X_mean is None or X_std is None:
        X_mean = np.mean(X)
        X_std = np.std(X)
    X_normalized = (X - X_mean) / X_std
    return X_normalized, X_mean, X_std

def normalize_minmax(X, user_min, user_max):
    """사용자 제공 min-max 값으로 [0, 1] 범위로 스케일링"""
    # 사용자 제공 범위로 클리핑
    X_clipped = np.clip(X, user_min, user_max)
    # [user_min, user_max] → [0, 1] 매핑
    X_scaled = (X_clipped - user_min) / (user_max - user_min)
    return X_scaled

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

def generate_folder_name(save_format, normalization_mode):
    """설정에 따른 데이터셋 폴더명 생성"""
    if normalization_mode == 'original':
        return f"{N_TRAINING}_{save_format}_original"
    else:
        return f"{N_TRAINING}_{save_format}_{normalization_mode}"

def save_tiff_image(img_array, save_path):
    """이미지를 32비트 부동소수점 TIFF 파일로 저장"""
    tifffile.imwrite(save_path, img_array.astype(np.float32))

def save_png_image(img_array, save_path):
    """이미지를 16비트 PNG 파일로 저장"""
    # [0, 1] 범위를 [0, 65535]로 스케일링
    img_16bit = (img_array * 65535).astype(np.uint16)
    cv2.imwrite(save_path, img_16bit)

def compute_domain_statistics():
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
            _, X_mean, X_std = normalize_zscore(X_train)
            domain_stats[domain] = {'mean': X_mean, 'std': X_std}
            
            # 정규화 후 통계량 확인
            X_normalized, _, _ = normalize_zscore(X_train, X_mean, X_std)
            
            print(f"  도메인 {domain}:")
            print(f"    원본: mean={X_mean:.6f}, std={X_std:.6f}")
            print(f"    정규화 후: min={X_normalized.min():.6f}, max={X_normalized.max():.6f}, mean={X_normalized.mean():.6f}, std={X_normalized.std():.6f}")
        else:
            print(f"  ⚠️ 경고: {train_path} 파일을 찾을 수 없습니다.")
    
    return domain_stats

def process_single_domain(domain, domain_paths, domain_stats, folder_name, save_format, normalization_mode):
    """도메인별 데이터 처리"""
    print(f"\n🔄 도메인 {domain} 처리 중... (형식: {save_format}, 정규화: {normalization_mode})")
    
    # 저장 경로 설정
    save_dirs = {}
    for data_type in ['train/good', 'test/good', 'test/fault']:
        save_dir = f"datasets/{BASE_FOLDER}/{folder_name}/domain_{domain}/{data_type}"
        save_dirs[data_type] = save_dir
        
        # 디렉토리 생성
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
    
    # 데이터 처리 매핑 (data_key, save_key, max_samples, description, shuffle)
    data_mapping = [
        ('train_normal', 'train/good', N_TRAINING, "정상 훈련", True),
        ('test_normal', 'test/good', N_TESTING, "정상 테스트", False),
        ('test_fault', 'test/fault', N_TESTING, "고장 테스트", False)
    ]
    
    # 정규화 모드에 따라 필요한 변수만 준비
    stats = {}
    user_min = user_max = None
    
    if normalization_mode == 'zscore':
        stats = domain_stats.get(domain, {})
        if 'mean' not in stats or 'std' not in stats:
            print(f"  ⚠️ 경고: 도메인 {domain}의 통계량이 없습니다. 해당 도메인을 건너뜁니다.")
            return
    elif normalization_mode == 'minmax':
        domain_config = DOMAIN_CONFIG[domain]
        user_min = domain_config['user_min']
        user_max = domain_config['user_max']
    
    for data_key, save_key, max_samples, description, shuffle in data_mapping:
        mat_path = domain_paths[data_key]
        save_dir = save_dirs[save_key]

        if not os.path.exists(mat_path):
            print(f"  ⚠️ {description}: {mat_path} 파일 없음")
            continue

        print(f"  📂 {description} 데이터 처리 중... (shuffle={shuffle})")

        # mat 파일 로드
        mat_data = scipy.io.loadmat(mat_path)
        image_data = mat_data['Xdata']

        # 샘플 수 결정
        actual_samples = image_data.shape[3] if len(image_data.shape) > 3 else 1
        num_samples = min(max_samples, actual_samples)

        # 인덱스 생성 (shuffle 옵션에 따라)
        if shuffle:
            np.random.seed(RANDOM_SEED)
            indices = np.random.choice(actual_samples, num_samples, replace=False)
            print(f"    🔀 랜덤 샘플링: {num_samples}개 선택 (전체 {actual_samples}개 중)")
        else:
            indices = np.arange(num_samples)
            print(f"    📋 순차 샘플링: 처음 {num_samples}개 선택")

        # 이미지 저장
        for idx, i in enumerate(indices):
            img = image_data[:, :, 0, i]

            # 파일 확장자 결정
            file_ext = 'tiff' if save_format == 'tiff' else 'png'
            # 파일명은 순차적으로 저장 (000000.png, 000001.png, ...)
            filename = f'{idx:06d}.{file_ext}'
            save_path = os.path.join(save_dir, filename)

            # 정규화 방식에 따른 처리
            if normalization_mode == 'original':
                # 원본 데이터 그대로 저장 (스케일링 없음)
                processed_img = img

            elif normalization_mode == 'zscore':
                # Z-score 정규화 방식
                processed_img, _, _ = normalize_zscore(img, stats['mean'], stats['std'])

            elif normalization_mode == 'minmax':
                # Min-Max 스케일링 방식 (도메인별 설정 사용)
                processed_img = normalize_minmax(img, user_min, user_max)

            else:
                raise ValueError(f"Unknown normalization mode: {normalization_mode}")

            # 저장 형식에 따른 저장
            if save_format == 'tiff':
                save_tiff_image(processed_img, save_path)
            elif save_format == 'png':
                save_png_image(processed_img, save_path)
            else:
                raise ValueError(f"Unknown save format: {save_format}")
            
            # 진행상황 출력
            if (idx + 1) % 10000 == 0:
                print(f"    진행: {idx + 1}/{num_samples}")
        
        print(f"  ✅ {description}: {num_samples}개 저장 완료")

def create_hdmap_datasets():
    """여러 처리 모드로 HDMAP 데이터셋 준비 (z-score + min-max 정규화 지원)"""
    print("="*80)
    print("🚀 HDMAP 데이터셋 변환 시작 (다중 모드)")
    print("="*80)
    print(f"훈련 샘플 수: {N_TRAINING:,}")
    print(f"테스트 샘플 수: {N_TESTING:,}")
    print(f"저장 형식: {len(SAVE_FORMATS)}개 ({', '.join(SAVE_FORMATS)})")
    print(f"정규화 방식: {len(NORMALIZATION_MODES)}개 ({', '.join(NORMALIZATION_MODES)})")
    
    print(f"\n정규화 설정:")
    print(f"  - Original: 원본 데이터 그대로 저장 (스케일링 없음)")
    print(f"  - Z-score: 전역 통계량 기반")
    print(f"  - Min-Max: 도메인별 사용자 제공 범위 → [0, 1]")
    for domain, config in DOMAIN_CONFIG.items():
        print(f"    도메인 {domain}: [{config['user_min']}, {config['user_max']}]")
    
    print("="*80)
    
    # 1. 경로 준비
    paths = generate_paths()
    
    # 2. z-score 방식용 전역 통계량 계산 (z-score 모드가 활성화된 경우만)
    domain_stats = {}
    if 'zscore' in NORMALIZATION_MODES:
        domain_stats = compute_domain_statistics()
    else:
        print("🔢 Z-score 모드가 비활성화되어 통계량 계산을 건너뜁니다.")
    
    # 3. 각 정규화 방식, 저장 형식별로 데이터 처리
    for normalization_mode in NORMALIZATION_MODES:
        for save_format in SAVE_FORMATS:
            print(f"\n🔄 처리: {normalization_mode.upper()} - {save_format.upper()}")
            folder_name = generate_folder_name(save_format, normalization_mode)
            
            for domain in DOMAIN_CONFIG.keys():
                domain_paths = paths[domain]
                process_single_domain(domain, domain_paths, domain_stats, folder_name, save_format, normalization_mode)

    print(f"\n✅ 모든 처리 완료!")
    
    # 생성된 폴더 요약
    print(f"\n📁 생성된 데이터셋 폴더:")
    for normalization_mode in NORMALIZATION_MODES:
        for save_format in SAVE_FORMATS:
            folder_name = generate_folder_name(save_format, normalization_mode)
            print(f"  - datasets/{BASE_FOLDER}/{folder_name}/")
    
    print(f"\n각 폴더에는 domain_A, domain_B, domain_C, domain_D가 포함되어 있습니다.")

if __name__ == "__main__":
    # HDMAP 데이터셋 생성 (Z-score, Min-Max 정규화)
    create_hdmap_datasets()