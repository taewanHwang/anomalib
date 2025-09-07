#!/usr/bin/env python3
"""HDMAP 데이터셋 준비 스크립트.

이 스크립트는 mat 파일 형태의 원본 HDMAP 데이터를 PNG 이미지로 변환합니다.
다양한 전처리 옵션(resize, padding)을 제공하여 이상 탐지 모델 학습에 적합한 형태로 변환합니다.

HDMAP (Health Data Map): 설비 상태 진단을 위한 센서 데이터를 시각화한 2D 맵 형태의 데이터
"""

import os
import shutil
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image, ImageOps
from scipy import ndimage


def apply_sobel_filter(img_array):
    """Sobel 필터를 적용하여 x, y 방향 엣지를 검출합니다.
    
    Args:
        img_array: 입력 이미지 배열 (numpy array)
        
    Returns:
        tuple: (sobel_x, sobel_y) - x방향 및 y방향 sobel 필터 결과
    """
    # 이미지 정규화 (0-1 범위)
    img_normalized = (img_array - img_array.min()) / (img_array.max() - img_array.min())
    
    # Sobel 필터 적용
    sobel_x = ndimage.sobel(img_normalized, axis=1)  # x 방향 (세로 엣지)
    sobel_y = ndimage.sobel(img_normalized, axis=0)  # y 방향 (가로 엣지)
    
    # 절댓값 취하고 0-1 범위로 정규화
    sobel_x = np.abs(sobel_x)
    sobel_y = np.abs(sobel_y)
    
    return sobel_x, sobel_y


def create_3channel_image(original, sobel_x, sobel_y):
    """원본 이미지와 Sobel 필터 결과를 합쳐 3채널 RGB 이미지를 생성합니다.
    
    Args:
        original: 원본 이미지 배열
        sobel_x: x방향 sobel 필터 결과
        sobel_y: y방향 sobel 필터 결과
        
    Returns:
        PIL.Image: 3채널 RGB 이미지 (R=원본, G=sobel_x, B=sobel_y)
    """
    # 각 채널을 0-255 범위로 정규화
    r_channel = ((original - original.min()) / (original.max() - original.min()) * 255).astype(np.uint8)
    g_channel = (sobel_x / sobel_x.max() * 255).astype(np.uint8) if sobel_x.max() > 0 else np.zeros_like(sobel_x, dtype=np.uint8)
    b_channel = (sobel_y / sobel_y.max() * 255).astype(np.uint8) if sobel_y.max() > 0 else np.zeros_like(sobel_y, dtype=np.uint8)
    
    # 3채널 이미지 생성
    rgb_array = np.stack([r_channel, g_channel, b_channel], axis=2)
    
    return Image.fromarray(rgb_array, mode='RGB')


def process_image_resize_with_padding(img_array, target_size=(256, 256)):
    """이미지를 비율 유지하면서 리사이즈하고 패딩 추가.
    
    Args:
        img_array: 입력 이미지 배열 (numpy array)
        target_size: 목표 크기 (height, width)
        
    Returns:
        PIL.Image: 처리된 이미지
    """
    # 이미지 정규화 (0-255 범위로 변환)
    img_normalized = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_normalized)  # 자동으로 그레이스케일로 인식
    
    # 비율 유지하면서 리사이즈
    w, h = img_pil.size
    scale = min(target_size[0]/h, target_size[1]/w)  # 작은 축에 맞춰 스케일 계산
    new_w = int(w * scale)
    new_h = int(h * scale)
    img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 중앙 정렬로 패딩 추가
    pad_h = max(0, target_size[0] - new_h)
    pad_w = max(0, target_size[1] - new_w)
    padding = (pad_w//2, pad_h//2, (pad_w+1)//2, (pad_h+1)//2)  # 좌, 상, 우, 하
    img_padded = ImageOps.expand(img_pil, padding, fill=0)  # 검은색으로 패딩
        
    return img_padded


def process_image_pad_only(img_array, target_size=(256, 256)):
    """원본 크기 유지하면서 패딩만 추가.
    
    Args:
        img_array: 입력 이미지 배열
        target_size: 목표 크기 (height, width)
        
    Returns:
        PIL.Image: 처리된 이미지
    """
    # 이미지 정규화
    img_normalized = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_normalized)
    
    # 원본 크기 유지하면서 패딩
    w, h = img_pil.size
    pad_h = max(0, target_size[0] - h)
    pad_w = max(0, target_size[1] - w)
    padding = (pad_w//2, pad_h//2, (pad_w+1)//2, (pad_h+1)//2)
    img_padded = ImageOps.expand(img_pil, padding, fill=0)
    
    return img_padded


def process_image_resize_only(img_array, target_size=(256, 256)):
    """단순 리사이즈 (비율 무시).
    
    Args:
        img_array: 입력 이미지 배열
        target_size: 목표 크기 (height, width)
        
    Returns:
        PIL.Image: 처리된 이미지
    """
    # 이미지 정규화
    img_normalized = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_normalized)
    
    # 강제 리사이즈 (비율 무시)
    img_resized = img_pil.resize(target_size)
    
    return img_resized


def save_image_with_depth(img_input, save_path, bit_depth='8bit'):
    """이미지를 지정된 비트 심도로 저장.
    
    Args:
        img_input: PIL Image 또는 numpy 배열
        save_path: 저장 경로
        bit_depth: 비트 심도 ('8bit' 또는 '16bit')
    """
    # PIL Image인 경우 numpy 배열로 변환
    if isinstance(img_input, Image.Image):
        img_array = np.array(img_input)
    else:
        img_array = img_input

    if bit_depth == '8bit':
        # 8비트 변환 (일반적인 PNG)
        img_normalized = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_normalized)
        img_pil.save(save_path)
    
    elif bit_depth == '16bit':
        # 16비트 변환 (더 높은 정밀도)
        img_normalized = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 65535).astype(np.uint16)
        img_pil = Image.fromarray(img_normalized.astype('uint16'))
        img_pil.save(save_path, format='PNG')


def prepare_hdmap_dataset():
    """HDMAP 데이터셋 준비 메인 함수."""
    
    # =============================================================================
    # 설정 파라미터 (필요에 따라 수정)
    # =============================================================================
    N_training = 1000  # 훈련 샘플 수
    N_testing = 2000    # 테스트 샘플 수
    bit_depth = '8bit'  # 비트 심도 ('8bit' 또는 '16bit')
    target_size = (224, 224)  # 목표 이미지 크기
    
    # 폴더 구조 설정
    base_folder = "HDMAP"  # 최상위 폴더
    folder_name = f"{N_training}_{bit_depth}_original"  # 원본 크기 저장 폴더명
    
    # =============================================================================
    # 데이터 경로 매핑 설정
    # =============================================================================
    # mat 파일 경로와 저장 경로 매핑 정의
    # 각 도메인별로 정상(good)과 이상(fault) 데이터를 train/test로 분리
    path_mapping = [
        # Domain-A: Class1의 1번 센서 데이터
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Normal/Normal2_LSSm0.3_HSS0/Class1/1/HDMap_train_test/1_TSA_DIF_train.mat',
            'save_dir': f'datasets/{base_folder}/{folder_name}/domain_A/train/good',
            'slice_from': 0,
            'slice_to': N_training,
            'description': 'Domain-A 정상 훈련 데이터'
        },
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Normal/Normal2_LSSm0.3_HSS0/Class1/1/HDMap_train_test/1_TSA_DIF_test.mat',
            'save_dir': f'datasets/{base_folder}/{folder_name}/domain_A/test/good',
            'slice_from': 0,
            'slice_to': N_testing,
            'description': 'Domain-A 정상 테스트 데이터'
        },
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Planet_fault_ring/1.42_LSSm0.3_HSS0/Class1/1/HDMap_train_test/1_TSA_DIF_test.mat',
            'save_dir': f'datasets/{base_folder}/{folder_name}/domain_A/test/fault',
            'slice_from': 0,
            'slice_to': N_testing,
            'description': 'Domain-A 결함 테스트 데이터'
        },
        
        # Domain-B: Class3의 1번 센서 데이터
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Normal/Normal2_LSSm0.3_HSS0/Class3/1/HDMap_train_test/1_TSA_DIF_train.mat',
            'save_dir': f'datasets/{base_folder}/{folder_name}/domain_B/train/good',
            'slice_from': 0,
            'slice_to': N_training,
            'description': 'Domain-B 정상 훈련 데이터'
        },
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Normal/Normal2_LSSm0.3_HSS0/Class3/1/HDMap_train_test/1_TSA_DIF_test.mat',
            'save_dir': f'datasets/{base_folder}/{folder_name}/domain_B/test/good',
            'slice_from': 0,
            'slice_to': N_testing,
            'description': 'Domain-B 정상 테스트 데이터'
        },
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Planet_fault_ring/1.42_LSSm0.3_HSS0/Class3/1/HDMap_train_test/1_TSA_DIF_test.mat',
            'save_dir': f'datasets/{base_folder}/{folder_name}/domain_B/test/fault',
            'slice_from': 0,
            'slice_to': N_testing,
            'description': 'Domain-B 결함 테스트 데이터'
        },

        # Domain-C: Class1의 3번 센서 데이터
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Normal/Normal2_LSSm0.3_HSS0/Class1/1/HDMap_train_test/3_TSA_DIF_train.mat',
            'save_dir': f'datasets/{base_folder}/{folder_name}/domain_C/train/good',
            'slice_from': 0,
            'slice_to': N_training,
            'description': 'Domain-C 정상 훈련 데이터'
        },
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Normal/Normal2_LSSm0.3_HSS0/Class1/1/HDMap_train_test/3_TSA_DIF_test.mat',
            'save_dir': f'datasets/{base_folder}/{folder_name}/domain_C/test/good',
            'slice_from': 0,
            'slice_to': N_testing,
            'description': 'Domain-C 정상 테스트 데이터'
        },
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Planet_fault_ring/1.42_LSSm0.3_HSS0/Class1/1/HDMap_train_test/3_TSA_DIF_test.mat',
            'save_dir': f'datasets/{base_folder}/{folder_name}/domain_C/test/fault',
            'slice_from': 0,
            'slice_to': N_testing,
            'description': 'Domain-C 결함 테스트 데이터'
        },
        
        # Domain-D: Class3의 3번 센서 데이터
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Normal/Normal2_LSSm0.3_HSS0/Class3/1/HDMap_train_test/3_TSA_DIF_train.mat',
            'save_dir': f'datasets/{base_folder}/{folder_name}/domain_D/train/good',
            'slice_from': 0,
            'slice_to': N_training,
            'description': 'Domain-D 정상 훈련 데이터'
        },
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Normal/Normal2_LSSm0.3_HSS0/Class3/1/HDMap_train_test/3_TSA_DIF_test.mat',
            'save_dir': f'datasets/{base_folder}/{folder_name}/domain_D/test/good',
            'slice_from': 0,
            'slice_to': N_testing,
            'description': 'Domain-D 정상 테스트 데이터'
        },
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Planet_fault_ring/1.42_LSSm0.3_HSS0/Class3/1/HDMap_train_test/3_TSA_DIF_test.mat',
            'save_dir': f'datasets/{base_folder}/{folder_name}/domain_D/test/fault',
            'slice_from': 0,
            'slice_to': N_testing,
            'description': 'Domain-D 결함 테스트 데이터'
        },
    ]
    
    # =============================================================================
    # 데이터 변환 및 저장
    # =============================================================================
    print("="*80)
    print("HDMAP 데이터셋 변환 시작")
    print("="*80)
    print(f"훈련 샘플 수: {N_training}")
    print(f"테스트 샘플 수: {N_testing}")
    print(f"목표 이미지 크기: {target_size}")
    print(f"비트 심도: {bit_depth}")
    print(f"최상위 폴더: {base_folder}")
    print(f"원본 데이터 폴더: {folder_name}")
    print("="*80)
    
    # HDMAP 최상위 디렉토리 생성
    base_dir = Path("datasets") / base_folder
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        print(f"HDMAP 최상위 디렉토리 생성: {base_dir}")
    else:
        print(f"HDMAP 최상위 디렉토리 확인: {base_dir}")
    
    # 각 데이터 세트별로 처리
    for idx, item in enumerate(path_mapping, 1):
        print(f"\n[{idx}/{len(path_mapping)}] {item['description']} 처리 중...")
        
        # 다양한 전처리 버전의 저장 디렉토리 설정
        original_save_dir = item['save_dir']  # 원본 (크기 그대로)
        resize_pad_save_dir = item['save_dir'].replace(folder_name, f"{N_training}_{bit_depth}_resize_pad_{target_size[0]}x{target_size[1]}")  # 리사이즈+패딩
        pad_save_dir = item['save_dir'].replace(folder_name, f"{N_training}_{bit_depth}_pad_to_{target_size[0]}x{target_size[1]}")  # 패딩만
        resize_save_dir = item['save_dir'].replace(folder_name, f"{N_training}_{bit_depth}_resize_{target_size[0]}x{target_size[1]}")  # 리사이즈만
        
        # 3채널 버전 저장 디렉토리 설정
        original_3ch_save_dir = item['save_dir'].replace(folder_name, f"{N_training}_{bit_depth}_3ch_original")  # 3채널 원본
        resize_pad_3ch_save_dir = item['save_dir'].replace(folder_name, f"{N_training}_{bit_depth}_3ch_resize_pad_{target_size[0]}x{target_size[1]}")  # 3채널 리사이즈+패딩
        pad_3ch_save_dir = item['save_dir'].replace(folder_name, f"{N_training}_{bit_depth}_3ch_pad_to_{target_size[0]}x{target_size[1]}")  # 3채널 패딩만
        resize_3ch_save_dir = item['save_dir'].replace(folder_name, f"{N_training}_{bit_depth}_3ch_resize_{target_size[0]}x{target_size[1]}")  # 3채널 리사이즈만
        
        # 디렉토리 초기화 및 생성
        all_dirs = [original_save_dir, resize_pad_save_dir, pad_save_dir, resize_save_dir,
                   original_3ch_save_dir, resize_pad_3ch_save_dir, pad_3ch_save_dir, resize_3ch_save_dir]
        for dir_path in all_dirs:
            if os.path.exists(dir_path):
                print(f"  기존 디렉토리 삭제: {dir_path}")
                shutil.rmtree(dir_path)
            os.makedirs(dir_path, exist_ok=True)
        
        # mat 파일 존재 확인
        if not os.path.exists(item['mat_path']):
            print(f"  ⚠️ 경고: {item['mat_path']} 파일을 찾을 수 없습니다. 건너뜁니다.")
            continue
        
        # mat 파일 읽기
        try:
            mat_data = scipy.io.loadmat(item['mat_path'])
            image_data = mat_data['Xdata']  # 'Xdata' 키에서 이미지 데이터 추출
            print(f"  파일 경로: {item['mat_path']}")
            print(f"  이미지 데이터 shape: {image_data.shape}")
        except Exception as e:
            print(f"  ❌ 오류: {item['mat_path']} 읽기 실패 - {e}")
            continue
        
        # 데이터 범위 확인
        actual_samples = image_data.shape[3] if len(image_data.shape) > 3 else 1
        max_slice = min(item['slice_to'], actual_samples)
        
        print(f"  추출 범위: {item['slice_from']} ~ {max_slice-1} (총 {max_slice - item['slice_from']}개)")
        
        # 지정된 범위의 샘플에 대해 이미지 저장
        for i in range(item['slice_from'], max_slice):
            # 3차원 데이터에서 i번째 슬라이스 추출
            img = image_data[:, :, 0, i]
            
            # 파일명 생성 (6자리 숫자로 패딩)
            filename = f'{i:06d}.png'
            
            # Sobel 필터 적용
            sobel_x, sobel_y = apply_sobel_filter(img)
            
            # 1. 원본 이미지 저장 (1채널)
            original_path = os.path.join(original_save_dir, filename)
            save_image_with_depth(img, original_path, bit_depth)
            
            # 2. 리사이즈+패딩 버전 저장 (1채널)
            processed_resize_pad = process_image_resize_with_padding(img, target_size)
            resize_pad_path = os.path.join(resize_pad_save_dir, filename)
            save_image_with_depth(processed_resize_pad, resize_pad_path, bit_depth)
            
            # 3. 패딩만 버전 저장 (1채널)
            processed_pad = process_image_pad_only(img, target_size)
            pad_path = os.path.join(pad_save_dir, filename)
            save_image_with_depth(processed_pad, pad_path, bit_depth)
            
            # 4. 리사이즈만 버전 저장 (1채널)
            processed_resize = process_image_resize_only(img, target_size)
            resize_path = os.path.join(resize_save_dir, filename)
            save_image_with_depth(processed_resize, resize_path, bit_depth)
            
            # 5. 3채널 원본 저장
            img_3ch = create_3channel_image(img, sobel_x, sobel_y)
            original_3ch_path = os.path.join(original_3ch_save_dir, filename)
            img_3ch.save(original_3ch_path)
            
            # 6. 3채널 리사이즈+패딩 버전 저장
            processed_3ch = create_3channel_image(img, sobel_x, sobel_y)
            processed_3ch_resize_pad = processed_3ch.resize(
                (min(target_size[1], int(processed_3ch.width * min(target_size[0]/processed_3ch.height, target_size[1]/processed_3ch.width))),
                 min(target_size[0], int(processed_3ch.height * min(target_size[0]/processed_3ch.height, target_size[1]/processed_3ch.width)))),
                Image.Resampling.LANCZOS
            )
            pad_h = max(0, target_size[0] - processed_3ch_resize_pad.height)
            pad_w = max(0, target_size[1] - processed_3ch_resize_pad.width)
            padding = (pad_w//2, pad_h//2, (pad_w+1)//2, (pad_h+1)//2)
            processed_3ch_resize_pad = ImageOps.expand(processed_3ch_resize_pad, padding, fill=(0, 0, 0))
            resize_pad_3ch_path = os.path.join(resize_pad_3ch_save_dir, filename)
            processed_3ch_resize_pad.save(resize_pad_3ch_path)
            
            # 7. 3채널 패딩만 버전 저장
            processed_3ch_pad = create_3channel_image(img, sobel_x, sobel_y)
            pad_h = max(0, target_size[0] - processed_3ch_pad.height)
            pad_w = max(0, target_size[1] - processed_3ch_pad.width)
            padding = (pad_w//2, pad_h//2, (pad_w+1)//2, (pad_h+1)//2)
            processed_3ch_pad = ImageOps.expand(processed_3ch_pad, padding, fill=(0, 0, 0))
            pad_3ch_path = os.path.join(pad_3ch_save_dir, filename)
            processed_3ch_pad.save(pad_3ch_path)
            
            # 8. 3채널 리사이즈만 버전 저장
            processed_3ch_resize = create_3channel_image(img, sobel_x, sobel_y)
            processed_3ch_resize = processed_3ch_resize.resize(target_size, Image.Resampling.LANCZOS)
            resize_3ch_path = os.path.join(resize_3ch_save_dir, filename)
            processed_3ch_resize.save(resize_3ch_path)
            
            # 진행상황 출력 (매 1000개마다)
            if (i + 1) % 1000 == 0:
                print(f"    진행: {i + 1}/{max_slice}")
        
        print(f"  ✅ 완료: {max_slice - item['slice_from']}개 이미지 저장")
        print(f"    - 1채널 원본: {original_save_dir}")
        print(f"    - 1채널 리사이즈+패딩: {resize_pad_save_dir}")
        print(f"    - 3채널 원본: {original_3ch_save_dir}")
        print(f"    - 3채널 리사이즈+패딩: {resize_pad_3ch_save_dir}")
    
    print("\n" + "="*80)
    print("🎉 HDMAP 데이터셋 변환 완료!")
    print("="*80)
    print("생성된 데이터셋 구조:")
    print("datasets/")
    print(f"└── {base_folder}/")
    print("    # 1채널 버전:")
    print(f"    ├── {folder_name}/                                           # 원본 크기 (31x95)")
    print(f"    ├── {N_training}_{bit_depth}_resize_pad_{target_size[0]}x{target_size[1]}/      # 리사이즈+패딩 → {target_size[0]}x{target_size[1]} (권장)")
    print(f"    ├── {N_training}_{bit_depth}_pad_to_{target_size[0]}x{target_size[1]}/          # 원본 크기 유지 + 패딩 → {target_size[0]}x{target_size[1]}")
    print(f"    ├── {N_training}_{bit_depth}_resize_{target_size[0]}x{target_size[1]}/          # 강제 리사이즈 → {target_size[0]}x{target_size[1]}")
    print("    # 3채널 RGB 버전 (R=원본, G=Sobel X, B=Sobel Y):")
    print(f"    ├── {N_training}_{bit_depth}_3ch_original/                   # 원본 크기 (31x95)")
    print(f"    ├── {N_training}_{bit_depth}_3ch_resize_pad_{target_size[0]}x{target_size[1]}/  # 리사이즈+패딩 → {target_size[0]}x{target_size[1]} (권장)")
    print(f"    ├── {N_training}_{bit_depth}_3ch_pad_to_{target_size[0]}x{target_size[1]}/      # 원본 크기 유지 + 패딩 → {target_size[0]}x{target_size[1]}")
    print(f"    └── {N_training}_{bit_depth}_3ch_resize_{target_size[0]}x{target_size[1]}/      # 강제 리사이즈 → {target_size[0]}x{target_size[1]}")
    print("\n각 폴더 구조:")
    print("└── domain_X/")
    print("    ├── train/good/     # 정상 훈련 데이터")
    print("    └── test/")
    print("        ├── good/       # 정상 테스트 데이터")
    print("        └── fault/      # 결함 테스트 데이터")


if __name__ == "__main__":
    prepare_hdmap_dataset()
