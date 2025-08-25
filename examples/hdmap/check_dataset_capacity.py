#!/usr/bin/env python3
"""HDMAP 데이터셋 용량 확인 스크립트.

이 스크립트는 각 데이터셋별로 사용 가능한 최대 샘플 수를 확인합니다.
N_train, N_test 설정 시 참고할 수 있는 정보를 제공합니다.
"""

import os
import scipy.io
from pathlib import Path


def check_dataset_capacity():
    """HDMAP 데이터셋별 사용 가능한 최대 샘플 수를 확인."""
    
    # 데이터 경로 매핑 정의 (prepare_hdmap_dataset.py와 동일)
    path_mapping = [
        # Domain-A: Class1의 1번 센서 데이터
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Normal/Normal2_LSSm0.3_HSS0/Class1/1/HDMap_train_test/1_TSA_DIF_train.mat',
            'description': 'Domain-A 정상 훈련 데이터'
        },
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Normal/Normal2_LSSm0.3_HSS0/Class1/1/HDMap_train_test/1_TSA_DIF_test.mat',
            'description': 'Domain-A 정상 테스트 데이터'
        },
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Planet_fault_ring/1.42_LSSm0.3_HSS0/Class1/1/HDMap_train_test/1_TSA_DIF_test.mat',
            'description': 'Domain-A 결함 테스트 데이터'
        },
        
        # Domain-B: Class3의 1번 센서 데이터
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Normal/Normal2_LSSm0.3_HSS0/Class3/1/HDMap_train_test/1_TSA_DIF_train.mat',
            'description': 'Domain-B 정상 훈련 데이터'
        },
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Normal/Normal2_LSSm0.3_HSS0/Class3/1/HDMap_train_test/1_TSA_DIF_test.mat',
            'description': 'Domain-B 정상 테스트 데이터'
        },
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Planet_fault_ring/1.42_LSSm0.3_HSS0/Class3/1/HDMap_train_test/1_TSA_DIF_test.mat',
            'description': 'Domain-B 결함 테스트 데이터'
        },

        # Domain-C: Class1의 3번 센서 데이터
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Normal/Normal2_LSSm0.3_HSS0/Class1/1/HDMap_train_test/3_TSA_DIF_train.mat',
            'description': 'Domain-C 정상 훈련 데이터'
        },
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Normal/Normal2_LSSm0.3_HSS0/Class1/1/HDMap_train_test/3_TSA_DIF_test.mat',
            'description': 'Domain-C 정상 테스트 데이터'
        },
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Planet_fault_ring/1.42_LSSm0.3_HSS0/Class1/1/HDMap_train_test/3_TSA_DIF_test.mat',
            'description': 'Domain-C 결함 테스트 데이터'
        },
        
        # Domain-D: Class3의 3번 센서 데이터
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Normal/Normal2_LSSm0.3_HSS0/Class3/1/HDMap_train_test/3_TSA_DIF_train.mat',
            'description': 'Domain-D 정상 훈련 데이터'
        },
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Normal/Normal2_LSSm0.3_HSS0/Class3/1/HDMap_train_test/3_TSA_DIF_test.mat',
            'description': 'Domain-D 정상 테스트 데이터'
        },
        {
            'mat_path': 'datasets/raw/KRISS_share_nipa2023/Planet_fault_ring/1.42_LSSm0.3_HSS0/Class3/1/HDMap_train_test/3_TSA_DIF_test.mat',
            'description': 'Domain-D 결함 테스트 데이터'
        },
    ]
    
    print("="*80)
    print("📊 HDMAP 데이터셋 용량 확인")
    print("="*80)
    
    dataset_info = {}
    
    # 각 데이터셋별로 확인
    for idx, item in enumerate(path_mapping, 1):
        print(f"\n[{idx}/{len(path_mapping)}] {item['description']} 확인 중...")
        
        # mat 파일 존재 확인
        if not os.path.exists(item['mat_path']):
            print(f"  ❌ 파일 없음: {item['mat_path']}")
            dataset_info[item['description']] = {
                'max_samples': 0,
                'file_exists': False,
                'file_path': item['mat_path']
            }
            continue
        
        # mat 파일 읽기
        try:
            mat_data = scipy.io.loadmat(item['mat_path'])
            image_data = mat_data['Xdata']
            
            # 데이터 차원에 따른 샘플 수 계산
            if len(image_data.shape) == 4:
                max_samples = image_data.shape[3]  # 4차원: (height, width, channels, samples)
            elif len(image_data.shape) == 3:
                max_samples = image_data.shape[2]  # 3차원: (height, width, samples)
            else:
                max_samples = 1  # 2차원: 단일 이미지
            
            print(f"  ✅ 파일 경로: {item['mat_path']}")
            print(f"  📐 데이터 shape: {image_data.shape}")
            print(f"  📊 최대 샘플 수: {max_samples:,}개")
            
            dataset_info[item['description']] = {
                'max_samples': max_samples,
                'file_exists': True,
                'file_path': item['mat_path'],
                'data_shape': image_data.shape
            }
            
        except Exception as e:
            print(f"  ❌ 읽기 실패: {e}")
            dataset_info[item['description']] = {
                'max_samples': 0,
                'file_exists': False,
                'file_path': item['mat_path'],
                'error': str(e)
            }
    
    # 요약 정보 출력
    print("\n" + "="*80)
    print("📋 데이터셋 용량 요약")
    print("="*80)
    
    # 도메인별로 그룹화
    domain_groups = {}
    for desc, info in dataset_info.items():
        if info['file_exists']:
            # 도메인 추출 (예: "Domain-A 정상 훈련 데이터" -> "Domain-A")
            domain = desc.split()[0]
            if domain not in domain_groups:
                domain_groups[domain] = {}
            
            # 데이터 타입 추출 (예: "Domain-A 정상 훈련 데이터" -> "정상 훈련")
            data_type = ' '.join(desc.split()[1:])
            domain_groups[domain][data_type] = info['max_samples']
    
    # 도메인별 요약 출력
    for domain, data_types in domain_groups.items():
        print(f"\n{domain}:")
        for data_type, max_samples in data_types.items():
            print(f"  {data_type}: {max_samples:,}개")
    
    # 전체 최대값 계산
    all_samples = [info['max_samples'] for info in dataset_info.values() if info['file_exists']]
    if all_samples:
        min_samples = min(all_samples)
        max_samples = max(all_samples)
        print(f"\n📈 전체 통계:")
        print(f"  최소 샘플 수: {min_samples:,}개")
        print(f"  최대 샘플 수: {max_samples:,}개")
        print(f"  권장 N_train: {min_samples:,}개 (모든 데이터셋에서 사용 가능)")
        print(f"  권장 N_test: {min(min_samples//10, 100):,}개 (전체의 10% 이하)")
    
    # 도메인별 최소값 계산
    print(f"\n🏷️  도메인별 권장 설정:")
    for domain, data_types in domain_groups.items():
        domain_min = min(data_types.values())
        print(f"  {domain}: N_train ≤ {domain_min:,}개, N_test ≤ {min(domain_min//10, 100):,}개")
    
    return dataset_info


if __name__ == "__main__":
    check_dataset_capacity()
