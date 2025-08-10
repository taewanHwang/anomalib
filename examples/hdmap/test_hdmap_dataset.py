#!/usr/bin/env python3
"""HDMAP Dataset 테스트 스크립트.

HDMAPDataset 클래스의 기본 동작을 확인하는 테스트 스크립트입니다.
도메인 전이 학습을 위한 기본 사용법을 보여줍니다.
"""

from pathlib import Path

from anomalib.data.datasets.image.hdmap import HDMAPDataset


def test_hdmap_dataset():
    """HDMAPDataset 기본 동작 테스트."""
    print("="*60)
    print("HDMAP Dataset 테스트 시작")
    print("="*60)
    
    # 데이터 경로 설정
    root_path = "./datasets/HDMAP/1000_8bit_resize_256x256"
    
    try:
        # Domain A 훈련 데이터 로드
        print("\n1. Domain A 훈련 데이터 로드 중...")
        dataset_A_train = HDMAPDataset(
            root=root_path,
            domain="domain_A",
            split="train"
        )
        
        print(f"✅ Dataset 이름: {dataset_A_train.name}")
        print(f"✅ 도메인: {dataset_A_train.domain}")
        print(f"✅ 분할: {dataset_A_train.split}")
        print(f"✅ 샘플 수: {len(dataset_A_train)}")
        print(f"✅ 작업 유형: {dataset_A_train.task}")
        
        # 샘플 데이터 확인
        print("\n2. 샘플 데이터 확인...")
        if len(dataset_A_train) > 0:
            sample = dataset_A_train[0]
            print(f"✅ 이미지 형태: {sample.image.shape}")
            print(f"✅ 라벨: {sample.gt_label}")
            print(f"✅ 이미지 경로: {sample.image_path}")
            print(f"✅ 마스크 경로: {sample.mask_path}")
        
        # Domain B 테스트 데이터 로드 (도메인 전이 학습용)
        print("\n3. Domain B 테스트 데이터 로드 중...")
        dataset_B_test = HDMAPDataset(
            root=root_path,
            domain="domain_B", 
            split="test"
        )
        
        print(f"✅ Domain B 샘플 수: {len(dataset_B_test)}")
        
        # 라벨 분포 확인
        print("\n4. Domain A 훈련 데이터 라벨 분포...")
        labels = [dataset_A_train[i].gt_label.item() for i in range(len(dataset_A_train))]
        good_count = labels.count(0)
        fault_count = labels.count(1)
        print(f"✅ 정상(good): {good_count}개")
        print(f"✅ 결함(fault): {fault_count}개")
        
        print("\n5. Domain B 테스트 데이터 라벨 분포...")
        labels_B = [dataset_B_test[i].gt_label.item() for i in range(len(dataset_B_test))]
        good_count_B = labels_B.count(0)
        fault_count_B = labels_B.count(1)
        print(f"✅ 정상(good): {good_count_B}개")
        print(f"✅ 결함(fault): {fault_count_B}개")
        
        # 샘플 DataFrame 확인
        print("\n6. Samples DataFrame 구조 확인...")
        print(f"✅ DataFrame 열: {list(dataset_A_train.samples.columns)}")
        print(f"✅ DataFrame 형태: {dataset_A_train.samples.shape}")
        
        # pandas 출력 옵션 설정 (잘림 방지)
        import pandas as pd
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 100)
        
        print("\n✅ 첫 3개 샘플 (전체 열 표시):")
        for i in range(min(3, len(dataset_A_train.samples))):
            sample_row = dataset_A_train.samples.iloc[i]
            print(f"\n--- 샘플 {i+1} ---")
            for col in dataset_A_train.samples.columns:
                print(f"  {col}: {sample_row[col]}")
        
        print("\n✅ 각 열별 샘플 값:")
        for col in dataset_A_train.samples.columns:
            print(f"  - {col}: {dataset_A_train.samples[col].iloc[0]}")
        
        # 데이터 타입 확인
        print(f"\n✅ 데이터 타입:")
        for col in dataset_A_train.samples.columns:
            print(f"  - {col}: {dataset_A_train.samples[col].dtype}")
        
        # 고유값 확인 (카테고리 열들)
        print(f"\n✅ 도메인 고유값: {dataset_A_train.samples['domain'].unique()}")
        print(f"✅ 분할 고유값: {dataset_A_train.samples['split'].unique()}")
        print(f"✅ 라벨 고유값: {dataset_A_train.samples['label'].unique()}")
        print(f"✅ 라벨 인덱스 고유값: {dataset_A_train.samples['label_index'].unique()}")
        
        print("\n" + "="*60)
        print("🎉 HDMAPDataset 테스트 완료!")
        print("✅ 모든 기능이 정상적으로 동작합니다.")
        print("✅ 도메인 전이 학습 준비 완료!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        print("💡 데이터 경로와 폴더 구조를 확인해주세요.")
        return False


def test_all_domains():
    """모든 도메인 로드 테스트."""
    print("\n" + "="*60)
    print("모든 도메인 로드 테스트")
    print("="*60)
    
    root_path = "./datasets/HDMAP/1000_8bit_resize_256x256"
    domains = ["domain_A", "domain_B", "domain_C", "domain_D"]
    
    for domain in domains:
        try:
            dataset = HDMAPDataset(
                root=root_path,
                domain=domain,
                split="train"
            )
            print(f"✅ {domain}: {len(dataset)}개 샘플")
        except Exception as e:
            print(f"❌ {domain}: {e}")


if __name__ == "__main__":
    # 기본 테스트 실행
    success = test_hdmap_dataset()
    
    if success:
        # 모든 도메인 테스트
        test_all_domains()
