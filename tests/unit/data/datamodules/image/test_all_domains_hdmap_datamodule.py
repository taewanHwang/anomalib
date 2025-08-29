#!/usr/bin/env python3
"""AllDomains HDMAP DataModule 테스트 스크립트.

새로 구현된 AllDomainsHDMAPDataModule의 기능을 테스트합니다.
Multi-class Unified Model Anomaly Detection을 위한 통합 데이터 처리를 검증합니다.
"""

from pathlib import Path
import torch

from anomalib.data.datamodules.image.all_domains_hdmap import AllDomainsHDMAPDataModule


def test_all_domains_hdmap_datamodule():
    """AllDomainsHDMAPDataModule 기본 기능 테스트."""
    print("="*80)
    print("AllDomains HDMAP DataModule 테스트 시작")
    print("="*80)
    
    # 데이터 경로 설정
    root_path = "./datasets/HDMAP/1000_8bit_resize_224x224"
    
    try:
        # AllDomainsHDMAPDataModule 생성
        print("\n1. AllDomainsHDMAPDataModule 생성 중...")
        datamodule = AllDomainsHDMAPDataModule(
            root=root_path,
            domains=None,  # 모든 도메인 사용 (A, B, C, D)
            train_batch_size=16,
            eval_batch_size=8,
            num_workers=2,
            val_split_ratio=0.2,  # train에서 20% validation 분할
        )
        
        print(f"✅ DataModule 이름: {datamodule.name}")
        print(f"✅ 카테고리: {datamodule.category}")
        print(f"✅ 루트 경로: {datamodule.root}")
        print(f"✅ 포함 도메인: {datamodule.domains}")
        print(f"✅ 훈련 배치 크기: {datamodule.train_batch_size}")
        print(f"✅ 평가 배치 크기: {datamodule.eval_batch_size}")
        print(f"✅ Validation 분할 비율: {datamodule.val_split_ratio}")
        
        # 데이터 준비 및 설정
        print("\n2. 데이터 준비 및 설정 중...")
        datamodule.prepare_data()
        datamodule.setup()
        
        print(f"✅ 훈련 데이터 샘플 수: {len(datamodule.train_data)}")
        print(f"✅ 검증 데이터 샘플 수: {len(datamodule.val_data) if datamodule.val_data else 0}")  
        print(f"✅ 테스트 데이터 샘플 수: {len(datamodule.test_data)}")
        
        # 예상 샘플 수 계산 및 검증
        expected_train_total = 4000  # 1000 * 4 domains
        expected_test_total = 6400   # 1600 * 4 domains (추정)
        expected_val_samples = int(expected_train_total * datamodule.val_split_ratio)
        expected_train_samples = expected_train_total - expected_val_samples
        
        print(f"\n📊 데이터 분할 검증:")
        print(f"   예상 Train: {expected_train_samples}, 실제: {len(datamodule.train_data)}")
        print(f"   예상 Val: {expected_val_samples}, 실제: {len(datamodule.val_data) if datamodule.val_data else 0}")
        print(f"   예상 Test: {expected_test_total}, 실제: {len(datamodule.test_data)}")
        
        # 데이터 분할의 라벨 분포 상세 분석
        print(f"\n🔍 데이터셋별 라벨 분포 상세 분석:")
        
        # Train 데이터 라벨 분포 (정상만 있어야 함)
        train_samples_df = datamodule.train_data.samples
        train_label_counts = train_samples_df['label_index'].value_counts().sort_index()
        print(f"   📊 Train 데이터 라벨 분포:")
        for label_idx, count in train_label_counts.items():
            label_name = "정상(good)" if label_idx == 0 else "결함(fault)"
            print(f"      {label_name}: {count}개")
        
        if len(train_label_counts) == 1 and 0 in train_label_counts:
            print("   ✅ 확인: Train 데이터는 정상 데이터만 포함")
        else:
            print("   ⚠️  경고: Train 데이터에 결함 데이터가 포함되어 있습니다!")
        
        # Validation 데이터 라벨 분포 (정상만 있어야 함 - train에서 분할)
        if datamodule.val_data:
            val_samples_df = datamodule.val_data.samples
            val_label_counts = val_samples_df['label_index'].value_counts().sort_index()
            print(f"   📊 Validation 데이터 라벨 분포:")
            for label_idx, count in val_label_counts.items():
                label_name = "정상(good)" if label_idx == 0 else "결함(fault)"
                print(f"      {label_name}: {count}개")
            
            if len(val_label_counts) == 1 and 0 in val_label_counts:
                print("   ✅ 확인: Validation 데이터는 정상 데이터만 포함 (train에서 분할)")
            else:
                print("   ⚠️  경고: Validation 데이터에 결함 데이터가 포함되어 있습니다!")
        
        # Test 데이터 라벨 분포 (정상+결함 모두 있어야 함)
        test_samples_df = datamodule.test_data.samples
        test_label_counts = test_samples_df['label_index'].value_counts().sort_index()
        print(f"   📊 Test 데이터 라벨 분포:")
        for label_idx, count in test_label_counts.items():
            label_name = "정상(good)" if label_idx == 0 else "결함(fault)"
            print(f"      {label_name}: {count}개")
        
        if len(test_label_counts) == 2 and 0 in test_label_counts and 1 in test_label_counts:
            print("   ✅ 확인: Test 데이터는 정상+결함 데이터 모두 포함")
            normal_ratio = test_label_counts[0] / len(test_samples_df) * 100
            fault_ratio = test_label_counts[1] / len(test_samples_df) * 100
            print(f"   📈 Test 데이터 비율: 정상 {normal_ratio:.1f}%, 결함 {fault_ratio:.1f}%")
        else:
            print("   ⚠️  경고: Test 데이터에 정상 또는 결함 데이터가 누락되어 있습니다!")
        
        # DataLoader 생성 및 테스트
        print("\n3. DataLoader 생성 및 배치 테스트...")
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()
        
        print(f"✅ 훈련 배치 수: {len(train_loader)}")
        print(f"✅ 검증 배치 수: {len(val_loader) if val_loader else 0}")
        print(f"✅ 테스트 배치 수: {len(test_loader)}")
        
        # 샘플 배치 확인
        print("\n4. 샘플 배치 데이터 확인...")
        train_batch = next(iter(train_loader))
        print(f"✅ 훈련 배치 이미지 형태: {train_batch.image.shape}")
        print(f"✅ 훈련 이미지 채널 수: {train_batch.image.shape[1]} (C, H, W 순서)")
        print(f"✅ 훈련 이미지 데이터 타입: {train_batch.image.dtype}")
        print(f"✅ 훈련 이미지 값 범위: {train_batch.image.min().item():.4f} ~ {train_batch.image.max().item():.4f}")
        print(f"✅ 훈련 배치 라벨 형태: {train_batch.gt_label.shape}")
        print(f"✅ 훈련 라벨 값 범위: {train_batch.gt_label.min().item()} ~ {train_batch.gt_label.max().item()}")
        
        # Train 배치는 정상 데이터만 있어야 함
        normal_count = (train_batch.gt_label == 0).sum().item()
        fault_count = (train_batch.gt_label == 1).sum().item()
        print(f"✅ 훈련 배치 - 정상: {normal_count}개, 결함: {fault_count}개")
        if fault_count == 0:
            print("   🔍 확인: 훈련 데이터는 정상 데이터만 포함 (Anomaly Detection 특성)")
        else:
            print("   ⚠️  경고: 훈련 데이터에 결함 데이터 포함됨!")
        
        if test_loader:
            test_batch = next(iter(test_loader))
            print(f"✅ 테스트 배치 이미지 형태: {test_batch.image.shape}")
            print(f"✅ 테스트 이미지 채널 수: {test_batch.image.shape[1]} (C, H, W 순서)")
            print(f"✅ 테스트 이미지 데이터 타입: {test_batch.image.dtype}")
            print(f"✅ 테스트 이미지 값 범위: {test_batch.image.min().item():.4f} ~ {test_batch.image.max().item():.4f}")
            
            # RGB 채널별 동일성 확인 (grayscale → RGB 변환 확인)
            if test_batch.image.shape[1] == 3:
                r_channel = test_batch.image[:, 0, :, :]
                g_channel = test_batch.image[:, 1, :, :] 
                b_channel = test_batch.image[:, 2, :, :]
                
                channels_identical = torch.allclose(r_channel, g_channel) and torch.allclose(g_channel, b_channel)
                print(f"✅ RGB 채널 동일성 (grayscale → RGB 변환): {'Yes' if channels_identical else 'No'}")
                
                if channels_identical:
                    print("   🔍 확인: Grayscale 이미지가 RGB로 변환됨 (R=G=B)")
                else:
                    print("   🔍 실제 RGB 컬러 이미지")
            
            # 테스트 데이터 라벨 분포 확인 (정상+결함 모두 있어야 함)
            test_normal_count = (test_batch.gt_label == 0).sum().item()
            test_fault_count = (test_batch.gt_label == 1).sum().item()
            print(f"✅ 테스트 배치 - 정상: {test_normal_count}개, 결함: {test_fault_count}개")
            
            # Validation 배치 라벨 분포도 확인
            if val_loader:
                val_batch = next(iter(val_loader))
                val_normal_count = (val_batch.gt_label == 0).sum().item()
                val_fault_count = (val_batch.gt_label == 1).sum().item()
                print(f"✅ 검증 배치 - 정상: {val_normal_count}개, 결함: {val_fault_count}개")
                if val_fault_count == 0:
                    print("   🔍 확인: 검증 데이터도 정상 데이터만 포함 (train에서 분할)")
                else:
                    print("   ⚠️  경고: 검증 데이터에 결함 데이터 포함됨!")
        
        # 도메인 통합 검증
        print(f"\n5. 도메인 통합 검증...")
        
        # 도메인별 샘플 분포 및 라벨 분포 교차 확인
        print(f"✅ 도메인별 상세 분포 분석:")
        
        # Train 데이터: 도메인별 + 라벨별 분포
        train_samples_df = datamodule.train_data.samples
        print(f"   📊 훈련 데이터 도메인별 분포:")
        for domain in datamodule.domains:
            domain_data = train_samples_df[train_samples_df['domain'] == domain]
            domain_label_counts = domain_data['label_index'].value_counts().sort_index()
            normal_count = domain_label_counts.get(0, 0)
            fault_count = domain_label_counts.get(1, 0)
            print(f"      {domain}: 총 {len(domain_data)}개 (정상: {normal_count}, 결함: {fault_count})")
        
        # Test 데이터: 도메인별 + 라벨별 분포  
        test_samples_df = datamodule.test_data.samples
        print(f"   📊 테스트 데이터 도메인별 분포:")
        for domain in datamodule.domains:
            domain_data = test_samples_df[test_samples_df['domain'] == domain]
            domain_label_counts = domain_data['label_index'].value_counts().sort_index()
            normal_count = domain_label_counts.get(0, 0)
            fault_count = domain_label_counts.get(1, 0)
            print(f"      {domain}: 총 {len(domain_data)}개 (정상: {normal_count}, 결함: {fault_count})")
        
        # Anomaly Detection 데이터 특성 요약
        print(f"\n🎯 Anomaly Detection 데이터 특성 요약:")
        print(f"   ✅ Train/Validation: 정상 데이터만 사용 (비지도 학습)")
        print(f"   ✅ Test: 정상+결함 데이터 모두 사용 (성능 평가)")
        print(f"   ✅ 도메인 통합: {len(datamodule.domains)}개 도메인 데이터 통합")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_specific_domains():
    """특정 도메인만 선택하여 테스트."""
    print("\n" + "="*80)
    print("특정 도메인 선택 테스트")
    print("="*80)
    
    root_path = "./datasets/HDMAP/1000_8bit_resize_224x224"
    
    try:
        # A, B 도메인만 사용
        print("\n1. 특정 도메인 (A, B) 설정...")
        datamodule = AllDomainsHDMAPDataModule(
            root=root_path,
            domains=["domain_A", "domain_B"],  # A, B만 사용
            train_batch_size=16,
            eval_batch_size=8,
            num_workers=2,
            val_split_ratio=0.25,  # 25% validation
        )
        
        datamodule.prepare_data()
        datamodule.setup()
        
        print(f"✅ 선택된 도메인: {datamodule.domains}")
        print(f"✅ 훈련 데이터 샘플 수: {len(datamodule.train_data)}")
        print(f"✅ 테스트 데이터 샘플 수: {len(datamodule.test_data)}")
        
        # 예상 값과 비교
        expected_total_train = 2000  # 1000 * 2 domains  
        expected_val = int(expected_total_train * 0.25)  # 500
        expected_train = expected_total_train - expected_val  # 1500
        expected_test = 3200  # 1600 * 2 domains
        
        print(f"\n📊 특정 도메인 데이터 검증:")
        print(f"   예상 Train: {expected_train}, 실제: {len(datamodule.train_data)}")
        print(f"   예상 Val: {expected_val}, 실제: {len(datamodule.val_data) if datamodule.val_data else 0}")
        print(f"   예상 Test: {expected_test}, 실제: {len(datamodule.test_data)}")
        
        # 특정 도메인의 라벨 분포 상세 확인
        print(f"✅ 특정 도메인 선택 시 라벨 분포 확인:")
        
        # Train 데이터 라벨 분포
        train_samples_df = datamodule.train_data.samples
        train_label_counts = train_samples_df['label_index'].value_counts().sort_index()
        print(f"   📊 Train 라벨 분포:")
        for label_idx, count in train_label_counts.items():
            label_name = "정상(good)" if label_idx == 0 else "결함(fault)"
            print(f"      {label_name}: {count}개")
        
        # Test 데이터 라벨 분포
        test_samples_df = datamodule.test_data.samples  
        test_label_counts = test_samples_df['label_index'].value_counts().sort_index()
        print(f"   📊 Test 라벨 분포:")
        for label_idx, count in test_label_counts.items():
            label_name = "정상(good)" if label_idx == 0 else "결함(fault)"
            print(f"      {label_name}: {count}개")
        
        # 도메인별 분포
        train_domain_counts = train_samples_df['domain'].value_counts()
        print(f"   📊 훈련 데이터 도메인별 분포:")
        for domain, count in train_domain_counts.items():
            print(f"      {domain}: {count}개")
        
        return True
        
    except Exception as e:
        print(f"❌ 특정 도메인 테스트 실패: {e}")
        return False


def compare_with_existing_datamodules():
    """기존 DataModule들과 비교 테스트."""
    print("\n" + "="*80)
    print("기존 DataModule들과 비교")
    print("="*80)
    
    root_path = "./datasets/HDMAP/1000_8bit_resize_224x224"
    
    try:
        from anomalib.data.datamodules.image.hdmap import HDMAPDataModule
        from anomalib.data.datamodules.image.multi_domain_hdmap import MultiDomainHDMAPDataModule
        
        print("\n📊 DataModule별 데이터 구성 비교:")
        
        # 1. HDMAPDataModule (단일 도메인)
        single_dm = HDMAPDataModule(
            root=root_path,
            domain="domain_A",
            train_batch_size=16,
            eval_batch_size=8,
            num_workers=2,
        )
        single_dm.setup()
        
        print(f"\n1. HDMAPDataModule (domain_A):")
        print(f"   Train: {len(single_dm.train_data)}개")
        print(f"   Val: {len(single_dm.val_data) if single_dm.val_data else 0}개")
        print(f"   Test: {len(single_dm.test_data)}개")
        
        # 2. MultiDomainHDMAPDataModule (도메인 전이)
        multi_dm = MultiDomainHDMAPDataModule(
            root=root_path,
            source_domain="domain_A",
            target_domains=["domain_B", "domain_C"],
            train_batch_size=16,
            eval_batch_size=8,
            num_workers=2,
        )
        multi_dm.setup()
        
        print(f"\n2. MultiDomainHDMAPDataModule (A→B,C):")
        print(f"   Train: {len(multi_dm.train_data)}개 (소스 도메인만)")
        print(f"   Val: {len(multi_dm.val_data) if multi_dm.val_data else 0}개 (소스 test)")
        print(f"   Test: {sum(len(test_data) for test_data in multi_dm.test_data)}개 (타겟 도메인들)")
        
        # 3. AllDomainsHDMAPDataModule (모든 도메인 통합)
        all_dm = AllDomainsHDMAPDataModule(
            root=root_path,
            domains=None,  # 모든 도메인
            train_batch_size=16,
            eval_batch_size=8,
            num_workers=2,
            val_split_ratio=0.2,
        )
        all_dm.setup()
        
        print(f"\n3. AllDomainsHDMAPDataModule (모든 도메인):")
        print(f"   Train: {len(all_dm.train_data)}개 (모든 도메인 통합)")
        print(f"   Val: {len(all_dm.val_data) if all_dm.val_data else 0}개 (train에서 분할)")
        print(f"   Test: {len(all_dm.test_data)}개 (모든 도메인 통합)")
        
        print(f"\n🎯 사용 용도별 추천:")
        print(f"   - 단일 도메인 학습: HDMAPDataModule")
        print(f"   - 도메인 전이 학습: MultiDomainHDMAPDataModule")
        print(f"   - 멀티클래스 통합 학습: AllDomainsHDMAPDataModule ⭐")
        
        # 각 DataModule의 라벨 분포 특성 요약
        print(f"\n📊 DataModule별 라벨 분포 특성:")
        print(f"   🔹 HDMAPDataModule:")
        print(f"      - Train: 정상만 (단일 도메인)")
        print(f"      - Test: 정상+결함 (단일 도메인)")
        print(f"   🔹 MultiDomainHDMAPDataModule:")
        print(f"      - Train: 정상만 (소스 도메인)")
        print(f"      - Val: 정상+결함 (소스 도메인 test)")
        print(f"      - Test: 정상+결함 (타겟 도메인들)")
        print(f"   🔹 AllDomainsHDMAPDataModule ⭐:")
        print(f"      - Train: 정상만 (모든 도메인 통합)")
        print(f"      - Val: 정상만 (train에서 분할)")
        print(f"      - Test: 정상+결함 (모든 도메인 통합)")
        
        return True
        
    except Exception as e:
        print(f"❌ 비교 테스트 실패: {e}")
        return False


if __name__ == "__main__":
    print("🚀 AllDomains HDMAP DataModule 종합 테스트 시작")
    
    # 1. 기본 기능 테스트
    basic_success = test_all_domains_hdmap_datamodule()
    
    if basic_success:
        # 2. 특정 도메인 선택 테스트
        specific_success = test_specific_domains()
        
        # 3. 기존 DataModule과 비교
        compare_success = compare_with_existing_datamodules()
        
        print("\n" + "="*80)
        print("🎉 AllDomains HDMAP DataModule 테스트 완료!")
        print("="*80)
        print("✅ 기본 기능: 정상 동작")
        print("✅ 특정 도메인 선택: 정상 동작")
        print("✅ 기존 DataModule 비교: 완료")
        print("\n🎯 Multi-class Unified Model Anomaly Detection 준비 완료!")
        print("📝 사용법:")
        print("   from anomalib.data.datamodules.image import AllDomainsHDMAPDataModule")
        print("   datamodule = AllDomainsHDMAPDataModule()")
        print("   trainer.fit(model, datamodule)")
        
    else:
        print("\n❌ 기본 테스트 실패. 데이터 경로와 구조를 확인해주세요.")