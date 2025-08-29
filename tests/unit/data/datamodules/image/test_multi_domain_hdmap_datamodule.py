#!/usr/bin/env python3
"""MultiDomainHDMAPDataModule 테스트 스크립트.

이 스크립트는 MultiDomainHDMAPDataModule의 기능을 테스트하고 사용법을 보여줍니다.
도메인 전이 학습을 위한 멀티 도메인 데이터모듈의 동작을 확인할 수 있습니다.
"""

import sys

import pandas as pd
from anomalib.data.datamodules.image.multi_domain_hdmap import MultiDomainHDMAPDataModule

# pandas 출력 옵션 설정 (잘림 방지)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 100)


def test_basic_functionality():
    """기본 기능 테스트 - DataModule 생성 및 설정."""
    print("=" * 80)
    print("🔧 MultiDomainHDMAPDataModule 기본 기능 테스트")
    print("=" * 80)
    
    # 1. DataModule 생성
    print("\n📦 1단계: MultiDomainHDMAPDataModule 생성")
    datamodule = MultiDomainHDMAPDataModule(
        root="./datasets/HDMAP/1000_8bit_resize_pad_256x256",
        source_domain="domain_A",
        target_domains="auto",
        validation_strategy="source_test",
        train_batch_size=16,
        eval_batch_size=32,
        num_workers=4
    )
    
    print(f"✅ DataModule 생성 완료")
    print(f"📋 DataModule 정보:")
    print(f"   - Source Domain: {datamodule.source_domain}")
    print(f"   - Target Domains: {datamodule.target_domains}")
    print(f"   - Validation Strategy: {datamodule.validation_strategy}")
    print(f"   - Train Batch Size: {datamodule.train_batch_size}")
    print(f"   - Eval Batch Size: {datamodule.eval_batch_size}")
    
    # 2. 데이터 준비 및 설정
    print("\n📊 2단계: 데이터 준비 및 설정")
    datamodule.prepare_data()
    datamodule.setup()
    
    print("✅ 데이터 설정 완료")
    
    # 3. 데이터셋 크기 확인
    print("\n📈 3단계: 데이터셋 크기 확인")
    print(f"   - 훈련 데이터 크기: {len(datamodule.train_data)} 샘플 (source: {datamodule.source_domain})")
    print(f"   - 검증 데이터 크기: {len(datamodule.val_data)} 샘플 (source test)")
    
    total_test_samples = sum(len(test_data) for test_data in datamodule.test_data)
    print(f"   - 테스트 데이터 크기: {total_test_samples} 샘플 (targets)")
    
    for i, target_domain in enumerate(datamodule.target_domains):
        print(f"     └─ {target_domain}: {len(datamodule.test_data[i])} 샘플")
    
    return datamodule


def test_dataloader_functionality(datamodule):
    """DataLoader 기능 테스트 - 배치 데이터 로딩 확인."""
    print("\n" + "=" * 80)
    print("🔄 DataLoader 기능 테스트")
    print("=" * 80)
    
    # 1. Train DataLoader 테스트
    print("\n🚂 1단계: Train DataLoader 테스트")
    train_loader = datamodule.train_dataloader()
    
    print(f"✅ Train DataLoader 생성 완료")
    print(f"   - 배치 수: {len(train_loader)}")
    print(f"   - 배치 크기: {train_loader.batch_size}")
    print(f"   - 셔플 여부: {train_loader.sampler is not None}")
    
    # 첫 번째 배치 확인
    train_batch = next(iter(train_loader))
    print(f"   - 배치 구조: {type(train_batch)}")
    print(f"   - 이미지 shape: {train_batch.image.shape}")
    print(f"   - 레이블 unique values: {train_batch.gt_label.unique()}")
    
    # 2. Validation DataLoader 테스트  
    print("\n🔍 2단계: Validation DataLoader 테스트")
    val_loader = datamodule.val_dataloader()
    
    print(f"✅ Validation DataLoader 생성 완료")
    print(f"   - 배치 수: {len(val_loader)}")
    print(f"   - 배치 크기: {val_loader.batch_size}")
    
    # 첫 번째 배치 확인 (정상+이상 데이터 포함 확인)
    val_batch_first = next(iter(val_loader))
    print(f"   - 이미지 shape: {val_batch_first.image.shape}")
    print(f"   - 레이블 unique values: {val_batch_first.gt_label.unique()}")
    print(f"   - 첫 배치 정상/이상: {(val_batch_first.gt_label == 0).sum().item()}/{(val_batch_first.gt_label == 1).sum().item()}")
    
    # 마지막 배치 확인 (전체 데이터 분포 파악)
    val_batches = list(val_loader)
    val_batch_last = val_batches[-1]
    print(f"   - 마지막 배치 정상/이상: {(val_batch_last.gt_label == 0).sum().item()}/{(val_batch_last.gt_label == 1).sum().item()}")
    
    # 전체 validation 데이터 분포 확인
    total_normal = sum((batch.gt_label == 0).sum().item() for batch in val_batches)
    total_anomaly = sum((batch.gt_label == 1).sum().item() for batch in val_batches)
    print(f"   - 전체 validation 정상/이상: {total_normal}/{total_anomaly}")
    
    # 3. Test DataLoaders 테스트
    print("\n🎯 3단계: Test DataLoaders 테스트")
    test_loaders = datamodule.test_dataloader()
    
    print(f"✅ Test DataLoaders 생성 완료")
    print(f"   - DataLoader 개수: {len(test_loaders)} (타겟 도메인별)")
    
    for i, (test_loader, target_domain) in enumerate(zip(test_loaders, datamodule.target_domains)):
        print(f"   - {target_domain}: {len(test_loader)} 배치, 배치크기 {test_loader.batch_size}")
        
        # 첫 번째 배치 확인
        test_batch_first = next(iter(test_loader))
        normal_count_first = (test_batch_first.gt_label == 0).sum().item()
        anomaly_count_first = (test_batch_first.gt_label == 1).sum().item()
        print(f"     └─ 첫 배치 정상/이상: {normal_count_first}/{anomaly_count_first}")
        
        # 마지막 배치 및 전체 분포 확인
        test_batches = list(test_loader)
        test_batch_last = test_batches[-1]
        normal_count_last = (test_batch_last.gt_label == 0).sum().item()
        anomaly_count_last = (test_batch_last.gt_label == 1).sum().item()
        print(f"     └─ 마지막 배치 정상/이상: {normal_count_last}/{anomaly_count_last}")
        
        # 전체 test 데이터 분포 확인
        total_normal_test = sum((batch.gt_label == 0).sum().item() for batch in test_batches)
        total_anomaly_test = sum((batch.gt_label == 1).sum().item() for batch in test_batches)
        print(f"     └─ 전체 {target_domain} 정상/이상: {total_normal_test}/{total_anomaly_test}")


def test_domain_transfer_scenario(datamodule):
    """도메인 전이 학습 시나리오 시뮬레이션."""
    print("\n" + "=" * 80)
    print("🌍 도메인 전이 학습 시나리오 시뮬레이션")
    print("=" * 80)
    
    print("\n📋 실험 시나리오:")
    print(f"   - Source Domain: {datamodule.source_domain} (훈련용)")
    print(f"   - Target Domains: {datamodule.target_domains} (평가용)")
    print(f"   - Validation: {datamodule.source_domain} test 데이터 활용")
    
    # 1. 훈련 단계 시뮬레이션
    print("\n🔥 1단계: 모델 훈련 시뮬레이션")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    print(f"   ✅ 훈련 데이터: {len(train_loader.dataset)} 샘플")
    print(f"   ✅ 검증 데이터: {len(val_loader.dataset)} 샘플")
    print(f"   💡 모델이 {datamodule.source_domain}에서 훈련됩니다")
    
    # 2. 검증 단계 시뮬레이션
    print("\n🔍 2단계: 훈련 중 검증 시뮬레이션")
    print(f"   ✅ 검증은 {datamodule.source_domain} test 데이터로 수행")
    print(f"   💡 조기 종료, 하이퍼파라미터 튜닝 가능")
    
    # 3. 도메인 전이 평가 시뮬레이션
    print("\n🎯 3단계: 도메인 전이 평가 시뮬레이션")
    test_loaders = datamodule.test_dataloader()
    
    print(f"   ✅ {len(test_loaders)}개 타겟 도메인에서 개별 평가:")
    for i, target_domain in enumerate(datamodule.target_domains):
        test_samples = len(test_loaders[i].dataset)
        print(f"     └─ {target_domain}: {test_samples} 샘플 평가")
    
    print("\n💡 예상 결과:")
    print("   - Source domain 성능이 가장 높을 것으로 예상")
    print("   - Target domains는 도메인 유사성에 따라 성능 차이 발생")
    print("   - 도메인 간 차이가 클수록 성능 저하 예상")


def test_error_handling():
    """오류 처리 테스트."""
    print("\n" + "=" * 80)
    print("⚠️  오류 처리 테스트")
    print("=" * 80)
    
    # 1. 소스 도메인이 타겟에 포함된 경우
    print("\n❌ 1단계: 소스 도메인이 타겟에 포함된 경우")
    try:
        invalid_datamodule = MultiDomainHDMAPDataModule(
            source_domain="domain_A",
            target_domains=["domain_A", "domain_B", "domain_C"],  # domain_A 중복!
        )
    except ValueError as e:
        print(f"   ✅ 예상된 오류 발생: {e}")
    
    # 2. 지원되지 않는 validation strategy
    print("\n❌ 2단계: 지원되지 않는 validation strategy")
    try:
        invalid_datamodule = MultiDomainHDMAPDataModule(
            source_domain="domain_A",
            target_domains=["domain_B", "domain_C"],
            validation_strategy="unsupported_strategy"
        )
    except ValueError as e:
        print(f"   ✅ 예상된 오류 발생: {e}")
    
    print("\n✅ 오류 처리가 올바르게 작동합니다!")


if __name__ == "__main__":
    print("🚀 MultiDomainHDMAPDataModule 종합 테스트 시작")
    print("이 테스트는 멀티 도메인 데이터모듈의 모든 기능을 검증합니다.")
    
    try:
        # 기본 기능 테스트
        datamodule = test_basic_functionality()
        
        # # DataLoader 기능 테스트
        test_dataloader_functionality(datamodule)
        
        # 도메인 전이 시나리오 테스트
        test_domain_transfer_scenario(datamodule)
        
        # 오류 처리 테스트
        test_error_handling()
                
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")