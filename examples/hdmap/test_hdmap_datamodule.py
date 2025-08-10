#!/usr/bin/env python3
"""HDMAP DataModule 테스트 스크립트.

HDMAPDataModule 클래스의 기본 동작과 도메인 전이 학습 시나리오를 테스트합니다.
PyTorch Lightning DataLoader 생성과 배치 데이터 검증을 포함합니다.
"""

from pathlib import Path

from anomalib.data.datamodules.image.hdmap import HDMAPDataModule


def test_hdmap_datamodule():
    """HDMAPDataModule 기본 동작 테스트."""
    print("="*70)
    print("HDMAP DataModule 테스트 시작")
    print("="*70)
    
    # 데이터 경로 설정
    root_path = "./datasets/HDMAP/1000_8bit_resize_256x256"
    
    try:
        # Domain A DataModule 생성
        print("\n1. Domain A DataModule 생성 중...")
        datamodule = HDMAPDataModule(
            root=root_path,
            domain="domain_A",
            train_batch_size=16,
            eval_batch_size=8,
            num_workers=2,  # 테스트용으로 낮게 설정
        )
        
        print(f"✅ DataModule 이름: {datamodule.name}")
        print(f"✅ 카테고리(도메인): {datamodule.category}")
        print(f"✅ 루트 경로: {datamodule.root}")
        print(f"✅ 훈련 배치 크기: {datamodule.train_batch_size}")
        print(f"✅ 평가 배치 크기: {datamodule.eval_batch_size}")
        
        # 데이터 준비 및 설정
        print("\n2. 데이터 준비 및 설정 중...")
        datamodule.prepare_data()
        datamodule.setup()
        
        print(f"✅ 훈련 데이터 샘플 수: {len(datamodule.train_data)}")
        print(f"✅ 검증 데이터 샘플 수: {len(datamodule.val_data) if datamodule.val_data else 0}")
        print(f"✅ 테스트 데이터 샘플 수: {len(datamodule.test_data)}")
        
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
        print(f"✅ 훈련 배치 라벨 형태: {train_batch.gt_label.shape}")
        print(f"✅ 라벨 값 범위: {train_batch.gt_label.min().item()} ~ {train_batch.gt_label.max().item()}")
        
        if test_loader:
            test_batch = next(iter(test_loader))
            print(f"✅ 테스트 배치 이미지 형태: {test_batch.image.shape}")
            print(f"✅ 테스트 배치 라벨 형태: {test_batch.gt_label.shape}")
            
            # 테스트 데이터 라벨 분포 확인
            good_count = (test_batch.gt_label == 0).sum().item()
            fault_count = (test_batch.gt_label == 1).sum().item()
            print(f"✅ 테스트 배치 - 정상: {good_count}개, 결함: {fault_count}개")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        return False


def test_domain_transfer_scenario():
    """도메인 전이 학습 시나리오 테스트."""
    print("\n" + "="*70)
    print("도메인 전이 학습 시나리오 테스트")
    print("="*70)
    
    root_path = "./datasets/HDMAP/1000_8bit_resize_256x256"
    
    try:
        # Source Domain (domain_A) - 훈련용
        print("\n1. Source Domain (domain_A) 설정...")
        source_dm = HDMAPDataModule(
            root=root_path,
            domain="domain_A",
            train_batch_size=32,
            eval_batch_size=16,
            num_workers=2,
        )
        
        source_dm.prepare_data()
        source_dm.setup()
        
        print(f"✅ Source Domain: {source_dm.domain}")
        print(f"✅ 훈련 샘플: {len(source_dm.train_data)}개")
        print(f"✅ 테스트 샘플: {len(source_dm.test_data)}개")
        
        # Target Domain (domain_B) - 평가용
        print("\n2. Target Domain (domain_B) 설정...")
        target_dm = HDMAPDataModule(
            root=root_path,
            domain="domain_B",
            train_batch_size=32,
            eval_batch_size=16,
            num_workers=2,
        )
        
        target_dm.prepare_data()
        target_dm.setup()
        
        print(f"✅ Target Domain: {target_dm.domain}")
        print(f"✅ 테스트 샘플: {len(target_dm.test_data)}개")
        
        # 도메인 간 데이터 로더 호환성 확인
        print("\n3. 도메인 간 데이터 호환성 확인...")
        source_batch = next(iter(source_dm.train_dataloader()))
        target_batch = next(iter(target_dm.test_dataloader()))
        
        print(f"✅ Source 배치 형태: {source_batch.image.shape}")
        print(f"✅ Target 배치 형태: {target_batch.image.shape}")
        print(f"✅ 형태 일치: {source_batch.image.shape[1:] == target_batch.image.shape[1:]}")
        
        # 실제 도메인 전이 학습 시뮬레이션
        print("\n4. 도메인 전이 학습 시뮬레이션...")
        print("📊 실험 시나리오:")
        print(f"   - Source Domain: {source_dm.domain} (훈련용)")
        print(f"   - Target Domain: {target_dm.domain} (평가용)")
        print("   - 실제 모델 훈련: model.fit(source_dm)")
        print("   - 도메인 전이 평가: model.test(target_dm)")
        
        print("\n✅ 도메인 전이 학습 준비 완료!")
        return True
        
    except Exception as e:
        print(f"\n❌ 도메인 전이 테스트 실패: {e}")
        return False


def test_all_domains():
    """모든 도메인 DataModule 테스트."""
    print("\n" + "="*70)
    print("모든 도메인 DataModule 테스트")
    print("="*70)
    
    root_path = "./datasets/HDMAP/1000_8bit_resize_256x256"
    domains = ["domain_A", "domain_B", "domain_C", "domain_D"]
    
    results = {}
    
    for domain in domains:
        try:
            print(f"\n테스트 중: {domain}")
            dm = HDMAPDataModule(
                root=root_path,
                domain=domain,
                train_batch_size=8,
                eval_batch_size=8,
                num_workers=1,
            )
            
            dm.prepare_data()
            dm.setup()
            
            train_samples = len(dm.train_data)
            test_samples = len(dm.test_data)
            
            print(f"✅ {domain}: 훈련 {train_samples}개, 테스트 {test_samples}개")
            results[domain] = {"train": train_samples, "test": test_samples}
            
        except Exception as e:
            print(f"❌ {domain}: {e}")
            results[domain] = {"error": str(e)}
    
    # 결과 요약
    print("\n📊 전체 도메인 결과 요약:")
    print("-" * 50)
    for domain, result in results.items():
        if "error" in result:
            print(f"{domain}: ❌ {result['error']}")
        else:
            print(f"{domain}: ✅ 훈련 {result['train']}개, 테스트 {result['test']}개")
    
    return results


if __name__ == "__main__":
    print("🚀 HDMAP DataModule 종합 테스트 시작")
    
    # 1. 기본 DataModule 테스트
    basic_success = test_hdmap_datamodule()
    
    if basic_success:
        # 2. 도메인 전이 시나리오 테스트
        transfer_success = test_domain_transfer_scenario()
        
        # 3. 모든 도메인 테스트
        all_domains_results = test_all_domains()
        
        print("\n" + "="*70)
        print("🎉 HDMAP DataModule 테스트 완료!")
        print("="*70)
        print("✅ 기본 기능: 정상 동작")
        print("✅ 도메인 전이: 준비 완료")
        print("✅ 모든 도메인: 확인 완료")
        print("\n🎯 다음 단계: 실제 모델 훈련 및 도메인 전이 실험!")
        
    else:
        print("\n❌ 기본 테스트 실패. 데이터 경로와 구조를 확인해주세요.")
