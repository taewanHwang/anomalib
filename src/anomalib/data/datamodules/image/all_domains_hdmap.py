#!/usr/bin/env python3
"""AllDomains HDMAP DataModule / 전체 도메인 통합 HDMAP 데이터모듈.

이 모듈은 모든 HDMAP 도메인(A, B, C, D)의 데이터를 통합하여 처리하는 DataModule을 제공합니다.
Multi-class Unified Model Anomaly Detection을 위해 설계되었습니다.

This module provides a DataModule that processes all HDMAP domains (A, B, C, D) in an integrated manner.
Designed for Multi-class Unified Model Anomaly Detection.

사용 예시 / Usage Example:
    >>> from anomalib.data.datamodules.image import AllDomainsHDMAPDataModule
    >>> 
    >>> # 모든 도메인 통합 학습 / Integrated training on all domains
    >>> datamodule = AllDomainsHDMAPDataModule(
    ...     root="./datasets/HDMAP/1000_8bit_resize_224x224",
    ...     train_batch_size=32,
    ...     eval_batch_size=32,
    ...     val_split_ratio=0.2  # test에서 20% validation 분할 (MVTec 방식)
    ... )
    >>> datamodule.setup()
    >>> 
    >>> # 통합된 데이터로더들 / Integrated dataloaders
    >>> train_loader = datamodule.train_dataloader()  # 모든 도메인의 train 데이터
    >>> val_loader = datamodule.val_dataloader()      # test에서 분할한 validation 데이터 (MVTec 방식)
    >>> test_loader = datamodule.test_dataloader()    # 모든 도메인의 test 데이터
    >>> 
    >>> # 모델 학습 / Model training
    >>> trainer.fit(model, datamodule)
    >>> trainer.test(model, datamodule)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import ConcatDataset

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.hdmap import HDMAPDataset, make_hdmap_dataset, DOMAINS
from anomalib.data.utils import TestSplitMode, ValSplitMode


def make_all_domains_hdmap_dataset(
    root: str | Path,
    split: str | None = None,
    domains: list[str] | None = None,
) -> pd.DataFrame:
    """Create HDMAP samples from all specified domains / 지정된 모든 도메인에서 HDMAP 샘플 생성.
    
    기존 make_hdmap_dataset 함수를 확장하여 여러 도메인의 데이터를 통합합니다.
    Extends the existing make_hdmap_dataset function to integrate data from multiple domains.
    
    Args:
        root (str | Path): Path to dataset root directory / 데이터셋 루트 디렉토리 경로
        split (str | None, optional): Dataset split (train or test). 
            If None, includes both splits. / 데이터셋 분할 (train 또는 test). None이면 모든 분할 포함.
        domains (list[str] | None, optional): List of domains to include.
            If None, includes all domains (A, B, C, D). / 포함할 도메인 리스트. None이면 모든 도메인 포함.
    
    Returns:
        pd.DataFrame: Integrated dataset samples from all domains / 모든 도메인에서 통합된 데이터셋 샘플
        
    Example:
        >>> # 모든 도메인의 train 데이터 / Train data from all domains
        >>> train_samples = make_all_domains_hdmap_dataset(root, split="train")
        >>> len(train_samples)  # 4000 (1000 * 4 domains)
        >>> 
        >>> # 특정 도메인들의 test 데이터 / Test data from specific domains
        >>> test_samples = make_all_domains_hdmap_dataset(
        ...     root, split="test", domains=["domain_A", "domain_B"]
        ... )
    """
    if domains is None:
        domains = list(DOMAINS)  # ["domain_A", "domain_B", "domain_C", "domain_D"]
    
    all_samples = []
    
    for domain in domains:
        try:
            # 기존 make_hdmap_dataset 함수 재사용 / Reuse existing make_hdmap_dataset function
            domain_samples = make_hdmap_dataset(
                root=root,
                domain=domain,
                split=split,
            )
            all_samples.append(domain_samples)
            
        except (RuntimeError, ValueError) as e:
            # 오류 발생 시 해당 도메인은 건너뛰기
            continue
    
    if not all_samples:
        raise RuntimeError(f"No valid samples found in any domain. Root: {root}, Split: {split}")
    
    # 모든 도메인의 샘플을 하나로 통합 / Integrate samples from all domains
    integrated_samples = pd.concat(all_samples, ignore_index=True)
    
    return integrated_samples


class AllDomainsHDMAPDataset(HDMAPDataset):
    """All Domains HDMAP Dataset / 전체 도메인 통합 HDMAP 데이터셋.
    
    기존 HDMAPDataset을 확장하여 모든 도메인의 데이터를 통합 처리합니다.
    Extends the existing HDMAPDataset to handle integrated data from all domains.
    
    **상속 관계**: AllDomainsHDMAPDataset ← HDMAPDataset ← AnomalibDataset
    **Inheritance**: AllDomainsHDMAPDataset ← HDMAPDataset ← AnomalibDataset
    
    **핵심 차이점**: 단일 도메인 대신 모든 도메인의 데이터를 통합
    **Key Difference**: Integrates data from all domains instead of a single domain
    """
    
    def __init__(
        self,
        root: Path | str = "./datasets/HDMAP/1000_8bit_resize_224x224",
        domains: list[str] | None = None,
        split: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize AllDomains HDMAP Dataset / 전체 도메인 통합 HDMAP 데이터셋 초기화.
        
        Args:
            root (Path | str): Path to dataset root directory.
            domains (list[str] | None): List of domains to include. 
                If None, includes all domains.
            split (str | None): Dataset split (train/test). 
                If None, includes both splits.
            **kwargs: Additional arguments passed to parent class.
        """
        # 부모 클래스 초기화를 건너뛰고 직접 AnomalibDataset 초기화
        # Skip HDMAPDataset.__init__ and directly initialize AnomalibDataset
        from anomalib.data.datasets.base.image import AnomalibDataset
        AnomalibDataset.__init__(self, **kwargs)
        
        self.root = Path(root)
        self.domains = domains or list(DOMAINS)
        self.split = split
        
        # 통합된 샘플 생성 / Create integrated samples
        self.samples = make_all_domains_hdmap_dataset(
            root=self.root,
            split=self.split,
            domains=self.domains,
        )


class AllDomainsHDMAPDataModule(AnomalibDataModule):
    """All Domains HDMAP DataModule / 전체 도메인 통합 HDMAP 데이터모듈.
    
    **목적**: Multi-class Unified Model Anomaly Detection을 위해 모든 HDMAP 도메인의 데이터를 통합 처리
    **Purpose**: Integrate all HDMAP domains for Multi-class Unified Model Anomaly Detection
    
    **데이터 분할 전략 / Data Split Strategy**:
    - Train: 모든 도메인(A~D)의 train 데이터 통합 / Integrate train data from all domains (A~D)
    - Validation: Train 데이터에서 분할 생성 / Split from train data  
    - Test: 모든 도메인(A~D)의 test 데이터 통합 / Integrate test data from all domains (A~D)
    
    **기존과의 차이점 / Differences from existing modules**:
    - HDMAPDataModule: 단일 도메인만 처리 / Processes only single domain
    - MultiDomainHDMAPDataModule: 도메인 전이 학습 (소스→타겟) / Domain transfer learning (source→target)
    - AllDomainsHDMAPDataModule: 모든 도메인 통합 학습 / Unified learning across all domains
    
    Args:
        root (Path | str): Path to the root folder containing HDMAP dataset.
            Defaults to "./datasets/HDMAP/1000_8bit_resize_224x224".
            / HDMAP 데이터셋을 포함하는 루트 폴더 경로.
        domains (list[str] | None): List of domains to include. 
            If None, includes all domains ["domain_A", "domain_B", "domain_C", "domain_D"].
            Defaults to None. / 포함할 도메인 리스트. None이면 모든 도메인 포함.
        train_batch_size (int): Training batch size. Defaults to 32.
            / 훈련 배치 크기.
        eval_batch_size (int): Evaluation batch size. Defaults to 32.
            / 평가 배치 크기.
        num_workers (int): Number of workers for data loading. Defaults to 8.
            / 데이터 로딩 워커 수.
        val_split_mode (ValSplitMode): Validation split mode. 
            Defaults to ValSplitMode.FROM_TEST (test에서 validation 분할, MVTec 방식).
            / 검증 데이터 분할 방식. 기본값은 test에서 분할.
        val_split_ratio (float): Fraction of train data for validation. Defaults to 0.2.
            / 검증용 train 데이터 비율.
        
    Examples:
        기본 사용법 / Basic usage:
        
        >>> datamodule = AllDomainsHDMAPDataModule(
        ...     root="./datasets/HDMAP/1000_8bit_resize_224x224",
        ...     train_batch_size=32,
        ...     val_split_ratio=0.2
        ... )
        >>> datamodule.setup()
        >>> 
        >>> # 통합된 데이터로더 / Integrated dataloaders
        >>> train_loader = datamodule.train_dataloader()  # 모든 도메인 train (80%)
        >>> val_loader = datamodule.val_dataloader()      # 모든 도메인 test에서 분할 (20%, MVTec 방식)  
        >>> test_loader = datamodule.test_dataloader()    # 모든 도메인 test (100%)
        >>> 
        >>> print(f"Train samples: {len(datamodule.train_data)}")  # ~3200 (4000 * 0.8)
        >>> print(f"Val samples: {len(datamodule.val_data)}")      # ~800 (4000 * 0.2)
        >>> print(f"Test samples: {len(datamodule.test_data)}")    # 6400 (1600 * 4 domains)
        
        특정 도메인만 선택 / Select specific domains:
        
        >>> datamodule = AllDomainsHDMAPDataModule(
        ...     domains=["domain_A", "domain_B"],  # A, B 도메인만 사용
        ...     val_split_ratio=0.25
        ... )
    """
    
    def __init__(
        self,
        root: Path | str = "./datasets/HDMAP/1000_8bit_resize_224x224",
        domains: list[str] | None = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        val_split_mode: ValSplitMode | str = ValSplitMode.FROM_TEST,  # test에서 validation 분할 (MVTec 방식)
        val_split_ratio: float = 0.2,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,  # 별도 test 디렉토리 사용
        test_split_ratio: float = 0.2,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize All Domains HDMAP DataModule / 전체 도메인 통합 HDMAP 데이터모듈 초기화."""
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            val_split_mode=val_split_mode,  # train에서 validation 분할
            val_split_ratio=val_split_ratio,
            test_split_mode=test_split_mode,  # 별도 test 디렉토리 사용  
            test_split_ratio=test_split_ratio,
            seed=seed,
            **kwargs,
        )
        
        self.root = Path(root)
        self.domains = domains or list(DOMAINS)  # 기본값: 모든 도메인
        
        # 도메인 유효성 검사 / Validate domains
        for domain in self.domains:
            if domain not in DOMAINS:
                raise ValueError(f"Invalid domain '{domain}'. Available domains: {DOMAINS}")
    
    @property
    def name(self) -> str:
        """Get the name of the datamodule / 데이터모듈 이름 가져오기."""
        domains_str = "_".join(self.domains)
        return f"AllDomainsHDMAPDataModule_{domains_str}"
    
    @property  
    def category(self) -> str:
        """Get the category of the datamodule / 데이터모듈 카테고리 가져오기."""
        return f"all_domains_{len(self.domains)}"
    
    def _setup(self, _stage: str | None = None) -> None:
        """Set up datasets for all domains / 모든 도메인에 대한 데이터셋 설정.
        
        **핵심 로직**: 모든 지정된 도메인의 train/test 데이터를 통합하여 데이터셋을 생성합니다.
        **Core Logic**: Create datasets by integrating train/test data from all specified domains.
        """
        # 모든 도메인의 훈련 데이터 통합 / Integrate training data from all domains
        self.train_data = AllDomainsHDMAPDataset(
            root=self.root,
            domains=self.domains,
            split="train",
            augmentations=self.train_augmentations,
        )
        
        # 모든 도메인의 테스트 데이터 통합 / Integrate test data from all domains
        self.test_data = AllDomainsHDMAPDataset(
            root=self.root,
            domains=self.domains,
            split="test", 
            augmentations=self.test_augmentations,
        )
    
    def prepare_data(self) -> None:
        """Prepare data if needed / 필요시 데이터 준비.
        
        모든 지정된 도메인의 데이터 경로를 검증합니다.
        Validates data paths for all specified domains.
        """
        # 루트 경로 확인 / Check root path
        if not self.root.exists():
            msg = (
                f"HDMAP dataset not found at {self.root}. "
                f"Please run 'examples/hdmap/prepare_hdmap_dataset.py' first to prepare the data."
            )
            raise FileNotFoundError(msg)
        
        # 각 도메인별 경로 확인 / Check each domain path
        for domain in self.domains:
            domain_path = self.root / domain
            if not domain_path.exists():
                msg = (
                    f"Domain '{domain}' not found at {domain_path}. "
                    f"Available domains should be: {DOMAINS}"
                )
                raise FileNotFoundError(msg)
    
    def __repr__(self) -> str:
        """String representation of the datamodule / 데이터모듈의 문자열 표현.""" 
        return (
            f"{self.__class__.__name__}(\n"
            f"  root={self.root},\n"
            f"  domains={self.domains},\n"
            f"  train_batch_size={self.train_batch_size},\n"
            f"  eval_batch_size={self.eval_batch_size},\n"
            f"  num_workers={self.num_workers},\n"
            f"  val_split_mode={self.val_split_mode},\n"
            f"  val_split_ratio={self.val_split_ratio}\n"
            f")"
        )