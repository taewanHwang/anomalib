"""MultiDomain HDMAP DataModule / 멀티 도메인 HDMAP 데이터모듈.

이 모듈은 여러 도메인을 동시에 처리하는 HDMAP 데이터모듈을 제공합니다.
도메인 전이 학습(Domain Transfer Learning)을 위해 설계되었습니다.
This module provides HDMAP data module that handles multiple domains simultaneously.
Designed for Domain Transfer Learning scenarios.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from anomalib.data.datamodules.base import AnomalibDataModule
from anomalib.data.datasets.image.hdmap import HDMAPDataset
from anomalib.data.utils import Split, ValSplitMode


class MultiDomainHDMAPDataModule(AnomalibDataModule):
    """MultiDomain HDMAP Datamodule / 멀티 도메인 HDMAP 데이터모듈.

    **도메인 전이 학습용**: 이 클래스는 하나의 소스 도메인에서 모델을 훈련하고,
    여러 타겟 도메인에서 성능을 평가하는 도메인 전이 학습을 지원합니다.
    **For Domain Transfer Learning**: This class supports domain transfer learning where
    a model is trained on one source domain and evaluated on multiple target domains.

    **데이터 분할 전략 / Data Split Strategy**:
    - Train: 소스 도메인의 train 데이터 (정상 데이터만) / Source domain train data (normal only)
    - Validation: 소스 도메인의 test 데이터 (정상+이상) / Source domain test data (normal+anomalous) 
    - Test: 타겟 도메인들의 test 데이터 / Target domains test data

    **상속 관계**: MultiDomainHDMAPDataModule ← AnomalibDataModule ← LightningDataModule
    **Inheritance**: MultiDomainHDMAPDataModule ← AnomalibDataModule ← LightningDataModule

    Args:
        root (Path | str): Path to the root folder containing HDMAP dataset.
            Defaults to "./datasets/HDMAP/1000_8bit_resize_pad_256x256".
            / HDMAP 데이터셋을 포함하는 루트 폴더 경로.
        source_domain (str): Source domain name for training (e.g., "domain_A").
            Defaults to "domain_A".
            / 훈련용 소스 도메인 이름 (예: "domain_A").
        target_domains (list[str] | str): List of target domain names for test.
            If "auto", automatically uses all domains except source_domain.
            Defaults to "auto".
            / 평가용 타겟 도메인 이름 리스트. "auto"이면 source를 제외한 모든 도메인 자동 사용.
        validation_strategy (str): Validation data strategy. Currently only supports "source_test".
            Defaults to "source_test".
            / 검증 데이터 전략. 현재는 "source_test"만 지원.
        train_batch_size (int): Training batch size. Defaults to 32.
            / 훈련 배치 크기.
        eval_batch_size (int): Evaluation batch size for validation and test. Defaults to 32.
            / 검증 및 테스트 배치 크기.
        num_workers (int): Number of workers for data loading. Defaults to 8.
            / 데이터 로딩 워커 수.

    Examples:
        기본 사용법 / Basic usage:

        >>> from anomalib.data.datamodules.image import MultiDomainHDMAPDataModule
        >>> # 수동으로 타겟 도메인 지정 / Manually specify target domains
        >>> datamodule = MultiDomainHDMAPDataModule(
        ...     source_domain="domain_A",
        ...     target_domains=["domain_B", "domain_C"]
        ... )
        >>> 
        >>> # 자동으로 타겟 도메인 설정 (source 제외한 나머지) / Auto target domains (all except source)
        >>> datamodule = MultiDomainHDMAPDataModule(
        ...     source_domain="domain_A",
        ...     target_domains="auto"
        ... )  # target_domains는 자동으로 ["domain_B", "domain_C", "domain_D"]가 됨
        >>> datamodule.setup()
        >>> 
        >>> # 훈련용 데이터로더 / Training dataloader
        >>> train_loader = datamodule.train_dataloader()
        >>> 
        >>> # 검증용 데이터로더 (소스 도메인 test) / Validation dataloader (source domain test)
        >>> val_loader = datamodule.val_dataloader()
        >>> 
        >>> # 테스트용 데이터로더들 (타겟 도메인들) / Test dataloaders (target domains)
        >>> test_loaders = datamodule.test_dataloader()
    """

    def __init__(
        self,
        root: Path | str = "./datasets/HDMAP/1000_8bit_resize_pad_256x256",
        source_domain: str = "domain_A",
        target_domains: list[str] | str = "auto",
        validation_strategy: str = "source_test",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        **kwargs: Any,
    ) -> None:
        """Initialize MultiDomain HDMAP DataModule / 멀티 도메인 HDMAP 데이터모듈 초기화."""
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            val_split_mode=ValSplitMode.NONE,  # validation 별도 관리 - 소스 도메인 test를 validation으로 직접 사용
            # val_split_mode=NONE으로 설정하는 이유:
            # 1. 기본 AnomalibDataModule의 validation split 로직을 비활성화
            # 2. 대신 소스 도메인의 test 데이터를 validation으로 직접 사용
            # 3. 이상 탐지에서 train 데이터는 정상만 있어서 split하면 불균형 validation 발생
            # 4. MultiDomain 환경에서는 커스텀 validation 전략 필요
            **kwargs,
        )

        self.root = Path(root)
        self.source_domain = source_domain
        
        # target_domains 자동 설정: "auto"이면 전체 도메인에서 source 제외
        # Automatic target_domains setting: exclude source from all domains if "auto"
        if target_domains == "auto":
            all_domains = ["domain_A", "domain_B", "domain_C", "domain_D"]
            self.target_domains = [d for d in all_domains if d != source_domain]
        else:
            self.target_domains = target_domains
            
        self.validation_strategy = validation_strategy

        # 지원되는 validation 전략 확인 / Check supported validation strategies
        if validation_strategy != "source_test":
            msg = f"Unsupported validation_strategy: {validation_strategy}. Only 'source_test' is supported."
            raise ValueError(msg)

        # 소스 도메인이 타겟 도메인에 포함되지 않도록 확인
        # Ensure source domain is not included in target domains
        if self.source_domain in self.target_domains:
            msg = f"Source domain '{self.source_domain}' cannot be in target_domains {self.target_domains}"
            raise ValueError(msg)

    def setup(self, stage: str | None = None) -> None:
        """Set up the datamodule / 데이터모듈 설정.
        
        **중요**: 부모 클래스의 setup() 메서드를 완전히 오버라이드합니다.
        부모의 _create_test_split() 등의 로직을 건너뛰고 멀티 도메인 전용 로직을 사용합니다.
        **Important**: Completely overrides parent's setup() method.
        Skips parent's _create_test_split() etc. and uses multi-domain specific logic.
        """
        self._setup(stage)

    def _setup(self, _stage: str | None = None) -> None:
        """Set up datasets for each domain / 각 도메인별 데이터셋 설정.

        **핵심 로직**: 소스 도메인에서는 train/validation을 모두 생성하고,
        타겟 도메인들에서는 test만 생성합니다.
        **Core Logic**: Create both train/validation for source domain,
        and only test for target domains.
        """
        
        # 소스 도메인 훈련 데이터 (정상 데이터만)
        # Source domain training data (normal data only)
        self.train_data = HDMAPDataset(
            root=self.root,
            domain=self.source_domain,
            split=Split.TRAIN,
        )
        
        # 소스 도메인 검증 데이터 (정상+이상 데이터)
        # Source domain validation data (normal+anomalous data)
        self.val_data = HDMAPDataset(
            root=self.root,
            domain=self.source_domain,
            split=Split.TEST,  # source test를 validation으로 사용
        )

        # 2. 타겟 도메인 테스트 데이터셋들 생성 / Create target domain test datasets
        self.test_data = []
        
        for target_domain in self.target_domains:
            target_test_data = HDMAPDataset(
                root=self.root,
                domain=target_domain,
                split=Split.TEST,
            )
            self.test_data.append(target_test_data)

    def prepare_data(self) -> None:
        """Prepare data if needed / 필요시 데이터 준비.
        
        **참고**: HDMAP 데이터는 이미 전처리되어 있으므로 별도 준비 작업이 필요하지 않습니다.
        **Note**: HDMAP data is already preprocessed, so no additional preparation is needed.
        """
        # HDMAP 데이터는 prepare_hdmap_dataset.py로 이미 준비됨
        # HDMAP data is already prepared by prepare_hdmap_dataset.py

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Create training dataloader from source domain / 소스 도메인에서 훈련 데이터로더 생성.
        
        **핵심**: 소스 도메인의 train 데이터만 사용합니다 (정상 데이터만 포함).
        **Key**: Only uses source domain train data (contains normal data only).
        
        Returns:
            DataLoader: Source domain training dataloader / 소스 도메인 훈련 데이터로더.
        """
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.train_batch_size,
            shuffle=True,  # 훈련시에는 셔플 적용
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.external_collate_fn or self.train_data.collate_fn,  # Anomalib 커스텀 collate 함수
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Create validation dataloader from source domain test data / 소스 도메인 테스트 데이터에서 검증 데이터로더 생성.
        
        **핵심**: 소스 도메인의 test 데이터를 validation으로 사용합니다 (정상+이상 데이터 포함).
        이는 이상 탐지에서 올바른 검증을 위해 필요합니다.
        **Key**: Uses source domain test data as validation (contains normal+anomalous data).
        This is necessary for proper validation in anomaly detection.
        
        Returns:
            DataLoader: Source domain validation dataloader / 소스 도메인 검증 데이터로더.
        """
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.eval_batch_size,
            shuffle=False,  # 검증시에는 셔플 비적용
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.external_collate_fn or self.val_data.collate_fn,  # Anomalib 커스텀 collate 함수
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Create test dataloaders from target domains / 타겟 도메인들에서 테스트 데이터로더들 생성.
        
        **핵심**: 모든 타겟 도메인의 test 데이터로 데이터로더 리스트를 생성합니다.
        각 도메인별로 개별 평가가 가능합니다.
        **Key**: Creates list of dataloaders from all target domains test data.
        Enables individual evaluation for each domain.
                
        Returns:
            list[DataLoader]: List of target domain test dataloaders / 타겟 도메인 테스트 데이터로더 리스트.
        """
        test_dataloaders = []
        
        for i, target_domain in enumerate(self.target_domains):
            test_loader = DataLoader(
                dataset=self.test_data[i],
                batch_size=self.eval_batch_size,
                shuffle=False,  # 테스트시에는 셔플 비적용
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
                collate_fn=self.external_collate_fn or self.test_data[i].collate_fn,  # Anomalib 커스텀 collate 함수
            )
            test_dataloaders.append(test_loader)
            
        return test_dataloaders

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """Create prediction dataloaders / 예측용 데이터로더들 생성.
        
        **참고**: 예측은 테스트와 동일하게 타겟 도메인들을 사용합니다.
        **Note**: Prediction uses the same target domains as test.
        
        Returns:
            list[DataLoader]: Prediction dataloaders / 예측용 데이터로더들.
        """
        return self.test_dataloader()

    def __repr__(self) -> str:
        """String representation of the datamodule / 데이터모듈의 문자열 표현."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  root={self.root},\n"
            f"  source_domain={self.source_domain},\n"
            f"  target_domains={self.target_domains},\n"
            f"  validation_strategy={self.validation_strategy},\n"
            f"  train_batch_size={self.train_batch_size},\n"
            f"  eval_batch_size={self.eval_batch_size},\n"
            f"  num_workers={self.num_workers}\n"
            f")"
        )
