#!/usr/bin/env python3
"""HDMAP Data Module / HDMAP 데이터 모듈.

This module provides PyTorch Lightning DataModule implementation for the HDMAP dataset.
Supports domain transfer learning tasks with multiple industrial domains.

이 모듈은 HDMAP 데이터셋에 대한 PyTorch Lightning DataModule 구현을 제공합니다.
여러 산업 도메인을 통한 도메인 전이 학습 작업을 지원합니다.

Example usage for domain transfer learning:
도메인 전이 학습을 위한 사용 예시:

    >>> # Source domain for training / 훈련용 소스 도메인
    >>> source_dm = HDMAPDataModule(domain="domain_A", train_batch_size=32)
    >>> source_dm.setup()
    
    >>> # Target domain for evaluation / 평가용 타겟 도메인
    >>> target_dm = HDMAPDataModule(domain="domain_B", eval_batch_size=32)
    >>> target_dm.setup()
    
    >>> # Train on source, test on target / 소스에서 훈련, 타겟에서 테스트
    >>> trainer.fit(model, source_dm)
    >>> trainer.test(model, target_dm)
"""

from pathlib import Path
from typing import Any

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.hdmap import HDMAPDataset
from anomalib.data.utils import TestSplitMode, ValSplitMode


class HDMAPDataModule(AnomalibDataModule):
    """HDMAP Datamodule / HDMAP 데이터모듈.

    **상속 관계**: 이 클래스는 AnomalibDataModule을 상속받아 HDMAP 데이터셋에
    특화된 구현을 제공합니다. 추상 메서드 _setup()을 실제로 구현한 예시입니다.
    **Inheritance**: This class inherits from AnomalibDataModule to provide HDMAP dataset-specific
    implementation. This is an example of implementing the abstract method _setup().

    **도메인 전이 학습**: 각 도메인을 독립적으로 설정하여 source/target domain 실험을 지원합니다.
    **Domain Transfer Learning**: Supports source/target domain experiments by independently configuring each domain.

    Args:
        root (Path | str): Path to the root folder containing HDMAP dataset.
            Defaults to ``"./datasets/HDMAP/1000_8bit_resize_256x256"``.
            / HDMAP 데이터셋을 포함하는 루트 폴더 경로. 기본값은 전처리된 HDMAP 폴더
        domain (str): Domain name (domain_A, domain_B, domain_C, domain_D).
            Defaults to ``"domain_A"``.
            / 도메인 이름 (domain_A, domain_B, domain_C, domain_D). 기본값은 ``"domain_A"``
        train_batch_size (int, optional): Training batch size. Defaults to ``32``.
            / 훈련 배치 크기. 기본값은 ``32``
        eval_batch_size (int, optional): Test batch size. Defaults to ``32``.
            / 테스트 배치 크기. 기본값은 ``32``
        num_workers (int, optional): Number of workers. Defaults to ``8``.
            / 워커 수. 기본값은 ``8``
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
            / 검증 서브셋을 얻는 방법을 결정하는 설정
        val_split_ratio (float): Fraction of train or test images held out for validation.
            / 검증용으로 분리할 훈련 또는 테스트 이미지 비율
        test_split_mode (TestSplitMode | None): Setting that determines how the test subset is obtained.
            / 테스트 서브셋을 얻는 방법을 결정하는 설정
        test_split_ratio (float): Fraction of train images held out for testing.
            / 테스트용으로 분리할 훈련 이미지 비율
        seed (int | None, optional): Seed which may be set to a fixed value for reproducible experiments.
            / 재현 가능한 실험을 위해 고정값으로 설정할 수 있는 시드

    Example:
        Single domain usage:
        단일 도메인 사용법:

        >>> datamodule = HDMAPDataModule(
        ...     root="./datasets/HDMAP/1000_8bit_resize_256x256",
        ...     domain="domain_A",
        ...     train_batch_size=32,
        ...     eval_batch_size=32
        ... )
        >>> datamodule.setup()
        >>> train_loader = datamodule.train_dataloader()

        Domain transfer learning:
        도메인 전이 학습:

        >>> # Source domain training / 소스 도메인 훈련
        >>> source_dm = HDMAPDataModule(domain="domain_A")
        >>> source_dm.setup()
        
        >>> # Target domain evaluation / 타겟 도메인 평가
        >>> target_dm = HDMAPDataModule(domain="domain_B")
        >>> target_dm.setup()
        
        >>> # Cross-domain evaluation / 도메인 간 평가
        >>> model.fit(source_dm)
        >>> results = model.test(target_dm)
    """

    def __init__(
        self,
        root: Path | str = "./datasets/HDMAP/1000_8bit_resize_256x256",
        domain: str = "domain_A",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        val_split_mode: ValSplitMode | str = ValSplitMode.FROM_TEST,
        val_split_ratio: float = 0.2,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        # 부모 클래스 초기화 / Initialize parent class
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            seed=seed,
            **kwargs,
        )

        # HDMAP 특화 속성 설정 / Set HDMAP-specific attributes
        self.root = Path(root)
        self.domain = domain

        # HDMAP은 별도 테스트 디렉토리 제공, 기본값: 테스트에서 검증 분할 (MVTec 방식)
        # HDMAP provides separate test directory, default: split validation from test (MVTec style)
        # 테스트 데이터에서 일부를 검증용으로 분할하여 균형잡힌 평가
        # Split some test data for validation to ensure balanced evaluation

    @property
    def name(self) -> str:
        """Get the name of the datamodule / 데이터모듈 이름 가져오기.

        Returns:
            str: Name of the datamodule including domain information.
                / 도메인 정보를 포함한 데이터모듈 이름

        Example:
            >>> dm = HDMAPDataModule(domain="domain_A")
            >>> dm.name
            'HDMAPDataModule_domain_A'
        """
        return f"HDMAPDataModule_{self.domain}"

    @property
    def category(self) -> str:
        """Get the category (domain) of the datamodule / 데이터모듈의 카테고리(도메인) 가져오기.

        Returns:
            str: Domain name for this datamodule.
                / 이 데이터모듈의 도메인 이름
        """
        return self.domain

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting / 데이터셋 설정 및 동적 서브셋 분할 수행.

        **중요**: 이것이 바로 부모 클래스의 추상 메서드 _setup()을 구현한 것입니다!
        HDMAPDataset을 사용해서 실제 데이터 로딩 로직을 구현합니다.
        **Important**: This implements the parent class's abstract method _setup()!
        Uses HDMAPDataset to implement actual data loading logic.

        **도메인 전이 학습**: 단일 도메인만 로드하여 source/target domain 실험을 지원합니다.
        **Domain Transfer Learning**: Loads only a single domain to support source/target domain experiments.

        Args:
            _stage (str | None): Current stage (fit, test, predict). Not used in HDMAP.
                / 현재 단계 (fit, test, predict). HDMAP에서는 사용하지 않음
        """
        # 훈련용 데이터셋 생성 / Create training dataset
        self.train_data = HDMAPDataset(
            root=self.root,
            domain=self.domain,
            split="train",
            augmentations=self.train_augmentations,
        )

        # 테스트용 데이터셋 생성 / Create test dataset  
        self.test_data = HDMAPDataset(
            root=self.root,
            domain=self.domain,
            split="test",
            augmentations=self.test_augmentations,
        )

        # 검증 세트는 부모 클래스에서 자동으로 생성됨 / Validation set is automatically created by parent class
        # val_split_mode에 따라 test 데이터에서 분할하거나 train 데이터에서 분할
        # Split from test data or train data according to val_split_mode

    def prepare_data(self) -> None:
        """Download the dataset if not available / 데이터셋이 없는 경우 다운로드.

        **PyTorch Lightning 패턴**: prepare_data()는 데이터 다운로드와 전처리를 담당하며,
        여러 GPU/노드에서 중복 실행되지 않도록 보장됩니다.
        **PyTorch Lightning Pattern**: prepare_data() handles data download and preprocessing,
        ensuring it doesn't run multiple times across GPUs/nodes.

        **HDMAP 특성**: HDMAP 데이터는 mat 파일에서 PNG로 변환된 로컬 데이터이므로
        별도의 다운로드가 필요하지 않습니다. 데이터 경로 존재 여부만 확인합니다.
        **HDMAP Characteristics**: HDMAP data is local data converted from mat files to PNG,
        so no separate download is needed. Only checks if data path exists.
        """
        # HDMAP 데이터 경로 확인 / Check HDMAP data path
        if not self.root.exists():
            msg = (
                f"HDMAP dataset not found at {self.root}. "
                f"Please run 'examples/hdmap/prepare_hdmap_dataset.py' first to prepare the data."
            )
            raise FileNotFoundError(msg)

        # 도메인별 경로 확인 / Check domain-specific path
        domain_path = self.root / self.domain
        if not domain_path.exists():
            msg = (
                f"Domain '{self.domain}' not found at {domain_path}. "
                f"Available domains should be: domain_A, domain_B, domain_C, domain_D"
            )
            raise FileNotFoundError(msg)

        print(f"✅ HDMAP 데이터 확인 완료: {self.root}")
        print(f"✅ 도메인 '{self.domain}' 데이터 확인 완료: {domain_path}")
