# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MVTec AD Data Module / MVTec AD 데이터 모듈.

This module provides a PyTorch Lightning DataModule for the MVTec AD dataset. If
the dataset is not available locally, it will be downloaded and extracted
automatically.

이 모듈은 MVTec AD 데이터셋을 위한 PyTorch Lightning DataModule을 제공합니다.
데이터셋이 로컬에 없으면 자동으로 다운로드되고 압축이 해제됩니다.

**교육적 가치**: MVTec AD는 이상 탐지 연구에서 가장 널리 사용되는 벤치마크 데이터셋으로,
산업 환경에서의 제품 결함 탐지를 시뮬레이션합니다. 15개의 다른 제품 카테고리를 제공하여
다양한 이상 탐지 시나리오를 학습할 수 있습니다.

Examples / 사용 예시들:
    
    **예시 1**: 기본 사용법 - 가장 간단한 형태 / Basic usage - simplest form::

        >>> from anomalib.data import MVTecAD
        >>> datamodule = MVTecAD(
        ...     root="./datasets/MVTecAD",
        ...     category="bottle"  # 15개 카테고리 중 하나 선택
        ... )
        >>> datamodule.setup()  # 데이터 준비
        >>> train_loader = datamodule.train_dataloader()
        >>> print(f"Training batches: {len(train_loader)}")

    **예시 2**: 다른 카테고리와 배치 크기 조정 / Different category and batch size::

        >>> datamodule = MVTecAD(
        ...     category="cable",  # 케이블 결함 탐지
        ...     train_batch_size=16,  # 작은 배치 크기
        ...     eval_batch_size=8,
        ...     num_workers=4  # CPU 코어 수에 맞게 조정
        ... )
        
    **예시 3**: 데이터 증강 적용 / Data augmentation application::

        >>> from torchvision.transforms import RandomRotation, ColorJitter
        >>> from torchvision.transforms import Compose
        >>> 
        >>> # 훈련용 데이터 증강 정의
        >>> train_transforms = Compose([
        ...     RandomRotation(10),  # 10도 회전
        ...     ColorJitter(brightness=0.2, contrast=0.2)  # 밝기/대비 조정
        ... ])
        >>> 
        >>> datamodule = MVTecAD(
        ...     category="screw",
        ...     train_augmentations=train_transforms,
        ...     # 검증/테스트에는 증강 적용하지 않음 (일반적인 관례)
        ... )

    **예시 4**: 검증 세트 분할 모드 변경 / Validation split mode configuration::

        >>> from anomalib.data.utils import ValSplitMode
        >>> 
        >>> # 테스트 데이터에서 검증 세트 분리 (더 엄격한 평가)
        >>> datamodule = MVTecAD(
        ...     category="transistor",
        ...     val_split_mode=ValSplitMode.FROM_TEST,
        ...     val_split_ratio=0.2,  # 테스트 데이터의 20%를 검증용으로
        ...     seed=42  # 재현 가능한 분할
        ... )
        >>> 
        >>> # 합성 이상 검증 세트 생성 (고급 기법)
        >>> datamodule_synthetic = MVTecAD(
        ...     category="wood",
        ...     val_split_mode=ValSplitMode.SYNTHETIC,
        ...     val_split_ratio=0.3,  # 정상 데이터의 30%에서 합성 이상 생성
        ... )

    **예시 5**: 실제 훈련 파이프라인에서 사용 / Usage in actual training pipeline::

        >>> import lightning as L
        >>> from anomalib.models import PatchCore
        >>> 
        >>> # 데이터모듈과 모델 설정
        >>> datamodule = MVTecAD(
        ...     category="metal_nut",
        ...     train_batch_size=32,
        ...     eval_batch_size=32,
        ... )
        >>> model = PatchCore()
        >>> 
        >>> # PyTorch Lightning 트레이너로 학습
        >>> trainer = L.Trainer(
        ...     max_epochs=10,
        ...     accelerator="auto",  # GPU 자동 감지
        ... )
        >>> 
        >>> # 전체 학습 파이프라인 실행
        >>> trainer.fit(model, datamodule)
        >>> 
        >>> # 테스트 수행
        >>> test_results = trainer.test(model, datamodule)
        >>> print(f"Test AUROC: {test_results[0]['test_AUROC']:.3f}")

    **교육적 팁들 / Educational Tips**:
        - 각 카테고리는 서로 다른 결함 유형을 가짐 (스크래치, 균열, 변색 등)
        - Each category has different defect types (scratches, cracks, discoloration, etc.)
        - 배치 크기는 GPU 메모리와 성능의 균형점을 고려하여 설정
        - Batch size should balance GPU memory and performance
        - seed 설정으로 실험의 재현성 보장 가능
        - Setting seed ensures experiment reproducibility

Notes:
    The dataset will be automatically downloaded and converted to the required
    format when first used. The directory structure after preparation will be::

        datasets/
        └── MVTecAD/
            ├── bottle/
            ├── cable/
            └── ...

License:
    MVTec AD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0).
    https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference:
    Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger,
    Carsten Steger: The MVTec Anomaly Detection Dataset: A Comprehensive
    Real-World Dataset for Unsupervised Anomaly Detection; in: International
    Journal of Computer Vision 129(4):1038-1059, 2021,
    DOI: 10.1007/s11263-020-01400-4.

    Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: MVTec AD —
    A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection;
    in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
    9584-9592, 2019, DOI: 10.1109/CVPR.2019.00982.
"""

import logging
from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.mvtecad import MVTecADDataset
from anomalib.data.utils import DownloadInfo, Split, TestSplitMode, ValSplitMode, download_and_extract
from anomalib.utils import deprecate

logger = logging.getLogger(__name__)


DOWNLOAD_INFO = DownloadInfo(
    name="mvtecad",
    url="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/"
    "download/420938113-1629952094/mvtec_anomaly_detection.tar.xz",
    hashsum="cf4313b13603bec67abb49ca959488fx7eedce2a9f7795ec54446c649ac98cd3d",
)


class MVTecAD(AnomalibDataModule):
    """MVTec AD Datamodule / MVTec AD 데이터모듈.

    **상속 관계**: 이 클래스는 AnomalibDataModule을 상속받아 구체적인 데이터셋(MVTec AD)에 
    특화된 구현을 제공합니다. 추상 메서드 _setup()을 실제로 구현한 예시입니다.
    
    **MVTec AD 데이터셋**: 산업용 이상 탐지를 위한 대표적인 벤치마크 데이터셋으로,
    15개 카테고리(bottle, cable, screw 등)의 제품 이미지를 포함합니다.

    Args:
        root (Path | str): Path to the root of the dataset.
            / 데이터셋 루트 경로. Defaults to ``"./datasets/MVTecAD"``.
        category (str): Category of the MVTec AD dataset (e.g. ``"bottle"`` or
            ``"cable"``). / MVTec AD 데이터셋의 카테고리 (예: ``"bottle"`` 또는 ``"cable"``). 
            Defaults to ``"bottle"``.
        train_batch_size (int, optional): Training batch size.
            / 훈련 배치 크기. Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            / 평가(테스트) 배치 크기. Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            / 데이터 로딩 워커 수. Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations to apply dto the training images
            / 훈련 이미지에 적용할 데이터 증강. Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            / 검증 이미지에 적용할 데이터 증강. Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            / 테스트 이미지에 적용할 데이터 증강. Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
            / 단계별 증강이 제공되지 않은 경우 적용할 일반 데이터 증강.
        test_split_mode (TestSplitMode): Method to create test set.
            / 테스트 세트 생성 방법. Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of data to use for testing.
            / 테스트에 사용할 데이터 비율. Defaults to ``0.2``.
        val_split_mode (ValSplitMode): Method to create validation set.
            / 검증 세트 생성 방법. Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction of data to use for validation.
            / 검증에 사용할 데이터 비율. Defaults to ``0.5``.
        seed (int | None, optional): Seed for reproducibility.
            / 재현성을 위한 시드. Defaults to ``None``.

    Example:
        Create MVTec AD datamodule with default settings / 기본 설정으로 MVTec AD 데이터모듈 생성::

            >>> datamodule = MVTecAD()
            >>> datamodule.setup()
            >>> i, data = next(enumerate(datamodule.train_dataloader()))
            >>> data.keys()
            dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

            >>> data["image"].shape
            torch.Size([32, 3, 256, 256])

        Change the category / 카테고리 변경::

            >>> datamodule = MVTecAD(category="cable")

        Create validation set from test data / 테스트 데이터에서 검증 세트 생성::

            >>> datamodule = MVTecAD(
            ...     val_split_mode=ValSplitMode.FROM_TEST,
            ...     val_split_ratio=0.1
            ... )

        Create synthetic validation set / 합성 검증 세트 생성::

            >>> datamodule = MVTecAD(
            ...     val_split_mode=ValSplitMode.SYNTHETIC,
            ...     val_split_ratio=0.2
            ... )
    """

    def __init__(
        self,
        root: Path | str = "./datasets/MVTecAD",
        category: str = "bottle",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,  # MVTecAD는 별도 테스트 디렉토리 제공
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.SAME_AS_TEST,  # 기본값: 테스트 세트를 검증으로도 사용
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        # 부모 클래스(AnomalibDataModule) 초기화 - 공통 설정들을 부모에게 전달
        # Initialize parent class (AnomalibDataModule) - pass common settings to parent
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        # MVTecAD 데이터셋 특화 설정 - MVTecAD specific configurations
        self.root = Path(root)  # 데이터셋 경로 / Dataset path
        self.category = category  # 제품 카테고리 (bottle, cable, screw 등) / Product category

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting / 데이터셋 설정 및 동적 서브셋 분할 수행.

        **중요**: 이것이 바로 부모 클래스의 추상 메서드 _setup()을 구현한 것입니다!
        MVTecADDataset을 사용해서 실제 데이터 로딩 로직을 구현합니다.

        This method may be overridden in subclass for custom splitting behaviour.
        이 메서드는 커스텀 분할 동작을 위해 서브클래스에서 재정의될 수 있습니다.

        Note:
            The stage argument is not used here. This is because, for a given
            instance of an AnomalibDataModule subclass, all three subsets are
            created at the first call of setup(). This is to accommodate the
            subset splitting behaviour of anomaly tasks, where the validation set
            is usually extracted from the test set, and the test set must
            therefore be created as early as the `fit` stage.
            
            stage 인수는 여기서 사용되지 않습니다. 이는 AnomalibDataModule 서브클래스의 
            특정 인스턴스에 대해 setup()의 첫 번째 호출 시 모든 세 개의 서브셋이 생성되기 때문입니다.
            이는 검증 세트가 일반적으로 테스트 세트에서 추출되는 이상 탐지 태스크의 
            서브셋 분할 동작을 수용하기 위함입니다.
        """
        # MVTecADDataset을 사용해서 훈련 데이터와 테스트 데이터 생성
        # Create training and test data using MVTecADDataset
        self.train_data = MVTecADDataset(
            split=Split.TRAIN,  # 훈련 분할 / Training split
            root=self.root,
            category=self.category,
        )
        self.test_data = MVTecADDataset(
            split=Split.TEST,   # 테스트 분할 / Test split
            root=self.root,
            category=self.category,
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available / 데이터셋이 없는 경우 다운로드.

        This method checks if the specified dataset is available in the file
        system. If not, it downloads and extracts the dataset into the
        appropriate directory.
        
        이 메서드는 지정된 데이터셋이 파일 시스템에 있는지 확인합니다. 
        없으면 데이터셋을 다운로드하고 적절한 디렉토리에 압축을 해제합니다.

        **PyTorch Lightning 패턴**: prepare_data()는 데이터 다운로드와 전처리를 담당하며,
        여러 GPU/노드에서 중복 실행되지 않도록 보장됩니다.

        Example:
            Assume the dataset is not available on the file system / 
            파일 시스템에 데이터셋이 없다고 가정::

                >>> datamodule = MVTecAD(
                ...     root="./datasets/MVTecAD",
                ...     category="bottle"
                ... )
                >>> datamodule.prepare_data()

            Directory structure after download / 다운로드 후 디렉토리 구조::

                datasets/
                └── MVTecAD/
                    ├── bottle/
                    ├── cable/
                    └── ...
        """
        # 지정된 카테고리 디렉토리가 존재하는지 확인 / Check if specified category directory exists
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")  # 데이터셋 발견 / Dataset found
        else:
            # 데이터셋 다운로드 및 압축 해제 / Download and extract dataset
            download_and_extract(self.root, DOWNLOAD_INFO)


@deprecate(since="2.1.0", remove="2.3.0", use="MVTecAD")
class MVTec(MVTecAD):
    """MVTec datamodule class (Deprecated) / MVTec 데이터모듈 클래스 (사용 중지됨).

    This class is deprecated and will be removed in a future version.
    Please use MVTecAD instead.
    
    이 클래스는 사용 중지되었으며 향후 버전에서 제거될 예정입니다.
    대신 MVTecAD를 사용하시기 바랍니다.
    
    **학습 포인트**: 이는 소프트웨어 버전 관리에서 하위 호환성을 유지하면서
    API를 점진적으로 변경하는 방법의 예시입니다.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
