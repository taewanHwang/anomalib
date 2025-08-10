# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base Anomalib data module.

This module provides the base data module class used across Anomalib. It handles
dataset splitting, validation set creation, and dataloader configuration.

The module contains:
    - :class:`AnomalibDataModule`: Base class for all Anomalib data modules

Example:
    Create a datamodule from a config file::

        >>> from anomalib.data import AnomalibDataModule
        >>> data_config = "examples/configs/data/mvtec.yaml"
        >>> datamodule = AnomalibDataModule.from_config(config_path=data_config)

    Override config with additional arguments::

        >>> override_kwargs = {"data.train_batch_size": 8}
        >>> datamodule = AnomalibDataModule.from_config(
        ...     config_path=data_config,
        ...     **override_kwargs
        ... )
"""

import copy
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from lightning.pytorch import LightningDataModule
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.v2 import Compose, Resize, Transform

from anomalib import TaskType
from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.transforms.utils import extract_transforms_by_type
from anomalib.data.utils import TestSplitMode, ValSplitMode, random_split, split_by_label
from anomalib.data.utils.synthetic import SyntheticAnomalyDataset
from anomalib.utils.attrs import get_nested_attr

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas import DataFrame


logger = logging.getLogger(__name__)


class AnomalibDataModule(LightningDataModule, ABC):
    """Base Anomalib data module / 기본 Anomalib 데이터 모듈.

    This class extends PyTorch Lightning's ``LightningDataModule`` to provide
    common functionality for anomaly detection datasets.
    
    이 클래스는 PyTorch Lightning의 ``LightningDataModule``을 확장하여
    이상 탐지 데이터셋을 위한 공통 기능을 제공합니다.

    Args:
        train_batch_size (int): Batch size used by the train dataloader.
            / 훈련 데이터로더에서 사용할 배치 크기
        eval_batch_size (int): Batch size used by the val and test dataloaders.
            / 검증 및 테스트 데이터로더에서 사용할 배치 크기
        num_workers (int): Number of workers used by the train, val and test dataloaders.
            / 훈련, 검증, 테스트 데이터로더에서 사용할 워커 수
        train_augmentations (Transform | None): Augmentations to apply dto the training images
            / 훈련 이미지에 적용할 데이터 증강. Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            / 검증 이미지에 적용할 데이터 증강. Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            / 테스트 이미지에 적용할 데이터 증강. Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
            / 단계별 증강이 제공되지 않은 경우 적용할 일반 데이터 증강
        val_split_mode (ValSplitMode | str): Method to obtain validation set.
            / 검증 세트를 얻는 방법.
            **중요**: 이 설정은 모델 성능 검증과 하이퍼파라미터 튜닝에 큰 영향을 미칩니다.
            Options:
                - ``none``: No validation set / 검증 세트 없음
                - ``same_as_test``: Use test set as validation / 테스트 세트를 검증으로 사용
                - ``from_test``: Sample from test set / 테스트 세트에서 샘플링
                - ``synthetic``: Generate synthetic anomalies / 합성 이상 생성
        val_split_ratio (float): Fraction of data to use for validation
            / 검증에 사용할 데이터 비율
        test_split_mode (TestSplitMode | str | None): Method to obtain test set.
            / 테스트 세트를 얻는 방법.
            **중요**: 이 설정은 최종 모델 평가의 신뢰성을 결정합니다.
            잘못 설정하면 과적합을 제대로 탐지하지 못할 수 있습니다.
            Options:
                - ``none``: No test split / 테스트 분할 없음
                - ``from_dir``: Use separate test directory / 별도의 테스트 디렉토리 사용
                - ``synthetic``: Generate synthetic anomalies / 합성 이상 생성
            Defaults to ``None``.
        test_split_ratio (float | None): Fraction of data to use for testing.
            / 테스트에 사용할 데이터 비율. Defaults to ``None``.
        seed (int | None): Random seed for reproducible splitting.
            / 재현 가능한 분할을 위한 랜덤 시드. Defaults to ``None``.
    """

    def __init__(
        self,
        train_batch_size: int,
        eval_batch_size: int,
        num_workers: int,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        val_split_mode: ValSplitMode | str | None = None,
        val_split_ratio: float | None = None,
        test_split_mode: TestSplitMode | str | None = None,
        test_split_ratio: float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        # split_mode 설정은 매우 중요합니다 - 데이터 분할 전략이 모델 성능 평가에 직접적인 영향을 미칩니다
        self.test_split_mode = TestSplitMode(test_split_mode) if test_split_mode else TestSplitMode.NONE
        self.test_split_ratio = test_split_ratio or 0.5
        self.val_split_mode = ValSplitMode(val_split_mode) if val_split_mode else ValSplitMode.NONE
        self.val_split_ratio = val_split_ratio or 0.5
        self.seed = seed

        # 증강 우선순위 로직: 단계별 증강이 제공되면 우선 사용, 없으면 일반 증강 사용
        self.train_augmentations = train_augmentations or augmentations
        self.val_augmentations = val_augmentations or augmentations
        self.test_augmentations = test_augmentations or augmentations

        self.train_data: AnomalibDataset
        self.val_data: AnomalibDataset
        self.test_data: AnomalibDataset

        self._samples: DataFrame | None = None
        self._category: str = ""

        self._is_setup = False  # flag to track if setup has been called / setup이 호출되었는지 추적하는 플래그

        self.external_collate_fn: Callable | None = None

    @property
    def name(self) -> str:
        """Name of the datamodule / 데이터모듈의 이름.

        Returns:
            str: Class name of the datamodule
            / 데이터모듈의 클래스 이름
        """
        return self.__class__.__name__

    def setup(self, stage: str | None = None) -> None:
        """Set up train, validation and test data / 훈련, 검증, 테스트 데이터 설정.

        This method handles the data splitting logic based on the configured
        modes.
        
        이 메서드는 설정된 모드에 따라 데이터 분할 로직을 처리합니다.

        Args:
            stage (str | None): Current stage (fit/validate/test/predict).
                / 현재 단계 (fit/validate/test/predict).
                Defaults to ``None``.
        """
        has_subset = any(hasattr(self, subset) for subset in ["train_data", "val_data", "test_data"])  # 서브셋 존재 확인
        if not has_subset or not self._is_setup:
            self._setup(stage)
            self._create_test_split()
            self._create_val_split()
            if isinstance(stage, TrainerFn):
                # only set flag if called from trainer
                # 트레이너에서 호출된 경우에만 플래그 설정
                self._is_setup = True

        self._update_augmentations()

    def _update_augmentations(self) -> None:
        """Update the augmentations for each subset / 각 서브셋에 대한 데이터 증강 업데이트."""
        for subset_name in ["train", "val", "test"]:
            subset = getattr(self, f"{subset_name}_data", None)
            augmentations = getattr(self, f"{subset_name}_augmentations", None)
            model_transform = get_nested_attr(self, "trainer.model.pre_processor.transform", None)

            if subset:
                if model_transform:
                    # If model transform exists, update augmentations with model-specific transforms
                    # 모델 변환이 존재하면, 모델별 변환으로 증강 업데이트
                    self._update_subset_augmentations(subset, augmentations, model_transform)
                else:
                    # If no model transform, just apply the user-specified augmentations
                    # 모델 변환이 없으면, 사용자 지정 증강만 적용
                    subset.augmentations = augmentations

    @staticmethod
    def _update_subset_augmentations(
        dataset: AnomalibDataset,
        augmentations: Transform | None,
        model_transform: Transform,
    ) -> None:
        """Update the augmentations of the dataset / 데이터셋의 증강 업데이트.

        This method passes the user-specified augmentations to a dataset subset. If the model transforms contain
        a Resize transform, it will be appended to the augmentations. This will ensure that resizing takes place
        before collating, which reduces the usage of shared memory by the Dataloader workers.
        
        이 메서드는 사용자가 지정한 증강을 데이터셋 서브셋에 전달합니다. 모델 변환에 Resize 변환이 포함된 경우,
        이를 증강에 추가합니다. 이는 collating 전에 크기 조정이 이루어지도록 하여 Dataloader 워커들의
        공유 메모리 사용량을 줄입니다.

        Args:
            dataset (AnomalibDataset): Dataset to update.
                / 업데이트할 데이터셋
            augmentations (Transform): Augmentations to apply to the dataset.
                / 데이터셋에 적용할 증강
            model_transform (Transform): Transform object from the model PreProcessor.
                / 모델 PreProcessor의 변환 객체
        """
        model_resizes = extract_transforms_by_type(model_transform, Resize)

        if model_resizes:
            model_resize = model_resizes[0]
            for aug_resize in extract_transforms_by_type(augmentations, Resize):  # warn user if resizes inconsistent / 크기 조정이 일관되지 않으면 사용자에게 경고
                if model_resize.size != aug_resize.size:
                    msg = f"Conflicting resize shapes found between augmentations and model transforms. You are using \
                        a Resize transform in your input data augmentations. Please be aware that the model also \
                        applies a Resize transform with a different output size. The final effective input size as \
                        seen by the model will be determined by the model transforms, not the augmentations. To change \
                        the effective input size, please change the model transforms in the PreProcessor module. \
                        Augmentations: {aug_resize.size}, Model transforms: {model_resize.size}"
                    logger.warning(msg)
                if model_resize.interpolation != aug_resize.interpolation:
                    msg = f"Conflicting interpolation method found between augmentations and model transforms. You are \
                        using a Resize transform in your input data augmentations. Please be aware that the model also \
                        applies a Resize transform with a different interpolation method. Using multiple interpolation \
                        methods can lead to unexpected behaviour, so it is recommended to use the same interpolation \
                        method between augmentations and model transforms. Augmentations: {aug_resize.interpolation}, \
                        Model transforms: {model_resize.interpolation}"
                    logger.warning(msg)
                if model_resize.antialias != aug_resize.antialias:
                    msg = f"Conflicting antialiasing setting found between augmentations and model transforms. You are \
                        using a Resize transform in your input data augmentations. Please be aware that the model also \
                        applies a Resize transform with a different antialising setting. Using conflicting \
                        antialiasing settings can lead to unexpected behaviour, so it is recommended to use the same \
                        antialiasing setting between augmentations and model transforms. Augmentations: \
                        antialias={aug_resize.antialias}, Model transforms: antialias={model_resize.antialias}"
                    logger.warning(msg)

            # append model resize to augmentations / 모델 크기 조정을 증강에 추가
            if isinstance(augmentations, Resize):
                augmentations = model_resize
            elif isinstance(augmentations, Compose):
                augmentations = Compose([*augmentations.transforms, model_resize])
            elif isinstance(augmentations, Transform):
                augmentations = Compose([augmentations, model_resize])
            elif augmentations is None:
                augmentations = model_resize

        dataset.augmentations = augmentations

    @abstractmethod
    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting / 데이터셋 설정 및 동적 서브셋 분할 수행.

        This method should be implemented by subclasses to define dataset-specific
        setup logic.
        
        이 메서드는 데이터셋별 설정 로직을 정의하기 위해 서브클래스에서 구현되어야 합니다.
        
        **중요**: 이 메서드는 추상 메서드(abstract method)입니다. 
        AnomalibDataModule을 상속받는 모든 클래스는 반드시 이 메서드를 구현해야 합니다.
        예: MVTecADDataModule, FolderDataModule 등에서 각자의 데이터 로딩 로직을 구현합니다.

        Note:
            The ``stage`` argument is not used since all subsets are created on
            first call to accommodate validation set extraction from test set.
            
            ``stage`` 인수는 테스트 세트에서 검증 세트 추출을 수용하기 위해 
            첫 번째 호출 시 모든 서브셋이 생성되므로 사용되지 않습니다.

        Args:
            _stage (str | None): Current stage (unused).
                / 현재 단계 (사용되지 않음).
                Defaults to ``None``.

        Raises:
            NotImplementedError: When not implemented by subclass
            / 서브클래스에서 구현되지 않은 경우. 이 에러가 발생하면 
            상속받은 클래스에서 _setup 메서드를 구현해야 합니다.
        """
        raise NotImplementedError

    @property
    def category(self) -> str:
        """Get dataset category name / 데이터셋 카테고리 이름 가져오기.

        Returns:
            str: Name of the current category
            / 현재 카테고리의 이름
        """
        return self._category

    @category.setter
    def category(self, category: str) -> None:
        """Set dataset category name / 데이터셋 카테고리 이름 설정.

        Args:
            category (str): Category name to set
            / 설정할 카테고리 이름
        """
        self._category = category

    @property
    def task(self) -> TaskType:
        """Get the task type / 태스크 타입 가져오기.

        Returns:
            TaskType: Type of anomaly task (classification/segmentation)
            / 이상 탐지 태스크 타입 (분류/분할)

        Raises:
            AttributeError: If no datasets have been set up yet
            / 아직 데이터셋이 설정되지 않은 경우
        """
        if hasattr(self, "train_data"):
            return self.train_data.task
        if hasattr(self, "val_data"):
            return self.val_data.task
        if hasattr(self, "test_data"):
            return self.test_data.task
        msg = "This datamodule does not have any datasets. Did you call setup?"
        raise AttributeError(msg)

    def _create_test_split(self) -> None:
        """Create the test split based on configured mode / 설정된 모드에 따라 테스트 분할 생성.

        This handles splitting normal/anomalous samples and optionally creating
        synthetic anomalies.
        
        이는 정상/이상 샘플 분할을 처리하고 선택적으로 합성 이상을 생성합니다.
        """
        if self.test_data.has_normal:
            # split test data into normal and anomalous / 테스트 데이터를 정상과 이상으로 분할
            normal_test_data, self.test_data = split_by_label(self.test_data)
        elif self.test_split_mode != TestSplitMode.NONE:
            # sample normal images from training set if none provided
            # 제공되지 않은 경우 훈련 세트에서 정상 이미지 샘플링
            logger.info(
                "No normal test images found. Sampling from training set using ratio of %0.2f",
                self.test_split_ratio,
            )
            if self.test_split_ratio is not None:
                self.train_data, normal_test_data = random_split(
                    self.train_data,
                    self.test_split_ratio,
                    seed=self.seed,
                )

        if self.test_split_mode == TestSplitMode.FROM_DIR:
            self.test_data += normal_test_data
        elif self.test_split_mode == TestSplitMode.SYNTHETIC:
            self.test_data = SyntheticAnomalyDataset.from_dataset(normal_test_data)
        elif self.test_split_mode != TestSplitMode.NONE:
            msg = f"Unsupported Test Split Mode: {self.test_split_mode}"
            raise ValueError(msg)

    def _create_val_split(self) -> None:
        """Create validation split based on configured mode / 설정된 모드에 따라 검증 분할 생성.

        This handles sampling from train/test sets and optionally creating
        synthetic anomalies.
        
        이는 훈련/테스트 세트에서 샘플링을 처리하고 선택적으로 합성 이상을 생성합니다.
        """
        if self.val_split_mode == ValSplitMode.FROM_DIR:
            # If the validation split mode is FROM_DIR, we don't need to create a validation set
            # 검증 분할 모드가 FROM_DIR인 경우, 검증 세트를 생성할 필요 없음
            return
        if self.val_split_mode == ValSplitMode.FROM_TRAIN:
            # randomly sample from train set / 훈련 세트에서 무작위 샘플링
            self.train_data, self.val_data = random_split(
                self.train_data,
                self.val_split_ratio,
                label_aware=True,
                seed=self.seed,
            )
        elif self.val_split_mode == ValSplitMode.FROM_TEST:
            # randomly sample from test set / 테스트 세트에서 무작위 샘플링
            self.test_data, self.val_data = random_split(
                self.test_data,
                self.val_split_ratio,
                label_aware=True,
                seed=self.seed,
            )
        elif self.val_split_mode == ValSplitMode.SAME_AS_TEST:
            # equal to test set / 테스트 세트와 동일
            self.val_data = copy.deepcopy(self.test_data)
        elif self.val_split_mode == ValSplitMode.SYNTHETIC:
            # create synthetic anomalies from training samples / 훈련 샘플에서 합성 이상 생성
            self.train_data, normal_val_data = random_split(
                self.train_data,
                self.val_split_ratio,
                seed=self.seed,
            )
            self.val_data = SyntheticAnomalyDataset.from_dataset(normal_val_data)
        elif self.val_split_mode != ValSplitMode.NONE:
            msg = f"Unknown validation split mode: {self.val_split_mode}"
            raise ValueError(msg)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Get training dataloader / 훈련 데이터로더 가져오기.

        **개념 설명**: 여기서 사용하는 DataLoader는 PyTorch에서 제공하는 표준 DataLoader입니다.
        DataLoader는 데이터셋을 효율적으로 배치(batch) 단위로 로딩하고, 
        멀티프로세싱, 셔플링 등의 기능을 제공합니다.

        Returns:
            DataLoader: Training dataloader
            / 훈련 데이터로더 (PyTorch 표준 DataLoader 객체)
        """
        # PyTorch DataLoader 생성 - 배치 처리, 셔플링, 멀티프로세싱 등을 담당
        # Creating PyTorch DataLoader - handles batching, shuffling, multiprocessing, etc.
        return DataLoader(
            dataset=self.train_data,
            shuffle=True,  # 훈련 시 데이터 순서 무작위화 / Randomize data order during training
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,  # 멀티프로세싱 워커 수 / Number of multiprocessing workers
            collate_fn=self.external_collate_fn or self.train_data.collate_fn,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Get validation dataloader / 검증 데이터로더 가져오기.

        Returns:
            DataLoader: Validation dataloader
            / 검증 데이터로더 (PyTorch 표준 DataLoader 객체)
        """
        return DataLoader(
            dataset=self.val_data,
            shuffle=False,  # 검증 시에는 셔플링하지 않음 / No shuffling during validation
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.external_collate_fn or self.val_data.collate_fn,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Get test dataloader / 테스트 데이터로더 가져오기.

        Returns:
            DataLoader: Test dataloader
            / 테스트 데이터로더 (PyTorch 표준 DataLoader 객체)
        """
        return DataLoader(
            dataset=self.test_data,
            shuffle=False,  # 테스트 시에는 셔플링하지 않음 / No shuffling during testing
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.external_collate_fn or self.test_data.collate_fn,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """Get prediction dataloader / 예측 데이터로더 가져오기.

        By default uses the test dataloader.
        기본적으로 테스트 데이터로더를 사용합니다.

        Returns:
            DataLoader: Prediction dataloader
            / 예측 데이터로더
        """
        return self.test_dataloader()

    @classmethod
    def from_config(
        cls: type["AnomalibDataModule"],
        config_path: str | Path,
        **kwargs,
    ) -> "AnomalibDataModule":
        """Create datamodule instance from config file / 설정 파일로부터 데이터모듈 인스턴스 생성.

        Args:
            config_path (str | Path): Path to config file
                / 설정 파일 경로
            **kwargs: Additional args to override config
                / 설정을 재정의할 추가 인수들

        Returns:
            AnomalibDataModule: Instantiated datamodule
            / 인스턴스화된 데이터모듈

        Raises:
            FileNotFoundError: If config file not found
                / 설정 파일을 찾을 수 없는 경우
            ValueError: If instantiated object is not AnomalibDataModule
                / 인스턴스화된 객체가 AnomalibDataModule이 아닌 경우

        Example:
            Load from config file / 설정 파일에서 로드::

                >>> config_path = "examples/configs/data/mvtec.yaml"
                >>> datamodule = AnomalibDataModule.from_config(config_path)

            Override config values / 설정 값 재정의::

                >>> datamodule = AnomalibDataModule.from_config(
                ...     config_path,
                ...     data_train_batch_size=8
                ... )
        """
        from jsonargparse import ArgumentParser

        if not Path(config_path).exists():
            msg = f"Configuration file not found: {config_path}"
            raise FileNotFoundError(msg)

        data_parser = ArgumentParser()
        data_parser.add_subclass_arguments(AnomalibDataModule, "data", required=False, fail_untyped=False)
        args = ["--data", str(config_path)]
        for key, value in kwargs.items():
            args.extend([f"--{key}", str(value)])
        config = data_parser.parse_args(args=args)
        instantiated_classes = data_parser.instantiate_classes(config)
        datamodule = instantiated_classes.get("data")
        if isinstance(datamodule, AnomalibDataModule):
            return datamodule

        msg = f"Datamodule is not an instance of AnomalibDataModule: {datamodule}"
        raise ValueError(msg)
