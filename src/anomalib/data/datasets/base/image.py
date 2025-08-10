# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Anomalib dataset base class / Anomalib 데이터셋 기본 클래스.

This module provides the base dataset class for Anomalib datasets. The dataset is based on a
dataframe that contains the information needed by the dataloader to load each dataset item
into memory.

이 모듈은 Anomalib 데이터셋의 기본 클래스를 제공합니다. 데이터셋은 데이터로더가 각 데이터셋 항목을
메모리로 로드하는 데 필요한 정보가 포함된 데이터프레임을 기반으로 합니다.

The samples dataframe must be set from the subclass using the setter of the ``samples``
property.

샘플 데이터프레임은 ``samples`` 속성의 setter를 사용하여 하위 클래스에서 설정되어야 합니다.

The DataFrame must include at least the following columns:
    - ``split`` (str): The subset to which the dataset item is assigned (e.g., 'train',
      'test'). / 데이터셋 항목이 할당된 서브셋 (예: 'train', 'test')
    - ``image_path`` (str): Path to the file system location where the image is stored.
      / 이미지가 저장된 파일 시스템 위치의 경로
    - ``label_index`` (int): Index of the anomaly label, typically 0 for 'normal' and 1 for
      'anomalous'. / 이상 라벨의 인덱스, 일반적으로 정상은 0, 이상은 1
    - ``mask_path`` (str, optional): Path to the ground truth masks (for anomalous images
      only). Required if task is 'segmentation'.
      / 실제 마스크 경로 (이상 이미지만 해당). 작업이 'segmentation'인 경우 필수

데이터프레임은 최소한 다음 열들을 포함해야 합니다:

Example DataFrame:
    >>> df = pd.DataFrame({
    ...     'image_path': ['path/to/image.png'],
    ...     'label': ['anomalous'],
    ...     'label_index': [1],
    ...     'mask_path': ['path/to/mask.png'],
    ...     'split': ['train']
    ... })
"""

import copy
import logging
from abc import ABC
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform
from torchvision.tv_tensors import Mask

from anomalib import TaskType
from anomalib.data.dataclasses import DatasetItem, ImageBatch, ImageItem
from anomalib.data.utils import LabelName, read_image, read_mask

_EXPECTED_COLUMNS = ["image_path", "split"]

logger = logging.getLogger(__name__)


class AnomalibDataset(Dataset, ABC):
    """Base class for Anomalib datasets / Anomalib 데이터셋의 기본 클래스.

    The dataset is designed to work with image-based anomaly detection tasks. It supports
    both classification and segmentation tasks.

    이 데이터셋은 이미지 기반 이상 탐지 작업을 위해 설계되었습니다. 분류와 세그멘테이션 작업을 모두 지원합니다.

    **상속 관계**: 이 클래스는 PyTorch의 Dataset을 상속받아 이상 탐지에 특화된 기능을 제공합니다.
    **Inheritance**: This class inherits from PyTorch's Dataset, providing anomaly detection-specific functionality.

    Args:
        augmentations (Transform | None, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``. / 입력 이미지에 적용할 증강. 기본값은 ``None``

    Example:
        >>> from torchvision.transforms.v2 import Resize
        >>> dataset = AnomalibDataset(augmentations=Resize((256, 256)))
        >>> len(dataset)  # Get dataset length / 데이터셋 길이 가져오기
        100
        >>> item = dataset[0]  # Get first item / 첫 번째 항목 가져오기
        >>> item.image.shape
        torch.Size([3, 256, 256])

    Note:
        The example above is illustrative and may need to be adjusted based on the specific dataset structure.
        위 예시는 설명용이며 특정 데이터셋 구조에 따라 조정이 필요할 수 있습니다.
    """

    def __init__(self, augmentations: Transform | None = None) -> None:
        super().__init__()
        self.augmentations = augmentations
        self._samples: DataFrame | None = None
        self._category: str | None = None

    @property
    def name(self) -> str:
        """Get the name of the dataset / 데이터셋 이름 가져오기.

        Returns:
            str: Name of the dataset derived from the class name, with 'Dataset' suffix
                removed if present. / 클래스 이름에서 파생된 데이터셋 이름, 'Dataset' 접미사가 있으면 제거

        Example:
            >>> dataset = AnomalibDataset()
            >>> dataset.name
            'Anomalib'
        """
        class_name = self.__class__.__name__

        # Remove the `_dataset` suffix from the class name
        if class_name.endswith("Dataset"):
            class_name = class_name[:-7]

        return class_name

    def __len__(self) -> int:
        """Get length of the dataset / 데이터셋 길이 가져오기.

        Returns:
            int: Number of samples in the dataset. / 데이터셋의 샘플 수

        Raises:
            RuntimeError: If samples DataFrame is not set. / samples DataFrame이 설정되지 않은 경우
        """
        return len(self.samples)

    def subsample(self, indices: Sequence[int], inplace: bool = False) -> "AnomalibDataset":
        """Create a subset of the dataset using the provided indices / 제공된 인덱스를 사용하여 데이터셋 부분집합 생성.

        Args:
            indices (Sequence[int]): Indices at which the dataset is to be subsampled.
                / 데이터셋을 부분 샘플링할 인덱스들
            inplace (bool, optional): When true, modify the instance itself. Defaults to
                ``False``. / True인 경우 인스턴스 자체를 수정. 기본값은 ``False``

        Returns:
            AnomalibDataset: Subsampled dataset. / 부분 샘플링된 데이터셋

        Raises:
            ValueError: If duplicate indices are provided. / 중복 인덱스가 제공된 경우

        Example:
            >>> dataset = AnomalibDataset()
            >>> subset = dataset.subsample([0, 1, 2])
            >>> len(subset)
            3
        """
        if len(set(indices)) != len(indices):
            msg = "No duplicates allowed in indices."
            raise ValueError(msg)
        dataset = self if inplace else copy.deepcopy(self)
        dataset.samples = self.samples.iloc[indices].reset_index(drop=True)
        return dataset

    @property
    def samples(self) -> DataFrame:
        """Get the samples DataFrame / 샘플 DataFrame 가져오기.

        Returns:
            DataFrame: DataFrame containing dataset samples. / 데이터셋 샘플을 포함하는 DataFrame

        Raises:
            RuntimeError: If samples DataFrame has not been set. / samples DataFrame이 설정되지 않은 경우
        """
        if self._samples is None:
            msg = (
                "Dataset does not have a samples dataframe. Ensure that a dataframe has "
                "been assigned to `dataset.samples`."
            )
            raise RuntimeError(msg)
        return self._samples

    @samples.setter
    def samples(self, samples: DataFrame) -> None:
        """Set the samples DataFrame / 샘플 DataFrame 설정.

        Args:
            samples (DataFrame): DataFrame containing dataset samples. / 데이터셋 샘플을 포함하는 DataFrame

        Raises:
            TypeError: If samples is not a pandas DataFrame. / samples이 pandas DataFrame이 아닌 경우
            ValueError: If required columns are missing. / 필수 열이 누락된 경우
            FileNotFoundError: If any image paths do not exist. / 이미지 경로가 존재하지 않는 경우

        Example:
            >>> df = pd.DataFrame({
            ...     'image_path': ['image.png'],
            ...     'split': ['train']
            ... })
            >>> dataset = AnomalibDataset()
            >>> dataset.samples = df
        """
        # validate the passed samples by checking the
        if not isinstance(samples, DataFrame):
            msg = f"samples must be a pandas.DataFrame, found {type(samples)}"
            raise TypeError(msg)

        if not all(col in samples.columns for col in _EXPECTED_COLUMNS):
            msg = f"samples must have (at least) columns {_EXPECTED_COLUMNS}, found {samples.columns}"
            raise ValueError(msg)

        if not samples["image_path"].apply(lambda p: Path(p).exists()).all():
            msg = "missing file path(s) in samples"
            raise FileNotFoundError(msg)

        self._samples = samples.sort_values(by="image_path", ignore_index=True)

    @property
    def category(self) -> str | None:
        """Get the category of the dataset / 데이터셋 카테고리 가져오기.

        Returns:
            str | None: Dataset category if set, else None. / 설정된 경우 데이터셋 카테고리, 그렇지 않으면 None
        """
        return self._category

    @category.setter
    def category(self, category: str) -> None:
        """Set the category of the dataset / 데이터셋 카테고리 설정.

        Args:
            category (str): Category to assign to the dataset. / 데이터셋에 할당할 카테고리
        """
        self._category = category

    @property
    def has_normal(self) -> bool:
        """Check if the dataset contains normal samples / 데이터셋에 정상 샘플이 포함되어 있는지 확인.

        Returns:
            bool: True if dataset contains normal samples, False otherwise. 
                / 데이터셋에 정상 샘플이 포함되어 있으면 True, 그렇지 않으면 False
        """
        return LabelName.NORMAL in list(self.samples.label_index)

    @property
    def has_anomalous(self) -> bool:
        """Check if the dataset contains anomalous samples / 데이터셋에 이상 샘플이 포함되어 있는지 확인.

        Returns:
            bool: True if dataset contains anomalous samples, False otherwise.
                / 데이터셋에 이상 샘플이 포함되어 있으면 True, 그렇지 않으면 False
        """
        return LabelName.ABNORMAL in list(self.samples.label_index)

    @property
    def task(self) -> TaskType:
        """Get the task type from the dataset / 데이터셋에서 작업 유형 가져오기.

        Returns:
            TaskType: Type of task (classification or segmentation).
                / 작업 유형 (분류 또는 세그멘테이션)

        Raises:
            ValueError: If task type is unknown. / 작업 유형을 알 수 없는 경우
        """
        return TaskType(self.samples.attrs["task"])

    def __getitem__(self, index: int) -> DatasetItem:
        """Get dataset item for the given index / 주어진 인덱스에 대한 데이터셋 항목 가져오기.

        **핵심 메서드**: 이 메서드는 PyTorch Dataset의 핵심으로, DataLoader에서 호출되어 배치를 구성합니다.
        **Key Method**: This is the core of PyTorch Dataset, called by DataLoader to construct batches.

        Args:
            index (int): Index to get the item. / 항목을 가져올 인덱스

        Returns:
            DatasetItem: Dataset item containing image and ground truth (if available).
                / 이미지와 실제값(사용 가능한 경우)을 포함하는 데이터셋 항목

        Example:
            >>> dataset = AnomalibDataset()
            >>> item = dataset[0]
            >>> isinstance(item.image, torch.Tensor)
            True
        """
        # DataFrame에서 해당 인덱스의 정보 추출 / Extract information for the given index from DataFrame
        image_path = self.samples.iloc[index].image_path
        mask_path = self.samples.iloc[index].mask_path
        label_index = self.samples.iloc[index].label_index

        # 이미지 읽기 (텐서로 변환) / Read the image (convert to tensor)
        image = read_image(image_path, as_tensor=True)

        # 마스크 초기화 / Initialize mask as None
        gt_mask = None

        # 작업 유형에 따른 처리 / Process based on task type
        if self.task == TaskType.SEGMENTATION:
            if label_index == LabelName.NORMAL:
                # 정상 샘플에 대해 zero 마스크 생성 / Create zero mask for normal samples
                gt_mask = Mask(torch.zeros(image.shape[-2:])).to(torch.uint8)
            elif label_index == LabelName.ABNORMAL:
                # 이상 샘플에 대해 마스크 읽기 / Read mask for anomalous samples
                gt_mask = read_mask(mask_path, as_tensor=True)
            # UNKNOWN의 경우 gt_mask는 None으로 유지 / For UNKNOWN, gt_mask remains None

        # 증강이 사용 가능한 경우 적용 / Apply augmentations if available
        if self.augmentations:
            if self.task == TaskType.CLASSIFICATION:
                # 분류 작업: 이미지만 증강 / Classification task: augment image only
                image = self.augmentations(image)
            elif self.task == TaskType.SEGMENTATION:
                # 세그멘테이션 작업: 이미지와 마스크 모두 증강 / Segmentation task: augment both image and mask
                # 이미지와 마스크가 모두 필요한 증강의 경우:
                # - UNKNOWN 샘플에는 임시 zero 마스크 사용
                # - 하지만 UNKNOWN의 경우 최종 gt_mask는 None으로 유지
                # For augmentations that require both image and mask:
                # - Use a temporary zero mask for UNKNOWN samples
                # - But preserve the final gt_mask as None for UNKNOWN
                temp_mask = gt_mask if gt_mask is not None else Mask(torch.zeros(image.shape[-2:])).to(torch.uint8)
                image, augmented_mask = self.augmentations(image, temp_mask)
                # 증강 전에 gt_mask가 None이 아니었던 경우에만 업데이트
                # Only update gt_mask if it wasn't None before augmentations
                if gt_mask is not None:
                    gt_mask = augmented_mask

        # gt_label 텐서 생성 (UNKNOWN의 경우 None) / Create gt_label tensor (None for UNKNOWN)
        gt_label = None if label_index == LabelName.UNKNOWN else torch.tensor(label_index)

        # 데이터셋 항목 반환 / Return the dataset item
        return ImageItem(
            image=image,
            gt_mask=gt_mask,
            gt_label=gt_label,
            image_path=image_path,
            mask_path=mask_path,
        )

    def __add__(self, other_dataset: "AnomalibDataset") -> "AnomalibDataset":
        """Concatenate this dataset with another dataset / 이 데이터셋을 다른 데이터셋과 연결.

        Args:
            other_dataset (AnomalibDataset): Dataset to concatenate with. / 연결할 데이터셋

        Returns:
            AnomalibDataset: Concatenated dataset. / 연결된 데이터셋

        Raises:
            TypeError: If datasets are not of the same type. / 데이터셋이 같은 유형이 아닌 경우

        Example:
            >>> dataset1 = AnomalibDataset()
            >>> dataset2 = AnomalibDataset()
            >>> combined = dataset1 + dataset2
        """
        if not isinstance(other_dataset, self.__class__):
            msg = "Cannot concatenate datasets that are not of the same type."
            raise TypeError(msg)
        dataset = copy.deepcopy(self)
        dataset.samples = pd.concat([self.samples, other_dataset.samples], ignore_index=True)
        return dataset

    @property
    def collate_fn(self) -> Callable:
        """Get the collate function for batching dataset items / 데이터셋 항목을 배칭하기 위한 collate 함수 가져오기.

        **교육적 설명**: collate_fn은 DataLoader에서 개별 샘플들을 배치로 결합할 때 사용되는 함수입니다.
        **Educational Note**: collate_fn is used by DataLoader to combine individual samples into batches.

        Returns:
            Callable: Collate function from ImageBatch. / ImageBatch의 collate 함수

        Note:
            By default, this returns ImageBatch's collate function. Override this property
            for other dataset types.
            기본적으로 ImageBatch의 collate 함수를 반환합니다. 다른 데이터셋 유형의 경우 이 속성을 오버라이드하세요.
        """
        return ImageBatch.collate
