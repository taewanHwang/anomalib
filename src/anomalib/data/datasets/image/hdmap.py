#!/usr/bin/env python3
"""HDMAP Dataset / HDMAP 데이터셋.

This module provides PyTorch Dataset implementation for the HDMAP (Health Data Map) dataset.
The dataset supports domain transfer learning tasks with multiple industrial domains.

이 모듈은 HDMAP (Health Data Map) 데이터셋에 대한 PyTorch Dataset 구현을 제공합니다.
데이터셋은 여러 산업 도메인을 통한 도메인 전이 학습 작업을 지원합니다.

Dataset Structure / 데이터셋 구조:
    datasets/HDMAP/{dataset_version}/
    ├── domain_A/
    │   ├── train/good/       # 정상 훈련 데이터
    │   └── test/
    │       ├── good/         # 정상 테스트 데이터
    │       └── fault/        # 결함 테스트 데이터
    ├── domain_B/
    ├── domain_C/
    └── domain_D/

Domains / 도메인:
    - domain_A: Class1 Sensor1 (정상 운영 조건)
    - domain_B: Class3 Sensor1 (다른 운영 조건)  
    - domain_C: Class1 Sensor3 (다른 센서 위치)
    - domain_D: Class3 Sensor3 (다른 운영 조건 + 센서 위치)

"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import LabelName, Split, validate_path
from anomalib.utils import deprecate

IMG_EXTENSIONS = (".tiff", ".tif", ".TIFF", ".TIF")
DOMAINS = (
    "domain_A",
    "domain_B", 
    "domain_C",
    "domain_D",
)


class HDMAPDataset(AnomalibDataset):
    """HDMAP dataset class / HDMAP 데이터셋 클래스.

    Dataset class for loading and processing HDMAP dataset images. Supports
    domain transfer learning and binary classification tasks (good vs fault).

    HDMAP 데이터셋 이미지를 로딩하고 처리하기 위한 데이터셋 클래스입니다. 
    도메인 전이 학습과 이진 분류 작업(정상 vs 결함)을 지원합니다.

    **상속 관계**: 이 클래스는 AnomalibDataset을 상속받아 HDMAP 데이터셋에 특화된 구현을 제공합니다.
    **Inheritance**: This class inherits from AnomalibDataset, providing HDMAP dataset-specific implementation.

    **4개 도메인**: domain_A, domain_B, domain_C, domain_D
    **4 Domains**: Representing different operational conditions and sensor configurations.

    **도메인 전이 학습**: 각 도메인을 독립적으로 로드하여 source/target domain 실험 가능
    **Domain Transfer Learning**: Load each domain independently for source/target domain experiments.

    Args:
        root (Path | str): Path to root directory containing the TIFF dataset.
            Defaults to ``"./datasets/HDMAP/10000_16bit_tiff_original"``.
            / TIFF 데이터셋을 포함하는 루트 디렉토리 경로. 기본값은 원본 TIFF 폴더
        domain (str): Domain name, must be one of ``DOMAINS``.
            Defaults to ``"domain_A"``.
            / 도메인 이름, ``DOMAINS`` 중 하나여야 함. 기본값은 ``"domain_A"``
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.
            / 입력 이미지에 적용할 증강. 기본값은 ``None``
        split (str | Split | None, optional): Dataset split - usually
            ``Split.TRAIN`` or ``Split.TEST``. Defaults to ``None``.
            / 데이터셋 분할 - 일반적으로 ``Split.TRAIN`` 또는 ``Split.TEST``. 기본값은 ``None``
        target_size (tuple[int, int] | None, optional): Target image size (H, W) for resizing.
            Defaults to ``None`` (no resizing).
            / 리사이즈 타겟 크기 (H, W). 기본값은 ``None`` (리사이즈 안함)
        resize_method (str, optional): Resize method - "resize", "black_padding", or "noise_padding".
            Defaults to ``"resize"``.
            / 리사이즈 방법 - "resize", "black_padding", "noise_padding" 중 하나

    Example:
        >>> from pathlib import Path
        >>> from anomalib.data.datasets import HDMAPDataset
        >>> # 단일 도메인 로드 / Load single domain
        >>> dataset = HDMAPDataset(
        ...     root=Path("./datasets/HDMAP/1000_8bit_resize_256x256"),
        ...     domain="domain_A",
        ...     split="train"
        ... )

        For classification tasks, each sample contains:
        분류 작업의 경우, 각 샘플은 다음을 포함합니다:

        >>> sample = dataset[0]
        >>> sample.image.shape      # torch.Size([3, 256, 256]) or configured size
        >>> sample.gt_label         # torch.tensor(0) for good, torch.tensor(1) for fault

        Domain Transfer Learning example:
        도메인 전이 학습 예시:

        >>> # Source domain for training / 훈련용 소스 도메인
        >>> source_dataset = HDMAPDataset(domain="domain_A", split="train")
        >>> # Target domain for testing / 테스트용 타겟 도메인  
        >>> target_dataset = HDMAPDataset(domain="domain_B", split="test")
    """

    def __init__(
        self,
        root: Path | str = "./datasets/HDMAP/10000_16bit_tiff_original",
        domain: str = "domain_A",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
        target_size: tuple[int, int] | None = None,
        resize_method: str = "resize",
    ) -> None:
        # 부모 클래스(AnomalibDataset) 초기화 / Initialize parent class (AnomalibDataset)
        super().__init__(augmentations=augmentations)

        # 도메인 유효성 검사 / Validate domain
        if domain not in DOMAINS:
            msg = f"Domain '{domain}' not found. Available domains: {DOMAINS}"
            raise ValueError(msg)

        # HDMAP 특화 속성 설정 / Set HDMAP-specific attributes
        self.root = Path(root)
        self.domain = domain  # 선택된 도메인 / Selected domain
        self.split = split  # 데이터 분할 (train/test) / Data split (train/test)
        self.target_size = target_size  # 타겟 이미지 크기 (H, W)
        self.resize_method = resize_method  # 리사이즈 방법
        
        # 리사이즈 방법 유효성 검사
        valid_methods = {"resize", "black_padding", "noise_padding"}
        if self.resize_method not in valid_methods:
            msg = f"Invalid resize_method '{self.resize_method}'. Valid methods: {valid_methods}"
            raise ValueError(msg)
        
        # 실제 데이터셋 구성 - make_hdmap_dataset 함수로 DataFrame 생성
        # Construct actual dataset - create DataFrame using make_hdmap_dataset function
        self.samples = make_hdmap_dataset(
            self.root,
            domain=self.domain,
            split=self.split,
            extensions=IMG_EXTENSIONS,
        )

    def load_and_resize_image(self, image_path: str) -> np.ndarray:
        """Load and resize TIFF image from file path.
        
        32bit TIFF 파일을 로딩하고 지정된 방법으로 리사이즈합니다.
        정규화는 수행하지 않습니다 (32bit TIFF는 이미 적절한 범위).
        
        Args:
            image_path: Path to TIFF image file
            
        Returns:
            Resized image as numpy array (C, H, W) in float32 format
        """
        # 32bit TIFF 파일 로딩 (정규화 불필요)
        with Image.open(image_path) as img:
            img_array = np.array(img).astype(np.float32)
                
            # 그레이스케일을 RGB로 변환
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array, img_array, img_array], axis=0)  # (3, H, W)
            elif len(img_array.shape) == 3:
                img_array = np.transpose(img_array, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        
        # 리사이즈 적용 (필요한 경우)
        if self.target_size is not None:
            img_array = self.resize_image(img_array)
            
        return img_array
    
    def resize_image(self, img: np.ndarray) -> np.ndarray:
        """
        이미지를 지정된 방법으로 리사이즈/패딩
        utils_data_loader_v2.py의 로직을 기반으로 구현
        
        Args:
            img: 입력 이미지 (C, H, W)
            
        Returns:
            resized_img: 리사이즈된 이미지 (C, target_H, target_W)
        """
        target_h, target_w = self.target_size
        current_h, current_w = img.shape[1], img.shape[2]
        
        if self.resize_method == 'resize':
            # 단순 리사이즈 (nearest neighbor)
            img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, C, H, W)
            resized = F.interpolate(img_tensor, size=(target_h, target_w), mode='nearest')
            return resized.squeeze(0).numpy()  # (C, H, W)
            
        elif self.resize_method == 'black_padding':
            # 검은색 패딩
            if current_h >= target_h and current_w >= target_w:
                # 크기가 더 크면 중앙 크롭
                start_h = (current_h - target_h) // 2
                start_w = (current_w - target_w) // 2
                return img[:, start_h:start_h+target_h, start_w:start_w+target_w]
            else:
                # 패딩 필요
                pad_h = max(0, target_h - current_h)
                pad_w = max(0, target_w - current_w)
                
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                
                img_tensor = torch.from_numpy(img)
                padded = F.pad(img_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)
                return padded.numpy()
                
        elif self.resize_method == 'noise_padding':
            # 노이즈 패딩
            if current_h >= target_h and current_w >= target_w:
                # 크기가 더 크면 중앙 크롭
                start_h = (current_h - target_h) // 2
                start_w = (current_w - target_w) // 2
                return img[:, start_h:start_h+target_h, start_w:start_w+target_w]
            else:
                # 노이즈 패딩 필요
                pad_h = max(0, target_h - current_h)
                pad_w = max(0, target_w - current_w)
                
                # 원본 데이터의 통계 계산
                img_mean = np.mean(img)
                img_std = np.std(img) * 0.1
                
                # 목표 크기의 빈 배열 생성
                result = np.zeros((img.shape[0], target_h, target_w), dtype=img.dtype)
                
                # 원본 이미지를 중앙에 배치
                start_h = pad_h // 2
                start_w = pad_w // 2
                result[:, start_h:start_h+current_h, start_w:start_w+current_w] = img
                
                # 패딩 영역에 노이즈 추가
                if pad_h > 0:
                    # 상단 패딩
                    if start_h > 0:
                        noise_top = np.random.normal(img_mean, img_std, (img.shape[0], start_h, target_w))
                        result[:, :start_h, :] = noise_top
                    # 하단 패딩
                    end_h = start_h + current_h
                    if end_h < target_h:
                        noise_bottom = np.random.normal(img_mean, img_std, (img.shape[0], target_h - end_h, target_w))
                        result[:, end_h:, :] = noise_bottom
                
                if pad_w > 0:
                    # 좌측 패딩
                    if start_w > 0:
                        noise_left = np.random.normal(img_mean, img_std, (img.shape[0], current_h, start_w))
                        result[:, start_h:start_h+current_h, :start_w] = noise_left
                    # 우측 패딩
                    end_w = start_w + current_w
                    if end_w < target_w:
                        noise_right = np.random.normal(img_mean, img_std, (img.shape[0], current_h, target_w - end_w))
                        result[:, start_h:start_h+current_h, end_w:] = noise_right
                
                return result
        else:
            raise ValueError(f"지원하지 않는 리사이즈 방법: {self.resize_method}")

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get dataset item with custom TIFF loading and resizing.
        
        Args:
            index: Sample index
            
        Returns:
            Dict containing image, label, and metadata
        """
        # 부모 클래스의 기본 __getitem__ 호출
        item = super().__getitem__(index)
        
        # 커스텀 TIFF 로딩으로 이미지 교체
        sample = self.samples.iloc[index]
        image_path = sample.image_path
        
        # 커스텀 이미지 로딩 및 리사이즈
        image = self.load_and_resize_image(image_path)
        
        # torch tensor로 변환
        custom_image = torch.from_numpy(image).float()
        
        # 새로운 item 생성 (기존 item의 모든 속성 복사 + 커스텀 이미지)
        from dataclasses import replace
        new_item = replace(item, image=custom_image)
        
        return new_item


def make_hdmap_dataset(
    root: str | Path,
    domain: str = "domain_A",
    split: str | Split | None = None,
    extensions: Sequence[str] | None = None,
) -> DataFrame:
    """Create HDMAP samples by parsing the domain-specific directory structure / 도메인별 디렉토리 구조를 파싱하여 HDMAP 샘플 생성.

    **핵심 함수**: 이 함수는 HDMAP의 도메인별 디렉토리 구조를 분석하여 pandas DataFrame을 생성합니다.
    **Key Function**: This function analyzes HDMAP's domain-specific directory structure to create a pandas DataFrame.

    The files are expected to follow the structure:
    파일들은 다음 구조를 따라야 합니다:
        ``path/to/dataset/domain_X/split/label/XXXXXX.png``

    Args:
        root (Path | str): Path to dataset root directory / 데이터셋 루트 디렉토리 경로
        domain (str): Domain name (domain_A, domain_B, domain_C, domain_D)
            / 도메인 이름 (domain_A, domain_B, domain_C, domain_D)
        split (str | Split | None, optional): Dataset split (train or test)
            Defaults to ``None``. / 데이터셋 분할 (train 또는 test). 기본값은 ``None``
        extensions (Sequence[str] | None, optional): Valid file extensions
            Defaults to ``None``. / 유효한 파일 확장자. 기본값은 ``None``

    Returns:
        DataFrame: Dataset samples with columns: / 다음 열을 가진 데이터셋 샘플:
            - path: Base path to dataset / 데이터셋 기본 경로
            - domain: Domain name / 도메인 이름
            - split: Dataset split (train/test) / 데이터셋 분할 (train/test)
            - label: Class label (good/fault) / 클래스 라벨 (good/fault)
            - image_path: Path to image file / 이미지 파일 경로
            - label_index: Numeric label (0=good, 1=fault) / 숫자 라벨 (0=정상, 1=결함)

    Example:
        >>> root = Path("./datasets/HDMAP/1000_8bit_resize_256x256")
        >>> samples = make_hdmap_dataset(root, domain="domain_A", split="train")
        >>> samples.head()
           path              domain   split label image_path                    label_index
        0  datasets/HDMAP/... domain_A train good  [...]/domain_A/train/good/000000.png  0
        1  datasets/HDMAP/... domain_A train good  [...]/domain_A/train/good/000001.png  0

    Raises:
        RuntimeError: If no valid images are found / 유효한 이미지를 찾을 수 없는 경우
        ValueError: If domain is not valid / 도메인이 유효하지 않은 경우
    """
    # 기본 확장자 설정 / Set default extensions
    if extensions is None:
        extensions = IMG_EXTENSIONS

    # 도메인 유효성 검사 / Validate domain
    if domain not in DOMAINS:
        msg = f"Domain '{domain}' not found. Available domains: {DOMAINS}"
        raise ValueError(msg)

    # 경로 검증 및 도메인별 경로 설정 / Validate path and set domain-specific path
    root = validate_path(root)
    domain_path = root / domain

    if not domain_path.exists():
        msg = f"Domain path '{domain_path}' does not exist"
        raise RuntimeError(msg)

    # 도메인 내 파일 스캔 (domain_X/split/label/image.png 패턴)
    # Scan files within domain (domain_X/split/label/image.png pattern)
    samples_list = []
    
    # 파일 경로를 정렬하여 스캔
    all_files = sorted(domain_path.glob("**/*")) 
    
    for file_path in all_files:
        if file_path.suffix in extensions:
            # 파일 경로에서 정보 추출: domain_X/split/label/filename.png
            # Extract info from file path: domain_X/split/label/filename.png
            parts = file_path.parts
            if len(parts) >= 4:  # 최소 4개 레벨 필요 / Need at least 4 levels
                # root/domain_X/split/label/filename.png에서 마지막 3개 추출
                # Extract last 3 from root/domain_X/split/label/filename.png
                split_name = parts[-3]  # train or test
                label_name = parts[-2]  # good or fault
                filename = parts[-1]    # 000000.png
                
                samples_list.append((str(root), domain, split_name, label_name, str(file_path)))

    if not samples_list:
        msg = f"Found 0 images in {domain_path}"
        raise RuntimeError(msg)

    # DataFrame 생성 / Create DataFrame
    samples = DataFrame(samples_list, columns=["path", "domain", "split", "label", "image_path"])

    # 라벨 인덱스 생성: 정상(0), 결함(1) / Create label index: good(0), fault(1)
    samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label == "fault"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)

    # HDMAP은 마스크가 없으므로 classification 작업으로 설정
    # HDMAP has no masks, so set as classification task
    samples["mask_path"] = ""  # 빈 문자열로 마스크 경로 설정 / Set empty string for mask path
    
    # 작업 유형 설정: classification (마스크 없음)
    # Set task type: classification (no masks)
    samples.attrs["task"] = "classification"

    # 특정 분할만 필터링 (요청된 경우) / Filter for specific split (if requested)
    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples
