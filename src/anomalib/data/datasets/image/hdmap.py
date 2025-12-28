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

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from PIL import Image
from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import LabelName, Split, validate_path
from anomalib.utils import deprecate

logger = logging.getLogger(__name__)

IMG_EXTENSIONS = (".tiff", ".tif", ".TIFF", ".TIF", ".png", ".PNG")
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
        resize_method (str, optional): Resize method. Available options:
            - ``"resize"``: Nearest neighbor interpolation (default)
            - ``"resize_bilinear"``: Bilinear interpolation (smoother)
            - ``"resize_aspect_padding"``: Preserve aspect ratio, scale to fit, then pad
            - ``"black_padding"``: No resize, just add black padding
            - ``"noise_padding"``: No resize, just add noise padding
            Defaults to ``"resize"``.
            / 리사이즈 방법:
            - ``"resize"``: Nearest neighbor 보간 (기본값)
            - ``"resize_bilinear"``: Bilinear 보간 (부드러움)
            - ``"resize_aspect_padding"``: 비율 유지 + 최대 확대 + 패딩
            - ``"black_padding"``: 리사이즈 없이 검은색 패딩만
            - ``"noise_padding"``: 리사이즈 없이 노이즈 패딩만

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
        valid_methods = {"resize", "resize_bilinear", "resize_aspect_padding", "black_padding", "noise_padding"}
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

        # 데이터 로딩 검증 로깅 플래그 / Data loading verification logging flag
        self._logged_data_stats = False
        self._log_interval = 1000  # 이후 n개마다 로깅

    def load_and_resize_image(self, image_path: str) -> np.ndarray:
        """Load and resize image from file path.

        TIFF(32bit float) 파일은 tifffile로 로딩하여 float32 정밀도를 유지합니다.
        PNG(16bit) 파일은 PIL로 로딩하고 [0, 1] 범위로 정규화합니다.

        **중요**: TIFF 파일은 tifffile을 사용하여 로딩합니다. 이는 PIL보다 정확하게
        float32 데이터를 보존합니다 (NO clipping, 원본 값 범위 유지).

        Args:
            image_path: Path to image file

        Returns:
            Resized image as numpy array (C, H, W) in float32 format
        """
        # TIFF 파일인지 확인
        if image_path.lower().endswith(('.tiff', '.tif')):
            # tifffile로 TIFF 로딩 (float32 정밀도 유지, NO clipping)
            img_array = tifffile.imread(image_path).astype(np.float32)
        else:
            # 기타 파일은 PIL로 로딩
            with Image.open(image_path) as img:
                if img.mode == 'I;16':
                    # 16bit PNG (0~65535 → 0~1 정규화)
                    img_array = np.array(img, dtype=np.uint16).astype(np.float32) / 65535.0
                else:
                    # 기타 모드는 float32로 변환
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

        elif self.resize_method == 'resize_bilinear':
            # Bilinear interpolation 리사이즈 (부드러운 보간)
            img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, C, H, W)
            resized = F.interpolate(img_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
            return resized.squeeze(0).numpy()  # (C, H, W)

        elif self.resize_method == 'resize_aspect_padding':
            # Aspect ratio 유지하면서 최대 확대 후 패딩
            # 1. 비율 계산: 가로/세로 중 더 큰 스케일 팩터 사용
            scale_h = target_h / current_h
            scale_w = target_w / current_w
            scale = min(scale_h, scale_w)  # aspect ratio 유지하면서 fit

            # 2. 새 크기 계산
            new_h = int(current_h * scale)
            new_w = int(current_w * scale)

            # 3. Bilinear interpolation으로 리사이즈
            img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, C, H, W)
            resized = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
            resized = resized.squeeze(0)  # (C, new_h, new_w)

            # 4. 패딩 추가 (중앙 배치)
            pad_h = target_h - new_h
            pad_w = target_w - new_w
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)
            return padded.numpy()  # (C, target_h, target_w)

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

        # 데이터 로딩 검증 로깅 (첫 번째 샘플에서만)
        if not self._logged_data_stats:
            self._log_data_statistics(custom_image, image_path)
            self._logged_data_stats = True

        # 새로운 item 생성 (기존 item의 모든 속성 복사 + 커스텀 이미지)
        from dataclasses import replace
        new_item = replace(item, image=custom_image)

        return new_item

    def _log_data_statistics(self, image: torch.Tensor, image_path: str) -> None:
        """Log data loading statistics for verification.

        데이터 로딩이 올바르게 수행되었는지 확인하기 위한 통계 로깅.
        - 값 범위 (min, max)
        - 평균, 표준편차
        - 데이터 타입
        - NaN/Inf 확인

        Args:
            image: Loaded image tensor (C, H, W)
            image_path: Path to the image file
        """
        is_tiff = image_path.lower().endswith(('.tiff', '.tif'))
        file_type = "TIFF" if is_tiff else "PNG"

        # 통계 계산
        img_min = image.min().item()
        img_max = image.max().item()
        img_mean = image.mean().item()
        img_std = image.std().item()
        has_nan = torch.isnan(image).any().item()
        has_inf = torch.isinf(image).any().item()

        # 클리핑 여부 확인 (TIFF는 [0,1] 범위 밖도 허용)
        is_clipped = (img_min >= 0.0 and img_max <= 1.0)
        clipping_status = "CLIPPED to [0,1]" if is_clipped else "NOT clipped (raw values)"

        # 상태 메시지 결정
        if is_tiff and not is_clipped:
            status_msg = "  [OK] TIFF float32 preserved (NO clipping)"
        elif is_tiff and is_clipped:
            status_msg = "  [WARN] TIFF values in [0,1] - possible clipping or normalized data"
        else:
            status_msg = f"  [INFO] {file_type} loaded normally"

        # domain 속성 안전하게 가져오기 (AllDomainsHDMAPDataset은 domains 사용)
        domain_info = getattr(self, 'domain', None) or getattr(self, 'domains', 'unknown')
        if isinstance(domain_info, list):
            domain_info = ','.join(domain_info)

        logger.info(
            f"[HDMAPDataset] Data Loading Verification ({domain_info}/{self.split}):\n"
            f"  File: {Path(image_path).name} ({file_type})\n"
            f"  Shape: {tuple(image.shape)}, dtype: {image.dtype}\n"
            f"  Range: [{img_min:.6f}, {img_max:.6f}] - {clipping_status}\n"
            f"  Mean: {img_mean:.6f}, Std: {img_std:.6f}\n"
            f"  NaN: {has_nan}, Inf: {has_inf}\n"
            f"{status_msg}"
        )


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
