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

from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import LabelName, Split, validate_path
from anomalib.utils import deprecate

IMG_EXTENSIONS = (".png", ".PNG")
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
        root (Path | str): Path to root directory containing the dataset.
            Defaults to ``"./datasets/HDMAP/1000_8bit_resize_256x256"``.
            / 데이터셋을 포함하는 루트 디렉토리 경로. 기본값은 HDMAP 전처리된 폴더
        domain (str): Domain name, must be one of ``DOMAINS``.
            Defaults to ``"domain_A"``.
            / 도메인 이름, ``DOMAINS`` 중 하나여야 함. 기본값은 ``"domain_A"``
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.
            / 입력 이미지에 적용할 증강. 기본값은 ``None``
        split (str | Split | None, optional): Dataset split - usually
            ``Split.TRAIN`` or ``Split.TEST``. Defaults to ``None``.
            / 데이터셋 분할 - 일반적으로 ``Split.TRAIN`` 또는 ``Split.TEST``. 기본값은 ``None``

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
        root: Path | str = "./datasets/HDMAP/1000_8bit_resize_256x256",
        domain: str = "domain_A",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
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
        
        # 실제 데이터셋 구성 - make_hdmap_dataset 함수로 DataFrame 생성
        # Construct actual dataset - create DataFrame using make_hdmap_dataset function
        self.samples = make_hdmap_dataset(
            self.root,
            domain=self.domain,
            split=self.split,
            extensions=IMG_EXTENSIONS,
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
    for file_path in domain_path.glob("**/*"):
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
