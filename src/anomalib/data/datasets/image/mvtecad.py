# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MVTec AD Dataset / MVTec AD 데이터셋.

This module provides PyTorch Dataset implementation for the MVTec AD dataset. The
dataset will be downloaded and extracted automatically if not found locally.

이 모듈은 MVTec AD 데이터셋에 대한 PyTorch Dataset 구현을 제공합니다. 데이터셋이 로컬에서 발견되지 않으면
자동으로 다운로드되고 추출됩니다.

The dataset contains 15 categories of industrial objects with both normal and
anomalous samples. Each category includes RGB images and pixel-level ground truth
masks for anomaly segmentation.

데이터셋은 정상 및 이상 샘플이 모두 포함된 15개 카테고리의 산업 객체를 포함합니다. 각 카테고리에는
이상 세그멘테이션을 위한 RGB 이미지와 픽셀 수준의 실제 마스크가 포함됩니다.

**교육적 가치**: MVTec AD는 이상 탐지 연구에서 가장 널리 사용되는 벤치마크 데이터셋으로,
산업 환경에서의 제품 결함 탐지를 시뮬레이션합니다.
**Educational Value**: MVTec AD is the most widely used benchmark dataset in anomaly detection research,
simulating product defect detection in industrial environments.

License:
    MVTec AD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference:
    Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., & Steger, C. (2021).
    The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for
    Unsupervised Anomaly Detection. International Journal of Computer Vision,
    129(4), 1038-1059.

    Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019). MVTec AD —
    A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. In
    IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
    9584-9592.
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
CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)


class MVTecADDataset(AnomalibDataset):
    """MVTec AD dataset class / MVTec AD 데이터셋 클래스.

    Dataset class for loading and processing MVTec AD dataset images. Supports
    both classification and segmentation tasks.

    MVTec AD 데이터셋 이미지를 로딩하고 처리하기 위한 데이터셋 클래스입니다. 분류와 세그멘테이션 작업을 모두 지원합니다.

    **상속 관계**: 이 클래스는 AnomalibDataset을 상속받아 MVTec AD 데이터셋에 특화된 구현을 제공합니다.
    **Inheritance**: This class inherits from AnomalibDataset, providing MVTec AD dataset-specific implementation.

    **15개 카테고리**: bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, 
    screw, tile, toothbrush, transistor, wood, zipper
    **15 Categories**: Industrial objects representing various defect types in manufacturing.

    Args:
        root (Path | str): Path to root directory containing the dataset.
            Defaults to ``"./datasets/MVTecAD"``.
            / 데이터셋을 포함하는 루트 디렉토리 경로. 기본값은 ``"./datasets/MVTecAD"``
        category (str): Category name, must be one of ``CATEGORIES``.
            Defaults to ``"bottle"``.
            / 카테고리 이름, ``CATEGORIES`` 중 하나여야 함. 기본값은 ``"bottle"``
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.
            / 입력 이미지에 적용할 증강. 기본값은 ``None``
        split (str | Split | None, optional): Dataset split - usually
            ``Split.TRAIN`` or ``Split.TEST``. Defaults to ``None``.
            / 데이터셋 분할 - 일반적으로 ``Split.TRAIN`` 또는 ``Split.TEST``. 기본값은 ``None``

    Example:
        >>> from pathlib import Path
        >>> from anomalib.data.datasets import MVTecADDataset
        >>> dataset = MVTecADDataset(
        ...     root=Path("./datasets/MVTecAD"),
        ...     category="bottle",
        ...     split="train"
        ... )

        For classification tasks, each sample contains:
        분류 작업의 경우, 각 샘플은 다음을 포함합니다:

        >>> sample = dataset[0]
        >>> list(sample.keys())
        ['image_path', 'label', 'image']

        For segmentation tasks, samples also include mask paths and masks:
        세그멘테이션 작업의 경우, 샘플에는 마스크 경로와 마스크도 포함됩니다:

        >>> dataset.task = "segmentation"
        >>> sample = dataset[0]
        >>> list(sample.keys())
        ['image_path', 'label', 'image', 'mask_path', 'mask']

        Images are PyTorch tensors with shape ``(C, H, W)``, masks have shape ``(H, W)``:
        이미지는 ``(C, H, W)`` 형태의 PyTorch 텐서이고, 마스크는 ``(H, W)`` 형태입니다:

        >>> sample["image"].shape, sample["mask"].shape
        (torch.Size([3, 256, 256]), torch.Size([256, 256]))
    """

    def __init__(
        self,
        root: Path | str = "./datasets/MVTecAD",
        category: str = "bottle",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        # 부모 클래스(AnomalibDataset) 초기화 / Initialize parent class (AnomalibDataset)
        super().__init__(augmentations=augmentations)

        # MVTec AD 특화 속성 설정 / Set MVTec AD-specific attributes
        self.root_category = Path(root) / Path(category)  # 카테고리별 데이터 경로 / Category-specific data path
        self.category = category  # 선택된 카테고리 / Selected category
        self.split = split  # 데이터 분할 (train/test) / Data split (train/test)
        
        # 실제 데이터셋 구성 - make_mvtec_ad_dataset 함수로 DataFrame 생성
        # Construct actual dataset - create DataFrame using make_mvtec_ad_dataset function
        self.samples = make_mvtec_ad_dataset(
            self.root_category,
            split=self.split,
            extensions=IMG_EXTENSIONS,  # .png, .PNG 파일만 처리 / Process only .png, .PNG files
        )


def make_mvtec_ad_dataset(
    root: str | Path,
    split: str | Split | None = None,
    extensions: Sequence[str] | None = None,
) -> DataFrame:
    """Create MVTec AD samples by parsing the data directory structure / 데이터 디렉토리 구조를 파싱하여 MVTec AD 샘플 생성.

    **핵심 함수**: 이 함수는 MVTec AD의 디렉토리 구조를 분석하여 pandas DataFrame을 생성합니다.
    **Key Function**: This function analyzes MVTec AD directory structure to create a pandas DataFrame.

    The files are expected to follow the structure:
    파일들은 다음 구조를 따라야 합니다:
        ``path/to/dataset/split/category/image_filename.png``
        ``path/to/dataset/ground_truth/category/mask_filename.png``

    Args:
        root (Path | str): Path to dataset root directory / 데이터셋 루트 디렉토리 경로
        split (str | Split | None, optional): Dataset split (train or test)
            Defaults to ``None``. / 데이터셋 분할 (train 또는 test). 기본값은 ``None``
        extensions (Sequence[str] | None, optional): Valid file extensions
            Defaults to ``None``. / 유효한 파일 확장자. 기본값은 ``None``

    Returns:
        DataFrame: Dataset samples with columns: / 다음 열을 가진 데이터셋 샘플:
            - path: Base path to dataset / 데이터셋 기본 경로
            - split: Dataset split (train/test) / 데이터셋 분할 (train/test)
            - label: Class label / 클래스 라벨
            - image_path: Path to image file / 이미지 파일 경로
            - mask_path: Path to mask file (if available) / 마스크 파일 경로 (사용 가능한 경우)
            - label_index: Numeric label (0=normal, 1=abnormal) / 숫자 라벨 (0=정상, 1=이상)

    Example:
        >>> root = Path("./datasets/MVTecAD/bottle")
        >>> samples = make_mvtec_dataset(root, split="train")
        >>> samples.head()
           path                split label image_path           mask_path label_index
        0  datasets/MVTecAD/bottle train good  [...]/good/105.png           0
        1  datasets/MVTecAD/bottle train good  [...]/good/017.png           0

    Raises:
        RuntimeError: If no valid images are found / 유효한 이미지를 찾을 수 없는 경우
        MisMatchError: If anomalous images and masks don't match / 이상 이미지와 마스크가 일치하지 않는 경우
    """
    # 기본 확장자 설정 / Set default extensions
    if extensions is None:
        extensions = IMG_EXTENSIONS

    # 경로 검증 및 파일 스캔 / Validate path and scan files
    root = validate_path(root)
    samples_list = [(str(root), *f.parts[-3:]) for f in root.glob(r"**/*") if f.suffix in extensions]
    if not samples_list:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    # DataFrame 생성 / Create DataFrame
    samples = DataFrame(samples_list, columns=["path", "split", "label", "image_path"])

    # 이미지 경로를 절대 경로로 변환 / Modify image_path column by converting to absolute path
    samples["image_path"] = samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path

    # 라벨 인덱스 생성: 정상(0), 이상(1) / Create label index for normal (0) and anomalous (1) images
    samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)

    # 마스크와 샘플을 분리 / Separate masks from samples
    mask_samples = samples.loc[samples.split == "ground_truth"].sort_values(
        by="image_path",
        ignore_index=True,
    )
    samples = samples[samples.split != "ground_truth"].sort_values(
        by="image_path",
        ignore_index=True,
    )

    # 이상 테스트 이미지에 마스크 경로 할당 / Assign mask paths to anomalous test images
    samples["mask_path"] = None
    samples.loc[
        (samples.split == "test") & (samples.label_index == LabelName.ABNORMAL),
        "mask_path",
    ] = mask_samples.image_path.to_numpy()

    # 올바른 마스크 파일이 올바른 테스트 이미지와 연결되었는지 확인
    # Assert that the right mask files are associated with the right test images
    abnormal_samples = samples.loc[samples.label_index == LabelName.ABNORMAL]
    if (
        len(abnormal_samples)
        and not abnormal_samples.apply(
            lambda x: Path(x.image_path).stem in Path(x.mask_path).stem,
            axis=1,
        ).all()
    ):
        msg = (
            "Mismatch between anomalous images and ground truth masks. Make sure "
            "mask files in 'ground_truth' folder follow the same naming "
            "convention as the anomalous images (e.g. image: '000.png', "
            "mask: '000.png' or '000_mask.png')."
        )
        raise MisMatchError(msg)

    # 작업 유형 추론: 마스크가 있으면 segmentation, 없으면 classification
    # Infer the task type: segmentation if masks available, classification otherwise
    samples.attrs["task"] = "classification" if (samples["mask_path"] == "").all() else "segmentation"

    # 특정 분할만 필터링 (요청된 경우) / Filter for specific split (if requested)
    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples


@deprecate(since="2.1.0", remove="2.3.0", use="MVTecADDataset")
class MVTecDataset(MVTecADDataset):
    """MVTec dataset class (Deprecated) / MVTec 데이터셋 클래스 (사용 중단).

    This class is deprecated and will be removed in a future version.
    Please use MVTecADDataset instead.

    이 클래스는 사용이 중단되었으며 향후 버전에서 제거될 예정입니다.
    대신 MVTecADDataset을 사용해 주세요.

    **교육적 팁**: 소프트웨어 개발에서 버전 관리와 호환성을 위해 사용하는 deprecation 패턴입니다.
    **Educational Tip**: This is a deprecation pattern used in software development for version management and compatibility.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
