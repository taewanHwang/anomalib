"""Convert PNG images to LBP (Local Binary Pattern) representation.

Input: 1000_png_bilinear (256x256 RGB PNG images with bilinear interpolation)
Output: 1000_png_bilinear_lbp (256x256 RGB PNG images - LBP texture)

Usage:
    python convert_png_to_lbp.py \
        --input-dir datasets/HDMAP/1000_png_bilinear \
        --output-dir datasets/HDMAP/1000_png_bilinear_lbp
"""

import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_lbp(
    image_array: np.ndarray,
    n_points: int = 8,
    radius: int = 1,
    method: str = "uniform",
) -> np.ndarray:
    """Apply Local Binary Pattern to grayscale image.

    Args:
        image_array: Grayscale image array (H, W)
        n_points: Number of circularly symmetric neighbor points
        radius: Radius of circle
        method: LBP method ('uniform', 'default', 'ror', 'nri_uniform', 'var')

    Returns:
        LBP image normalized to 0-255
    """
    # Apply LBP
    lbp = local_binary_pattern(image_array, P=n_points, R=radius, method=method)

    # Normalize to 0-255
    lbp_min, lbp_max = lbp.min(), lbp.max()
    if lbp_max > lbp_min:
        lbp_normalized = (lbp - lbp_min) / (lbp_max - lbp_min) * 255
    else:
        lbp_normalized = lbp * 0

    return lbp_normalized.astype(np.uint8)


def convert_single_image(args: tuple) -> tuple[str, bool, str]:
    """Convert a single PNG image to LBP representation.

    Args:
        args: Tuple of (input_path, output_path, n_points, radius, method)

    Returns:
        Tuple of (input_path, success, error_message)
    """
    input_path, output_path, n_points, radius, method = args

    try:
        # Load PNG image
        img = Image.open(input_path)

        # Convert to grayscale if RGB
        if img.mode == 'RGB':
            img_gray = img.convert('L')
        elif img.mode == 'L':
            img_gray = img
        else:
            img_gray = img.convert('L')

        # Convert to numpy array
        arr = np.array(img_gray, dtype=np.float32)

        # Apply LBP
        lbp_result = apply_lbp(arr, n_points=n_points, radius=radius, method=method)

        # Convert to PIL Image (grayscale)
        img_lbp = Image.fromarray(lbp_result, mode='L')

        # Convert to RGB (3 channels for consistency)
        img_rgb = img_lbp.convert('RGB')

        # Save as PNG
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img_rgb.save(output_path, 'PNG')

        return str(input_path), True, ""

    except Exception as e:
        return str(input_path), False, str(e)


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    n_points: int = 8,
    radius: int = 1,
    method: str = "uniform",
    num_workers: int = 8,
) -> None:
    """Convert all PNG images in a directory to LBP.

    Args:
        input_dir: Input directory containing PNG files
        output_dir: Output directory for LBP PNG files
        n_points: Number of circularly symmetric neighbor points
        radius: Radius of circle
        method: LBP method
        num_workers: Number of parallel workers
    """
    # Find all PNG files
    png_files = list(input_dir.rglob("*.png"))
    logger.info(f"Found {len(png_files)} PNG files")

    if not png_files:
        logger.warning("No PNG files found!")
        return

    # Prepare conversion tasks
    tasks = []
    for png_path in png_files:
        # Preserve directory structure
        rel_path = png_path.relative_to(input_dir)
        output_path = output_dir / rel_path
        tasks.append((str(png_path), str(output_path), n_points, radius, method))

    # Process with parallel workers
    success_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(convert_single_image, task): task for task in tasks}

        with tqdm(total=len(tasks), desc=f"Converting to LBP (P={n_points}, R={radius})") as pbar:
            for future in as_completed(futures):
                input_path, success, error_msg = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    logger.error(f"Failed to convert {input_path}: {error_msg}")
                pbar.update(1)

    logger.info(f"Conversion complete: {success_count} success, {error_count} errors")


def main():
    parser = argparse.ArgumentParser(description="Convert PNG to LBP representation")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png_bilinear",
        help="Input directory containing PNG files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png_bilinear_lbp",
        help="Output directory for LBP PNG files",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=8,
        help="Number of circularly symmetric neighbor points (default: 8)",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=1,
        help="Radius of circle (default: 1)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="uniform",
        choices=["uniform", "default", "ror", "nri_uniform", "var"],
        help="LBP method (default: uniform)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"LBP parameters: P={args.n_points}, R={args.radius}, method={args.method}")

    convert_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        n_points=args.n_points,
        radius=args.radius,
        method=args.method,
        num_workers=args.num_workers,
    )

    # Show sample output
    sample_outputs = list(output_dir.rglob("*.png"))[:3]
    if sample_outputs:
        logger.info("\nSample outputs:")
        for png_path in sample_outputs:
            img = Image.open(png_path)
            logger.info(f"  {png_path.name}: {img.size}, mode={img.mode}")


if __name__ == "__main__":
    main()
