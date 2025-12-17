"""Convert HDMAP TIFF images to PNG format.

Conversion settings:
- Square size (max dimension)
- Nearest neighbor interpolation (preserve grid pattern)
- 3 channels (RGB)
- 8-bit PNG output
"""

import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_single_image(args: tuple) -> tuple[str, bool, str]:
    """Convert a single TIFF image to PNG.

    Args:
        args: Tuple of (input_path, output_path, target_size)

    Returns:
        Tuple of (input_path, success, error_message)
    """
    input_path, output_path, target_size = args

    try:
        # Load TIFF (32-bit float, mode 'F')
        img = Image.open(input_path)
        arr = np.array(img, dtype=np.float32)

        # Normalize to 0-255 range
        # Data is already minmax normalized to ~0-1 range
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr_normalized = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr_normalized = arr

        # Convert to 8-bit
        arr_uint8 = (arr_normalized * 255).astype(np.uint8)

        # Convert to PIL Image (grayscale)
        img_gray = Image.fromarray(arr_uint8, mode='L')

        # Resize to square using NEAREST interpolation (preserve grid pattern)
        h, w = arr.shape
        max_dim = max(h, w)

        if target_size is not None:
            final_size = target_size
        else:
            final_size = max_dim

        # Resize to square
        img_resized = img_gray.resize((final_size, final_size), Image.Resampling.NEAREST)

        # Convert to RGB (3 channels)
        img_rgb = img_resized.convert('RGB')

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
    target_size: int | None = None,
    num_workers: int = 8,
) -> None:
    """Convert all TIFF images in a directory to PNG.

    Args:
        input_dir: Input directory containing TIFF files
        output_dir: Output directory for PNG files
        target_size: Target square size (None = use max dimension)
        num_workers: Number of parallel workers
    """
    # Find all TIFF files
    tiff_files = list(input_dir.rglob("*.tiff")) + list(input_dir.rglob("*.tif"))
    logger.info(f"Found {len(tiff_files)} TIFF files")

    if not tiff_files:
        logger.warning("No TIFF files found!")
        return

    # Prepare conversion tasks
    tasks = []
    for tiff_path in tiff_files:
        # Preserve directory structure
        rel_path = tiff_path.relative_to(input_dir)
        output_path = output_dir / rel_path.with_suffix('.png')
        tasks.append((str(tiff_path), str(output_path), target_size))

    # Process with parallel workers
    success_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(convert_single_image, task): task for task in tasks}

        with tqdm(total=len(tasks), desc="Converting") as pbar:
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
    parser = argparse.ArgumentParser(description="Convert HDMAP TIFF to PNG")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_tiff_minmax",
        help="Input directory containing TIFF files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png",
        help="Output directory for PNG files",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=256,
        help="Target square size (default: 256)",
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
    logger.info(f"Target size: {args.target_size}x{args.target_size}")
    logger.info(f"Interpolation: NEAREST (grid preserved)")

    convert_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        target_size=args.target_size,
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
