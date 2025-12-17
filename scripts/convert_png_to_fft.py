"""Convert PNG images to 2D FFT magnitude or phase spectrum.

Input: 1000_png (256x256 RGB PNG images)
Output:
  - 1000_png_2dfft (256x256 RGB PNG images - FFT magnitude spectrum)
  - 1000_png_2dfft_phase (256x256 RGB PNG images - FFT phase spectrum)

The FFT transformation:
1. Convert RGB to grayscale
2. Apply 2D FFT
3. Shift zero-frequency to center
4. Compute magnitude (log scale) or phase
5. Normalize to 0-255
6. Convert back to RGB (3 channels)

Usage:
    # Magnitude (default)
    python convert_png_to_fft.py --output-type magnitude

    # Phase
    python convert_png_to_fft.py --output-type phase --output-dir datasets/HDMAP/1000_png_2dfft_phase
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


def apply_2d_fft(image_array: np.ndarray, output_type: str = "magnitude") -> np.ndarray:
    """Apply 2D FFT and return magnitude or phase spectrum.

    Args:
        image_array: Grayscale image array (H, W)
        output_type: "magnitude" or "phase"

    Returns:
        Spectrum normalized to 0-255
    """
    # Apply 2D FFT
    f_transform = np.fft.fft2(image_array)

    # Shift zero-frequency component to center
    f_shift = np.fft.fftshift(f_transform)

    if output_type == "magnitude":
        # Compute magnitude spectrum (log scale for better visualization)
        magnitude = np.abs(f_shift)

        # Log scale (add 1 to avoid log(0))
        spectrum = np.log1p(magnitude)

    elif output_type == "phase":
        # Compute phase spectrum (-pi to pi)
        phase = np.angle(f_shift)

        # Shift from [-pi, pi] to [0, 2*pi] for better visualization
        spectrum = phase + np.pi

        # Log scale for phase (optional, but keeps consistency)
        # Phase is already bounded, so log scale helps spread the values
        spectrum = np.log1p(spectrum)

    else:
        raise ValueError(f"Unknown output_type: {output_type}. Use 'magnitude' or 'phase'.")

    # Normalize to 0-255
    spec_min, spec_max = spectrum.min(), spectrum.max()
    if spec_max > spec_min:
        spectrum_normalized = (spectrum - spec_min) / (spec_max - spec_min) * 255
    else:
        spectrum_normalized = spectrum * 0

    return spectrum_normalized.astype(np.uint8)


def convert_single_image(args: tuple) -> tuple[str, bool, str]:
    """Convert a single PNG image to FFT magnitude or phase spectrum.

    Args:
        args: Tuple of (input_path, output_path, output_type)

    Returns:
        Tuple of (input_path, success, error_message)
    """
    input_path, output_path, output_type = args

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

        # Apply 2D FFT
        fft_spectrum = apply_2d_fft(arr, output_type=output_type)

        # Convert to PIL Image (grayscale)
        img_fft = Image.fromarray(fft_spectrum, mode='L')

        # Convert to RGB (3 channels for consistency)
        img_rgb = img_fft.convert('RGB')

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
    output_type: str = "magnitude",
    num_workers: int = 8,
) -> None:
    """Convert all PNG images in a directory to FFT magnitude or phase spectrum.

    Args:
        input_dir: Input directory containing PNG files
        output_dir: Output directory for FFT PNG files
        output_type: "magnitude" or "phase"
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
        tasks.append((str(png_path), str(output_path), output_type))

    # Process with parallel workers
    success_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(convert_single_image, task): task for task in tasks}

        with tqdm(total=len(tasks), desc="Converting to FFT") as pbar:
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
    parser = argparse.ArgumentParser(description="Convert PNG to 2D FFT magnitude or phase spectrum")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png",
        help="Input directory containing PNG files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for FFT PNG files (default: auto-generated based on output-type)",
    )
    parser.add_argument(
        "--output-type",
        type=str,
        default="magnitude",
        choices=["magnitude", "phase"],
        help="Output type: 'magnitude' or 'phase' (default: magnitude)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # Auto-generate output directory if not specified
    if args.output_dir is None:
        if args.output_type == "magnitude":
            output_dir = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png_2dfft")
        else:  # phase
            output_dir = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png_2dfft_phase")
    else:
        output_dir = Path(args.output_dir)

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output type: {args.output_type}")
    logger.info(f"Transformation: 2D FFT {args.output_type.capitalize()} Spectrum (log scale, normalized)")

    convert_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        output_type=args.output_type,
        num_workers=args.num_workers,
    )

    # Show sample output comparison
    sample_outputs = list(output_dir.rglob("*.png"))[:3]
    if sample_outputs:
        logger.info("\nSample outputs:")
        for png_path in sample_outputs:
            img = Image.open(png_path)
            logger.info(f"  {png_path.name}: {img.size}, mode={img.mode}")


if __name__ == "__main__":
    main()
