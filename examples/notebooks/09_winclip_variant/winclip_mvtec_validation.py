"""WinCLIP MVTec-AD Validation Script.

Purpose: Verify WinCLIP model performance on MVTec-AD dataset.
Dataset: MVTec-AD (15 categories)
Model: WinCLIP (Zero-shot / Few-shot)

WinCLIP is a zero-shot/few-shot model that uses CLIP embeddings and
sliding window approach to detect anomalies without training.

Modes:
    - Zero-shot (k_shot=0): No reference images, text prompts only
    - Few-shot (k_shot=1,2,4): Uses k normal reference images
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models.image.winclip import WinClip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# All 15 MVTec-AD categories
ALL_CATEGORIES = [
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
]

# Default k_shot modes to test
DEFAULT_K_SHOTS = [0, 1, 2, 4]


def evaluate_category(
    category: str,
    k_shot: int,
    result_dir: Path,
    gpu_id: int = 0,
) -> dict:
    """Evaluate WinCLIP on a single MVTec-AD category.

    Args:
        category: MVTec-AD category name
        k_shot: Number of reference images (0 for zero-shot)
        result_dir: Directory to save results
        gpu_id: GPU device ID to use

    Returns:
        Dictionary with evaluation results
    """
    mode_name = "zero-shot" if k_shot == 0 else f"{k_shot}-shot"
    logger.info(f"Evaluating {category} with {mode_name} mode...")

    # Create output directory for this category/k_shot
    category_dir = result_dir / category / mode_name
    category_dir.mkdir(parents=True, exist_ok=True)

    # Setup data module
    # Use absolute path to anomalib root datasets
    anomalib_root = Path(__file__).parent.parent.parent.parent  # examples/notebooks/09_winclip_variant -> anomalib
    dataset_root = anomalib_root / "datasets" / "MVTecAD"

    datamodule = MVTecAD(
        root=str(dataset_root),
        category=category,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=8,
    )

    # Create WinCLIP model
    model = WinClip(
        k_shot=k_shot,
        scales=(2, 3),  # Default scales from paper
    )

    # Create engine (no training needed for WinCLIP)
    engine = Engine(
        accelerator="gpu",
        devices=[gpu_id],
        default_root_dir=str(category_dir),
    )

    # Test (WinCLIP doesn't need training)
    logger.info(f"Testing {category} ({mode_name})...")
    test_results = engine.test(model=model, datamodule=datamodule)

    # Extract metrics
    metrics = {}
    if test_results and len(test_results) > 0:
        metrics = test_results[0]

    # Build results dict
    results = {
        "category": category,
        "k_shot": k_shot,
        "mode": mode_name,
        "metrics": {k: float(v) if isinstance(v, (int, float, torch.Tensor)) else str(v)
                   for k, v in metrics.items()},
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    results_file = category_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Log summary
    image_auroc = metrics.get("image_AUROC", metrics.get("test_image_AUROC", "N/A"))
    pixel_auroc = metrics.get("pixel_AUROC", metrics.get("test_pixel_AUROC", "N/A"))

    if isinstance(image_auroc, (int, float, torch.Tensor)):
        image_auroc = float(image_auroc)
        logger.info(f"{category} ({mode_name}): Image AUROC = {image_auroc:.4f} ({image_auroc*100:.2f}%)")
    if isinstance(pixel_auroc, (int, float, torch.Tensor)):
        pixel_auroc = float(pixel_auroc)
        logger.info(f"{category} ({mode_name}): Pixel AUROC = {pixel_auroc:.4f} ({pixel_auroc*100:.2f}%)")

    return results


def main():
    """Run WinCLIP validation on MVTec-AD."""
    parser = argparse.ArgumentParser(description="WinCLIP MVTec-AD Validation")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["bottle"],
        choices=ALL_CATEGORIES + ["all"],
        help="Categories to test (use 'all' for all categories)",
    )
    parser.add_argument(
        "--k-shots",
        nargs="+",
        type=int,
        default=DEFAULT_K_SHOTS,
        help="k_shot values to test (default: 0 1 2 4)",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="./results/winclip_mvtec_validation",
        help="Directory to save results",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID",
    )
    args = parser.parse_args()

    # Expand 'all' to all categories
    categories = ALL_CATEGORIES if "all" in args.categories else args.categories

    # Setup result directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(args.result_dir) / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment settings
    settings = {
        "categories": categories,
        "k_shots": args.k_shots,
        "gpu": args.gpu,
        "timestamp": timestamp,
    }
    with open(result_dir / "experiment_settings.json", "w") as f:
        json.dump(settings, f, indent=2)

    logger.info(f"Starting WinCLIP validation on MVTec-AD")
    logger.info(f"Categories: {categories}")
    logger.info(f"k_shot modes: {args.k_shots}")
    logger.info(f"Results will be saved to: {result_dir}")

    # Run experiments
    all_results = []
    for category in categories:
        for k_shot in args.k_shots:
            try:
                results = evaluate_category(
                    category=category,
                    k_shot=k_shot,
                    result_dir=result_dir,
                    gpu_id=args.gpu,
                )
                all_results.append(results)
            except Exception as e:
                logger.error(f"Failed to evaluate {category} with k_shot={k_shot}: {e}")
                all_results.append({
                    "category": category,
                    "k_shot": k_shot,
                    "error": str(e),
                })

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: WinCLIP MVTec-AD Validation Results")
    print("=" * 80)

    # Group by k_shot
    for k_shot in args.k_shots:
        mode_name = "Zero-shot" if k_shot == 0 else f"{k_shot}-shot"
        print(f"\n{mode_name} Results:")
        print("-" * 60)

        for result in all_results:
            if result.get("k_shot") == k_shot:
                category = result["category"]
                if "error" in result:
                    print(f"  {category}: ERROR - {result['error']}")
                else:
                    metrics = result.get("metrics", {})
                    image_auroc = metrics.get("image_AUROC", metrics.get("test_image_AUROC", "N/A"))
                    pixel_auroc = metrics.get("pixel_AUROC", metrics.get("test_pixel_AUROC", "N/A"))

                    if isinstance(image_auroc, (int, float)):
                        image_str = f"{image_auroc * 100:.2f}%"
                    else:
                        image_str = str(image_auroc)

                    if isinstance(pixel_auroc, (int, float)):
                        pixel_str = f"{pixel_auroc * 100:.2f}%"
                    else:
                        pixel_str = str(pixel_auroc)

                    print(f"  {category}: Image AUROC = {image_str}, Pixel AUROC = {pixel_str}")

    print("\n" + "=" * 80)

    # Save final summary
    summary_file = result_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {result_dir}")
    print(f"Summary file: {summary_file}")


if __name__ == "__main__":
    main()
