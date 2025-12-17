"""Dinomaly (DINOv2) MVTec-AD Validation Script.

Purpose: Verify Dinomaly model performance on MVTec-AD dataset.
Dataset: MVTec-AD (15 categories)
Model: Dinomaly with DINOv2 backbone (Vision Transformer-based)

This script tests Dinomaly on MVTec-AD to compare with UniNet results.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.metrics import AUROC, Evaluator
from anomalib.models import Dinomaly

logging.basicConfig(level=logging.INFO)
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

# Dinomaly default settings
DEFAULT_SETTINGS = {
    "encoder_name": "dinov2reg_vit_base_14",  # Options: dinov2reg_vit_small_14, dinov2reg_vit_base_14, dinov2reg_vit_large_14
    "bottleneck_dropout": 0.2,
    "decoder_depth": 8,
    "batch_size": 8,
    "max_steps": 5000,  # Dinomaly uses max_steps instead of max_epochs
    "check_val_every_n_epoch": 1,
}


def train_and_evaluate(
    category: str,
    output_dir: Path,
    settings: dict,
    gpu_id: int = 0,
) -> dict:
    """Train Dinomaly on a single MVTec-AD category and evaluate.

    Args:
        category: MVTec-AD category name
        output_dir: Directory to save results
        settings: Training settings dict
        gpu_id: GPU device ID to use

    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Starting experiment for category: {category}")
    logger.info(f"Using encoder: {settings['encoder_name']}")

    # Create output directory for this category
    category_dir = output_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    # Setup data module
    datamodule = MVTecAD(
        root="./datasets/MVTecAD",
        category=category,
        train_batch_size=settings["batch_size"],
        eval_batch_size=settings["batch_size"],
        num_workers=8,
        seed=42,
    )

    # Setup evaluator with AUROC metrics
    val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
    test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
    evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])

    # Create Dinomaly model
    model = Dinomaly(
        encoder_name=settings["encoder_name"],
        bottleneck_dropout=settings["bottleneck_dropout"],
        decoder_depth=settings["decoder_depth"],
        evaluator=evaluator,
        pre_processor=True,
    )

    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_image_AUROC",
            mode="max",
            patience=10,
        ),
        ModelCheckpoint(
            dirpath=category_dir / "checkpoints",
            filename="best-{step}-{val_image_AUROC:.4f}",
            monitor="val_image_AUROC",
            mode="max",
            save_top_k=1,
            save_weights_only=True,
        ),
    ]

    # Create engine - Dinomaly uses max_steps
    engine = Engine(
        max_steps=settings["max_steps"],
        accelerator="gpu",
        devices=[gpu_id],
        check_val_every_n_epoch=settings["check_val_every_n_epoch"],
        callbacks=callbacks,
        default_root_dir=str(category_dir),
        gradient_clip_val=0.1,  # Dinomaly specific
        num_sanity_val_steps=0,  # Dinomaly specific
    )

    # Train
    logger.info(f"Training {category}...")
    engine.fit(model=model, datamodule=datamodule)

    # Test
    logger.info(f"Testing {category}...")
    test_results = engine.test(model=model, datamodule=datamodule)

    # Extract results
    results = {
        "category": category,
        "encoder": settings["encoder_name"],
        "test_results": test_results[0] if test_results else {},
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in settings.items()},
    }

    # Save results
    results_file = category_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results for {category}: {test_results}")
    return results


def main():
    """Run Dinomaly validation on MVTec-AD."""
    parser = argparse.ArgumentParser(description="Dinomaly MVTec-AD Validation")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["bottle"],
        choices=ALL_CATEGORIES + ["all"],
        help="Categories to test (use 'all' for all categories)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/dinomaly_mvtec_validation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="dinov2reg_vit_base_14",
        choices=["dinov2reg_vit_small_14", "dinov2reg_vit_base_14", "dinov2reg_vit_large_14"],
        help="DINOv2 encoder variant",
    )
    args = parser.parse_args()

    # Expand 'all' to all categories
    categories = ALL_CATEGORIES if "all" in args.categories else args.categories

    # Setup output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update settings with command line args
    settings = DEFAULT_SETTINGS.copy()
    settings["max_steps"] = args.max_steps
    settings["batch_size"] = args.batch_size
    settings["encoder_name"] = args.encoder

    # Save experiment settings
    with open(output_dir / "experiment_settings.json", "w") as f:
        json.dump(
            {
                "categories": categories,
                "settings": settings,
                "timestamp": timestamp,
            },
            f,
            indent=2,
        )

    # Run experiments
    all_results = []
    for category in categories:
        try:
            results = train_and_evaluate(
                category=category,
                output_dir=output_dir,
                settings=settings,
                gpu_id=args.gpu,
            )
            all_results.append(results)
        except Exception as e:
            logger.error(f"Failed to train {category}: {e}")
            all_results.append({"category": category, "error": str(e)})

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Dinomaly (DINOv2) MVTec-AD Validation Results")
    print("=" * 60)

    for result in all_results:
        category = result["category"]
        if "error" in result:
            print(f"{category}: ERROR - {result['error']}")
        else:
            test_results = result.get("test_results", {})
            auroc = test_results.get("test_image_AUROC", "N/A")
            encoder = result.get("encoder", "N/A")
            if isinstance(auroc, float):
                auroc_str = f"{auroc * 100:.2f}%"
            else:
                auroc_str = str(auroc)
            print(f"{category} ({encoder}): Image AUROC = {auroc_str}")

    print("=" * 60)

    # Save final summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"Summary file: {summary_file}")

    # Comparison note
    print("\n" + "-" * 60)
    print("COMPARISON WITH UniNet:")
    print("-" * 60)
    print("UniNet (wide_resnet50_2): bottle = 99.92% AUROC")
    print("Dinomaly (DINOv2): Check results above")
    print("-" * 60)


if __name__ == "__main__":
    main()
