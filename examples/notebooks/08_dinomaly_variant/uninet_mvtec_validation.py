"""UniNet MVTec-AD Validation Script.

Purpose: Verify that the anomalib UniNet implementation reproduces paper performance.
Dataset: MVTec-AD (15 categories)
Expected: Image AUROC > 99% (paper claims 99.9%+)

This script tests UniNet on MVTec-AD to diagnose whether low HDMAP performance
is due to implementation issues or dataset characteristics.
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
from anomalib.models import UniNet

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

# Categories that use temperature=0.1 according to original UniNet paper
SPECIAL_TEMP_CATEGORIES = ["transistor", "pill", "cable"]

# Original UniNet settings from the paper
ORIGINAL_SETTINGS = {
    "student_backbone": "wide_resnet50_2",
    "teacher_backbone": "wide_resnet50_2",
    "temperature": 2.0,  # default, some categories use 0.1
    "learning_rate": 0.005,
    "weight_decay": 1e-5,
    "warmup_epochs": 0,
    "batch_size": 8,
    "max_epochs": 100,
    "image_size": (256, 256),
    "check_val_every_n_epoch": 10,
}


def get_temperature_for_category(category: str) -> float:
    """Get the appropriate temperature for a category based on paper settings."""
    if category in SPECIAL_TEMP_CATEGORIES:
        return 0.1
    return 2.0


def train_and_evaluate(
    category: str,
    output_dir: Path,
    settings: dict,
    gpu_id: int = 0,
) -> dict:
    """Train UniNet on a single MVTec-AD category and evaluate.

    Args:
        category: MVTec-AD category name
        output_dir: Directory to save results
        settings: Training settings dict
        gpu_id: GPU device ID to use

    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Starting experiment for category: {category}")

    # Get category-specific temperature
    temperature = get_temperature_for_category(category)
    logger.info(f"Using temperature={temperature} for {category}")

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

    # Create model with original UniNet settings
    model = UniNet(
        student_backbone=settings["student_backbone"],
        teacher_backbone=settings["teacher_backbone"],
        temperature=temperature,
        learning_rate=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        warmup_epochs=settings["warmup_epochs"],
        evaluator=evaluator,
        pre_processor=True,  # Use default preprocessor for MVTec
    )

    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_image_AUROC",
            mode="max",
            patience=20,  # Be patient since we check every 10 epochs
        ),
        ModelCheckpoint(
            dirpath=category_dir / "checkpoints",
            filename="best-{epoch}-{val_image_AUROC:.4f}",
            monitor="val_image_AUROC",
            mode="max",
            save_top_k=1,
            save_weights_only=True,  # Avoid recursion error
        ),
    ]

    # Create engine
    engine = Engine(
        max_epochs=settings["max_epochs"],
        accelerator="gpu",
        devices=[gpu_id],
        check_val_every_n_epoch=settings["check_val_every_n_epoch"],
        callbacks=callbacks,
        default_root_dir=str(category_dir),
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
        "temperature": temperature,
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
    """Run UniNet validation on MVTec-AD."""
    parser = argparse.ArgumentParser(description="UniNet MVTec-AD Validation")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["bottle"],  # Default to one category for quick test
        choices=ALL_CATEGORIES + ["all"],
        help="Categories to test (use 'all' for all categories)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/uninet_mvtec_validation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    args = parser.parse_args()

    # Expand 'all' to all categories
    categories = ALL_CATEGORIES if "all" in args.categories else args.categories

    # Setup output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update settings with command line args
    settings = ORIGINAL_SETTINGS.copy()
    settings["max_epochs"] = args.epochs
    settings["batch_size"] = args.batch_size

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
    print("SUMMARY: UniNet MVTec-AD Validation Results")
    print("=" * 60)

    for result in all_results:
        category = result["category"]
        if "error" in result:
            print(f"{category}: ERROR - {result['error']}")
        else:
            test_results = result.get("test_results", {})
            auroc = test_results.get("test_image_AUROC", "N/A")
            temp = result.get("temperature", "N/A")
            if isinstance(auroc, float):
                auroc_str = f"{auroc * 100:.2f}%"
            else:
                auroc_str = str(auroc)
            print(f"{category} (T={temp}): Image AUROC = {auroc_str}")

    print("=" * 60)

    # Save final summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"Summary file: {summary_file}")

    # Diagnostic message
    print("\n" + "-" * 60)
    print("DIAGNOSTIC INTERPRETATION:")
    print("-" * 60)
    print("If AUROC > 99%: anomalib UniNet implementation is correct")
    print("  -> Low HDMAP performance is due to dataset characteristics")
    print("If AUROC 70-90%: Implementation may have issues")
    print("  -> Need detailed comparison with original UniNet code")
    print("If AUROC < 70%: Serious implementation error")
    print("  -> Consider using original UniNet repository directly")
    print("-" * 60)


if __name__ == "__main__":
    main()
