"""
HDMAP Validation with APE-based Adaptive Bottleneck Dropout.

This script tests the DinomalyAdaptive model on HDMAP PNG dataset
to validate the effectiveness of Angular Power Entropy (APE)-based adaptive dropout.

APE measures directional concentration of power in the frequency domain:
- Low APE (strong directional pattern) → Higher overfit risk → Higher dropout
- High APE (isotropic/complex pattern) → Lower overfit risk → Lower dropout

Comparison:
1. Standard Dinomaly (fixed dropout)
2. DinomalyAdaptive (adaptive dropout based on APE)
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.metrics import AUROC
from anomalib.metrics.evaluator import Evaluator
from anomalib.models.image.dinomaly import Dinomaly, DinomalyAdaptive

from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import TensorBoardLogger


class ValidationAUROCHistory(Callback):
    """Callback to track validation AUROC history during training."""

    def __init__(self):
        super().__init__()
        self.history = []  # List of {step, val_image_AUROC}

    def on_validation_epoch_end(self, trainer, pl_module):
        """Record validation AUROC after each validation epoch."""
        metrics = trainer.callback_metrics
        val_auroc = metrics.get("val_image_AUROC", None)
        if val_auroc is not None:
            self.history.append({
                "step": trainer.global_step,
                "val_image_AUROC": float(val_auroc),
            })
            logger.info(
                f"[ValidationHistory] Step {trainer.global_step}: "
                f"val_image_AUROC = {val_auroc:.4f} ({val_auroc*100:.2f}%)"
            )

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Dataset paths
HDMAP_PNG_ROOT = "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png"

# Default settings
DEFAULT_SETTINGS = {
    "batch_size": 8,
    "learning_rate": 1e-4,
    "max_steps": 5000,
    "val_check_interval": 100,  # Must be <= training batches (125 for HDMAP)
    "encoder_name": "dinov2reg_vit_large_14",
}

ADAPTIVE_DROPOUT_SETTINGS_DEFAULT = {
    "base_dropout": 0.3,
    "min_dropout": 0.1,
    "max_dropout": 0.6,
    "dropout_sensitivity": 4.0,
    "normal_ape": 0.78,  # Default (will be overridden by domain-specific values)
}

# Domain-specific normal APE values from EDA (mean APE from normal training data)
# APE: Angular Power Entropy - measures directional concentration in frequency domain
# Lower APE = more directional/regular pattern, Higher APE = more isotropic/complex
DOMAIN_APE_MAP = {
    "domain_A": 0.777,
    "domain_B": 0.713,
    "domain_C": 0.866,
    "domain_D": 0.816,
}


def create_model(
    model_type: str,
    evaluator: Evaluator,
    adaptive_settings: dict | None = None,
    use_adaptive: bool = True,
) -> Dinomaly | DinomalyAdaptive:
    """Create model based on type."""
    encoder_name = DEFAULT_SETTINGS["encoder_name"]

    if adaptive_settings is None:
        adaptive_settings = ADAPTIVE_DROPOUT_SETTINGS_DEFAULT

    if model_type == "dinomaly":
        return Dinomaly(
            encoder_name=encoder_name,
            bottleneck_dropout=0.2,  # Fixed dropout
            evaluator=evaluator,
            pre_processor=True,
        )
    elif model_type == "dinomaly_adaptive":
        return DinomalyAdaptive(
            encoder_name=encoder_name,
            base_dropout=adaptive_settings["base_dropout"],
            min_dropout=adaptive_settings["min_dropout"],
            max_dropout=adaptive_settings["max_dropout"],
            dropout_sensitivity=adaptive_settings["dropout_sensitivity"],
            normal_ape=adaptive_settings.get("normal_ape", 0.78),
            use_adaptive_dropout=use_adaptive,
            evaluator=evaluator,
            pre_processor=True,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_experiment(
    model_type: str,
    domain: str,
    max_steps: int,
    gpu_id: int,
    output_dir: Path,
    adaptive_settings: dict | None = None,
    use_adaptive: bool = True,
) -> dict:
    """Run a single experiment."""
    if adaptive_settings is None:
        adaptive_settings = ADAPTIVE_DROPOUT_SETTINGS_DEFAULT

    logger.info(f"Starting experiment: {model_type} on {domain} ({max_steps} steps)")
    if model_type == "dinomaly_adaptive":
        logger.info(f"Adaptive settings: {adaptive_settings}")

    experiment_dir = output_dir / f"{model_type}_{domain}_{max_steps}steps"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Setup data module
    domain_path = Path(HDMAP_PNG_ROOT) / domain
    datamodule = Folder(
        name=f"HDMAP_{domain}",
        root=domain_path,
        normal_dir="train/good",
        abnormal_dir="test/fault",
        normal_test_dir="test/good",
        train_batch_size=DEFAULT_SETTINGS["batch_size"],
        eval_batch_size=DEFAULT_SETTINGS["batch_size"],
        num_workers=8,
        seed=42,
        val_split_mode="from_test",
        val_split_ratio=0.1,
    )

    # Setup evaluator
    val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
    test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
    evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])

    # Create model
    model = create_model(model_type, evaluator, adaptive_settings=adaptive_settings, use_adaptive=use_adaptive)

    # Setup training callbacks
    val_history_callback = ValidationAUROCHistory()
    callbacks = [
        ModelCheckpoint(
            dirpath=experiment_dir / "checkpoints",
            filename="best-{step}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
        ),
        val_history_callback,
    ]

    # Setup TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=str(experiment_dir),
        name="tensorboard",
        version="",
    )

    engine = Engine(
        max_steps=max_steps,
        accelerator="gpu",
        devices=[gpu_id],
        val_check_interval=DEFAULT_SETTINGS["val_check_interval"],
        callbacks=callbacks,
        default_root_dir=str(experiment_dir),
        logger=tb_logger,
    )

    # Train
    logger.info("Starting training...")
    engine.fit(model, datamodule=datamodule)

    # Test using engine.test() for accurate AUROC (same method as TensorBoard validation)
    logger.info("Evaluating model with engine.test()...")
    test_results = engine.test(model, datamodule=datamodule)

    # Extract AUROC from test results
    # test_results is a list of dicts with metrics like "test_image_AUROC"
    auroc = 0.0
    if test_results and len(test_results) > 0:
        auroc = test_results[0].get("test_image_AUROC", 0.0)
    logger.info(f"Test AUROC (from Evaluator): {auroc:.4f} ({auroc*100:.2f}%)")

    # Get adaptive dropout stats if available
    dropout_stats = {}
    if hasattr(model, 'model') and hasattr(model.model, 'get_dropout_stats'):
        dropout_stats = model.model.get_dropout_stats()

    # Save results (only key adaptive parameters)
    adaptive_key_settings = None
    if model_type == "dinomaly_adaptive" and adaptive_settings:
        adaptive_key_settings = {
            "sensitivity": adaptive_settings.get("dropout_sensitivity"),
            "normal_ape": adaptive_settings.get("normal_ape"),
            "base_dropout": adaptive_settings.get("base_dropout"),
        }

    # Get validation history from callback
    validation_history = val_history_callback.history

    # Log summary of validation progress
    if validation_history:
        best_val = max(validation_history, key=lambda x: x["val_image_AUROC"])
        logger.info(
            f"Validation Summary: {len(validation_history)} checkpoints, "
            f"Best val_AUROC = {best_val['val_image_AUROC']:.4f} at step {best_val['step']}"
        )

    results = {
        "model_type": model_type,
        "domain": domain,
        "max_steps": max_steps,
        "use_adaptive_dropout": use_adaptive,
        "adaptive_settings": adaptive_key_settings,
        "validation_history": validation_history,  # All intermediate val_image_AUROC values
        "test_image_AUROC": auroc,  # Final test AUROC
        "dropout_statistics": dropout_stats,
        "timestamp": datetime.now().isoformat(),
    }

    with open(experiment_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Test AUROC = {auroc:.4f} ({auroc*100:.2f}%)")
    logger.info(f"Results saved to: {experiment_dir}")

    return results


def compare_models(
    domain: str,
    max_steps: int,
    gpu_id: int,
    output_dir: Path,
) -> dict:
    """Compare standard Dinomaly vs DinomalyAdaptive on same domain."""
    results = {}

    # Run standard Dinomaly
    logger.info("=" * 60)
    logger.info("Running Standard Dinomaly (fixed dropout)")
    logger.info("=" * 60)
    results["dinomaly"] = run_experiment(
        model_type="dinomaly",
        domain=domain,
        max_steps=max_steps,
        gpu_id=gpu_id,
        output_dir=output_dir,
    )

    # Run DinomalyAdaptive
    logger.info("=" * 60)
    logger.info("Running DinomalyAdaptive (APE-based adaptive dropout)")
    logger.info("=" * 60)
    results["dinomaly_adaptive"] = run_experiment(
        model_type="dinomaly_adaptive",
        domain=domain,
        max_steps=max_steps,
        gpu_id=gpu_id,
        output_dir=output_dir,
    )

    # Print comparison
    logger.info("=" * 60)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 60)
    logger.info(f"Domain: {domain}, Steps: {max_steps}")
    logger.info(f"Standard Dinomaly:     Test AUROC = {results['dinomaly']['test_image_AUROC']*100:.2f}%")
    logger.info(f"DinomalyAdaptive:      Test AUROC = {results['dinomaly_adaptive']['test_image_AUROC']*100:.2f}%")

    diff = results['dinomaly_adaptive']['test_image_AUROC'] - results['dinomaly']['test_image_AUROC']
    logger.info(f"Improvement:           {diff*100:+.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="HDMAP APE-based Adaptive Dropout Validation")
    parser.add_argument(
        "--model",
        type=str,
        choices=["dinomaly", "dinomaly_adaptive", "compare"],
        default="dinomaly_adaptive",
        help="Model type to test",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="domain_A",
        choices=["domain_A", "domain_B", "domain_C", "domain_D", "all"],
        help="Domain to test",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hdmap_adaptive_validation",
        help="Output directory",
    )
    # Adaptive dropout settings
    parser.add_argument(
        "--base-dropout",
        type=float,
        default=0.3,
        help="Base dropout probability (default: 0.3)",
    )
    parser.add_argument(
        "--min-dropout",
        type=float,
        default=0.1,
        help="Minimum dropout probability (default: 0.1)",
    )
    parser.add_argument(
        "--max-dropout",
        type=float,
        default=0.6,
        help="Maximum dropout probability (default: 0.6)",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=4.0,
        help="Dropout sensitivity to APE (default: 4.0). When 0, uses fixed base_dropout.",
    )
    parser.add_argument(
        "--normal-ape",
        type=float,
        default=0.78,
        help="Reference APE from normal samples. For adaptive experiments (sensitivity > 0), "
             "domain-specific values are used automatically from EDA results: "
             "A=0.777, B=0.713, C=0.866, D=0.816",
    )

    args = parser.parse_args()

    # Build adaptive settings from args
    adaptive_settings = {
        "base_dropout": args.base_dropout,
        "min_dropout": args.min_dropout,
        "max_dropout": args.max_dropout,
        "dropout_sensitivity": args.sensitivity,
        "normal_ape": args.normal_ape,
    }

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment settings (only key adaptive parameters)
    settings = {
        "model": args.model,
        "domain": args.domain,
        "max_steps": args.max_steps,
        "gpu": args.gpu,
        "sensitivity": args.sensitivity,
        "base_dropout": args.base_dropout,
        "normal_ape": args.normal_ape,
        "timestamp": timestamp,
    }
    with open(output_dir / "experiment_settings.json", "w") as f:
        json.dump(settings, f, indent=2)

    domains = ["domain_A", "domain_B", "domain_C", "domain_D"] if args.domain == "all" else [args.domain]

    all_results = {}
    for domain in domains:
        # For adaptive experiments (sensitivity > 0), use domain-specific normal_ape
        domain_adaptive_settings = adaptive_settings.copy()
        if args.sensitivity > 0 and domain in DOMAIN_APE_MAP:
            domain_adaptive_settings["normal_ape"] = DOMAIN_APE_MAP[domain]
            logger.info(f"Using domain-specific normal_ape for {domain}: {DOMAIN_APE_MAP[domain]}")

        if args.model == "compare":
            all_results[domain] = compare_models(
                domain=domain,
                max_steps=args.max_steps,
                gpu_id=args.gpu,
                output_dir=output_dir,
            )
        else:
            all_results[domain] = run_experiment(
                model_type=args.model,
                domain=domain,
                max_steps=args.max_steps,
                gpu_id=args.gpu,
                output_dir=output_dir,
                adaptive_settings=domain_adaptive_settings,
            )

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
