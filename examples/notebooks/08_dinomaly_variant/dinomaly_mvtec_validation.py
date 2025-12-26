"""Dinomaly (DINOv2) MVTec-AD Validation Script.

Purpose: Verify Dinomaly model performance on MVTec-AD dataset.
Dataset: MVTec-AD (15 categories)
Model: DinomalyAdaptive with Unified Information-Control Dropout

Unified Dropout Formula (from Dinomaly paper):
    dropout_p = p_min + (p_max - p_min) * p_time * p_struct

Where:
    - p_time = min(1.0, global_step / t_warmup)  # time-based curriculum
    - p_struct = sigmoid(sensitivity * (normal_ape - APE))  # sample-based

Modes:
    - Ablation (sensitivity=0): p_struct=1.0, matches original Dinomaly paper
    - Adaptive (sensitivity>0): Sample-specific dropout based on APE
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import TensorBoardLogger

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.metrics import AUROC, Evaluator
from anomalib.models.image.dinomaly import DinomalyAdaptive

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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


class GradientMonitorCallback(Callback):
    """Callback to monitor gradients and detect gradient explosion.

    Logs to TensorBoard:
    - grad/total_norm: Total gradient norm across all parameters
    - grad/max_norm: Maximum gradient norm among all layers
    - grad/encoder_norm: Gradient norm of encoder parameters
    - grad/decoder_norm: Gradient norm of decoder parameters
    - weights/total_norm: Total weight norm
    - weights/max_value: Maximum absolute weight value
    - debug/has_nan_grad: 1 if any gradient is NaN, 0 otherwise
    - debug/has_inf_grad: 1 if any gradient is Inf, 0 otherwise
    """

    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_after_backward(self, trainer, pl_module):
        """Log gradient statistics after backward pass."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        # Collect gradient statistics
        total_norm = 0.0
        max_norm = 0.0
        encoder_norm = 0.0
        decoder_norm = 0.0
        has_nan = False
        has_inf = False

        layer_norms = {}

        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad = param.grad.data

                # Check for NaN/Inf
                if grad.isnan().any():
                    has_nan = True
                if grad.isinf().any():
                    has_inf = True

                # Compute norms
                param_norm = grad.norm(2).item()
                total_norm += param_norm ** 2
                max_norm = max(max_norm, param_norm)

                # Categorize by layer type
                if 'encoder' in name.lower() or 'backbone' in name.lower():
                    encoder_norm += param_norm ** 2
                elif 'decoder' in name.lower() or 'bottleneck' in name.lower():
                    decoder_norm += param_norm ** 2

                # Track individual layer norms (top-level modules only)
                module_name = name.split('.')[0]
                if module_name not in layer_norms:
                    layer_norms[module_name] = 0.0
                layer_norms[module_name] += param_norm ** 2

        total_norm = total_norm ** 0.5
        encoder_norm = encoder_norm ** 0.5
        decoder_norm = decoder_norm ** 0.5

        # Log to TensorBoard
        if trainer.logger:
            trainer.logger.experiment.add_scalar('grad/total_norm', total_norm, trainer.global_step)
            trainer.logger.experiment.add_scalar('grad/max_norm', max_norm, trainer.global_step)
            trainer.logger.experiment.add_scalar('grad/encoder_norm', encoder_norm, trainer.global_step)
            trainer.logger.experiment.add_scalar('grad/decoder_norm', decoder_norm, trainer.global_step)
            trainer.logger.experiment.add_scalar('debug/has_nan_grad', int(has_nan), trainer.global_step)
            trainer.logger.experiment.add_scalar('debug/has_inf_grad', int(has_inf), trainer.global_step)

            # Log per-module gradient norms
            for module_name, norm_sq in layer_norms.items():
                trainer.logger.experiment.add_scalar(
                    f'grad_by_module/{module_name}',
                    norm_sq ** 0.5,
                    trainer.global_step
                )

        # Log weight statistics
        total_weight_norm = 0.0
        max_weight_value = 0.0

        for name, param in pl_module.named_parameters():
            if param.data is not None:
                total_weight_norm += param.data.norm(2).item() ** 2
                max_weight_value = max(max_weight_value, param.data.abs().max().item())

        total_weight_norm = total_weight_norm ** 0.5

        if trainer.logger:
            trainer.logger.experiment.add_scalar('weights/total_norm', total_weight_norm, trainer.global_step)
            trainer.logger.experiment.add_scalar('weights/max_value', max_weight_value, trainer.global_step)

        # Warning log if gradient explosion detected
        if has_nan or has_inf or total_norm > 100:
            logger.warning(
                f"[GradientMonitor] Step {trainer.global_step}: "
                f"total_norm={total_norm:.4f}, max_norm={max_norm:.4f}, "
                f"has_nan={has_nan}, has_inf={has_inf}"
            )


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

# Dinomaly default settings with Unified Information-Control Dropout
DEFAULT_SETTINGS = {
    "encoder_name": "dinov2reg_vit_base_14",  # Options: dinov2reg_vit_small_14, dinov2reg_vit_base_14, dinov2reg_vit_large_14
    "decoder_depth": 8,
    "batch_size": 8,
    "max_steps": 5000,  # Dinomaly uses max_steps instead of max_epochs
    "check_val_every_n_epoch": 1,
    # Unified dropout parameters
    "dropout_sensitivity": 0.0,  # 0 = ablation mode (original Dinomaly paper)
    "normal_ape": 0.78,  # Reference APE from normal samples
    "p_min": 0.0,  # Minimum dropout
    "p_max": 0.9,  # Maximum dropout (Dinomaly paper spec)
    "t_warmup": 1000,  # Warmup steps (Dinomaly paper spec)
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

    # Create DinomalyAdaptive model with Unified Information-Control Dropout
    model = DinomalyAdaptive(
        encoder_name=settings["encoder_name"],
        decoder_depth=settings["decoder_depth"],
        dropout_sensitivity=settings["dropout_sensitivity"],
        normal_ape=settings["normal_ape"],
        p_min=settings["p_min"],
        p_max=settings["p_max"],
        t_warmup=settings["t_warmup"],
        evaluator=evaluator,
        pre_processor=True,
    )

    # Log dropout mode
    if settings["dropout_sensitivity"] == 0:
        logger.info(f"Dropout mode: ABLATION (sensitivity=0, progressive 0%→{settings['p_max']*100:.0f}% over {settings['t_warmup']} steps)")
    else:
        logger.info(f"Dropout mode: ADAPTIVE (sensitivity={settings['dropout_sensitivity']}, normal_ape={settings['normal_ape']})")

    # Setup callbacks (no early stopping to ensure full training until max_steps)
    val_history_callback = ValidationAUROCHistory()
    gradient_monitor_callback = GradientMonitorCallback(log_every_n_steps=50)
    callbacks = [
        ModelCheckpoint(
            dirpath=category_dir / "checkpoints",
            filename="best-{step}-{val_image_AUROC:.4f}",
            monitor="val_image_AUROC",
            mode="max",
            save_top_k=1,
            save_weights_only=True,
        ),
        val_history_callback,
        gradient_monitor_callback,
    ]

    # Setup TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=str(category_dir),
        name="tensorboard",
        version="",
    )

    # Create engine - Dinomaly uses max_steps
    engine = Engine(
        max_steps=settings["max_steps"],
        accelerator="gpu",
        devices=[gpu_id],
        check_val_every_n_epoch=settings["check_val_every_n_epoch"],
        callbacks=callbacks,
        default_root_dir=str(category_dir),
        gradient_clip_val=settings.get("gradient_clip_val", 0.1),  # Configurable for experiments
        num_sanity_val_steps=0,  # Dinomaly specific
        logger=tb_logger,
    )

    # Train
    logger.info(f"Training {category}...")
    engine.fit(model=model, datamodule=datamodule)

    # Test
    logger.info(f"Testing {category}...")
    test_results = engine.test(model=model, datamodule=datamodule)

    # Get validation history from callback
    validation_history = val_history_callback.history

    # Log summary of validation progress
    if validation_history:
        best_val = max(validation_history, key=lambda x: x["val_image_AUROC"])
        logger.info(
            f"Validation Summary: {len(validation_history)} checkpoints, "
            f"Best val_AUROC = {best_val['val_image_AUROC']:.4f} at step {best_val['step']}"
        )

    # Extract test AUROC
    test_auroc = 0.0
    if test_results and len(test_results) > 0:
        test_auroc = test_results[0].get("test_image_AUROC", 0.0)

    # Extract results
    dropout_mode = "ablation" if settings["dropout_sensitivity"] == 0 else "adaptive"
    results = {
        "category": category,
        "encoder": settings["encoder_name"],
        "max_steps": settings["max_steps"],
        "gradient_clip_val": settings.get("gradient_clip_val", 0.1),
        "dropout_mode": dropout_mode,
        "dropout_sensitivity": settings["dropout_sensitivity"],
        "normal_ape": settings["normal_ape"],
        "p_min": settings["p_min"],
        "p_max": settings["p_max"],
        "t_warmup": settings["t_warmup"],
        "validation_history": validation_history,  # All intermediate val_image_AUROC values
        "test_image_AUROC": test_auroc,
        "test_results": test_results[0] if test_results else {},
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in settings.items()},
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    results_file = category_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Test AUROC for {category}: {test_auroc:.4f} ({test_auroc*100:.2f}%)")
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
    parser.add_argument(
        "--gradient-clip-val",
        type=float,
        default=0.1,
        help="Gradient clipping value (0 to disable, default: 0.1)",
    )
    parser.add_argument(
        "--no-gradient-clip",
        action="store_true",
        help="Disable gradient clipping (for NaN experiment)",
    )
    # Unified Information-Control Dropout parameters
    parser.add_argument(
        "--dropout-sensitivity",
        type=float,
        default=0.0,
        help="Dropout sensitivity (alpha). 0=ablation mode (original paper), >0=adaptive mode",
    )
    parser.add_argument(
        "--normal-ape",
        type=float,
        default=0.78,
        help="Reference APE from normal samples (mu_normal). Domain defaults: A=0.777, B=0.713, C=0.866, D=0.816",
    )
    parser.add_argument(
        "--p-min",
        type=float,
        default=0.0,
        help="Minimum dropout probability (default: 0.0)",
    )
    parser.add_argument(
        "--p-max",
        type=float,
        default=0.9,
        help="Maximum dropout probability (default: 0.9, per Dinomaly paper)",
    )
    parser.add_argument(
        "--t-warmup",
        type=int,
        default=1000,
        help="Warmup steps for dropout curriculum (default: 1000, per Dinomaly paper)",
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
    settings["gradient_clip_val"] = 0.0 if args.no_gradient_clip else args.gradient_clip_val
    # Unified dropout parameters
    settings["dropout_sensitivity"] = args.dropout_sensitivity
    settings["normal_ape"] = args.normal_ape
    settings["p_min"] = args.p_min
    settings["p_max"] = args.p_max
    settings["t_warmup"] = args.t_warmup

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
    print("\n" + "=" * 70)
    print("SUMMARY: DinomalyAdaptive MVTec-AD Validation Results")
    print("=" * 70)
    dropout_mode = "ABLATION" if settings["dropout_sensitivity"] == 0 else "ADAPTIVE"
    print(f"Dropout Mode: {dropout_mode}")
    if settings["dropout_sensitivity"] == 0:
        print(f"  Progressive dropout: 0% → {settings['p_max']*100:.0f}% over {settings['t_warmup']} steps")
    else:
        print(f"  Sensitivity: {settings['dropout_sensitivity']}, Normal APE: {settings['normal_ape']}")
        print(f"  Range: {settings['p_min']*100:.0f}% - {settings['p_max']*100:.0f}%, Warmup: {settings['t_warmup']} steps")
    print("-" * 70)

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

    # Usage examples
    print("\n" + "-" * 70)
    print("USAGE EXAMPLES:")
    print("-" * 70)
    print("# Ablation mode (original Dinomaly paper: 0%→90% over 1000 steps)")
    print("python dinomaly_mvtec_validation.py --categories bottle --dropout-sensitivity 0")
    print("")
    print("# Adaptive mode (APE-based sample-specific dropout)")
    print("python dinomaly_mvtec_validation.py --categories bottle --dropout-sensitivity 4.0 --normal-ape 0.78")
    print("-" * 70)


if __name__ == "__main__":
    main()
