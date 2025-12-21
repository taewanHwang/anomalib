"""
HDMAP Validation with Structure-Feature-Adaptive Discarding.

This script tests the DinomalyAdaptive model on HDMAP PNG dataset
to validate the effectiveness of structure-feature-based adaptive discarding.

Key insight (per Dinomaly paper correction):
- **Dropout** in MLP should be FIXED at 0.2 (not warmed up)
- **Discarding rate (k%)** in hard mining loss should use warmup 0%→90%
- Structure-feature-adaptive modulation is applied to the discarding rate

Available structure features:
- APE (Angular Power Entropy): Frequency-domain, GPU-batch optimized
- OE (Orientational Entropy): Spatial-domain, gradient-based

Structure features measure pattern regularity:
- Low value (structured/regular) → Higher k% (discard more easy points)
- High value (complex/irregular) → Lower k% (preserve learning signal)

Comparison:
1. Standard Dinomaly (baseline discarding warmup: 0%→90%)
2. DinomalyAdaptive (structure-feature-adaptive discarding)
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
from anomalib.models.image.dinomaly import Dinomaly, DinomalyAdaptive, DinomalyHDMAP

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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Dataset paths
HDMAP_PNG_ROOT = "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png"
HDMAP_TIFF_ROOT = "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_tiff_minmax"

# Default settings
DEFAULT_SETTINGS = {
    "batch_size": 32,  # Increased for faster training
    "learning_rate": 1e-4,
    "max_steps": 10000,  # Extended for longer experiments
    "val_check_interval": 100,  # Must be <= training batches
    "encoder_name": "dinov2reg_vit_large_14",
}

# APE-Adaptive Discarding default settings
# Formula: k_discard = k_min + (k_max - k_min) * k_time * k_struct
# Baseline mode (sensitivity=0): k_struct=1.0, matches original Dinomaly paper
ADAPTIVE_DISCARD_SETTINGS_DEFAULT = {
    "discard_sensitivity": 0.0,  # 0 = baseline mode (original paper)
    "normal_ape": 0.866,  # Default for Domain C
    "k_min": 0.0,
    "k_max": 0.9,  # Per Dinomaly paper
    "t_warmup": 1000,  # Per Dinomaly paper
    "ema_alpha": 0.0,  # No EMA smoothing
    "bottleneck_dropout": 0.2,  # Fixed dropout (not warmed up)
}

# Legacy alias for backward compatibility
ADAPTIVE_DROPOUT_SETTINGS_DEFAULT = ADAPTIVE_DISCARD_SETTINGS_DEFAULT

# HDMAP Loss default settings (Row-wise structure sensitive loss)
HDMAP_LOSS_SETTINGS_DEFAULT = {
    "row_weight": 0.3,        # Weight for row-wise loss component
    "top_k_ratio": 0.15,      # Top 15% rows to focus on
    "k_min": 0.0,             # Minimum discard rate
    "k_max": 0.9,             # Maximum discard rate
    "t_warmup": 1000,         # Warmup steps for discard rate
    "factor": 0.1,            # Gradient reduction factor for easy points
    "row_var_weight": 0.1,    # Row variance regularization weight
    "bottleneck_dropout": 0.2,  # Fixed bottleneck dropout
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

# Domain-specific normal OE values from TIFF EDA (mean OE from train_good data)
# OE: Orientational Entropy - measures gradient direction diversity in spatial domain
# Lower OE = more directional gradients, Higher OE = more isotropic/uniform gradients
# TIFF EDA Results (Dec 2025):
#   OE has highest Cohen's d across all domains (best separability)
#   - domain_A: d=4.27, domain_B: d=10.42, domain_C: d=2.88, domain_D: d=4.87
DOMAIN_OE_MAP = {
    "domain_A": 0.9355,  # train_good mean from TIFF EDA
    "domain_B": 0.9149,  # train_good mean from TIFF EDA
    "domain_C": 0.9731,  # train_good mean from TIFF EDA
    "domain_D": 0.9520,  # train_good mean from TIFF EDA
}


def create_model(
    model_type: str,
    evaluator: Evaluator,
    adaptive_settings: dict | None = None,
    hdmap_settings: dict | None = None,
    use_adaptive: bool = True,
) -> Dinomaly | DinomalyAdaptive | DinomalyHDMAP:
    """Create model based on type."""
    encoder_name = DEFAULT_SETTINGS["encoder_name"]

    if adaptive_settings is None:
        adaptive_settings = ADAPTIVE_DISCARD_SETTINGS_DEFAULT
    if hdmap_settings is None:
        hdmap_settings = HDMAP_LOSS_SETTINGS_DEFAULT

    if model_type == "dinomaly":
        return Dinomaly(
            encoder_name=encoder_name,
            bottleneck_dropout=0.2,  # Fixed dropout
            evaluator=evaluator,
            pre_processor=True,
        )
    elif model_type == "dinomaly_adaptive":
        # Structure-Feature-Adaptive Discarding
        # k_discard = k_min + (k_max - k_min) * k_time * k_struct
        # Legacy parameter support: dropout_sensitivity -> discard_sensitivity
        sensitivity = adaptive_settings.get("discard_sensitivity",
                                            adaptive_settings.get("dropout_sensitivity", 0.0))
        k_min = adaptive_settings.get("k_min", adaptive_settings.get("p_min", 0.0))
        k_max = adaptive_settings.get("k_max", adaptive_settings.get("p_max", 0.9))

        # Structure feature type (ape or oe)
        structure_feature = adaptive_settings.get("structure_feature", "ape")

        # Normal value: use normal_value if set, else fallback to legacy normal_ape
        normal_value = adaptive_settings.get("normal_value")
        if normal_value is None:
            normal_value = adaptive_settings.get("normal_ape")  # Legacy fallback

        return DinomalyAdaptive(
            encoder_name=encoder_name,
            structure_feature=structure_feature,
            discard_sensitivity=sensitivity,
            normal_value=normal_value,  # None means use feature computer's default
            k_min=k_min,
            k_max=k_max,
            t_warmup=adaptive_settings.get("t_warmup", 1000),
            ema_alpha=adaptive_settings.get("ema_alpha", 0.0),
            bottleneck_dropout=adaptive_settings.get("bottleneck_dropout", 0.2),
            use_adaptive_discard=use_adaptive,
            evaluator=evaluator,
            pre_processor=True,
        )
    elif model_type == "dinomaly_hdmap":
        # HDMAP-specific Row-wise Loss
        # Optimized for horizontal line defect detection in Domain C
        return DinomalyHDMAP(
            encoder_name=encoder_name,
            row_weight=hdmap_settings.get("row_weight", 0.3),
            top_k_ratio=hdmap_settings.get("top_k_ratio", 0.15),
            k_min=hdmap_settings.get("k_min", 0.0),
            k_max=hdmap_settings.get("k_max", 0.9),
            t_warmup=hdmap_settings.get("t_warmup", 1000),
            factor=hdmap_settings.get("factor", 0.1),
            row_var_weight=hdmap_settings.get("row_var_weight", 0.1),
            bottleneck_dropout=hdmap_settings.get("bottleneck_dropout", 0.2),
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
    hdmap_settings: dict | None = None,
    use_adaptive: bool = True,
    data_root: str = "png",
    batch_size: int = 32,
) -> dict:
    """Run a single experiment."""
    if adaptive_settings is None:
        adaptive_settings = ADAPTIVE_DROPOUT_SETTINGS_DEFAULT
    if hdmap_settings is None:
        hdmap_settings = HDMAP_LOSS_SETTINGS_DEFAULT

    logger.info(f"Starting experiment: {model_type} on {domain} ({max_steps} steps)")
    if model_type == "dinomaly_adaptive":
        logger.info(f"Adaptive settings: {adaptive_settings}")
    elif model_type == "dinomaly_hdmap":
        logger.info(f"HDMAP settings: {hdmap_settings}")

    experiment_dir = output_dir / f"{model_type}_{domain}_{max_steps}steps"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Setup data module - select PNG or TIFF based on data_root
    if data_root == "tiff":
        base_path = Path(HDMAP_TIFF_ROOT)
        logger.info(f"Using TIFF dataset: {HDMAP_TIFF_ROOT}")
    else:
        base_path = Path(HDMAP_PNG_ROOT)
        logger.info(f"Using PNG dataset: {HDMAP_PNG_ROOT}")
    domain_path = base_path / domain
    datamodule = Folder(
        name=f"HDMAP_{domain}",
        root=domain_path,
        normal_dir="train/good",
        abnormal_dir="test/fault",
        normal_test_dir="test/good",
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=8,
        seed=42,
        val_split_mode="from_test",
        val_split_ratio=0.1,
    )
    logger.info(f"Using batch_size={batch_size}")

    # Calculate appropriate val_check_interval based on batch_size
    # HDMAP has ~1000 training samples, so batches_per_epoch = 1000 / batch_size
    estimated_batches_per_epoch = max(1, 1000 // batch_size)
    val_check_interval = min(DEFAULT_SETTINGS["val_check_interval"], estimated_batches_per_epoch)
    logger.info(f"Using val_check_interval={val_check_interval} (estimated {estimated_batches_per_epoch} batches/epoch)")

    # Setup evaluator
    val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
    test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
    evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])

    # Create model
    model = create_model(
        model_type,
        evaluator,
        adaptive_settings=adaptive_settings,
        hdmap_settings=hdmap_settings,
        use_adaptive=use_adaptive,
    )

    # Setup training callbacks
    val_history_callback = ValidationAUROCHistory()
    gradient_monitor_callback = GradientMonitorCallback(log_every_n_steps=50)
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
        gradient_monitor_callback,
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
        val_check_interval=val_check_interval,
        callbacks=callbacks,
        default_root_dir=str(experiment_dir),
        logger=tb_logger,
        gradient_clip_val=0.1,  # Prevent gradient explosion (NaN loss)
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
    data_root: str = "png",
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
        data_root=data_root,
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
        data_root=data_root,
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
    parser = argparse.ArgumentParser(description="HDMAP Structure-Feature-Adaptive Discarding Validation")
    parser.add_argument(
        "--model",
        type=str,
        choices=["dinomaly", "dinomaly_adaptive", "dinomaly_hdmap", "compare"],
        default="dinomaly_adaptive",
        help="Model type to test: dinomaly (baseline), dinomaly_adaptive (APE/OE), dinomaly_hdmap (row-wise)",
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
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and evaluation (default: 32)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to repeat the experiment (for overnight runs, default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hdmap_adaptive_validation",
        help="Output directory",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="png",
        choices=["png", "tiff"],
        help="Dataset type: 'png' (normalized 0-255) or 'tiff' (original 32-bit float). "
             "TIFF recommended for accurate structure feature computation.",
    )
    # APE-Adaptive Discarding settings
    # Formula: k_discard = k_min + (k_max - k_min) * k_time * k_struct
    parser.add_argument(
        "--discard-sensitivity",
        type=float,
        default=0.0,
        help="Discard sensitivity (alpha). 0=baseline mode (original paper), >0=APE-adaptive mode",
    )
    parser.add_argument(
        "--normal-ape",
        type=float,
        default=None,
        help="(DEPRECATED: use --normal-value) Reference APE from normal samples",
    )
    parser.add_argument(
        "--k-min",
        type=float,
        default=0.0,
        help="Minimum discarding rate (default: 0.0)",
    )
    parser.add_argument(
        "--k-max",
        type=float,
        default=0.9,
        help="Maximum discarding rate (default: 0.9, per Dinomaly paper)",
    )
    parser.add_argument(
        "--t-warmup",
        type=int,
        default=1000,
        help="Warmup steps for discarding (default: 1000, per Dinomaly paper)",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.0,
        help="EMA smoothing factor for k_struct (default: 0.0 = no smoothing)",
    )
    parser.add_argument(
        "--bottleneck-dropout",
        type=float,
        default=0.2,
        help="Fixed dropout rate for bottleneck (default: 0.2, per Dinomaly paper)",
    )
    # Structure feature selection
    parser.add_argument(
        "--structure-feature",
        type=str,
        default="ape",
        choices=["ape", "oe"],
        help="Structure feature type: 'ape' (Angular Power Entropy) or 'oe' (Orientational Entropy)",
    )
    parser.add_argument(
        "--normal-value",
        type=float,
        default=None,
        help="Reference feature value from normal samples. If None, uses domain-specific default. "
             "APE: A=0.777, B=0.713, C=0.866, D=0.816. OE: TBD (requires EDA)",
    )
    # HDMAP Loss settings (for dinomaly_hdmap model)
    parser.add_argument(
        "--row-weight",
        type=float,
        default=0.3,
        help="Weight for row-wise loss component (default: 0.3)",
    )
    parser.add_argument(
        "--top-k-ratio",
        type=float,
        default=0.15,
        help="Top-k%% rows to focus on (default: 0.15 = 15%%)",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=0.1,
        help="Gradient reduction factor for easy points (default: 0.1)",
    )
    parser.add_argument(
        "--row-var-weight",
        type=float,
        default=0.1,
        help="Row variance regularization weight (default: 0.1)",
    )
    # Legacy aliases for backward compatibility
    parser.add_argument(
        "--dropout-sensitivity",
        type=float,
        default=None,
        help="(DEPRECATED: use --discard-sensitivity) Alias for discard-sensitivity",
    )
    parser.add_argument(
        "--p-min",
        type=float,
        default=None,
        help="(DEPRECATED: use --k-min) Alias for k-min",
    )
    parser.add_argument(
        "--p-max",
        type=float,
        default=None,
        help="(DEPRECATED: use --k-max) Alias for k-max",
    )

    args = parser.parse_args()

    # Handle legacy parameter aliases
    discard_sensitivity = args.discard_sensitivity
    if args.dropout_sensitivity is not None:
        logger.warning("--dropout-sensitivity is deprecated, use --discard-sensitivity instead")
        discard_sensitivity = args.dropout_sensitivity

    k_min = args.k_min
    if args.p_min is not None:
        logger.warning("--p-min is deprecated, use --k-min instead")
        k_min = args.p_min

    k_max = args.k_max
    if args.p_max is not None:
        logger.warning("--p-max is deprecated, use --k-max instead")
        k_max = args.p_max

    # Handle normal_value (new) vs normal_ape (legacy)
    normal_value = args.normal_value
    if args.normal_ape is not None:
        logger.warning("--normal-ape is deprecated, use --normal-value instead")
        normal_value = args.normal_ape

    # Structure feature type
    structure_feature = args.structure_feature

    # Build structure-feature-adaptive discarding settings from args
    adaptive_settings = {
        "structure_feature": structure_feature,
        "discard_sensitivity": discard_sensitivity,
        "normal_value": normal_value,  # None means use domain-specific default
        "k_min": k_min,
        "k_max": k_max,
        "t_warmup": args.t_warmup,
        "ema_alpha": args.ema_alpha,
        "bottleneck_dropout": args.bottleneck_dropout,
    }

    # Build HDMAP loss settings from args
    hdmap_settings = {
        "row_weight": args.row_weight,
        "top_k_ratio": args.top_k_ratio,
        "k_min": k_min,
        "k_max": k_max,
        "t_warmup": args.t_warmup,
        "factor": args.factor,
        "row_var_weight": args.row_var_weight,
        "bottleneck_dropout": args.bottleneck_dropout,
    }

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment settings
    settings = {
        "model": args.model,
        "domain": args.domain,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "gpu": args.gpu,
        "data_root": args.data_root,
        "structure_feature": structure_feature,
        "discard_sensitivity": discard_sensitivity,
        "normal_value": normal_value,
        "k_min": k_min,
        "k_max": k_max,
        "t_warmup": args.t_warmup,
        "ema_alpha": args.ema_alpha,
        "bottleneck_dropout": args.bottleneck_dropout,
        # HDMAP-specific settings
        "row_weight": args.row_weight,
        "top_k_ratio": args.top_k_ratio,
        "factor": args.factor,
        "row_var_weight": args.row_var_weight,
        "timestamp": timestamp,
    }
    with open(output_dir / "experiment_settings.json", "w") as f:
        json.dump(settings, f, indent=2)

    # Log experiment configuration
    if args.repeats > 1:
        logger.info(f"Running {args.repeats} repeated experiments with batch_size={args.batch_size}, max_steps={args.max_steps}")

    domains = ["domain_A", "domain_B", "domain_C", "domain_D"] if args.domain == "all" else [args.domain]

    all_results = {}

    # Repeat experiments for statistical significance (overnight runs)
    for repeat_idx in range(args.repeats):
        if args.repeats > 1:
            logger.info(f"\n{'='*60}")
            logger.info(f"REPEAT {repeat_idx + 1}/{args.repeats}")
            logger.info(f"{'='*60}")
            repeat_output_dir = output_dir / f"repeat_{repeat_idx + 1:02d}"
            repeat_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            repeat_output_dir = output_dir

        repeat_results = {}
        for domain in domains:
            # For adaptive experiments (sensitivity > 0), use domain-specific normal_value
            domain_adaptive_settings = adaptive_settings.copy()
            if discard_sensitivity > 0 and domain_adaptive_settings.get("normal_value") is None:
                # Apply domain-specific default based on structure feature type
                if structure_feature == "ape" and domain in DOMAIN_APE_MAP:
                    domain_adaptive_settings["normal_value"] = DOMAIN_APE_MAP[domain]
                    logger.info(f"Using domain-specific APE for {domain}: {DOMAIN_APE_MAP[domain]}")
                elif structure_feature == "oe" and domain in DOMAIN_OE_MAP:
                    domain_adaptive_settings["normal_value"] = DOMAIN_OE_MAP[domain]
                    logger.info(f"Using domain-specific OE for {domain}: {DOMAIN_OE_MAP[domain]}")

            if args.model == "compare":
                repeat_results[domain] = compare_models(
                    domain=domain,
                    max_steps=args.max_steps,
                    gpu_id=args.gpu,
                    output_dir=repeat_output_dir,
                    data_root=args.data_root,
                )
            else:
                repeat_results[domain] = run_experiment(
                    model_type=args.model,
                    domain=domain,
                    max_steps=args.max_steps,
                    gpu_id=args.gpu,
                    output_dir=repeat_output_dir,
                    adaptive_settings=domain_adaptive_settings,
                    hdmap_settings=hdmap_settings,
                    data_root=args.data_root,
                    batch_size=args.batch_size,
                )

        # Save repeat summary
        if args.repeats > 1:
            repeat_key = f"repeat_{repeat_idx + 1:02d}"
            all_results[repeat_key] = repeat_results
            with open(repeat_output_dir / "summary.json", "w") as f:
                json.dump(repeat_results, f, indent=2)
        else:
            all_results = repeat_results

    # Save overall summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary statistics for repeated experiments
    if args.repeats > 1:
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY OF REPEATED EXPERIMENTS")
        logger.info(f"{'='*60}")
        for domain in domains:
            aurocs = []
            for repeat_key in all_results:
                if domain in all_results[repeat_key]:
                    auroc = all_results[repeat_key][domain].get("test_image_AUROC", 0)
                    aurocs.append(auroc)
            if aurocs:
                mean_auroc = sum(aurocs) / len(aurocs)
                std_auroc = (sum((x - mean_auroc) ** 2 for x in aurocs) / len(aurocs)) ** 0.5
                logger.info(f"{domain}: {mean_auroc*100:.2f}% ± {std_auroc*100:.2f}% (n={len(aurocs)})")

    logger.info(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
