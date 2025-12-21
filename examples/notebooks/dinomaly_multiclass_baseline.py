"""
Dinomaly Multi-Class Baseline for HDMAP Dataset.

This script implements the UNIFIED (multi-class) training approach from the original
Dinomaly paper (ECCV 2025). Instead of training separate models for each domain,
we train a SINGLE model on ALL 4 HDMAP domains (A, B, C, D) combined.

Key insight from original paper:
- Multi-class unified training: All classes are combined into one training dataset
- The model learns to detect anomalies across ALL classes without class-specific tuning
- This tests the model's generalization capability

Reference:
- dinomaly_mvtec_uni.py: Multi-class unified training on MVTec (15 classes)
- dinomaly_mvtec_sep.py: Single-class training on MVTec (per-class models)

HDMAP Dataset Structure:
    datasets/HDMAP/1000_png/
    ├── domain_A/
    │   ├── train/good/
    │   └── test/{good,fault}/
    ├── domain_B/
    ├── domain_C/
    └── domain_D/
"""

from __future__ import annotations

import argparse
import atexit
import errno
import json
import logging
import os
import random
import shutil
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.metrics import AUROC
from anomalib.metrics.evaluator import Evaluator
from anomalib.models.image.dinomaly import Dinomaly

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


# =============================================================================
# Constants
# =============================================================================

# Dataset paths (use environment variables for portability)
DEFAULT_HDMAP_PNG_ROOT = "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png"
DEFAULT_HDMAP_TIFF_ROOT = "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_tiff_minmax"

HDMAP_PNG_ROOT = os.getenv("HDMAP_PNG_ROOT", DEFAULT_HDMAP_PNG_ROOT)
HDMAP_TIFF_ROOT = os.getenv("HDMAP_TIFF_ROOT", DEFAULT_HDMAP_TIFF_ROOT)

# Allowed domains (whitelist for security)
ALLOWED_DOMAINS = frozenset({"domain_A", "domain_B", "domain_C", "domain_D"})
DOMAINS = list(ALLOWED_DOMAINS)

# Training constants (from original Dinomaly paper)
DEFAULT_BOTTLENECK_DROPOUT = 0.2  # Fixed per original paper
DEFAULT_MAX_STEPS = 10000
DEFAULT_BATCH_SIZE = 16
DEFAULT_IMAGE_SIZE = 448
DEFAULT_CROP_SIZE = 392
DEFAULT_GRADIENT_CLIP_VAL = 0.1

# Validation constants
VALIDATION_CHECK_EVERY_N_STEPS = 1000
MAX_VAL_CHECK_INTERVAL = 500
DEFAULT_VAL_CHECK_INTERVAL = 100

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# Logging Setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_domain(domain: str) -> None:
    """Validate domain name is safe and allowed.

    Args:
        domain: Domain name to validate.

    Raises:
        ValueError: If domain is invalid or contains unsafe characters.
    """
    if domain not in ALLOWED_DOMAINS:
        raise ValueError(f"Invalid domain: {domain}. Allowed: {ALLOWED_DOMAINS}")
    if any(char in domain for char in ['/', '\\', '..', '\0']):
        raise ValueError(f"Domain contains invalid characters: {domain}")


def format_auroc(auroc: float) -> str:
    """Format AUROC as percentage string.

    Args:
        auroc: AUROC value (0.0 to 1.0).

    Returns:
        Formatted string like "95.23%".
    """
    return f"{auroc * 100:.2f}%"


def safe_mean(values: list[float]) -> float:
    """Calculate mean with division by zero protection.

    Args:
        values: List of float values.

    Returns:
        Mean value, or 0.0 if list is empty.
    """
    return sum(values) / len(values) if values else 0.0


def create_safe_symlink(src: Path, dst: Path) -> bool:
    """Safely create symlink with proper error handling.

    Args:
        src: Source file path.
        dst: Destination symlink path.

    Returns:
        True if symlink was created successfully, False otherwise.
    """
    # Validate source exists and is a file
    if not src.exists() or not src.is_file():
        logger.warning(f"Invalid source file: {src}")
        return False

    # Sanitize filename to prevent path traversal
    safe_name = src.name.replace('/', '_').replace('\\', '_').replace('..', '_')
    dst = dst.parent / safe_name

    try:
        os.symlink(src, dst)
        return True
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Symlink already exists - verify it points to correct target
            try:
                if dst.resolve() != src.resolve():
                    logger.warning(f"Conflicting symlink exists: {dst}")
                    dst.unlink()
                    os.symlink(src, dst)
            except OSError:
                pass  # Could not resolve, skip
        else:
            logger.error(f"Failed to create symlink {dst}: {e}")
            return False
    return True


@contextmanager
def temporary_combined_dataset(data_root: Path, domains: list[str]):
    """Context manager for temporary combined dataset with cleanup.

    Args:
        data_root: Root path of dataset.
        domains: List of domain names.

    Yields:
        Path to combined dataset root.
    """
    combined_root = data_root / "_combined_multiclass"
    try:
        yield combined_root
    finally:
        # Cleanup on exit
        if combined_root.exists():
            logger.info(f"Cleaning up temporary dataset: {combined_root}")
            try:
                shutil.rmtree(combined_root)
            except OSError as e:
                logger.warning(f"Failed to cleanup {combined_root}: {e}")


def evaluate_domain(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    """Evaluate model on a single domain.

    Args:
        model: PyTorch model to evaluate.
        dataloader: DataLoader for the domain.
        device: Device to run inference on.

    Returns:
        Tuple of (labels, scores) lists.
    """
    labels: list[int] = []
    scores: list[float] = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            batch_labels = batch["label"]

            # Forward pass
            output = model(images)

            # Robust score extraction
            if hasattr(output, "pred_score"):
                batch_scores = output.pred_score.cpu()
            elif isinstance(output, dict) and "pred_score" in output:
                batch_scores = output["pred_score"].cpu()
            elif torch.is_tensor(output):
                batch_scores = output.cpu()
            else:
                raise ValueError(f"Unexpected output format: {type(output)}")

            labels.extend(batch_labels.numpy().tolist())
            scores.extend(batch_scores.numpy().tolist())

            # Cleanup GPU memory
            del images, output

    return labels, scores


def compute_auroc_safe(labels: list[int], scores: list[float]) -> float:
    """Compute AUROC with edge case handling.

    Args:
        labels: Ground truth labels.
        scores: Predicted scores.

    Returns:
        AUROC value, or 0.0 if cannot be computed.
    """
    if len(set(labels)) < 2:
        logger.warning("Cannot compute AUROC: only one class present")
        return 0.0
    try:
        return roc_auc_score(labels, scores)
    except Exception as e:
        logger.error(f"Failed to compute AUROC: {e}")
        return 0.0


# =============================================================================
# Dataset
# =============================================================================

class HDMAPMultiClassDataset(torch.utils.data.Dataset):
    """Multi-class dataset combining all HDMAP domains for unified training.

    Args:
        root: Path to HDMAP dataset root.
        domains: List of domain names to include.
        split: "train" or "test".
        transform: Image transformations.
    """

    def __init__(
        self,
        root: str | Path,
        domains: list[str],
        split: str = "train",
        transform: transforms.Compose | None = None,
    ) -> None:
        """Initialize multi-class dataset."""
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # Validate domains
        for domain in domains:
            validate_domain(domain)
        self.domains = domains

        self.samples: list[dict[str, Any]] = []
        self.domain_ids: dict[str, int] = {}

        for domain_idx, domain in enumerate(domains):
            self.domain_ids[domain] = domain_idx
            self._load_domain_samples(domain, domain_idx)

        logger.info(f"Loaded {len(self.samples)} samples from {len(domains)} domains ({split})")

        # Efficient counting using Counter
        domain_counts = Counter(s["domain"] for s in self.samples)
        for domain in domains:
            logger.info(f"  {domain}: {domain_counts.get(domain, 0)} samples")

    def _load_domain_samples(self, domain: str, domain_idx: int) -> None:
        """Load samples from a single domain.

        Args:
            domain: Domain name.
            domain_idx: Domain index.
        """
        if self.split == "train":
            domain_path = self.root / domain / "train" / "good"
            if not domain_path.exists():
                raise FileNotFoundError(f"Training path not found: {domain_path}")

            png_files = list(domain_path.glob("*.png"))
            if not png_files:
                logger.warning(f"No PNG files found in {domain_path}")

            for img_path in png_files:
                self.samples.append({
                    "image_path": str(img_path),
                    "domain": domain,
                    "domain_idx": domain_idx,
                    "label": 0,  # Good
                })
        else:
            # Test: both good and fault
            for label_name, label_value in [("good", 0), ("fault", 1)]:
                domain_path = self.root / domain / "test" / label_name
                if domain_path.exists():
                    for img_path in domain_path.glob("*.png"):
                        self.samples.append({
                            "image_path": str(img_path),
                            "domain": domain,
                            "domain_idx": domain_idx,
                            "label": label_value,
                        })

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with image, label, domain info.
        """
        sample = self.samples[idx]

        # Load image with proper resource management
        with Image.open(sample["image_path"]) as img:
            image = img.convert("RGB")
            if self.transform:
                image = self.transform(image)

        return {
            "image": image,
            "label": sample["label"],  # Let DataLoader handle batching
            "domain_idx": sample["domain_idx"],
            "image_path": sample["image_path"],
            "domain": sample["domain"],
        }


# =============================================================================
# Callbacks
# =============================================================================

class ValidationAUROCPerDomain(Callback):
    """Callback to track validation AUROC per domain during training.

    Args:
        test_dataloaders: Dictionary mapping domain names to DataLoaders.
    """

    def __init__(self, test_dataloaders: dict[str, DataLoader]) -> None:
        """Initialize callback."""
        super().__init__()
        self.test_dataloaders = test_dataloaders
        self.history: list[dict[str, Any]] = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Evaluate per-domain AUROC at end of each epoch.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: Lightning module being trained.
        """
        if trainer.global_step % VALIDATION_CHECK_EVERY_N_STEPS != 0:
            return

        pl_module.eval()
        device = pl_module.device

        domain_results: dict[str, float] = {}
        all_labels: list[int] = []
        all_scores: list[float] = []

        with torch.no_grad():
            for domain, dataloader in self.test_dataloaders.items():
                # Use shared evaluation function
                domain_labels, domain_scores = evaluate_domain(
                    pl_module, dataloader, device
                )

                auroc = compute_auroc_safe(domain_labels, domain_scores)
                domain_results[domain] = auroc
                all_labels.extend(domain_labels)
                all_scores.extend(domain_scores)

                logger.info(f"Step {trainer.global_step} - {domain}: AUROC = {format_auroc(auroc)}")

        # Overall AUROC
        overall_auroc = compute_auroc_safe(all_labels, all_scores)
        logger.info(f"Step {trainer.global_step} - Overall: AUROC = {format_auroc(overall_auroc)}")

        # Log to TensorBoard
        if trainer.logger:
            for domain, auroc in domain_results.items():
                trainer.logger.experiment.add_scalar(
                    f"val_auroc/{domain}", auroc, trainer.global_step
                )
            trainer.logger.experiment.add_scalar(
                "val_auroc/overall", overall_auroc, trainer.global_step
            )

        self.history.append({
            "step": trainer.global_step,
            "domain_aurocs": domain_results,
            "overall_auroc": overall_auroc,
        })

        pl_module.train()


# =============================================================================
# Transforms
# =============================================================================

def get_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    crop_size: int = DEFAULT_CROP_SIZE,
) -> transforms.Compose:
    """Get data transforms matching original Dinomaly paper.

    Args:
        image_size: Size to resize images to.
        crop_size: Size to center crop to.

    Returns:
        Composed transforms.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# =============================================================================
# Experiment Functions
# =============================================================================

def create_combined_dataset(
    data_root: Path,
    domains: list[str],
    combined_root: Path,
) -> tuple[int, int, int]:
    """Create combined dataset with symlinks.

    Args:
        data_root: Root path of original dataset.
        domains: List of domain names.
        combined_root: Path for combined dataset.

    Returns:
        Tuple of (train_count, test_good_count, test_fault_count).
    """
    # Create directories
    train_dir = combined_root / "train" / "good"
    train_dir.mkdir(parents=True, exist_ok=True)

    test_good_dir = combined_root / "test" / "good"
    test_good_dir.mkdir(parents=True, exist_ok=True)

    test_fault_dir = combined_root / "test" / "fault"
    test_fault_dir.mkdir(parents=True, exist_ok=True)

    train_count = 0
    test_good_count = 0
    test_fault_count = 0

    for domain in domains:
        validate_domain(domain)
        domain_root = data_root / domain

        # Training samples
        train_path = domain_root / "train" / "good"
        if not train_path.exists():
            raise FileNotFoundError(f"Training path not found: {train_path}")

        for img_path in train_path.glob("*.png"):
            link_path = train_dir / f"{domain}_{img_path.name}"
            if create_safe_symlink(img_path, link_path):
                train_count += 1

        # Test good samples
        test_good_path = domain_root / "test" / "good"
        if test_good_path.exists():
            for img_path in test_good_path.glob("*.png"):
                link_path = test_good_dir / f"{domain}_{img_path.name}"
                if create_safe_symlink(img_path, link_path):
                    test_good_count += 1

        # Test fault samples
        test_fault_path = domain_root / "test" / "fault"
        if test_fault_path.exists():
            for img_path in test_fault_path.glob("*.png"):
                link_path = test_fault_dir / f"{domain}_{img_path.name}"
                if create_safe_symlink(img_path, link_path):
                    test_fault_count += 1

    return train_count, test_good_count, test_fault_count


def run_multiclass_experiment(
    data_root: str,
    domains: list[str],
    max_steps: int,
    gpu_id: int,
    output_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    crop_size: int = DEFAULT_CROP_SIZE,
    encoder_name: str = "dinov2reg_vit_base_14",
) -> dict[str, Any]:
    """Run multi-class unified training experiment.

    This follows the original Dinomaly paper's unified training approach:
    - Combine all domains into single training dataset
    - Train one model on all domains
    - Evaluate per-domain and overall AUROC

    Args:
        data_root: Path to dataset root.
        domains: List of domain names.
        max_steps: Maximum training steps.
        gpu_id: GPU device ID.
        output_dir: Output directory for results.
        batch_size: Training batch size.
        image_size: Image resize size.
        crop_size: Center crop size.
        encoder_name: DINOv2 encoder variant.

    Returns:
        Dictionary containing experiment results.
    """
    set_seed(42)

    logger.info("=" * 60)
    logger.info("MULTI-CLASS UNIFIED TRAINING")
    logger.info("=" * 60)
    logger.info(f"Domains: {domains}")
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Encoder: {encoder_name}")

    experiment_dir = output_dir / "multiclass_unified"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Create transforms
    transform = get_transforms(image_size, crop_size)

    # Create per-domain test dataloaders for validation callback
    test_dataloaders: dict[str, DataLoader] = {}
    for domain in domains:
        test_dataset = HDMAPMultiClassDataset(
            root=data_root,
            domains=[domain],
            split="test",
            transform=transform,
        )
        test_dataloaders[domain] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )

    # Setup evaluator
    val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
    test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
    evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])

    # Create model (standard Dinomaly with fixed dropout)
    model = Dinomaly(
        encoder_name=encoder_name,
        bottleneck_dropout=DEFAULT_BOTTLENECK_DROPOUT,
        evaluator=evaluator,
        pre_processor=True,
    )

    # Setup callbacks
    per_domain_callback = ValidationAUROCPerDomain(test_dataloaders)

    callbacks = [
        ModelCheckpoint(
            dirpath=experiment_dir / "checkpoints",
            filename="best-{step}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
        ),
        per_domain_callback,
    ]

    # Setup TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=str(experiment_dir),
        name="tensorboard",
        version="",
    )

    data_root_path = Path(data_root)

    # Use context manager for automatic cleanup
    with temporary_combined_dataset(data_root_path, domains) as combined_root:
        logger.info("Creating combined dataset...")

        train_count, test_good_count, test_fault_count = create_combined_dataset(
            data_root_path, domains, combined_root
        )

        logger.info(f"Created combined dataset at: {combined_root}")
        logger.info(f"  Train samples: {train_count}")
        logger.info(f"  Test good: {test_good_count}")
        logger.info(f"  Test fault: {test_fault_count}")

        if train_count == 0:
            raise RuntimeError("No training samples found!")

        # Create datamodule from combined dataset
        datamodule = Folder(
            name="HDMAP_MultiClass",
            root=combined_root,
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

        # Calculate val_check_interval
        estimated_batches_per_epoch = max(1, train_count // batch_size)
        val_check_interval = min(MAX_VAL_CHECK_INTERVAL, estimated_batches_per_epoch)
        logger.info(f"Using val_check_interval={val_check_interval}")

        engine = Engine(
            max_steps=max_steps,
            accelerator="gpu",
            devices=[gpu_id],
            val_check_interval=val_check_interval,
            callbacks=callbacks,
            default_root_dir=str(experiment_dir),
            logger=tb_logger,
            gradient_clip_val=DEFAULT_GRADIENT_CLIP_VAL,
        )

        # Train
        logger.info("Starting training...")
        engine.fit(model, datamodule=datamodule)

        # Test
        logger.info("Evaluating model...")
        test_results = engine.test(model, datamodule=datamodule)

    # Extract overall AUROC
    if not test_results:
        logger.warning("Test failed to produce results")
        overall_auroc = 0.0
    else:
        overall_auroc = test_results[0].get("test_image_AUROC", 0.0)

    logger.info(f"Overall Test AUROC: {format_auroc(overall_auroc)}")

    # Evaluate per-domain (final evaluation)
    logger.info("\nPer-Domain AUROC (Final):")
    model.eval()
    device = next(model.parameters()).device

    domain_results: dict[str, float] = {}
    for domain in domains:
        labels, scores = evaluate_domain(model, test_dataloaders[domain], device)
        auroc = compute_auroc_safe(labels, scores)
        domain_results[domain] = auroc
        logger.info(f"  {domain}: {format_auroc(auroc)}")

    # Compute mean
    mean_auroc = safe_mean(list(domain_results.values()))
    logger.info(f"  Mean: {format_auroc(mean_auroc)}")

    # Save results
    results: dict[str, Any] = {
        "experiment_type": "multiclass_unified",
        "domains": domains,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "encoder_name": encoder_name,
        "overall_auroc": float(overall_auroc),
        "per_domain_auroc": {k: float(v) for k, v in domain_results.items()},
        "mean_domain_auroc": float(mean_auroc),
        "training_history": per_domain_callback.history,
        "timestamp": datetime.now().isoformat(),
    }

    with open(experiment_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {experiment_dir}")

    return results


def run_singleclass_experiments(
    data_root: str,
    domains: list[str],
    max_steps: int,
    gpu_id: int,
    output_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    encoder_name: str = "dinov2reg_vit_base_14",
) -> dict[str, Any]:
    """Run single-class (per-domain) training for comparison.

    This is the traditional approach where each domain has its own model.

    Args:
        data_root: Path to dataset root.
        domains: List of domain names.
        max_steps: Maximum training steps.
        gpu_id: GPU device ID.
        output_dir: Output directory for results.
        batch_size: Training batch size.
        encoder_name: DINOv2 encoder variant.

    Returns:
        Dictionary containing per-domain results.
    """
    logger.info("=" * 60)
    logger.info("SINGLE-CLASS (PER-DOMAIN) TRAINING")
    logger.info("=" * 60)

    all_results: dict[str, dict[str, Any]] = {}

    for domain in domains:
        set_seed(42)
        validate_domain(domain)

        logger.info(f"\n{'='*40}")
        logger.info(f"Training on: {domain}")
        logger.info(f"{'='*40}")

        experiment_dir = output_dir / f"singleclass_{domain}"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        domain_path = Path(data_root) / domain
        if not domain_path.exists():
            raise FileNotFoundError(f"Domain path not found: {domain_path}")

        # Setup evaluator
        val_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="val_image_")
        test_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="test_image_")
        evaluator = Evaluator(val_metrics=[val_auroc], test_metrics=[test_auroc])

        # Create datamodule
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

        # Create model
        model = Dinomaly(
            encoder_name=encoder_name,
            bottleneck_dropout=DEFAULT_BOTTLENECK_DROPOUT,
            evaluator=evaluator,
            pre_processor=True,
        )

        # Setup TensorBoard logger
        tb_logger = TensorBoardLogger(
            save_dir=str(experiment_dir),
            name="tensorboard",
            version="",
        )

        # Val check interval
        estimated_batches = max(1, 1000 // batch_size)
        val_check_interval = min(DEFAULT_VAL_CHECK_INTERVAL, estimated_batches)

        engine = Engine(
            max_steps=max_steps,
            accelerator="gpu",
            devices=[gpu_id],
            val_check_interval=val_check_interval,
            default_root_dir=str(experiment_dir),
            logger=tb_logger,
            gradient_clip_val=DEFAULT_GRADIENT_CLIP_VAL,
        )

        # Train
        engine.fit(model, datamodule=datamodule)

        # Test
        test_results = engine.test(model, datamodule=datamodule)

        if not test_results:
            logger.warning(f"Test failed for {domain}")
            auroc = 0.0
        else:
            auroc = test_results[0].get("test_image_AUROC", 0.0)

        logger.info(f"{domain} Test AUROC: {format_auroc(auroc)}")

        all_results[domain] = {
            "auroc": float(auroc),
            "max_steps": max_steps,
        }

        # Save domain results
        with open(experiment_dir / "results.json", "w") as f:
            json.dump(all_results[domain], f, indent=2)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SINGLE-CLASS SUMMARY")
    logger.info("=" * 60)
    for domain, result in all_results.items():
        logger.info(f"  {domain}: {format_auroc(result['auroc'])}")

    aurocs = [r["auroc"] for r in all_results.values()]
    mean_auroc = safe_mean(aurocs)
    logger.info(f"  Mean: {format_auroc(mean_auroc)}")

    return all_results


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Dinomaly Multi-Class Baseline for HDMAP")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["multiclass", "singleclass", "compare"],
        default="compare",
        help="Training mode: multiclass (unified), singleclass (per-domain), or compare (both)",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=DOMAINS,
        help="Domains to include (default: all 4)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum training steps (default: {DEFAULT_MAX_STEPS}, same as original paper)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE}, same as original paper)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=HDMAP_PNG_ROOT,
        help="Path to HDMAP dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/dinomaly_multiclass_baseline",
        help="Output directory",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="dinov2reg_vit_base_14",
        choices=["dinov2reg_vit_small_14", "dinov2reg_vit_base_14", "dinov2reg_vit_large_14"],
        help="DINOv2 encoder variant",
    )

    args = parser.parse_args()

    # Validate domains
    for domain in args.domains:
        validate_domain(domain)

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment settings
    settings = {
        "mode": args.mode,
        "domains": args.domains,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "encoder": args.encoder,
        "data_root": args.data_root,
        "timestamp": timestamp,
    }
    with open(output_dir / "experiment_settings.json", "w") as f:
        json.dump(settings, f, indent=2)

    results: dict[str, Any] = {}

    if args.mode in ["multiclass", "compare"]:
        results["multiclass"] = run_multiclass_experiment(
            data_root=args.data_root,
            domains=args.domains,
            max_steps=args.max_steps,
            gpu_id=args.gpu,
            output_dir=output_dir,
            batch_size=args.batch_size,
            encoder_name=args.encoder,
        )

    if args.mode in ["singleclass", "compare"]:
        results["singleclass"] = run_singleclass_experiments(
            data_root=args.data_root,
            domains=args.domains,
            max_steps=args.max_steps,
            gpu_id=args.gpu,
            output_dir=output_dir,
            batch_size=args.batch_size,
            encoder_name=args.encoder,
        )

    # Comparison summary
    if args.mode == "compare":
        logger.info("\n" + "=" * 60)
        logger.info("MULTICLASS vs SINGLECLASS COMPARISON")
        logger.info("=" * 60)

        multiclass_mean = results["multiclass"]["mean_domain_auroc"]
        singleclass_aurocs = [r["auroc"] for r in results["singleclass"].values()]
        singleclass_mean = safe_mean(singleclass_aurocs)

        logger.info(f"Multi-class (Unified) Mean AUROC: {format_auroc(multiclass_mean)}")
        logger.info(f"Single-class (Per-domain) Mean AUROC: {format_auroc(singleclass_mean)}")
        logger.info(f"Difference: {(multiclass_mean - singleclass_mean)*100:+.2f}%")

        # Per-domain comparison
        logger.info("\nPer-Domain Comparison:")
        for domain in args.domains:
            multi_auroc = results["multiclass"]["per_domain_auroc"].get(domain, 0)
            single_auroc = results["singleclass"].get(domain, {}).get("auroc", 0)
            diff = multi_auroc - single_auroc
            logger.info(
                f"  {domain}: Multi={format_auroc(multi_auroc)}, "
                f"Single={format_auroc(single_auroc)}, Diff={diff*100:+.2f}%"
            )

    # Save final summary
    with open(output_dir / "final_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    logger.info(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
