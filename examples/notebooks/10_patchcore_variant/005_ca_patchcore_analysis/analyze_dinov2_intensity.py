#!/usr/bin/env python3
"""Analysis 3: DINOv2 Intensity Sensitivity Test.

Test whether DINOv2 features are intensity-invariant by applying
brightness/contrast adjustments and measuring feature distance.

Key Question: If DINOv2 features are intensity-invariant, cold fault patches
might have similar features to warm normal patches, explaining CA-PatchCore failure.

Usage:
    CUDA_VISIBLE_DEVICES=0 python analyze_dinov2_intensity.py \
        --domain domain_C \
        --n-samples 20 \
        --brightness-range -0.15 0.15 \
        --contrast-range 0.7 1.3

Outputs:
    - Sensitivity curves: L2 distance vs brightness delta
    - Sensitivity curves: L2 distance vs contrast factor
    - Cold-to-warm simulation analysis
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import tifffile
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results" / "dinov2_sensitivity"
ANOMALIB_ROOT = SCRIPT_DIR.parent.parent.parent.parent
DATASET_ROOT = ANOMALIB_ROOT / "datasets" / "HDMAP" / "1000_tiff_minmax"


def get_config() -> dict:
    """Get model configuration."""
    return {
        "backbone": "vit_base_patch14_dinov2",
        "layers": ["blocks.8"],
        "target_size": (518, 518),
        "batch_size": 1,
    }


def setup_model():
    """Setup DINOv2 feature extractor."""
    from anomalib.models import Patchcore
    from anomalib.pre_processing import PreProcessor
    from torchvision.transforms.v2 import Compose, Normalize

    config = get_config()

    pre_processor = PreProcessor(
        transform=Compose([
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )

    model = Patchcore(
        backbone=config["backbone"],
        layers=config["layers"],
        pre_trained=True,
        coreset_sampling_ratio=1.0,
        num_neighbors=1,
        pre_processor=pre_processor,
    )

    return model


def extract_features(model, images: torch.Tensor, device) -> torch.Tensor:
    """Extract patch embeddings from images."""
    model.model.eval()
    pre_processor = model.pre_processor

    with torch.no_grad():
        images = images.to(device)
        normalized = pre_processor(images)

        feature_extractor = model.model.feature_extractor
        feature_pooler = model.model.feature_pooler

        features = feature_extractor(normalized)
        features = {layer: feature_pooler(feature) for layer, feature in features.items()}

        embedding = model.model.generate_embedding(features)
        embedding = model.model.reshape_embedding(embedding)

    return embedding


def load_tiff_image(image_path: Path, target_size: Tuple[int, int] = (518, 518)) -> np.ndarray:
    """Load and resize TIFF image."""
    from PIL import Image

    img = tifffile.imread(str(image_path)).astype(np.float32)

    # Resize using bilinear interpolation
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(target_size, Image.BILINEAR)
    img = np.array(pil_img, dtype=np.float32)

    return img


def numpy_to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert numpy image to tensor for model input."""
    # Ensure 3 channels (RGB from grayscale)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=0)
    elif len(img.shape) == 3 and img.shape[-1] == 3:
        img = img.transpose(2, 0, 1)

    return torch.from_numpy(img).unsqueeze(0).float()


def apply_brightness_shift(image: np.ndarray, delta: float) -> np.ndarray:
    """Shift brightness by delta."""
    return np.clip(image + delta, 0, 1)


def apply_contrast_scale(image: np.ndarray, factor: float) -> np.ndarray:
    """Scale contrast by factor."""
    mean = image.mean()
    return np.clip((image - mean) * factor + mean, 0, 1)


def compute_feature_distance(
    original_features: torch.Tensor,
    transformed_features: torch.Tensor
) -> Dict:
    """Compute distance metrics between original and transformed features."""
    # L2 distance per patch
    l2_dist = torch.norm(original_features - transformed_features, dim=1)

    # Cosine similarity per patch
    cos_sim = torch.nn.functional.cosine_similarity(
        original_features, transformed_features, dim=1
    )

    return {
        "l2_mean": float(l2_dist.mean()),
        "l2_std": float(l2_dist.std()),
        "l2_max": float(l2_dist.max()),
        "cosine_mean": float(cos_sim.mean()),
        "cosine_std": float(cos_sim.std()),
        "cosine_min": float(cos_sim.min()),
    }


def analyze_brightness_sensitivity(
    model,
    images: List[np.ndarray],
    brightness_deltas: List[float],
    device
) -> Dict:
    """Analyze feature sensitivity to brightness changes."""
    results = {delta: [] for delta in brightness_deltas}

    for img in tqdm(images, desc="Brightness sensitivity"):
        # Extract original features
        img_tensor = numpy_to_tensor(img)
        original_feat = extract_features(model, img_tensor, device).squeeze(0)

        for delta in brightness_deltas:
            # Apply brightness shift
            transformed_img = apply_brightness_shift(img, delta)
            transformed_tensor = numpy_to_tensor(transformed_img)
            transformed_feat = extract_features(model, transformed_tensor, device).squeeze(0)

            # Compute distances
            dist = compute_feature_distance(original_feat, transformed_feat)
            results[delta].append(dist)

    # Aggregate
    aggregated = {}
    for delta in brightness_deltas:
        aggregated[delta] = {
            "l2_mean": np.mean([r["l2_mean"] for r in results[delta]]),
            "l2_std": np.std([r["l2_mean"] for r in results[delta]]),
            "cosine_mean": np.mean([r["cosine_mean"] for r in results[delta]]),
            "cosine_std": np.std([r["cosine_mean"] for r in results[delta]]),
        }

    return aggregated


def analyze_contrast_sensitivity(
    model,
    images: List[np.ndarray],
    contrast_factors: List[float],
    device
) -> Dict:
    """Analyze feature sensitivity to contrast changes."""
    results = {factor: [] for factor in contrast_factors}

    for img in tqdm(images, desc="Contrast sensitivity"):
        # Extract original features
        img_tensor = numpy_to_tensor(img)
        original_feat = extract_features(model, img_tensor, device).squeeze(0)

        for factor in contrast_factors:
            # Apply contrast scaling
            transformed_img = apply_contrast_scale(img, factor)
            transformed_tensor = numpy_to_tensor(transformed_img)
            transformed_feat = extract_features(model, transformed_tensor, device).squeeze(0)

            # Compute distances
            dist = compute_feature_distance(original_feat, transformed_feat)
            results[factor].append(dist)

    # Aggregate
    aggregated = {}
    for factor in contrast_factors:
        aggregated[factor] = {
            "l2_mean": np.mean([r["l2_mean"] for r in results[factor]]),
            "l2_std": np.std([r["l2_mean"] for r in results[factor]]),
            "cosine_mean": np.mean([r["cosine_mean"] for r in results[factor]]),
            "cosine_std": np.std([r["cosine_mean"] for r in results[factor]]),
        }

    return aggregated


def analyze_cold_warm_simulation(
    model,
    cold_images: List[np.ndarray],
    warm_images: List[np.ndarray],
    device
) -> Dict:
    """Compare actual cold-warm feature distance vs simulated brightness shift.

    Simulates: cold + delta ≈ warm (where delta is the mean intensity gap)
    """
    # Compute mean intensity gap
    cold_means = [img.mean() for img in cold_images]
    warm_means = [img.mean() for img in warm_images]
    mean_cold = np.mean(cold_means)
    mean_warm = np.mean(warm_means)
    intensity_gap = mean_warm - mean_cold

    print(f"\nCold mean intensity: {mean_cold:.4f}")
    print(f"Warm mean intensity: {mean_warm:.4f}")
    print(f"Intensity gap (warm - cold): {intensity_gap:.4f}")

    # 1. Actual cold vs warm feature distance
    actual_distances = []
    for cold_img, warm_img in zip(cold_images[:len(warm_images)], warm_images):
        cold_tensor = numpy_to_tensor(cold_img)
        warm_tensor = numpy_to_tensor(warm_img)

        cold_feat = extract_features(model, cold_tensor, device).squeeze(0)
        warm_feat = extract_features(model, warm_tensor, device).squeeze(0)

        dist = compute_feature_distance(cold_feat, warm_feat)
        actual_distances.append(dist)

    # 2. Simulated: cold + gap vs original cold
    simulated_distances = []
    for cold_img in cold_images:
        cold_tensor = numpy_to_tensor(cold_img)
        cold_feat = extract_features(model, cold_tensor, device).squeeze(0)

        # Simulate warming
        warmed_img = apply_brightness_shift(cold_img, intensity_gap)
        warmed_tensor = numpy_to_tensor(warmed_img)
        warmed_feat = extract_features(model, warmed_tensor, device).squeeze(0)

        dist = compute_feature_distance(cold_feat, warmed_feat)
        simulated_distances.append(dist)

    return {
        "intensity_gap": intensity_gap,
        "actual_cold_warm_dist": {
            "l2_mean": np.mean([d["l2_mean"] for d in actual_distances]),
            "l2_std": np.std([d["l2_mean"] for d in actual_distances]),
            "cosine_mean": np.mean([d["cosine_mean"] for d in actual_distances]),
        },
        "simulated_cold_warmed_dist": {
            "l2_mean": np.mean([d["l2_mean"] for d in simulated_distances]),
            "l2_std": np.std([d["l2_mean"] for d in simulated_distances]),
            "cosine_mean": np.mean([d["cosine_mean"] for d in simulated_distances]),
        },
    }


def visualize_sensitivity(
    brightness_results: Dict,
    contrast_results: Dict,
    cold_warm_results: Dict,
    output_path: Path
):
    """Create sensitivity visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Brightness sensitivity (L2 distance)
    ax = axes[0, 0]
    deltas = sorted(brightness_results.keys())
    l2_means = [brightness_results[d]["l2_mean"] for d in deltas]
    l2_stds = [brightness_results[d]["l2_std"] for d in deltas]

    ax.errorbar(deltas, l2_means, yerr=l2_stds, marker='o', capsize=3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Brightness Delta")
    ax.set_ylabel("L2 Feature Distance")
    ax.set_title("DINOv2 Sensitivity to Brightness")
    ax.grid(True, alpha=0.3)

    # Annotate cold-warm gap
    if cold_warm_results:
        gap = cold_warm_results["intensity_gap"]
        ax.axvline(x=gap, color='red', linestyle=':', label=f'Cold→Warm gap ({gap:.3f})')
        ax.legend()

    # 2. Brightness sensitivity (Cosine similarity)
    ax = axes[0, 1]
    cos_means = [brightness_results[d]["cosine_mean"] for d in deltas]
    cos_stds = [brightness_results[d]["cosine_std"] for d in deltas]

    ax.errorbar(deltas, cos_means, yerr=cos_stds, marker='s', capsize=3, color='green')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Brightness Delta")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("DINOv2 Feature Similarity vs Brightness")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.9, 1.01)

    # 3. Contrast sensitivity
    ax = axes[1, 0]
    factors = sorted(contrast_results.keys())
    l2_means = [contrast_results[f]["l2_mean"] for f in factors]
    l2_stds = [contrast_results[f]["l2_std"] for f in factors]

    ax.errorbar(factors, l2_means, yerr=l2_stds, marker='o', capsize=3, color='purple')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Contrast Factor")
    ax.set_ylabel("L2 Feature Distance")
    ax.set_title("DINOv2 Sensitivity to Contrast")
    ax.grid(True, alpha=0.3)

    # 4. Cold-warm comparison
    ax = axes[1, 1]
    if cold_warm_results:
        categories = ['Actual\nCold vs Warm', 'Simulated\nCold + Shift']
        l2_values = [
            cold_warm_results["actual_cold_warm_dist"]["l2_mean"],
            cold_warm_results["simulated_cold_warmed_dist"]["l2_mean"]
        ]
        cos_values = [
            cold_warm_results["actual_cold_warm_dist"]["cosine_mean"],
            cold_warm_results["simulated_cold_warmed_dist"]["cosine_mean"]
        ]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax.bar(x - width/2, l2_values, width, label='L2 Distance', color='steelblue')
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, cos_values, width, label='Cosine Sim', color='coral')

        ax.set_ylabel('L2 Distance', color='steelblue')
        ax2.set_ylabel('Cosine Similarity', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_title('Cold vs Warm: Actual vs Simulated')

        # Add value labels
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9, color='coral')

        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, 'Cold-warm analysis not available',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {output_path}")


def load_samples(domain_path: Path, n_samples: int, target_size: Tuple[int, int]) -> Tuple[List, List]:
    """Load cold and warm good samples."""
    good_path = domain_path / "test" / "good"
    good_files = sorted(good_path.glob("*.tiff"))

    cold_images = []
    warm_images = []

    for f in good_files:
        idx = int(f.stem)
        img = load_tiff_image(f, target_size)
        if idx < 500:
            cold_images.append(img)
        else:
            warm_images.append(img)

    # Subsample
    if len(cold_images) > n_samples:
        step = len(cold_images) // n_samples
        cold_images = cold_images[::step][:n_samples]
    if len(warm_images) > n_samples:
        step = len(warm_images) // n_samples
        warm_images = warm_images[::step][:n_samples]

    return cold_images, warm_images


def main():
    parser = argparse.ArgumentParser(description="Analyze DINOv2 intensity sensitivity")
    parser.add_argument("--domain", type=str, required=True,
                        choices=["domain_A", "domain_B", "domain_C", "domain_D"])
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Number of samples to analyze")
    parser.add_argument("--brightness-min", type=float, default=-0.15)
    parser.add_argument("--brightness-max", type=float, default=0.15)
    parser.add_argument("--contrast-min", type=float, default=0.7)
    parser.add_argument("--contrast-max", type=float, default=1.3)
    args = parser.parse_args()

    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / args.domain / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Analysis 3: DINOv2 Intensity Sensitivity Test")
    print(f"Domain: {args.domain}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Load samples
    domain_path = DATASET_ROOT / args.domain
    print(f"Loading samples from: {domain_path}")
    cold_images, warm_images = load_samples(domain_path, args.n_samples, config["target_size"])
    print(f"Loaded: {len(cold_images)} cold, {len(warm_images)} warm")

    # Use all images for sensitivity analysis
    all_images = cold_images + warm_images

    # Setup model
    print("\nLoading DINOv2 model...")
    model = setup_model()
    model.model.to(device)
    model.model.eval()

    # Define test ranges
    brightness_deltas = np.linspace(args.brightness_min, args.brightness_max, 11).tolist()
    contrast_factors = np.linspace(args.contrast_min, args.contrast_max, 11).tolist()

    print(f"\nBrightness deltas: {[f'{d:.3f}' for d in brightness_deltas]}")
    print(f"Contrast factors: {[f'{f:.2f}' for f in contrast_factors]}")

    # Run analyses
    print("\n1. Analyzing brightness sensitivity...")
    brightness_results = analyze_brightness_sensitivity(
        model, all_images, brightness_deltas, device
    )

    print("\n2. Analyzing contrast sensitivity...")
    contrast_results = analyze_contrast_sensitivity(
        model, all_images, contrast_factors, device
    )

    print("\n3. Analyzing cold-warm simulation...")
    cold_warm_results = analyze_cold_warm_simulation(
        model, cold_images, warm_images, device
    )

    # Create visualization
    print("\nCreating visualization...")
    visualize_sensitivity(
        brightness_results, contrast_results, cold_warm_results,
        output_dir / "sensitivity_analysis.png"
    )

    # Save results
    results = {
        "domain": args.domain,
        "n_samples": args.n_samples,
        "brightness": {str(k): v for k, v in brightness_results.items()},
        "contrast": {str(k): v for k, v in contrast_results.items()},
        "cold_warm_simulation": cold_warm_results,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")

    # Find sensitivity at cold-warm gap
    gap = cold_warm_results["intensity_gap"]
    closest_delta = min(brightness_deltas, key=lambda x: abs(x - gap))
    gap_sensitivity = brightness_results[closest_delta]["l2_mean"]

    print(f"\nCold-Warm Intensity Gap: {gap:.4f}")
    print(f"L2 distance at gap ({closest_delta:.3f}): {gap_sensitivity:.4f}")

    print(f"\nActual Cold vs Warm:")
    print(f"  L2 distance: {cold_warm_results['actual_cold_warm_dist']['l2_mean']:.4f}")
    print(f"  Cosine sim:  {cold_warm_results['actual_cold_warm_dist']['cosine_mean']:.4f}")

    print(f"\nSimulated Cold + Shift:")
    print(f"  L2 distance: {cold_warm_results['simulated_cold_warmed_dist']['l2_mean']:.4f}")
    print(f"  Cosine sim:  {cold_warm_results['simulated_cold_warmed_dist']['cosine_mean']:.4f}")

    # Interpretation
    max_brightness_l2 = max(v["l2_mean"] for v in brightness_results.values())
    actual_l2 = cold_warm_results["actual_cold_warm_dist"]["l2_mean"]
    intensity_explains_ratio = gap_sensitivity / actual_l2 if actual_l2 > 0 else 0

    print(f"\n*** INTERPRETATION ***")
    if cold_warm_results['actual_cold_warm_dist']['cosine_mean'] > 0.95:
        print("DINOv2 features are HIGHLY SIMILAR between cold and warm.")
        print("This means DINOv2 is relatively INTENSITY-INVARIANT.")
    else:
        print("DINOv2 features show SIGNIFICANT difference between cold and warm.")
        print("This means DINOv2 is SENSITIVE to intensity.")

    print(f"\nIntensity-only explains {intensity_explains_ratio*100:.1f}% of cold-warm difference.")

    if intensity_explains_ratio > 0.7:
        print("=> Intensity is the DOMINANT factor in cold-warm feature difference.")
    else:
        print("=> Structural/textural differences also contribute significantly.")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
