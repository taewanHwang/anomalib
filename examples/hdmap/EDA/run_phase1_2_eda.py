"""Phase 1-2: CutPaste Augmentation Frequency Analysis.

Analyzes how CutPaste augmentation affects image frequency characteristics.
Compares FFT spectrum before and after CutPaste to understand:
1. What frequency components CutPaste introduces
2. How paste boundary affects frequency spectrum
3. Relationship between CutPaste parameters and frequency changes

Usage:
    python run_phase1_2_eda.py --all_domains --parallel --n_workers 16
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.anomalib.models.image.draem_cutpaste_clf.synthetic_generator_v2 import (
    CutPasteSyntheticGenerator,
)
from examples.hdmap.EDA.frequency_filters import (
    get_frequency_band_energy,
    get_magnitude_spectrum,
    create_distance_from_center,
)

# Default paths and parameters
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "HDMAP" / "1000_tiff_minmax"
DEFAULT_OUTPUT_ROOT = Path(__file__).parent / "results" / "phase1_2"
DEFAULT_NUM_WORKERS = 16
DEFAULT_N_SAMPLES = 100  # Number of images to analyze per domain

# CutPaste parameters (user specified)
CUTPASTE_PARAMS = {
    "cut_w_range": (10, 80),
    "cut_h_range": (1, 2),
    "a_fault_start": 0,
    "a_fault_range_end": 2,
    "probability": 1.0,  # Always apply for analysis
    "norm": True,
}

# Target image size (before resize to model input)
TARGET_SIZE = (31, 95)  # (H, W)

DOMAINS = ["domain_A", "domain_B", "domain_C", "domain_D"]


def load_and_resize_image(img_path: Path, target_size: tuple[int, int]) -> np.ndarray:
    """Load image and resize to target size.

    Args:
        img_path: Path to image file
        target_size: Target size (H, W)

    Returns:
        Resized image as numpy array (H, W)
    """
    img = Image.open(img_path)
    # Resize to target size (PIL uses W, H)
    img_resized = img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)

    # Handle multi-channel (take first channel)
    if img_array.ndim == 3:
        img_array = img_array[..., 0]

    return img_array


def apply_cutpaste_and_analyze(
    image: np.ndarray,
    generator: CutPasteSyntheticGenerator,
    n_bands: int = 5,
) -> dict:
    """Apply CutPaste and analyze frequency changes.

    Args:
        image: Original image (H, W)
        generator: CutPaste generator
        n_bands: Number of frequency bands

    Returns:
        Dictionary with analysis results
    """
    # Convert to tensor (C, H, W)
    img_tensor = torch.from_numpy(image).unsqueeze(0).float()  # (1, H, W)

    # Apply CutPaste
    synthetic, mask, severity, patch_info = generator(img_tensor, return_patch_info=True)

    # Convert back to numpy
    synthetic_np = synthetic.squeeze().numpy()  # (H, W)
    mask_np = mask.squeeze().numpy()  # (H, W)

    # Get frequency band energies
    original_bands = get_frequency_band_energy(image, n_bands)
    synthetic_bands = get_frequency_band_energy(synthetic_np, n_bands)

    # Compute difference image and its frequency
    diff_image = synthetic_np - image
    diff_bands = get_frequency_band_energy(diff_image, n_bands)

    # Compute energy change
    band_changes = {}
    for band_name in original_bands:
        band_changes[band_name] = synthetic_bands[band_name] - original_bands[band_name]

    return {
        "original_bands": original_bands,
        "synthetic_bands": synthetic_bands,
        "diff_bands": diff_bands,
        "band_changes": band_changes,
        "patch_info": patch_info,
        "original_image": image,
        "synthetic_image": synthetic_np,
        "diff_image": diff_image,
        "mask": mask_np,
    }


def analyze_domain(
    domain: str,
    data_root: Path,
    output_dir: Path,
    n_samples: int = DEFAULT_N_SAMPLES,
    n_bands: int = 5,
    seed: int = 42,
) -> dict:
    """Analyze CutPaste frequency effects for a domain.

    Args:
        domain: Domain name
        data_root: Data root directory
        output_dir: Output directory
        n_samples: Number of samples to analyze
        n_bands: Number of frequency bands
        seed: Random seed

    Returns:
        Analysis results dictionary
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    domain_dir = data_root / domain
    train_dir = domain_dir / "train" / "good"

    # Get image paths
    image_paths = list(train_dir.glob("*.tiff")) + list(train_dir.glob("*.tif"))
    if not image_paths:
        raise FileNotFoundError(f"No TIFF images found in {train_dir}")

    # Sample images
    if len(image_paths) > n_samples:
        image_paths = random.sample(image_paths, n_samples)

    print(f"[{domain}] Analyzing {len(image_paths)} images...")

    # Create generator
    generator = CutPasteSyntheticGenerator(**CUTPASTE_PARAMS)

    # Collect results
    all_original_bands = []
    all_synthetic_bands = []
    all_diff_bands = []
    all_band_changes = []
    all_patch_info = []

    # Store a few examples for visualization
    example_results = []

    for i, img_path in enumerate(tqdm(image_paths, desc=f"[{domain}] Processing")):
        # Load and resize image
        image = load_and_resize_image(img_path, TARGET_SIZE)

        # Apply CutPaste and analyze
        result = apply_cutpaste_and_analyze(image, generator, n_bands)

        all_original_bands.append(result["original_bands"])
        all_synthetic_bands.append(result["synthetic_bands"])
        all_diff_bands.append(result["diff_bands"])
        all_band_changes.append(result["band_changes"])
        all_patch_info.append(result["patch_info"])

        # Store first 5 examples for visualization
        if i < 5:
            example_results.append(result)

    # Compute statistics
    band_names = list(all_original_bands[0].keys())

    stats = {
        "domain": domain,
        "n_samples": len(image_paths),
        "target_size": TARGET_SIZE,
        "cutpaste_params": CUTPASTE_PARAMS,
        "band_names": band_names,
    }

    # Original bands statistics
    stats["original_bands"] = {}
    for band in band_names:
        values = [r[band] for r in all_original_bands]
        stats["original_bands"][band] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    # Synthetic bands statistics
    stats["synthetic_bands"] = {}
    for band in band_names:
        values = [r[band] for r in all_synthetic_bands]
        stats["synthetic_bands"][band] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    # Difference bands statistics (frequency of diff image)
    stats["diff_bands"] = {}
    for band in band_names:
        values = [r[band] for r in all_diff_bands]
        stats["diff_bands"][band] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    # Band changes (synthetic - original)
    stats["band_changes"] = {}
    for band in band_names:
        values = [r[band] for r in all_band_changes]
        stats["band_changes"][band] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    # Patch info statistics
    patch_widths = [p["cut_w"] for p in all_patch_info if p["has_anomaly"]]
    patch_heights = [p["cut_h"] for p in all_patch_info if p["has_anomaly"]]
    amplitudes = [p["a_fault"] for p in all_patch_info if p["has_anomaly"]]
    coverages = [p["coverage_percentage"] for p in all_patch_info if p["has_anomaly"]]

    stats["patch_statistics"] = {
        "width": {"mean": float(np.mean(patch_widths)), "std": float(np.std(patch_widths))},
        "height": {"mean": float(np.mean(patch_heights)), "std": float(np.std(patch_heights))},
        "amplitude": {"mean": float(np.mean(amplitudes)), "std": float(np.std(amplitudes))},
        "coverage_pct": {"mean": float(np.mean(coverages)), "std": float(np.std(coverages))},
    }

    # Create visualizations
    domain_output = output_dir / domain
    domain_output.mkdir(parents=True, exist_ok=True)

    create_visualizations(stats, example_results, domain_output)

    # Save results
    with open(domain_output / "phase1_2_results.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def create_visualizations(
    stats: dict,
    example_results: list[dict],
    output_dir: Path,
) -> None:
    """Create visualization plots.

    Args:
        stats: Statistics dictionary
        example_results: List of example analysis results
        output_dir: Output directory
    """
    band_names = stats["band_names"]
    domain = stats["domain"]

    # 1. Bar chart: Original vs Synthetic vs Diff band energies
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.arange(len(band_names))
    width = 0.35

    # Original vs Synthetic
    ax = axes[0]
    original_means = [stats["original_bands"][b]["mean"] * 100 for b in band_names]
    synthetic_means = [stats["synthetic_bands"][b]["mean"] * 100 for b in band_names]

    bars1 = ax.bar(x - width/2, original_means, width, label='Original', alpha=0.8)
    bars2 = ax.bar(x + width/2, synthetic_means, width, label='After CutPaste', alpha=0.8)

    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Energy (%)')
    ax.set_title(f'{domain}: Original vs CutPaste Applied')
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Band changes
    ax = axes[1]
    changes = [stats["band_changes"][b]["mean"] * 100 for b in band_names]
    change_stds = [stats["band_changes"][b]["std"] * 100 for b in band_names]
    colors = ['green' if c > 0 else 'red' for c in changes]

    bars = ax.bar(x, changes, color=colors, alpha=0.8)
    ax.errorbar(x, changes, yerr=change_stds, fmt='none', color='black', capsize=3)

    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Energy Change (%p)')
    ax.set_title(f'{domain}: Energy Change (CutPaste - Original)')
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, rotation=45)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    # Diff image frequency distribution
    ax = axes[2]
    diff_means = [stats["diff_bands"][b]["mean"] * 100 for b in band_names]

    bars = ax.bar(x, diff_means, color='purple', alpha=0.8)

    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Energy (%)')
    ax.set_title(f'{domain}: Difference Image (CutPaste - Original) Frequency')
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, rotation=45)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "band_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Example images with FFT spectrums
    if example_results:
        n_examples = min(3, len(example_results))
        fig, axes = plt.subplots(n_examples, 5, figsize=(20, 4 * n_examples))

        if n_examples == 1:
            axes = axes.reshape(1, -1)

        for i, result in enumerate(example_results[:n_examples]):
            # Original image
            ax = axes[i, 0]
            ax.imshow(result["original_image"], cmap='gray')
            ax.set_title('Original')
            ax.axis('off')

            # Original FFT
            ax = axes[i, 1]
            orig_spectrum = get_magnitude_spectrum(result["original_image"])
            ax.imshow(orig_spectrum, cmap='jet')
            ax.set_title('Original FFT')
            ax.axis('off')

            # Synthetic (CutPaste applied)
            ax = axes[i, 2]
            ax.imshow(result["synthetic_image"], cmap='gray')
            # Overlay mask boundary
            mask = result["mask"]
            ax.contour(mask, colors='red', linewidths=1)
            patch_info = result["patch_info"]
            ax.set_title(f'CutPaste (amp={patch_info["a_fault"]:.2f})')
            ax.axis('off')

            # Synthetic FFT
            ax = axes[i, 3]
            synth_spectrum = get_magnitude_spectrum(result["synthetic_image"])
            ax.imshow(synth_spectrum, cmap='jet')
            ax.set_title('CutPaste FFT')
            ax.axis('off')

            # Difference FFT
            ax = axes[i, 4]
            diff_spectrum = get_magnitude_spectrum(result["diff_image"])
            ax.imshow(diff_spectrum, cmap='jet')
            ax.set_title('Difference FFT')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / "example_images_fft.png", dpi=150, bbox_inches='tight')
        plt.close()

    # 3. Patch statistics
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    patch_stats = stats["patch_statistics"]

    # Width distribution info
    ax = axes[0]
    ax.bar(['Mean'], [patch_stats["width"]["mean"]], yerr=[patch_stats["width"]["std"]], capsize=5)
    ax.set_title(f'Patch Width\n({CUTPASTE_PARAMS["cut_w_range"][0]}-{CUTPASTE_PARAMS["cut_w_range"][1]} range)')
    ax.set_ylabel('Pixels')
    ax.grid(axis='y', alpha=0.3)

    # Height distribution info
    ax = axes[1]
    ax.bar(['Mean'], [patch_stats["height"]["mean"]], yerr=[patch_stats["height"]["std"]], capsize=5)
    ax.set_title(f'Patch Height\n({CUTPASTE_PARAMS["cut_h_range"][0]}-{CUTPASTE_PARAMS["cut_h_range"][1]} range)')
    ax.set_ylabel('Pixels')
    ax.grid(axis='y', alpha=0.3)

    # Amplitude distribution info
    ax = axes[2]
    ax.bar(['Mean'], [patch_stats["amplitude"]["mean"]], yerr=[patch_stats["amplitude"]["std"]], capsize=5)
    ax.set_title(f'Amplitude\n({CUTPASTE_PARAMS["a_fault_start"]}-{CUTPASTE_PARAMS["a_fault_range_end"]} range)')
    ax.set_ylabel('Value')
    ax.grid(axis='y', alpha=0.3)

    # Coverage distribution info
    ax = axes[3]
    ax.bar(['Mean'], [patch_stats["coverage_pct"]["mean"]], yerr=[patch_stats["coverage_pct"]["std"]], capsize=5)
    ax.set_title('Patch Coverage')
    ax.set_ylabel('Percentage (%)')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "patch_statistics.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_report(all_stats: dict, output_dir: Path) -> None:
    """Create summary report across all domains.

    Args:
        all_stats: Statistics for all domains
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    domains = list(all_stats.keys())
    band_names = all_stats[domains[0]]["band_names"]

    # Summary tables
    summary = {
        "domains": domains,
        "band_names": band_names,
        "cutpaste_params": CUTPASTE_PARAMS,
        "target_size": TARGET_SIZE,
    }

    # Original bands
    summary["original_bands"] = {}
    for domain in domains:
        summary["original_bands"][domain] = {
            b: all_stats[domain]["original_bands"][b]["mean"] * 100
            for b in band_names
        }

    # Band changes
    summary["band_changes"] = {}
    for domain in domains:
        summary["band_changes"][domain] = {
            b: all_stats[domain]["band_changes"][b]["mean"] * 100
            for b in band_names
        }

    # Diff image bands (frequency distribution of CutPaste-introduced changes)
    summary["diff_bands"] = {}
    for domain in domains:
        summary["diff_bands"][domain] = {
            b: all_stats[domain]["diff_bands"][b]["mean"] * 100
            for b in band_names
        }

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Create cross-domain comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.arange(len(band_names))
    width = 0.2

    # 1. Original band comparison
    ax = axes[0, 0]
    for i, domain in enumerate(domains):
        values = [summary["original_bands"][domain][b] for b in band_names]
        ax.bar(x + i * width, values, width, label=domain.replace("_", " ").title(), alpha=0.8)

    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Energy (%)')
    ax.set_title('Original Image Energy Distribution')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(band_names, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Band changes comparison
    ax = axes[0, 1]
    for i, domain in enumerate(domains):
        values = [summary["band_changes"][domain][b] for b in band_names]
        ax.bar(x + i * width, values, width, label=domain.replace("_", " ").title(), alpha=0.8)

    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Energy Change (%p)')
    ax.set_title('CutPaste Energy Change (After - Before)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(band_names, rotation=45)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 3. Diff image frequency distribution
    ax = axes[1, 0]
    for i, domain in enumerate(domains):
        values = [summary["diff_bands"][domain][b] for b in band_names]
        ax.bar(x + i * width, values, width, label=domain.replace("_", " ").title(), alpha=0.8)

    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Energy (%)')
    ax.set_title('CutPaste Difference Image Frequency Distribution')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(band_names, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 4. Text summary
    ax = axes[1, 1]
    ax.axis('off')

    text = "CutPaste Parameters:\n"
    text += f"  cut_w_range: {CUTPASTE_PARAMS['cut_w_range']}\n"
    text += f"  cut_h_range: {CUTPASTE_PARAMS['cut_h_range']}\n"
    text += f"  a_fault: {CUTPASTE_PARAMS['a_fault_start']} - {CUTPASTE_PARAMS['a_fault_range_end']}\n"
    text += f"  norm: {CUTPASTE_PARAMS['norm']}\n\n"
    text += f"Image Size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}\n\n"
    text += "Key Findings:\n"

    # Find which band has most change
    avg_changes = {}
    for b in band_names:
        avg_changes[b] = np.mean([summary["band_changes"][d][b] for d in domains])

    most_increased = max(avg_changes, key=avg_changes.get)
    most_decreased = min(avg_changes, key=avg_changes.get)

    text += f"  - Most increased band: {most_increased} ({avg_changes[most_increased]:+.3f}%p)\n"
    text += f"  - Most decreased band: {most_decreased} ({avg_changes[most_decreased]:+.3f}%p)\n\n"

    # Diff image dominant frequency
    avg_diff = {}
    for b in band_names:
        avg_diff[b] = np.mean([summary["diff_bands"][d][b] for d in domains])

    dominant_diff_band = max(avg_diff, key=avg_diff.get)
    text += f"Diff Image Dominant Band: {dominant_diff_band} ({avg_diff[dominant_diff_band]:.1f}%)"

    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / "cross_domain_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Create README
    create_readme(summary, all_stats, output_dir)


def create_readme(summary: dict, all_stats: dict, output_dir: Path) -> None:
    """Create README markdown file.

    Args:
        summary: Summary data
        all_stats: Full statistics
        output_dir: Output directory
    """
    domains = summary["domains"]
    band_names = summary["band_names"]

    md = "# Phase 1-2: CutPaste Augmentation Frequency Analysis\n\n"

    md += "## 분석 개요\n\n"
    md += f"- **목표**: CutPaste augmentation이 이미지 주파수 특성에 미치는 영향 분석\n"
    md += f"- **데이터셋**: `1000_tiff_minmax` (Normal 이미지)\n"
    md += f"- **이미지 크기**: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} (resized)\n"
    md += f"- **주파수 대역**: {len(band_names)}개 ({', '.join(band_names)})\n\n"

    md += "## CutPaste 파라미터\n\n"
    md += "```python\n"
    md += f"cut_w_range = {CUTPASTE_PARAMS['cut_w_range']}\n"
    md += f"cut_h_range = {CUTPASTE_PARAMS['cut_h_range']}\n"
    md += f"a_fault_range = ({CUTPASTE_PARAMS['a_fault_start']}, {CUTPASTE_PARAMS['a_fault_range_end']})\n"
    md += f"norm = {CUTPASTE_PARAMS['norm']}\n"
    md += "```\n\n"

    md += "## 결과 요약\n\n"

    # Original bands table
    md += "### Original 이미지 에너지 분포 (%)\n\n"
    md += "| 도메인 |"
    for b in band_names:
        md += f" {b} |"
    md += "\n"
    md += "|--------|" + "------|" * len(band_names) + "\n"

    for domain in domains:
        md += f"| {domain.replace('_', ' ').title()} |"
        for b in band_names:
            val = summary["original_bands"][domain][b]
            md += f" {val:.2f} |"
        md += "\n"

    md += "\n"

    # Band changes table
    md += "### CutPaste 적용 후 변화 (%p)\n\n"
    md += "| 도메인 |"
    for b in band_names:
        md += f" {b} |"
    md += "\n"
    md += "|--------|" + "------|" * len(band_names) + "\n"

    for domain in domains:
        md += f"| {domain.replace('_', ' ').title()} |"
        for b in band_names:
            val = summary["band_changes"][domain][b]
            sign = "+" if val >= 0 else ""
            # Bold if significant change
            if abs(val) > 0.5:
                md += f" **{sign}{val:.2f}** |"
            else:
                md += f" {sign}{val:.2f} |"
        md += "\n"

    md += "\n"

    # Diff image frequency table
    md += "### CutPaste가 도입한 변화의 주파수 분포 (%)\n\n"
    md += "*Diff Image (CutPaste - Original)의 FFT 에너지 분포*\n\n"
    md += "| 도메인 |"
    for b in band_names:
        md += f" {b} |"
    md += "\n"
    md += "|--------|" + "------|" * len(band_names) + "\n"

    for domain in domains:
        md += f"| {domain.replace('_', ' ').title()} |"
        for b in band_names:
            val = summary["diff_bands"][domain][b]
            # Bold if dominant
            if val > 30:
                md += f" **{val:.2f}** |"
            else:
                md += f" {val:.2f} |"
        md += "\n"

    md += "\n"

    # Key findings
    md += "## 핵심 발견\n\n"

    avg_changes = {}
    for b in band_names:
        avg_changes[b] = np.mean([summary["band_changes"][d][b] for d in domains])

    avg_diff = {}
    for b in band_names:
        avg_diff[b] = np.mean([summary["diff_bands"][d][b] for d in domains])

    most_increased = max(avg_changes, key=avg_changes.get)
    most_decreased = min(avg_changes, key=avg_changes.get)
    dominant_diff = max(avg_diff, key=avg_diff.get)

    md += f"1. **CutPaste → {most_increased} 에너지 증가** ({avg_changes[most_increased]:+.3f}%p 평균)\n"
    md += f"2. **CutPaste → {most_decreased} 에너지 감소** ({avg_changes[most_decreased]:+.3f}%p 평균)\n"
    md += f"3. **CutPaste가 도입하는 변화는 주로 {dominant_diff} 대역** ({avg_diff[dominant_diff]:.1f}% 에너지)\n\n"

    md += "## 생성 파일\n\n"
    md += "각 도메인 폴더:\n"
    md += "- `band_comparison.png`: 에너지 분포 비교\n"
    md += "- `example_images_fft.png`: 예시 이미지와 FFT\n"
    md += "- `patch_statistics.png`: 패치 통계\n"
    md += "- `phase1_2_results.json`: 상세 결과\n\n"
    md += "루트 폴더:\n"
    md += "- `cross_domain_comparison.png`: 도메인간 비교\n"
    md += "- `summary.json`: 요약 데이터\n"

    with open(output_dir / "README.md", "w") as f:
        f.write(md)


def main():
    parser = argparse.ArgumentParser(description="Phase 1-2: CutPaste Frequency Analysis")
    parser.add_argument("--domain", type=str, default=None, help="Single domain to analyze")
    parser.add_argument("--all_domains", action="store_true", help="Analyze all domains")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT, help="Data root path")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output path")
    parser.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES, help="Samples per domain")
    parser.add_argument("--n_bands", type=int, default=5, help="Number of frequency bands")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Determine domains to analyze
    if args.all_domains:
        domains = DOMAINS
    elif args.domain:
        domains = [args.domain]
    else:
        domains = ["domain_A"]

    print(f"Phase 1-2: CutPaste Frequency Analysis")
    print(f"Data root: {args.data_root}")
    print(f"Output: {args.output_root}")
    print(f"Domains: {domains}")
    print(f"Samples per domain: {args.n_samples}")
    print(f"CutPaste params: {CUTPASTE_PARAMS}")
    print()

    # Analyze each domain
    all_stats = {}
    for domain in domains:
        stats = analyze_domain(
            domain=domain,
            data_root=args.data_root,
            output_dir=args.output_root,
            n_samples=args.n_samples,
            n_bands=args.n_bands,
            seed=args.seed,
        )
        all_stats[domain] = stats

    # Create summary
    if len(domains) > 1:
        create_summary_report(all_stats, args.output_root)

    print("\nDone! Results saved to:", args.output_root)


if __name__ == "__main__":
    main()
