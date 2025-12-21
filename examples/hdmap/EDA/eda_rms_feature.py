#!/usr/bin/env python3
"""
RMS (Root Mean Square) Feature EDA for HDMAP TIFF Dataset.

RMS는 이미지의 전체적인 에너지/강도를 측정하는 지표입니다.
- RMS = sqrt(mean(pixel_values^2))
- 이미지의 전반적인 신호 세기를 표현
- 정상 샘플과 비정상 샘플의 에너지 분포 차이를 분석

분석 대상:
- 4개 도메인: domain_A, domain_B, domain_C, domain_D
- 3개 split: train_good, test_good, test_fault
- 정상 vs 비정상 분포 비교, Cohen's d 계산
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Dataset path - TIFF original!
HDMAP_TIFF_ROOT = "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_tiff_minmax"
DOMAINS = ["domain_A", "domain_B", "domain_C", "domain_D"]


def load_tiff_images(domain_path: Path, split: str, label: str, max_samples: int = None) -> List[np.ndarray]:
    """Load TIFF images from a directory (preserving original values)."""
    img_dir = domain_path / split / label
    if not img_dir.exists():
        print(f"    Warning: Directory not found: {img_dir}")
        return []

    img_files = sorted(img_dir.glob("*.tiff")) + sorted(img_dir.glob("*.tif"))
    if max_samples:
        img_files = img_files[:max_samples]

    if not img_files:
        print(f"    Warning: No TIFF files found in {img_dir}")
        return []

    images = []
    for img_path in img_files:
        try:
            img = Image.open(img_path)
            # TIFF mode 'F' = 32-bit float, preserve original values!
            arr = np.array(img, dtype=np.float32)
            images.append(arr)
        except Exception as e:
            print(f"    Error loading {img_path}: {e}")
            continue

    return images


# ============================================================
# RMS Feature Computation
# ============================================================
def compute_rms(image: np.ndarray) -> float:
    """
    Compute RMS (Root Mean Square) of image pixel values.

    RMS = sqrt(mean(pixel_values^2))

    이미지의 전체적인 에너지/강도를 측정하는 지표입니다.
    - 높은 RMS: 전반적으로 높은 픽셀 값 (밝거나 강한 신호)
    - 낮은 RMS: 전반적으로 낮은 픽셀 값 (어둡거나 약한 신호)

    Args:
        image: 2D grayscale image (H, W)

    Returns:
        RMS value (float)
    """
    # RMS = sqrt(mean(x^2))
    mean_squared = np.mean(image ** 2)
    rms = np.sqrt(mean_squared)
    return float(rms)


def compute_rms_statistics(image: np.ndarray) -> Dict[str, float]:
    """
    RMS 외에 추가적인 통계를 계산합니다.

    Returns:
        Dictionary with RMS and related statistics
    """
    rms = compute_rms(image)

    # 비교를 위한 다른 지표들
    mean_val = float(np.mean(image))
    std_val = float(np.std(image))

    # RMS는 mean^2 + std^2의 제곱근과 같습니다
    # RMS^2 = mean^2 + variance
    rms_theoretical = np.sqrt(mean_val**2 + std_val**2)

    return {
        "rms": rms,
        "mean": mean_val,
        "std": std_val,
        "rms_theoretical": float(rms_theoretical),  # 검증용
        "min": float(np.min(image)),
        "max": float(np.max(image)),
        "energy": float(np.sum(image ** 2)),  # Total energy
    }


def compute_all_rms_features(images: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute RMS features for a list of images."""
    rms_values = []
    mean_values = []
    std_values = []
    energy_values = []

    for img in images:
        stats = compute_rms_statistics(img)
        rms_values.append(stats["rms"])
        mean_values.append(stats["mean"])
        std_values.append(stats["std"])
        energy_values.append(stats["energy"])

    return {
        "rms": np.array(rms_values),
        "mean": np.array(mean_values),
        "std": np.array(std_values),
        "energy": np.array(energy_values),
    }


def compute_summary_statistics(values: np.ndarray) -> Dict:
    """Compute summary statistics."""
    if len(values) == 0:
        return {}

    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
    }


def run_eda_for_domain(domain: str, max_samples: int = None) -> Dict:
    """Run RMS EDA for a single domain."""
    domain_path = Path(HDMAP_TIFF_ROOT) / domain

    print(f"\n{'='*60}")
    print(f"Processing {domain} - RMS Feature Analysis")
    print(f"{'='*60}")

    results = {"domain": domain, "splits": {}}

    splits = [("train", "good"), ("test", "good"), ("test", "fault")]

    for split, label in splits:
        split_key = f"{split}_{label}"
        print(f"\n  Loading {split_key}...")

        images = load_tiff_images(domain_path, split, label, max_samples)
        if not images:
            print(f"    No images found or loaded")
            continue

        print(f"    Loaded {len(images)} TIFF images")
        print(f"    Image shape: {images[0].shape}")
        print(f"    Computing RMS features...")

        features = compute_all_rms_features(images)

        # Print sample statistics
        sample_rms = features["rms"][:3] if len(features["rms"]) >= 3 else features["rms"]
        print(f"    Sample RMS values: {sample_rms}")

        results["splits"][split_key] = {
            "stats": {name: compute_summary_statistics(values) for name, values in features.items()},
            "raw": {name: values.tolist() for name, values in features.items()},
        }

    return results


def compute_separability(all_results: List[Dict]) -> Dict:
    """
    Compute separability metrics between normal and fault samples.

    Metrics computed:
    - Cohen's d: Effect size (standardized mean difference)
    - t-statistic and p-value: Statistical significance test
    - Mean difference and percentage difference
    """
    separability = {}
    feature_names = ["rms", "mean", "std", "energy"]

    for domain_result in all_results:
        domain = domain_result["domain"]
        separability[domain] = {}

        splits = domain_result["splits"]
        if "test_good" not in splits or "test_fault" not in splits:
            print(f"  Warning: Missing test splits for {domain}")
            continue

        for feature in feature_names:
            normal_raw = splits["test_good"].get("raw", {}).get(feature, [])
            fault_raw = splits["test_fault"].get("raw", {}).get(feature, [])

            if not normal_raw or not fault_raw:
                continue

            normal = np.array(normal_raw)
            fault = np.array(fault_raw)

            # Cohen's d (effect size)
            pooled_std = np.sqrt((np.var(normal) + np.var(fault)) / 2)
            if pooled_std > 1e-10:
                cohens_d = (np.mean(fault) - np.mean(normal)) / pooled_std
            else:
                cohens_d = 0.0

            # t-test
            t_stat, p_value = stats.ttest_ind(fault, normal, equal_var=False)

            # Mean difference
            mean_diff = np.mean(fault) - np.mean(normal)
            if np.mean(normal) != 0:
                diff_percent = (mean_diff / np.mean(normal)) * 100
            else:
                diff_percent = 0.0

            separability[domain][feature] = {
                "cohens_d": float(cohens_d),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "normal_mean": float(np.mean(normal)),
                "normal_std": float(np.std(normal)),
                "fault_mean": float(np.mean(fault)),
                "fault_std": float(np.std(fault)),
                "mean_diff": float(mean_diff),
                "diff_percent": float(diff_percent),
                "direction": "fault_higher" if np.mean(fault) > np.mean(normal) else "fault_lower",
            }

    return separability


def print_separability_analysis(separability: Dict):
    """Print detailed separability analysis."""
    print("\n" + "="*140)
    print("SEPARABILITY ANALYSIS - RMS Feature (Normal vs Fault)")
    print("="*140)

    header = f"{'Domain':<12} | {'Feature':<10} | {'Cohen d':>10} | {'t-stat':>10} | {'p-value':>10} | {'Normal':>12} | {'Fault':>12} | {'Diff %':>10} | {'Direction':<12} | {'Sig':<5}"
    print(header)
    print("-"*140)

    for domain in sorted(separability.keys()):
        features = separability[domain]
        for feature, metrics in sorted(features.items()):
            d = metrics["cohens_d"]
            t_stat = metrics["t_statistic"]
            p_val = metrics["p_value"]
            normal_mean = metrics["normal_mean"]
            fault_mean = metrics["fault_mean"]
            diff_pct = metrics["diff_percent"]
            direction = metrics["direction"]

            # Significance markers
            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            else:
                sig = ""

            # Effect size interpretation
            abs_d = abs(d)
            if abs_d >= 0.8:
                effect = " [L]"  # Large
            elif abs_d >= 0.5:
                effect = " [M]"  # Medium
            elif abs_d >= 0.2:
                effect = " [S]"  # Small
            else:
                effect = ""

            line = f"{domain:<12} | {feature:<10} | {d:>10.4f} | {t_stat:>10.4f} | {p_val:>10.6f} | {normal_mean:>12.4f} | {fault_mean:>12.4f} | {diff_pct:>+9.1f}% | {direction:<12} | {sig:<5}{effect}"
            print(line)

    print("\nEffect Size Legend: [S]=Small(0.2-0.5), [M]=Medium(0.5-0.8), [L]=Large(>=0.8)")
    print("Significance: * p<0.05, ** p<0.01, *** p<0.001")


def create_distribution_plots(all_results: List[Dict], output_dir: Path):
    """Create distribution comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # RMS Feature에 집중
    feature = "rms"

    # 1. Histogram comparison for each domain
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, domain_result in enumerate(all_results):
        ax = axes[idx]
        domain = domain_result["domain"]

        splits = domain_result["splits"]

        # Plot distributions
        for split_key, color, label, alpha in [
            ("train_good", "blue", "Train Good", 0.5),
            ("test_good", "green", "Test Good", 0.5),
            ("test_fault", "red", "Test Fault", 0.6)
        ]:
            if split_key in splits:
                data = splits[split_key].get("raw", {}).get(feature, [])
                if data:
                    ax.hist(data, bins=30, color=color, alpha=alpha, label=label, edgecolor='black')

        ax.set_title(f"{domain} - RMS Distribution", fontsize=12, fontweight='bold')
        ax.set_xlabel("RMS Value", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "rms_histogram_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")

    # 2. Box plot comparison
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    for idx, domain_result in enumerate(all_results):
        ax = axes[idx]
        domain = domain_result["domain"]

        data_to_plot = []
        labels = []
        colors = []

        for split_key in ["train_good", "test_good", "test_fault"]:
            if split_key in domain_result["splits"]:
                raw_data = domain_result["splits"][split_key].get("raw", {}).get(feature, [])
                if raw_data:
                    data_to_plot.append(raw_data)
                    labels.append(split_key.replace("_", "\n"))
                    colors.append("red" if "fault" in split_key else "blue")

        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True)

            # Color boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            ax.set_title(f"{domain}", fontsize=12, fontweight='bold')
            ax.set_ylabel("RMS Value", fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle("RMS Feature - Box Plot Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = output_dir / "rms_boxplot_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")

    # 3. Violin plot for better distribution visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    for idx, domain_result in enumerate(all_results):
        ax = axes[idx]
        domain = domain_result["domain"]

        data_to_plot = []
        labels = []
        colors = []

        for split_key in ["train_good", "test_good", "test_fault"]:
            if split_key in domain_result["splits"]:
                raw_data = domain_result["splits"][split_key].get("raw", {}).get(feature, [])
                if raw_data:
                    data_to_plot.append(raw_data)
                    labels.append(split_key.replace("_", "\n"))
                    colors.append("red" if "fault" in split_key else "blue")

        if data_to_plot:
            parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)),
                                 showmeans=True, showmedians=True)

            # Color violin bodies
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i] if i < len(colors) else 'gray')
                pc.set_alpha(0.7)

            ax.set_xticks(range(len(data_to_plot)))
            ax.set_xticklabels(labels)
            ax.set_title(f"{domain}", fontsize=12, fontweight='bold')
            ax.set_ylabel("RMS Value", fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle("RMS Feature - Violin Plot Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = output_dir / "rms_violin_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")


def create_separability_heatmap(separability: Dict, output_dir: Path):
    """Create heatmap of Cohen's d values."""
    domains = sorted(separability.keys())
    features = ["rms", "mean", "std", "energy"]

    # Create matrix
    matrix = np.zeros((len(domains), len(features)))

    for i, domain in enumerate(domains):
        for j, feature in enumerate(features):
            if feature in separability[domain]:
                matrix[i, j] = separability[domain][feature]["cohens_d"]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)

    # Set ticks
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(domains)))
    ax.set_xticklabels(features)
    ax.set_yticklabels(domains)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cohen's d", rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(domains)):
        for j in range(len(features)):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=10)

    ax.set_title("Separability Heatmap (Cohen's d)\nPositive: Fault > Normal, Negative: Fault < Normal",
                fontsize=12, fontweight='bold')
    ax.set_xlabel("Feature", fontsize=10)
    ax.set_ylabel("Domain", fontsize=10)

    plt.tight_layout()
    plot_path = output_dir / "separability_heatmap.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")


def create_domain_summary_report(all_results: List[Dict], separability: Dict, output_dir: Path):
    """Create a text summary report."""
    report_path = output_dir / "rms_eda_summary.txt"

    with open(report_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("RMS FEATURE EDA SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("RMS (Root Mean Square) 정의:\n")
        f.write("  RMS = sqrt(mean(pixel_values^2))\n")
        f.write("  이미지의 전체적인 에너지/강도를 측정하는 지표\n\n")

        # Per-domain analysis
        for domain_result in all_results:
            domain = domain_result["domain"]
            f.write("\n" + "-"*80 + "\n")
            f.write(f"{domain} Analysis\n")
            f.write("-"*80 + "\n\n")

            splits = domain_result["splits"]

            # Print statistics for each split
            for split_key in ["train_good", "test_good", "test_fault"]:
                if split_key in splits:
                    stats = splits[split_key]["stats"].get("rms", {})
                    if stats:
                        f.write(f"  {split_key}:\n")
                        f.write(f"    Count:  {stats.get('count', 0)}\n")
                        f.write(f"    Mean:   {stats.get('mean', 0):.4f}\n")
                        f.write(f"    Std:    {stats.get('std', 0):.4f}\n")
                        f.write(f"    Median: {stats.get('median', 0):.4f}\n")
                        f.write(f"    Range:  [{stats.get('min', 0):.4f}, {stats.get('max', 0):.4f}]\n")
                        f.write(f"    IQR:    {stats.get('iqr', 0):.4f}\n\n")

            # Separability metrics
            if domain in separability and "rms" in separability[domain]:
                sep = separability[domain]["rms"]
                f.write(f"  Separability (test_good vs test_fault):\n")
                f.write(f"    Cohen's d:        {sep['cohens_d']:.4f}\n")
                f.write(f"    t-statistic:      {sep['t_statistic']:.4f}\n")
                f.write(f"    p-value:          {sep['p_value']:.6f}\n")
                f.write(f"    Normal mean:      {sep['normal_mean']:.4f}\n")
                f.write(f"    Fault mean:       {sep['fault_mean']:.4f}\n")
                f.write(f"    Difference:       {sep['mean_diff']:.4f} ({sep['diff_percent']:+.1f}%)\n")
                f.write(f"    Direction:        {sep['direction']}\n")

                # Interpretation
                abs_d = abs(sep['cohens_d'])
                if abs_d >= 0.8:
                    interpretation = "Large effect (strong discrimination)"
                elif abs_d >= 0.5:
                    interpretation = "Medium effect (moderate discrimination)"
                elif abs_d >= 0.2:
                    interpretation = "Small effect (weak discrimination)"
                else:
                    interpretation = "Negligible effect (no discrimination)"

                f.write(f"    Interpretation:   {interpretation}\n\n")

        # Overall summary
        f.write("\n" + "="*80 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("="*80 + "\n\n")

        # Rank domains by absolute Cohen's d
        domain_scores = []
        for domain in separability.keys():
            if "rms" in separability[domain]:
                d = abs(separability[domain]["rms"]["cohens_d"])
                domain_scores.append((domain, d))

        domain_scores.sort(key=lambda x: x[1], reverse=True)

        f.write("Domain Ranking (by RMS discrimination capability):\n")
        for rank, (domain, d) in enumerate(domain_scores, 1):
            f.write(f"  {rank}. {domain}: |Cohen's d| = {d:.4f}\n")

        f.write("\n결론:\n")
        if domain_scores:
            best_domain, best_d = domain_scores[0]
            f.write(f"  - RMS 피처는 {best_domain}에서 가장 높은 분별력을 보임 (|d|={best_d:.4f})\n")

            # Overall assessment
            avg_d = np.mean([d for _, d in domain_scores])
            f.write(f"  - 평균 effect size: {avg_d:.4f}\n")

            if avg_d >= 0.8:
                f.write("  - RMS는 전반적으로 강한 분별력을 가진 피처입니다.\n")
            elif avg_d >= 0.5:
                f.write("  - RMS는 전반적으로 중간 정도의 분별력을 가진 피처입니다.\n")
            elif avg_d >= 0.2:
                f.write("  - RMS는 전반적으로 약한 분별력을 가진 피처입니다.\n")
            else:
                f.write("  - RMS는 분별력이 거의 없는 피처입니다.\n")

    print(f"  Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="RMS Feature EDA on HDMAP TIFF Data")
    parser.add_argument("--output-dir", type=str, default="results/rms_eda",
                        help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per split (for testing)")
    args = parser.parse_args()

    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("HDMAP RMS FEATURE EDA")
    print("RMS = sqrt(mean(pixel_values^2))")
    print("="*70)

    # Run EDA for all domains
    all_results = []
    for domain in DOMAINS:
        result = run_eda_for_domain(domain, max_samples=args.max_samples)
        all_results.append(result)

    # Compute separability
    print("\n" + "="*70)
    print("Computing Separability Metrics...")
    print("="*70)
    separability = compute_separability(all_results)
    print_separability_analysis(separability)

    # Create visualizations
    print("\n" + "="*70)
    print("Creating Visualizations...")
    print("="*70)
    create_distribution_plots(all_results, output_dir)
    create_separability_heatmap(separability, output_dir)

    # Create summary report
    print("\n" + "="*70)
    print("Creating Summary Report...")
    print("="*70)
    create_domain_summary_report(all_results, separability, output_dir)

    # Save JSON results
    results_file = output_dir / "rms_eda_results.json"
    with open(results_file, "w") as f:
        # Prepare results for JSON (exclude raw data for file size)
        results_for_json = []
        for r in all_results:
            r_copy = {"domain": r["domain"], "splits": {}}
            for split_key, split_data in r["splits"].items():
                r_copy["splits"][split_key] = {"stats": split_data.get("stats", {})}
            results_for_json.append(r_copy)

        json.dump({
            "results": results_for_json,
            "separability": separability
        }, f, indent=2)

    print(f"\n  Results saved to: {results_file}")

    # Print final summary
    print("\n" + "="*70)
    print("RMS EDA COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - rms_histogram_comparison.png")
    print("  - rms_boxplot_comparison.png")
    print("  - rms_violin_comparison.png")
    print("  - separability_heatmap.png")
    print("  - rms_eda_summary.txt")
    print("  - rms_eda_results.json")

    print("\n추천 사항:")
    # Find best and worst domains
    domain_scores = []
    for domain in separability.keys():
        if "rms" in separability[domain]:
            d = abs(separability[domain]["rms"]["cohens_d"])
            direction = separability[domain]["rms"]["direction"]
            domain_scores.append((domain, d, direction))

    if domain_scores:
        domain_scores.sort(key=lambda x: x[1], reverse=True)
        best_domain, best_d, best_dir = domain_scores[0]
        worst_domain, worst_d, worst_dir = domain_scores[-1]

        print(f"  - 가장 높은 분별력: {best_domain} (|Cohen's d| = {best_d:.4f}, {best_dir})")
        print(f"  - 가장 낮은 분별력: {worst_domain} (|Cohen's d| = {worst_d:.4f}, {worst_dir})")

        avg_d = np.mean([d for _, d, _ in domain_scores])
        if avg_d >= 0.5:
            print(f"  - RMS 피처는 fault detection에 유용한 피처로 보입니다.")
        else:
            print(f"  - RMS 피처의 분별력이 제한적입니다. 다른 피처와 조합하여 사용을 권장합니다.")


if __name__ == "__main__":
    main()
