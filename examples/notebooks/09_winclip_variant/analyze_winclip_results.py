#!/usr/bin/env python3
"""
WinCLIP HDMAP Results Analyzer.

This script parses and analyzes WinCLIP experiment results,
computing statistics per domain and k_shot mode.

Usage:
    python examples/notebooks/09_winclip_variant/analyze_winclip_results.py \
        --result-dir results/winclip_hdmap_validation

    # Multiple experiments comparison
    python examples/notebooks/09_winclip_variant/analyze_winclip_results.py \
        --result-dir results/winclip_hdmap_exp1 results/winclip_hdmap_exp2 --compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_winclip_results(result_dir: Path) -> list[dict[str, Any]]:
    """Load all WinCLIP experiment results from a directory.

    Args:
        result_dir: Path to experiment results directory.

    Returns:
        List of result dictionaries.
    """
    all_results = []

    # Find all timestamp directories
    for exp_folder in sorted(result_dir.iterdir()):
        if not exp_folder.is_dir():
            continue

        # Load summary.json if exists
        summary_file = exp_folder / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                results = json.load(f)
                for r in results:
                    r["_exp_folder"] = str(exp_folder)
                    r["_timestamp"] = exp_folder.name
                all_results.extend(results)
        else:
            # Load individual domain results
            for domain_folder in exp_folder.iterdir():
                if domain_folder.is_dir() and domain_folder.name.startswith("domain_"):
                    for mode_folder in domain_folder.iterdir():
                        if mode_folder.is_dir():
                            results_file = mode_folder / "results.json"
                            if results_file.exists():
                                with open(results_file) as f:
                                    result = json.load(f)
                                    result["_exp_folder"] = str(exp_folder)
                                    result["_timestamp"] = exp_folder.name
                                    all_results.append(result)

    return all_results


def compute_statistics(values: list[float]) -> dict[str, float]:
    """Compute mean, std, min, max for a list of values."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": len(values),
    }


def format_pct(value: float, width: int = 6) -> str:
    """Format a value as percentage string."""
    return f"{value * 100:>{width}.2f}%"


def format_pct_with_std(mean: float, std: float) -> str:
    """Format mean ± std as percentage string."""
    return f"{mean * 100:.2f}% ± {std * 100:.2f}%"


def analyze_winclip_experiment(result_dir: Path) -> dict[str, Any]:
    """Analyze WinCLIP experiment directory.

    Args:
        result_dir: Path to experiment results directory.

    Returns:
        Analysis results dictionary.
    """
    results = load_winclip_results(result_dir)

    if not results:
        print(f"No results found in {result_dir}")
        return {}

    # Group results by k_shot and domain
    by_kshot = {}  # k_shot -> [results]
    by_domain = {}  # domain -> [results]
    by_kshot_domain = {}  # (k_shot, domain) -> result

    domains = set()
    k_shots = set()
    class_names = set()
    timestamps = set()

    for r in results:
        if "error" in r:
            continue

        k_shot = r.get("k_shot", 0)
        domain = r.get("domain", "unknown")
        class_name = r.get("class_name", "unknown")
        timestamp = r.get("_timestamp", "unknown")

        domains.add(domain)
        k_shots.add(k_shot)
        class_names.add(class_name)
        timestamps.add(timestamp)

        if k_shot not in by_kshot:
            by_kshot[k_shot] = []
        by_kshot[k_shot].append(r)

        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(r)

        by_kshot_domain[(k_shot, domain)] = r

    # Compute per-k_shot statistics
    kshot_stats = {}
    for k_shot in sorted(k_shots):
        results_k = by_kshot.get(k_shot, [])

        auroc_values = []
        f1_values = []

        for r in results_k:
            metrics = r.get("metrics", {})
            if "image_AUROC" in metrics:
                auroc_values.append(metrics["image_AUROC"])
            if "image_F1Score" in metrics:
                f1_values.append(metrics["image_F1Score"])

        kshot_stats[k_shot] = {
            "auroc": compute_statistics(auroc_values),
            "f1": compute_statistics(f1_values),
            "num_domains": len(results_k),
        }

    # Compute per-domain statistics
    domain_stats = {}
    for domain in sorted(domains):
        results_d = by_domain.get(domain, [])

        auroc_values = []
        f1_values = []

        for r in results_d:
            metrics = r.get("metrics", {})
            if "image_AUROC" in metrics:
                auroc_values.append(metrics["image_AUROC"])
            if "image_F1Score" in metrics:
                f1_values.append(metrics["image_F1Score"])

        domain_stats[domain] = {
            "auroc": compute_statistics(auroc_values),
            "f1": compute_statistics(f1_values),
            "num_kshots": len(results_d),
        }

    return {
        "result_dir": str(result_dir),
        "num_results": len(results),
        "domains": sorted(domains),
        "k_shots": sorted(k_shots),
        "class_names": list(class_names),
        "timestamps": sorted(timestamps),
        "kshot_stats": kshot_stats,
        "domain_stats": domain_stats,
        "by_kshot_domain": by_kshot_domain,
        "raw_results": results,
    }


def print_analysis(analysis: dict[str, Any]) -> None:
    """Print analysis results in a formatted way."""
    if not analysis:
        return

    print("\n" + "=" * 80)
    print(f"WINCLIP HDMAP ANALYSIS: {Path(analysis['result_dir']).name}")
    print("=" * 80)

    print(f"\nTotal results: {analysis['num_results']}")
    print(f"Domains: {', '.join(analysis['domains'])}")
    print(f"k_shot modes: {', '.join(map(str, analysis['k_shots']))}")
    print(f"Class names: {', '.join(analysis['class_names'])}")
    print(f"Experiments: {', '.join(analysis['timestamps'])}")

    # Per-k_shot summary table
    print("\n" + "-" * 80)
    print("PER K-SHOT MODE SUMMARY (Mean across domains)")
    print("-" * 80)

    header = f"{'Mode':<12} {'Image AUROC':>20} {'Image F1':>20} {'#Domains':>10}"
    print(header)
    print("-" * len(header))

    for k_shot in analysis["k_shots"]:
        stats = analysis["kshot_stats"].get(k_shot, {})
        mode_name = "Zero-shot" if k_shot == 0 else f"{k_shot}-shot"

        auroc_stats = stats.get("auroc", {})
        f1_stats = stats.get("f1", {})
        num_domains = stats.get("num_domains", 0)

        if auroc_stats.get("count", 0) > 0:
            auroc_str = format_pct_with_std(auroc_stats["mean"], auroc_stats["std"])
        else:
            auroc_str = "N/A"

        if f1_stats.get("count", 0) > 0:
            f1_str = format_pct_with_std(f1_stats["mean"], f1_stats["std"])
        else:
            f1_str = "N/A"

        print(f"{mode_name:<12} {auroc_str:>20} {f1_str:>20} {num_domains:>10}")

    # Per-domain summary table
    print("\n" + "-" * 80)
    print("PER DOMAIN SUMMARY (Mean across k_shot modes)")
    print("-" * 80)

    header = f"{'Domain':<12} {'Image AUROC':>20} {'Image F1':>20} {'#Modes':>10}"
    print(header)
    print("-" * len(header))

    for domain in analysis["domains"]:
        stats = analysis["domain_stats"].get(domain, {})

        auroc_stats = stats.get("auroc", {})
        f1_stats = stats.get("f1", {})
        num_kshots = stats.get("num_kshots", 0)

        if auroc_stats.get("count", 0) > 0:
            auroc_str = format_pct_with_std(auroc_stats["mean"], auroc_stats["std"])
        else:
            auroc_str = "N/A"

        if f1_stats.get("count", 0) > 0:
            f1_str = format_pct_with_std(f1_stats["mean"], f1_stats["std"])
        else:
            f1_str = "N/A"

        print(f"{domain:<12} {auroc_str:>20} {f1_str:>20} {num_kshots:>10}")

    # Detailed results matrix (k_shot x domain)
    print("\n" + "-" * 80)
    print("DETAILED RESULTS MATRIX (Image AUROC)")
    print("-" * 80)

    # Header
    header = f"{'Mode':<12}"
    for domain in analysis["domains"]:
        header += f" {domain:>12}"
    header += f" {'Mean':>12}"
    print(header)
    print("-" * len(header))

    for k_shot in analysis["k_shots"]:
        mode_name = "Zero-shot" if k_shot == 0 else f"{k_shot}-shot"
        row = f"{mode_name:<12}"

        auroc_values = []
        for domain in analysis["domains"]:
            result = analysis["by_kshot_domain"].get((k_shot, domain))
            if result and "metrics" in result:
                auroc = result["metrics"].get("image_AUROC", None)
                if auroc is not None:
                    row += f" {auroc * 100:>11.2f}%"
                    auroc_values.append(auroc)
                else:
                    row += f" {'N/A':>12}"
            else:
                row += f" {'N/A':>12}"

        # Mean
        if auroc_values:
            mean_auroc = np.mean(auroc_values)
            row += f" {mean_auroc * 100:>11.2f}%"
        else:
            row += f" {'N/A':>12}"

        print(row)

    # F1 Score Matrix
    print("\n" + "-" * 80)
    print("DETAILED RESULTS MATRIX (Image F1 Score)")
    print("-" * 80)

    # Header
    header = f"{'Mode':<12}"
    for domain in analysis["domains"]:
        header += f" {domain:>12}"
    header += f" {'Mean':>12}"
    print(header)
    print("-" * len(header))

    for k_shot in analysis["k_shots"]:
        mode_name = "Zero-shot" if k_shot == 0 else f"{k_shot}-shot"
        row = f"{mode_name:<12}"

        f1_values = []
        for domain in analysis["domains"]:
            result = analysis["by_kshot_domain"].get((k_shot, domain))
            if result and "metrics" in result:
                f1 = result["metrics"].get("image_F1Score", None)
                if f1 is not None:
                    row += f" {f1 * 100:>11.2f}%"
                    f1_values.append(f1)
                else:
                    row += f" {'N/A':>12}"
            else:
                row += f" {'N/A':>12}"

        # Mean
        if f1_values:
            mean_f1 = np.mean(f1_values)
            row += f" {mean_f1 * 100:>11.2f}%"
        else:
            row += f" {'N/A':>12}"

        print(row)


def compare_experiments(analyses: list[tuple[Path, dict[str, Any]]]) -> None:
    """Compare multiple experiments side by side."""
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPARISON")
    print("=" * 80)

    # Get all k_shots across experiments
    all_kshots = set()
    for _, analysis in analyses:
        all_kshots.update(analysis.get("k_shots", []))

    # Compare per k_shot
    for k_shot in sorted(all_kshots):
        mode_name = "Zero-shot" if k_shot == 0 else f"{k_shot}-shot"
        print(f"\n{mode_name} Mode:")
        print("-" * 60)

        header = f"{'Experiment':<30} {'Image AUROC':>15} {'Image F1':>15}"
        print(header)
        print("-" * len(header))

        for result_dir, analysis in analyses:
            exp_name = result_dir.name[:28]
            stats = analysis.get("kshot_stats", {}).get(k_shot, {})

            auroc_stats = stats.get("auroc", {})
            f1_stats = stats.get("f1", {})

            if auroc_stats.get("count", 0) > 0:
                auroc_str = f"{auroc_stats['mean'] * 100:.2f}%"
            else:
                auroc_str = "N/A"

            if f1_stats.get("count", 0) > 0:
                f1_str = f"{f1_stats['mean'] * 100:.2f}%"
            else:
                f1_str = "N/A"

            print(f"{exp_name:<30} {auroc_str:>15} {f1_str:>15}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze WinCLIP HDMAP experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--result-dir",
        nargs="+",
        type=Path,
        required=True,
        help="Path(s) to experiment result directories",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple experiments side by side",
    )

    args = parser.parse_args()

    analyses = []
    for result_dir in args.result_dir:
        if not result_dir.exists():
            print(f"Warning: {result_dir} does not exist, skipping...")
            continue

        analysis = analyze_winclip_experiment(result_dir)
        if analysis:
            analyses.append((result_dir, analysis))
            print_analysis(analysis)

    if args.compare and len(analyses) > 1:
        compare_experiments(analyses)


if __name__ == "__main__":
    main()
