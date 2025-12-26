#!/usr/bin/env python3
"""
Experiment Results Analyzer for Dinomaly Experiments.

This script parses and analyzes experiment results from multiple seeds,
computing statistics and generating comparison tables.

Usage:
    python examples/notebooks/analyze_results.py --result-dir results/dinomaly_baseline
    python examples/notebooks/analyze_results.py --result-dir results/dinomaly_gem
    python examples/notebooks/analyze_results.py --result-dir results/dinomaly_topk
    python examples/notebooks/analyze_results.py --result-dir results/dinomaly_topk_ablation
    python examples/notebooks/analyze_results.py --result-dir results/dinomaly_baseline results/dinomaly_gem --compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_experiment_results(result_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all experiment results from a directory.

    Args:
        result_dir: Path to experiment results directory.

    Returns:
        Dictionary mapping seed to results dict.
    """
    results = {}

    for exp_folder in sorted(result_dir.iterdir()):
        if not exp_folder.is_dir():
            continue

        # Extract seed from folder name (e.g., 20251224_091049_seed42 -> 42)
        folder_name = exp_folder.name
        if "_seed" in folder_name:
            seed = folder_name.split("_seed")[-1]
        else:
            seed = folder_name

        # Find results.json
        results_file = exp_folder / "multiclass_unified" / "results.json"
        if not results_file.exists():
            results_file = exp_folder / "results.json"

        if results_file.exists():
            with open(results_file) as f:
                results[seed] = json.load(f)
                results[seed]["_folder"] = str(exp_folder)

    return results


def compute_statistics(values: list[float]) -> dict[str, float]:
    """Compute mean, std, min, max for a list of values."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def format_pct(value: float, width: int = 6) -> str:
    """Format a value as percentage string."""
    return f"{value * 100:>{width}.2f}%"


def format_pct_with_std(mean: float, std: float) -> str:
    """Format mean ± std as percentage string."""
    return f"{mean * 100:.2f}% ± {std * 100:.2f}%"


def analyze_single_experiment(result_dir: Path) -> dict[str, Any]:
    """Analyze a single experiment directory with multiple seeds.

    Args:
        result_dir: Path to experiment results directory.

    Returns:
        Analysis results dictionary.
    """
    results = load_experiment_results(result_dir)

    if not results:
        print(f"No results found in {result_dir}")
        return {}

    # Extract method info from first result
    first_result = next(iter(results.values()))
    method = first_result.get("method", "Baseline")
    max_steps = first_result.get("max_steps", "N/A")
    encoder = first_result.get("encoder_name", "N/A")

    # Collect metrics across seeds
    seeds = sorted(results.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    domains = first_result.get("domains", [])

    # Metric names to analyze
    metric_names = ["auroc", "tpr_at_fpr_1pct", "tpr_at_fpr_5pct",
                    "precision", "recall", "f1_score", "accuracy"]

    # Per-seed summary
    seed_summaries = {}
    for seed in seeds:
        r = results[seed]

        # Handle both old format (per_domain_auroc) and new format (per_domain_metrics)
        if "per_domain_metrics" in r:
            domain_metrics = r["per_domain_metrics"]
        elif "per_domain_auroc" in r:
            # Old format - only has AUROC
            domain_metrics = {d: {"auroc": v} for d, v in r["per_domain_auroc"].items()}
        else:
            domain_metrics = {}

        seed_summaries[seed] = {
            "overall_auroc": r.get("overall_auroc", 0.0),
            "domain_metrics": domain_metrics,
            "mean_metrics": r.get("mean_metrics", {}),
            "overall_pooled_metrics": r.get("overall_pooled_metrics", {}),
        }

    # Compute cross-seed statistics for each metric
    cross_seed_stats = {}
    for metric in metric_names:
        values = []
        for seed in seeds:
            mean_metrics = seed_summaries[seed].get("mean_metrics", {})
            if metric in mean_metrics:
                values.append(mean_metrics[metric])
            elif metric == "auroc" and "mean_domain_auroc" in results[seed]:
                values.append(results[seed]["mean_domain_auroc"])
        cross_seed_stats[metric] = compute_statistics(values)

    # Per-domain cross-seed statistics
    domain_stats = {}
    for domain in domains:
        domain_stats[domain] = {}
        for metric in metric_names:
            values = []
            for seed in seeds:
                dm = seed_summaries[seed].get("domain_metrics", {}).get(domain, {})
                if metric in dm:
                    values.append(dm[metric])
            if values:
                domain_stats[domain][metric] = compute_statistics(values)

    return {
        "method": method,
        "max_steps": max_steps,
        "encoder": encoder,
        "num_seeds": len(seeds),
        "seeds": seeds,
        "domains": domains,
        "seed_summaries": seed_summaries,
        "cross_seed_stats": cross_seed_stats,
        "domain_stats": domain_stats,
    }


def print_analysis(analysis: dict[str, Any], result_dir: Path) -> None:
    """Print analysis results in a formatted way."""
    if not analysis:
        return

    print("\n" + "=" * 80)
    print(f"EXPERIMENT ANALYSIS: {result_dir.name}")
    print("=" * 80)

    print(f"\nMethod: {analysis['method']}")
    print(f"Max Steps: {analysis['max_steps']}")
    print(f"Encoder: {analysis['encoder']}")
    print(f"Seeds: {', '.join(analysis['seeds'])} (n={analysis['num_seeds']})")

    # Per-seed results table
    print("\n" + "-" * 80)
    print("PER-SEED RESULTS (Per-Domain Mean)")
    print("-" * 80)

    header = f"{'Seed':<8} {'AUROC':>10} {'TPR@1%':>10} {'TPR@5%':>10} {'Prec':>10} {'Recall':>10} {'F1':>10} {'Acc':>10}"
    print(header)
    print("-" * len(header))

    for seed in analysis["seeds"]:
        summary = analysis["seed_summaries"][seed]
        mm = summary.get("mean_metrics", {})

        auroc = mm.get("auroc", summary.get("overall_auroc", 0) or 0)
        tpr1 = mm.get("tpr_at_fpr_1pct", 0)
        tpr5 = mm.get("tpr_at_fpr_5pct", 0)
        prec = mm.get("precision", 0)
        recall = mm.get("recall", 0)
        f1 = mm.get("f1_score", 0)
        acc = mm.get("accuracy", 0)

        print(f"{seed:<8} {format_pct(auroc, 8)} {format_pct(tpr1, 8)} {format_pct(tpr5, 8)} "
              f"{format_pct(prec, 8)} {format_pct(recall, 8)} {format_pct(f1, 8)} {format_pct(acc, 8)}")

    # Cross-seed statistics
    print("\n" + "-" * 80)
    print("CROSS-SEED STATISTICS (Mean ± Std)")
    print("-" * 80)

    stats = analysis["cross_seed_stats"]
    metrics_display = [
        ("AUROC", "auroc"),
        ("TPR@FPR=1%", "tpr_at_fpr_1pct"),
        ("TPR@FPR=5%", "tpr_at_fpr_5pct"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1_score"),
        ("Accuracy", "accuracy"),
    ]

    for display_name, metric_key in metrics_display:
        if metric_key in stats and stats[metric_key]["mean"] > 0:
            s = stats[metric_key]
            print(f"{display_name:<15}: {format_pct_with_std(s['mean'], s['std']):<20} "
                  f"(min: {format_pct(s['min'], 5)}, max: {format_pct(s['max'], 5)})")

    # Per-domain breakdown
    print("\n" + "-" * 80)
    print("PER-DOMAIN STATISTICS (Mean ± Std across seeds)")
    print("-" * 80)

    domain_stats = analysis["domain_stats"]
    if domain_stats:
        header = f"{'Domain':<12} {'AUROC':>18} {'TPR@1%':>18} {'TPR@5%':>18}"
        print(header)
        print("-" * len(header))

        for domain in analysis["domains"]:
            if domain in domain_stats:
                ds = domain_stats[domain]
                auroc_str = format_pct_with_std(ds.get("auroc", {}).get("mean", 0),
                                                 ds.get("auroc", {}).get("std", 0))
                tpr1_str = format_pct_with_std(ds.get("tpr_at_fpr_1pct", {}).get("mean", 0),
                                                ds.get("tpr_at_fpr_1pct", {}).get("std", 0))
                tpr5_str = format_pct_with_std(ds.get("tpr_at_fpr_5pct", {}).get("mean", 0),
                                                ds.get("tpr_at_fpr_5pct", {}).get("std", 0))
                print(f"{domain:<12} {auroc_str:>18} {tpr1_str:>18} {tpr5_str:>18}")


def compare_experiments(analyses: list[tuple[Path, dict[str, Any]]]) -> None:
    """Compare multiple experiments side by side."""
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPARISON")
    print("=" * 80)

    # Header
    print(f"\n{'Metric':<20}", end="")
    for result_dir, analysis in analyses:
        method = analysis.get("method", result_dir.name)
        print(f"{method:>25}", end="")
    print()
    print("-" * (20 + 25 * len(analyses)))

    # Compare metrics
    metrics_display = [
        ("AUROC", "auroc"),
        ("TPR@FPR=1%", "tpr_at_fpr_1pct"),
        ("TPR@FPR=5%", "tpr_at_fpr_5pct"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1_score"),
        ("Accuracy", "accuracy"),
    ]

    for display_name, metric_key in metrics_display:
        print(f"{display_name:<20}", end="")
        for _, analysis in analyses:
            stats = analysis.get("cross_seed_stats", {}).get(metric_key, {})
            if stats and stats.get("mean", 0) > 0:
                value_str = format_pct_with_std(stats["mean"], stats["std"])
            else:
                value_str = "N/A"
            print(f"{value_str:>25}", end="")
        print()

    # Per-domain comparison for AUROC
    print("\n" + "-" * 80)
    print("PER-DOMAIN AUROC COMPARISON")
    print("-" * 80)

    # Get all domains
    all_domains = set()
    for _, analysis in analyses:
        all_domains.update(analysis.get("domains", []))

    print(f"{'Domain':<12}", end="")
    for result_dir, analysis in analyses:
        method = analysis.get("method", result_dir.name)
        print(f"{method:>25}", end="")
    print()
    print("-" * (12 + 25 * len(analyses)))

    for domain in sorted(all_domains):
        print(f"{domain:<12}", end="")
        for _, analysis in analyses:
            ds = analysis.get("domain_stats", {}).get(domain, {}).get("auroc", {})
            if ds and ds.get("mean", 0) > 0:
                value_str = format_pct_with_std(ds["mean"], ds["std"])
            else:
                value_str = "N/A"
            print(f"{value_str:>25}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Dinomaly experiment results",
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

        analysis = analyze_single_experiment(result_dir)
        if analysis:
            analyses.append((result_dir, analysis))
            print_analysis(analysis, result_dir)

    if args.compare and len(analyses) > 1:
        compare_experiments(analyses)


if __name__ == "__main__":
    main()
