#!/usr/bin/env python3
"""CA-WinCLIP Experiment Results Analyzer.

Parses experiment.json files from results/winclip_hdmap_ca and displays
a summary table of all experiments with key metrics.

Usage:
    python analyze_ca_results.py                    # Analyze all results
    python analyze_ca_results.py --domain domain_C  # Filter by domain
    python analyze_ca_results.py --sort cross       # Sort by cross-condition AUROC
    python analyze_ca_results.py --csv results.csv  # Export to CSV
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Default results directory
DEFAULT_RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "results" / "winclip_hdmap_ca"


def load_experiment(exp_dir: Path) -> Optional[Dict]:
    """Load experiment.json from a directory."""
    exp_file = exp_dir / "experiment.json"
    if not exp_file.exists():
        return None

    try:
        with open(exp_file) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def analyze_results(
    results_dir: Path,
    domain_filter: Optional[str] = None,
    sort_by: str = "timestamp",
    show_failed: bool = False,
) -> List[Dict]:
    """Analyze all experiment results in directory.

    Args:
        results_dir: Path to results directory.
        domain_filter: Optional domain to filter by.
        sort_by: Sort key ('timestamp', 'overall', 'cross', 'gating', 'mode').
        show_failed: Include failed experiments.

    Returns:
        List of experiment summaries.
    """
    experiments = []

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        exp_data = load_experiment(exp_dir)
        if exp_data is None:
            continue

        # Skip failed experiments unless requested
        if exp_data.get("status") != "completed" and not show_failed:
            continue

        config = exp_data.get("experiment_config", {})
        results = exp_data.get("results", {})

        # Filter by domain if specified
        domain = config.get("domain", "unknown")
        if domain_filter and domain != domain_filter:
            continue

        # Extract metrics
        auroc = results.get("auroc_metrics", {}) if isinstance(results, dict) else {}

        summary = {
            "timestamp": exp_dir.name,
            "status": exp_data.get("status", "unknown"),
            "domain": domain,
            "mode": config.get("gating_mode", "unknown"),
            "k": config.get("k_per_bank", 0),
            "overall": auroc.get("overall_auroc", 0) * 100,
            "cold": auroc.get("cold_only_auroc", 0) * 100,
            "warm": auroc.get("warm_only_auroc", 0) * 100,
            "cross": auroc.get("cold_fault_vs_warm_good_auroc", 0) * 100,
            "gating": (results.get("gating_accuracy", 0) or 0) * 100,
            "duration": exp_data.get("duration_seconds", 0),
            "error": exp_data.get("error"),
        }
        experiments.append(summary)

    # Sort results
    sort_keys = {
        "timestamp": lambda x: x["timestamp"],
        "overall": lambda x: -x["overall"],
        "cross": lambda x: -x["cross"],
        "gating": lambda x: -x["gating"],
        "mode": lambda x: x["mode"],
    }
    if sort_by in sort_keys:
        experiments.sort(key=sort_keys[sort_by])

    return experiments


def print_table(experiments: List[Dict], show_duration: bool = True) -> None:
    """Print experiments as a formatted table."""
    if not experiments:
        print("No experiments found.")
        return

    # Table header
    print("\n" + "=" * 110)
    print("CA-WinCLIP Experiment Results")
    print("=" * 110)

    if show_duration:
        header = f"{'Timestamp':<18} {'Domain':<10} {'Mode':<10} {'k':>3} {'Overall':>9} {'Cold':>9} {'Warm':>9} {'Cross':>9} {'Gate%':>7} {'Time':>8}"
    else:
        header = f"{'Timestamp':<18} {'Domain':<10} {'Mode':<10} {'k':>3} {'Overall':>9} {'Cold':>9} {'Warm':>9} {'Cross':>9} {'Gate%':>7}"

    print(header)
    print("-" * 110)

    for exp in experiments:
        if exp["status"] == "completed":
            if show_duration:
                row = (
                    f"{exp['timestamp']:<18} "
                    f"{exp['domain']:<10} "
                    f"{exp['mode']:<10} "
                    f"{exp['k']:>3} "
                    f"{exp['overall']:>8.2f}% "
                    f"{exp['cold']:>8.2f}% "
                    f"{exp['warm']:>8.2f}% "
                    f"{exp['cross']:>8.2f}% "
                    f"{exp['gating']:>6.1f}% "
                    f"{format_duration(exp['duration']):>8}"
                )
            else:
                row = (
                    f"{exp['timestamp']:<18} "
                    f"{exp['domain']:<10} "
                    f"{exp['mode']:<10} "
                    f"{exp['k']:>3} "
                    f"{exp['overall']:>8.2f}% "
                    f"{exp['cold']:>8.2f}% "
                    f"{exp['warm']:>8.2f}% "
                    f"{exp['cross']:>8.2f}% "
                    f"{exp['gating']:>6.1f}%"
                )
        else:
            row = (
                f"{exp['timestamp']:<18} "
                f"{exp['domain']:<10} "
                f"{exp['mode']:<10} "
                f"{exp['k']:>3} "
                f"{'FAILED':<50}"
            )
        print(row)

    print("=" * 110)
    print(f"Total: {len(experiments)} experiments")


def print_comparison(experiments: List[Dict]) -> None:
    """Print comparison summary grouped by mode."""
    if not experiments:
        return

    # Group by mode
    by_mode = {}
    for exp in experiments:
        if exp["status"] != "completed":
            continue
        mode = exp["mode"]
        if mode not in by_mode:
            by_mode[mode] = []
        by_mode[mode].append(exp)

    print("\n" + "=" * 80)
    print("Comparison by Gating Mode")
    print("=" * 80)

    # Calculate averages
    summaries = []
    for mode, exps in sorted(by_mode.items()):
        avg_overall = sum(e["overall"] for e in exps) / len(exps)
        avg_cross = sum(e["cross"] for e in exps) / len(exps)
        avg_gating = sum(e["gating"] for e in exps) / len(exps)
        summaries.append({
            "mode": mode,
            "count": len(exps),
            "overall": avg_overall,
            "cross": avg_cross,
            "gating": avg_gating,
        })

    # Sort by cross-condition AUROC
    summaries.sort(key=lambda x: -x["cross"])

    print(f"{'Mode':<12} {'Count':>6} {'Avg Overall':>12} {'Avg Cross':>12} {'Avg Gate%':>12}")
    print("-" * 80)
    for s in summaries:
        print(f"{s['mode']:<12} {s['count']:>6} {s['overall']:>11.2f}% {s['cross']:>11.2f}% {s['gating']:>11.1f}%")
    print("=" * 80)


def export_csv(experiments: List[Dict], output_path: Path) -> None:
    """Export results to CSV file."""
    import csv

    with open(output_path, 'w', newline='') as f:
        fieldnames = ["timestamp", "status", "domain", "mode", "k",
                      "overall", "cold", "warm", "cross", "gating", "duration"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for exp in experiments:
            row = {k: exp.get(k, "") for k in fieldnames}
            writer.writerow(row)

    print(f"\nExported {len(experiments)} experiments to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze CA-WinCLIP experiment results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Path to results directory",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["domain_A", "domain_B", "domain_C", "domain_D"],
        help="Filter by domain",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="timestamp",
        choices=["timestamp", "overall", "cross", "gating", "mode"],
        help="Sort results by field",
    )
    parser.add_argument(
        "--show-failed",
        action="store_true",
        help="Include failed experiments",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Export results to CSV file",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show comparison summary by mode",
    )
    args = parser.parse_args()

    # Analyze results
    experiments = analyze_results(
        args.results_dir,
        domain_filter=args.domain,
        sort_by=args.sort,
        show_failed=args.show_failed,
    )

    # Print table
    print_table(experiments)

    # Print comparison if requested
    if args.compare:
        print_comparison(experiments)

    # Export to CSV if requested
    if args.csv:
        export_csv(experiments, args.csv)


if __name__ == "__main__":
    main()
