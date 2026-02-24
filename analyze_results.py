"""Analyze experiment results across runs.

Reads results.json files from results/ directories and computes
summary statistics (mean, std, min, max) for key metrics.

Usage:
    python3.12 analyze_results.py                    # all results
    python3.12 analyze_results.py --tag sweep         # only dirs containing "sweep"
    python3.12 analyze_results.py --tag baseline      # only baseline runs
    python3.12 analyze_results.py --mode basic        # only basic mode runs
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_results(results_dir: Path, tag: str = None, mode: str = None):
    """Load all results.json files, optionally filtering by tag/mode."""
    runs = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        results_path = run_dir / "results.json"
        config_path = run_dir / "config.json"
        if not results_path.exists():
            continue

        # Tag filter: check if tag appears in directory name
        if tag and tag not in run_dir.name:
            continue

        with open(results_path) as f:
            results = json.load(f)
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        # Mode filter
        if mode and results.get("mode", config.get("mode")) != mode:
            continue

        runs.append({
            "dir": run_dir.name,
            "results": results,
            "config": config,
        })
    return runs


def summarize_runs(runs):
    """Compute summary statistics across runs."""
    if not runs:
        return None

    gen_ns = [r["results"]["gen_n"] for r in runs]
    mdl_bits = [r["results"]["total_mdl_bits"] for r in runs]
    weight_bits = [r["results"]["weight_bits"] for r in runs]
    mean_accs = [r["results"].get("mean_det_accuracy", 0) for r in runs]

    return {
        "n_runs": len(runs),
        "gen_n": {
            "mean": np.mean(gen_ns),
            "std": np.std(gen_ns),
            "min": np.min(gen_ns),
            "max": np.max(gen_ns),
            "median": np.median(gen_ns),
            "values": gen_ns,
        },
        "mdl_bits": {
            "mean": np.mean(mdl_bits),
            "std": np.std(mdl_bits),
            "min": np.min(mdl_bits),
            "max": np.max(mdl_bits),
            "values": mdl_bits,
        },
        "weight_bits": {
            "mean": np.mean(weight_bits),
            "std": np.std(weight_bits),
            "min": np.min(weight_bits),
            "max": np.max(weight_bits),
            "values": weight_bits,
        },
        "mean_det_accuracy": {
            "mean": np.mean(mean_accs),
            "std": np.std(mean_accs),
            "values": mean_accs,
        },
    }


def print_run_table(runs):
    """Print a table of individual run results."""
    print(f"\n{'Run directory':<65} {'Seed':>4} {'gen_n':>8} {'|H|':>6} {'fail@':>7}")
    print("-" * 95)
    for r in runs:
        seed = r["config"].get("seed", "?")
        res = r["results"]
        fail = res.get("first_failure_n")
        fail_str = str(fail) if fail else "none"
        print(f"{r['dir']:<65} {seed:>4} {res['gen_n']:>8} {res['total_mdl_bits']:>6} {fail_str:>7}")


def print_summary(summary):
    """Print summary statistics."""
    print(f"\n{'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Median':>10}")
    print("-" * 80)
    for key in ["gen_n", "mdl_bits", "weight_bits"]:
        s = summary[key]
        med = s.get("median", "")
        med_str = f"{med:>10.1f}" if med != "" else f"{'':>10}"
        print(f"{key:<25} {s['mean']:>10.1f} {s['std']:>10.1f} "
              f"{s['min']:>10.0f} {s['max']:>10.0f} {med_str}")

    s = summary["mean_det_accuracy"]
    print(f"{'mean_det_accuracy':<25} {s['mean']:>10.4f} {s['std']:>10.4f} "
          f"{'':>10} {'':>10} {'':>10}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Path to results directory")
    parser.add_argument("--tag", type=str, default=None,
                        help="Filter runs by substring in directory name")
    parser.add_argument("--mode", type=str, default=None,
                        help="Filter by mode (basic/shared/baseline_*)")
    parser.add_argument("--json", action="store_true",
                        help="Output summary as JSON")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs = load_results(results_dir, tag=args.tag, mode=args.mode)

    if not runs:
        print(f"No results found in {results_dir}"
              + (f" matching tag='{args.tag}'" if args.tag else "")
              + (f" mode='{args.mode}'" if args.mode else ""))
        return

    print(f"Found {len(runs)} runs"
          + (f" matching tag='{args.tag}'" if args.tag else "")
          + (f" mode='{args.mode}'" if args.mode else ""))

    print_run_table(runs)

    summary = summarize_runs(runs)
    if args.json:
        # Convert numpy types for JSON serialization
        clean = {}
        for k, v in summary.items():
            if isinstance(v, dict):
                clean[k] = {
                    sk: (sv.tolist() if hasattr(sv, "tolist") else sv)
                    for sk, sv in v.items()
                }
            else:
                clean[k] = v
        print(json.dumps(clean, indent=2))
    else:
        print_summary(summary)


if __name__ == "__main__":
    main()
