"""Analyze experiment results across runs.

Reads results.json files from results/ directories and computes
summary statistics (mean, std, min, max) for key metrics.

Usage:
    python3.12 scripts/analysis/summarize_experiment_results.py                 # all results
    python3.12 scripts/analysis/summarize_experiment_results.py --tag sweep     # only dirs containing "sweep"
    python3.12 scripts/analysis/summarize_experiment_results.py --tag baseline  # only baseline runs
    python3.12 scripts/analysis/summarize_experiment_results.py --mode basic    # only basic mode runs
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_results(results_dir: Path, tag: str = None, mode: str = None):
    """Load all results.json files, optionally filtering by tag/mode."""
    runs = []
    for results_path in sorted(results_dir.rglob("results.json")):
        run_dir = results_path.parent
        config_path = run_dir / "config.json"

        # Tag filter: check if tag appears in directory name
        run_rel = run_dir.relative_to(results_dir).as_posix()
        if tag and tag not in run_rel:
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
            "dir": run_rel,
            "results": results,
            "config": config,
        })
    return runs


def summarize_runs(runs):
    """Compute summary statistics across runs."""
    if not runs:
        return None

    def metric(values):
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
            "values": values,
        }

    summary = {"n_runs": len(runs)}
    key_map = {
        "gen_n": "gen_n",
        "mdl_bits": "total_mdl_bits",
        "weight_bits": "weight_bits",
        "mean_det_accuracy": "mean_det_accuracy",
        "train_acc": "train_acc",
        "test_acc": "test_acc",
    }
    for summary_key, result_key in key_map.items():
        vals = [r["results"][result_key] for r in runs if result_key in r["results"]]
        if vals:
            summary[summary_key] = metric(vals)

    return summary


def print_run_table(runs):
    """Print a table of individual run results."""
    has_mdl_cols = any("gen_n" in r["results"] for r in runs)
    if has_mdl_cols:
        print(f"\n{'Run directory':<65} {'Seed':>4} {'gen_n':>8} {'|H|':>6} {'fail@':>7}")
        print("-" * 95)
        for r in runs:
            seed = r["config"].get("seed", "?")
            res = r["results"]
            fail = res.get("first_failure_n")
            fail_str = str(fail) if fail else "none"
            gen_n = res.get("gen_n", "")
            mdl = res.get("total_mdl_bits", "")
            print(f"{r['dir']:<65} {seed:>4} {gen_n:>8} {mdl:>6} {fail_str:>7}")
        return

    print(f"\n{'Run directory':<65} {'Seed':>4} {'Train':>8} {'Test':>8}")
    print("-" * 95)
    for r in runs:
        seed = r["config"].get("seed", "?")
        res = r["results"]
        tr = res.get("train_acc", "")
        te = res.get("test_acc", "")
        tr_str = f"{tr:.4f}" if isinstance(tr, (int, float)) else str(tr)
        te_str = f"{te:.4f}" if isinstance(te, (int, float)) else str(te)
        print(f"{r['dir']:<65} {seed:>4} {tr_str:>8} {te_str:>8}")


def print_summary(summary):
    """Print summary statistics."""
    print(f"\n{'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Median':>10}")
    print("-" * 80)
    ordered = ["gen_n", "mdl_bits", "weight_bits", "train_acc", "test_acc", "mean_det_accuracy"]
    for key in ordered:
        if key not in summary:
            continue
        s = summary[key]
        med = s.get("median", "")
        is_acc = "acc" in key
        med_str = f"{med:>10.4f}" if med != "" else f"{'':>10}"
        minmax_fmt = "{:>10.4f}" if is_acc else "{:>10.0f}"
        print(
            f"{key:<25} {s['mean']:>10.4f} {s['std']:>10.4f} "
            f"{minmax_fmt.format(s['min'])} {minmax_fmt.format(s['max'])} {med_str}"
        )


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
