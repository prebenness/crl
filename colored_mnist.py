# Run with: python colored_mnist.py config/colored_mnist/vib_pair_sweep.yaml
# Requirements:
#   pip install jax jaxlib flax optax torch torchvision matplotlib pyyaml tqdm

import os
os.environ["WANDB_SILENT"] = "true"

import sys
import time
from pathlib import Path
import json

import numpy as np
import wandb
import pandas as pd

import jax
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from src.config import load_config
from src.utils.checkpointing import (
    TeeLogger,
    save_config,
    make_experiment_dir,
    utc_timestamp,
)
from src.datasets.datasets import (
    build_dataset, dataset_to_jax_arrays, DATASET_NUM_CLASSES,
)
from src.models.ib_classifiers import VIBClassifier, ULAMLPVarClassifier
from src.models.classifiers import StdClassifier, ULAMLPClassifier
from src.models.mdl_classifiers import GumbelSoftmaxMLP
from src.mdl.coding import grid_values_and_codelengths
from src.training.train_state import (
    create_state_inner, create_state_outer, create_state_mdl,
)
from src.training.steps import (
    make_train_step, make_train_step_pair, make_eval_step,
    make_train_step_mdl, make_train_step_mdl_pair, make_eval_step_mdl,
)
from src.training.epochs import (
    make_train_epoch, make_train_epoch_pair, make_eval_epoch,
    make_train_epoch_mdl, make_train_epoch_mdl_pair,
)
from src.training.runners import (
    run_train_eval, run_train_eval_pair,
    run_train_eval_mdl, run_train_eval_mdl_pair,
)
from src.utils.plotting.colored_mnist_plots import wandb_summary_plot


# ---- Model construction (architecture stays as code, not config) ----

def _make_mdl_mlp(cfg):
    """Construct a GumbelSoftmaxMLP from config, caching the rational grid."""
    if not hasattr(_make_mdl_mlp, "_cache"):
        _make_mdl_mlp._cache = {}
    key = (cfg.mdl.n_max, cfg.mdl.m_max)
    if key not in _make_mdl_mlp._cache:
        _make_mdl_mlp._cache[key] = grid_values_and_codelengths(*key)
    gv, gc = _make_mdl_mlp._cache[key]
    return GumbelSoftmaxMLP(
        num_classes=cfg.model.num_classes,
        grid_values=gv,
        grid_codelengths=gc,
    )


INNER_MODELS = {
    "vib_cnn": lambda cfg: VIBClassifier(
        bottleneck_width=cfg.model.bottleneck_width,
        num_classes=cfg.model.num_classes,
    ),
    "ula_mlp_var": lambda cfg: ULAMLPVarClassifier(
        num_classes=cfg.model.num_classes,
    ),
    "mdl_mlp": _make_mdl_mlp,
}

OUTER_MODELS = {
    "std_cnn": lambda cfg: StdClassifier(
        rep_dim=cfg.model.outer_rep_dim,
        num_classes=cfg.model.num_classes,
    ),
    "ula_mlp": lambda cfg: ULAMLPClassifier(
        rep_dim=cfg.model.outer_rep_dim,
        num_classes=cfg.model.num_classes,
    ),
}


def _make_sweep_plot(results, cfg, out_path):
    """Generate and save the matplotlib sweep plot."""
    xs = np.array([r["lambda"] for r in results])

    plt.figure()
    if cfg.model.mode in ("pair", "mdl_pair"):
        plt.scatter(xs, [r["train_acc2"] for r in results],
                    label="Outer train", marker="o")
        plt.scatter(xs, [r["test_acc2"] for r in results],
                    label="Outer test", marker="x")
        plt.scatter(xs, [r["train_acc1"] for r in results],
                    label="Inner train", marker="s")
        plt.scatter(xs, [r["test_acc1"] for r in results],
                    label="Inner test", marker="^")
    else:
        plt.scatter(xs, [r["train_acc"] for r in results],
                    label="Train", marker="o")
        plt.scatter(xs, [r["test_acc"] for r in results],
                    label="Test", marker="x")

    if cfg.sweep.log_sweep:
        plt.xscale("symlog", linthresh=0.01)

    is_mdl = cfg.model.mode.startswith("mdl")
    plt.xlabel("MDL lambda" if is_mdl else "Lambda")
    plt.ylabel("Accuracy")
    label = "MDL penalty" if is_mdl else "Information capacity"
    plt.title(f"{cfg.dataset.name} accuracy vs {label}")
    plt.legend()
    plt.ylim(-0.02, 1.02)

    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.35)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot: {out_path}")


def _print_timing_table(results, dataset_name):
    results = sorted(results, key=lambda r: r["lambda"])
    print(f"\n{dataset_name} results")
    print("Lambda\t\tRun Time\tTrain Acc.\tTest Acc.")
    for r in results:
        print(f"{r['lambda']:.3E}\t\t{r['run_time']:.2f}"
              f"\t{r['train_acc']:.3f}\t\t{r['test_acc']:.3f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python colored_mnist.py <config.yaml>",
              file=sys.stderr)
        sys.exit(1)

    cfg = load_config(sys.argv[1])

    expected_nc = DATASET_NUM_CLASSES.get(cfg.dataset.name)
    if expected_nc is not None and cfg.model.num_classes != expected_nc:
        print(
            f"ERROR: dataset '{cfg.dataset.name}' has {expected_nc} classes "
            f"but config sets model.num_classes={cfg.model.num_classes}. "
            f"Fix the config to set num_classes: {expected_nc}.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("JAX devices:", jax.devices())
    jax.config.update("jax_default_matmul_precision", "high")

    cfg_path = Path(sys.argv[1]).resolve()
    cfg_name = cfg_path.stem
    timestamp = utc_timestamp()
    experiment_group_id = (
        f"{timestamp}-sweep-lambda"
        f"-{cfg.sweep.lambda_min_exp}-{cfg.sweep.lambda_max_exp}"
        f"-{cfg.sweep.lambda_steps}"
    )

    # Build JIT-compiled functions from config
    mode = cfg.model.mode

    if mode in ("single", "pair"):
        train_step = make_train_step(cfg)
        train_step_pair = make_train_step_pair(cfg)
        eval_step = make_eval_step(cfg)

        train_epoch = make_train_epoch(train_step)
        train_epoch_pair = make_train_epoch_pair(train_step_pair)
        eval_epoch = make_eval_epoch(eval_step)

    if mode in ("mdl", "mdl_pair"):
        mdl_step_warmup = make_train_step_mdl(cfg, soft_forward=True)
        mdl_step_bridge = make_train_step_mdl(
            cfg, soft_forward=False, deterministic_st=True,
        )
        mdl_step_train = make_train_step_mdl(cfg, soft_forward=False)
        mdl_eval_step = make_eval_step_mdl(cfg)

        mdl_epoch_warmup = make_train_epoch_mdl(mdl_step_warmup)
        mdl_epoch_bridge = make_train_epoch_mdl(mdl_step_bridge)
        mdl_epoch_train = make_train_epoch_mdl(mdl_step_train)
        mdl_eval_epoch = make_eval_epoch(mdl_eval_step)

    if mode == "mdl_pair":
        mdl_pair_step_warmup = make_train_step_mdl_pair(cfg, soft_forward=True)
        mdl_pair_step_bridge = make_train_step_mdl_pair(
            cfg, soft_forward=False, deterministic_st=True,
        )
        mdl_pair_step_train = make_train_step_mdl_pair(cfg, soft_forward=False)
        vib_eval_step = make_eval_step(cfg)

        mdl_pair_epoch_warmup = make_train_epoch_mdl_pair(mdl_pair_step_warmup)
        mdl_pair_epoch_bridge = make_train_epoch_mdl_pair(mdl_pair_step_bridge)
        mdl_pair_epoch_train = make_train_epoch_mdl_pair(mdl_pair_step_train)
        outer_eval_epoch = make_eval_epoch(vib_eval_step)

    # Load data
    print("Loading datasets and converting to JAX arrays...")
    t0 = time.time()

    train_ds = build_dataset(
        cfg.dataset.name, train=True,
        p_corr=cfg.dataset.p_train, seed=cfg.training.seed,
    )
    test_ds = build_dataset(
        cfg.dataset.name, train=False,
        p_corr=cfg.dataset.p_test, seed=cfg.training.seed + 1,
    )
    x_train, y_train = dataset_to_jax_arrays(train_ds)
    x_test, y_test = dataset_to_jax_arrays(test_ds)

    print("Data shapes")
    print("Train shape, min, max:",
          x_train.shape, x_train.min(), x_train.max())
    print("Test shape, min, max :",
          x_test.shape, x_test.min(), x_test.max())
    print(f"Data ready in {time.time()-t0:.2f}s")

    # Lambda sweep
    all_results = []
    sweep_start = time.time()

    # Create results directory for this sweep
    sweep_name = f"{timestamp}_{cfg_name}_{cfg.dataset.name}_{mode}"
    sweep_dir = make_experiment_dir("colored_mnist", sweep_name)
    runs_dir = sweep_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    save_config(sweep_dir, {
        "experiment": "colored_mnist",
        "sweep_name": sweep_name,
        "config_path": str(cfg_path),
        "mode": mode,
        "dataset": cfg.dataset.name,
        "seed": cfg.training.seed,
        "epochs": cfg.training.epochs,
        "batch_size": cfg.training.batch_size,
        "lr": cfg.training.lr,
        "n_lambdas": int(len(cfg.lambdas)),
        "lambdas": [float(v) for v in cfg.lambdas],
    })

    for lamb in cfg.lambdas:
        lamb = float(lamb)
        inner_model = INNER_MODELS[cfg.model.inner](cfg)

        print(f"\n--- Sweep (lamb={lamb:.1e}) ---")

        # Per-lambda run directory
        run_dir = runs_dir / f"lambda_{lamb:.1e}"
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(run_dir, {
            "experiment": "colored_mnist",
            "sweep_name": sweep_name,
            "mode": mode, "lambda": lamb,
            "dataset": cfg.dataset.name,
            "config_path": str(cfg_path),
            "seed": cfg.training.seed,
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "lr": cfg.training.lr,
        })

        run_config = {
            "lambda": lamb,
            "dataset": cfg.dataset.name,
            "type": "sweep",
        }
        run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            group=experiment_group_id,
            name=f"cmnist-lamb_{lamb:.1e}",
            config=run_config,
            reinit=True,
        )

        t_start = time.time()

        with TeeLogger(run_dir / "train.log"):
            if mode == "single":
                res = run_train_eval(
                    x_train, y_train, x_test, y_test,
                    inner_model, cfg, lamb, wandb_run=run,
                    create_state_fn=create_state_inner,
                    train_epoch_fn=train_epoch,
                    eval_epoch_fn=eval_epoch,
                    run_dir=run_dir,
                )
            elif mode == "pair":
                outer_model = OUTER_MODELS[cfg.model.outer](cfg)
                res = run_train_eval_pair(
                    x_train, y_train, x_test, y_test,
                    inner_model, outer_model, cfg, lamb, wandb_run=run,
                    create_inner_fn=create_state_inner,
                    create_outer_fn=create_state_outer,
                    train_epoch_pair_fn=train_epoch_pair,
                    eval_epoch_fn=eval_epoch,
                    run_dir=run_dir,
                )
            elif mode == "mdl":
                res = run_train_eval_mdl(
                    x_train, y_train, x_test, y_test,
                    inner_model, cfg, lamb, wandb_run=run,
                    create_state_fn=create_state_mdl,
                    train_epoch_warmup_fn=mdl_epoch_warmup,
                    train_epoch_bridge_fn=mdl_epoch_bridge,
                    train_epoch_fn=mdl_epoch_train,
                    eval_epoch_fn=mdl_eval_epoch,
                    run_dir=run_dir,
                )
            elif mode == "mdl_pair":
                outer_model = OUTER_MODELS[cfg.model.outer](cfg)
                res = run_train_eval_mdl_pair(
                    x_train, y_train, x_test, y_test,
                    inner_model, outer_model, cfg, lamb, wandb_run=run,
                    create_inner_fn=create_state_mdl,
                    create_outer_fn=create_state_outer,
                    train_epoch_warmup_fn=mdl_pair_epoch_warmup,
                    train_epoch_bridge_fn=mdl_pair_epoch_bridge,
                    train_epoch_fn=mdl_pair_epoch_train,
                    eval_inner_epoch_fn=mdl_eval_epoch,
                    eval_outer_epoch_fn=outer_eval_epoch,
                    run_dir=run_dir,
                )
            else:
                raise ValueError(f"Unknown model.mode: {cfg.model.mode!r}")

        res["run_time"] = time.time() - t_start
        res["dataset"] = cfg.dataset.name
        res["lambda"] = lamb

        run.summary["final_test_acc"] = res["test_acc"]
        run.summary["final_train_acc"] = res["train_acc"]
        run.finish()

        all_results.append(res)

    total_time = time.time() - sweep_start
    print("\n" + "#" * 70)
    print(f"Total sweep wall time: {total_time/60:.2f} minutes")
    print("#" * 70)

    # Summary W&B run
    summary_run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        group=experiment_group_id,
        name="experiment-summary",
        job_type="summary",
    )
    wandb_summary_plot(all_data=all_results, wandb_run=summary_run)
    summary_run.log({
        "raw_data": wandb.Table(dataframe=pd.DataFrame(all_results)),
    })

    plot_path = sweep_dir / "sweep_plot.png"
    _make_sweep_plot(all_results, cfg, plot_path)

    with open(sweep_dir / "summary.json", "w") as f:
        json.dump({
            "experiment": "colored_mnist",
            "sweep_name": sweep_name,
            "config_path": str(cfg_path),
            "mode": mode,
            "dataset": cfg.dataset.name,
            "n_runs": len(all_results),
            "total_time_sec": total_time,
            "results": all_results,
        }, f, indent=2)
    pd.DataFrame(all_results).to_csv(sweep_dir / "summary.csv", index=False)

    summary_run.finish()

    _print_timing_table(all_results, cfg.dataset.name)


if __name__ == "__main__":
    main()
