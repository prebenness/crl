# Run with: python colored_mnist.py config/colored_mnist/vib_pair_sweep.yaml
# Requirements:
#   pip install jax jaxlib flax optax torch torchvision matplotlib pyyaml tqdm

import argparse
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

from src.config import load_config, apply_overrides
from src.utils.checkpointing import (
    TeeLogger,
    save_config,
    make_experiment_dir,
    utc_timestamp,
    load_checkpoint,
    resolve_resume_checkpoint,
    resolve_resume_start_epoch,
)
from src.datasets.datasets import (
    build_dataset, dataset_to_jax_arrays, DATASET_NUM_CLASSES,
)
from src.models.ib_classifiers import VIBClassifier, ULAMLPVarClassifier
from src.models.classifiers import StdClassifier, ULAMLPClassifier, OracleMLP, CBAOMMlp
from src.models.ipsn_classifier import IPSNMlp
from src.models.mdl_classifiers import GumbelSoftmaxMLP
from mdl.src.mdl.coding import grid_values_and_codelengths
from src.training.train_state import (
    create_state_inner, create_state_outer, create_state_mdl,
    create_state_mdl_shared, create_state_oracle, create_state_cba_om,
    create_state_ipsn,
)
from src.training.steps import (
    make_train_step, make_train_step_pair, make_eval_step,
    make_train_step_mdl, make_train_step_mdl_pair, make_eval_step_mdl,
    make_train_step_mdl_shared, make_train_step_mdl_shared_pair,
    make_eval_step_mdl_shared,
    make_train_step_oracle, make_eval_step_oracle,
    make_train_step_oracle_pair,
    make_train_step_cba_om, make_eval_step_cba_om,
    make_train_step_ipsn, make_eval_step_ipsn,
    make_train_step_pair_mmd, make_train_step_oracle_pair_mmd,
    make_train_step_mdl_pair_mmd, make_train_step_mdl_shared_pair_mmd,
)
from src.training.epochs import (
    make_train_epoch, make_train_epoch_pair, make_eval_epoch,
    make_train_epoch_mdl, make_train_epoch_mdl_pair,
    make_train_epoch_cba_om,
    make_train_epoch_ipsn,
    make_train_epoch_pair_mmd, make_train_epoch_mdl_pair_mmd,
)
from src.training.runners import (
    run_train_eval, run_train_eval_pair,
    run_train_eval_mdl, run_train_eval_mdl_pair,
    run_train_eval_oracle_pair,
    run_train_eval_cba_om,
    run_train_eval_ipsn,
    _forward_vib, _forward_oracle, _forward_mdl, _forward_mdl_shared,
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
        mode_forward=cfg.mdl.mode_forward,
        init_cl_scale=cfg.mdl.init_cl_scale,
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
    "oracle_mlp": lambda cfg: OracleMLP(
        num_classes=cfg.model.num_classes,
    ),
    "ula_mlp": lambda cfg: ULAMLPClassifier(
        rep_dim=cfg.model.outer_rep_dim,
        num_classes=cfg.model.num_classes,
    ),
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
    "cba_om_mlp": lambda cfg: CBAOMMlp(
        num_classes=cfg.model.num_classes,
        num_colors=cfg.cba_om.num_colors,
        embed_dim=cfg.cba_om.embed_dim,
    ),
    "ipsn_mlp": lambda cfg: IPSNMlp(
        num_classes=cfg.model.num_classes,
        num_colors=cfg.ipsn.num_colors,
        c_dim=cfg.ipsn.c_dim,
        b_dim=cfg.ipsn.b_dim,
        embed_dim=cfg.ipsn.embed_dim,
        decoder_hidden=cfg.ipsn.decoder_hidden,
        grad_rev_scale=cfg.ipsn.grad_rev_scale,
    ),
}


def parse_args(argv=None):
    """Parse CLI arguments and optional dataloader overrides.

    Any trailing ``section.key=value`` arguments override the YAML config.
    Example:
        python colored_mnist.py config.yaml training.epochs=20 hsic.weight=0.3
    """
    parser = argparse.ArgumentParser(
        description="Run the colored-MNIST lambda sweep from a YAML config.",
    )
    parser.add_argument("config", help="Path to the experiment YAML config.")
    parser.add_argument(
        "overrides", nargs="*",
        help="Config overrides in section.key=value format.",
    )
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=None,
        help="Override dataloader.num_workers for PyTorch dataset loading.",
    )
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=None,
        help="Override dataloader.prefetch_factor (used only when workers > 0).",
    )
    parser.add_argument(
        "--dataloader-pin-memory",
        dest="dataloader_pin_memory",
        action="store_true",
        help="Override dataloader.pin_memory=True.",
    )
    parser.add_argument(
        "--no-dataloader-pin-memory",
        dest="dataloader_pin_memory",
        action="store_false",
        help="Override dataloader.pin_memory=False.",
    )
    parser.add_argument(
        "--dataloader-persistent-workers",
        dest="dataloader_persistent_workers",
        action="store_true",
        help="Override dataloader.persistent_workers=True.",
    )
    parser.add_argument(
        "--no-dataloader-persistent-workers",
        dest="dataloader_persistent_workers",
        action="store_false",
        help="Override dataloader.persistent_workers=False.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a run_dir to resume training from.",
    )
    parser.set_defaults(
        dataloader_pin_memory=None,
        dataloader_persistent_workers=None,
    )
    return parser.parse_args(argv)


def _apply_cli_overrides(cfg, args):
    """Apply optional CLI overrides after YAML is loaded."""
    if args.dataloader_workers is not None:
        cfg.dataloader.num_workers = args.dataloader_workers
    if args.dataloader_prefetch_factor is not None:
        cfg.dataloader.prefetch_factor = args.dataloader_prefetch_factor
    if args.dataloader_pin_memory is not None:
        cfg.dataloader.pin_memory = args.dataloader_pin_memory
    if args.dataloader_persistent_workers is not None:
        cfg.dataloader.persistent_workers = args.dataloader_persistent_workers


def _dataloader_settings(cfg):
    """Return the active PyTorch dataloader settings as a plain dict."""
    return {
        "num_workers": int(cfg.dataloader.num_workers),
        "pin_memory": bool(cfg.dataloader.pin_memory),
        "persistent_workers": bool(cfg.dataloader.persistent_workers),
        "prefetch_factor": int(cfg.dataloader.prefetch_factor),
    }


def _make_sweep_plot(results, cfg, out_path):
    """Generate and save the matplotlib sweep plot."""
    xs = np.array([r["lambda"] for r in results])

    plt.figure()
    if cfg.model.mode.endswith("pair"):
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
    print("Lambda\t\tRun Time\tTrain Acc.\tBest Test Acc.\tBest Epoch")
    for r in results:
        print(f"{r['lambda']:.3E}\t\t{r['run_time']:.2f}"
              f"\t{r['train_acc']:.3f}\t\t{r['test_acc']:.3f}"
              f"\t\t{r.get('best_epoch', '-')}")


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.overrides:
        apply_overrides(cfg, args.overrides)
    _apply_cli_overrides(cfg, args)

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

    cfg_path = Path(args.config).resolve()
    cfg_name = cfg_path.stem
    timestamp = utc_timestamp()
    experiment_group_id = (
        f"{timestamp}-sweep-lambda"
        f"-{cfg.sweep.lambda_min_exp}-{cfg.sweep.lambda_max_exp}"
        f"-{cfg.sweep.lambda_steps}"
    )
    dataloader_settings = _dataloader_settings(cfg)

    # Build JIT-compiled functions from config
    mode = cfg.model.mode

    use_mmd = cfg.model.outer_loss == "mmd"

    if mode in ("single", "pair"):
        train_step = make_train_step(cfg)
        train_step_pair = make_train_step_pair(cfg)
        eval_step = make_eval_step(cfg)

        train_epoch = make_train_epoch(train_step)
        train_epoch_pair = make_train_epoch_pair(train_step_pair)
        eval_epoch = make_eval_epoch(eval_step)

        if mode == "pair" and use_mmd:
            mmd_step_pair = make_train_step_pair_mmd(cfg)
            mmd_epoch_pair = make_train_epoch_pair_mmd(mmd_step_pair)

    if mode == "cba_om":
        cba_om_train_step = make_train_step_cba_om(cfg)
        cba_om_eval_step = make_eval_step_cba_om(cfg)

        cba_om_train_epoch = make_train_epoch_cba_om(cba_om_train_step)
        cba_om_eval_epoch = make_eval_epoch(cba_om_eval_step)

    if mode == "ipsn":
        ipsn_train_step = make_train_step_ipsn(cfg)
        ipsn_eval_step = make_eval_step_ipsn(cfg)

        ipsn_train_epoch = make_train_epoch_ipsn(ipsn_train_step)
        ipsn_eval_epoch = make_eval_epoch(ipsn_eval_step)

    if mode in ("oracle_train", "erm"):
        oracle_train_step = make_train_step_oracle(cfg)
        oracle_eval_step = make_eval_step_oracle(cfg)

        oracle_train_epoch = make_train_epoch(oracle_train_step)
        oracle_eval_epoch = make_eval_epoch(oracle_eval_step)

    if mode == "oracle_pair":
        oracle_pair_step = make_train_step_oracle_pair(cfg)
        oracle_eval_step = make_eval_step_oracle(cfg)
        outer_eval_step_op = make_eval_step_oracle(cfg)

        oracle_pair_epoch = make_train_epoch_pair(oracle_pair_step)
        oracle_inner_eval_epoch = make_eval_epoch(oracle_eval_step)
        oracle_outer_eval_epoch = make_eval_epoch(outer_eval_step_op)

        if use_mmd:
            mmd_oracle_pair_step = make_train_step_oracle_pair_mmd(cfg)
            mmd_oracle_pair_epoch = make_train_epoch_pair_mmd(
                mmd_oracle_pair_step,
            )

    if mode in ("mdl", "mdl_pair"):
        mdl_step_train = make_train_step_mdl(cfg)
        mdl_eval_step = make_eval_step_mdl(cfg)

        mdl_epoch_train = make_train_epoch_mdl(mdl_step_train)
        mdl_eval_epoch = make_eval_epoch(mdl_eval_step)

    if mode in ("mdl_shared", "mdl_shared_pair"):
        mdl_shared_step_train = make_train_step_mdl_shared(cfg)
        mdl_shared_eval_step = make_eval_step_mdl_shared(cfg)

        mdl_shared_epoch_train = make_train_epoch_mdl(mdl_shared_step_train)
        mdl_shared_eval_epoch = make_eval_epoch(mdl_shared_eval_step)

    if mode == "mdl_pair":
        mdl_pair_step_train = make_train_step_mdl_pair(cfg)
        vib_eval_step = make_eval_step(cfg)

        mdl_pair_epoch_train = make_train_epoch_mdl_pair(mdl_pair_step_train)
        outer_eval_epoch = make_eval_epoch(vib_eval_step)

        if use_mmd:
            mmd_mdl_pair_step = make_train_step_mdl_pair_mmd(cfg)
            mmd_mdl_pair_epoch = make_train_epoch_mdl_pair_mmd(
                mmd_mdl_pair_step,
            )

    if mode == "mdl_shared_pair":
        mdl_shared_pair_step_train = make_train_step_mdl_shared_pair(cfg)
        vib_eval_step = make_eval_step(cfg)

        mdl_shared_pair_epoch_train = make_train_epoch_mdl_pair(
            mdl_shared_pair_step_train,
        )

        if use_mmd:
            mmd_mdl_shared_pair_step = make_train_step_mdl_shared_pair_mmd(cfg)
            mmd_mdl_shared_pair_epoch = make_train_epoch_mdl_pair_mmd(
                mmd_mdl_shared_pair_step,
            )
        outer_eval_epoch = make_eval_epoch(vib_eval_step)

    # Load data
    print("Loading datasets and converting to JAX arrays...")
    print("PyTorch DataLoader settings:", dataloader_settings)
    t0 = time.time()

    train_ds = build_dataset(
        cfg.dataset.name, train=True,
        p_corr=cfg.dataset.p_train, seed=cfg.training.seed,
    )
    test_ds = build_dataset(
        cfg.dataset.name, train=False,
        p_corr=cfg.dataset.p_test, seed=cfg.training.seed + 1,
    )
    x_train, y_train = dataset_to_jax_arrays(
        train_ds,
        batch_size=cfg.training.batch_size,
        **dataloader_settings,
    )
    x_test, y_test = dataset_to_jax_arrays(
        test_ds,
        batch_size=cfg.training.batch_size,
        **dataloader_settings,
    )

    print("Data shapes")
    print("Train shape, min, max:",
          x_train.shape, x_train.min(), x_train.max())
    print("Test shape, min, max :",
          x_test.shape, x_test.min(), x_test.max())
    print(f"Data ready in {time.time()-t0:.2f}s")

    # Extract color labels for oracle training
    if mode == "oracle_train":
        import jax.numpy as jnp
        y_train = jnp.array(train_ds.colors, dtype=jnp.int32)
        y_test = jnp.array(test_ds.colors, dtype=jnp.int32)
        print(f"Oracle mode: using color labels (train unique: "
              f"{len(set(train_ds.colors.tolist()))}, "
              f"test unique: {len(set(test_ds.colors.tolist()))})")

    # Extract colour labels for CBA-OM outer model
    if mode == "cba_om":
        import jax.numpy as jnp
        oracle_ckpt = cfg.model.oracle_checkpoint
        if oracle_ckpt:
            # Phase 2: use trained oracle model to predict colours
            oracle_model = INNER_MODELS["oracle_mlp"](cfg)
            oracle_params = load_checkpoint(oracle_ckpt)
            print(f"  CBA-OM: loaded oracle checkpoint: {oracle_ckpt}")
            oracle_state = create_state_oracle(
                jrandom.PRNGKey(0), oracle_model,
                (cfg.training.batch_size,) + x_train.shape[1:], cfg,
            )
            oracle_state = oracle_state.replace(params=oracle_params)
            from src.training.runners import precompute_predictions
            s_train = precompute_predictions(
                oracle_state, x_train, cfg.training.batch_size,
                lambda state, x: state.apply_fn(
                    {"params": state.params}, x, train=False,
                )[0],
            )
            oracle_acc = float((s_train == jnp.array(
                train_ds.colors, dtype=jnp.int32)).mean())
            print(f"  CBA-OM: oracle colour accuracy on train: {oracle_acc:.4f}")
        else:
            # Phase 1: use ground-truth colour labels
            s_train = jnp.array(train_ds.colors, dtype=jnp.int32)
            print(f"CBA-OM mode: using ground-truth colour labels as oracle "
                  f"(train unique: {len(set(train_ds.colors.tolist()))})")

    # Extract colour labels for IPSN (always as soft distributions)
    if mode == "ipsn":
        import jax.numpy as jnp
        num_colors = cfg.ipsn.num_colors
        oracle_ckpt = cfg.model.oracle_checkpoint
        if oracle_ckpt:
            oracle_model = INNER_MODELS["oracle_mlp"](cfg)
            oracle_params = load_checkpoint(oracle_ckpt)
            print(f"  IPSN: loaded oracle checkpoint: {oracle_ckpt}")
            oracle_state = create_state_oracle(
                jrandom.PRNGKey(0), oracle_model,
                (cfg.training.batch_size,) + x_train.shape[1:], cfg,
            )
            oracle_state = oracle_state.replace(params=oracle_params)
            # Hard labels via argmax — convert to one-hot
            from src.training.runners import precompute_predictions
            s_hard = precompute_predictions(
                oracle_state, x_train, cfg.training.batch_size,
                lambda state, x: state.apply_fn(
                    {"params": state.params}, x, train=False,
                )[0],
            )
            s_train = jax.nn.one_hot(s_hard, num_colors)
            oracle_acc = float((s_hard == jnp.array(
                train_ds.colors, dtype=jnp.int32)).mean())
            print(f"  IPSN: oracle colour accuracy on train: {oracle_acc:.4f}")
        else:
            # Ground truth — one-hot encode
            s_hard = jnp.array(train_ds.colors, dtype=jnp.int32)
            s_train = jax.nn.one_hot(s_hard, num_colors)
            print(f"IPSN mode: using ground-truth colour labels (one-hot) "
                  f"(train unique: {len(set(train_ds.colors.tolist()))})")

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
        "dataloader": dataloader_settings,
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
            "dataloader": dataloader_settings,
        })

        # Resume handling
        resume_source = args.resume or cfg.checkpointing.resume_from
        if resume_source:
            ckpt_path, ckpt_kind = resolve_resume_checkpoint(
                resume_source, cfg.checkpointing.ckpt_select,
            )
            resume_params = load_checkpoint(ckpt_path)
            resume_start_epoch = resolve_resume_start_epoch(
                resume_source, ckpt_kind,
                default_final_epoch=cfg.training.epochs,
            )
            print(f"  Resuming from {ckpt_path} ({ckpt_kind}, "
                  f"epoch {resume_start_epoch})")
        else:
            resume_params = None
            resume_start_epoch = 0

        run_config = {
            "lambda": lamb,
            "dataset": cfg.dataset.name,
            "type": "sweep",
            "dataloader": dataloader_settings,
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
        print(f"Detailed training log: {run_dir / 'train.log'}")

        with TeeLogger(run_dir / "train.log"):
            if mode == "single":
                res = run_train_eval(
                    x_train, y_train, x_test, y_test,
                    inner_model, cfg, lamb, wandb_run=run,
                    create_state_fn=create_state_inner,
                    train_epoch_fn=train_epoch,
                    eval_epoch_fn=eval_epoch,
                    run_dir=run_dir,
                    start_epoch=resume_start_epoch,
                    init_params=resume_params,
                )
            elif mode == "pair":
                outer_model = OUTER_MODELS[cfg.model.outer](cfg)
                mmd_kw = {}
                if use_mmd:
                    mmd_kw = {
                        "inner_forward_fn": _forward_vib,
                        "train_epoch_pair_mmd_fn": mmd_epoch_pair,
                    }
                res = run_train_eval_pair(
                    x_train, y_train, x_test, y_test,
                    inner_model, outer_model, cfg, lamb, wandb_run=run,
                    create_inner_fn=create_state_inner,
                    create_outer_fn=create_state_outer,
                    train_epoch_pair_fn=train_epoch_pair,
                    eval_epoch_fn=eval_epoch,
                    run_dir=run_dir,
                    start_epoch=resume_start_epoch,
                    init_inner_params=resume_params,
                    **mmd_kw,
                )
            elif mode == "mdl":
                res = run_train_eval_mdl(
                    x_train, y_train, x_test, y_test,
                    inner_model, cfg, lamb, wandb_run=run,
                    create_state_fn=create_state_mdl,
                    train_epoch_fn=mdl_epoch_train,
                    eval_epoch_fn=mdl_eval_epoch,
                    run_dir=run_dir,
                    start_epoch=resume_start_epoch,
                    init_params=resume_params,
                )
            elif mode == "mdl_pair":
                outer_model = OUTER_MODELS[cfg.model.outer](cfg)
                mmd_kw = {}
                if use_mmd:
                    mmd_kw = {
                        "inner_forward_fn": _forward_mdl,
                        "train_epoch_mmd_fn": mmd_mdl_pair_epoch,
                    }
                res = run_train_eval_mdl_pair(
                    x_train, y_train, x_test, y_test,
                    inner_model, outer_model, cfg, lamb, wandb_run=run,
                    create_inner_fn=create_state_mdl,
                    create_outer_fn=create_state_outer,
                    train_epoch_fn=mdl_pair_epoch_train,
                    eval_inner_epoch_fn=mdl_eval_epoch,
                    eval_outer_epoch_fn=outer_eval_epoch,
                    run_dir=run_dir,
                    start_epoch=resume_start_epoch,
                    init_inner_params=resume_params,
                    **mmd_kw,
                )
            elif mode == "mdl_shared":
                res = run_train_eval_mdl(
                    x_train, y_train, x_test, y_test,
                    inner_model, cfg, lamb, wandb_run=run,
                    create_state_fn=create_state_mdl_shared,
                    train_epoch_fn=mdl_shared_epoch_train,
                    eval_epoch_fn=mdl_shared_eval_epoch,
                    run_dir=run_dir,
                    start_epoch=resume_start_epoch,
                    init_params=resume_params,
                )
            elif mode == "mdl_shared_pair":
                outer_model = OUTER_MODELS[cfg.model.outer](cfg)
                mmd_kw = {}
                if use_mmd:
                    mmd_kw = {
                        "inner_forward_fn": _forward_mdl_shared,
                        "train_epoch_mmd_fn": mmd_mdl_shared_pair_epoch,
                    }
                res = run_train_eval_mdl_pair(
                    x_train, y_train, x_test, y_test,
                    inner_model, outer_model, cfg, lamb, wandb_run=run,
                    create_inner_fn=create_state_mdl_shared,
                    create_outer_fn=create_state_outer,
                    train_epoch_fn=mdl_shared_pair_epoch_train,
                    eval_inner_epoch_fn=mdl_shared_eval_epoch,
                    eval_outer_epoch_fn=outer_eval_epoch,
                    run_dir=run_dir,
                    start_epoch=resume_start_epoch,
                    init_inner_params=resume_params,
                    **mmd_kw,
                )
            elif mode in ("oracle_train", "erm"):
                res = run_train_eval(
                    x_train, y_train, x_test, y_test,
                    inner_model, cfg, lamb, wandb_run=run,
                    create_state_fn=create_state_oracle,
                    train_epoch_fn=oracle_train_epoch,
                    eval_epoch_fn=oracle_eval_epoch,
                    run_dir=run_dir,
                    start_epoch=resume_start_epoch,
                    init_params=resume_params,
                )
            elif mode == "oracle_pair":
                outer_model = OUTER_MODELS[cfg.model.outer](cfg)
                oracle_ckpt = cfg.model.oracle_checkpoint
                if not oracle_ckpt:
                    raise ValueError(
                        "oracle_pair mode requires model.oracle_checkpoint "
                        "to be set in the config"
                    )
                oracle_params = load_checkpoint(oracle_ckpt)
                print(f"  Loaded oracle checkpoint: {oracle_ckpt}")
                mmd_kw = {}
                if use_mmd:
                    mmd_kw = {
                        "inner_forward_fn": _forward_oracle,
                        "train_epoch_pair_mmd_fn": mmd_oracle_pair_epoch,
                    }
                res = run_train_eval_oracle_pair(
                    x_train, y_train, x_test, y_test,
                    inner_model, outer_model, cfg, lamb, wandb_run=run,
                    create_inner_fn=create_state_oracle,
                    create_outer_fn=create_state_outer,
                    train_epoch_pair_fn=oracle_pair_epoch,
                    eval_epoch_fn=oracle_inner_eval_epoch,
                    run_dir=run_dir,
                    start_epoch=resume_start_epoch,
                    init_inner_params=oracle_params,
                    init_outer_params=resume_params,
                    **mmd_kw,
                )
            elif mode == "cba_om":
                outer_model = OUTER_MODELS[cfg.model.outer](cfg)
                res = run_train_eval_cba_om(
                    x_train, y_train, s_train, x_test, y_test,
                    outer_model, cfg, lamb, wandb_run=run,
                    create_state_fn=create_state_cba_om,
                    train_epoch_fn=cba_om_train_epoch,
                    eval_epoch_fn=cba_om_eval_epoch,
                    run_dir=run_dir,
                    start_epoch=resume_start_epoch,
                    init_params=resume_params,
                )
            elif mode == "ipsn":
                ipsn_model = OUTER_MODELS[cfg.model.outer](cfg)
                res = run_train_eval_ipsn(
                    x_train, y_train, s_train, x_test, y_test,
                    ipsn_model, cfg, lamb, wandb_run=run,
                    create_state_fn=create_state_ipsn,
                    train_epoch_fn=ipsn_train_epoch,
                    eval_epoch_fn=ipsn_eval_epoch,
                    run_dir=run_dir,
                    start_epoch=resume_start_epoch,
                    init_params=resume_params,
                )
            else:
                raise ValueError(f"Unknown model.mode: {cfg.model.mode!r}")

        res["run_time"] = time.time() - t_start
        res["dataset"] = cfg.dataset.name
        res["lambda"] = lamb

        run.summary["final_test_acc"] = res["test_acc"]
        run.summary["final_train_acc"] = res["train_acc"]
        run.summary["best_test_acc"] = res.get("best_test_acc", res["test_acc"])
        run.summary["best_epoch"] = res.get("best_epoch", res.get("epoch"))
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
            "dataloader": dataloader_settings,
            "n_runs": len(all_results),
            "total_time_sec": total_time,
            "results": all_results,
        }, f, indent=2)
    pd.DataFrame(all_results).to_csv(sweep_dir / "summary.csv", index=False)

    summary_run.finish()

    _print_timing_table(all_results, cfg.dataset.name)


if __name__ == "__main__":
    main()
