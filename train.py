"""Unified training entry point for all spurious correlation benchmarks.

Supports: cMNIST (cmnist_ula), cCIFAR10 (ccifar10), Waterbirds (waterbirds),
CelebA (celeba), CivilComments (civilcomments).

Usage:
    python train.py config/ccifar10/resnet18_single.yaml
    python train.py config/waterbirds/resnet50_pair.yaml
"""

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
import jax.numpy as jnp

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
from src.models.resnet import ResNet18, ResNet50
from src.training.train_state import (
    create_state_resnet, create_state_oracle,
)
from src.training.steps import (
    make_train_step_resnet, make_eval_step_resnet,
    make_train_step_resnet_pair,
    make_train_step_oracle, make_eval_step_oracle,
)
from src.training.epochs import (
    make_train_epoch, make_train_epoch_pair, make_eval_epoch,
)
from src.training.runners import run_train_eval, run_train_eval_pair
from src.datasets.streaming import (
    StreamingLoader, streaming_train_epoch, streaming_eval_epoch,
)


# ---- Model registry ----

def _make_resnet18_cifar(cfg):
    return ResNet18(num_classes=cfg.model.num_classes, cifar_mode=True)


def _make_resnet50(cfg):
    return ResNet50(num_classes=cfg.model.num_classes)


def _make_bert(cfg):
    from src.models.bert import BertClassifier
    return BertClassifier(num_classes=cfg.model.num_classes)


MODELS = {
    "resnet18_cifar": _make_resnet18_cifar,
    "resnet50": _make_resnet50,
    "bert": _make_bert,
}

# Models that use BatchNorm (need ResNet step functions / create_state_resnet)
BN_MODELS = {"resnet18_cifar", "resnet50"}


# ---- Augmentation pipelines ----

def _get_augment_fns(dataset_name):
    """Return (train_augment_fn, eval_transform_fn) for a dataset."""
    if dataset_name == "ccifar10":
        from src.datasets.augmentations import normalize_cifar10
        return (
            lambda rng, x: normalize_cifar10(x),
            normalize_cifar10,
        )
    elif dataset_name == "waterbirds":
        from src.datasets.augmentations import (
            waterbirds_train_augment, waterbirds_eval_transform,
        )
        return waterbirds_train_augment, waterbirds_eval_transform
    elif dataset_name == "celeba":
        from src.datasets.augmentations import (
            celeba_train_augment, celeba_eval_transform,
        )
        return celeba_train_augment, celeba_eval_transform
    else:
        return None, None


# ---- Data loading modes ----
MATERIALIZABLE = {"cmnist_ula", "ccifar10", "waterbirds", "colored_mnist",
                  "irm_colored_mnist", "standard_mnist_bin"}
STREAMING = {"celeba", "civilcomments"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train spurious correlation benchmarks.",
    )
    parser.add_argument("config", help="Path to experiment YAML config.")
    parser.add_argument(
        "overrides", nargs="*",
        help="Config overrides in section.key=value format.",
    )
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a run_dir to resume training from.")
    return parser.parse_args(argv)


def _build_step_fns(cfg, mode, model_name, train_augment_fn, eval_transform_fn):
    """Build JIT-compiled step and epoch functions based on mode and model type."""
    uses_bn = model_name in BN_MODELS

    if uses_bn:
        if mode == "single":
            train_step = make_train_step_resnet(cfg, augment_fn=train_augment_fn)
            eval_step = make_eval_step_resnet(cfg, transform_fn=eval_transform_fn)
            train_epoch = make_train_epoch(train_step)
            eval_epoch = make_eval_epoch(eval_step)
            return {"train_epoch": train_epoch, "eval_epoch": eval_epoch}

        elif mode == "pair":
            pair_step = make_train_step_resnet_pair(cfg, augment_fn=train_augment_fn)
            eval_step = make_eval_step_resnet(cfg, transform_fn=eval_transform_fn)
            pair_epoch = make_train_epoch_pair(pair_step)
            eval_epoch = make_eval_epoch(eval_step)
            return {"pair_epoch": pair_epoch, "eval_epoch": eval_epoch}
    else:
        # Non-BN models (BERT, etc.) use oracle-style steps
        if mode == "single":
            train_step = make_train_step_oracle(cfg)
            eval_step = make_eval_step_oracle(cfg)
            train_epoch = make_train_epoch(train_step)
            eval_epoch = make_eval_epoch(eval_step)
            return {"train_epoch": train_epoch, "eval_epoch": eval_epoch}

    raise ValueError(f"Unsupported mode={mode!r} + model={model_name!r}")


def _create_state(rng, model, input_shape, cfg, model_name, wd_key="weight_decay_inner"):
    """Create the right state type for a model."""
    if model_name in BN_MODELS:
        return create_state_resnet(rng, model, input_shape, cfg, weight_decay_key=wd_key)
    else:
        return create_state_oracle(rng, model, input_shape, cfg)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.overrides:
        apply_overrides(cfg, args.overrides)

    expected_nc = DATASET_NUM_CLASSES.get(cfg.dataset.name)
    if expected_nc is not None and cfg.model.num_classes != expected_nc:
        print(
            f"ERROR: dataset '{cfg.dataset.name}' has {expected_nc} classes "
            f"but config sets model.num_classes={cfg.model.num_classes}.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("JAX devices:", jax.devices())
    jax.config.update("jax_default_matmul_precision", "high")

    cfg_path = Path(args.config).resolve()
    cfg_name = cfg_path.stem
    timestamp = utc_timestamp()
    mode = cfg.model.mode

    # Compute p_corr from beta if set
    if cfg.dataset.beta > 0:
        p_corr_train = 1.0 - cfg.dataset.beta
    else:
        p_corr_train = cfg.dataset.p_train

    train_augment_fn, eval_transform_fn = _get_augment_fns(cfg.dataset.name)

    # Build model(s)
    if cfg.model.inner not in MODELS:
        raise ValueError(
            f"Unknown model '{cfg.model.inner}'. Available: {list(MODELS.keys())}"
        )
    inner_model = MODELS[cfg.model.inner](cfg)

    outer_model = None
    if mode == "pair":
        outer_key = cfg.model.outer if cfg.model.outer else cfg.model.inner
        if outer_key not in MODELS:
            raise ValueError(
                f"Unknown outer model '{outer_key}'. Available: {list(MODELS.keys())}"
            )
        outer_model = MODELS[outer_key](cfg)

    # Build JIT-compiled functions
    step_fns = _build_step_fns(cfg, mode, cfg.model.inner,
                                train_augment_fn, eval_transform_fn)

    # Load data
    print(f"Loading {cfg.dataset.name} dataset...")
    t0 = time.time()

    streaming_mode = cfg.dataset.name in STREAMING

    if not streaming_mode:
        train_ds = build_dataset(
            cfg.dataset.name, train=True,
            p_corr=p_corr_train, seed=cfg.training.seed,
            split="train",
        )
        test_ds = build_dataset(
            cfg.dataset.name, train=False,
            p_corr=cfg.dataset.p_test, seed=cfg.training.seed + 1,
            split="test",
        )
        x_train, y_train = dataset_to_jax_arrays(
            train_ds, batch_size=cfg.training.batch_size,
        )
        x_test, y_test = dataset_to_jax_arrays(
            test_ds, batch_size=cfg.training.batch_size,
        )
        print(f"Train: {x_train.shape}, Test: {x_test.shape}")
    else:
        # Streaming: build PyTorch datasets + StreamingLoaders
        train_ds = build_dataset(
            cfg.dataset.name, split="train",
            p_corr=p_corr_train, seed=cfg.training.seed,
        )
        test_ds = build_dataset(
            cfg.dataset.name, split="test",
            p_corr=cfg.dataset.p_test, seed=cfg.training.seed + 1,
        )
        # For CivilComments, data is 1D tokens not images
        to_nhwc = cfg.dataset.name != "civilcomments"
        train_loader = StreamingLoader(
            train_ds, batch_size=cfg.training.batch_size,
            shuffle=True, drop_last=True, to_nhwc=to_nhwc,
        )
        test_loader = StreamingLoader(
            test_ds, batch_size=cfg.training.batch_size,
            shuffle=False, drop_last=False, to_nhwc=to_nhwc,
        )
        print(f"Train: {len(train_ds)} samples ({len(train_loader)} batches), "
              f"Test: {len(test_ds)} samples ({len(test_loader)} batches)")

    print(f"Data ready in {time.time()-t0:.2f}s")

    # Lambda sweep
    all_results = []
    sweep_start = time.time()

    sweep_name = f"{timestamp}_{cfg_name}_{cfg.dataset.name}_{mode}"
    sweep_dir = make_experiment_dir(cfg.dataset.name, sweep_name)
    runs_dir = sweep_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    save_config(sweep_dir, {
        "experiment": cfg.dataset.name,
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

    experiment_group_id = f"{timestamp}-{cfg.dataset.name}-{mode}"

    for lamb in cfg.lambdas:
        lamb = float(lamb)
        print(f"\n--- Sweep (lamb={lamb:.1e}) ---")

        run_dir = runs_dir / f"lambda_{lamb:.1e}"
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(run_dir, {
            "experiment": cfg.dataset.name,
            "mode": mode, "lambda": lamb,
            "config_path": str(cfg_path),
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
        else:
            resume_params = None
            resume_start_epoch = 0

        run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            group=experiment_group_id,
            name=f"{cfg.dataset.name}-lamb_{lamb:.1e}",
            config={"lambda": lamb, "dataset": cfg.dataset.name, "mode": mode},
            reinit=True,
        )

        t_start = time.time()

        with TeeLogger(run_dir / "train.log"):
            if streaming_mode:
                res = _run_streaming(
                    train_loader, test_loader, inner_model, cfg, lamb,
                    run, step_fns, run_dir, resume_start_epoch,
                )
            elif mode == "single":
                res = run_train_eval(
                    x_train, y_train, x_test, y_test,
                    inner_model, cfg, lamb, wandb_run=run,
                    create_state_fn=lambda rng, m, s, c: _create_state(
                        rng, m, s, c, cfg.model.inner),
                    train_epoch_fn=step_fns["train_epoch"],
                    eval_epoch_fn=step_fns["eval_epoch"],
                    run_dir=run_dir,
                    start_epoch=resume_start_epoch,
                    init_params=resume_params,
                )
            elif mode == "pair":
                res = run_train_eval_pair(
                    x_train, y_train, x_test, y_test,
                    inner_model, outer_model, cfg, lamb, wandb_run=run,
                    create_inner_fn=lambda rng, m, s, c: _create_state(
                        rng, m, s, c, cfg.model.inner),
                    create_outer_fn=lambda rng, m, s, c: _create_state(
                        rng, m, s, c, cfg.model.inner, "weight_decay_outer"),
                    train_epoch_pair_fn=step_fns["pair_epoch"],
                    eval_epoch_fn=step_fns["eval_epoch"],
                    run_dir=run_dir,
                    start_epoch=resume_start_epoch,
                    init_inner_params=resume_params,
                )
            else:
                raise ValueError(f"Unknown mode: {mode!r}")

        res["run_time"] = time.time() - t_start
        res["dataset"] = cfg.dataset.name
        res["lambda"] = lamb

        run.summary["final_test_acc"] = res["test_acc"]
        run.summary["final_train_acc"] = res["train_acc"]
        run.summary["best_test_acc"] = res.get("best_test_acc", res["test_acc"])
        run.finish()

        all_results.append(res)

    total_time = time.time() - sweep_start
    print(f"\nTotal sweep wall time: {total_time/60:.2f} minutes")

    # Save summary
    with open(sweep_dir / "summary.json", "w") as f:
        json.dump({
            "experiment": cfg.dataset.name,
            "config_path": str(cfg_path),
            "mode": mode,
            "n_runs": len(all_results),
            "total_time_sec": total_time,
            "results": all_results,
        }, f, indent=2)

    summary_run = wandb.init(
        entity=cfg.wandb.entity, project=cfg.wandb.project,
        group=experiment_group_id, name="summary", job_type="summary",
    )
    summary_run.log({"raw_data": wandb.Table(dataframe=pd.DataFrame(all_results))})
    summary_run.finish()


def _run_streaming(train_loader, test_loader, model, cfg, lamb,
                   wandb_run, step_fns, run_dir, start_epoch):
    """Streaming training loop for datasets too large to materialize."""
    from src.utils.checkpointing import (
        save_checkpoint, save_results, checkpoint_path, save_checkpoint_meta,
    )
    from jax import random as jrandom

    rng = jrandom.PRNGKey(cfg.training.seed)

    # Peek at one batch to get input shape
    sample_x, sample_y = next(iter(train_loader))
    input_shape = (cfg.training.batch_size,) + sample_x.shape[1:]

    rng, rng_init = jrandom.split(rng)
    state = _create_state(rng_init, model, input_shape, cfg, cfg.model.inner)

    # Build raw step functions for streaming (not epoch-level)
    uses_bn = cfg.model.inner in BN_MODELS
    if uses_bn:
        from src.training.steps import make_train_step_resnet, make_eval_step_resnet
        from src.datasets.augmentations import (
            celeba_train_augment, celeba_eval_transform,
        )
        aug_fn, eval_fn = celeba_train_augment, celeba_eval_transform
        if cfg.dataset.name == "civilcomments":
            aug_fn, eval_fn = None, None
        train_step = make_train_step_resnet(cfg, augment_fn=aug_fn)
        eval_step = make_eval_step_resnet(cfg, transform_fn=eval_fn)
    else:
        from src.training.steps import make_train_step_oracle, make_eval_step_oracle
        train_step = make_train_step_oracle(cfg)
        eval_step = make_eval_step_oracle(cfg)

    best = {"test_acc": -1.0, "params": None, "epoch": 0}

    for ep in range(start_epoch, cfg.training.epochs):
        t0 = time.time()
        rng, rng_epoch = jrandom.split(rng)

        state, metrics = streaming_train_epoch(
            state, train_loader, train_step, rng_epoch, lamb, cfg.training.alpha,
        )

        rng, rng_eval = jrandom.split(rng)
        te_loss, te_acc = streaming_eval_epoch(
            state, test_loader, eval_step, rng_eval,
        )

        current_acc = float(te_acc)
        if current_acc > best["test_acc"]:
            best["test_acc"] = current_acc
            best["params"] = state.params
            best["epoch"] = ep + 1
            if run_dir is not None:
                save_checkpoint(state.params,
                                checkpoint_path(run_dir, "best.npz"))
                save_checkpoint_meta(run_dir, ep + 1, best["test_acc"],
                                     best_checkpoint_epoch=ep + 1)
            print(f"    >>> NEW BEST test_acc={current_acc:.4f} (epoch {ep+1})")

        train_acc = float(metrics.get("accuracy", 0.0))
        results = {
            "epoch": ep + 1,
            "train_acc": train_acc,
            "test_acc": current_acc,
            "test_nll_nats": float(te_loss),
            "best_test_acc": best["test_acc"],
            "best_epoch": best["epoch"],
        }
        wandb_run.log(results)

        print(
            f"  Epoch {ep+1}/{cfg.training.epochs}"
            f"  train_acc {train_acc:.4f}"
            f" test_acc {current_acc:.4f}"
            f"  {time.time()-t0:.2f}s"
        )

    results["best_test_acc"] = best["test_acc"]
    results["best_epoch"] = best["epoch"]
    results["n_restarts"] = 0

    if run_dir is not None:
        save_checkpoint(state.params, checkpoint_path(run_dir, "final.npz"))
        save_checkpoint_meta(run_dir, ep + 1, best["test_acc"],
                             best_checkpoint_epoch=best["epoch"])
        save_results(run_dir, results)

    return results


if __name__ == "__main__":
    main()
