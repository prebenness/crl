"""Full training-loop runners with W&B logging and checkpointing."""

import time

import jax.numpy as jnp
from jax import random as jrandom

from src.datasets.datasets import make_epoch_batches
from src.mdl.training import anneal_tau
from src.utils.checkpointing import save_checkpoint, save_results


def run_train_eval(x_train, y_train, x_test, y_test, model, cfg, lamb,
                   wandb_run, *, create_state_fn, train_epoch_fn,
                   eval_epoch_fn, run_dir=None):
    """Single-model (inner only) training loop."""
    rng = jrandom.PRNGKey(cfg.training.seed)
    input_shape = (cfg.training.batch_size,) + x_train.shape[1:]
    rng, rng_init = jrandom.split(rng)

    state = create_state_fn(rng_init, model, input_shape, cfg)

    for ep in range(cfg.training.epochs):
        t0 = time.time()

        seed_tr = cfg.training.seed * 10_000 + ep
        seed_te = cfg.training.seed * 20_000 + ep

        rng, rng_epoch = jrandom.split(rng)

        xb, yb = make_epoch_batches(
            x_train, y_train, cfg.training.batch_size, seed_tr,
        )
        xt, yt = make_epoch_batches(
            x_test, y_test, cfg.training.batch_size, seed_te,
        )

        state, metrics = train_epoch_fn(
            state, xb, yb, rng_epoch, lamb, cfg.training.alpha,
        )

        rng, rng_eval = jrandom.split(rng)
        te_loss, te_acc = eval_epoch_fn(state, xt, yt, rng_eval)

        results = {
            "epoch": ep + 1,
            "train_loss": float(metrics["loss"]),
            "train_acc": float(metrics["acc"]),
            "train_kl": float(metrics["kl"]),
            "train_recon": float(metrics["recon"]),
            "train_beta": float(metrics["beta"]),
            "test_loss": float(te_loss),
            "test_acc": float(te_acc),
            "cap": float(lamb),
            "kl_err": float(lamb - metrics["kl"]),
        }
        wandb_run.log(results)

        print(
            f"  Epoch {ep+1}/{cfg.training.epochs}"
            f"  Acc. Train: {float(metrics['acc']):.4f}"
            f" Test: {float(te_acc):.4f}"
            f"  Loss Train: {float(metrics['loss']):.4f}"
            f" Test: {float(te_loss):.4f}"
            f"  Train CE: {float(metrics['ce']):.4f}"
            f" Recon {float(metrics['recon']):.4f}"
            f"  KL: {float(metrics['kl']):.4f}"
            f"  Beta: {float(metrics['beta']):.3f}"
            f"      Time: {time.time()-t0:.2f}s"
        )

    if run_dir is not None:
        save_checkpoint(state.params, run_dir / "checkpoint_final.npz")
        save_results(run_dir, results)

    return results


def run_train_eval_pair(x_train, y_train, x_test, y_test, inner_model,
                        outer_model, cfg, lamb, wandb_run,
                        *, create_inner_fn, create_outer_fn,
                        train_epoch_pair_fn, eval_epoch_fn, run_dir=None):
    """Dual-model (inner + outer) training loop."""
    rng = jrandom.PRNGKey(cfg.training.seed)
    input_shape = (cfg.training.batch_size,) + x_train.shape[1:]
    rng, r1, r2 = jrandom.split(rng, 3)

    inner_state = create_inner_fn(r1, inner_model, input_shape, cfg)
    outer_state = create_outer_fn(r2, outer_model, input_shape, cfg)

    for ep in range(cfg.training.epochs):
        t0 = time.time()
        seed_tr = cfg.training.seed * 10_000 + ep
        seed_te = cfg.training.seed * 20_000 + ep
        rng, rng_epoch = jrandom.split(rng)

        xb, yb = make_epoch_batches(
            x_train, y_train, cfg.training.batch_size, seed_tr,
        )
        xt, yt = make_epoch_batches(
            x_test, y_test, cfg.training.batch_size, seed_te,
        )

        inner_state, outer_state, metrics = train_epoch_pair_fn(
            inner_state, outer_state, xb, yb, rng_epoch,
            lamb, cfg.training.alpha, cfg.hsic.weight,
            num_classes=cfg.model.num_classes,
        )

        rng, rng_eval1, rng_eval2 = jrandom.split(rng, 3)
        te_loss1, te_acc1 = eval_epoch_fn(
            inner_state, xt, yt, rng_eval1,
        )
        te_loss2, te_acc2 = eval_epoch_fn(
            outer_state, xt, yt, rng_eval2,
        )

        results = {
            "epoch": ep + 1,
            "train_acc1": float(metrics["acc1"]),
            "train_loss1": float(metrics["loss1"]),
            "train_kl1": float(metrics["kl1"]),
            "train_recon1": float(metrics["recon1"]),
            "train_beta1": float(metrics["beta1"]),
            "test_acc1": float(te_acc1),
            "test_loss1": float(te_loss1),
            "train_acc2": float(metrics["acc2"]),
            "train_loss2": float(metrics["loss2"]),
            "train_ce2": float(metrics["ce2"]),
            "test_acc2": float(te_acc2),
            "test_loss2": float(te_loss2),
            "hsic": float(metrics["hsic"]),
            "cap": float(lamb),
        }
        wandb_run.log(results)

        print(
            f"  Epoch {ep+1}/{cfg.training.epochs}"
            f" | Inner acc tr {results['train_acc1']:.4f}"
            f" te {results['test_acc1']:.4f}"
            f" | Outer acc tr {results['train_acc2']:.4f}"
            f" te {results['test_acc2']:.4f}"
            f" | hsic {results['hsic']:.4f}"
            f" | {time.time()-t0:.2f}s"
        )

    if run_dir is not None:
        save_checkpoint(inner_state.params, run_dir / "inner_final.npz")
        save_checkpoint(outer_state.params, run_dir / "outer_final.npz")
        save_results(run_dir, results)

    results["train_acc"] = results["train_acc2"]
    results["test_acc"] = results["test_acc2"]
    results["lambda"] = float(lamb)
    return results


def run_train_eval_mdl(x_train, y_train, x_test, y_test, model, cfg, lamb,
                       wandb_run, *, create_state_fn,
                       train_epoch_warmup_fn, train_epoch_fn,
                       eval_epoch_fn, run_dir=None):
    """MDL single-model training loop with warmup and tau annealing."""
    base_rng = jrandom.PRNGKey(cfg.training.seed)
    rng_init_inner = jrandom.fold_in(base_rng, 0)
    rng_train = jrandom.fold_in(base_rng, 2)
    rng_eval = jrandom.fold_in(base_rng, 3)
    input_shape = (cfg.training.batch_size,) + x_train.shape[1:]

    state = create_state_fn(rng_init_inner, model, input_shape, cfg)
    n_train = x_train.shape[0]

    for ep in range(cfg.training.epochs):
        t0 = time.time()

        tau = anneal_tau(ep, cfg.training.epochs,
                         cfg.mdl.tau_start, cfg.mdl.tau_end)
        state = state.replace(tau=jnp.array(tau, dtype=jnp.float32))

        is_warmup = ep < cfg.mdl.warmup_epochs
        epoch_fn = train_epoch_warmup_fn if is_warmup else train_epoch_fn

        seed_tr = cfg.training.seed * 10_000 + ep
        seed_te = cfg.training.seed * 20_000 + ep
        rng_train, rng_epoch = jrandom.split(rng_train)

        xb, yb = make_epoch_batches(
            x_train, y_train, cfg.training.batch_size, seed_tr,
        )
        xt, yt = make_epoch_batches(
            x_test, y_test, cfg.training.batch_size, seed_te,
        )

        state, metrics = epoch_fn(
            state, xb, yb, rng_epoch, lamb, n_train,
        )

        rng_eval, rng_eval_epoch = jrandom.split(rng_eval)
        te_loss, te_acc = eval_epoch_fn(state, xt, yt, rng_eval_epoch)

        results = {
            "epoch": ep + 1,
            "train_loss": float(metrics["loss"]),
            "train_acc": float(metrics["acc"]),
            "train_ce": float(metrics["ce"]),
            "train_hyp_cl": float(metrics["hyp_cl"]),
            "train_entropy": float(metrics["entropy"]),
            "train_hyp_cl_nats": float(metrics["hyp_cl"]),
            "train_entropy_nats": float(metrics["entropy"]),
            "tau": float(tau),
            "test_loss": float(te_loss),
            "test_acc": float(te_acc),
            "cap": float(lamb),
        }
        wandb_run.log(results)

        print(
            f"  Epoch {ep+1}/{cfg.training.epochs}"
            f"  Acc tr {float(metrics['acc']):.4f} te {float(te_acc):.4f}"
            f"  CE {float(metrics['ce']):.4f}"
            f"  HypCL(nats) {float(metrics['hyp_cl']):.1f}"
            f"  H(nats) {float(metrics['entropy']):.1f}"
            f"  tau {float(tau):.4f}"
            f"{'  [warmup]' if is_warmup else ''}"
            f"  {time.time()-t0:.2f}s"
        )

    if run_dir is not None:
        save_checkpoint(state.params, run_dir / "checkpoint_final.npz")
        save_results(run_dir, results)

    return results


def run_train_eval_mdl_pair(x_train, y_train, x_test, y_test, inner_model,
                            outer_model, cfg, lamb, wandb_run,
                            *, create_inner_fn, create_outer_fn,
                            train_epoch_warmup_fn, train_epoch_fn,
                            eval_inner_epoch_fn, eval_outer_epoch_fn,
                            run_dir=None):
    """MDL dual-model (MDL inner + standard outer) training loop."""
    base_rng = jrandom.PRNGKey(cfg.training.seed)
    # Match the inner-model init/train/eval streams used in run_train_eval_mdl
    # so mdl and mdl_pair are directly comparable.
    rng_init_inner = jrandom.fold_in(base_rng, 0)
    rng_init_outer = jrandom.fold_in(base_rng, 1)
    rng_train = jrandom.fold_in(base_rng, 2)
    rng_eval = jrandom.fold_in(base_rng, 3)
    input_shape = (cfg.training.batch_size,) + x_train.shape[1:]

    inner_state = create_inner_fn(rng_init_inner, inner_model, input_shape, cfg)
    outer_state = create_outer_fn(rng_init_outer, outer_model, input_shape, cfg)
    n_train = x_train.shape[0]

    for ep in range(cfg.training.epochs):
        t0 = time.time()

        tau = anneal_tau(ep, cfg.training.epochs,
                         cfg.mdl.tau_start, cfg.mdl.tau_end)
        inner_state = inner_state.replace(
            tau=jnp.array(tau, dtype=jnp.float32),
        )

        is_warmup = ep < cfg.mdl.warmup_epochs
        epoch_fn = train_epoch_warmup_fn if is_warmup else train_epoch_fn

        seed_tr = cfg.training.seed * 10_000 + ep
        seed_te = cfg.training.seed * 20_000 + ep
        rng_train, rng_epoch = jrandom.split(rng_train)

        xb, yb = make_epoch_batches(
            x_train, y_train, cfg.training.batch_size, seed_tr,
        )
        xt, yt = make_epoch_batches(
            x_test, y_test, cfg.training.batch_size, seed_te,
        )

        inner_state, outer_state, metrics = epoch_fn(
            inner_state, outer_state, xb, yb, rng_epoch,
            lamb, n_train, cfg.hsic.weight,
            num_classes=cfg.model.num_classes,
        )

        rng_eval, rng_eval_inner = jrandom.split(rng_eval)
        rng_eval_outer = jrandom.fold_in(rng_eval_inner, 1)
        te_loss1, te_acc1 = eval_inner_epoch_fn(
            inner_state, xt, yt, rng_eval_inner,
        )
        te_loss2, te_acc2 = eval_outer_epoch_fn(
            outer_state, xt, yt, rng_eval_outer,
        )

        results = {
            "epoch": ep + 1,
            "train_acc1": float(metrics["acc1"]),
            "train_loss1": float(metrics["loss1"]),
            "train_ce1": float(metrics["ce1"]),
            "train_hyp_cl": float(metrics["hyp_cl"]),
            "train_entropy": float(metrics["entropy"]),
            "train_hyp_cl_nats": float(metrics["hyp_cl"]),
            "train_entropy_nats": float(metrics["entropy"]),
            "test_acc1": float(te_acc1),
            "test_loss1": float(te_loss1),
            "train_acc2": float(metrics["acc2"]),
            "train_loss2": float(metrics["loss2"]),
            "train_ce2": float(metrics["ce2"]),
            "test_acc2": float(te_acc2),
            "test_loss2": float(te_loss2),
            "hsic": float(metrics["hsic"]),
            "tau": float(tau),
            "cap": float(lamb),
        }
        wandb_run.log(results)

        print(
            f"  Epoch {ep+1}/{cfg.training.epochs}"
            f" | Inner acc tr {results['train_acc1']:.4f}"
            f" te {results['test_acc1']:.4f}"
            f" | Outer acc tr {results['train_acc2']:.4f}"
            f" te {results['test_acc2']:.4f}"
            f" | hsic {results['hsic']:.4f}"
            f" | tau {float(tau):.4f}"
            f"{'  [warmup]' if is_warmup else ''}"
            f" | {time.time()-t0:.2f}s"
        )

    if run_dir is not None:
        save_checkpoint(inner_state.params, run_dir / "inner_final.npz")
        save_checkpoint(outer_state.params, run_dir / "outer_final.npz")
        save_results(run_dir, results)

    results["train_acc"] = results["train_acc2"]
    results["test_acc"] = results["test_acc2"]
    results["lambda"] = float(lamb)
    return results
