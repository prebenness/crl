"""Full training-loop runners with W&B logging and checkpointing."""

import time

import jax.numpy as jnp
from jax import random as jrandom

from src.datasets.datasets import make_epoch_batches
from src.mdl.training import anneal_tau_st_phase
from src.utils.checkpointing import save_checkpoint, save_results, checkpoint_path


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
            "train_acc": float(metrics["accuracy"]),
            "test_acc": float(te_acc),
            "objective_total_nats": float(metrics["objective_total_nats"]),
            "data_nll_nats": float(metrics["data_nll_nats"]),
            "reconstruction_nll_nats": float(metrics["reconstruction_nll_nats"]),
            "kl_latent_nats": float(metrics["kl_latent_nats"]),
            "beta_controller": float(metrics["beta_controller"]),
            "test_nll_nats": float(te_loss),
            "capacity_target_nats": float(lamb),
            "kl_target_gap_nats": float(lamb - metrics["kl_latent_nats"]),
        }
        wandb_run.log(results)

        print(
            f"  Epoch {ep+1}/{cfg.training.epochs}"
            f"  train_acc {float(metrics['accuracy']):.4f}"
            f" test_acc {float(te_acc):.4f}"
            f"  objective_total {float(metrics['objective_total_nats']):.4f}n"
            f" test_nll {float(te_loss):.4f}n"
            f"  data_nll {float(metrics['data_nll_nats']):.4f}n"
            f" recon_nll {float(metrics['reconstruction_nll_nats']):.4f}n"
            f"  kl_latent {float(metrics['kl_latent_nats']):.4f}n"
            f"  beta_controller {float(metrics['beta_controller']):.3f}"
            f"      Time: {time.time()-t0:.2f}s"
        )

    if run_dir is not None:
        save_checkpoint(state.params, checkpoint_path(run_dir, "final.npz"))
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
            "train_acc1": float(metrics["accuracy_inner"]),
            "test_acc1": float(te_acc1),
            "objective_total_nats": float(metrics["objective_total_nats"]),
            "data_nll_nats": float(metrics["data_nll_nats"]),
            "reconstruction_nll_nats": float(metrics["reconstruction_nll_nats"]),
            "kl_latent_nats": float(metrics["kl_latent_nats"]),
            "beta_controller": float(metrics["beta_controller"]),
            "test_nll_nats_inner": float(te_loss1),
            "train_acc2": float(metrics["accuracy_outer"]),
            "test_acc2": float(te_acc2),
            "objective_total_outer_nats": float(metrics["objective_total_outer_nats"]),
            "data_nll_outer_nats": float(metrics["data_nll_outer_nats"]),
            "test_nll_nats_outer": float(te_loss2),
            "hsic": float(metrics["hsic"]),
            "capacity_target_nats": float(lamb),
            "kl_target_gap_nats": float(lamb - metrics["kl_latent_nats"]),
        }
        wandb_run.log(results)

        print(
            f"  Epoch {ep+1}/{cfg.training.epochs}"
            f" | inner_train_acc {results['train_acc1']:.4f}"
            f" inner_test_acc {results['test_acc1']:.4f}"
            f" | inner_objective_total {results['objective_total_nats']:.4f}n"
            f" | outer_train_acc {results['train_acc2']:.4f}"
            f" outer_test_acc {results['test_acc2']:.4f}"
            f" | outer_objective_total {results['objective_total_outer_nats']:.4f}n"
            f" | hsic {results['hsic']:.4f}"
            f" | {time.time()-t0:.2f}s"
        )

    if run_dir is not None:
        save_checkpoint(inner_state.params, checkpoint_path(run_dir, "inner_final.npz"))
        save_checkpoint(outer_state.params, checkpoint_path(run_dir, "outer_final.npz"))
        save_results(run_dir, results)

    results["train_acc"] = results["train_acc2"]
    results["test_acc"] = results["test_acc2"]
    results["lambda"] = float(lamb)
    return results


def run_train_eval_mdl(x_train, y_train, x_test, y_test, model, cfg, lamb,
                       wandb_run, *, create_state_fn,
                       train_epoch_warmup_fn, train_epoch_bridge_fn,
                       train_epoch_fn,
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

        tau = anneal_tau_st_phase(
            ep, cfg.training.epochs, cfg.mdl.warmup_epochs,
            cfg.mdl.tau_start, cfg.mdl.tau_end,
        )
        state = state.replace(tau=jnp.array(tau, dtype=jnp.float32))

        is_warmup = ep < cfg.mdl.warmup_epochs
        is_bridge = (
            train_epoch_bridge_fn is not None
            and cfg.mdl.warmup_epochs > 0
            and cfg.mdl.warmup_epochs < cfg.training.epochs
            and ep == cfg.mdl.warmup_epochs
        )
        if is_warmup:
            epoch_fn = train_epoch_warmup_fn
        elif is_bridge:
            epoch_fn = train_epoch_bridge_fn
        else:
            epoch_fn = train_epoch_fn

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

        phase = "warmup" if is_warmup else ("bridge" if is_bridge else "st")

        results = {
            "epoch": ep + 1,
            "phase": phase,
            "tau": float(tau),
            "lambda_reg": float(lamb),
            "train_acc": float(metrics["accuracy"]),
            "test_acc": float(te_acc),
            "objective_total_nats": float(metrics["objective_total_nats"]),
            "data_nll_nats": float(metrics["data_nll_nats"]),
            "complexity_expected_nats": float(metrics["complexity_expected_nats"]),
            "entropy_weights_nats": float(metrics["entropy_weights_nats"]),
            "reg_complexity_weighted_nats": float(metrics["reg_complexity_weighted_nats"]),
            "reg_entropy_bonus_nats": float(metrics["reg_entropy_bonus_nats"]),
            "reg_net_nats": float(metrics["reg_net_nats"]),
            "objective_total_bits": float(metrics["objective_total_bits"]),
            "data_nll_bits": float(metrics["data_nll_bits"]),
            "complexity_expected_bits": float(metrics["complexity_expected_bits"]),
            "entropy_weights_bits": float(metrics["entropy_weights_bits"]),
            "reg_complexity_weighted_bits": float(metrics["reg_complexity_weighted_bits"]),
            "reg_entropy_bonus_bits": float(metrics["reg_entropy_bonus_bits"]),
            "reg_net_bits": float(metrics["reg_net_bits"]),
            "test_nll_nats": float(te_loss),
            "test_nll_bits": float(te_loss / jnp.log(2.0)),
        }
        wandb_run.log(results)

        print(
            f"  Epoch {ep+1}/{cfg.training.epochs}"
            f"  train_acc {float(metrics['accuracy']):.4f}"
            f" test_acc {float(te_acc):.4f}"
            f"  objective_total {float(metrics['objective_total_nats']):.4f}n"
            f"  data_nll {float(metrics['data_nll_nats']):.4f}n"
            f"  reg_net {float(metrics['reg_net_nats']):.4f}n"
            f" (reg_complexity_weighted {float(metrics['reg_complexity_weighted_nats']):.4f}n"
            f" - reg_entropy_bonus {float(metrics['reg_entropy_bonus_nats']):.4f}n)"
            f"  complexity_expected {float(metrics['complexity_expected_nats']):.1f}n"
            f"  entropy_weights {float(metrics['entropy_weights_nats']):.1f}n"
            f"  tau {float(tau):.4f}"
            f"{'  [warmup]' if is_warmup else ('  [bridge]' if is_bridge else '')}"
            f"  {time.time()-t0:.2f}s"
        )

    if run_dir is not None:
        save_checkpoint(state.params, checkpoint_path(run_dir, "final.npz"))
        save_results(run_dir, results)

    return results


def run_train_eval_mdl_pair(x_train, y_train, x_test, y_test, inner_model,
                            outer_model, cfg, lamb, wandb_run,
                            *, create_inner_fn, create_outer_fn,
                            train_epoch_warmup_fn, train_epoch_bridge_fn,
                            train_epoch_fn,
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

        tau = anneal_tau_st_phase(
            ep, cfg.training.epochs, cfg.mdl.warmup_epochs,
            cfg.mdl.tau_start, cfg.mdl.tau_end,
        )
        inner_state = inner_state.replace(
            tau=jnp.array(tau, dtype=jnp.float32),
        )

        is_warmup = ep < cfg.mdl.warmup_epochs
        is_bridge = (
            train_epoch_bridge_fn is not None
            and cfg.mdl.warmup_epochs > 0
            and cfg.mdl.warmup_epochs < cfg.training.epochs
            and ep == cfg.mdl.warmup_epochs
        )
        if is_warmup:
            epoch_fn = train_epoch_warmup_fn
        elif is_bridge:
            epoch_fn = train_epoch_bridge_fn
        else:
            epoch_fn = train_epoch_fn

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

        phase = "warmup" if is_warmup else ("bridge" if is_bridge else "st")

        results = {
            "epoch": ep + 1,
            "phase": phase,
            "tau": float(tau),
            "lambda_reg": float(lamb),
            "train_acc1": float(metrics["accuracy_inner"]),
            "objective_total_nats": float(metrics["objective_total_nats"]),
            "data_nll_nats": float(metrics["data_nll_nats"]),
            "complexity_expected_nats": float(metrics["complexity_expected_nats"]),
            "entropy_weights_nats": float(metrics["entropy_weights_nats"]),
            "reg_complexity_weighted_nats": float(metrics["reg_complexity_weighted_nats"]),
            "reg_entropy_bonus_nats": float(metrics["reg_entropy_bonus_nats"]),
            "reg_net_nats": float(metrics["reg_net_nats"]),
            "objective_total_bits": float(metrics["objective_total_bits"]),
            "data_nll_bits": float(metrics["data_nll_bits"]),
            "complexity_expected_bits": float(metrics["complexity_expected_bits"]),
            "entropy_weights_bits": float(metrics["entropy_weights_bits"]),
            "reg_complexity_weighted_bits": float(metrics["reg_complexity_weighted_bits"]),
            "reg_entropy_bonus_bits": float(metrics["reg_entropy_bonus_bits"]),
            "reg_net_bits": float(metrics["reg_net_bits"]),
            "test_acc1": float(te_acc1),
            "test_nll_nats_inner": float(te_loss1),
            "test_nll_bits_inner": float(te_loss1 / jnp.log(2.0)),
            "train_acc2": float(metrics["accuracy_outer"]),
            "objective_total_outer_nats": float(metrics["objective_total_outer_nats"]),
            "data_nll_outer_nats": float(metrics["data_nll_outer_nats"]),
            "test_acc2": float(te_acc2),
            "test_nll_nats_outer": float(te_loss2),
            "test_nll_bits_outer": float(te_loss2 / jnp.log(2.0)),
            "hsic": float(metrics["hsic"]),
        }
        wandb_run.log(results)

        print(
            f"  Epoch {ep+1}/{cfg.training.epochs}"
            f" | inner_train_acc {results['train_acc1']:.4f}"
            f" inner_test_acc {results['test_acc1']:.4f}"
            f" | inner_objective_total {results['objective_total_nats']:.4f}n"
            f" inner_data_nll {results['data_nll_nats']:.4f}n"
            f" inner_reg_net {results['reg_net_nats']:.4f}n"
            f" | outer_train_acc {results['train_acc2']:.4f}"
            f" outer_test_acc {results['test_acc2']:.4f}"
            f" | hsic {results['hsic']:.4f}"
            f" | tau {float(tau):.4f}"
            f"{'  [warmup]' if is_warmup else ('  [bridge]' if is_bridge else '')}"
            f" | {time.time()-t0:.2f}s"
        )

    if run_dir is not None:
        save_checkpoint(inner_state.params, checkpoint_path(run_dir, "inner_final.npz"))
        save_checkpoint(outer_state.params, checkpoint_path(run_dir, "outer_final.npz"))
        save_results(run_dir, results)

    results["train_acc"] = results["train_acc2"]
    results["test_acc"] = results["test_acc2"]
    results["lambda"] = float(lamb)
    return results
