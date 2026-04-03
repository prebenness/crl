"""Full training-loop runners with W&B logging and checkpointing."""

import time
import jax
import jax.numpy as jnp
from jax import random as jrandom

from src.datasets.datasets import make_epoch_batches, make_eval_batches
from mdl.src.mdl.training import anneal_tau
from src.utils.checkpointing import (
    save_checkpoint, save_results, checkpoint_path, save_checkpoint_meta,
)


def precompute_predictions(state, x, batch_size, forward_fn):
    """Run forward_fn on all of x in batches, return argmax predictions.

    Args:
        state: model train state (params used by forward_fn).
        x: [N, ...] full training data.
        batch_size: batch size for forward passes.
        forward_fn: callable(state, x_batch) -> logits [B, C].

    Returns:
        [N] int32 array of predicted class labels.
    """
    xb, _, counts = make_eval_batches(x, jnp.zeros(x.shape[0], dtype=jnp.int32),
                                       batch_size)
    n_batches = xb.shape[0]

    @jax.jit
    def _predict_batch(x_batch):
        logits = forward_fn(state, x_batch)
        return jnp.argmax(logits, axis=-1)

    all_preds = []
    for i in range(n_batches):
        preds = _predict_batch(xb[i])
        all_preds.append(preds[:int(counts[i])])
    return jnp.concatenate(all_preds, axis=0)


def _forward_vib(state, x_batch):
    """Single-sample VIB forward for prediction precomputation."""
    rng = jrandom.PRNGKey(0)
    logits, _ = state.apply_fn(
        {"params": state.params}, x_batch, train=True,
        rngs={"noise": rng},
    )
    return logits


def _forward_oracle(state, x_batch):
    """Deterministic oracle/ERM forward for prediction precomputation."""
    logits, _ = state.apply_fn(
        {"params": state.params}, x_batch, train=False,
    )
    return logits


def _forward_mdl(state, x_batch):
    """Deterministic MDL forward for prediction precomputation."""
    logits, _ = state.apply_fn(
        {"params": state.params}, x_batch, tau=1.0, train=False,
    )
    return logits


def _forward_mdl_shared(state, x_batch):
    """Deterministic shared-MDL forward for prediction precomputation."""
    logits, _ = state.apply_fn(
        {"params": {"logits": state.params["logits"]}},
        x_batch, tau=1.0, train=False,
    )
    return logits


def compute_mmd_weights(y, s, num_classes, smoothing_eps=1e-6, w_max=500.0):
    """Compute importance weights w_i = q(s_i) / p_smooth(s_i | y_i).

    q(s) = 1/K (uniform). p(s|y) estimated from counts with Laplace smoothing.

    Args:
        y: [N] int32 true labels.
        s: [N] int32 predicted color labels.
        num_classes: K (used for both class and color dimensions).
        smoothing_eps: Laplace smoothing strength.
        w_max: maximum weight clip.

    Returns:
        [N] float32 importance weights.
    """
    K = num_classes
    q_s = 1.0 / K  # uniform target

    # Compute p(s|y) as a [K, K] matrix via one-hot outer product
    y_onehot = jax.nn.one_hot(y, K)        # [N, K]
    s_onehot = jax.nn.one_hot(s, K)        # [N, K]
    counts = y_onehot.T @ s_onehot         # [K, K]

    row_sums = counts.sum(axis=1, keepdims=True)
    p_hat = counts / jnp.maximum(row_sums, 1.0)

    # Laplace smoothing: p_smooth = (1 - eps) * p_hat + eps / K
    p_smooth = (1.0 - smoothing_eps) * p_hat + smoothing_eps / K

    # Per-example weights
    p_s_given_y = p_smooth[y, s]
    w = q_s / p_s_given_y
    w = jnp.clip(w, 0.0, w_max)
    return w


def _get_patience_cfg(cfg):
    """Read patience settings from cfg, tolerating missing checkpointing."""
    ckpt = getattr(cfg, "checkpointing", None)
    es = getattr(ckpt, "early_stopping_patience", 0) if ckpt else 0
    rs = getattr(ckpt, "restart_patience", 0) if ckpt else 0
    return es, rs


def _check_patience(state, best, epochs_since_best, n_restarts,
                    early_stopping_patience, restart_patience, ep, label=""):
    """Check early stopping / restart-with-patience after each epoch.

    Returns (state, epochs_since_best, n_restarts, should_stop).
    """
    # Early stopping
    if early_stopping_patience > 0 and epochs_since_best >= early_stopping_patience:
        print(f"  {label}Early stopping at epoch {ep+1} "
              f"(no improvement for {early_stopping_patience} epochs)")
        return state, epochs_since_best, n_restarts, True

    # Restart with patience
    if (restart_patience > 0
            and best["params"] is not None
            and epochs_since_best >= restart_patience):
        reloaded = jax.tree.map(jnp.copy, best["params"])
        state = state.replace(
            params=reloaded,
            opt_state=state.tx.init(reloaded),
        )
        epochs_since_best = 0
        n_restarts += 1
        print(f"  {label}RESTART #{n_restarts} at epoch {ep+1} "
              f"-> best checkpoint (epoch {best['epoch']})")

    return state, epochs_since_best, n_restarts, False


# ---- Single-model runner ----

def run_train_eval(x_train, y_train, x_test, y_test, model, cfg, lamb,
                   wandb_run, *, create_state_fn, train_epoch_fn,
                   eval_epoch_fn, run_dir=None,
                   start_epoch=0, init_params=None):
    """Single-model (inner only) training loop."""
    rng = jrandom.PRNGKey(cfg.training.seed)
    input_shape = (cfg.training.batch_size,) + x_train.shape[1:]
    rng, rng_init = jrandom.split(rng)

    state = create_state_fn(rng_init, model, input_shape, cfg)
    if init_params is not None:
        state = state.replace(params=init_params)

    # Deterministic eval batches covering all test samples (padded).
    xt, yt, te_counts = make_eval_batches(
        x_test, y_test, cfg.training.batch_size,
    )

    early_stopping_patience, restart_patience = _get_patience_cfg(cfg)
    best = {"test_acc": -1.0, "params": None, "epoch": 0}
    epochs_since_best = 0
    n_restarts = 0

    for ep in range(start_epoch, cfg.training.epochs):
        t0 = time.time()

        seed_tr = cfg.training.seed * 10_000 + ep

        rng, rng_epoch = jrandom.split(rng)

        xb, yb = make_epoch_batches(
            x_train, y_train, cfg.training.batch_size, seed_tr,
        )

        state, metrics = train_epoch_fn(
            state, xb, yb, rng_epoch, lamb, cfg.training.alpha,
        )

        rng, rng_eval = jrandom.split(rng)
        te_loss, te_acc = eval_epoch_fn(state, xt, yt, rng_eval, te_counts)

        # Best-checkpoint tracking
        current_acc = float(te_acc)
        if current_acc > best["test_acc"]:
            best["test_acc"] = current_acc
            best["params"] = jax.tree.map(jnp.copy, state.params)
            best["epoch"] = ep + 1
            epochs_since_best = 0
            if run_dir is not None:
                save_checkpoint(state.params,
                                checkpoint_path(run_dir, "best.npz"))
                save_checkpoint_meta(run_dir, ep + 1, best["test_acc"],
                                     best_checkpoint_epoch=ep + 1)
            print(f"    >>> NEW BEST test_acc={current_acc:.4f} "
                  f"(epoch {ep+1})")
        else:
            epochs_since_best += 1

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
            "best_test_acc": best["test_acc"],
            "best_epoch": best["epoch"],
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

        state, epochs_since_best, n_restarts, should_stop = _check_patience(
            state, best, epochs_since_best, n_restarts,
            early_stopping_patience, restart_patience, ep,
        )
        if should_stop:
            break

    results["test_acc"] = best["test_acc"]
    results["best_test_acc"] = best["test_acc"]
    results["best_epoch"] = best["epoch"]
    results["n_restarts"] = n_restarts

    if run_dir is not None:
        save_checkpoint(state.params, checkpoint_path(run_dir, "final.npz"))
        save_checkpoint_meta(run_dir, ep + 1, best["test_acc"],
                             best_checkpoint_epoch=best["epoch"])
        save_results(run_dir, results)

    return results


# ---- CBA-OM runner ----

def run_train_eval_cba_om(x_train, y_train, s_train, x_test, y_test,
                          model, cfg, lamb, wandb_run, *,
                          create_state_fn, train_epoch_fn,
                          eval_epoch_fn, run_dir=None,
                          start_epoch=0, init_params=None):
    """CBA-OM training loop: colour-conditioned training, marginalised eval."""
    rng = jrandom.PRNGKey(cfg.training.seed)
    input_shape = (cfg.training.batch_size,) + x_train.shape[1:]
    rng, rng_init = jrandom.split(rng)

    state = create_state_fn(rng_init, model, input_shape, cfg)
    if init_params is not None:
        state = state.replace(params=init_params)

    xt, yt, te_counts = make_eval_batches(
        x_test, y_test, cfg.training.batch_size,
    )

    early_stopping_patience, restart_patience = _get_patience_cfg(cfg)
    best = {"test_acc": -1.0, "params": None, "epoch": 0}
    epochs_since_best = 0
    n_restarts = 0

    for ep in range(start_epoch, cfg.training.epochs):
        t0 = time.time()
        seed_tr = cfg.training.seed * 10_000 + ep
        rng, rng_epoch = jrandom.split(rng)

        xb, yb, sb = make_epoch_batches(
            x_train, y_train, cfg.training.batch_size, seed_tr, s_train,
        )

        state, metrics = train_epoch_fn(
            state, xb, yb, sb, rng_epoch, lamb, cfg.training.alpha,
        )

        rng, rng_eval = jrandom.split(rng)
        te_loss, te_acc = eval_epoch_fn(state, xt, yt, rng_eval, te_counts)

        current_acc = float(te_acc)
        if current_acc > best["test_acc"]:
            best["test_acc"] = current_acc
            best["params"] = jax.tree.map(jnp.copy, state.params)
            best["epoch"] = ep + 1
            epochs_since_best = 0
            if run_dir is not None:
                save_checkpoint(state.params,
                                checkpoint_path(run_dir, "best.npz"))
                save_checkpoint_meta(run_dir, ep + 1, best["test_acc"],
                                     best_checkpoint_epoch=ep + 1)
            print(f"    >>> NEW BEST test_acc={current_acc:.4f} "
                  f"(epoch {ep+1})")
        else:
            epochs_since_best += 1

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
            "best_test_acc": best["test_acc"],
            "best_epoch": best["epoch"],
        }
        wandb_run.log(results)

        print(
            f"  Epoch {ep+1}/{cfg.training.epochs}"
            f"  train_acc {float(metrics['accuracy']):.4f}"
            f" test_acc {float(te_acc):.4f}"
            f"  ce_loss {float(metrics['data_nll_nats']):.4f}n"
            f"  best {best['test_acc']:.4f}"
            f"  ({time.time()-t0:.2f}s)"
        )

        state, epochs_since_best, n_restarts, should_stop = _check_patience(
            state, best, epochs_since_best, n_restarts,
            early_stopping_patience, restart_patience, ep,
        )
        if should_stop:
            break

    results["best_test_acc"] = best["test_acc"]
    results["best_epoch"] = best["epoch"]
    results["n_restarts"] = n_restarts

    if run_dir is not None:
        save_checkpoint(state.params, checkpoint_path(run_dir, "final.npz"))
        save_checkpoint_meta(run_dir, ep + 1, best["test_acc"],
                             best_checkpoint_epoch=best["epoch"])
        save_results(run_dir, results)

    return results


def run_train_eval_ipsn(x_train, y_train, s_train, x_test, y_test,
                        model, cfg, lamb, wandb_run, *,
                        create_state_fn, train_epoch_fn,
                        eval_epoch_fn, run_dir=None,
                        start_epoch=0, init_params=None):
    """IPSN training loop: disentangled encoder, content-only eval."""
    rng = jrandom.PRNGKey(cfg.training.seed)
    input_shape = (cfg.training.batch_size,) + x_train.shape[1:]
    rng, rng_init = jrandom.split(rng)

    state = create_state_fn(rng_init, model, input_shape, cfg)
    if init_params is not None:
        state = state.replace(params=init_params)

    xt, yt, te_counts = make_eval_batches(
        x_test, y_test, cfg.training.batch_size,
    )

    early_stopping_patience, restart_patience = _get_patience_cfg(cfg)
    best = {"test_acc": -1.0, "params": None, "epoch": 0}
    epochs_since_best = 0
    n_restarts = 0

    for ep in range(start_epoch, cfg.training.epochs):
        t0 = time.time()
        seed_tr = cfg.training.seed * 10_000 + ep
        rng, rng_epoch = jrandom.split(rng)

        xb, yb, sb = make_epoch_batches(
            x_train, y_train, cfg.training.batch_size, seed_tr, s_train,
        )

        state, metrics = train_epoch_fn(
            state, xb, yb, sb, rng_epoch, lamb, cfg.training.alpha,
        )

        rng, rng_eval = jrandom.split(rng)
        te_loss, te_acc = eval_epoch_fn(state, xt, yt, rng_eval, te_counts)

        current_acc = float(te_acc)
        if current_acc > best["test_acc"]:
            best["test_acc"] = current_acc
            best["params"] = jax.tree.map(jnp.copy, state.params)
            best["epoch"] = ep + 1
            epochs_since_best = 0
            if run_dir is not None:
                save_checkpoint(state.params,
                                checkpoint_path(run_dir, "best.npz"))
                save_checkpoint_meta(run_dir, ep + 1, best["test_acc"],
                                     best_checkpoint_epoch=ep + 1)
            print(f"    >>> NEW BEST test_acc={current_acc:.4f} "
                  f"(epoch {ep+1})")
        else:
            epochs_since_best += 1

        results = {
            "epoch": ep + 1,
            "train_acc": float(metrics["accuracy"]),
            "test_acc": float(te_acc),
            "objective_total_nats": float(metrics["objective_total_nats"]),
            "data_nll_nats": float(metrics["data_nll_nats"]),
            "reconstruction_nll_nats": float(metrics["reconstruction_nll_nats"]),
            "kl_latent_nats": float(metrics["kl_latent_nats"]),
            "beta_controller": float(metrics["beta_controller"]),
            "ce_color": float(metrics["ce_color"]),
            "ce_adversary": float(metrics["ce_adversary"]),
            "test_nll_nats": float(te_loss),
            "capacity_target_nats": float(lamb),
            "kl_target_gap_nats": float(lamb - metrics["kl_latent_nats"]),
            "best_test_acc": best["test_acc"],
            "best_epoch": best["epoch"],
        }
        wandb_run.log(results)

        print(
            f"  Epoch {ep+1}/{cfg.training.epochs}"
            f"  train_acc {float(metrics['accuracy']):.4f}"
            f" test_acc {float(te_acc):.4f}"
            f"  ce_digit {float(metrics['data_nll_nats']):.4f}"
            f"  ce_color {float(metrics['ce_color']):.4f}"
            f"  ce_adv {float(metrics['ce_adversary']):.4f}"
            f"  recon {float(metrics['reconstruction_nll_nats']):.4f}"
            f"  best {best['test_acc']:.4f}"
            f"  ({time.time()-t0:.2f}s)"
        )

        state, epochs_since_best, n_restarts, should_stop = _check_patience(
            state, best, epochs_since_best, n_restarts,
            early_stopping_patience, restart_patience, ep,
        )
        if should_stop:
            break

    results["best_test_acc"] = best["test_acc"]
    results["best_epoch"] = best["epoch"]
    results["n_restarts"] = n_restarts

    if run_dir is not None:
        save_checkpoint(state.params, checkpoint_path(run_dir, "final.npz"))
        save_checkpoint_meta(run_dir, ep + 1, best["test_acc"],
                             best_checkpoint_epoch=best["epoch"])
        save_results(run_dir, results)

    return results


# ---- Pair-model runner ----

def run_train_eval_pair(x_train, y_train, x_test, y_test, inner_model,
                        outer_model, cfg, lamb, wandb_run,
                        *, create_inner_fn, create_outer_fn,
                        train_epoch_pair_fn, eval_epoch_fn, run_dir=None,
                        start_epoch=0, init_inner_params=None,
                        init_outer_params=None,
                        inner_forward_fn=None,
                        train_epoch_pair_mmd_fn=None):
    """Dual-model (inner + outer) training loop.

    When ``cfg.model.outer_loss == "mmd"``, the caller must also provide
    ``inner_forward_fn`` and ``train_epoch_pair_mmd_fn``.
    """
    use_mmd = cfg.model.outer_loss == "mmd"
    rng = jrandom.PRNGKey(cfg.training.seed)
    input_shape = (cfg.training.batch_size,) + x_train.shape[1:]
    rng, r1, r2 = jrandom.split(rng, 3)

    inner_state = create_inner_fn(r1, inner_model, input_shape, cfg)
    outer_state = create_outer_fn(r2, outer_model, input_shape, cfg)
    if init_inner_params is not None:
        inner_state = inner_state.replace(params=init_inner_params)
    if init_outer_params is not None:
        outer_state = outer_state.replace(params=init_outer_params)

    xt, yt, te_counts = make_eval_batches(
        x_test, y_test, cfg.training.batch_size,
    )

    early_stopping_patience, restart_patience = _get_patience_cfg(cfg)
    best = {
        "test_acc": -1.0,
        "inner_params": None, "outer_params": None,
        "epoch": 0,
    }
    epochs_since_best = 0
    n_restarts = 0

    for ep in range(start_epoch, cfg.training.epochs):
        t0 = time.time()
        seed_tr = cfg.training.seed * 10_000 + ep
        rng, rng_epoch = jrandom.split(rng)

        if use_mmd:
            s_train = precompute_predictions(
                inner_state, x_train, cfg.training.batch_size,
                inner_forward_fn,
            )
            w_train = compute_mmd_weights(
                y_train, s_train, cfg.model.num_classes,
                cfg.mmd.smoothing_eps, cfg.mmd.w_max,
            )
            xb, yb, sb, wb = make_epoch_batches(
                x_train, y_train, cfg.training.batch_size, seed_tr,
                s_train, w_train,
            )
            inner_state, outer_state, metrics = train_epoch_pair_mmd_fn(
                inner_state, outer_state, xb, yb, sb, wb, rng_epoch,
                lamb, cfg.training.alpha, cfg.mmd.weight,
                num_classes=cfg.model.num_classes,
            )
        else:
            xb, yb = make_epoch_batches(
                x_train, y_train, cfg.training.batch_size, seed_tr,
            )
            inner_state, outer_state, metrics = train_epoch_pair_fn(
                inner_state, outer_state, xb, yb, rng_epoch,
                lamb, cfg.training.alpha, cfg.hsic.weight,
                num_classes=cfg.model.num_classes,
            )

        rng, rng_eval1, rng_eval2 = jrandom.split(rng, 3)
        te_loss1, te_acc1 = eval_epoch_fn(
            inner_state, xt, yt, rng_eval1, te_counts,
        )
        te_loss2, te_acc2 = eval_epoch_fn(
            outer_state, xt, yt, rng_eval2, te_counts,
        )

        # Best-checkpoint tracking (outer test accuracy)
        current_acc = float(te_acc2)
        if current_acc > best["test_acc"]:
            best["test_acc"] = current_acc
            best["inner_params"] = jax.tree.map(jnp.copy, inner_state.params)
            best["outer_params"] = jax.tree.map(jnp.copy, outer_state.params)
            best["epoch"] = ep + 1
            epochs_since_best = 0
            if run_dir is not None:
                save_checkpoint(inner_state.params,
                                checkpoint_path(run_dir, "inner_best.npz"))
                save_checkpoint(outer_state.params,
                                checkpoint_path(run_dir, "outer_best.npz"))
                save_checkpoint_meta(run_dir, ep + 1, best["test_acc"],
                                     best_checkpoint_epoch=ep + 1)
            print(f"    >>> NEW BEST outer_test_acc={current_acc:.4f} "
                  f"(epoch {ep+1})")
        else:
            epochs_since_best += 1

        mmd_val = float(metrics.get("mmd", 0.0))
        hsic_val = float(metrics.get("hsic", 0.0))
        penalty_label = "mmd" if use_mmd else "hsic"
        penalty_val = mmd_val if use_mmd else hsic_val

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
            "hsic": hsic_val,
            "mmd": mmd_val,
            "capacity_target_nats": float(lamb),
            "kl_target_gap_nats": float(lamb - metrics["kl_latent_nats"]),
            "best_test_acc": best["test_acc"],
            "best_epoch": best["epoch"],
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
            f" | {penalty_label} {penalty_val:.4f}"
            f" | {time.time()-t0:.2f}s"
        )

        # Patience check — restart both models together
        if early_stopping_patience > 0 and epochs_since_best >= early_stopping_patience:
            print(f"  Early stopping at epoch {ep+1} "
                  f"(no improvement for {early_stopping_patience} epochs)")
            break

        if (restart_patience > 0
                and best["inner_params"] is not None
                and epochs_since_best >= restart_patience):
            ri = jax.tree.map(jnp.copy, best["inner_params"])
            ro = jax.tree.map(jnp.copy, best["outer_params"])
            inner_state = inner_state.replace(
                params=ri, opt_state=inner_state.tx.init(ri),
            )
            outer_state = outer_state.replace(
                params=ro, opt_state=outer_state.tx.init(ro),
            )
            epochs_since_best = 0
            n_restarts += 1
            print(f"  RESTART #{n_restarts} at epoch {ep+1} "
                  f"-> best checkpoint (epoch {best['epoch']})")

    results["best_test_acc"] = best["test_acc"]
    results["best_epoch"] = best["epoch"]
    results["n_restarts"] = n_restarts

    if run_dir is not None:
        save_checkpoint(inner_state.params,
                        checkpoint_path(run_dir, "inner_final.npz"))
        save_checkpoint(outer_state.params,
                        checkpoint_path(run_dir, "outer_final.npz"))
        save_checkpoint_meta(run_dir, ep + 1, best["test_acc"],
                             best_checkpoint_epoch=best["epoch"])
        save_results(run_dir, results)

    results["train_acc"] = results["train_acc2"]
    results["test_acc"] = best["test_acc"]
    results["lambda"] = float(lamb)
    return results


# ---- Oracle pair-model runner ----

def run_train_eval_oracle_pair(x_train, y_train, x_test, y_test,
                               inner_model, outer_model, cfg, lamb,
                               wandb_run, *, create_inner_fn,
                               create_outer_fn, train_epoch_pair_fn,
                               eval_epoch_fn, run_dir=None,
                               start_epoch=0, init_inner_params=None,
                               init_outer_params=None,
                               inner_forward_fn=None,
                               train_epoch_pair_mmd_fn=None):
    """Oracle pair training loop: frozen inner, trainable outer."""
    use_mmd = cfg.model.outer_loss == "mmd"
    rng = jrandom.PRNGKey(cfg.training.seed)
    input_shape = (cfg.training.batch_size,) + x_train.shape[1:]
    rng, r1, r2 = jrandom.split(rng, 3)

    inner_state = create_inner_fn(r1, inner_model, input_shape, cfg)
    outer_state = create_outer_fn(r2, outer_model, input_shape, cfg)
    if init_inner_params is not None:
        inner_state = inner_state.replace(params=init_inner_params)
    if init_outer_params is not None:
        outer_state = outer_state.replace(params=init_outer_params)

    xt, yt, te_counts = make_eval_batches(
        x_test, y_test, cfg.training.batch_size,
    )

    early_stopping_patience, restart_patience = _get_patience_cfg(cfg)
    best = {
        "test_acc": -1.0,
        "outer_params": None,
        "epoch": 0,
    }
    epochs_since_best = 0
    n_restarts = 0

    # Oracle inner is frozen — S and W are constant across epochs.
    if use_mmd:
        s_train = precompute_predictions(
            inner_state, x_train, cfg.training.batch_size,
            inner_forward_fn,
        )
        w_train = compute_mmd_weights(
            y_train, s_train, cfg.model.num_classes,
            cfg.mmd.smoothing_eps, cfg.mmd.w_max,
        )

    for ep in range(start_epoch, cfg.training.epochs):
        t0 = time.time()
        seed_tr = cfg.training.seed * 10_000 + ep
        rng, rng_epoch = jrandom.split(rng)

        if use_mmd:
            xb, yb, sb, wb = make_epoch_batches(
                x_train, y_train, cfg.training.batch_size, seed_tr,
                s_train, w_train,
            )
            inner_state, outer_state, metrics = train_epoch_pair_mmd_fn(
                inner_state, outer_state, xb, yb, sb, wb, rng_epoch,
                lamb, cfg.training.alpha, cfg.mmd.weight,
                num_classes=cfg.model.num_classes,
            )
        else:
            xb, yb = make_epoch_batches(
                x_train, y_train, cfg.training.batch_size, seed_tr,
            )
            inner_state, outer_state, metrics = train_epoch_pair_fn(
                inner_state, outer_state, xb, yb, rng_epoch,
                lamb, cfg.training.alpha, cfg.hsic.weight,
                num_classes=cfg.model.num_classes,
            )

        rng, rng_eval1, rng_eval2 = jrandom.split(rng, 3)
        te_loss1, te_acc1 = eval_epoch_fn(
            inner_state, xt, yt, rng_eval1, te_counts,
        )
        te_loss2, te_acc2 = eval_epoch_fn(
            outer_state, xt, yt, rng_eval2, te_counts,
        )

        # Best-checkpoint tracking (outer test accuracy only)
        current_acc = float(te_acc2)
        if current_acc > best["test_acc"]:
            best["test_acc"] = current_acc
            best["outer_params"] = jax.tree.map(jnp.copy, outer_state.params)
            best["epoch"] = ep + 1
            epochs_since_best = 0
            if run_dir is not None:
                save_checkpoint(outer_state.params,
                                checkpoint_path(run_dir, "outer_best.npz"))
                save_checkpoint_meta(run_dir, ep + 1, best["test_acc"],
                                     best_checkpoint_epoch=ep + 1)
            print(f"    >>> NEW BEST outer_test_acc={current_acc:.4f} "
                  f"(epoch {ep+1})")
        else:
            epochs_since_best += 1

        mmd_val = float(metrics.get("mmd", 0.0))
        hsic_val = float(metrics.get("hsic", 0.0))
        penalty_label = "mmd" if use_mmd else "hsic"
        penalty_val = mmd_val if use_mmd else hsic_val

        results = {
            "epoch": ep + 1,
            "train_acc1": float(metrics["accuracy_inner"]),
            "test_acc1": float(te_acc1),
            "test_nll_nats_inner": float(te_loss1),
            "train_acc2": float(metrics["accuracy_outer"]),
            "test_acc2": float(te_acc2),
            "objective_total_outer_nats": float(metrics["objective_total_outer_nats"]),
            "data_nll_outer_nats": float(metrics["data_nll_outer_nats"]),
            "test_nll_nats_outer": float(te_loss2),
            "hsic": hsic_val,
            "mmd": mmd_val,
            "best_test_acc": best["test_acc"],
            "best_epoch": best["epoch"],
        }
        wandb_run.log(results)

        print(
            f"  Epoch {ep+1}/{cfg.training.epochs}"
            f" | oracle_acc {results['train_acc1']:.4f}"
            f" (test {results['test_acc1']:.4f})"
            f" | outer_train_acc {results['train_acc2']:.4f}"
            f" outer_test_acc {results['test_acc2']:.4f}"
            f" | outer_ce {results['data_nll_outer_nats']:.4f}n"
            f" | {penalty_label} {penalty_val:.4f}"
            f" | {time.time()-t0:.2f}s"
        )

        # Patience check — only restart outer (inner is frozen)
        if early_stopping_patience > 0 and epochs_since_best >= early_stopping_patience:
            print(f"  Early stopping at epoch {ep+1} "
                  f"(no improvement for {early_stopping_patience} epochs)")
            break

        if (restart_patience > 0
                and best["outer_params"] is not None
                and epochs_since_best >= restart_patience):
            ro = jax.tree.map(jnp.copy, best["outer_params"])
            outer_state = outer_state.replace(
                params=ro, opt_state=outer_state.tx.init(ro),
            )
            epochs_since_best = 0
            n_restarts += 1
            print(f"  RESTART #{n_restarts} at epoch {ep+1} "
                  f"-> best checkpoint (epoch {best['epoch']})")

    results["best_test_acc"] = best["test_acc"]
    results["best_epoch"] = best["epoch"]
    results["n_restarts"] = n_restarts

    if run_dir is not None:
        save_checkpoint(outer_state.params,
                        checkpoint_path(run_dir, "outer_final.npz"))
        save_checkpoint_meta(run_dir, ep + 1, best["test_acc"],
                             best_checkpoint_epoch=best["epoch"])
        save_results(run_dir, results)

    results["train_acc"] = results["train_acc2"]
    results["test_acc"] = best["test_acc"]
    results["lambda"] = float(lamb)
    return results


# ---- MDL single-model runner ----

def run_train_eval_mdl(x_train, y_train, x_test, y_test, model, cfg, lamb,
                       wandb_run, *, create_state_fn,
                       train_epoch_fn,
                       eval_epoch_fn, run_dir=None,
                       start_epoch=0, init_params=None):
    """MDL single-model training loop with tau annealing."""
    base_rng = jrandom.PRNGKey(cfg.training.seed)
    rng_init_inner = jrandom.fold_in(base_rng, 0)
    rng_train = jrandom.fold_in(base_rng, 2)
    rng_eval = jrandom.fold_in(base_rng, 3)
    input_shape = (cfg.training.batch_size,) + x_train.shape[1:]

    state = create_state_fn(rng_init_inner, model, input_shape, cfg)
    if init_params is not None:
        state = state.replace(params=init_params)
    n_train = x_train.shape[0]

    xt, yt, te_counts = make_eval_batches(
        x_test, y_test, cfg.training.batch_size,
    )

    early_stopping_patience, restart_patience = _get_patience_cfg(cfg)
    best = {"test_acc": -1.0, "params": None, "epoch": 0}
    epochs_since_best = 0
    n_restarts = 0

    for ep in range(start_epoch, cfg.training.epochs):
        t0 = time.time()

        tau = anneal_tau(ep, cfg.training.epochs, cfg.mdl.tau_start, cfg.mdl.tau_end)
        state = state.replace(tau=jnp.array(tau, dtype=jnp.float32))

        seed_tr = cfg.training.seed * 10_000 + ep
        rng_train, rng_epoch = jrandom.split(rng_train)

        xb, yb = make_epoch_batches(
            x_train, y_train, cfg.training.batch_size, seed_tr,
        )

        state, metrics = train_epoch_fn(
            state, xb, yb, rng_epoch, lamb, n_train,
        )

        rng_eval, rng_eval_epoch = jrandom.split(rng_eval)
        te_loss, te_acc = eval_epoch_fn(state, xt, yt, rng_eval_epoch, te_counts)

        # Best-checkpoint tracking
        current_acc = float(te_acc)
        if current_acc > best["test_acc"]:
            best["test_acc"] = current_acc
            best["params"] = jax.tree.map(jnp.copy, state.params)
            best["epoch"] = ep + 1
            epochs_since_best = 0
            if run_dir is not None:
                save_checkpoint(state.params,
                                checkpoint_path(run_dir, "best.npz"))
                save_checkpoint_meta(run_dir, ep + 1, best["test_acc"],
                                     best_checkpoint_epoch=ep + 1)
            print(f"    >>> NEW BEST test_acc={current_acc:.4f} "
                  f"(epoch {ep+1})")
        else:
            epochs_since_best += 1

        results = {
            "epoch": ep + 1,
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
            "best_test_acc": best["test_acc"],
            "best_epoch": best["epoch"],
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
            f"  {time.time()-t0:.2f}s"
        )

        state, epochs_since_best, n_restarts, should_stop = _check_patience(
            state, best, epochs_since_best, n_restarts,
            early_stopping_patience, restart_patience, ep,
        )
        if should_stop:
            break

    results["test_acc"] = best["test_acc"]
    results["best_test_acc"] = best["test_acc"]
    results["best_epoch"] = best["epoch"]
    results["n_restarts"] = n_restarts

    if run_dir is not None:
        save_checkpoint(state.params, checkpoint_path(run_dir, "final.npz"))
        save_checkpoint_meta(run_dir, ep + 1, best["test_acc"],
                             best_checkpoint_epoch=best["epoch"])
        save_results(run_dir, results)

    return results


# ---- MDL pair-model runner ----

def run_train_eval_mdl_pair(x_train, y_train, x_test, y_test, inner_model,
                            outer_model, cfg, lamb, wandb_run,
                            *, create_inner_fn, create_outer_fn,
                            train_epoch_fn,
                            eval_inner_epoch_fn, eval_outer_epoch_fn,
                            run_dir=None,
                            start_epoch=0, init_inner_params=None,
                            init_outer_params=None,
                            inner_forward_fn=None,
                            train_epoch_mmd_fn=None):
    """MDL dual-model (MDL inner + standard outer) training loop."""
    use_mmd = cfg.model.outer_loss == "mmd"
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
    if init_inner_params is not None:
        inner_state = inner_state.replace(params=init_inner_params)
    if init_outer_params is not None:
        outer_state = outer_state.replace(params=init_outer_params)
    n_train = x_train.shape[0]

    xt, yt, te_counts = make_eval_batches(
        x_test, y_test, cfg.training.batch_size,
    )

    early_stopping_patience, restart_patience = _get_patience_cfg(cfg)
    best = {
        "test_acc": -1.0,
        "inner_params": None, "outer_params": None,
        "epoch": 0,
    }
    epochs_since_best = 0
    n_restarts = 0

    for ep in range(start_epoch, cfg.training.epochs):
        t0 = time.time()

        tau = anneal_tau(ep, cfg.training.epochs, cfg.mdl.tau_start, cfg.mdl.tau_end)
        inner_state = inner_state.replace(
            tau=jnp.array(tau, dtype=jnp.float32),
        )

        seed_tr = cfg.training.seed * 10_000 + ep
        rng_train, rng_epoch = jrandom.split(rng_train)

        if use_mmd:
            s_train = precompute_predictions(
                inner_state, x_train, cfg.training.batch_size,
                inner_forward_fn,
            )
            w_train = compute_mmd_weights(
                y_train, s_train, cfg.model.num_classes,
                cfg.mmd.smoothing_eps, cfg.mmd.w_max,
            )
            xb, yb, sb, wb = make_epoch_batches(
                x_train, y_train, cfg.training.batch_size, seed_tr,
                s_train, w_train,
            )
            inner_state, outer_state, metrics = train_epoch_mmd_fn(
                inner_state, outer_state, xb, yb, sb, wb, rng_epoch,
                lamb, n_train, cfg.mmd.weight,
                num_classes=cfg.model.num_classes,
            )
        else:
            xb, yb = make_epoch_batches(
                x_train, y_train, cfg.training.batch_size, seed_tr,
            )
            inner_state, outer_state, metrics = train_epoch_fn(
                inner_state, outer_state, xb, yb, rng_epoch,
                lamb, n_train, cfg.hsic.weight,
                num_classes=cfg.model.num_classes,
            )

        rng_eval, rng_eval_inner = jrandom.split(rng_eval)
        rng_eval_outer = jrandom.fold_in(rng_eval_inner, 1)
        te_loss1, te_acc1 = eval_inner_epoch_fn(
            inner_state, xt, yt, rng_eval_inner, te_counts,
        )
        te_loss2, te_acc2 = eval_outer_epoch_fn(
            outer_state, xt, yt, rng_eval_outer, te_counts,
        )

        # Best-checkpoint tracking (outer test accuracy)
        current_acc = float(te_acc2)
        if current_acc > best["test_acc"]:
            best["test_acc"] = current_acc
            best["inner_params"] = jax.tree.map(jnp.copy, inner_state.params)
            best["outer_params"] = jax.tree.map(jnp.copy, outer_state.params)
            best["epoch"] = ep + 1
            epochs_since_best = 0
            if run_dir is not None:
                save_checkpoint(inner_state.params,
                                checkpoint_path(run_dir, "inner_best.npz"))
                save_checkpoint(outer_state.params,
                                checkpoint_path(run_dir, "outer_best.npz"))
                save_checkpoint_meta(run_dir, ep + 1, best["test_acc"],
                                     best_checkpoint_epoch=ep + 1)
            print(f"    >>> NEW BEST outer_test_acc={current_acc:.4f} "
                  f"(epoch {ep+1})")
        else:
            epochs_since_best += 1

        results = {
            "epoch": ep + 1,
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
            "hsic": float(metrics.get("hsic", 0.0)),
            "mmd": float(metrics.get("mmd", 0.0)),
            "best_test_acc": best["test_acc"],
            "best_epoch": best["epoch"],
        }
        wandb_run.log(results)

        penalty_label = "mmd" if use_mmd else "hsic"
        penalty_val = results["mmd"] if use_mmd else results["hsic"]
        print(
            f"  Epoch {ep+1}/{cfg.training.epochs}"
            f" | inner_train_acc {results['train_acc1']:.4f}"
            f" inner_test_acc {results['test_acc1']:.4f}"
            f" | inner_objective_total {results['objective_total_nats']:.4f}n"
            f" inner_data_nll {results['data_nll_nats']:.4f}n"
            f" inner_reg_net {results['reg_net_nats']:.4f}n"
            f" | outer_train_acc {results['train_acc2']:.4f}"
            f" outer_test_acc {results['test_acc2']:.4f}"
            f" | {penalty_label} {penalty_val:.4f}"
            f" | tau {float(tau):.4f}"
            f" | {time.time()-t0:.2f}s"
        )

        # Patience check — restart both models together
        if early_stopping_patience > 0 and epochs_since_best >= early_stopping_patience:
            print(f"  Early stopping at epoch {ep+1} "
                  f"(no improvement for {early_stopping_patience} epochs)")
            break

        if (restart_patience > 0
                and best["inner_params"] is not None
                and epochs_since_best >= restart_patience):
            ri = jax.tree.map(jnp.copy, best["inner_params"])
            ro = jax.tree.map(jnp.copy, best["outer_params"])
            inner_state = inner_state.replace(
                params=ri, opt_state=inner_state.tx.init(ri),
            )
            outer_state = outer_state.replace(
                params=ro, opt_state=outer_state.tx.init(ro),
            )
            epochs_since_best = 0
            n_restarts += 1
            print(f"  RESTART #{n_restarts} at epoch {ep+1} "
                  f"-> best checkpoint (epoch {best['epoch']})")

    results["best_test_acc"] = best["test_acc"]
    results["best_epoch"] = best["epoch"]
    results["n_restarts"] = n_restarts

    if run_dir is not None:
        save_checkpoint(inner_state.params,
                        checkpoint_path(run_dir, "inner_final.npz"))
        save_checkpoint(outer_state.params,
                        checkpoint_path(run_dir, "outer_final.npz"))
        save_checkpoint_meta(run_dir, ep + 1, best["test_acc"],
                             best_checkpoint_epoch=best["epoch"])
        save_results(run_dir, results)

    results["train_acc"] = results["train_acc2"]
    results["test_acc"] = best["test_acc"]
    results["lambda"] = float(lamb)
    return results
