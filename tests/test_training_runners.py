"""Tests for training runner result schemas."""

from types import SimpleNamespace

import jax.numpy as jnp
import optax
import pytest
from flax.training import train_state

from src.training.runners import (
    run_train_eval, run_train_eval_pair, run_train_eval_oracle_pair,
)


class _DummyWandbRun:
    def __init__(self):
        self.logged = []

    def log(self, payload):
        self.logged.append(payload)


def _ckpt_cfg(es=0, rs=0):
    return SimpleNamespace(
        early_stopping_patience=es,
        restart_patience=rs,
    )


def _make_single_cfg(**ckpt_kw):
    return SimpleNamespace(
        training=SimpleNamespace(
            seed=0,
            batch_size=2,
            epochs=1,
            alpha=0.25,
        ),
        checkpointing=_ckpt_cfg(**ckpt_kw),
    )


def _make_pair_cfg(**ckpt_kw):
    return SimpleNamespace(
        training=SimpleNamespace(
            seed=0,
            batch_size=2,
            epochs=1,
            alpha=0.25,
        ),
        hsic=SimpleNamespace(weight=0.5),
        model=SimpleNamespace(num_classes=2),
        checkpointing=_ckpt_cfg(**ckpt_kw),
    )


# ---- Existing schema tests (updated for new fields) ----

def test_run_train_eval_uses_explicit_vib_metric_names():
    """Single-model VIB runner should expose explicit result keys."""
    cfg = _make_single_cfg()
    run = _DummyWandbRun()
    x = jnp.zeros((4, 1), dtype=jnp.float32)
    y = jnp.zeros((4,), dtype=jnp.int32)

    def create_state_fn(rng, model, input_shape, cfg_):
        return SimpleNamespace(params={"w": jnp.array(1.0)})

    def train_epoch_fn(state, xb, yb, rng, lamb, alpha):
        metrics = {
            "accuracy": jnp.array(0.50),
            "objective_total_nats": jnp.array(1.20),
            "data_nll_nats": jnp.array(0.80),
            "reconstruction_nll_nats": jnp.array(0.30),
            "kl_latent_nats": jnp.array(0.40),
            "beta_controller": jnp.array(0.70),
        }
        return state, metrics

    def eval_epoch_fn(state, xb, yb, rng, counts=None):
        return jnp.array(0.90), jnp.array(0.60)

    results = run_train_eval(
        x, y, x, y,
        model=None,
        cfg=cfg,
        lamb=0.1,
        wandb_run=run,
        create_state_fn=create_state_fn,
        train_epoch_fn=train_epoch_fn,
        eval_epoch_fn=eval_epoch_fn,
        run_dir=None,
    )

    assert results["train_acc"] == pytest.approx(0.5)
    assert results["test_acc"] == pytest.approx(0.6)
    assert results["objective_total_nats"] == pytest.approx(1.2)
    assert results["data_nll_nats"] == pytest.approx(0.8)
    assert results["reconstruction_nll_nats"] == pytest.approx(0.3)
    assert results["kl_latent_nats"] == pytest.approx(0.4)
    assert results["beta_controller"] == pytest.approx(0.7)
    assert results["test_nll_nats"] == pytest.approx(0.9)
    assert results["capacity_target_nats"] == pytest.approx(0.1)
    assert results["kl_target_gap_nats"] == pytest.approx(0.1 - 0.4)
    assert "train_loss" not in results
    assert "test_loss" not in results
    assert "cap" not in results
    assert run.logged and "objective_total_nats" in run.logged[0]

    # New best-tracking fields
    assert results["best_test_acc"] == pytest.approx(0.6)
    assert results["best_epoch"] == 1
    assert results["n_restarts"] == 0


def test_run_train_eval_pair_uses_explicit_vib_hsic_metric_names():
    """Pair-model VIB/HSIC runner should expose explicit result keys."""
    cfg = _make_pair_cfg()
    run = _DummyWandbRun()
    x = jnp.zeros((4, 1), dtype=jnp.float32)
    y = jnp.zeros((4,), dtype=jnp.int32)

    def create_inner_fn(rng, model, input_shape, cfg_):
        return SimpleNamespace(params={"w": jnp.array(1.0)}, role="inner")

    def create_outer_fn(rng, model, input_shape, cfg_):
        return SimpleNamespace(params={"w": jnp.array(2.0)}, role="outer")

    def train_epoch_pair_fn(inner_state, outer_state, xb, yb, rng, lamb,
                            alpha, hsic_w, num_classes):
        metrics = {
            "accuracy_inner": jnp.array(0.55),
            "objective_total_nats": jnp.array(1.30),
            "data_nll_nats": jnp.array(0.85),
            "reconstruction_nll_nats": jnp.array(0.25),
            "kl_latent_nats": jnp.array(0.60),
            "beta_controller": jnp.array(0.90),
            "accuracy_outer": jnp.array(0.70),
            "objective_total_outer_nats": jnp.array(0.45),
            "data_nll_outer_nats": jnp.array(0.35),
            "hsic": jnp.array(0.15),
        }
        return inner_state, outer_state, metrics

    def eval_epoch_fn(state, xb, yb, rng, counts=None):
        if state.role == "inner":
            return jnp.array(0.95), jnp.array(0.58)
        return jnp.array(0.40), jnp.array(0.72)

    results = run_train_eval_pair(
        x, y, x, y,
        inner_model=None,
        outer_model=None,
        cfg=cfg,
        lamb=0.2,
        wandb_run=run,
        create_inner_fn=create_inner_fn,
        create_outer_fn=create_outer_fn,
        train_epoch_pair_fn=train_epoch_pair_fn,
        eval_epoch_fn=eval_epoch_fn,
        run_dir=None,
    )

    assert results["train_acc1"] == pytest.approx(0.55)
    assert results["test_acc1"] == pytest.approx(0.58)
    assert results["objective_total_nats"] == pytest.approx(1.3)
    assert results["data_nll_nats"] == pytest.approx(0.85)
    assert results["reconstruction_nll_nats"] == pytest.approx(0.25)
    assert results["kl_latent_nats"] == pytest.approx(0.6)
    assert results["beta_controller"] == pytest.approx(0.9)
    assert results["test_nll_nats_inner"] == pytest.approx(0.95)
    assert results["train_acc2"] == pytest.approx(0.7)
    assert results["test_acc2"] == pytest.approx(0.72)
    assert results["objective_total_outer_nats"] == pytest.approx(0.45)
    assert results["data_nll_outer_nats"] == pytest.approx(0.35)
    assert results["test_nll_nats_outer"] == pytest.approx(0.4)
    assert results["hsic"] == pytest.approx(0.15)
    assert results["capacity_target_nats"] == pytest.approx(0.2)
    assert results["kl_target_gap_nats"] == pytest.approx(0.2 - 0.6)
    assert results["train_acc"] == pytest.approx(0.7)
    assert results["test_acc"] == pytest.approx(0.72)
    assert results["lambda"] == pytest.approx(0.2)
    assert "train_loss1" not in results
    assert "train_loss2" not in results
    assert "train_ce2" not in results
    assert "cap" not in results
    assert run.logged and "objective_total_outer_nats" in run.logged[0]

    # New best-tracking fields
    assert results["best_test_acc"] == pytest.approx(0.72)
    assert results["best_epoch"] == 1
    assert results["n_restarts"] == 0


# ---- Tests for new training loop features ----

def _make_real_state(lr=1e-3):
    """Create a real Flax TrainState for patience/restart tests."""
    params = {"w": jnp.ones((2, 2))}
    tx = optax.adamw(lr)
    return train_state.TrainState(
        step=0,
        apply_fn=lambda *a, **kw: None,
        params=params,
        tx=tx,
        opt_state=tx.init(params),
    )


def test_early_stopping_breaks_loop():
    """Runner should stop early when patience is exceeded."""
    cfg = _make_single_cfg(es=2)
    cfg.training.epochs = 10
    run = _DummyWandbRun()
    x = jnp.zeros((4, 1), dtype=jnp.float32)
    y = jnp.zeros((4,), dtype=jnp.int32)

    call_count = [0]

    def create_state_fn(rng, model, input_shape, cfg_):
        return _make_real_state()

    def train_epoch_fn(state, xb, yb, rng, lamb, alpha):
        call_count[0] += 1
        metrics = {
            "accuracy": jnp.array(0.50),
            "objective_total_nats": jnp.array(1.0),
            "data_nll_nats": jnp.array(0.5),
            "reconstruction_nll_nats": jnp.array(0.2),
            "kl_latent_nats": jnp.array(0.3),
            "beta_controller": jnp.array(0.5),
        }
        return state, metrics

    # Accuracy peaks at epoch 1 then drops — should stop at epoch 3
    acc_seq = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
    acc_idx = [0]

    def eval_epoch_fn(state, xb, yb, rng, counts=None):
        acc = acc_seq[min(acc_idx[0], len(acc_seq) - 1)]
        acc_idx[0] += 1
        return jnp.array(0.5), jnp.array(acc)

    results = run_train_eval(
        x, y, x, y,
        model=None, cfg=cfg, lamb=0.1, wandb_run=run,
        create_state_fn=create_state_fn,
        train_epoch_fn=train_epoch_fn,
        eval_epoch_fn=eval_epoch_fn,
        run_dir=None,
    )

    # Should have trained 3 epochs: best at 1, patience exceeded at 3
    assert call_count[0] == 3
    assert results["best_test_acc"] == pytest.approx(0.8)
    assert results["best_epoch"] == 1


def test_restart_with_patience_reloads_best_params():
    """Runner should reload best params and reset optimizer on restart."""
    cfg = _make_single_cfg(rs=2)
    cfg.training.epochs = 6
    run = _DummyWandbRun()
    x = jnp.zeros((4, 1), dtype=jnp.float32)
    y = jnp.zeros((4,), dtype=jnp.int32)

    param_snapshots = []

    def create_state_fn(rng, model, input_shape, cfg_):
        return _make_real_state()

    def train_epoch_fn(state, xb, yb, rng, lamb, alpha):
        # Modify params each epoch so we can detect restart
        new_params = {"w": state.params["w"] + 0.1}
        state = state.replace(params=new_params)
        metrics = {
            "accuracy": jnp.array(0.50),
            "objective_total_nats": jnp.array(1.0),
            "data_nll_nats": jnp.array(0.5),
            "reconstruction_nll_nats": jnp.array(0.2),
            "kl_latent_nats": jnp.array(0.3),
            "beta_controller": jnp.array(0.5),
        }
        return state, metrics

    # Accuracy: high at epoch 1, then drops — restart should fire at epoch 3
    acc_seq = [0.9, 0.5, 0.4, 0.3, 0.2, 0.1]
    acc_idx = [0]

    def eval_epoch_fn(state, xb, yb, rng, counts=None):
        acc = acc_seq[min(acc_idx[0], len(acc_seq) - 1)]
        acc_idx[0] += 1
        param_snapshots.append(float(state.params["w"].ravel()[0]))
        return jnp.array(0.5), jnp.array(acc)

    results = run_train_eval(
        x, y, x, y,
        model=None, cfg=cfg, lamb=0.1, wandb_run=run,
        create_state_fn=create_state_fn,
        train_epoch_fn=train_epoch_fn,
        eval_epoch_fn=eval_epoch_fn,
        run_dir=None,
    )

    assert results["n_restarts"] >= 1
    assert results["best_test_acc"] == pytest.approx(0.9)

    # After restart, params should drop back toward the best value (1.1)
    # Before restart: 1.1, 1.2, 1.3 (restart fires here)
    # After restart: params reset to 1.1, then train adds 0.1 -> 1.2, ...
    # So param_snapshots[3] should be ~1.2 (best=1.1 + one train step)
    assert param_snapshots[3] == pytest.approx(1.2, abs=0.01)


def test_oracle_pair_runner_schema():
    """Oracle pair runner should expose oracle-specific result keys."""
    cfg = _make_pair_cfg()
    run = _DummyWandbRun()
    x = jnp.zeros((4, 1), dtype=jnp.float32)
    y = jnp.zeros((4,), dtype=jnp.int32)

    def create_inner_fn(rng, model, input_shape, cfg_):
        return SimpleNamespace(params={"w": jnp.array(1.0)})

    def create_outer_fn(rng, model, input_shape, cfg_):
        return SimpleNamespace(params={"w": jnp.array(2.0)})

    def train_epoch_pair_fn(inner_state, outer_state, xb, yb, rng, lamb,
                            alpha, hsic_w, num_classes):
        metrics = {
            "accuracy_inner": jnp.array(0.95),
            "accuracy_outer": jnp.array(0.70),
            "objective_total_outer_nats": jnp.array(0.45),
            "data_nll_outer_nats": jnp.array(0.35),
            "hsic": jnp.array(0.15),
        }
        return inner_state, outer_state, metrics

    def eval_epoch_fn(state, xb, yb, rng, counts=None):
        return jnp.array(0.40), jnp.array(0.72)

    results = run_train_eval_oracle_pair(
        x, y, x, y,
        inner_model=None,
        outer_model=None,
        cfg=cfg,
        lamb=0.2,
        wandb_run=run,
        create_inner_fn=create_inner_fn,
        create_outer_fn=create_outer_fn,
        train_epoch_pair_fn=train_epoch_pair_fn,
        eval_epoch_fn=eval_epoch_fn,
        run_dir=None,
    )

    assert results["train_acc1"] == pytest.approx(0.95)
    assert results["test_acc1"] == pytest.approx(0.72)
    assert results["train_acc2"] == pytest.approx(0.70)
    assert results["test_acc2"] == pytest.approx(0.72)
    assert results["objective_total_outer_nats"] == pytest.approx(0.45)
    assert results["data_nll_outer_nats"] == pytest.approx(0.35)
    assert results["hsic"] == pytest.approx(0.15)
    assert results["train_acc"] == pytest.approx(0.70)
    assert results["test_acc"] == pytest.approx(0.72)
    assert results["lambda"] == pytest.approx(0.2)
    assert results["best_test_acc"] == pytest.approx(0.72)
    assert results["best_epoch"] == 1
    assert results["n_restarts"] == 0
    # Should NOT have VIB-specific keys
    assert "kl_latent_nats" not in results
    assert "reconstruction_nll_nats" not in results
    assert "beta_controller" not in results
    assert run.logged and "hsic" in run.logged[0]


def test_best_tracking_across_improving_epochs():
    """Best-tracking should update when accuracy improves."""
    cfg = _make_single_cfg()
    cfg.training.epochs = 3
    run = _DummyWandbRun()
    x = jnp.zeros((4, 1), dtype=jnp.float32)
    y = jnp.zeros((4,), dtype=jnp.int32)

    def create_state_fn(rng, model, input_shape, cfg_):
        return SimpleNamespace(params={"w": jnp.array(1.0)})

    def train_epoch_fn(state, xb, yb, rng, lamb, alpha):
        metrics = {
            "accuracy": jnp.array(0.50),
            "objective_total_nats": jnp.array(1.0),
            "data_nll_nats": jnp.array(0.5),
            "reconstruction_nll_nats": jnp.array(0.2),
            "kl_latent_nats": jnp.array(0.3),
            "beta_controller": jnp.array(0.5),
        }
        return state, metrics

    acc_seq = [0.6, 0.7, 0.8]
    acc_idx = [0]

    def eval_epoch_fn(state, xb, yb, rng, counts=None):
        acc = acc_seq[acc_idx[0]]
        acc_idx[0] += 1
        return jnp.array(0.5), jnp.array(acc)

    results = run_train_eval(
        x, y, x, y,
        model=None, cfg=cfg, lamb=0.1, wandb_run=run,
        create_state_fn=create_state_fn,
        train_epoch_fn=train_epoch_fn,
        eval_epoch_fn=eval_epoch_fn,
        run_dir=None,
    )

    assert results["best_test_acc"] == pytest.approx(0.8)
    assert results["best_epoch"] == 3
    assert results["n_restarts"] == 0
