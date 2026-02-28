"""Tests for training runner result schemas."""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from src.training.runners import run_train_eval, run_train_eval_pair


class _DummyWandbRun:
    def __init__(self):
        self.logged = []

    def log(self, payload):
        self.logged.append(payload)


def _make_single_cfg():
    return SimpleNamespace(
        training=SimpleNamespace(
            seed=0,
            batch_size=2,
            epochs=1,
            alpha=0.25,
        ),
    )


def _make_pair_cfg():
    return SimpleNamespace(
        training=SimpleNamespace(
            seed=0,
            batch_size=2,
            epochs=1,
            alpha=0.25,
        ),
        hsic=SimpleNamespace(weight=0.5),
        model=SimpleNamespace(num_classes=2),
    )


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

    def eval_epoch_fn(state, xb, yb, rng):
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

    def eval_epoch_fn(state, xb, yb, rng):
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
