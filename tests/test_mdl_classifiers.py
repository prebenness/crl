"""Tests for GumbelSoftmaxMLP and MDL training integration.

Covers:
    - Forward pass shapes for all 3 modes (soft, Gumbel-ST, argmax)
    - Aux dict contains expected keys with correct shapes
    - MDL loss computation produces finite values
    - Single training step runs and produces sensible metrics
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random as jrandom
import numpy as np

from src.mdl.coding import grid_values_and_codelengths
from src.models.mdl_classifiers import GumbelSoftmaxMLP, kaiming_categorical_init


@pytest.fixture
def grid():
    """Small rational grid for tests."""
    return grid_values_and_codelengths(n_max=3, m_max=3)


@pytest.fixture
def model(grid):
    gv, gc = grid
    return GumbelSoftmaxMLP(
        num_classes=10,
        grid_values=gv,
        grid_codelengths=gc,
        h1=16,
        bottleneck=8,
        h3=16,
    )


@pytest.fixture
def dummy_input():
    """Batch of 4 fake 28x28x3 images."""
    return jnp.ones((4, 28, 28, 3), dtype=jnp.float32)


@pytest.fixture
def params(model, dummy_input):
    rng = jrandom.PRNGKey(0)
    return model.init(rng, dummy_input, tau=1.0, train=False)["params"]


class TestGumbelSoftmaxMLPForward:
    """Verify forward pass shapes for all three modes."""

    def test_soft_forward_shapes(self, model, params, dummy_input):
        logits, aux = model.apply(
            {"params": params}, dummy_input,
            tau=1.0, train=True, rng=jrandom.PRNGKey(1),
            soft_forward=True,
        )
        assert logits.shape == (4, 10)
        assert aux["z"].shape == (4, 8)  # bottleneck dim
        assert aux["mu"].shape == (4, 8)

    def test_gumbel_st_shapes(self, model, params, dummy_input):
        logits, aux = model.apply(
            {"params": params}, dummy_input,
            tau=1.0, train=True, rng=jrandom.PRNGKey(2),
        )
        assert logits.shape == (4, 10)
        assert aux["z"].shape == (4, 8)

    def test_argmax_shapes(self, model, params, dummy_input):
        logits, aux = model.apply(
            {"params": params}, dummy_input,
            tau=1.0, train=False,
        )
        assert logits.shape == (4, 10)
        assert aux["z"].shape == (4, 8)

    def test_deterministic_eval_is_consistent(self, model, params, dummy_input):
        logits1, _ = model.apply(
            {"params": params}, dummy_input, tau=1.0, train=False,
        )
        logits2, _ = model.apply(
            {"params": params}, dummy_input, tau=1.0, train=False,
        )
        np.testing.assert_array_equal(logits1, logits2)


class TestAuxOutputs:
    """Verify aux dict contains expected keys with valid values."""

    def test_expected_codelength_positive(self, model, params, dummy_input):
        _, aux = model.apply(
            {"params": params}, dummy_input,
            tau=1.0, train=True, rng=jrandom.PRNGKey(3),
            soft_forward=True,
        )
        assert "expected_codelength" in aux
        assert float(aux["expected_codelength"]) > 0

    def test_all_probs_shape_and_sum(self, model, params, dummy_input, grid):
        gv, _ = grid
        M = len(gv)

        _, aux = model.apply(
            {"params": params}, dummy_input,
            tau=1.0, train=True, rng=jrandom.PRNGKey(4),
            soft_forward=True,
        )
        all_probs = aux["all_probs"]
        assert all_probs.ndim == 2
        assert all_probs.shape[1] == M
        # Each row should sum to ~1
        row_sums = jnp.sum(all_probs, axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


class TestKaimingCategoricalInit:
    """Verify the Kaiming categorical initializer produces non-degenerate weights."""

    def test_init_produces_peaked_distributions(self, model, params):
        logits = params["logits"]
        # Every row should have a dominant peak (peak_logit=10.0)
        max_vals = jnp.max(logits, axis=-1)
        assert float((max_vals > 1.0).mean()) > 0.99

    def test_soft_forward_not_near_zero(self, model, params, dummy_input):
        logits, _ = model.apply(
            {"params": params}, dummy_input,
            tau=2.0, train=True, rng=jrandom.PRNGKey(10),
            soft_forward=True,
        )
        # With Kaiming init, soft forward should produce non-trivial logits
        assert float(jnp.abs(logits).max()) > 0.1

    def test_warmup_gradient_nonzero(self, model, params, dummy_input):
        y = jnp.array([0, 1, 2, 3])

        def loss_fn(p):
            logits, _ = model.apply(
                {"params": p}, dummy_input,
                tau=2.0, train=True, rng=jrandom.PRNGKey(11),
                soft_forward=True,
            )
            import optax
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

        grads = jax.grad(loss_fn)(params)
        grad_norm = jnp.sqrt(sum(
            jnp.sum(g**2) for g in jax.tree.leaves(grads)
        ))
        assert float(grad_norm) > 1e-6


class TestMDLLoss:
    """Smoke test the MDL loss function."""

    def test_mdl_loss_finite(self, model, params, dummy_input):
        from src.training.steps import _mdl_loss

        y = jnp.array([0, 1, 2, 3])
        rng = jrandom.PRNGKey(5)

        total_loss, (logits, data_nll_nats, complexity_expected_nats, entropy_weights_nats, z) = _mdl_loss(
            model.apply, params, dummy_input, y,
            rng=rng, tau=1.0, mdl_lambda=0.01,
            n_train=100, n_samples=1, soft_forward=True,
        )

        assert jnp.isfinite(total_loss)
        assert jnp.isfinite(data_nll_nats)
        assert jnp.isfinite(complexity_expected_nats)
        assert jnp.isfinite(entropy_weights_nats)
        assert float(data_nll_nats) > 0
        assert float(complexity_expected_nats) > 0
        assert float(entropy_weights_nats) >= 0


class TestMDLTrainStep:
    """Smoke test a single MDL training step."""

    def test_single_step_runs(self, model, dummy_input):
        from src.training.steps import make_train_step_mdl
        from src.training.train_state import MDLTrainState
        import optax

        # Minimal config mock
        class _MDLCfg:
            n_samples = 1
        class _Cfg:
            mdl = _MDLCfg()

        step_fn = make_train_step_mdl(_Cfg(), soft_forward=True)

        rng = jrandom.PRNGKey(6)
        params = model.init(rng, dummy_input, tau=1.0, train=False)["params"]
        tx = optax.adam(1e-3)

        state = MDLTrainState(
            step=0,
            apply_fn=model.apply,
            params=params,
            tx=tx,
            opt_state=tx.init(params),
            tau=jnp.array(1.0, dtype=jnp.float32),
        )

        y = jnp.array([0, 1, 2, 3])
        rng_step = jrandom.PRNGKey(7)

        state2, metrics = step_fn(
            state, (dummy_input, y), rng_step,
            mdl_lambda=0.01, n_train=100,
        )

        assert jnp.isfinite(metrics["objective_total_nats"])
        assert 0 <= float(metrics["accuracy"]) <= 1
        assert state2.step == 1
