"""Smoke tests for the fixed-beta DST training mode."""

import argparse
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random as jrandom

from src.mdl.coding import build_rational_grid, rational_codelength
from src.mdl.data import make_anbn_dataset, sequences_to_padded_arrays, NUM_SYMBOLS
from src.mdl.lstm import GumbelSoftmaxLSTM
from src.mdl.training import (
    create_mdl_state,
    make_train_step,
    make_fused_epoch_fn_fixed_tau,
)
from differentiable_mdl import _zero_adam_moments, _maybe_restart


def _make_small_setup(hidden_size=2, n_max=3, m_max=3, num_train=20, seed=0):
    """Create a minimal model + data for testing."""
    grid_fractions = build_rational_grid(n_max, m_max)
    grid_values = np.array([float(f) for f in grid_fractions], dtype=np.float32)
    grid_codelengths = np.array(
        [rational_codelength(f) for f in grid_fractions], dtype=np.float32,
    )
    M = len(grid_values)

    model = GumbelSoftmaxLSTM(
        hidden_size=hidden_size,
        input_size=NUM_SYMBOLS,
        output_size=NUM_SYMBOLS,
        grid_values=grid_values,
        grid_codelengths=grid_codelengths,
        init_cl_scale=1.0,
    )

    train_inputs, train_targets = make_anbn_dataset(
        num_strings=num_train, p=0.3, seed=seed,
    )
    x_train, y_train, mask_train = sequences_to_padded_arrays(
        train_inputs, train_targets,
    )
    return model, grid_values, grid_codelengths, x_train, y_train, mask_train


class TestFusedEpochFixedTau:
    """Test the fixed-tau fused epoch function."""

    def test_tau_unchanged_after_fused_steps(self):
        """State.tau should remain constant after fused training steps."""
        model, gv, gc, x, y, mask = _make_small_setup()
        N = x.shape[0]
        tau_fixed = 0.7

        rng = jrandom.PRNGKey(0)
        rng, init_rng = jrandom.split(rng)

        state = create_mdl_state(
            init_rng, model,
            seq_len=x.shape[1],
            batch_size=N,
            lr=0.01,
            tau_init=tau_fixed,
        )

        step_nojit = make_train_step(
            mdl_lambda=1.0, n_train=N, n_samples=1,
            deterministic_st=True, jit=False,
        )
        fused = make_fused_epoch_fn_fixed_tau(step_nojit, x, y, mask)

        state, rng, metrics = fused(state, rng, 5)

        np.testing.assert_allclose(float(state.tau), tau_fixed, atol=1e-7)

    def test_fused_returns_valid_metrics(self):
        """Fused function should return a dict with expected metric keys."""
        model, gv, gc, x, y, mask = _make_small_setup()
        N = x.shape[0]

        rng = jrandom.PRNGKey(0)
        rng, init_rng = jrandom.split(rng)

        state = create_mdl_state(
            init_rng, model,
            seq_len=x.shape[1],
            batch_size=N,
            lr=0.01,
            tau_init=1.0,
        )

        step_nojit = make_train_step(
            mdl_lambda=1.0, n_train=N, n_samples=1,
            deterministic_st=True, jit=False,
        )
        fused = make_fused_epoch_fn_fixed_tau(step_nojit, x, y, mask)

        state, rng, metrics = fused(state, rng, 3)

        expected_keys = {
            "objective_total_bits", "data_nll_bits",
            "complexity_expected_bits", "entropy_weights_bits",
            "reg_complexity_weighted_bits", "reg_entropy_bonus_bits",
            "reg_net_bits",
        }
        assert expected_keys.issubset(set(metrics.keys()))
        # All metric values should be finite
        for k in expected_keys:
            assert np.isfinite(float(metrics[k])), f"{k} is not finite"

    def test_loss_decreases_over_steps(self):
        """Training should reduce the objective over multiple fused steps."""
        model, gv, gc, x, y, mask = _make_small_setup()
        N = x.shape[0]

        rng = jrandom.PRNGKey(42)
        rng, init_rng = jrandom.split(rng)

        state = create_mdl_state(
            init_rng, model,
            seq_len=x.shape[1],
            batch_size=N,
            lr=0.01,
            tau_init=1.0,
        )

        step_nojit = make_train_step(
            mdl_lambda=0.1, n_train=N, n_samples=1,
            deterministic_st=True, jit=False,
        )
        fused = make_fused_epoch_fn_fixed_tau(step_nojit, x, y, mask)

        # Run 5 steps, record loss
        state, rng, metrics_early = fused(state, rng, 5)
        loss_early = float(metrics_early["objective_total_bits"])

        # Run 50 more steps
        state, rng, metrics_late = fused(state, rng, 50)
        loss_late = float(metrics_late["objective_total_bits"])

        assert loss_late < loss_early, (
            f"Loss did not decrease: {loss_early:.4f} -> {loss_late:.4f}"
        )


class TestDSTFixedSmoke:
    """End-to-end smoke test for run_training_dst_fixed."""

    def test_run_training_dst_fixed_returns_valid_tuple(self):
        """run_training_dst_fixed should return (state, best_params, n_perfect, epoch)."""
        from differentiable_mdl import run_training_dst_fixed

        model, gv, gc, x, y, mask = _make_small_setup(num_train=20)
        N = x.shape[0]

        # Minimal args namespace
        args = argparse.Namespace(
            epochs=5,
            lr=0.01,
            mdl_lambda=1.0,
            tau_fixed=1.0,
            batch_size=0,
            hidden_size=2,
            log_every=2,
            eval_every=5,
            restart_patience=0,
            mode_forward=False,
        )

        rng = jrandom.PRNGKey(0)
        # Validation set: a few fixed strings
        from src.mdl.data import make_anbn_fixed_n
        val_inputs = []
        val_targets = []
        for n in [1, 2, 3]:
            inp, tgt = make_anbn_fixed_n(n)
            val_inputs.append(inp)
            val_targets.append(tgt)

        from src.mdl.coding import build_rational_grid
        grid = build_rational_grid(3, 3)

        result = run_training_dst_fixed(
            args, model, gv, gc,
            x, y, mask,
            val_inputs, val_targets, rng, grid,
            run_dir=None,
        )

        assert len(result) == 4
        state, best_params, best_n_perfect, best_epoch = result
        # State tau should still be tau_fixed
        np.testing.assert_allclose(float(state.tau), 1.0, atol=1e-7)
        # best_epoch should be > 0 if eval ran
        assert best_epoch > 0


class TestRestart:
    """Tests for the restart / backtracking mechanism."""

    def test_zero_adam_moments_preserves_count(self):
        """_zero_adam_moments should zero mu/nu but keep count."""
        tx = optax.adam(0.01)
        params = {"logits": jnp.ones((4, 3))}
        opt_state = tx.init(params)
        # Simulate a few update steps to get non-zero mu/nu and count
        grads = {"logits": jnp.ones((4, 3)) * 0.1}
        for _ in range(5):
            updates, opt_state = tx.update(grads, opt_state, params)
        adam_state = opt_state[0]
        assert int(adam_state.count) == 5
        assert float(jnp.sum(jnp.abs(adam_state.mu["logits"]))) > 0

        zeroed_state = _zero_adam_moments(opt_state)
        zeroed_adam = zeroed_state[0]
        # count preserved
        assert int(zeroed_adam.count) == 5
        # mu and nu zeroed
        np.testing.assert_array_equal(zeroed_adam.mu["logits"], 0.0)
        np.testing.assert_array_equal(zeroed_adam.nu["logits"], 0.0)

    def test_maybe_restart_fires_at_patience(self):
        """_maybe_restart should trigger when epochs_since_best >= patience."""
        model, gv, gc, x, y, mask = _make_small_setup()
        N = x.shape[0]
        rng = jrandom.PRNGKey(0)
        rng, init_rng = jrandom.split(rng)
        state = create_mdl_state(
            init_rng, model, seq_len=x.shape[1],
            batch_size=N, lr=0.01, tau_init=1.0,
        )
        # Train a few steps to diverge params from init
        step_nojit = make_train_step(
            mdl_lambda=1.0, n_train=N, n_samples=1,
            deterministic_st=True, jit=False,
        )
        fused = make_fused_epoch_fn_fixed_tau(step_nojit, x, y, mask)
        state, rng, _ = fused(state, rng, 10)

        # Build a fake best dict with the initial params
        best = {
            'params': jax.tree.map(jnp.zeros_like, state.params),
            'opt_state': jax.tree.map(lambda x: x.copy(), state.opt_state),
            'epoch': 5,
            'n_perfect': 0,
            'gen_n': 0,
            'complexity_bits': 999.0,
        }

        # Should NOT restart — below patience
        s, esb, did = _maybe_restart(state, best, 90, 100)
        assert not did
        assert esb == 90

        # Should restart — at patience
        params_before = state.params['logits'].copy()
        s, esb, did = _maybe_restart(state, best, 100, 100)
        assert did
        assert esb == 0
        # Params should now match best (zeros)
        np.testing.assert_array_equal(s.params['logits'], 0.0)
        # Adam moments should be zeroed
        adam_state = s.opt_state[0]
        np.testing.assert_array_equal(adam_state.mu['logits'], 0.0)
        np.testing.assert_array_equal(adam_state.nu['logits'], 0.0)
        # Count should be preserved from best's opt_state
        best_count = int(best['opt_state'][0].count)
        assert int(adam_state.count) == best_count

    def test_maybe_restart_disabled_when_patience_zero(self):
        """_maybe_restart should never fire with patience=0."""
        best = {'params': jnp.zeros(3), 'opt_state': None}
        state = "dummy"
        s, esb, did = _maybe_restart(state, best, 99999, 0)
        assert not did
        assert s == "dummy"

    def test_maybe_restart_skips_when_no_best(self):
        """_maybe_restart should not fire if best params is None."""
        best = {'params': None, 'opt_state': None}
        state = "dummy"
        s, esb, did = _maybe_restart(state, best, 99999, 100)
        assert not did
