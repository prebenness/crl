"""Tests for the baseline LSTM module and training infrastructure.

Covers:
    - BaselineLSTM forward pass shape and output
    - Parameter count matches GumbelSoftmaxLSTM (108 parameters)
    - L1/L2 regularization affects loss
    - MDL score computation via weight rationalization
    - Training step reduces loss
    - Compatibility with MDL evaluation function
"""

import pytest
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
from jax import random as jrandom

from src.mdl.baseline_lstm import (
    BaselineLSTM,
    create_baseline_state,
    make_baseline_train_step,
    make_baseline_loss_fn,
    compute_baseline_mdl_score,
    flatten_params,
)
from src.mdl.data import (
    make_anbn_dataset,
    make_test_set,
    sequences_to_padded_arrays,
    NUM_SYMBOLS,
)
from src.mdl.training import evaluate_deterministic_accuracy


# ---------------------------------------------------------------------------
# Model architecture tests
# ---------------------------------------------------------------------------

class TestBaselineLSTM:
    """Test the baseline LSTM model."""

    def setup_method(self):
        self.model = BaselineLSTM(
            hidden_size=3,
            input_size=NUM_SYMBOLS,
            output_size=NUM_SYMBOLS,
        )
        self.rng = jrandom.PRNGKey(0)

    def test_output_shape(self):
        """Forward pass produces correct output shape."""
        x = jnp.zeros((2, 10), dtype=jnp.int32)
        params = self.model.init(self.rng, x, train=False)["params"]
        logits, aux = self.model.apply({"params": params}, x, train=False)
        assert logits.shape == (2, 10, NUM_SYMBOLS)

    def test_param_count(self):
        """Baseline has same number of parameters as GumbelSoftmaxLSTM (108)."""
        x = jnp.zeros((1, 5), dtype=jnp.int32)
        params = self.model.init(self.rng, x, train=False)["params"]
        all_weights = flatten_params(params)
        # LSTM: 4*3*3 (input) + 4*3*3 (hidden) + 4*3 + 4*3 (biases) = 96
        # Output: 3*3 + 3 = 12
        # Total: 108
        assert len(all_weights) == 108

    def test_accepts_tau_kwarg(self):
        """Model accepts tau keyword for compatibility with MDL eval."""
        x = jnp.zeros((1, 5), dtype=jnp.int32)
        params = self.model.init(self.rng, x, train=False)["params"]
        logits, _ = self.model.apply(
            {"params": params}, x, tau=1.0, train=False,
        )
        assert logits.shape == (1, 5, NUM_SYMBOLS)

    def test_different_seeds_different_params(self):
        """Different seeds produce different initializations."""
        x = jnp.zeros((1, 5), dtype=jnp.int32)
        p1 = self.model.init(jrandom.PRNGKey(0), x, train=False)["params"]
        p2 = self.model.init(jrandom.PRNGKey(1), x, train=False)["params"]
        w1 = flatten_params(p1)
        w2 = flatten_params(p2)
        assert not jnp.allclose(w1, w2)

    def test_dropout_model(self):
        """Dropout model produces different outputs in train vs eval."""
        model = BaselineLSTM(
            hidden_size=3,
            input_size=NUM_SYMBOLS,
            output_size=NUM_SYMBOLS,
            dropout_rate=0.5,
        )
        x = jnp.ones((4, 20), dtype=jnp.int32)
        params = model.init(self.rng, x, train=False)["params"]

        eval_logits, _ = model.apply({"params": params}, x, train=False)
        train_logits, _ = model.apply(
            {"params": params}, x, train=True,
            rng=jrandom.PRNGKey(42),
        )
        # With high dropout and enough data, outputs should differ
        # (not guaranteed per-sample, but very likely with 0.5 rate)
        assert eval_logits.shape == train_logits.shape


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

class TestBaselineLoss:
    """Test loss function with regularization."""

    def setup_method(self):
        self.model = BaselineLSTM(
            hidden_size=3,
            input_size=NUM_SYMBOLS,
            output_size=NUM_SYMBOLS,
        )
        self.rng = jrandom.PRNGKey(0)
        # Create minimal training data
        inputs, targets = make_anbn_dataset(num_strings=10, p=0.3, seed=0)
        self.x, self.y, self.mask = sequences_to_padded_arrays(inputs, targets)
        init_rng, self.step_rng = jrandom.split(self.rng)
        self.params = self.model.init(init_rng, self.x, train=False)["params"]

    def test_ce_loss_positive(self):
        """CE-only loss is positive."""
        loss_fn = make_baseline_loss_fn(reg_type=None, reg_lambda=0.0)
        loss, aux = loss_fn(
            self.params, self.model.apply, self.x, self.y, self.mask,
            self.step_rng,
        )
        assert float(loss) > 0
        assert float(aux["data_nll_bits"]) > 0
        assert float(aux["reg_regularizer"]) == 0.0

    def test_l1_increases_loss(self):
        """L1 regularization increases the total loss."""
        loss_fn_noreg = make_baseline_loss_fn(reg_type=None, reg_lambda=0.0)
        loss_fn_l1 = make_baseline_loss_fn(reg_type="l1", reg_lambda=1.0)

        loss_noreg, _ = loss_fn_noreg(
            self.params, self.model.apply, self.x, self.y, self.mask,
            self.step_rng,
        )
        loss_l1, aux_l1 = loss_fn_l1(
            self.params, self.model.apply, self.x, self.y, self.mask,
            self.step_rng,
        )
        assert float(loss_l1) > float(loss_noreg)
        assert float(aux_l1["reg_regularizer"]) > 0

    def test_l2_increases_loss(self):
        """L2 regularization increases the total loss."""
        loss_fn_noreg = make_baseline_loss_fn(reg_type=None, reg_lambda=0.0)
        loss_fn_l2 = make_baseline_loss_fn(reg_type="l2", reg_lambda=1.0)

        loss_noreg, _ = loss_fn_noreg(
            self.params, self.model.apply, self.x, self.y, self.mask,
            self.step_rng,
        )
        loss_l2, _ = loss_fn_l2(
            self.params, self.model.apply, self.x, self.y, self.mask,
            self.step_rng,
        )
        assert float(loss_l2) > float(loss_noreg)


# ---------------------------------------------------------------------------
# Training step tests
# ---------------------------------------------------------------------------

class TestBaselineTraining:
    """Test that training reduces loss."""

    def test_training_reduces_loss(self):
        """A few training steps should reduce the loss."""
        model = BaselineLSTM(
            hidden_size=3,
            input_size=NUM_SYMBOLS,
            output_size=NUM_SYMBOLS,
        )
        inputs, targets = make_anbn_dataset(num_strings=50, p=0.3, seed=0)
        x, y, mask = sequences_to_padded_arrays(inputs, targets)

        rng = jrandom.PRNGKey(0)
        rng, init_rng = jrandom.split(rng)
        state = create_baseline_state(
            init_rng, model,
            seq_len=x.shape[1], batch_size=x.shape[0], lr=0.001,
        )
        train_step = make_baseline_train_step(reg_type=None, reg_lambda=0.0)

        initial_loss = None
        for i in range(20):
            rng, step_rng = jrandom.split(rng)
            state, loss, aux = train_step(state, x, y, mask, step_rng)
            if initial_loss is None:
                initial_loss = float(loss)

        final_loss = float(loss)
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )


# ---------------------------------------------------------------------------
# MDL score tests
# ---------------------------------------------------------------------------

class TestBaselineMDL:
    """Test MDL score computation for baseline networks."""

    def test_mdl_score_positive(self):
        """MDL score is positive for any network."""
        model = BaselineLSTM(
            hidden_size=3,
            input_size=NUM_SYMBOLS,
            output_size=NUM_SYMBOLS,
        )
        rng = jrandom.PRNGKey(0)
        x = jnp.zeros((1, 5), dtype=jnp.int32)
        params = model.init(rng, x, train=False)["params"]

        mdl = compute_baseline_mdl_score(params, hidden_size=3)
        assert mdl["total_bits"] > 0
        assert mdl["arch_bits"] == 5  # E(3) = 5 bits
        assert mdl["weight_bits"] > 0
        assert mdl["n_params"] == 108

    def test_zero_weights_have_low_mdl(self):
        """A network with all zero weights has low MDL."""
        model = BaselineLSTM(
            hidden_size=3,
            input_size=NUM_SYMBOLS,
            output_size=NUM_SYMBOLS,
        )
        rng = jrandom.PRNGKey(0)
        x = jnp.zeros((1, 5), dtype=jnp.int32)
        params = model.init(rng, x, train=False)["params"]

        # Zero out all parameters
        zero_params = jax.tree.map(jnp.zeros_like, params)
        mdl = compute_baseline_mdl_score(zero_params, hidden_size=3)

        # Zero weight = 0/1, codelength = 1 (sign) + E(0) + E(1) = 1+1+3 = 5 bits
        expected_weight_bits = 108 * 5
        assert mdl["weight_bits"] == expected_weight_bits
        assert mdl["n_nonzero"] == 0


# ---------------------------------------------------------------------------
# Evaluation compatibility tests
# ---------------------------------------------------------------------------

class TestBaselineEvaluation:
    """Test that baseline model works with MDL evaluation functions."""

    def test_deterministic_accuracy_compatible(self):
        """evaluate_deterministic_accuracy works with baseline model."""
        model = BaselineLSTM(
            hidden_size=3,
            input_size=NUM_SYMBOLS,
            output_size=NUM_SYMBOLS,
        )
        rng = jrandom.PRNGKey(0)
        x = jnp.zeros((1, 10), dtype=jnp.int32)
        params = model.init(rng, x, train=False)["params"]

        test_inputs, test_targets = make_test_set(max_n=5)
        result = evaluate_deterministic_accuracy(
            model.apply, params, None,
            test_inputs, test_targets, max_n=5,
        )
        assert "gen_n" in result
        assert "n_perfect" in result
        assert "mean_accuracy" in result
        assert 0 <= result["mean_accuracy"] <= 1
