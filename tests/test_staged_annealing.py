"""Tests for staged temperature annealing (item 3.A).

Covers:
    - Freeze mask initialization
    - Confidence mask computation
    - Weight freezing roundtrip (freeze → one-hot logits, mask union)
    - Gradient masking (frozen params get zero gradients)
    - Staged training smoke test (3 stages × 5 epochs)
"""

import pytest
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import random as jrandom

from src.mdl.training import (
    create_mdl_state,
    compute_confidence_mask,
    freeze_confident_weights,
    apply_freeze_mask_to_grads,
    make_train_step,
)
from src.mdl.lstm import GumbelSoftmaxLSTM
from src.mdl.coding import grid_values_and_codelengths
from src.mdl.data import NUM_SYMBOLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_small_model_and_state(rng, tau=1.0):
    """Create a minimal GumbelSoftmaxLSTM + MDLTrainState for testing."""
    grid_values, grid_codelengths = grid_values_and_codelengths(3, 3)
    model = GumbelSoftmaxLSTM(
        hidden_size=3,
        input_size=NUM_SYMBOLS,
        output_size=NUM_SYMBOLS,
        grid_values=grid_values,
        grid_codelengths=grid_codelengths,
    )
    state = create_mdl_state(
        rng, model, seq_len=6, batch_size=2, lr=1e-3, tau_init=tau,
    )
    return model, state, grid_values, grid_codelengths


# ---------------------------------------------------------------------------
# Freeze mask initialization
# ---------------------------------------------------------------------------

class TestFreezeMaskInit:

    def test_freeze_mask_init_zeros(self):
        """Fresh state should have all-zero freeze_mask."""
        rng = jrandom.PRNGKey(0)
        _, state, _, _ = _make_small_model_and_state(rng)
        assert state.freeze_mask.shape == (state.params["logits"].shape[0],)
        assert float(jnp.sum(state.freeze_mask)) == 0.0


# ---------------------------------------------------------------------------
# Confidence mask
# ---------------------------------------------------------------------------

class TestConfidenceMask:

    def test_uniform_logits_no_confidence(self):
        """Uniform logits (softmax = 1/M) should not trigger any mask."""
        M = 15
        logits = jnp.zeros((10, M))
        mask = compute_confidence_mask(logits, threshold=0.95)
        assert mask.shape == (10,)
        assert float(jnp.sum(mask)) == 0.0

    def test_peaked_logits_trigger_mask(self):
        """Logits with one very large entry should trigger mask."""
        M = 15
        logits = jnp.zeros((5, M))
        # Make first 3 parameters peaked (softmax → ~1.0 on column 0)
        logits = logits.at[:3, 0].set(100.0)
        mask = compute_confidence_mask(logits, threshold=0.95)
        assert float(jnp.sum(mask)) == 3.0
        np.testing.assert_array_equal(np.array(mask), [1, 1, 1, 0, 0])

    def test_threshold_boundary(self):
        """Mask should respect the threshold precisely."""
        # 2-class softmax: softmax([x, 0]) = [sigmoid(x), 1-sigmoid(x)]
        # sigmoid(3) ≈ 0.953, sigmoid(2) ≈ 0.881
        logits = jnp.array([[3.0, 0.0], [2.0, 0.0]])
        mask_95 = compute_confidence_mask(logits, threshold=0.95)
        assert float(mask_95[0]) == 1.0  # 0.953 > 0.95
        assert float(mask_95[1]) == 0.0  # 0.881 < 0.95


# ---------------------------------------------------------------------------
# Freeze confident weights
# ---------------------------------------------------------------------------

class TestFreezeConfidentWeights:

    def test_roundtrip_logits_become_one_hot(self):
        """Frozen weights should have one-hot logits × 100."""
        rng = jrandom.PRNGKey(42)
        _, state, _, _ = _make_small_model_and_state(rng)

        # Make first 2 parameters very peaked
        logits = state.params["logits"]
        M = logits.shape[1]
        peaked_logits = logits.at[0, 3].set(100.0).at[1, 5].set(100.0)
        state = state.replace(params={**state.params, "logits": peaked_logits})

        new_state = freeze_confident_weights(state, threshold=0.95)

        # First 2 should be frozen (one-hot × 100)
        assert float(new_state.freeze_mask[0]) == 1.0
        assert float(new_state.freeze_mask[1]) == 1.0
        assert float(new_state.params["logits"][0, 3]) == 100.0
        assert float(new_state.params["logits"][1, 5]) == 100.0
        # Non-argmax entries should be 0
        assert float(new_state.params["logits"][0, 0]) == 0.0
        assert float(new_state.params["logits"][1, 0]) == 0.0

    def test_mask_union_persists(self):
        """Once frozen, weights stay frozen even if called again."""
        rng = jrandom.PRNGKey(7)
        _, state, _, _ = _make_small_model_and_state(rng)

        # Manually set freeze_mask for param 0
        state = state.replace(
            freeze_mask=state.freeze_mask.at[0].set(1.0),
        )
        # Make param 0 logits uniform (no longer peaked)
        logits = state.params["logits"].at[0, :].set(0.0)
        state = state.replace(params={**state.params, "logits": logits})

        new_state = freeze_confident_weights(state, threshold=0.95)
        # Param 0 should still be frozen (union with existing mask)
        assert float(new_state.freeze_mask[0]) == 1.0


# ---------------------------------------------------------------------------
# Gradient masking
# ---------------------------------------------------------------------------

class TestGradientMasking:

    def test_frozen_grads_zeroed(self):
        """Frozen parameters should have zero gradients."""
        n_params, M = 5, 10
        grads = {"logits": jnp.ones((n_params, M))}
        freeze_mask = jnp.array([1.0, 0.0, 1.0, 0.0, 0.0])

        masked = apply_freeze_mask_to_grads(grads, freeze_mask)

        # Frozen rows (0, 2) should be zero
        np.testing.assert_allclose(np.array(masked["logits"][0]), 0.0)
        np.testing.assert_allclose(np.array(masked["logits"][2]), 0.0)
        # Unfrozen rows (1, 3, 4) should be unchanged
        np.testing.assert_allclose(np.array(masked["logits"][1]), 1.0)
        np.testing.assert_allclose(np.array(masked["logits"][3]), 1.0)
        np.testing.assert_allclose(np.array(masked["logits"][4]), 1.0)

    def test_no_mask_passes_through(self):
        """All-zero mask should not modify gradients."""
        n_params, M = 3, 8
        grads = {"logits": jnp.ones((n_params, M)) * 2.5}
        freeze_mask = jnp.zeros(n_params)

        masked = apply_freeze_mask_to_grads(grads, freeze_mask)
        np.testing.assert_allclose(np.array(masked["logits"]), 2.5)

    def test_other_grad_keys_preserved(self):
        """Non-logit gradient keys should pass through unchanged."""
        grads = {
            "logits": jnp.ones((3, 5)),
            "other_param": jnp.ones((2, 2)) * 7.0,
        }
        freeze_mask = jnp.array([1.0, 1.0, 1.0])

        masked = apply_freeze_mask_to_grads(grads, freeze_mask)
        np.testing.assert_allclose(np.array(masked["other_param"]), 7.0)


# ---------------------------------------------------------------------------
# Staged training smoke test
# ---------------------------------------------------------------------------

class TestStagedTrainingSmoke:

    def test_staged_training_3_stages(self):
        """Run 3 stages × 5 epochs each and verify state updates correctly."""
        from src.mdl.data import make_anbn_dataset, sequences_to_padded_arrays

        rng = jrandom.PRNGKey(99)

        # Tiny dataset
        train_inputs, train_targets = make_anbn_dataset(
            num_strings=8, p=0.3, seed=42,
        )
        x_train, y_train, mask_train = sequences_to_padded_arrays(
            train_inputs, train_targets,
        )

        rng, model_rng = jrandom.split(rng)
        _, state, grid_values, grid_codelengths = _make_small_model_and_state(
            model_rng, tau=1.0,
        )

        # Stage config: 3 stages with decreasing tau
        tau_levels = [1.0, 0.3, 0.01]
        epochs_per_stage = 5
        n_train = x_train.shape[0]

        for stage_idx, tau in enumerate(tau_levels):
            state = state.replace(tau=jnp.array(tau, dtype=jnp.float32))

            train_step = make_train_step(
                mdl_lambda=1.0,
                n_train=n_train,
                use_freeze_mask=True,
                jit=True,
            )

            for ep in range(epochs_per_stage):
                rng, step_rng = jrandom.split(rng)
                state, loss, aux = train_step(
                    state, x_train, y_train, mask_train, step_rng,
                )
                assert jnp.isfinite(loss), f"Non-finite loss at stage {stage_idx} epoch {ep}"

            # Freeze confident weights between stages (except after last)
            if stage_idx < len(tau_levels) - 1:
                state = freeze_confident_weights(state, threshold=0.95)

        # After 3 stages, check state is valid
        assert state.params["logits"].shape[0] == state.freeze_mask.shape[0]
        assert jnp.all(jnp.isfinite(state.params["logits"]))
        # Freeze mask should be non-negative
        assert float(jnp.min(state.freeze_mask)) >= 0.0
        assert float(jnp.max(state.freeze_mask)) <= 1.0
