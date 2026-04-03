"""Tests for MMD conditional invariance loss and weight computation."""

import jax
import jax.numpy as jnp
import pytest

from src.loss_fns.reg_loss_fns import class_cond_mmd_rbf
from src.training.runners import compute_mmd_weights


class TestClassCondMMD:
    """Tests for class_cond_mmd_rbf."""

    def test_identical_distributions_zero(self):
        """MMD should be ~0 when z distributions are the same across colors."""
        rng = jax.random.PRNGKey(42)
        B = 200
        D = 10
        z = jax.random.normal(rng, (B, D))
        # All same class, two colors assigned randomly
        y = jnp.zeros(B, dtype=jnp.int32)
        s = jnp.where(jnp.arange(B) < B // 2, 0, 1).astype(jnp.int32)

        mmd = class_cond_mmd_rbf(z, y, s, num_classes=2, num_colors=2)
        assert float(mmd) < 0.05, f"Expected ~0 MMD, got {float(mmd)}"

    def test_different_distributions_positive(self):
        """MMD should be positive when z distributions differ across colors."""
        rng = jax.random.PRNGKey(42)
        B = 200
        D = 10
        r1, r2 = jax.random.split(rng)
        # Color 0: mean=0, Color 1: mean=5 (very different)
        z0 = jax.random.normal(r1, (B // 2, D))
        z1 = jax.random.normal(r2, (B // 2, D)) + 5.0
        z = jnp.concatenate([z0, z1], axis=0)
        y = jnp.zeros(B, dtype=jnp.int32)
        s = jnp.concatenate([
            jnp.zeros(B // 2, dtype=jnp.int32),
            jnp.ones(B // 2, dtype=jnp.int32),
        ])

        mmd = class_cond_mmd_rbf(z, y, s, num_classes=2, num_colors=2)
        assert float(mmd) > 0.1, f"Expected positive MMD, got {float(mmd)}"

    def test_empty_cells_handled(self):
        """MMD should not crash when some (y,s) cells have < 2 examples."""
        z = jax.random.normal(jax.random.PRNGKey(0), (20, 5))
        y = jnp.zeros(20, dtype=jnp.int32)  # all class 0
        s = jnp.zeros(20, dtype=jnp.int32)  # all color 0, no color 1
        # Only one color present — no valid pairs
        mmd = class_cond_mmd_rbf(z, y, s, num_classes=2, num_colors=2)
        assert float(mmd) == pytest.approx(0.0, abs=1e-6)

    def test_multiclass(self):
        """MMD works with multiple classes and colors."""
        rng = jax.random.PRNGKey(99)
        B = 300
        D = 8
        z = jax.random.normal(rng, (B, D))
        y = jnp.tile(jnp.arange(3), B // 3).astype(jnp.int32)
        s = jnp.tile(jnp.arange(3), B // 3).astype(jnp.int32)

        mmd = class_cond_mmd_rbf(z, y, s, num_classes=3, num_colors=3)
        assert jnp.isfinite(mmd), "MMD should be finite"


class TestComputeMMDWeights:
    """Tests for compute_mmd_weights."""

    def test_uniform_labels_uniform_preds(self):
        """When p(s|y) is already uniform, weights should be ~1."""
        K = 4
        N = 400
        y = jnp.tile(jnp.arange(K), N // K).astype(jnp.int32)
        s = jnp.tile(jnp.arange(K), N // K).astype(jnp.int32)
        # Shuffle s to make p(s|y) uniform
        s = jax.random.permutation(jax.random.PRNGKey(0), s)

        w = compute_mmd_weights(y, s, K, smoothing_eps=1e-6, w_max=500.0)
        # All weights should be close to 1
        assert float(jnp.max(jnp.abs(w - 1.0))) < 0.5

    def test_biased_labels_high_weight(self):
        """When one (y,s) cell is rare, its weight should be larger."""
        K = 2
        N = 100
        # Class 0: 90 with color 0, 10 with color 1
        y = jnp.zeros(N, dtype=jnp.int32)
        s = jnp.concatenate([
            jnp.zeros(90, dtype=jnp.int32),
            jnp.ones(10, dtype=jnp.int32),
        ])

        w = compute_mmd_weights(y, s, K, smoothing_eps=1e-6, w_max=500.0)
        # Rare color should have higher weight
        w_common = float(w[0])
        w_rare = float(w[90])
        assert w_rare > w_common * 3, (
            f"Rare color weight {w_rare} should be much larger "
            f"than common {w_common}"
        )

    def test_weight_clipping(self):
        """Weights should be clipped to w_max."""
        K = 2
        N = 100
        y = jnp.zeros(N, dtype=jnp.int32)
        # 99 with color 0, 1 with color 1 — extreme imbalance
        s = jnp.concatenate([
            jnp.zeros(99, dtype=jnp.int32),
            jnp.ones(1, dtype=jnp.int32),
        ])

        w_max = 10.0
        w = compute_mmd_weights(y, s, K, smoothing_eps=1e-6, w_max=w_max)
        assert float(jnp.max(w)) <= w_max + 1e-6
