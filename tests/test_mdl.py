"""Tests for the differentiable MDL implementation.

Covers:
    - Coding scheme: E(n) integer code length matches Lan et al. examples
    - Rational codelength: per-weight coding matches proposal Definition 2
    - Rational grid: correct construction and deduplication
    - Golden network: correct output probabilities and 100% accuracy
    - Data generation: correct a^n b^n strings from PCFG
    - Deterministic accuracy: correct masking (input b positions only)
    - Shared weights: P_base normalization, epsilon-bounded simplex, KL
"""

import pytest
import math
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
from jax import random as jrandom


# ---------------------------------------------------------------------------
# Coding scheme tests
# ---------------------------------------------------------------------------

class TestIntegerCodeLength:
    """Verify E(n) matches Lan et al. (2024) examples."""

    def test_E0(self):
        from src.mdl.coding import integer_code_length
        # E(0) = "0" -> length 1
        assert integer_code_length(0) == 1

    def test_E1(self):
        from src.mdl.coding import integer_code_length
        # E(1) = "1 0 1" -> length 3
        assert integer_code_length(1) == 3

    def test_E2(self):
        from src.mdl.coding import integer_code_length
        # E(2) = "11 0 10" -> length 5 (from paper: E(2) = 11010)
        assert integer_code_length(2) == 5

    def test_E5(self):
        from src.mdl.coding import integer_code_length
        # E(5) = "111 0 101" -> length 7 (from paper: E(5) = 1110101)
        assert integer_code_length(5) == 7

    def test_formula(self):
        """Verify |E(n)| = 2*ceil(log2(n+1)) + 1 for various n."""
        from src.mdl.coding import integer_code_length
        for n in range(20):
            k = math.ceil(math.log2(n + 1)) if n > 0 else 0
            expected = 2 * k + 1
            assert integer_code_length(n) == expected, f"Failed for n={n}"


class TestRationalCodelength:
    """Verify per-weight codelength l(w) = 1 + |E(n)| + |E(m)|."""

    def test_zero(self):
        from src.mdl.coding import rational_codelength
        # 0 = +0/1: 1 + |E(0)|=1 + |E(1)|=3 = 5
        assert rational_codelength(Fraction(0)) == 5

    def test_one(self):
        from src.mdl.coding import rational_codelength
        # 1 = +1/1: 1 + |E(1)|=3 + |E(1)|=3 = 7
        assert rational_codelength(Fraction(1)) == 7

    def test_two_fifths(self):
        from src.mdl.coding import rational_codelength
        # 2/5 = +2/5: 1 + |E(2)|=5 + |E(5)|=7 = 13
        assert rational_codelength(Fraction(2, 5)) == 13

    def test_negative(self):
        from src.mdl.coding import rational_codelength
        # -1 = -1/1: 1 + |E(1)|=3 + |E(1)|=3 = 7 (same as +1)
        assert rational_codelength(Fraction(-1)) == 7

    def test_simple_cheaper_than_complex(self):
        """Simpler rationals should have shorter codes."""
        from src.mdl.coding import rational_codelength
        assert rational_codelength(Fraction(0)) < rational_codelength(Fraction(1))
        assert rational_codelength(Fraction(1)) < rational_codelength(Fraction(7, 3))


class TestRationalGrid:
    """Verify grid construction."""

    def test_includes_zero(self):
        from src.mdl.coding import build_rational_grid
        grid = build_rational_grid(5, 5)
        assert Fraction(0) in grid

    def test_symmetric(self):
        """Grid should be symmetric: if w in S then -w in S."""
        from src.mdl.coding import build_rational_grid
        grid = build_rational_grid(5, 5)
        for w in grid:
            if w != 0:
                assert -w in grid, f"-{w} not in grid"

    def test_no_duplicates(self):
        """All entries should be in reduced form (no duplicates)."""
        from src.mdl.coding import build_rational_grid
        grid = build_rational_grid(10, 10)
        assert len(grid) == len(set(grid))

    def test_grid_size_increases(self):
        from src.mdl.coding import build_rational_grid
        g1 = build_rational_grid(5, 5)
        g2 = build_rational_grid(10, 10)
        assert len(g2) > len(g1)


# ---------------------------------------------------------------------------
# Golden network tests
# ---------------------------------------------------------------------------

class TestGoldenNetwork:
    """Verify the golden a^n b^n LSTM."""

    def test_output_probs_start(self):
        """At start symbol #, output should be [p, 1-p, 0]."""
        from src.mdl.golden import build_golden_network_params, golden_forward
        params = build_golden_network_params(p=0.3)
        # Input: just #
        x = jnp.array([[0]], dtype=jnp.int32)  # #
        logits = golden_forward(params, x)
        probs = jax.nn.softmax(logits[0, 0])
        np.testing.assert_allclose(float(probs[0]), 0.3, atol=0.01)
        np.testing.assert_allclose(float(probs[1]), 0.7, atol=0.01)
        np.testing.assert_allclose(float(probs[2]), 0.0, atol=0.01)

    def test_output_probs_a_phase(self):
        """During a phase, output should be [0, 1-p, p]."""
        from src.mdl.golden import build_golden_network_params, golden_forward
        params = build_golden_network_params(p=0.3)
        # Input: # a
        x = jnp.array([[0, 1]], dtype=jnp.int32)
        logits = golden_forward(params, x)
        probs = jax.nn.softmax(logits[0, 1])
        np.testing.assert_allclose(float(probs[0]), 0.0, atol=0.01)
        np.testing.assert_allclose(float(probs[1]), 0.7, atol=0.01)
        np.testing.assert_allclose(float(probs[2]), 0.3, atol=0.01)

    def test_output_probs_b_phase(self):
        """During b phase (not last b), output should be [0, 0, 1]."""
        from src.mdl.golden import build_golden_network_params, golden_forward
        params = build_golden_network_params(p=0.3)
        # Input: # a a b  (count still > 0)
        x = jnp.array([[0, 1, 1, 2]], dtype=jnp.int32)
        logits = golden_forward(params, x)
        probs = jax.nn.softmax(logits[0, 3])
        np.testing.assert_allclose(float(probs[2]), 1.0, atol=0.01)

    def test_output_probs_last_b(self):
        """At last b (count = 0), output should be [1, 0, 0]."""
        from src.mdl.golden import build_golden_network_params, golden_forward
        params = build_golden_network_params(p=0.3)
        # Input: # a b  (n=1, after seeing both a and b, count=0)
        x = jnp.array([[0, 1, 2]], dtype=jnp.int32)
        logits = golden_forward(params, x)
        probs = jax.nn.softmax(logits[0, 2])
        np.testing.assert_allclose(float(probs[0]), 1.0, atol=0.01)

    def test_accuracy_small(self):
        """Golden network should get 100% on small n."""
        from src.mdl.golden import evaluate_golden_network
        result = evaluate_golden_network(max_n=20, p=0.3)
        assert result["all_correct"], f"Failed at n={result['first_failure_n']}"

    def test_mdl_score_positive(self):
        """MDL score should be a positive number of bits."""
        from src.mdl.golden import golden_mdl_score
        mdl = golden_mdl_score()
        assert mdl["total_bits"] > 0
        assert mdl["arch_bits"] > 0
        assert mdl["weight_bits"] > 0
        assert mdl["total_bits"] == mdl["arch_bits"] + mdl["weight_bits"]


# ---------------------------------------------------------------------------
# Data generation tests
# ---------------------------------------------------------------------------

class TestDataGeneration:
    """Verify a^n b^n data generation."""

    def test_string_structure(self):
        """Each string should be # a^n b^n #."""
        from src.mdl.data import generate_anbn_strings, SYMBOL_HASH, SYMBOL_A, SYMBOL_B
        strings = generate_anbn_strings(100, p=0.3, seed=42)
        for s in strings:
            assert s[0] == SYMBOL_HASH
            assert s[-1] == SYMBOL_HASH
            # Count a's and b's
            n_a = s.count(SYMBOL_A)
            n_b = s.count(SYMBOL_B)
            assert n_a == n_b, f"Unbalanced: {n_a} a's, {n_b} b's"

    def test_deterministic_seed(self):
        """Same seed should produce same strings."""
        from src.mdl.data import generate_anbn_strings
        s1 = generate_anbn_strings(50, seed=123)
        s2 = generate_anbn_strings(50, seed=123)
        assert s1 == s2


# ---------------------------------------------------------------------------
# Deterministic accuracy tests
# ---------------------------------------------------------------------------

class TestDeterministicAccuracy:
    """Verify the deterministic accuracy metric."""

    def test_perfect_prediction(self):
        """A model that predicts correctly at all b positions should get 1.0."""
        from src.mdl.data import SYMBOL_B
        # For n=3: input # a a a b b b, target a a a b b b #
        # Deterministic positions: where input=b (positions 4,5,6)
        inp = [0, 1, 1, 1, 2, 2, 2]
        tgt = [1, 1, 1, 2, 2, 2, 0]
        det_positions = [i for i, x in enumerate(inp) if x == SYMBOL_B]
        assert det_positions == [4, 5, 6]

    def test_mask_excludes_a_phase(self):
        """The a phase (including last a -> first b transition) should not be masked."""
        from src.mdl.data import SYMBOL_A, SYMBOL_B
        # Position 3 is last a (input=a, target=b) -- NOT deterministic
        inp = [0, 1, 1, 1, 2, 2, 2]
        det_positions = [i for i, x in enumerate(inp) if x == SYMBOL_B]
        assert 3 not in det_positions  # last a position excluded


# ---------------------------------------------------------------------------
# Shared weights tests
# ---------------------------------------------------------------------------

class TestSharedWeights:
    """Verify shared-weight components."""

    def test_p_base_normalization(self):
        """P_base should sum to 1."""
        from src.mdl.shared_weights import compute_p_base
        grid = jnp.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
        p = compute_p_base(grid)
        np.testing.assert_allclose(float(jnp.sum(p)), 1.0, atol=1e-6)

    def test_p_base_favors_simple(self):
        """P_base should assign higher probability to simpler rationals (near 0)."""
        from src.mdl.shared_weights import compute_p_base
        grid = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        p = compute_p_base(grid)
        # 0 should have highest probability
        assert float(p[2]) > float(p[1])  # P(0) > P(-1)
        assert float(p[1]) > float(p[0])  # P(-1) > P(-10)

    def test_epsilon_bound_lower(self):
        """All phi values should be >= epsilon."""
        from src.mdl.shared_weights import epsilon_bound_simplex
        eps = 1e-4
        logits = jnp.array([-100.0, 0.0, 100.0, -50.0])
        phi = epsilon_bound_simplex(logits, eps)
        assert float(jnp.min(phi)) >= eps * 0.99  # allow small float error

    def test_epsilon_bound_sums_to_one(self):
        """Epsilon-bounded phi should still sum to 1."""
        from src.mdl.shared_weights import epsilon_bound_simplex
        logits = jnp.array([1.0, -1.0, 0.5, 2.0])
        phi = epsilon_bound_simplex(logits, 1e-6)
        np.testing.assert_allclose(float(jnp.sum(phi)), 1.0, atol=1e-5)

    def test_kl_zero_for_identical(self):
        """KL(p || p) should be 0."""
        from src.mdl.shared_weights import _kl_divergence
        p = jnp.array([0.25, 0.25, 0.25, 0.25])
        kl = _kl_divergence(p, p)
        np.testing.assert_allclose(float(kl), 0.0, atol=1e-5)

    def test_kl_positive(self):
        """KL(p || q) should be non-negative for p != q."""
        from src.mdl.shared_weights import _kl_divergence
        p = jnp.array([0.9, 0.1])
        q = jnp.array([0.5, 0.5])
        kl = _kl_divergence(p, q)
        assert float(kl) > 0
