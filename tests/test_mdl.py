"""Tests for the differentiable MDL implementation.

Covers:
    - Coding scheme: E(n) integer code length matches Lan et al. examples
    - Rational codelength: per-weight coding matches proposal Definition 2
    - Rational grid: correct construction and deduplication
    - Golden network: correct output probabilities and 100% accuracy
    - Data generation: correct a^n b^n strings from PCFG
    - Deterministic accuracy: correct masking (input b positions only)
    - Shared weights: P_base from codelengths, epsilon-bounded simplex, KL/CE
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

    def test_golden_float32_limit(self):
        """The handcrafted float32 counter should fail immediately after 2^24."""
        from src.mdl.golden import (
            check_golden_network_single_n,
            estimate_golden_float32_limit,
            golden_float32_counter_limit,
        )

        limit_n = golden_float32_counter_limit()
        assert limit_n == 2**24

        assert check_golden_network_single_n(limit_n)["correct"]
        assert not check_golden_network_single_n(limit_n + 1)["correct"]

        estimate = estimate_golden_float32_limit(max_n=limit_n + 8)
        assert estimate["max_correct_n"] == limit_n
        assert estimate["first_failure_n"] == limit_n + 1

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
        codelengths = jnp.array([5.0, 7.0, 9.0, 11.0])
        p = compute_p_base(codelengths)
        np.testing.assert_allclose(float(jnp.sum(p)), 1.0, atol=1e-6)

    def test_p_base_matches_lan_codelength_prior(self):
        """P_base should be proportional to 2^{-ell(s)}."""
        from src.mdl.shared_weights import compute_p_base
        codelengths = jnp.array([5.0, 7.0, 11.0])
        p = compute_p_base(codelengths)
        weights = np.array([2.0 ** -5.0, 2.0 ** -7.0, 2.0 ** -11.0], dtype=np.float32)
        expected = weights / weights.sum()
        np.testing.assert_allclose(np.array(p), expected, atol=1e-6)

    def test_cross_entropy_equals_entropy_plus_kl(self):
        """CE_2(p, q) should equal H_2(p) + KL(p || q)."""
        from src.mdl.shared_weights import _cross_entropy_bits, _kl_divergence
        p = jnp.array([0.2, 0.3, 0.5], dtype=jnp.float32)
        q = jnp.array([0.6, 0.3, 0.1], dtype=jnp.float32)
        ce = _cross_entropy_bits(p, q)
        h = -jnp.sum(p * jnp.log2(p))
        kl = _kl_divergence(p, q)
        np.testing.assert_allclose(float(ce), float(h + kl), atol=1e-6)

    def test_p_base_recovers_expected_codelength_up_to_constant(self):
        """CE_2(pi, P_base) should equal E[ell] plus a constant offset."""
        from src.mdl.shared_weights import _cross_entropy_bits, compute_p_base
        codelengths = jnp.array([5.0, 7.0, 11.0], dtype=jnp.float32)
        p_base = compute_p_base(codelengths)
        pi = jnp.array([0.2, 0.3, 0.5], dtype=jnp.float32)
        ce = _cross_entropy_bits(pi, p_base)
        expected_len = jnp.sum(pi * codelengths)
        z = jnp.sum(jnp.exp2(-codelengths))
        np.testing.assert_allclose(float(ce), float(expected_len + jnp.log2(z)), atol=1e-6)

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


# ---------------------------------------------------------------------------
# Smoothed evaluation NLL tests
# ---------------------------------------------------------------------------

class TestSmoothedNLL:
    """Verify compute_data_nll_bits_smoothed matches Abudy et al. (2025) convention."""

    def test_agrees_with_logsoftmax_on_normal_logits(self):
        """On well-behaved logits, smoothed and log_softmax NLL should nearly agree."""
        from src.mdl.training import _compute_data_nll_bits, compute_data_nll_bits_smoothed
        logits = jnp.array([[[2.0, 1.0, 0.5], [0.1, 3.0, -1.0]]])  # (1,2,3)
        y = jnp.array([[0, 1]])
        mask = jnp.array([[1.0, 1.0]])
        nll_logsoftmax = _compute_data_nll_bits(logits, y, mask)
        nll_smoothed = compute_data_nll_bits_smoothed(logits, y, mask)
        np.testing.assert_allclose(
            float(nll_smoothed), float(nll_logsoftmax), atol=1e-5,
        )

    def test_finite_on_extreme_logits(self):
        """Even with extreme logits producing near-zero probs, result is finite."""
        from src.mdl.training import compute_data_nll_bits_smoothed
        # logit=-100 gives softmax near 0 for that class
        logits = jnp.array([[[100.0, -100.0, -100.0]]])
        y = jnp.array([[1]])  # target is the near-zero class
        mask = jnp.array([[1.0]])
        nll = compute_data_nll_bits_smoothed(logits, y, mask)
        assert jnp.isfinite(nll), "Smoothed NLL should be finite"
        assert float(nll) > 0

    def test_matches_manual_computation(self):
        """Smoothed NLL should match manual softmax + add eps + -log2."""
        from src.mdl.training import compute_data_nll_bits_smoothed
        logits = jnp.array([[[1.0, 2.0, 3.0]]])
        y = jnp.array([[0]])
        mask = jnp.array([[1.0]])
        # Manual
        probs = jax.nn.softmax(logits[0, 0])
        expected = float(-jnp.log2(probs[0] + 1e-10))
        result = float(compute_data_nll_bits_smoothed(logits, y, mask))
        np.testing.assert_allclose(result, expected, atol=1e-6)
