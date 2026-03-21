"""Tests for paper-comparable evaluation metrics (src/mdl/evaluation.py).

Covers:
    - Grammar weights: normalization, correct geometric distribution
    - Per-string NLL: correctness on known logits
    - Grammar-weighted test NLL: golden network sanity check (~2.94 bits)
    - Optimal |D:H|: golden test baseline matches expected value
    - Δ%: arithmetic correctness
    - Golden under regularisers: norms and MDL match expectations
"""

import pytest
import math

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Grammar weights
# ---------------------------------------------------------------------------

class TestGrammarWeights:
    """Verify grammar weight computation for aⁿbⁿ."""

    def test_sums_to_approximately_one(self):
        """Raw PCFG weights for n=0..max_n sum to ~1 for large max_n."""
        from src.mdl.evaluation import compute_anbn_grammar_weights
        w = compute_anbn_grammar_weights(1000, p=0.3, min_n=0)
        assert abs(w.sum() - 1.0) < 1e-10

    def test_geometric_ratios(self):
        """Adjacent weights should have ratio (1-p)."""
        from src.mdl.evaluation import compute_anbn_grammar_weights
        p = 0.3
        w = compute_anbn_grammar_weights(100, p=p, min_n=0)
        for i in range(10):
            ratio = w[i + 1] / w[i]
            assert abs(ratio - (1 - p)) < 1e-12

    def test_n0_is_largest(self):
        """With p=0.3, n=0 should have the highest weight."""
        from src.mdl.evaluation import compute_anbn_grammar_weights
        w = compute_anbn_grammar_weights(100, p=0.3, min_n=0)
        assert w[0] > w[1] > w[2]

    def test_shape_with_n0(self):
        from src.mdl.evaluation import compute_anbn_grammar_weights
        w = compute_anbn_grammar_weights(50, p=0.3, min_n=0)
        assert w.shape == (51,)  # n=0..50

    def test_shape_without_n0(self):
        from src.mdl.evaluation import compute_anbn_grammar_weights
        w = compute_anbn_grammar_weights(50, p=0.3, min_n=1)
        assert w.shape == (50,)  # n=1..50

    def test_n0_weight_equals_p(self):
        """P(0) = p * (1-p)^0 = p."""
        from src.mdl.evaluation import compute_anbn_grammar_weights
        w = compute_anbn_grammar_weights(10, p=0.3, min_n=0)
        assert abs(w[0] - 0.3) < 1e-12


# ---------------------------------------------------------------------------
# Per-string NLL
# ---------------------------------------------------------------------------

class TestPerStringNLL:
    """Verify per-string NLL computation on known logits."""

    def test_perfect_prediction_near_zero(self):
        """If model predicts the correct token with prob ~1, NLL ≈ 0."""
        from src.mdl.evaluation import compute_per_string_nll_bits

        def perfect_forward(x):
            B, T = x.shape
            # Return logits that put all mass on target
            # For a simple test: target is always token 1
            logits = jnp.full((B, T, 3), -100.0)
            logits = logits.at[:, :, 1].set(100.0)
            return logits

        inputs = [[0, 1, 1]]   # any tokens
        targets = [[1, 1, 1]]  # all token 1
        nll = compute_per_string_nll_bits(perfect_forward, inputs, targets)
        # Should be very close to zero (not exactly due to smoothing)
        assert nll[0] < 0.01

    def test_uniform_prediction(self):
        """Uniform prediction over 3 tokens: NLL = log2(3) per position."""
        from src.mdl.evaluation import compute_per_string_nll_bits

        def uniform_forward(x):
            B, T = x.shape
            return jnp.zeros((B, T, 3))  # uniform logits

        inputs = [[0, 1, 2]]
        targets = [[1, 2, 0]]
        nll = compute_per_string_nll_bits(
            uniform_forward, inputs, targets, smoothing=0.0,
        )
        # 3 positions × log2(3) ≈ 3 × 1.585 = 4.755
        expected = 3 * math.log2(3)
        assert abs(nll[0] - expected) < 0.01

    def test_sums_not_averages(self):
        """Verify NLL is summed over positions, not averaged."""
        from src.mdl.evaluation import compute_per_string_nll_bits

        def uniform_forward(x):
            B, T = x.shape
            return jnp.zeros((B, T, 3))

        # String of length 1
        nll_1 = compute_per_string_nll_bits(
            uniform_forward, [[0]], [[1]], smoothing=0.0,
        )
        # String of length 3
        nll_3 = compute_per_string_nll_bits(
            uniform_forward, [[0, 1, 2]], [[1, 2, 0]], smoothing=0.0,
        )
        # 3x length should give ~3x NLL
        assert abs(nll_3[0] / nll_1[0] - 3.0) < 0.01

    def test_multiple_strings(self):
        """Batched computation returns correct per-string values."""
        from src.mdl.evaluation import compute_per_string_nll_bits

        def uniform_forward(x):
            B, T = x.shape
            return jnp.zeros((B, T, 3))

        inputs = [[0], [0, 1], [0, 1, 2]]
        targets = [[1], [1, 2], [1, 2, 0]]
        nll = compute_per_string_nll_bits(
            uniform_forward, inputs, targets, smoothing=0.0,
        )
        per_pos = math.log2(3)
        assert abs(nll[0] - 1 * per_pos) < 0.01
        assert abs(nll[1] - 2 * per_pos) < 0.01
        assert abs(nll[2] - 3 * per_pos) < 0.01


# ---------------------------------------------------------------------------
# Grammar-weighted test NLL (golden network sanity)
# ---------------------------------------------------------------------------

class TestGrammarWeightedNLL:
    """Verify grammar-weighted NLL on the golden network matches ~2.94 bits."""

    def test_golden_test_dh_approximately_294(self):
        """Golden network test |D:H| should be ≈ 2.94 bits.

        Reference: Abudy et al. (2025, arXiv:2505.13398v2), line 803.
        The ideal predictor gives E[NLL] = E[n]*(-log2(1-p)) + (-log2(p))
        ≈ (7/3)*0.5146 + 1.737 ≈ 2.938, using E[n] = (1-p)/p over the
        full PCFG (including n=0).
        """
        from src.mdl.evaluation import compute_grammar_weighted_nll_bits
        from src.mdl.golden import build_golden_network_params, golden_forward

        params = build_golden_network_params(p=0.3)

        def golden_fwd(x):
            return golden_forward(params, x)

        # Use moderate max_n for test speed; result should be close
        result = compute_grammar_weighted_nll_bits(
            golden_fwd, max_n=200, p=0.3, batch_size=64,
        )

        # Should be approximately 2.94 bits (with n=0 included in PCFG)
        assert abs(result["data_dh_bits"] - 2.94) < 0.1, (
            f"Golden test |D:H| = {result['data_dh_bits']:.4f}, "
            f"expected ≈ 2.94"
        )

    def test_analytical_expected_value(self):
        """Cross-check against analytical formula for ideal predictor.

        For ideal (no epsilon) predictor on aⁿbⁿ with p=0.3, including
        n=0 in the PCFG expectation:
        E[NLL_total(n)] = n × (-log2(1-p)) + (-log2(p))
                        = n × 0.5146 + 1.7370
        E[NLL] = E[n] × 0.5146 + 1.7370
        where E[n] = (1-p)/p = 7/3 ≈ 2.333 for geometric(0.3) on {0,1,...}.
        So E[NLL] ≈ 1.201 + 1.737 = 2.938.
        """
        from src.mdl.evaluation import compute_anbn_grammar_weights

        p = 0.3
        max_n = 5000  # large enough for convergence
        weights = compute_anbn_grammar_weights(max_n, p=p, min_n=0)

        # Analytical per-string NLL for ideal predictor (n=0..max_n)
        ns = np.arange(0, max_n + 1)
        nll_per_n = ns * (-math.log2(1 - p)) + (-math.log2(p))
        expected_nll = np.sum(weights * nll_per_n)

        # Should be very close to 2.938
        assert abs(expected_nll - 2.938) < 0.01


# ---------------------------------------------------------------------------
# Optimal |D:H|
# ---------------------------------------------------------------------------

class TestOptimalDH:
    """Verify golden network optimal |D:H| computation."""

    def test_golden_h_bits_is_1137(self):
        """Golden LSTM |H| should be 1137 bits.

        This is our LSTM golden (Lan et al. 2024, arXiv:2402.10013v2),
        NOT the 139-bit free-form RNN golden from Abudy et al. (2025).
        The difference is due to LARGE=127 saturated gate weights.
        """
        from src.mdl.evaluation import compute_optimal_dh_test

        result = compute_optimal_dh_test(max_n=10, p=0.3)
        assert result["h_bits"] == 1137

    def test_returns_all_fields(self):
        from src.mdl.evaluation import compute_optimal_dh_test

        result = compute_optimal_dh_test(max_n=10, p=0.3)
        assert "data_dh_bits" in result
        assert "h_bits" in result
        assert "mdl_score" in result


# ---------------------------------------------------------------------------
# Δ%
# ---------------------------------------------------------------------------

class TestDeltaPct:
    """Verify Δ% computation."""

    def test_zero_gap(self):
        from src.mdl.evaluation import compute_delta_pct
        assert compute_delta_pct(2.94, 2.94) == 0.0

    def test_positive_gap(self):
        from src.mdl.evaluation import compute_delta_pct
        # 10% worse
        assert abs(compute_delta_pct(3.234, 2.94) - 10.0) < 0.01

    def test_negative_gap(self):
        """Score below optimal gives negative Δ% (unlikely but valid)."""
        from src.mdl.evaluation import compute_delta_pct
        assert compute_delta_pct(2.0, 4.0) == -50.0

    def test_zero_optimal(self):
        from src.mdl.evaluation import compute_delta_pct
        assert compute_delta_pct(1.0, 0.0) == float("inf")
        assert compute_delta_pct(0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Golden under regularisers
# ---------------------------------------------------------------------------

class TestGoldenRegularisers:
    """Verify golden network norms and MDL."""

    def test_mdl_is_1137(self):
        """LSTM golden |H| = 1137 bits (not 139; see TestOptimalDH)."""
        from src.mdl.evaluation import evaluate_golden_under_regularisers
        result = evaluate_golden_under_regularisers(max_n=10, p=0.3)
        assert result["mdl_bits"] == 1137

    def test_l1_dominated_by_large_weights(self):
        """L1 norm should be large due to LARGE=127 saturated gates."""
        from src.mdl.evaluation import evaluate_golden_under_regularisers
        result = evaluate_golden_under_regularisers(max_n=10, p=0.3)
        # With LARGE=127 and many gate weights, L1 should be >> 100
        assert result["l1_norm"] > 100

    def test_108_params(self):
        from src.mdl.evaluation import evaluate_golden_under_regularisers
        result = evaluate_golden_under_regularisers(max_n=10, p=0.3)
        assert result["n_params"] == 108

    def test_ce_is_finite(self):
        from src.mdl.evaluation import evaluate_golden_under_regularisers
        result = evaluate_golden_under_regularisers(max_n=10, p=0.3)
        assert np.isfinite(result["ce_test_bits"])
        assert result["ce_test_bits"] > 0


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

class TestFormatting:
    """Verify table formatting functions produce output."""

    def test_comparison_table_produces_string(self):
        from src.mdl.evaluation import format_abudy_comparison_table
        table = format_abudy_comparison_table(
            our_test_data_dh=3.5,
            our_train_data_dh=1600.0,
            our_h_bits=200,
            opt_test_data_dh=2.94,
            opt_train_data_dh=1531.77,
            golden_h_bits=139,
        )
        assert "Golden" in table
        assert "Ours" in table
        assert "Δ" in table

    def test_regulariser_table_produces_string(self):
        from src.mdl.evaluation import (
            evaluate_golden_under_regularisers,
            format_golden_regulariser_table,
        )
        result = evaluate_golden_under_regularisers(max_n=10, p=0.3)
        table = format_golden_regulariser_table(result)
        assert "L1" in table
        assert "L2" in table
        assert "MDL" in table
